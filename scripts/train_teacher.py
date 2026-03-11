# /home/vineet/PycharmProjects/TempoSyncDiff/scripts/train_teacher.py
# TempoSyncDiff - Teacher training (epoch-based, early stop, resume, minimal terminal output)
#
# Notes:
# - Supports data.type: "synthetic" (toy) and "manifest" (real video dataset via manifest/scanner dataset).
# - Ignores/strips unknown YAML keys like "augment" when constructing dataclasses.
# - Prints a single epoch summary line (train/val loss + "acc" proxy + time).
#   * "acc" proxy = sign-agreement between predicted noise and true noise (bounded [0,1]).
#
# Run:
#   export PYTHONPATH="$(pwd)"
#   python -u scripts/train_teacher.py --config configs/train_teacher_lrs3.yaml

import argparse
import math
import time
import yaml
from pathlib import Path
from dataclasses import is_dataclass, fields

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from temposyncdiff.utils.seed import seed_everything
from temposyncdiff.utils.io import ensure_dir, save_ckpt, load_ckpt
from temposyncdiff.data.datasets import SyntheticConfig, SyntheticTalkingHeadDataset
from temposyncdiff.models.vae import TinyVAE
from temposyncdiff.models.teacher_unet import TinyUNet
from temposyncdiff.models.identity_anchor import IdentityEncoder
from temposyncdiff.losses.temporal_losses import temporal_l1, mouth_flicker_proxy
from temposyncdiff.losses.id_loss import identity_loss
from temposyncdiff.losses.sync_loss import sync_proxy_loss


def _filter_kwargs_for_ctor(cls, kwargs: dict) -> dict:
    """
    Keep only keys accepted by a dataclass / constructor signature.
    This prevents errors like: __init__() got an unexpected keyword argument 'augment'
    """
    if is_dataclass(cls):
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in kwargs.items() if k in allowed}

    # Fallback: attempt to use __annotations__ if present
    ann = getattr(cls, "__annotations__", None)
    if isinstance(ann, dict) and ann:
        allowed = set(ann.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}

    # If we cannot introspect, just pass everything (may still error).
    return dict(kwargs)


def build_dataset_from_block(data_block: dict, seed: int):
    """
    data_block is the YAML dict under cfg["data"] (and optionally cfg["data_val"]).

    Supported:
      data.type: synthetic | synthetic_faces (alias) | manifest | real (alias)
    """
    data_cfg = dict(data_block)
    dtype = str(data_cfg.pop("type", "synthetic")).strip()

    # alias support
    if dtype == "synthetic_faces":
        dtype = "synthetic"
    if dtype == "real":
        dtype = "manifest"

    # allow YAML to contain augment without breaking dataclass constructors
    data_cfg.pop("augment", None)

    if dtype == "synthetic":
        dtag = SyntheticConfig(**_filter_kwargs_for_ctor(SyntheticConfig, data_cfg))
        ds = SyntheticTalkingHeadDataset(dtag, seed=seed)

    elif dtype == "manifest":
        from temposyncdiff.data.real_manifest_dataset import (
            ManifestVideoConfig,
            ManifestVideoDataset,
        )
        dtag = ManifestVideoConfig(**_filter_kwargs_for_ctor(ManifestVideoConfig, data_cfg))
        ds = ManifestVideoDataset(dtag)

    else:
        raise ValueError(f"Unknown data.type={dtype}")

    return ds


@torch.no_grad()
def apply_temporal_augment(video, ref, normalize: str):
    """
    video: [B,T,C,H,W], ref: [B,C,H,W]
    Temporal-consistent: same random decisions for all frames.
    """
    # horizontal flip (width dim)
    if torch.rand((), device=video.device) < 0.5:
        video = torch.flip(video, dims=[-1])
        ref = torch.flip(ref, dims=[-1])

    # brightness/contrast (same params for all frames)
    b = 1.0 + (torch.rand((), device=video.device) * 0.2 - 0.1)  # [-0.1, +0.1]
    c = 1.0 + (torch.rand((), device=video.device) * 0.2 - 0.1)

    mean_v = video.mean(dim=(-2, -1), keepdim=True)
    video = (video - mean_v) * c + mean_v
    video = video * b

    mean_r = ref.mean(dim=(-2, -1), keepdim=True)
    ref = (ref - mean_r) * c + mean_r
    ref = ref * b

    if normalize == "minus1_1":
        video = torch.clamp(video, -1.0, 1.0)
        ref = torch.clamp(ref, -1.0, 1.0)
    else:
        video = torch.clamp(video, 0.0, 1.0)
        ref = torch.clamp(ref, 0.0, 1.0)

    return video, ref


def _noise_sign_accuracy(pred_noise: torch.Tensor, true_noise: torch.Tensor) -> torch.Tensor:
    """
    A simple bounded "accuracy" proxy for denoising diffusion:
    sign-agreement between predicted residual and true residual.
    """
    # avoid sign(0) issues
    eps = 1e-12
    s_pred = torch.sign(pred_noise + eps)
    s_true = torch.sign(true_noise + eps)
    return (s_pred == s_true).float().mean()


def forward_teacher_loss(
    vae,
    idenc,
    denoiser,
    video,
    ref,
    viseme,
    device,
    amp_enabled: bool,
):
    """
    Returns:
      loss (scalar),
      stats dict with:
        loss_diff, loss_id, loss_temp, loss_sync, acc_proxy
    """
    B, T, C, H, W = video.shape
    video_flat = video.view(B * T, C, H, W)

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    w_id   = float(getattr(cfg, "get", lambda *_: {})("train", {}).get("w_id", 0.0)) if isinstance(cfg, dict) else 0.0
    w_temp = float(getattr(cfg, "get", lambda *_: {})("train", {}).get("w_temp", 0.0)) if isinstance(cfg, dict) else 0.0
    w_sync = float(getattr(cfg, "get", lambda *_: {})("train", {}).get("w_sync", 0.0)) if isinstance(cfg, dict) else 0.0
    w_rec  = float(getattr(cfg, "get", lambda *_: {})("train", {}).get("w_rec", 1.0)) if isinstance(cfg, dict) else 1.0

    with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
        z = vae.encode(video_flat)         # [B*T,latent,h,w]
        z = z.detach()                     # keep encoder frozen; decoder still trains
        z = z.view(B, T, *z.shape[1:])

        id_emb = idenc(ref)                # [B,latent_dim]
        cond = id_emb.repeat_interleave(T, dim=0)

        z0 = z.view(B * T, *z.shape[2:])
        noise = torch.randn_like(z0)
        zn = z0 + 0.2 * noise
        pred = denoiser(zn, cond)

        loss_diff = ((pred - noise) ** 2).mean()
        acc_proxy = _noise_sign_accuracy(pred, noise)

        zh = (zn - 0.1 * pred).view(B * T, *z.shape[2:])
        ih = vae.decode(zh).view(B, T, 3, H, W)
        # ---- FIX: pixel reconstruction loss prevents black collapse ----
        loss_rec = F.l1_loss(ih, video)

        id_pred = idenc(ih[:, 0])
        loss_id = identity_loss(id_pred, id_emb)
        loss_temp = temporal_l1(ih) + 0.5 * mouth_flicker_proxy(ih)

        # Sync loss optional (requires viseme). If your viseme tokens are constant,
        # keep it disabled (0.0). Otherwise uncomment:
        # loss_sync = sync_proxy_loss(ih, viseme)
        loss_sync = sync_proxy_loss(ih, viseme) if viseme is not None else torch.tensor(0.0, device=device)

        loss = loss_diff + (w_id * loss_id) + (w_temp * loss_temp) + (w_sync * loss_sync) + (w_rec * loss_rec)

    stats = {
        "loss_diff": loss_diff.detach(),
        "loss_id": loss_id.detach(),
        "loss_temp": loss_temp.detach(),
        "loss_sync": loss_sync.detach(),
        "loss_rec": loss_rec.detach(),
        "acc_proxy": acc_proxy.detach(),
    }
    return loss, stats


@torch.no_grad()
def run_validation(
    vae,
    idenc,
    denoiser,
    vdl,
    device,
    amp_enabled: bool,
    max_batches: int,
    augment: bool,
    normalize: str,
):
    vae.eval()
    idenc.eval()
    denoiser.eval()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for i, batch in enumerate(vdl, start=1):
        if max_batches is not None and i > max_batches:
            break

        video = batch["video"].to(device)  # [B,T,C,H,W]
        ref = batch["ref"].to(device)      # [B,C,H,W]
        viseme = batch.get("viseme", None)
        if viseme is not None:
            viseme = viseme.to(device)

        if augment:
            video, ref = apply_temporal_augment(video, ref, normalize=normalize)

        loss, stats = forward_teacher_loss(
            vae=vae,
            idenc=idenc,
            denoiser=denoiser,
            video=video,
            ref=ref,
            viseme=viseme,
            device=device,
            amp_enabled=amp_enabled,
        )
        total_loss += float(loss.item())
        total_acc += float(stats["acc_proxy"].item())
        n += 1

    return (total_loss / max(n, 1)), (total_acc / max(n, 1))


def main(cfg):
    seed = int(cfg.get("seed", 123))
    seed_everything(seed)

    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_dir = ensure_dir(cfg.get("out_dir", "results"))
    ckpt_dir = ensure_dir(str(Path(out_dir) / "checkpoints"))

    # -----------------------
    # Build train/val datasets
    # -----------------------
    train_data_block = cfg["data"]
    train_ds = build_dataset_from_block(train_data_block, seed=seed)

    dl = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        drop_last=True,
    )

    vdl = None
    if cfg.get("data_val") is not None:
        val_ds = build_dataset_from_block(cfg["data_val"], seed=seed)
        vdl = DataLoader(
            val_ds,
            batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
            shuffle=False,
            num_workers=int(cfg["train"].get("num_workers", 0)),
            drop_last=False,
        )

    # -------------
    # Build models
    # -------------
    latent_dim = int(cfg["model"]["latent_dim"])
    vae = TinyVAE(latent_dim=latent_dim).to(device)
    idenc = IdentityEncoder(emb=latent_dim).to(device)
    denoiser = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)

    # Freeze VAE (pretrained)
    for p in vae.parameters():
        p.requires_grad = False

    # ---- FIX: load pretrained VAE (optional) ----
    vae_ckpt = str(cfg.get("train", {}).get("vae_ckpt", "")).strip()
    if vae_ckpt and Path(vae_ckpt).exists():
        st = load_ckpt(vae_ckpt, map_location=device)
        if isinstance(st, dict) and "vae" in st:
            vae.load_state_dict(st["vae"])
        else:
            vae.load_state_dict(st)
        print(f"[teacher] loaded VAE from: {vae_ckpt}")

    # ---- FIX: freeze VAE (pretrained) ----
    for pp in vae.parameters():
        pp.requires_grad_(False)
    vae.eval()

    # ---- FIX: freeze IdentityEncoder to prevent collapse ----
    for pp in idenc.parameters():
        pp.requires_grad_(False)
    idenc.eval()

    opt = torch.optim.Adam(list(idenc.parameters()) + list(denoiser.parameters()), lr=float(cfg["train"]["lr"]))

    amp_enabled = bool(cfg["train"].get("amp", True) and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    
    # -----------------------
    # Warm-start (optional)
    # -----------------------
    # Use when you want to start from epoch 1 but initialize weights (and optimizer)
    # from a previous "best" checkpoint.
    warm = str(cfg["train"].get("warm_start_ckpt", "")).strip()
    if warm and Path(warm).exists():
        st = load_ckpt(warm, map_location=device)
        # weights
        if "vae" in st: vae.load_state_dict(st["vae"])
        if "idenc" in st: idenc.load_state_dict(st["idenc"])
        if "denoiser" in st: denoiser.load_state_dict(st["denoiser"])
        # optimizer/scaler (keeps momentum if present)
        if "opt" in st:
            try:
                opt.load_state_dict(st["opt"])
            except Exception as e:
                print(f"[teacher] warm_start: could not load optimizer state: {e}")
        if st.get("scaler") is not None and scaler is not None:
            try:
                scaler.load_state_dict(st["scaler"])
            except Exception as e:
                print(f"[teacher] warm_start: could not load scaler state: {e}")

        # reset counters so the next print is "epoch 1"
        start_epoch = 1
        best_val = 1e9
        bad_epochs = 0
        global_step = 0
        print(f"[teacher] Warm-started from {warm} (epoch counter reset to 1).")
    else:
    # -----------------------
        # Resume (optional)
        # -----------------------
        resume = str(cfg["train"].get("resume_ckpt", "")).strip()
        if resume and Path(resume).exists():
            st = load_ckpt(resume, map_location=device)
            vae.load_state_dict(st["vae"])
            idenc.load_state_dict(st["idenc"])
            denoiser.load_state_dict(st["denoiser"])
            opt.load_state_dict(st["opt"])
            if st.get("scaler") is not None and scaler is not None:
                scaler.load_state_dict(st["scaler"])

            start_epoch = int(st.get("epoch", 0)) + 1
            best_val = float(st.get("best_val", 1e9))
            bad_epochs = int(st.get("bad_epochs", 0))
            global_step = int(st.get("global_step", 0))
            print(f"[teacher] Resumed from {resume} @ epoch={start_epoch} best_val={best_val:.6f} bad_epochs={bad_epochs}")
        else:
            start_epoch = 1
            best_val = 1e9
            bad_epochs = 0
            global_step = 0


# -----------------------
    # Training config
    # -----------------------
    epochs = int(cfg["train"].get("epochs", 0))
    if epochs <= 0:
        steps = int(cfg["train"].get("steps", 0))
        if steps <= 0:
            raise ValueError("Set train.epochs (recommended) or train.steps (legacy).")
        epochs = max(1, math.ceil(steps / max(1, len(dl))))
        print(f"[teacher] train.epochs not set; using approx epochs={epochs} from steps={steps} and len(dl)={len(dl)}")

    patience = int(cfg["train"].get("patience", 8))
    min_delta = float(cfg["train"].get("min_delta", 1e-4))
    val_every = int(cfg["train"].get("val_every", 1))
    val_batches = cfg["train"].get("val_batches", 50)
    val_batches = None if val_batches is None else int(val_batches)

    normalize_train = str(cfg["data"].get("normalize", "0_1"))
    aug_train = bool(cfg["data"].get("augment", False))

    normalize_val = normalize_train
    aug_val = False
    if cfg.get("data_val") is not None:
        normalize_val = str(cfg["data_val"].get("normalize", normalize_train))
        aug_val = bool(cfg["data_val"].get("augment", False))

    # If you want *no step-by-step printing*, set log_every: 0 in YAML.
    log_every = int(cfg["train"].get("log_every", 25))

    # -----------------------
    # Epoch training loop
    # -----------------------
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        vae.eval()
        idenc.eval()
        denoiser.train()

        train_total = 0.0
        train_acc_total = 0.0
        train_n = 0

        for step_in_epoch, batch in enumerate(dl, start=1):
            global_step += 1

            video = batch["video"].to(device)  # [B,T,C,H,W]
            ref = batch["ref"].to(device)      # [B,C,H,W]
            viseme = batch.get("viseme", None)
            if viseme is not None:
                viseme = viseme.to(device)

            if aug_train:
                with torch.no_grad():
                    video, ref = apply_temporal_augment(video, ref, normalize=normalize_train)

            loss, stats = forward_teacher_loss(
                vae=vae,
                idenc=idenc,
                denoiser=denoiser,
                video=video,
                ref=ref,
                viseme=viseme,
                device=device,
                amp_enabled=amp_enabled,
            )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_total += float(loss.item())
            train_acc_total += float(stats["acc_proxy"].item())
            train_n += 1

            if False:  # step logging disabled:
                print(
                    f"[teacher] epoch {epoch}/{epochs} step {step_in_epoch}/{len(dl)} "
                    f"loss={loss.item():.4f} "
                    f"(diff={stats['loss_diff'].item():.4f}, id={stats['loss_id'].item():.4f}, "
                    f"temp={stats['loss_temp'].item():.4f}, sync={stats['loss_sync'].item():.4f})"
                )

        train_loss = train_total / max(train_n, 1)
        train_acc = train_acc_total / max(train_n, 1)

        # -----------------------
        # Validation + early stop
        # -----------------------
        if vdl is not None and (epoch % val_every == 0):
            val_loss, val_acc = run_validation(
                vae=vae,
                idenc=idenc,
                denoiser=denoiser,
                vdl=vdl,
                device=device,
                amp_enabled=amp_enabled,
                max_batches=val_batches,
                augment=aug_val,
                normalize=normalize_val,
            )

            mins = (time.time() - t0) / 60.0
            print(
                f"[teacher] epoch {epoch}: "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={mins:.2f}min"
            )

            improved = (val_loss < best_val - min_delta)
            if improved:
                best_val = float(val_loss)
                bad_epochs = 0
                save_ckpt(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "vae": vae.state_dict(),
                        "idenc": idenc.state_dict(),
                        "denoiser": denoiser.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "best_val": best_val,
                        "bad_epochs": bad_epochs,
                        "cfg": cfg,
                    },
                    str(Path(ckpt_dir) / "teacher_best.pt"),
                )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[teacher] Early stopping: no improvement for {patience} epochs.")
                    save_ckpt(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "vae": vae.state_dict(),
                            "idenc": idenc.state_dict(),
                            "denoiser": denoiser.state_dict(),
                            "opt": opt.state_dict(),
                            "scaler": scaler.state_dict() if scaler is not None else None,
                            "best_val": best_val,
                            "bad_epochs": bad_epochs,
                            "cfg": cfg,
                        },
                        str(Path(ckpt_dir) / "teacher_last.pt"),
                    )
                    break
        else:
            mins = (time.time() - t0) / 60.0
            print(f"[teacher] epoch {epoch}: train_loss={train_loss:.6f} train_acc={train_acc:.4f} time={mins:.2f}min")

        # Always save "last" each epoch (so resume works)
        save_ckpt(
            {
                "epoch": epoch,
                "global_step": global_step,
                "vae": vae.state_dict(),
                "idenc": idenc.state_dict(),
                "denoiser": denoiser.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_val": best_val,
                "bad_epochs": bad_epochs,
                "cfg": cfg,
            },
            str(Path(ckpt_dir) / "teacher_last.pt"),
        )

    # final export checkpoint
    save_ckpt(
        {
            "vae": vae.state_dict(),
            "idenc": idenc.state_dict(),
            "denoiser": denoiser.state_dict(),
            "cfg": cfg,
        },
        str(Path(ckpt_dir) / "teacher.pt"),
    )
    print("Saved:", str(Path(ckpt_dir) / "teacher.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)
