# /home/vineet/PycharmProjects/TempoSyncDiff/scripts/distill_student.py
# TempoSyncDiff - Student distillation (epoch-based, early stop, resume, minimal terminal output)
#
# Distillation objective:
#   Student denoiser learns to match teacher residual predictions in latent space.
#
# "Accuracy" proxy (train_acc/val_acc):
#   sign-agreement between student residual and teacher residual (bounded [0,1]).
#
# Run:
#   export PYTHONPATH="$(pwd)"
#   python -u scripts/distill_student.py --config configs/train_student_distill_lrs3.yaml

import argparse
import math
import time
import yaml
from pathlib import Path
from dataclasses import is_dataclass, fields

import torch
from torch.utils.data import DataLoader

from temposyncdiff.utils.seed import seed_everything
from temposyncdiff.utils.io import ensure_dir, save_ckpt, load_ckpt
from temposyncdiff.data.datasets import SyntheticConfig, SyntheticTalkingHeadDataset
from temposyncdiff.models.vae import TinyVAE
from temposyncdiff.models.teacher_unet import TinyUNet
from temposyncdiff.models.student_sampler import StudentSampler
from temposyncdiff.models.identity_anchor import IdentityEncoder

def _filter_kwargs_for_ctor(cls, kwargs: dict) -> dict:
    if is_dataclass(cls):
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in kwargs.items() if k in allowed}
    ann = getattr(cls, "__annotations__", None)
    if isinstance(ann, dict) and ann:
        allowed = set(ann.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    return dict(kwargs)


def build_dataset_from_block(data_block: dict, seed: int):
    data_cfg = dict(data_block)
    dtype = str(data_cfg.pop("type", "synthetic")).strip()

    if dtype == "synthetic_faces":
        dtype = "synthetic"
    if dtype == "real":
        dtype = "manifest"

    # IMPORTANT: YAML may contain keys that are not dataclass fields
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
    # identical to teacher augment (keep consistent)
    if torch.rand((), device=video.device) < 0.5:
        video = torch.flip(video, dims=[-1])
        ref = torch.flip(ref, dims=[-1])

    b = 1.0 + (torch.rand((), device=video.device) * 0.2 - 0.1)
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


def _sign_agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return (torch.sign(a + eps) == torch.sign(b + eps)).float().mean()


@torch.no_grad()
def run_validation(student, teacher, vae, idenc, vdl, device, amp_enabled: bool, max_batches: int, augment: bool, normalize: str):
    student.eval()
    teacher.eval()
    vae.eval()
    idenc.eval()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    for i, batch in enumerate(vdl, start=1):
        if max_batches is not None and i > max_batches:
            break

        video = batch["video"].to(device)
        ref = batch["ref"].to(device)

        if augment:
            video, ref = apply_temporal_augment(video, ref, normalize=normalize)

        B, T, C, H, W = video.shape
        video_flat = video.view(B * T, C, H, W)

        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            z = vae.encode(video_flat).detach().view(B, T, -1, H // 4, W // 4)
            id_emb = idenc(ref)
            cond = id_emb.repeat_interleave(T, dim=0)

            z0 = z.view(B * T, z.shape[2], z.shape[3], z.shape[4])
            noise = torch.randn_like(z0)
            zn = z0 + 0.2 * noise

            t_pred = teacher(zn, cond)
            s_pred = student(zn, cond)

            loss = ((s_pred - t_pred) ** 2).mean()
            acc = _sign_agreement(s_pred, t_pred)

        total_loss += float(loss.item())
        total_acc += float(acc.item())
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
    # Load teacher checkpoint
    # -----------------------
    teacher_ckpt_path = str(cfg["teacher_ckpt"])
    tckpt = load_ckpt(teacher_ckpt_path, map_location=device)

    latent_dim = int(tckpt["cfg"]["model"]["latent_dim"])
    vae = TinyVAE(latent_dim=latent_dim).to(device)
    idenc = IdentityEncoder(emb=latent_dim).to(device)
    teacher = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)

    vae.load_state_dict(tckpt["vae"])
    idenc.load_state_dict(tckpt["idenc"])
    teacher.load_state_dict(tckpt["denoiser"])
    teacher.eval()

    # -----------------------
    # Student model
    # -----------------------
    student = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)
    sampler_steps = int(cfg.get("student", {}).get("steps", 8))
    sampler = StudentSampler(student, steps=sampler_steps).to(device)  # optional; used by inference

    opt = torch.optim.Adam(student.parameters(), lr=float(cfg["train"]["lr"]))

    amp_enabled = bool(cfg["train"].get("amp", True) and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # -----------------------
    # Build train/val datasets
    # -----------------------
    # If student config doesn't define "data", reuse teacher's data block.
    train_data_block = cfg.get("data", tckpt["cfg"]["data"])
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
    # -----------------------
    # Warm-start (optional)
    # -----------------------
    warm = str(cfg["train"].get("warm_start_ckpt", "")).strip()
    if warm and Path(warm).exists():
        st = load_ckpt(warm, map_location=device)
        if "student" in st:
            student.load_state_dict(st["student_denoiser"])
        # optimizer/scaler
        if "opt" in st:
            try:
                opt.load_state_dict(st["opt"])
            except Exception as e:
                print(f"[student] warm_start: could not load optimizer state: {e}")
        if st.get("scaler") is not None and scaler is not None:
            try:
                scaler.load_state_dict(st["scaler"])
            except Exception as e:
                print(f"[student] warm_start: could not load scaler state: {e}")
        start_epoch = 1
        best_val = 1e9
        bad_epochs = 0
        global_step = 0
        print(f"[student] Warm-started from {warm} (epoch counter reset to 1).")
    else:


        # -----------------------
        # Resume (optional)
        # -----------------------
        resume = str(cfg["train"].get("resume_ckpt", "")).strip()
        if resume and Path(resume).exists():
            st = load_ckpt(resume, map_location=device)
            student.load_state_dict(st["student_denoiser"])
            opt.load_state_dict(st["opt"])
            if st.get("scaler") is not None and scaler is not None:
                scaler.load_state_dict(st["scaler"])

            start_epoch = int(st.get("epoch", 0)) + 1
            best_val = float(st.get("best_val", 1e9))
            bad_epochs = int(st.get("bad_epochs", 0))
            global_step = int(st.get("global_step", 0))
            print(f"[student] Resumed from {resume} @ epoch={start_epoch} best_val={best_val:.6f} bad_epochs={bad_epochs}")
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
        print(f"[student] train.epochs not set; using approx epochs={epochs} from steps={steps} and len(dl)={len(dl)}")

    patience = int(cfg["train"].get("patience", 8))
    min_delta = float(cfg["train"].get("min_delta", 1e-4))
    val_every = int(cfg["train"].get("val_every", 1))
    val_batches = cfg["train"].get("val_batches", 50)
    val_batches = None if val_batches is None else int(val_batches)

    normalize_train = str(train_data_block.get("normalize", "0_1"))
    aug_train = bool(train_data_block.get("augment", False))

    normalize_val = normalize_train
    aug_val = False
    if cfg.get("data_val") is not None:
        normalize_val = str(cfg["data_val"].get("normalize", normalize_train))
        aug_val = bool(cfg["data_val"].get("augment", False))

    log_every = int(cfg["train"].get("log_every", 25))

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        student.train()
        teacher.eval()
        vae.eval()
        idenc.eval()

        train_total = 0.0
        train_acc_total = 0.0
        train_n = 0

        for step_in_epoch, batch in enumerate(dl, start=1):
            global_step += 1

            video = batch["video"].to(device)
            ref = batch["ref"].to(device)

            if aug_train:
                with torch.no_grad():
                    video, ref = apply_temporal_augment(video, ref, normalize=normalize_train)

            B, T, C, H, W = video.shape
            video_flat = video.view(B * T, C, H, W)

            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                z = vae.encode(video_flat).detach().view(B, T, -1, H // 4, W // 4)
                id_emb = idenc(ref)
                cond = id_emb.repeat_interleave(T, dim=0)

                z0 = z.view(B * T, z.shape[2], z.shape[3], z.shape[4])
                noise = torch.randn_like(z0)
                zn = z0 + 0.2 * noise

                with torch.no_grad():
                    t_pred = teacher(zn, cond)

                s_pred = student(zn, cond)
                loss = ((s_pred - t_pred) ** 2).mean()
                acc = _sign_agreement(s_pred, t_pred)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_total += float(loss.item())
            train_acc_total += float(acc.item())
            train_n += 1

            if log_every > 0 and (global_step % log_every == 0):
                print(
                    f"[student] epoch {epoch}/{epochs} step {step_in_epoch}/{len(dl)} "
                    f"distill_loss={loss.item():.6f}"
                )

        train_loss = train_total / max(train_n, 1)
        train_acc = train_acc_total / max(train_n, 1)

        if vdl is not None and (epoch % val_every == 0):
            val_loss, val_acc = run_validation(
                student=student,
                teacher=teacher,
                vae=vae,
                idenc=idenc,
                vdl=vdl,
                device=device,
                amp_enabled=amp_enabled,
                max_batches=val_batches,
                augment=aug_val,
                normalize=normalize_val,
            )

            mins = (time.time() - t0) / 60.0
            print(
                f"[student] epoch {epoch}: "
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
                        "student": student.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "best_val": best_val,
                        "bad_epochs": bad_epochs,
                        "cfg": cfg,
                        "teacher_cfg": tckpt["cfg"],
                        # pack teacher components for single-checkpoint inference
                        "vae": vae.state_dict(),
                        "idenc": idenc.state_dict(),
                    },
                    str(Path(ckpt_dir) / "student_best.pt"),
                )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[student] Early stopping: no improvement for {patience} epochs.")
                    break
        else:
            mins = (time.time() - t0) / 60.0
            print(f"[student] epoch {epoch}: train_loss={train_loss:.6f} train_acc={train_acc:.4f} time={mins:.2f}min")

        # always save last
        save_ckpt(
            {
                "epoch": epoch,
                "global_step": global_step,
                "student": student.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_val": best_val,
                "bad_epochs": bad_epochs,
                "cfg": cfg,
                "teacher_cfg": tckpt["cfg"],
                "vae": vae.state_dict(),
                "idenc": idenc.state_dict(),
            },
            str(Path(ckpt_dir) / "student_last.pt"),
        )

    # final export checkpoint (for inference)
    save_ckpt(
        {
            "student": student.state_dict(),
            "vae": vae.state_dict(),
            "idenc": idenc.state_dict(),
            "cfg": cfg,
            "teacher_cfg": tckpt["cfg"],
        },
        str(Path(ckpt_dir) / "student.pt"),
    )
    print("Saved:", str(Path(ckpt_dir) / "student.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)
