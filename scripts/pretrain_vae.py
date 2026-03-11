import argparse, math, time, yaml
from pathlib import Path
from dataclasses import is_dataclass, fields

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from temposyncdiff.utils.seed import seed_everything
from temposyncdiff.utils.io import ensure_dir, save_ckpt
from temposyncdiff.data.datasets import SyntheticConfig, SyntheticTalkingHeadDataset
from temposyncdiff.models.vae import TinyVAE

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
    if dtype == "synthetic_faces": dtype = "synthetic"
    if dtype == "real": dtype = "manifest"
    data_cfg.pop("augment", None)

    if dtype == "synthetic":
        dtag = SyntheticConfig(**_filter_kwargs_for_ctor(SyntheticConfig, data_cfg))
        ds = SyntheticTalkingHeadDataset(dtag, seed=seed)
    elif dtype == "manifest":
        from temposyncdiff.data.real_manifest_dataset import ManifestVideoConfig, ManifestVideoDataset
        dtag = ManifestVideoConfig(**_filter_kwargs_for_ctor(ManifestVideoConfig, data_cfg))
        ds = ManifestVideoDataset(dtag)
    else:
        raise ValueError(f"Unknown data.type={dtype}")
    return ds

@torch.no_grad()
def _apply_aug(video, normalize: str):
    # very light augmentation (optional)
    if torch.rand((), device=video.device) < 0.5:
        video = torch.flip(video, dims=[-1])
    if normalize == "0_1":
        video = torch.clamp(video, 0.0, 1.0)
    else:
        video = torch.clamp(video, -1.0, 1.0)
    return video

def main(cfg):
    seed = int(cfg.get("seed", 123))
    seed_everything(seed)

    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    out_dir = ensure_dir(cfg.get("out_dir", "results"))
    ckpt_dir = ensure_dir(str(Path(out_dir) / "checkpoints"))

    latent_dim = int(cfg["model"]["latent_dim"])
    vae = TinyVAE(latent_dim=latent_dim).to(device)

    opt = torch.optim.Adam(vae.parameters(), lr=float(cfg["train"]["lr"]))
    amp = bool(cfg["train"].get("amp", False) and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    train_ds = build_dataset_from_block(cfg["data"], seed=seed)
    dl = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
                    num_workers=int(cfg["train"].get("num_workers", 0)), drop_last=True)

    vdl = None
    if cfg.get("data_val") is not None:
        val_ds = build_dataset_from_block(cfg["data_val"], seed=seed)
        vdl = DataLoader(val_ds, batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
                         shuffle=False, num_workers=int(cfg["train"].get("num_workers", 0)), drop_last=False)

    epochs = int(cfg["train"].get("epochs", 20))
    patience = int(cfg["train"].get("patience", 5))
    log_every = int(cfg["train"].get("log_every", 50))
    normalize = str(cfg["data"].get("normalize", "0_1"))

    best_val = 1e9
    bad = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        vae.train()
        total = 0.0
        n = 0

        for batch in dl:
            global_step += 1
            video = batch["video"].to(device)          # [B,T,C,H,W]
            B,T,C,H,W = video.shape
            video = _apply_aug(video, normalize=normalize)
            x = video.view(B*T, C, H, W)               # [B*T,C,H,W]

            with torch.amp.autocast(device_type=device_type, enabled=amp):
                z = vae.encode(x)
                xr = vae.decode(z)
                loss = F.l1_loss(xr, x)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += float(loss.item())
            n += 1

            if log_every > 0 and (global_step % log_every == 0):
                print(f"[vae] epoch {epoch}/{epochs} step {n}/{len(dl)} loss={loss.item():.6f}")

        train_loss = total / max(n, 1)

        val_loss = None
        if vdl is not None:
            vae.eval()
            vtot = 0.0
            vn = 0
            with torch.no_grad():
                for vb in vdl:
                    vvideo = vb["video"].to(device)
                    BB,TT,CC,HH,WW = vvideo.shape
                    vvideo = _apply_aug(vvideo, normalize=normalize)
                    vx = vvideo.view(BB*TT, CC, HH, WW)
                    z = vae.encode(vx)
                    vxr = vae.decode(z)
                    vloss = F.l1_loss(vxr, vx)
                    vtot += float(vloss.item())
                    vn += 1
            val_loss = vtot / max(vn, 1)

        mins = (time.time() - t0) / 60.0
        if val_loss is None:
            print(f"[vae] epoch {epoch}: train_loss={train_loss:.6f} time={mins:.2f}min")
        else:
            print(f"[vae] epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={mins:.2f}min")

        improved = (val_loss is not None and val_loss < best_val - 1e-5)
        if val_loss is not None and improved:
            best_val = float(val_loss)
            bad = 0
            save_ckpt({"vae": vae.state_dict(), "cfg": cfg}, str(Path(ckpt_dir) / "vae_pretrained.pt"))
            print("[vae] Saved best:", str(Path(ckpt_dir) / "vae_pretrained.pt"))
        elif val_loss is not None:
            bad += 1
            if bad >= patience:
                print(f"[vae] Early stopping: no improvement for {patience} epochs.")
                break

    # always save last
    save_ckpt({"vae": vae.state_dict(), "cfg": cfg}, str(Path(ckpt_dir) / "vae_pretrained_last.pt"))
    print("[vae] Saved last:", str(Path(ckpt_dir) / "vae_pretrained_last.pt"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)
