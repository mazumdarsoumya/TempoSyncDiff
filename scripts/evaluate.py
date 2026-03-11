# /home/vineet/PycharmProjects/TempoSyncDiff/scripts/evaluate.py
# TempoSyncDiff - evaluation script for teacher/student checkpoints.
#
# It evaluates the denoising/distillation objective on a validation/test dataloader and
# writes a JSON report under: <out_dir>/metrics/eval.json
#
# Run:
#   export PYTHONPATH="$(pwd)"
#   python -u scripts/evaluate.py --config configs/eval_lrs3.yaml

import argparse
import json
import time
import yaml
from pathlib import Path
from dataclasses import is_dataclass, fields

import torch
from torch.utils.data import DataLoader

from temposyncdiff.utils.io import ensure_dir, load_ckpt
from temposyncdiff.utils.seed import seed_everything
from temposyncdiff.models.vae import TinyVAE
from temposyncdiff.models.teacher_unet import TinyUNet
from temposyncdiff.models.identity_anchor import IdentityEncoder
from temposyncdiff.data.datasets import SyntheticConfig, SyntheticTalkingHeadDataset


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


def _sign_agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return (torch.sign(a + eps) == torch.sign(b + eps)).float().mean()


@torch.no_grad()
def eval_teacher(ckpt_path: str, dl: DataLoader, device: str, amp: bool):
    ckpt = load_ckpt(ckpt_path, map_location=device)
    latent_dim = int(ckpt["cfg"]["model"]["latent_dim"])

    vae = TinyVAE(latent_dim=latent_dim).to(device)
    idenc = IdentityEncoder(emb=latent_dim).to(device)
    denoiser = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)

    vae.load_state_dict(ckpt["vae"])
    idenc.load_state_dict(ckpt["idenc"])
    denoiser.load_state_dict(ckpt["denoiser"])

    vae.eval(); idenc.eval(); denoiser.eval()

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch in dl:
        video = batch["video"].to(device)
        ref = batch["ref"].to(device)
        B, T, C, H, W = video.shape
        video_flat = video.view(B*T, C, H, W)

        with torch.amp.autocast(device_type=device_type, enabled=amp):
            z = vae.encode(video_flat).detach().view(B, T, -1, H//4, W//4)
            id_emb = idenc(ref)
            cond = id_emb.repeat_interleave(T, dim=0)

            z0 = z.view(B*T, z.shape[2], z.shape[3], z.shape[4])
            noise = torch.randn_like(z0)
            zn = z0 + 0.2*noise
            pred = denoiser(zn, cond)

            loss = ((pred - noise)**2).mean()
            acc = _sign_agreement(pred, noise)

        total_loss += float(loss.item())
        total_acc += float(acc.item())
        n += 1

    return {"loss": total_loss/max(n,1), "acc": total_acc/max(n,1)}


@torch.no_grad()
def eval_student(student_ckpt_path: str, teacher_ckpt_path: str, dl: DataLoader, device: str, amp: bool):
    sckpt = load_ckpt(student_ckpt_path, map_location=device)
    tckpt = load_ckpt(teacher_ckpt_path, map_location=device)

    latent_dim = int(tckpt["cfg"]["model"]["latent_dim"])

    vae = TinyVAE(latent_dim=latent_dim).to(device)
    idenc = IdentityEncoder(emb=latent_dim).to(device)
    teacher = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)
    student = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)

    # teacher weights
    vae.load_state_dict(tckpt["vae"])
    idenc.load_state_dict(tckpt["idenc"])
    teacher.load_state_dict(tckpt["denoiser"])

    # student weights (prefer student_best/student.pt format)
    student.load_state_dict(sckpt["student_denoiser"])

    vae.eval(); idenc.eval(); teacher.eval(); student.eval()

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch in dl:
        video = batch["video"].to(device)
        ref = batch["ref"].to(device)
        B, T, C, H, W = video.shape
        video_flat = video.view(B*T, C, H, W)

        with torch.amp.autocast(device_type=device_type, enabled=amp):
            z = vae.encode(video_flat).detach().view(B, T, -1, H//4, W//4)
            id_emb = idenc(ref)
            cond = id_emb.repeat_interleave(T, dim=0)

            z0 = z.view(B*T, z.shape[2], z.shape[3], z.shape[4])
            noise = torch.randn_like(z0)
            zn = z0 + 0.2*noise

            t_pred = teacher(zn, cond)
            s_pred = student(zn, cond)

            loss = ((s_pred - t_pred)**2).mean()
            acc = _sign_agreement(s_pred, t_pred)

        total_loss += float(loss.item())
        total_acc += float(acc.item())
        n += 1

    return {"distill_loss": total_loss/max(n,1), "acc": total_acc/max(n,1)}


def main(cfg):
    seed = int(cfg.get("seed", 123))
    seed_everything(seed)

    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_dir = ensure_dir(cfg.get("out_dir", "results"))
    metrics_dir = ensure_dir(str(Path(out_dir) / "metrics"))

    data_block = cfg["data"]
    ds = build_dataset_from_block(data_block, seed=seed)

    dl = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=False,
    )

    amp = bool(cfg.get("amp", True) and device.startswith("cuda"))

    report = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "data": data_block,
    }

    if cfg.get("teacher_ckpt"):
        report["teacher"] = eval_teacher(cfg["teacher_ckpt"], dl, device=device, amp=amp)

    if cfg.get("student_ckpt") and cfg.get("teacher_ckpt"):
        report["student"] = eval_student(cfg["student_ckpt"], cfg["teacher_ckpt"], dl, device=device, amp=amp)

    out_path = Path(metrics_dir) / "eval.json"
    out_path.write_text(json.dumps(report, indent=2))
    print("[eval] wrote:", str(out_path))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)
