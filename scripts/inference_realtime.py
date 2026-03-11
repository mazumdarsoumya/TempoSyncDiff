# /home/vineet/PycharmProjects/TempoSyncDiff/scripts/inference_realtime.py
# TempoSyncDiff - simple inference / sample generation script (terminal-only).
#
# This script is intentionally lightweight: it loads a student checkpoint and generates
# a short clip (T frames) from a reference frame + (optional) fake viseme tokens.
#
# Output:
#   <out_dir>/samples/sample.mp4   (requires ffmpeg in PATH)
#
# Run:
#   export PYTHONPATH="$(pwd)"
#   python -u scripts/inference_realtime.py --config configs/infer_lrs3.yaml

import argparse
import os
import subprocess
import yaml
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from temposyncdiff.utils.io import ensure_dir, load_ckpt
from temposyncdiff.models.vae import TinyVAE
from temposyncdiff.models.teacher_unet import TinyUNet
from temposyncdiff.models.identity_anchor import IdentityEncoder
from temposyncdiff.data.audio_tokens import fake_viseme_tokens


def _load_image(path: str, size: int, normalize: str):
    img = Image.open(path).convert("RGB").resize((size, size))
    x = torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,C]
    x = x.permute(2, 0, 1).unsqueeze(0)                  # [1,C,H,W]
    if normalize == "minus1_1":
        x = x * 2.0 - 1.0
    return x


@torch.no_grad()
def main(cfg):
    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp = bool(cfg.get("amp", True) and device.startswith("cuda"))

    out_dir = ensure_dir(cfg.get("out_dir", "results"))
    samples_dir = ensure_dir(str(Path(out_dir) / "samples"))

    ckpt_path = str(cfg["student_ckpt"])
    ckpt = load_ckpt(ckpt_path, map_location=device)

    # teacher_cfg is stored in student checkpoint; fall back to cfg
    teacher_cfg = ckpt.get("teacher_cfg", {})
    latent_dim = int(teacher_cfg.get("model", {}).get("latent_dim", cfg.get("latent_dim", 64)))

    vae = TinyVAE(latent_dim=latent_dim).to(device)
    idenc = IdentityEncoder(emb=latent_dim).to(device)
    denoiser = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)

    # student checkpoint packs vae/idenc too; prefer those if present
    if "vae" in ckpt:
        vae.load_state_dict(ckpt["vae"])
    if "idenc" in ckpt:
        idenc.load_state_dict(ckpt["idenc"])
    denoiser.load_state_dict(ckpt["student_denoiser"])

    vae.eval(); idenc.eval(); denoiser.eval()

    size = int(cfg.get("image_size", 96))
    T = int(cfg.get("frames", 25))
    normalize = str(cfg.get("normalize", "0_1"))

    ref_img_path = str(cfg["ref_image"])
    ref = _load_image(ref_img_path, size=size, normalize=normalize).to(device)  # [1,3,H,W]

    # build conditioning
    id_emb = idenc(ref)  # [1,latent_dim]
    cond = id_emb.repeat_interleave(T, dim=0)  # [T, latent_dim]

    # initialize latents with noise
    # latent spatial dims assumed /4 relative to image size in TinyVAE
    h = size // 4
    w = size // 4
    z = torch.randn(T, latent_dim, h, w, device=device)

    # one-step "denoise" scaffold (your full sampler may do multi-step)
    with torch.amp.autocast(device_type=device_type, enabled=amp):
        pred = denoiser(z, cond)
        z_hat = z - 0.1 * pred
        frames = vae.decode(z_hat).clamp(-1, 1 if normalize == "minus1_1" else 1)

    # convert to uint8 images
    if normalize == "minus1_1":
        frames = (frames + 1.0) / 2.0
    frames = (frames.clamp(0, 1) * 255.0).byte()  # [T,3,H,W]

    # save frames to pngs
    tmp_dir = Path(samples_dir) / "tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i in range(T):
        img = frames[i].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(tmp_dir / f"{i:04d}.png")

    out_mp4 = Path(samples_dir) / "sample.mp4"
    fps = int(cfg.get("fps", 25))

    # ffmpeg encode
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "%04d.png"),
        "-pix_fmt", "yuv420p",
        str(out_mp4)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[infer] wrote:", str(out_mp4))
    except Exception as e:
        print("[infer] ffmpeg failed. You can still view PNGs in:", str(tmp_dir))
        print("Error:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)
