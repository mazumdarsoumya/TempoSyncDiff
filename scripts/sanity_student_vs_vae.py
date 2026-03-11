import argparse, os, json, math, subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from temposyncdiff.utils.io import load_ckpt, ensure_dir
from temposyncdiff.models.vae import TinyVAE
from temposyncdiff.models.teacher_unet import TinyUNet
from temposyncdiff.models.identity_anchor import IdentityEncoder


def _psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def _to_u8(x01: torch.Tensor) -> np.ndarray:
    # x01: [T,3,H,W] float in [0,1]
    return (x01.clamp(0, 1) * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()


def _write_mp4(frames_u8: np.ndarray, out_mp4: Path, fps: int = 25) -> str:
    # frames_u8: [T,H,W,3] uint8
    frames_dir = out_mp4.parent / (out_mp4.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames_u8.shape[0]):
        Image.fromarray(frames_u8[i]).save(frames_dir / f"{i:04d}.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%04d.png"),
        "-pix_fmt", "yuv420p",
        str(out_mp4)
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(frames_dir)


def _concat_h(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # [T,H,W,3] + [T,H,W,3] -> [T,H,2W,3]
    return np.concatenate([a, b], axis=2)


def _build_manifest_dataloader_from_teacher_cfg(teacher_cfg: dict, split: str, batch_size: int):
    # Lazy import to avoid cv2 import unless needed
    from temposyncdiff.data.real_manifest_dataset import ManifestVideoConfig, ManifestVideoDataset

    if split == "val":
        block = teacher_cfg.get("data_val") or teacher_cfg.get("data")
    else:
        block = teacher_cfg.get("data") or teacher_cfg.get("data_val")

    data_cfg = dict(block)
    data_cfg.pop("augment", None)

    # filter only dataclass fields
    allowed = set(ManifestVideoConfig.__dataclass_fields__.keys())
    data_cfg = {k: v for k, v in data_cfg.items() if k in allowed}

    ds = ManifestVideoDataset(ManifestVideoConfig(**data_cfg))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return dl


def _pick_state_dict(ckpt: dict):
    # Try common keys in order
    for k in ["student_denoiser","student","denoiser","unet","model","net"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k], k
    # If it looks like a raw state_dict itself
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt, "(root)"
    raise KeyError(f"Could not find model state dict in keys={list(ckpt.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="path to teacher ckpt (teacher_best.pt or teacher.pt)")
    ap.add_argument("--student", required=True, help="path to student ckpt (student_best.pt)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--batches", type=int, default=1, help="number of batches to evaluate (avg). Use >1 for stability.")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--sigma", type=float, default=0.2, help="noise scale used in training")
    ap.add_argument("--step", type=float, default=0.1, help="one-step update used in training")
    ap.add_argument("--no_viz", action="store_true", help="disable mp4/png writing")
    args = ap.parse_args()

    out_dir = Path(ensure_dir(args.out))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_ckpt = load_ckpt(args.teacher, map_location=device)
    student_ckpt = load_ckpt(args.student, map_location=device)

    # latent dim from teacher cfg (source of truth)
    teacher_cfg = teacher_ckpt.get("cfg", {})
    latent_dim = int(teacher_cfg.get("model", {}).get("latent_dim", 64))

    # Build modules (VAE + idenc always from teacher)
    vae = TinyVAE(latent_dim=latent_dim).to(device)
    idenc = IdentityEncoder(emb=latent_dim).to(device)

    # Teacher denoiser
    teacher_unet = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)

    vae.load_state_dict(teacher_ckpt["vae"])
    idenc.load_state_dict(teacher_ckpt["idenc"])
    teacher_unet.load_state_dict(teacher_ckpt["denoiser"])

    vae.eval(); idenc.eval(); teacher_unet.eval()

    # Student denoiser:
    # default assume same class as teacher. If your student arch is different,
    # update this import/class to match distill_student.py
    student_unet = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)
    sd, sd_key = _pick_state_dict(student_ckpt)
    try:
        student_unet.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"[sanity] student load strict=True failed from key '{sd_key}': {e}")
        print("[sanity] Retrying strict=False (if it still fails, student arch != TinyUNet).")
        student_unet.load_state_dict(sd, strict=False)
    student_unet.eval()

    # Data loader from teacher cfg (real videos)
    dl = _build_manifest_dataloader_from_teacher_cfg(teacher_cfg, split=args.split, batch_size=1)
    it = iter(dl)

    sums = {
        "mse_noisy_vs_vae": 0.0,
        "mse_teacher_vs_vae": 0.0,
        "mse_student_vs_vae": 0.0,
        "temporal_l1_vae": 0.0,
        "temporal_l1_teacher": 0.0,
        "temporal_l1_student": 0.0,
        "flicker_std_vae": 0.0,
        "flicker_std_teacher": 0.0,
        "flicker_std_student": 0.0,
    }
    n = 0

    saved = False

    for bi in range(args.batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        video = batch["video"].to(device)  # expected [-1,1] if normalize=minus1_1
        ref   = batch["ref"].to(device)

        B, T, C, H, W = video.shape
        video_flat = video.view(B*T, C, H, W)

        with torch.no_grad():
            # VAE recon (reference target for comparisons)
            z0 = vae.encode(video_flat)                        # [B*T,latent,h,w]
            vae_recon = vae.decode(z0).view(B, T, 3, H, W)     # [0,1]
            vae_recon = vae_recon.clamp(0, 1)

            # Build conditioning
            id_emb = idenc(ref)                                # uses same normalization as training
            cond = id_emb.repeat_interleave(T, dim=0)

            # Add noise in latent space, decode baseline
            noise = torch.randn_like(z0)
            zn = z0 + float(args.sigma) * noise
            noisy_decode = vae.decode(zn).view(B, T, 3, H, W).clamp(0, 1)

            # Teacher one-step denoise (on real latent + noise)
            pred_t = teacher_unet(zn, cond)
            zt = zn - float(args.step) * pred_t
            out_t = vae.decode(zt).view(B, T, 3, H, W).clamp(0, 1)

            # Student one-step denoise
            pred_s = student_unet(zn, cond)
            zs = zn - float(args.step) * pred_s
            out_s = vae.decode(zs).view(B, T, 3, H, W).clamp(0, 1)

            # Metrics VS VAE recon (NOT GT)
            mse_noisy  = torch.mean((noisy_decode - vae_recon) ** 2).item()
            mse_t      = torch.mean((out_t - vae_recon) ** 2).item()
            mse_s      = torch.mean((out_s - vae_recon) ** 2).item()

            # Temporal
            def temporal_l1(x):
                return torch.mean(torch.abs(x[:, 1:] - x[:, :-1])).item()

            def flicker_std(x):
                # frame mean std (per sample), then mean over batch
                fm = x.mean(dim=(2, 3, 4))  # [B,T]
                return fm.std(dim=1).mean().item()

            tl1_vae = temporal_l1(vae_recon)
            tl1_t   = temporal_l1(out_t)
            tl1_s   = temporal_l1(out_s)

            fk_vae = flicker_std(vae_recon)
            fk_t   = flicker_std(out_t)
            fk_s   = flicker_std(out_s)

        sums["mse_noisy_vs_vae"]   += mse_noisy
        sums["mse_teacher_vs_vae"] += mse_t
        sums["mse_student_vs_vae"] += mse_s
        sums["temporal_l1_vae"]    += tl1_vae
        sums["temporal_l1_teacher"]+= tl1_t
        sums["temporal_l1_student"]+= tl1_s
        sums["flicker_std_vae"]    += fk_vae
        sums["flicker_std_teacher"]+= fk_t
        sums["flicker_std_student"]+= fk_s
        n += 1

        # Save visualization only once (first batch)
        if (not args.no_viz) and (not saved):
            vr_u8 = _to_u8(vae_recon[0])
            nd_u8 = _to_u8(noisy_decode[0])
            tt_u8 = _to_u8(out_t[0])
            ss_u8 = _to_u8(out_s[0])

            # single frame comparison
            f0 = np.concatenate([vr_u8[0], nd_u8[0], tt_u8[0], ss_u8[0]], axis=1)
            Image.fromarray(f0).save(out_dir / "frame0_VAE_NOISY_TEACHER_STUDENT.png")

            # videos
            _write_mp4(vr_u8, out_dir / "vae_recon.mp4", fps=args.fps)
            _write_mp4(nd_u8, out_dir / "noisy_decode.mp4", fps=args.fps)
            _write_mp4(tt_u8, out_dir / "teacher_denoise.mp4", fps=args.fps)
            _write_mp4(ss_u8, out_dir / "student_denoise.mp4", fps=args.fps)
            _write_mp4(_concat_h(vr_u8, tt_u8), out_dir / "vae_vs_teacher.mp4", fps=args.fps)
            _write_mp4(_concat_h(vr_u8, ss_u8), out_dir / "vae_vs_student.mp4", fps=args.fps)
            _write_mp4(_concat_h(tt_u8, ss_u8), out_dir / "teacher_vs_student.mp4", fps=args.fps)

            saved = True

    if n == 0:
        raise RuntimeError("No batches evaluated. Check manifests / dataset.")

    # averages
    avg = {k: (v / n) for k, v in sums.items()}

    report = {
        "device": device,
        "teacher_ckpt": args.teacher,
        "student_ckpt": args.student,
        "split": args.split,
        "batches": n,

        # VS VAE recon (this is what you wanted)
        "vs_vae_recon": {
            "noisy":   {"mse": avg["mse_noisy_vs_vae"],   "psnr_db": _psnr_from_mse(avg["mse_noisy_vs_vae"])},
            "teacher": {"mse": avg["mse_teacher_vs_vae"], "psnr_db": _psnr_from_mse(avg["mse_teacher_vs_vae"])},
            "student": {"mse": avg["mse_student_vs_vae"], "psnr_db": _psnr_from_mse(avg["mse_student_vs_vae"])},

            "delta_psnr_teacher_minus_noisy_db": _psnr_from_mse(avg["mse_teacher_vs_vae"]) - _psnr_from_mse(avg["mse_noisy_vs_vae"]),
            "delta_psnr_student_minus_noisy_db": _psnr_from_mse(avg["mse_student_vs_vae"]) - _psnr_from_mse(avg["mse_noisy_vs_vae"]),
            "delta_psnr_student_minus_teacher_db": _psnr_from_mse(avg["mse_student_vs_vae"]) - _psnr_from_mse(avg["mse_teacher_vs_vae"]),
        },

        "temporal": {
            "temporal_l1": {
                "vae": avg["temporal_l1_vae"],
                "teacher": avg["temporal_l1_teacher"],
                "student": avg["temporal_l1_student"],
            },
            "flicker_std": {
                "vae": avg["flicker_std_vae"],
                "teacher": avg["flicker_std_teacher"],
                "student": avg["flicker_std_student"],
            },
        },

        "outputs": None if args.no_viz else {
            "frame0_png": str(out_dir / "frame0_VAE_NOISY_TEACHER_STUDENT.png"),
            "vae_recon_mp4": str(out_dir / "vae_recon.mp4"),
            "noisy_decode_mp4": str(out_dir / "noisy_decode.mp4"),
            "teacher_denoise_mp4": str(out_dir / "teacher_denoise.mp4"),
            "student_denoise_mp4": str(out_dir / "student_denoise.mp4"),
            "vae_vs_teacher_mp4": str(out_dir / "vae_vs_teacher.mp4"),
            "vae_vs_student_mp4": str(out_dir / "vae_vs_student.mp4"),
            "teacher_vs_student_mp4": str(out_dir / "teacher_vs_student.mp4"),
        }
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
