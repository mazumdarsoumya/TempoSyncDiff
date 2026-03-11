import argparse, json, math, os, random, subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from temposyncdiff.utils.io import load_ckpt
from temposyncdiff.models.vae import TinyVAE
from temposyncdiff.models.teacher_unet import TinyUNet
from temposyncdiff.models.identity_anchor import IdentityEncoder

def load_ref_image(path: str, size: int, normalize: str, device: str):
    img = Image.open(path).convert("RGB").resize((size, size))
    x = torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,3] in 0..1
    x = x.permute(2,0,1).unsqueeze(0)                    # [1,3,H,W]
    if normalize == "minus1_1":
        x = x * 2.0 - 1.0
    return x.to(device)

def to01(video, normalize: str):
    # video: [B,T,3,H,W] if minus1_1 -> map to 0..1
    if normalize == "minus1_1":
        return ((video + 1.0) / 2.0).clamp(0,1)
    return video.clamp(0,1)

def psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))

def write_mp4(frames_u8: np.ndarray, out_mp4: Path, fps: int):
    # frames_u8: [T,H,W,3] uint8
    tmp = out_mp4.parent / (out_mp4.stem + "_frames")
    tmp.mkdir(parents=True, exist_ok=True)
    for i in range(frames_u8.shape[0]):
        Image.fromarray(frames_u8[i]).save(tmp / f"{i:04d}.png")
    cmd = ["ffmpeg","-y","-framerate",str(fps),"-i",str(tmp/"%04d.png"),"-pix_fmt","yuv420p",str(out_mp4)]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def pick_state_dict(ckpt: dict):
    # student checkpoints may store state under various keys
    for k in ["student_denoiser", "student", "denoiser", "model", "net"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k], k
    raise KeyError(f"Could not find model state dict in keys={list(ckpt.keys())}")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="teacher_best.pt snapshot path")
    ap.add_argument("--student", required=True, help="student_best.pt snapshot path")
    ap.add_argument("--manifest", required=True, help="val manifest jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--noise_scale", type=float, default=0.2, help="sigma for zn = z0 + sigma*eps")
    ap.add_argument("--denoise_step", type=float, default=0.1, help="z_hat = zn - step*pred")
    ap.add_argument("--no_identity", action="store_true", help="ablation: zero cond")
    ap.add_argument("--ref_override", default="", help="path to ref image override (cross-ID)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- load teacher ckpt (vae+idenc+teacher denoiser + cfg) --------
    tckpt = load_ckpt(args.teacher, map_location=device)
    tcfg = tckpt.get("cfg", {})
    latent_dim = int(tcfg.get("model", {}).get("latent_dim", 64))

    # Pull data settings from teacher cfg if possible
    data_block = tcfg.get("data_val", tcfg.get("data", {}))
    image_size = int(data_block.get("image_size", 224))
    num_frames = int(data_block.get("num_frames", 50))
    normalize = str(data_block.get("normalize", "minus1_1"))
    ref_mode  = str(data_block.get("ref_mode", "first"))

    vae   = TinyVAE(latent_dim=latent_dim).to(device)
    idenc = IdentityEncoder(emb=latent_dim).to(device)
    tden  = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)

    vae.load_state_dict(tckpt["vae"])
    idenc.load_state_dict(tckpt["idenc"])
    tden.load_state_dict(tckpt["denoiser"])
    vae.eval(); idenc.eval(); tden.eval()

    # -------- load student ckpt (student denoiser weights) --------
    sckpt = load_ckpt(args.student, map_location=device)
    sd, sd_key = pick_state_dict(sckpt)

    sden = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)
    sden.load_state_dict(sd, strict=True)
    sden.eval()

    # -------- dataset --------
    from temposyncdiff.data.real_manifest_dataset import ManifestVideoConfig, ManifestVideoDataset
    cfg = ManifestVideoConfig(
        manifest=args.manifest,
        image_size=image_size,
        num_frames=num_frames,
        normalize=normalize,
        ref_mode=ref_mode,
    )
    ds = ManifestVideoDataset(cfg)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # pick indices
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset has 0 samples.")
    idxs = list(range(min(args.num_samples, n)))

    rows = []
    for si, (idx, batch) in enumerate(zip(idxs, dl), start=1):
        video = batch["video"].to(device)  # [1,T,3,H,W] (likely -1..1 if normalize=minus1_1)
        ref   = batch["ref"].to(device)    # [1,3,H,W]
        B,T,C,H,W = video.shape
        assert T == num_frames, f"Expected T={num_frames}, got {T}"

        # override identity ref if requested
        if args.ref_override.strip():
            ref = load_ref_image(args.ref_override.strip(), size=image_size, normalize=normalize, device=device)

        # conditioning
        if args.no_identity:
            id_emb = torch.zeros((1, latent_dim), device=device)
        else:
            id_emb = idenc(ref)
        cond = id_emb.repeat_interleave(T, dim=0)  # [T,latent]

        # flatten video for VAE
        vf = video.view(B*T, C, H, W)

        # VAE recon baseline (primary reference for metrics)
        z = vae.encode(vf)
        vae_rec = vae.decode(z).view(B, T, 3, H, W).clamp(0,1)

        # Create noisy latent from *real* latent (not pure noise)
        z0 = vae.encode(vf).view(B, T, -1, H//4, W//4)
        z0_bt = z0.view(B*T, z0.shape[2], z0.shape[3], z0.shape[4])
        eps = torch.randn_like(z0_bt)
        zn = z0_bt + float(args.noise_scale) * eps

        # noisy decode
        noisy = vae.decode(zn).view(B, T, 3, H, W).clamp(0,1)

        # teacher denoise
        t_pred = tden(zn, cond)
        zt = zn - float(args.denoise_step) * t_pred
        teacher = vae.decode(zt).view(B, T, 3, H, W).clamp(0,1)

        # student denoise
        s_pred = sden(zn, cond)
        zs = zn - float(args.denoise_step) * s_pred
        student = vae.decode(zs).view(B, T, 3, H, W).clamp(0,1)

        # metrics vs VAE recon (PRIMARY)
        def mse(a,b): return torch.mean((a-b)**2).item()
        mse_noisy = mse(noisy, vae_rec)
        mse_t = mse(teacher, vae_rec)
        mse_s = mse(student, vae_rec)

        psnr_noisy = psnr_from_mse(mse_noisy)
        psnr_t = psnr_from_mse(mse_t)
        psnr_s = psnr_from_mse(mse_s)

        # temporal proxies (computed on decoded videos)
        vid_v = vae_rec[0].permute(0,2,3,1).cpu().numpy()
        vid_t = teacher[0].permute(0,2,3,1).cpu().numpy()
        vid_s = student[0].permute(0,2,3,1).cpu().numpy()
        def temporal_l1(x): return float(np.abs(x[1:]-x[:-1]).mean())
        def flicker_std(x): return float(x.mean(axis=(1,2,3)).std())

        # save outputs
        sample_dir = out_dir / f"sample_{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        def to_u8(x01): return (x01.clamp(0,1)*255).byte().cpu().numpy()  # [T,3,H,W]
        u_vae = to_u8(vae_rec[0]).transpose(0,2,3,1)
        u_noi = to_u8(noisy[0]).transpose(0,2,3,1)
        u_t   = to_u8(teacher[0]).transpose(0,2,3,1)
        u_s   = to_u8(student[0]).transpose(0,2,3,1)

        # 4-up panel frames (VAE | Noisy | Teacher | Student)
        panel = np.concatenate([u_vae, u_noi, u_t, u_s], axis=2)  # concat width
        Image.fromarray(panel[0]).save(sample_dir/"frame0_VAE_NOISY_TEACHER_STUDENT.png")

        write_mp4(u_vae, sample_dir/"vae_recon.mp4", fps=args.fps)
        write_mp4(u_noi, sample_dir/"noisy_decode.mp4", fps=args.fps)
        write_mp4(u_t,   sample_dir/"teacher_denoise.mp4", fps=args.fps)
        write_mp4(u_s,   sample_dir/"student_denoise.mp4", fps=args.fps)
        write_mp4(panel, sample_dir/"panel_4up.mp4", fps=args.fps)

        rows.append({
            "sample_idx": idx,
            "mse_noisy_vs_vae": mse_noisy,
            "psnr_noisy_vs_vae_db": psnr_noisy,
            "mse_teacher_vs_vae": mse_t,
            "psnr_teacher_vs_vae_db": psnr_t,
            "mse_student_vs_vae": mse_s,
            "psnr_student_vs_vae_db": psnr_s,
            "delta_psnr_teacher_minus_noisy_db": psnr_t - psnr_noisy,
            "delta_psnr_student_minus_noisy_db": psnr_s - psnr_noisy,
            "delta_psnr_student_minus_teacher_db": psnr_s - psnr_t,
            "temporal_l1_vae": temporal_l1(vid_v),
            "temporal_l1_teacher": temporal_l1(vid_t),
            "temporal_l1_student": temporal_l1(vid_s),
            "flicker_std_vae": flicker_std(vid_v),
            "flicker_std_teacher": flicker_std(vid_t),
            "flicker_std_student": flicker_std(vid_s),
            "ref_override": bool(args.ref_override.strip()),
            "no_identity": bool(args.no_identity),
            "noise_scale": float(args.noise_scale),
            "denoise_step": float(args.denoise_step),
        })

    # write summary
    (out_dir/"metrics.json").write_text(json.dumps({
        "teacher_ckpt": args.teacher,
        "student_ckpt": args.student,
        "manifest": args.manifest,
        "num_samples": len(rows),
        "student_state_key": sd_key,
        "ref_override": args.ref_override,
        "no_identity": args.no_identity,
        "noise_scale": args.noise_scale,
        "denoise_step": args.denoise_step,
        "rows": rows,
    }, indent=2))

    # CSV
    import csv
    keys = list(rows[0].keys()) if rows else []
    with open(out_dir/"metrics.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[qual] wrote: {out_dir}")
    print(f"[qual] samples: {len(rows)}")

if __name__ == "__main__":
    main()
