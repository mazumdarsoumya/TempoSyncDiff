import os, json, math, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import subprocess

from temposyncdiff.utils.io import load_ckpt
from temposyncdiff.models.vae import TinyVAE
from temposyncdiff.models.teacher_unet import TinyUNet
from temposyncdiff.models.identity_anchor import IdentityEncoder

def write_mp4(frames_u8, path, fps=25):
    path = Path(path)
    tmp = path.parent / (path.stem + "_frames")
    tmp.mkdir(parents=True, exist_ok=True)
    for i in range(frames_u8.shape[0]):
        Image.fromarray(frames_u8[i]).save(tmp / f"{i:04d}.png")
    cmd = ["ffmpeg","-y","-framerate",str(fps),"-i",str(tmp/"%04d.png"),"-pix_fmt","yuv420p",str(path)]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(tmp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--step", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = load_ckpt(args.ckpt, map_location=device)
    cfg  = ckpt.get("cfg", {})
    data_block = cfg.get("data_val", cfg.get("data", {}))

    latent_dim = int(cfg.get("model", {}).get("latent_dim", 64))
    vae   = TinyVAE(latent_dim=latent_dim).to(device)
    idenc = IdentityEncoder(emb=latent_dim).to(device)
    den   = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)

    vae.load_state_dict(ckpt["vae"])
    idenc.load_state_dict(ckpt["idenc"])
    den.load_state_dict(ckpt["denoiser"])
    vae.eval(); idenc.eval(); den.eval()

    # dataset (manifest)
    from temposyncdiff.data.real_manifest_dataset import ManifestVideoConfig, ManifestVideoDataset
    data_cfg = dict(data_block); data_cfg.pop("augment", None)
    allowed = set(ManifestVideoConfig.__dataclass_fields__.keys())
    data_cfg = {k:v for k,v in data_cfg.items() if k in allowed}

    ds = ManifestVideoDataset(ManifestVideoConfig(**data_cfg))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    batch = next(iter(dl))
    video = batch["video"].to(device)   # [1,T,3,H,W] (often -1..1 if normalize=minus1_1)
    ref   = batch["ref"].to(device)

    B,T,C,H,W = video.shape
    T = min(T, args.max_frames)
    video = video[:, :T]
    video_flat = video.reshape(B*T, C, H, W)

    # convert GT to 0..1 for metric comparison if needed
    # (works fine even if already 0..1)
    gt01 = ((video + 1.0) / 2.0).clamp(0,1)

    with torch.no_grad():
        # VAE reconstruction baseline
        z = vae.encode(video_flat)
        r = vae.decode(z).reshape(B, T, 3, H, W).clamp(0,1)

    mse_vae = torch.mean((r - gt01)**2).item()
    psnr_vae = 10.0 * math.log10(1.0 / max(mse_vae, 1e-12))

    with torch.no_grad():
        # noisy latent baseline + denoise
        z0 = vae.encode(video_flat).reshape(B, T, -1, H//4, W//4)
        id_emb = idenc(ref)
        cond = id_emb.repeat_interleave(T, dim=0)

        z0_bt = z0.reshape(B*T, z0.shape[2], z0.shape[3], z0.shape[4])
        noise = torch.randn_like(z0_bt)
        zn = z0_bt + args.sigma * noise

        # baseline noisy decode
        y_noisy = vae.decode(zn).reshape(B, T, 3, H, W).clamp(0,1)

        # teacher denoise (single step, same as training scaffold)
        pred = den(zn, cond)
        zh = zn - args.step * pred
        y = vae.decode(zh).reshape(B, T, 3, H, W).clamp(0,1)

    mse_noisy = torch.mean((y_noisy - gt01)**2).item()
    psnr_noisy = 10.0 * math.log10(1.0 / max(mse_noisy, 1e-12))

    mse_dn = torch.mean((y - gt01)**2).item()
    psnr_dn = 10.0 * math.log10(1.0 / max(mse_dn, 1e-12))

    # temporal metrics
    vid = y[0].permute(0,2,3,1).cpu().numpy()
    gtv = gt01[0].permute(0,2,3,1).cpu().numpy()
    temporal_l1_pred = float(np.abs(vid[1:] - vid[:-1]).mean())
    temporal_l1_gt   = float(np.abs(gtv[1:] - gtv[:-1]).mean())
    flicker_pred = float(vid.mean(axis=(1,2,3)).std())
    flicker_gt   = float(gtv.mean(axis=(1,2,3)).std())

    def to_u8(x01):
        return (x01.clamp(0,1)*255.0).byte()

    gt_u8 = to_u8(gt01[0]).permute(0,2,3,1).cpu().numpy()
    r_u8  = to_u8(r[0]).permute(0,2,3,1).cpu().numpy()
    n_u8  = to_u8(y_noisy[0]).permute(0,2,3,1).cpu().numpy()
    y_u8  = to_u8(y[0]).permute(0,2,3,1).cpu().numpy()

    # Frame0 strip: GT | VAE | NOISY | DENOISE
    strip0 = np.concatenate([gt_u8[0], r_u8[0], n_u8[0], y_u8[0]], axis=1)
    Image.fromarray(strip0).save(out_dir/"frame0_GT_VAE_NOISY_DENOISE.png")

    fps = int(data_cfg.get("fps", 25)) if "fps" in data_cfg else 25
    write_mp4(gt_u8, out_dir/"gt.mp4", fps=fps)
    write_mp4(r_u8,  out_dir/"vae_recon.mp4", fps=fps)
    write_mp4(n_u8,  out_dir/"noisy_decode.mp4", fps=fps)
    write_mp4(y_u8,  out_dir/"teacher_denoise.mp4", fps=fps)

    # side-by-side: GT vs DENOISE
    sbs = np.concatenate([gt_u8, y_u8], axis=2)
    write_mp4(sbs, out_dir/"gt_vs_denoise.mp4", fps=fps)

    report = {
        "ckpt": args.ckpt,
        "device": device,
        "vae_recon": {"mse": mse_vae, "psnr_db": psnr_vae},
        "noisy_decode": {"mse": mse_noisy, "psnr_db": psnr_noisy},
        "teacher_denoise": {"mse": mse_dn, "psnr_db": psnr_dn},
        "delta_psnr_dn_minus_noisy_db": psnr_dn - psnr_noisy,
        "temporal": {
            "gt_temporal_l1": temporal_l1_gt,
            "pred_temporal_l1": temporal_l1_pred,
            "gt_flicker_std": flicker_gt,
            "pred_flicker_std": flicker_pred,
        },
        "outputs": {
            "frame0_png": str(out_dir/"frame0_GT_VAE_NOISY_DENOISE.png"),
            "gt_mp4": str(out_dir/"gt.mp4"),
            "vae_recon_mp4": str(out_dir/"vae_recon.mp4"),
            "noisy_decode_mp4": str(out_dir/"noisy_decode.mp4"),
            "teacher_denoise_mp4": str(out_dir/"teacher_denoise.mp4"),
            "gt_vs_denoise_mp4": str(out_dir/"gt_vs_denoise.mp4"),
        }
    }
    (out_dir/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
