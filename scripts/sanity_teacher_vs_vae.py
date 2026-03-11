import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import subprocess

from temposyncdiff.utils.io import load_ckpt, ensure_dir
from temposyncdiff.models.vae import TinyVAE
from temposyncdiff.models.teacher_unet import TinyUNet
from temposyncdiff.models.identity_anchor import IdentityEncoder
from temposyncdiff.data.real_manifest_dataset import ManifestVideoConfig, ManifestVideoDataset

def write_mp4(frames_u8, path, fps=25):
    path = Path(path)
    tmp = path.parent / (path.stem + "_frames")
    tmp.mkdir(parents=True, exist_ok=True)
    for i in range(frames_u8.shape[0]):
        Image.fromarray(frames_u8[i]).save(tmp / f"{i:04d}.png")
    cmd = ["ffmpeg","-y","-framerate",str(fps),"-i",str(tmp/"%04d.png"),
           "-pix_fmt","yuv420p",str(path)]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def psnr_from_mse(mse):
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5, help="how many clips to test")
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(ensure_dir(args.out))
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

    # dataset
    data_cfg = dict(data_block); data_cfg.pop("augment", None)
    allowed = set(ManifestVideoConfig.__dataclass_fields__.keys())
    data_cfg = {k:v for k,v in data_cfg.items() if k in allowed}
    ds = ManifestVideoDataset(ManifestVideoConfig(**data_cfg))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    sigma = float(args.sigma)
    alpha = float(args.alpha)

    noisy_mses, teacher_mses = [], []
    recon_tl1, teach_tl1 = [], []

    saved = False
    for idx, batch in enumerate(dl):
        if idx >= args.k:
            break
        video = batch["video"].to(device)  # [-1,1] if normalize=minus1_1
        ref   = batch["ref"].to(device)

        B,T,C,H,W = video.shape
        video_flat = video.view(B*T, C, H, W)

        # VAE recon reference (decode(encode(video)))
        z0 = vae.encode(video_flat)                       # [B*T,latent,h,w]
        recon = vae.decode(z0).view(B,T,3,H,W).clamp(0,1)

        # add noise in latent + decode baseline
        z0_bt = z0
        noise = torch.randn_like(z0_bt)
        zn    = z0_bt + sigma*noise
        noisy = vae.decode(zn).view(B,T,3,H,W).clamp(0,1)

        # teacher one-step denoise (NO pure noise generation)
        id_emb = idenc(ref)                 # [B,latent]
        cond   = id_emb.repeat_interleave(T, dim=0)
        pred   = den(zn, cond)
        zh     = zn - alpha*pred
        teach  = vae.decode(zh).view(B,T,3,H,W).clamp(0,1)

        noisy_mse  = torch.mean((noisy - recon)**2).item()
        teach_mse  = torch.mean((teach - recon)**2).item()
        noisy_mses.append(noisy_mse)
        teacher_mses.append(teach_mse)

        # temporal on decoded (vs itself)
        r = recon[0].permute(0,2,3,1).cpu().numpy()
        t = teach[0].permute(0,2,3,1).cpu().numpy()
        recon_tl1.append(float(np.abs(r[1:] - r[:-1]).mean()))
        teach_tl1.append(float(np.abs(t[1:] - t[:-1]).mean()))

        # save only first clip side-by-side
        if not saved:
            def to_u8(x): return (x.clamp(0,1)*255).byte()
            recon_u8 = to_u8(recon[0]).permute(0,2,3,1).cpu().numpy()
            noisy_u8 = to_u8(noisy[0]).permute(0,2,3,1).cpu().numpy()
            teach_u8 = to_u8(teach[0]).permute(0,2,3,1).cpu().numpy()

            Image.fromarray(np.concatenate([recon_u8[0], noisy_u8[0], teach_u8[0]], axis=1))\
                .save(out_dir/"frame0_RECON_NOISY_TEACH.png")

            write_mp4(np.concatenate([recon_u8, teach_u8], axis=2), out_dir/"recon_vs_teacher.mp4")
            write_mp4(np.concatenate([recon_u8, noisy_u8], axis=2),  out_dir/"recon_vs_noisy.mp4")
            saved = True

    noisy_mse = float(np.mean(noisy_mses))
    teach_mse = float(np.mean(teacher_mses))

    report = {
        "ckpt": args.ckpt,
        "device": device,
        "k": args.k,
        "sigma": sigma,
        "alpha": alpha,
        "vs_vae_recon": {
            "noisy_mse": noisy_mse,
            "teacher_mse": teach_mse,
            "noisy_psnr_db": psnr_from_mse(noisy_mse),
            "teacher_psnr_db": psnr_from_mse(teach_mse),
            "delta_psnr_teacher_minus_noisy_db": psnr_from_mse(teach_mse) - psnr_from_mse(noisy_mse),
        },
        "temporal": {
            "teacher_temporal_l1": float(np.mean(teach_tl1)),
            "vae_recon_temporal_l1": float(np.mean(recon_tl1)),
        },
        "outputs": {
            "recon_vs_teacher_mp4": str(out_dir/"recon_vs_teacher.mp4"),
            "recon_vs_noisy_mp4": str(out_dir/"recon_vs_noisy.mp4"),
        }
    }
    (out_dir/"report_vs_vae.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
