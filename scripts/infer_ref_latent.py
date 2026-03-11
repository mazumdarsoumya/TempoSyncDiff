import os, json, math
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

def _pick_sd(ckpt: dict, keys):
    for k in keys:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k], k
    raise KeyError(f"Could not find any of {keys} in keys={list(ckpt.keys())}")

def write_mp4(frames_u8, mp4_path: Path, fps=25):
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp4_path.parent / (mp4_path.stem + "_frames")
    tmp.mkdir(parents=True, exist_ok=True)
    for i in range(frames_u8.shape[0]):
        Image.fromarray(frames_u8[i]).save(tmp / f"{i:04d}.png")
    cmd = ["ffmpeg","-y","-framerate",str(fps),"-i",str(tmp/"%04d.png"),"-pix_fmt","yuv420p",str(mp4_path)]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(tmp)

@torch.no_grad()
def main():
    teacher_path = os.environ.get("TEACHER","").strip()
    student_path = os.environ.get("STUDENT","").strip()
    ref_path     = os.environ.get("REF","").strip()
    out_dir      = Path(os.environ.get("OUT","results/infer_ref_latent")).resolve()
    split        = os.environ.get("SPLIT","val").strip().lower()
    batches      = int(os.environ.get("BATCHES","1"))
    use_cuda     = os.environ.get("CUDA","1").strip() == "1"

    assert teacher_path, "Set TEACHER=/path/to/teacher_best.pt"
    assert ref_path, "Set REF=/path/to/ref.jpg"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"

    # --- Load teacher ckpt (weights + cfg) ---
    tckpt = load_ckpt(teacher_path, map_location=device)
    cfg   = tckpt.get("cfg", {})
    data_block = cfg.get("data_val", cfg.get("data", {}))
    latent_dim = int(cfg.get("model", {}).get("latent_dim", 64))

    vae   = TinyVAE(latent_dim=latent_dim).to(device)
    idenc = IdentityEncoder(emb=latent_dim).to(device)
    teacher = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)

    vae.load_state_dict(tckpt["vae"])
    idenc.load_state_dict(tckpt["idenc"])
    teacher.load_state_dict(tckpt["denoiser"])
    vae.eval(); idenc.eval(); teacher.eval()

    # --- Load student if provided ---
    student = None
    if student_path and Path(student_path).exists():
        sckpt = load_ckpt(student_path, map_location=device)
        sd, sd_key = _pick_sd(sckpt, ["student_denoiser", "student", "denoiser"])
        student = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)
        student.load_state_dict(sd, strict=True)
        student.eval()

    # --- Build one batch from manifest val/train (NOT GT judging, just VAE-space sanity) ---
    from temposyncdiff.data.real_manifest_dataset import ManifestVideoConfig, ManifestVideoDataset
    dc = dict(data_block)
    dc.pop("augment", None)
    allowed = set(ManifestVideoConfig.__dataclass_fields__.keys())
    dc = {k:v for k,v in dc.items() if k in allowed}

    ds = ManifestVideoDataset(ManifestVideoConfig(**dc))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # Also load ref image (for identity condition)
    size = int(dc.get("image_size", 224))
    normalize = str(dc.get("normalize","minus1_1"))
    img = Image.open(ref_path).convert("RGB").resize((size,size))
    ref = torch.from_numpy(np.array(img)).float()/255.0
    ref = ref.permute(2,0,1).unsqueeze(0).to(device)
    if normalize == "minus1_1":
        ref = ref*2-1

    fps = int(os.environ.get("FPS","25"))

    all_psnr = {"teacher": [], "student": [], "noisy": []}
    temporal = {"vae": [], "teacher": [], "student": []}
    flicker  = {"vae": [], "teacher": [], "student": []}

    # helper
    def to_u8(x01):
        return (x01.clamp(0,1)*255.0).byte()

    picked_frame0 = False

    for bi, batch in enumerate(dl, start=1):
        if bi > batches:
            break

        video = batch["video"].to(device)  # [1,T,3,H,W], likely in [-1,1] if minus1_1
        B,T,C,H,W = video.shape
        video_flat = video.view(B*T, C, H, W)

        # Convert dataset video to 0..1 for VAE comparison
        if normalize == "minus1_1":
            gt01 = ((video + 1.0)/2.0).clamp(0,1)
        else:
            gt01 = video.clamp(0,1)

        # (A) VAE recon baseline (this is your “anchor”, not GT perception)
        z_clean = vae.encode(video_flat).view(B, T, -1, H//4, W//4)
        recon = vae.decode(z_clean.view(B*T, -1, H//4, W//4)).view(B, T, 3, H, W).clamp(0,1)

        # (B) Build conditioning from ref (identity)
        id_emb = idenc(ref)                 # [1,latent]
        cond   = id_emb.repeat_interleave(T, dim=0)  # [T,latent]

        # (C) Noisy decode (what teacher/student must improve over)
        z0_bt = z_clean.view(B*T, z_clean.shape[2], z_clean.shape[3], z_clean.shape[4])
        noise = torch.randn_like(z0_bt)
        zn = z0_bt + 0.2*noise
        noisy = vae.decode(zn).view(B, T, 3, H, W).clamp(0,1)

        # (D) Teacher denoise (aligned with training)
        t_pred = teacher(zn, cond)
        zt = zn - 0.1*t_pred
        t_out = vae.decode(zt).view(B, T, 3, H, W).clamp(0,1)

        # (E) Student denoise if present
        s_out = None
        if student is not None:
            s_pred = student(zn, cond)
            zs = zn - 0.1*s_pred
            s_out = vae.decode(zs).view(B, T, 3, H, W).clamp(0,1)

        # metrics vs VAE recon (your requested judge)
        def mse(a,b): return torch.mean((a-b)**2).item()
        def psnr(m): return 10.0*math.log10(1.0/max(m,1e-12))

        m_noisy = mse(noisy, recon); all_psnr["noisy"].append(psnr(m_noisy))
        m_t     = mse(t_out, recon); all_psnr["teacher"].append(psnr(m_t))
        if s_out is not None:
            m_s = mse(s_out, recon); all_psnr["student"].append(psnr(m_s))

        # temporal metrics (lower temporal_l1 = more stable)
        def temp_stats(x):
            v = x[0].permute(0,2,3,1).detach().cpu().numpy()  # [T,H,W,3]
            d = np.abs(v[1:] - v[:-1]).mean()
            f = v.mean(axis=(1,2,3)).std()
            return float(d), float(f)

        d_vae, f_vae = temp_stats(recon)
        d_t,   f_t   = temp_stats(t_out)
        temporal["vae"].append(d_vae); flicker["vae"].append(f_vae)
        temporal["teacher"].append(d_t); flicker["teacher"].append(f_t)
        if s_out is not None:
            d_s, f_s = temp_stats(s_out)
            temporal["student"].append(d_s); flicker["student"].append(f_s)

        # write artifacts for batch 1 only
        if not picked_frame0:
            gt_u8    = to_u8(gt01[0]).permute(0,2,3,1).cpu().numpy()
            recon_u8 = to_u8(recon[0]).permute(0,2,3,1).cpu().numpy()
            noisy_u8 = to_u8(noisy[0]).permute(0,2,3,1).cpu().numpy()
            t_u8     = to_u8(t_out[0]).permute(0,2,3,1).cpu().numpy()
            if s_out is not None:
                s_u8 = to_u8(s_out[0]).permute(0,2,3,1).cpu().numpy()

            # montage frame0: VAE | Noisy | Teacher | Student (if any)
            row = [recon_u8[0], noisy_u8[0], t_u8[0]]
            if s_out is not None:
                row.append(s_u8[0])
            montage = np.concatenate(row, axis=1)
            Image.fromarray(montage).save(out_dir/"frame0_VAE_NOISY_TEACHER_STUDENT.png")

            write_mp4(recon_u8, out_dir/"vae_recon.mp4", fps=fps)
            write_mp4(noisy_u8, out_dir/"noisy_decode.mp4", fps=fps)
            write_mp4(t_u8,     out_dir/"teacher_denoise.mp4", fps=fps)
            if s_out is not None:
                write_mp4(s_u8, out_dir/"student_denoise.mp4", fps=fps)

            picked_frame0 = True

    # aggregate report
    def avg(x): 
        return float(np.mean(x)) if len(x) else None

    report = {
        "device": device,
        "teacher_ckpt": teacher_path,
        "student_ckpt": student_path if (student_path and Path(student_path).exists()) else None,
        "split": split,
        "batches": batches,
        "vs_vae_recon": {
            "avg_psnr_db": {
                "noisy": avg(all_psnr["noisy"]),
                "teacher": avg(all_psnr["teacher"]),
                "student": avg(all_psnr["student"]) if len(all_psnr["student"]) else None
            },
            "teacher_minus_noisy_db": (avg(all_psnr["teacher"]) - avg(all_psnr["noisy"])) if len(all_psnr["noisy"]) else None,
            "student_minus_noisy_db": (avg(all_psnr["student"]) - avg(all_psnr["noisy"])) if len(all_psnr["student"]) else None,
            "student_minus_teacher_db": (avg(all_psnr["student"]) - avg(all_psnr["teacher"])) if len(all_psnr["student"]) else None,
        },
        "temporal": {
            "temporal_l1": {k: avg(v) for k,v in temporal.items()},
            "flicker_std": {k: avg(v) for k,v in flicker.items()},
        },
        "outputs": {
            "frame0_png": str(out_dir/"frame0_VAE_NOISY_TEACHER_STUDENT.png"),
            "vae_recon_mp4": str(out_dir/"vae_recon.mp4"),
            "noisy_decode_mp4": str(out_dir/"noisy_decode.mp4"),
            "teacher_denoise_mp4": str(out_dir/"teacher_denoise.mp4"),
            "student_denoise_mp4": str(out_dir/"student_denoise.mp4") if (student_path and Path(student_path).exists()) else None,
        }
    }

    (out_dir/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
