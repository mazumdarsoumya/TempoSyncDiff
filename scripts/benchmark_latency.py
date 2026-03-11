import argparse, yaml
from pathlib import Path
import torch

from temposyncdiff.utils.seed import seed_everything
from temposyncdiff.utils.io import ensure_dir, load_ckpt
from temposyncdiff.utils.profiler import Timer
from temposyncdiff.models.teacher_unet import TinyUNet
from temposyncdiff.models.student_sampler import StudentSampler

def main(cfg):
    seed_everything(cfg.get("seed", 123))
    device = cfg.get("device","cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_dir = ensure_dir(cfg.get("out_dir","results"))
    bench_dir = ensure_dir(str(Path(out_dir)/"bench"))

    sckpt = load_ckpt(cfg["student_ckpt"], map_location=device)
    latent_dim = sckpt["teacher_cfg"]["model"]["latent_dim"]
    den = TinyUNet(latent_dim=latent_dim, cond_dim=latent_dim).to(device)
    den.load_state_dict(sckpt["student_denoiser"])
    den.eval()
    sampler = StudentSampler(den, steps=int(sckpt["cfg"]["student"]["steps"])).to(device)

    B = 8
    H = W = 32
    z = torch.randn(B, latent_dim, H, W, device=device)
    cond = torch.randn(B, latent_dim, device=device)

    warmup = int(cfg["benchmark"]["warmup"])
    iters = int(cfg["benchmark"]["iters"])

    for _ in range(warmup):
        _ = sampler.sample(z, cond)

    times = []
    for _ in range(iters):
        with Timer() as t:
            _ = sampler.sample(z, cond)
        times.append(t.dt)

    ms = 1000*sum(times)/len(times)
    fps = 1.0/(sum(times)/len(times))
    out = Path(bench_dir)/"latency.txt"
    out.write_text(f"device={device}\nsteps={sampler.steps}\nms_per_iter={ms:.3f}\nfps={fps:.2f}\n", encoding="utf-8")
    print(out.read_text())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)
