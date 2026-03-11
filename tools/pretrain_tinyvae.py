import argparse, yaml, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from temposyncdiff.utils.io import ensure_dir, save_ckpt
from temposyncdiff.utils.seed import seed_everything
from temposyncdiff.models.vae import TinyVAE

def build_ds(data_block: dict, seed: int):
    data_cfg = dict(data_block)
    dtype = str(data_cfg.pop("type", "manifest")).strip()
    if dtype == "real":
        dtype = "manifest"
    if dtype != "manifest":
        raise ValueError("pretrain_tinyvae expects data.type=manifest")
    data_cfg.pop("augment", None)

    from temposyncdiff.data.real_manifest_dataset import ManifestVideoConfig, ManifestVideoDataset
    cfg = ManifestVideoConfig(**data_cfg)
    return ManifestVideoDataset(cfg)

def to_0_1(x: torch.Tensor) -> torch.Tensor:
    # If dataset returns [-1,1], map to [0,1]
    if x.min().item() < -0.1:
        x = (x + 1.0) / 2.0
    return x.clamp(0, 1)

def main(cfg):
    seed = int(cfg.get("seed", 123))
    seed_everything(seed)

    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_dir = ensure_dir(cfg.get("out_dir", "results/vae_pretrain"))
    ckpt_dir = ensure_dir(str(Path(out_dir) / "checkpoints"))

    latent_dim = int(cfg.get("latent_dim", 64))
    vae = TinyVAE(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=float(cfg.get("lr", 1e-4)))

    ds = build_ds(cfg["data"], seed=seed)
    dl = DataLoader(ds, batch_size=int(cfg.get("batch_size", 8)),
                    shuffle=True, num_workers=int(cfg.get("num_workers", 0)),
                    drop_last=True)

    epochs = int(cfg.get("epochs", 5))
    log_every = int(cfg.get("log_every", 200))
    global_step = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        vae.train()
        tot = 0.0
        n = 0

        for batch in dl:
            global_step += 1
            vid = batch["video"].to(device)   # [B,T,C,H,W]
            B,T,C,H,W = vid.shape
            x = vid.view(B*T, C, H, W)
            x = to_0_1(x)

            z = vae.encode(x)
            r = vae.decode(z)
            loss = F.l1_loss(r, x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot += float(loss.item())
            n += 1

            if log_every > 0 and (global_step % log_every == 0):
                print(f"[vae] epoch {epoch}/{epochs} step {global_step} l1={loss.item():.6f}")

        print(f"[vae] epoch {epoch}: l1={tot/max(n,1):.6f} time={(time.time()-t0)/60:.2f}min")
        save_ckpt({"vae": vae.state_dict(), "cfg": cfg}, str(Path(ckpt_dir) / "vae_pre.pt"))

    print("Saved:", str(Path(ckpt_dir) / "vae_pre.pt"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)
