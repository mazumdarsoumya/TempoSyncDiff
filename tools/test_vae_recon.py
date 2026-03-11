import torch, yaml
import numpy as np
from PIL import Image
from pathlib import Path

from temposyncdiff.utils.io import load_ckpt, ensure_dir
from temposyncdiff.models.vae import TinyVAE

def load_img(path, size):
    img = Image.open(path).convert("RGB").resize((size, size))
    x = torch.from_numpy(np.array(img)).float()/255.0
    x = x.permute(2,0,1).unsqueeze(0)   # [1,3,H,W]
    return x

@torch.no_grad()
def main():
    ckpt = load_ckpt("results/checkpoints/student.pt", map_location="cpu")
    teacher_cfg = ckpt.get("teacher_cfg", {})
    latent_dim = int(teacher_cfg.get("model", {}).get("latent_dim", 64))

    vae = TinyVAE(latent_dim=latent_dim)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()

    size = 224
    x = load_img("data/examples/vkr.jpg", size=size)  # change path if needed
    z = vae.encode(x)
    xhat = vae.decode(z).clamp(0,1)

    out_dir = Path(ensure_dir("results/vae_debug"))
    Image.fromarray((x[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(out_dir/"ref.png")
    Image.fromarray((xhat[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(out_dir/"recon.png")
    print("WROTE:", out_dir/"ref.png", out_dir/"recon.png")

if __name__ == "__main__":
    main()