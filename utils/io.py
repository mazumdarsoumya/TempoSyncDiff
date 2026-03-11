from pathlib import Path
import torch

def ensure_dir(p: str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_ckpt(obj: dict, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    torch.save(obj, path)

def load_ckpt(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)
