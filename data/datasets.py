from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from .audio_tokens import fake_viseme_tokens

@dataclass
class SyntheticConfig:
    num_identities: int = 8
    clips_per_id: int = 16
    frames: int = 32
    image_size: int = 128

class SyntheticTalkingHeadDataset(Dataset):
    """Synthetic dataset: generates simple 'face-like' patterns and mouth motion driven by fake viseme tokens.
    This makes the repo runnable without any external dataset.
    """
    def __init__(self, cfg: SyntheticConfig, seed: int = 123):
        self.cfg = cfg
        # Note: keep a dataset-level seed only for shuffling augmentation.
        # Per-sample identity parameters are generated deterministically from id_idx
        # so that identity is stable across epochs/workers.
        self.seed = int(seed)
        self.N = cfg.num_identities * cfg.clips_per_id

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        cfg = self.cfg
        id_idx = idx // cfg.clips_per_id

        # Identity seed -> fixed color palette and face shape params
        # Deterministic per-identity palette (stable across calls)
        rng_id = np.random.default_rng(int(id_idx) + 1337)
        base = rng_id.normal(0, 1, size=(3,))
        base = (base - base.min()) / (base.max() - base.min() + 1e-6)

        frames = []
        # Deterministic per-clip viseme stream
        visemes = fake_viseme_tokens(cfg.frames, seed=int(self.seed) + int(idx) + 17*int(id_idx))

        for t in range(cfg.frames):
            img = np.ones((cfg.image_size, cfg.image_size, 3), dtype=np.float32) * 0.15
            # face oval
            yy, xx = np.ogrid[:cfg.image_size, :cfg.image_size]
            cy, cx = cfg.image_size//2, cfg.image_size//2
            ry, rx = cfg.image_size*0.38, cfg.image_size*0.30
            mask = ((yy-cy)**2)/(ry**2) + ((xx-cx)**2)/(rx**2) <= 1.0
            img[mask] = 0.4 + 0.4*base

            # eyes
            for ex in [cx-20, cx+20]:
                ey = cy-15
                rr = 7
                em = (yy-ey)**2 + (xx-ex)**2 <= rr**2
                img[em] = 0.05

            # mouth openness from viseme token (toy)
            v = int(visemes[t])
            open_amt = (v % 5) / 5.0
            my = cy+25
            mw = 36
            mh = int(6 + 18*open_amt)
            y0, y1 = my-mh//2, my+mh//2
            x0, x1 = cx-mw//2, cx+mw//2
            img[y0:y1, x0:x1, :] = 0.08  # mouth cavity
            # teeth band (stabilize target)
            if open_amt > 0.2:
                img[y0:y0+2, x0:x1, :] = 0.9

            frames.append(img.transpose(2,0,1))  # CHW

        video = torch.tensor(np.stack(frames, axis=0))  # T,C,H,W
        ref = video[0].clone()
        tokens = torch.tensor(visemes, dtype=torch.long)
        return {"video": video, "ref": ref, "viseme": tokens, "id": torch.tensor(id_idx, dtype=torch.long)}
