#/home/vineet/PycharmProjects/TempoSyncDiff/models/teacher_unet.py

import torch
import torch.nn as nn

class TinyUNet(nn.Module):
    """A tiny conditional denoiser for latents.
    Conditions (viseme tokens + identity) are embedded and injected via FiLM-like modulation.
    This is NOT a production talking-head model; it is a runnable scaffold.
    """
    def __init__(self, latent_dim=64, cond_dim=64, base=128):
        super().__init__()
        self.inp = nn.Conv2d(latent_dim, base, 3, 1, 1)
        self.down = nn.Conv2d(base, base, 3, 1, 1)
        self.mid = nn.Conv2d(base, base, 3, 1, 1)
        self.up = nn.Conv2d(base, base, 3, 1, 1)
        self.out = nn.Conv2d(base, latent_dim, 3, 1, 1)

        self.cond = nn.Sequential(nn.Linear(cond_dim, base*2), nn.SiLU(), nn.Linear(base*2, base*2))
        self.act = nn.SiLU()

    def forward(self, z, cond):
        # z: [B,C,H,W], cond: [B,cond_dim]
        h = self.act(self.inp(z))
        gamma_beta = self.cond(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        h = self.act(self.down(h))
        h = h * (1 + 0.1*gamma) + 0.1*beta
        h = self.act(self.mid(h))
        h = self.act(self.up(h))
        return self.out(h)
