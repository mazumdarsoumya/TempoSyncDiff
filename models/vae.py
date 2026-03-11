#/home/vineet/PycharmProjects/TempoSyncDiff/models/vae.py

import torch
import torch.nn as nn

class TinyVAE(nn.Module):
    """A tiny VAE-like encoder/decoder used for runnable scaffold.
    Replace with a real latent diffusion VAE (e.g., Stable Diffusion VAE) for real experiments.
    """
    def __init__(self, in_ch=3, latent_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, latent_dim, 1, 1, 0)
        )
        self.dec = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, in_ch, 3, 1, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)
