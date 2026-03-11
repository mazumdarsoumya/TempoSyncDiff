import torch
import torch.nn as nn

class IdentityEncoder(nn.Module):
    """Tiny identity encoder for scaffold.
    Replace with ArcFace/InsightFace embeddings for real experiments.
    """
    def __init__(self, in_ch=3, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Linear(32, emb)

    def forward(self, x):
        h = self.net(x).flatten(1)
        e = self.proj(h)
        e = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        return e
