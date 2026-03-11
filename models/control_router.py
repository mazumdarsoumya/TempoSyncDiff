import torch
import torch.nn as nn

class ControlRouter(nn.Module):
    """Toy condition arbitration: outputs weights for (audio/viseme, identity).
    In real system: layer-wise guidance weights for multiple controls.
    """
    def __init__(self, cond_in: int, hidden: int = 128, n_controls: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_in, hidden), nn.SiLU(),
            nn.Linear(hidden, n_controls)
        )

    def forward(self, cond):
        w = self.net(cond)
        return torch.softmax(w, dim=-1)
