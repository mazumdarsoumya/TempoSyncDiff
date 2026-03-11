import torch
import torch.nn.functional as F

def temporal_l1(video):
    # video: [B,T,C,H,W]
    return (video[:,1:] - video[:,:-1]).abs().mean()

def mouth_flicker_proxy(video):
    # proxy: focus on lower face region
    B,T,C,H,W = video.shape
    y0 = int(H*0.55)
    roi = video[:,:, :, y0:, :]
    return (roi[:,1:] - roi[:,:-1]).abs().mean()
