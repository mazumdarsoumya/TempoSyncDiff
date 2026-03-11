import torch

def sync_proxy_loss(video, viseme_tokens):
    # This is a toy proxy: encourage average mouth ROI intensity to correlate with token-derived openness.
    B,T,C,H,W = video.shape
    y0 = int(H*0.55)
    roi = video[:,:, :, y0:, :]
    mouth_energy = roi.mean(dim=(2,3,4))  # [B,T]
    openness = (viseme_tokens.float() % 5) / 5.0
    openness = openness.to(mouth_energy.device)
    # normalize
    mouth_energy = (mouth_energy - mouth_energy.mean(dim=1, keepdim=True)) / (mouth_energy.std(dim=1, keepdim=True)+1e-6)
    openness = (openness - openness.mean(dim=1, keepdim=True)) / (openness.std(dim=1, keepdim=True)+1e-6)
    # want high correlation => minimize (1-corr)
    corr = (mouth_energy * openness).mean(dim=1)
    return (1 - corr).mean()
