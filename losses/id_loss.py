import torch
import torch.nn.functional as F

def identity_loss(emb, ref_emb):
    # 1 - cosine similarity
    cos = (emb * ref_emb).sum(dim=1)
    return (1 - cos).mean()
