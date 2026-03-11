import torch
import torch.nn.functional as F

def token_classification_loss(logits, targets):
    return F.cross_entropy(logits, targets)
