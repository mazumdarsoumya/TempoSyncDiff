import numpy as np

def simple_mse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(((a-b)**2).mean())
