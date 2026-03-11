import numpy as np

# Toy phoneme/viseme tokenization for the scaffold.
# Replace with MFA/Whisper-phoneme aligner or Montreal Forced Aligner pipeline.

VISEME_SET = ["sil", "AA", "AE", "EH", "IH", "OH", "OO", "FV", "MBP", "WQ", "L", "CH", "TH", "SZ", "R"]

def fake_viseme_tokens(num_frames: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, len(VISEME_SET), size=(num_frames,))
    return tokens
