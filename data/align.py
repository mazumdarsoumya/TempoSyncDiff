"""Audio-to-frame alignment utilities.

This repo ships with a runnable synthetic dataset, so alignment is *not*
required for the demo. For real THG experiments, you typically need
audio↔video alignment to derive phoneme/viseme tokens per video frame.

Design goals for this module:
  1) Provide a clean interface for alignment.
  2) Offer a deterministic fallback that works without external tools.
  3) Make it easy to plug in a proper forced-aligner (MFA/WhisperX/etc.).

The `align_audio_to_frames` function returns integer viseme IDs per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .audio_tokens import VISEME_SET


@dataclass
class AlignConfig:
    fps: int = 25
    viseme_set: Sequence[str] = tuple(VISEME_SET)
    # If you have a TextGrid/CTM/JSON alignment file, you can point to it.
    # This scaffold does not implement all formats; add your parser here.
    alignment_path: Optional[str] = None


def _energy_envelope(wav: np.ndarray, sr: int, hop: int = 320) -> np.ndarray:
    """Cheap energy envelope for fallback alignment."""
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    # frame-wise RMS
    n = len(wav)
    frames = max(1, n // hop)
    env = np.empty((frames,), dtype=np.float32)
    for i in range(frames):
        s = i * hop
        e = min(n, s + hop)
        seg = wav[s:e]
        env[i] = float(np.sqrt(np.mean(seg * seg) + 1e-8))
    # normalize
    env = (env - env.min()) / (env.max() - env.min() + 1e-8)
    return env


def align_audio_to_frames(
    wav: np.ndarray,
    sr: int,
    num_frames: int,
    cfg: Optional[AlignConfig] = None,
    seed: int = 0,
) -> np.ndarray:
    """Return per-frame viseme token ids.

    Args:
        wav: audio waveform (float or int), shape [N] or [N,channels].
        sr: sampling rate.
        num_frames: number of video frames to align to.
        cfg: alignment config.
        seed: used only for the deterministic fallback.

    Returns:
        tokens: int array of shape [num_frames], values in [0, len(viseme_set)-1].

    Notes:
        - If you provide `cfg.alignment_path`, this scaffold currently falls back
          to energy-based alignment unless you add a parser.
        - For a production pipeline, replace the fallback with MFA/WhisperX and
          map phonemes → visemes.
    """
    cfg = cfg or AlignConfig()

    # TODO: Implement real parsing when alignment_path is provided.
    # For now, always use the fallback.
    env = _energy_envelope(wav, sr)
    # downsample envelope to frames
    x = np.linspace(0.0, 1.0, num=len(env), dtype=np.float32)
    xf = np.linspace(0.0, 1.0, num=num_frames, dtype=np.float32)
    env_f = np.interp(xf, x, env)

    # Map energy to a small subset of mouth-open visemes.
    # Use 5 bins -> {sil, MBP, AA, OH, CH} (indices depend on VISEME_SET).
    # If these labels are missing, fall back to uniform random.
    preferred = ["sil", "MBP", "AA", "OH", "CH"]
    idx_map = []
    for p in preferred:
        try:
            idx_map.append(int(cfg.viseme_set.index(p)))
        except ValueError:
            idx_map = []
            break

    if not idx_map:
        rng = np.random.default_rng(int(seed))
        return rng.integers(0, len(cfg.viseme_set), size=(num_frames,), dtype=np.int64)

    bins = np.clip((env_f * (len(idx_map) - 1)).astype(np.int64), 0, len(idx_map) - 1)
    tokens = np.array([idx_map[b] for b in bins], dtype=np.int64)
    return tokens
