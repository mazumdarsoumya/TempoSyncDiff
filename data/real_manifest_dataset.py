#/home/vineet/PycharmProjects/TempoSyncDiff/data/real_manifest_dataset.py

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset

@dataclass
class ManifestVideoConfig:
    manifest: str
    image_size: int = 224
    num_frames: int = 50
    normalize: str = "minus1_1"  # or "0_1"

    # Base ref selection from the DRIVING clip
    ref_mode: str = "first"      # "first" or "middle"

    # Mode B: cross-identity reference replacement
    # With this probability, we will take ref frame from a DIFFERENT manifest item/video.
    mismatch_fraction: float = 0.0

    # How to pick ref from the mismatched identity ("first" or "middle" or "random")
    mismatch_ref_pick: str = "first"

def _read_all_frames(mp4: str, image_size: int):
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read: {mp4}")
    return np.stack(frames, axis=0)  # [N,H,W,3]

def _read_one_ref_frame(mp4: str, image_size: int, pick: str = "first"):
    """
    Read a single RGB frame (resized) from a video for use as reference.
    pick: "first" | "middle" | "random"
    """
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4}")

    pick = (pick or "first").lower()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if pick == "middle" and n > 0:
        target = n // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, fr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
    elif pick == "random" and n > 1:
        target = int(np.random.randint(0, n))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, fr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = cap.read()

    cap.release()
    if not ok or fr is None:
        raise RuntimeError(f"Could not read ref frame from: {mp4}")

    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    fr = cv2.resize(fr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return fr  # [H,W,3] uint8

class ManifestVideoDataset(Dataset):
    def __init__(self, cfg: ManifestVideoConfig):
        self.cfg = cfg
        self.items = []
        with open(cfg.manifest, "r") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))

        # normalize ref_mode aliases
        rm = (self.cfg.ref_mode or "first").lower()
        if rm in ("mismatched", "mismatch", "xid", "cross", "cross_id", "cross_identity"):
            # driving ref_mode still decides where ref would come from IF not replaced,
            # but in Mode B we may replace it with another identity ref.
            # keep driving ref default to "first" if user gave alias.
            self.cfg.ref_mode = "first"

    def __len__(self):
        return len(self.items)

    def _normalize_video(self, video_tchw: torch.Tensor) -> torch.Tensor:
        # video_tchw: [T,3,H,W] float
        if self.cfg.normalize == "minus1_1":
            return (video_tchw / 127.5) - 1.0
        else:
            return video_tchw / 255.0

    def __getitem__(self, idx):
        it = self.items[idx]

        # ---- driving video + viseme tokens (audio conditioning) from this sample ----
        frames = _read_all_frames(it["video_mp4"], self.cfg.image_size)  # [N,H,W,3]
        vis_path = it.get("viseme_npy", None)
        if vis_path is not None and os.path.exists(vis_path):
            vtok = np.load(vis_path).astype(np.int64)
        else:
            vtok = np.zeros((frames.shape[0],), dtype=np.int64)  # [N]

        N = frames.shape[0]
        T = self.cfg.num_frames

        # choose window for driving clip
        if N >= T:
            start = np.random.randint(0, N - T + 1)
            frames = frames[start:start+T]
            vtok = vtok[start:start+T]
        else:
            pad = T - N
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)], axis=0)
            vtok = np.concatenate([vtok, np.repeat(vtok[-1:], pad, axis=0)], axis=0)

        video = torch.from_numpy(frames).float().permute(0, 3, 1, 2)  # [T,3,H,W]
        video = self._normalize_video(video)

        # base ref from driving clip
        base_ref = video[0] if self.cfg.ref_mode == "first" else video[T//2]

        # ---- Mode B: replace reference with another identity sometimes ----
        ref = base_ref
        mf = float(getattr(self.cfg, "mismatch_fraction", 0.0) or 0.0)
        if mf > 0.0 and len(self.items) > 1 and (np.random.rand() < mf):
            # pick a different index
            j = int(np.random.randint(0, len(self.items) - 1))
            if j >= idx:
                j += 1
            it2 = self.items[j]
            try:
                fr = _read_one_ref_frame(
                    it2["video_mp4"],
                    self.cfg.image_size,
                    pick=str(getattr(self.cfg, "mismatch_ref_pick", "first") or "first"),
                )
                ref2 = torch.from_numpy(fr).float().permute(2, 0, 1)  # [3,H,W]
                ref2 = self._normalize_video(ref2.unsqueeze(0))[0]
                ref = ref2
            except Exception:
                # if mismatched video fails to read, fall back to base ref
                ref = base_ref

        viseme = torch.from_numpy(vtok)

        return {"video": video, "ref": ref, "viseme": viseme}
