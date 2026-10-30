"""Inference on arbitrary-size co-registered image pairs by 256x256 tiling.

The models were trained on 256x256 crops at 0.5 m/px. Larger images are processed at their native resolution
(no resizing, which would change the ground sampling distance): the pair is padded by reflection to a multiple
of 256, cut into non-overlapping 256x256 tiles, predicted tile by tile, stitched and cropped back.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import to_tensor
from .models.registry import ChangeModel

TILE = 256


@torch.no_grad()
def predict_pair(model: ChangeModel, a_u8: np.ndarray, b_u8: np.ndarray, batch: int = 4) -> tuple[np.ndarray, float]:
    """a_u8, b_u8: (H, W, 3) uint8 with identical shapes. Returns (P(change) as float32 (H, W), seconds)."""
    if a_u8.shape != b_u8.shape:
        raise ValueError(f"images must have the same size, got {a_u8.shape} and {b_u8.shape}")
    dev = model.device
    a, b = to_tensor(a_u8)[None].to(dev), to_tensor(b_u8)[None].to(dev)
    h, w = a.shape[-2:]
    ph, pw = (-h) % TILE, (-w) % TILE
    if ph or pw:
        mode = "reflect" if ph < h and pw < w else "replicate"
        a, b = F.pad(a, (0, pw, 0, ph), mode=mode), F.pad(b, (0, pw, 0, ph), mode=mode)
    H, W = a.shape[-2:]
    ta = a.unfold(2, TILE, TILE).unfold(3, TILE, TILE)            # (1, 3, nh, nw, T, T)
    tb = b.unfold(2, TILE, TILE).unfold(3, TILE, TILE)
    nh, nw = ta.shape[2], ta.shape[3]
    ta = ta.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, TILE, TILE)
    tb = tb.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, TILE, TILE)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = torch.cat([model.predict(ta[i:i + batch], tb[i:i + batch]) for i in range(0, len(ta), batch)])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    out = out.reshape(nh, nw, TILE, TILE).permute(0, 2, 1, 3).reshape(H, W)
    return out[:h, :w].float().cpu().numpy(), secs
