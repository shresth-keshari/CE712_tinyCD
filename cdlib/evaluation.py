"""A minimal evaluation loop shared by the probe and robustness scripts."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import LevirCD
from .metrics import THRESHOLD, confusion


def per_image_counts(predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], split: str = "test",
                     names: Optional[list[str]] = None, batch: int = 8, device: str = "cuda",
                     crop: int = 0, desc: str = "", workers: int = 2) -> np.ndarray:
    """`predict(a, b)` -> P(change) (B, 1, H, W). Returns (N, 4) TP/FP/FN/TN per image.
    `crop` > 0 excludes a border of that many pixels from the counts on every side."""
    dl = DataLoader(LevirCD(split, names=names), batch_size=batch, shuffle=False, num_workers=workers)
    out = []
    for (a, b), m, _ in tqdm(dl, desc=desc, leave=False):
        pred = (predict(a.to(device), b.to(device))[:, 0] > THRESHOLD).cpu()
        if crop:
            pred, m = pred[:, crop:-crop, crop:-crop], m[:, crop:-crop, crop:-crop]
        out.append(confusion(pred, m))
    return np.concatenate(out)
