"""Change-class metrics exactly as defined in the TinyCD paper (Section 4.2).

TP, FP, FN, TN are counted for the *change* class and summed over **all pixels of the whole split**
(one global confusion matrix, not a per-image average). This is also what the BIT / ChangeFormer /
TinyCD evaluation code does (`ConfuseMatrixMeter`).

    Pr  = TP / (TP + FP)
    Rc  = TP / (TP + FN)
    F1  = 2 Pr Rc / (Pr + Rc)          (= 1 / mean(1/Pr, 1/Rc), the harmonic mean)
    IoU = TP / (TP + FP + FN)
    OA  = (TP + TN) / (TP + TN + FP + FN)
"""
from __future__ import annotations

import numpy as np
import torch

THRESHOLD = 0.5


def confusion(pred: torch.Tensor | np.ndarray, gt: torch.Tensor | np.ndarray) -> np.ndarray:
    """Per-image counts. pred, gt: bool arrays (N, H, W) or (H, W). Returns int64 (N, 4) = TP, FP, FN, TN."""
    if isinstance(pred, np.ndarray):
        pred, gt = torch.from_numpy(pred), torch.from_numpy(gt)
    pred, gt = pred.bool(), gt.bool()
    if pred.ndim == 2:
        pred, gt = pred[None], gt[None]
    dims = (1, 2)
    tp = (pred & gt).sum(dims)
    fp = (pred & ~gt).sum(dims)
    fn = (~pred & gt).sum(dims)
    tn = (~pred & ~gt).sum(dims)
    return torch.stack([tp, fp, fn, tn], 1).cpu().numpy().astype(np.int64)


def scores(counts) -> dict:
    """Metrics (in %) from summed counts. `counts`: (4,) or (N, 4) array of TP, FP, FN, TN."""
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim == 2:
        c = c.sum(0)
    tp, fp, fn, tn = c
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    oa = (tp + tn) / c.sum() if c.sum() else 0.0
    return {"precision": 100 * pr, "recall": 100 * rc, "f1": 100 * f1, "iou": 100 * iou, "oa": 100 * oa}


def per_image_f1(counts: np.ndarray) -> np.ndarray:
    """F1 (%) of each image; NaN where the image has no change and no predicted change (F1 undefined)."""
    tp, fp, fn = counts[:, 0].astype(float), counts[:, 1].astype(float), counts[:, 2].astype(float)
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 100 * 2 * tp / den, np.nan)


def bootstrap_f1(counts: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Image-level bootstrap of the global F1: resample the test images with replacement,
    re-sum their confusion matrices, recompute F1. Returns (low, high) percentile interval in %."""
    rng = np.random.default_rng(seed)
    n = len(counts)
    idx = rng.integers(0, n, size=(n_boot, n))
    s = counts[idx].sum(1).astype(float)  # (n_boot, 4)
    f1 = 100 * 2 * s[:, 0] / (2 * s[:, 0] + s[:, 1] + s[:, 2])
    return float(np.percentile(f1, 100 * alpha / 2)), float(np.percentile(f1, 100 * (1 - alpha / 2)))


def paired_bootstrap_delta_f1(counts_a: np.ndarray, counts_b: np.ndarray, n_boot: int = 2000, seed: int = 0,
                              alpha: float = 0.05):
    """Bootstrap of F1(a) - F1(b) where both models are scored on the *same* resampled images.
    Returns (delta, low, high, fraction of resamples with delta <= 0)."""
    rng = np.random.default_rng(seed)
    n = len(counts_a)
    idx = rng.integers(0, n, size=(n_boot, n))

    def f1(c):
        s = c[idx].sum(1).astype(float)
        return 100 * 2 * s[:, 0] / (2 * s[:, 0] + s[:, 1] + s[:, 2])

    d = f1(counts_a) - f1(counts_b)
    delta = scores(counts_a)["f1"] - scores(counts_b)["f1"]
    return (float(delta), float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2))),
            float((d <= 0).mean()))
