"""Object-level analysis (paper Section 4.4: are errors missed/hallucinated regions or poor boundaries?).

A changed building = a connected component of the ground-truth change mask (8-connectivity).
It counts as *detected* when at least 50% of its pixels are predicted as change.
False alarm: a test pair with **no** ground-truth change in which the model predicts >= 50 changed pixels.
One LEVIR-CD pixel is 0.5 m x 0.5 m = 0.25 m^2.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

DETECT_FRACTION = 0.5
FALSE_ALARM_MIN_PIXELS = 50
EIGHT = np.ones((3, 3), bool)
# Object-size bins in pixels (and m^2 at 0.25 m^2 / px).
SIZE_BINS = [(0, 100), (100, 400), (400, 1600), (1600, 10 ** 9)]
SIZE_LABELS = ["< 100 px (< 25 m²)", "100–400 px (25–100 m²)", "400–1600 px (100–400 m²)", "> 1600 px (> 400 m²)"]


def objects(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For one image: (sizes, detected) of every ground-truth change component."""
    lab, n = ndimage.label(gt, structure=EIGHT)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, bool)
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum_labels(np.ones_like(gt, dtype=np.int64), lab, idx).astype(int)
    hits = ndimage.sum_labels(pred.astype(np.int64), lab, idx)
    return sizes, hits >= DETECT_FRACTION * sizes


def predicted_objects(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    """(number of predicted components, number with no overlap at all with the ground truth) = hallucinations."""
    lab, n = ndimage.label(pred, structure=EIGHT)
    if n == 0:
        return 0, 0
    overlap = ndimage.sum_labels(gt.astype(np.int64), lab, np.arange(1, n + 1))
    return n, int((overlap == 0).sum())


def summarise(sizes: np.ndarray, detected: np.ndarray, pred_counts: np.ndarray, false_alarms: int,
              no_change_pairs: int) -> dict:
    out = {"objects": int(len(sizes)), "detected": int(detected.sum()),
           "object_recall_pct": float(100 * detected.mean()) if len(sizes) else float("nan"),
           "bins": []}
    for (lo, hi), lab in zip(SIZE_BINS, SIZE_LABELS):
        sel = (sizes >= lo) & (sizes < hi)
        out["bins"].append({"label": lab, "n": int(sel.sum()), "detected": int(detected[sel].sum()),
                            "recall_pct": float(100 * detected[sel].mean()) if sel.any() else float("nan")})
    n_pred, n_halluc = pred_counts.sum(0)
    out["predicted_objects"] = int(n_pred)
    out["hallucinated_objects"] = int(n_halluc)
    out["hallucinated_pct"] = float(100 * n_halluc / n_pred) if n_pred else float("nan")
    out["false_alarm_pairs"] = int(false_alarms)
    out["no_change_pairs"] = int(no_change_pairs)
    out["false_alarm_rate_pct"] = float(100 * false_alarms / no_change_pairs) if no_change_pairs else float("nan")
    return out
