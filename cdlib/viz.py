"""Small NumPy/Matplotlib helpers for images shown in the app. No image-processing software is involved."""
from __future__ import annotations

import numpy as np

# Error-map colours (paper-style): TP white, FP red, FN blue, TN black.
ERROR_COLOURS = {"TP": (255, 255, 255), "FP": (230, 50, 50), "FN": (50, 120, 255), "TN": (0, 0, 0)}


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RGB uint8 map of true/false positives/negatives for bool masks of shape (H, W)."""
    out = np.zeros((*gt.shape, 3), np.uint8)
    out[pred & gt] = ERROR_COLOURS["TP"]
    out[pred & ~gt] = ERROR_COLOURS["FP"]
    out[~pred & gt] = ERROR_COLOURS["FN"]
    return out


def boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels of `mask` that have at least one 4-neighbour outside the mask."""
    m = np.pad(mask, 1)
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return mask & ~inner


def outline(img: np.ndarray, mask: np.ndarray, colour=(255, 230, 0)) -> np.ndarray:
    """Draw the outline of `mask` on a copy of `img` (H, W, 3 uint8)."""
    out = img.copy()
    out[boundary(mask)] = colour
    return out


def tint(img: np.ndarray, mask: np.ndarray, colour=(255, 60, 60), alpha: float = 0.45) -> np.ndarray:
    out = img.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour, np.float32)
    return out.clip(0, 255).astype(np.uint8)


def colormap(arr: np.ndarray, cmap: str = "jet", vmin=None, vmax=None) -> np.ndarray:
    """Map a 2-D float array to RGB uint8 with a Matplotlib colour map (min-max scaled unless vmin/vmax given)."""
    import matplotlib

    a = arr.astype(np.float64)
    lo = a.min() if vmin is None else vmin
    hi = a.max() if vmax is None else vmax
    a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    return (matplotlib.colormaps[cmap](a.clip(0, 1))[..., :3] * 255).astype(np.uint8)


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour enlargement (keeps pixels visibly square when showing small feature maps)."""
    return img.repeat(factor, 0).repeat(factor, 1)
