"""Portable inference evidence: inputs, float probabilities, masks and provenance."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image


def png_bytes(arr: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="PNG")
    return out.getvalue()


def change_overlay(image: np.ndarray, pred: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    out = image.copy()
    out[pred] = np.rint(0.5 * out[pred] + 0.5 * np.array([255, 0, 80])).astype(np.uint8)
    if valid is not None:
        out[~valid] = [100, 100, 100]
    return out


def prediction_summary(prob: np.ndarray, threshold: float, valid: np.ndarray | None = None) -> dict:
    if prob.ndim != 2 or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError("Expected a finite HxW probability array in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    valid = np.ones(prob.shape, bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != prob.shape or not valid.any():
        raise ValueError("Valid mask must match the probability shape and contain usable pixels")
    n = int(valid.sum())
    changed = int(((prob > threshold) & valid).sum())
    return dict(threshold=float(threshold), valid_pixels=n, excluded_pixels=int(valid.size - n),
                predicted_changed_pixels=changed, predicted_changed_percent=100 * changed / n)


def evidence_zip(a: np.ndarray, b: np.ndarray, prob: np.ndarray, threshold: float, metadata: dict,
                 valid: np.ndarray | None = None, ground_truth: np.ndarray | None = None) -> bytes:
    summary = prediction_summary(prob, threshold, valid)
    valid = np.ones(prob.shape, bool) if valid is None else valid
    if a.shape != b.shape or a.shape[:2] != prob.shape:
        raise ValueError("Inputs and predictions must have matching dimensions")
    mask = (prob > threshold) & valid
    raw = io.BytesIO()
    np.save(raw, prob.astype(np.float32), allow_pickle=False)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, arr in (("before.png", a), ("after.png", b), ("mask.png", mask.astype(np.uint8) * 255),
                          ("valid.png", valid.astype(np.uint8) * 255),
                          ("overlay.png", change_overlay(b, mask, valid))):
            z.writestr(name, png_bytes(arr))
        if ground_truth is not None:
            z.writestr("ground_truth.png", png_bytes(ground_truth.astype(np.uint8) * 255))
        z.writestr("probability.npy", raw.getvalue())
        z.writestr("metadata.json", json.dumps(dict(metadata, prediction_summary=summary,
                    exported_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    return out.getvalue()
