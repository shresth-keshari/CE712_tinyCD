"""LEVIR-CD256 loading and integrity checks for inference-only replication.

Images are returned as float tensors in [0, 1] (C, H, W). Each model applies its own normalisation
(see `cdlib.models.registry`), so one data pipeline feeds every model.

Masks are binarised with `> 0`, exactly like the official TinyCD loader (`clip(mask * 255, 0, 1)`).
A handful of LEVIR-CD256 masks contain stray values 156/254 besides 0/255; `> 0` counts them as change.
"""
from __future__ import annotations


import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .paths import DATA_DIR

SPLITS = ("train", "val", "test")
EXPECTED_SIZES = {"train": 7120, "val": 1024, "test": 2048}


def read_list(split: str, data_dir=DATA_DIR) -> list[str]:
    with open(data_dir / "list" / f"{split}.txt") as f:
        return [line.strip() for line in f if line.strip()]


def load_pair(name: str, data_dir=DATA_DIR):
    """Return (A, B, mask) as uint8 HxWx3, uint8 HxWx3, bool HxW."""
    a = np.array(Image.open(data_dir / "A" / name).convert("RGB"))
    b = np.array(Image.open(data_dir / "B" / name).convert("RGB"))
    m = np.array(Image.open(data_dir / "label" / name)) > 0
    if m.ndim == 3:
        m = m.any(axis=2)
    return a, b, m


def to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_u8).permute(2, 0, 1).float().div_(255.0)


class LevirCD(Dataset):
    """Returns ((A, B), mask, name). A, B: float (3, H, W) in [0, 1]; mask: bool (H, W)."""

    def __init__(self, split: str, names: list[str] | None = None, data_dir=DATA_DIR):
        self.split, self.data_dir = split, data_dir
        self.names = names if names is not None else read_list(split, data_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        a, b, m = load_pair(name, self.data_dir)
        a, b, m = to_tensor(a), to_tensor(b), torch.from_numpy(m)
        return (a, b), m, name


def integrity_report(data_dir=DATA_DIR) -> dict:
    """Split sizes, file presence, disjointness, image sizes and mask values for every listed pair."""
    report = {"splits": {}, "problems": []}
    seen: dict[str, str] = {}
    for split in SPLITS:
        names = read_list(split, data_dir)
        changed_px = total_px = no_change_pairs = 0
        stray_px = stray_pairs = 0
        change_fracs = []
        for n in names:
            if n in seen:
                report["problems"].append(f"{n} in both {seen[n]} and {split}")
            seen[n] = split
            for sub in ("A", "B", "label"):
                if not (data_dir / sub / n).exists():
                    report["problems"].append(f"missing {sub}/{n}")
            a = Image.open(data_dir / "A" / n)
            b = Image.open(data_dir / "B" / n)
            raw = np.array(Image.open(data_dir / "label" / n))
            if a.size != (256, 256) or b.size != (256, 256) or raw.shape[:2] != (256, 256):
                report["problems"].append(f"bad size {n}")
            stray = np.isin(raw, (0, 255), invert=True).sum()
            if stray:
                stray_px += int(stray)
                stray_pairs += 1
            m = raw > 0
            c = int(m.sum())
            changed_px += c
            total_px += m.size
            no_change_pairs += c == 0
            change_fracs.append(c / m.size)
        fr = np.array(change_fracs)
        report["splits"][split] = {
            "pairs": len(names),
            "expected_pairs": EXPECTED_SIZES[split],
            "changed_pixel_pct": 100 * changed_px / total_px,
            "no_change_pairs_pct": 100 * no_change_pairs / len(names),
            "median_change_pct_changed_pairs": float(100 * np.median(fr[fr > 0])),
            "max_change_pct": float(100 * fr.max()),
            "stray_mask_pixels": stray_px,
            "stray_mask_pairs": stray_pairs,
            "change_fraction_hist": np.histogram(100 * fr[fr > 0], bins=20, range=(0, 80))[0].tolist(),
        }
    return report
