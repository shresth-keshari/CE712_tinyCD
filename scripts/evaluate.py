"""Evaluate one model on a full LEVIR-CD split (default: all 2,048 test pairs).

    python scripts/evaluate.py --model tinycd
    python scripts/evaluate.py --model changeformer --batch 4

Writes
  results/eval_<model>_<split>.json       global metrics, confusion counts, object-level summary, environment
  results/per_image_<model>_<split>.npz   per-image TP/FP/FN/TN (used for bootstrap CIs and the failure pages)

Threshold 0.5, no test-time tricks, nothing tuned on the test set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib import analysis  # noqa: E402
from cdlib.env import environment, pick_device  # noqa: E402
from cdlib.data import LevirCD  # noqa: E402
from cdlib.metrics import THRESHOLD, confusion, scores  # noqa: E402
from cdlib.models.registry import load_model  # noqa: E402
from cdlib.paths import RESULTS_DIR  # noqa: E402


def run(model_key: str, split: str, batch: int, device: str, workers: int) -> dict:
    model = load_model(model_key, device, check_hash=True)
    ds = LevirCD(split)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)

    counts, names = [], []
    sizes, detected, pred_objs = [], [], []
    false_alarms = no_change = 0
    t0 = time.time()
    for (a, b), m, nm in tqdm(dl, desc=f"{model_key}/{split}"):
        prob = model.predict(a.to(device), b.to(device))[:, 0]
        pred = (prob > THRESHOLD).cpu()
        counts.append(confusion(pred, m))
        names += list(nm)
        for p, g in zip(pred.numpy(), m.numpy()):
            s, d = analysis.objects(p, g)
            sizes.append(s)
            detected.append(d)
            pred_objs.append(analysis.predicted_objects(p, g))
            if not g.any():
                no_change += 1
                false_alarms += int(p.sum() >= analysis.FALSE_ALARM_MIN_PIXELS)
    elapsed = time.time() - t0

    counts = np.concatenate(counts)
    np.savez_compressed(RESULTS_DIR / f"per_image_{model_key}_{split}.npz", names=np.array(names), counts=counts)
    result = {
        "model": model_key,
        "meta": model.meta,
        "split": split,
        "pairs": len(ds),
        "threshold": THRESHOLD,
        "metrics": scores(counts),
        "counts": dict(zip(("tp", "fp", "fn", "tn"), counts.sum(0).tolist())),
        "objects": analysis.summarise(np.concatenate(sizes), np.concatenate(detected), np.array(pred_objs),
                                      false_alarms, no_change),
        "eval_seconds": elapsed,
        "batch": batch,
        "environment": environment(device),
    }
    (RESULTS_DIR / f"eval_{model_key}_{split}.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tinycd", "bit", "changeformer"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    r = run(args.model, args.split, args.batch, pick_device(args.device), args.workers)
    print(json.dumps(r["metrics"], indent=1))
    print(f"{r['pairs']} pairs in {r['eval_seconds']:.0f} s")
