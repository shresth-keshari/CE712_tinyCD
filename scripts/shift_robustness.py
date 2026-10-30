"""Co-registration sensitivity (paper Section 6, Limitations) -> results/shift.json

The paper argues that TinyCD's local, low-level design relies on the two images being spatially registered,
and that it "can show worse performances" than global (transformer) approaches under large shifts. The paper
does not test this. We test it with a controlled experiment:

  * shift the t2 image B by k pixels to the right and k pixels down (k = 0, 2, 4, 8, 16);
  * keep the ground truth unchanged;
  * score only the central 224x224 region (16-px border excluded for every k, so every k sees the same pixels
    and no wrapped-around content from `torch.roll` is scored).

This is a stress test that we designed, not a reproduction of a paper result.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.env import environment, pick_device  # noqa: E402
from cdlib.evaluation import per_image_counts  # noqa: E402
from cdlib.metrics import scores  # noqa: E402
from cdlib.models.registry import MODEL_KEYS, load_model  # noqa: E402
from cdlib.paths import RESULTS_DIR  # noqa: E402

SHIFTS = (0, 2, 4, 8, 16)
CROP = 16

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--models", nargs="+", default=list(MODEL_KEYS))
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    dev = pick_device(args.device)
    path = RESULTS_DIR / "shift.json"
    res = json.loads(path.read_text()) if path.exists() else {"results": {}}
    res.update({"environment": environment(dev), "shifts_px": list(SHIFTS), "border_excluded_px": CROP,
                "split": "test"})
    for key in args.models:
        model = load_model(key, dev)
        res["results"][key] = {}
        for k in SHIFTS:
            def pred(a, b, k=k):
                return model.predict(a, torch.roll(b, shifts=(k, k), dims=(2, 3)))
            c = per_image_counts(pred, device=dev, crop=CROP, batch=args.batch, desc=f"{key} k={k}")
            res["results"][key][str(k)] = scores(c)
            print(f"{key:12s} shift={k:2d}px F1={res['results'][key][str(k)]['f1']:.2f}")
        del model
        torch.cuda.empty_cache()
        path.write_text(json.dumps(res, indent=1))
