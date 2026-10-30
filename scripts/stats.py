"""Bootstrap confidence intervals from the per-image confusion counts -> results/stats.json

* 95% CI of each model's test F1: resample the 2,048 test images with replacement (2,000 times), re-sum the
  confusion matrix, recompute F1, take the 2.5th and 97.5th percentiles.
* Paired difference F1(TinyCD) - F1(other) on the *same* resampled images. If the 95% CI excludes 0, the
  difference is unlikely to be explained by which test images happened to be in the test set.

Caveat: this captures test-set sampling variability only, not training randomness (one released checkpoint
per model; no retraining with different seeds).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.metrics import bootstrap_f1, paired_bootstrap_delta_f1, scores  # noqa: E402
from cdlib.models.registry import MODEL_KEYS  # noqa: E402
from cdlib.paths import RESULTS_DIR  # noqa: E402

N_BOOT, SEED = 2000, 0

if __name__ == "__main__":
    counts = {}
    for k in MODEL_KEYS:
        p = RESULTS_DIR / f"per_image_{k}_test.npz"
        if p.exists():
            d = np.load(p)
            counts[k] = (d["names"], d["counts"])
    ref_names = counts["tinycd"][0]
    for k, (names, _) in counts.items():
        assert (names == ref_names).all(), f"{k}: image order differs"
    out = {"n_boot": N_BOOT, "seed": SEED, "ci": {}, "paired_vs_tinycd": {}}
    for k, (_, c) in counts.items():
        lo, hi = bootstrap_f1(c, N_BOOT, SEED)
        out["ci"][k] = {"f1": scores(c)["f1"], "low": lo, "high": hi}
        print(f"{k:12s} F1={scores(c)['f1']:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
        if k != "tinycd":
            d, dlo, dhi, p_le0 = paired_bootstrap_delta_f1(counts["tinycd"][1], c, N_BOOT, SEED)
            out["paired_vs_tinycd"][k] = {"delta": d, "low": dlo, "high": dhi, "frac_le_0": p_le0}
            print(f"   TinyCD - {k}: {d:+.2f}  95% CI [{dlo:+.2f}, {dhi:+.2f}]")
    (RESULTS_DIR / "stats.json").write_text(json.dumps(out, indent=1))
