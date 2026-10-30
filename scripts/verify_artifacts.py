"""Read-only consistency checks of saved inference evidence; writes a separate verification record."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.metrics import scores
from cdlib.models.registry import WEIGHTS, sha256
from cdlib.paths import RESULTS_DIR, DATA_DIR, DEMO_DIR


def verify(check_weights=False):
    checks, common_names = [], None
    for key in WEIGHTS:
        result = json.loads((RESULTS_DIR / f"eval_{key}_test.json").read_text())
        with np.load(RESULTS_DIR / f"per_image_{key}_test.npz", allow_pickle=False) as data:
            names, counts = data["names"], data["counts"]
            if counts.shape != (2048, 4) or len(set(names)) != 2048 or (counts < 0).any():
                raise ValueError(f"Invalid per-image counts for {key}")
            if not np.all(counts.sum(1) == 256 * 256):
                raise ValueError(f"Unexpected pixel totals for {key}")
            if common_names is not None and not np.array_equal(common_names, names):
                raise ValueError("Model test-pair lists differ")
            common_names = names.copy()
            if counts.sum(0).tolist() != [result["counts"][k] for k in ("tp", "fp", "fn", "tn")]:
                raise ValueError(f"Confusion totals disagree for {key}")
            if result["pairs"] != 2048 or result["threshold"] != 0.5:
                raise ValueError("Benchmark protocol changed")
            calculated = scores(counts)
            for metric, value in calculated.items():
                if not np.isclose(value, result["metrics"][metric], atol=1e-10, rtol=0):
                    raise ValueError(f"{key} {metric} does not match saved counts")
        check = dict(model=key, pairs=2048, metrics=calculated, counts_consistent=True)
        if check_weights:
            if sha256(WEIGHTS[key]["path"]) != WEIGHTS[key]["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {key}")
            check["checkpoint_sha256_verified"] = True
        checks.append(check)
    demos = json.loads((DEMO_DIR / "index.json").read_text())
    for demo in demos:
        for name in ("A.png", "B.png", "label.png", "prob_tinycd.png", "prob_bit.png", "prob_changeformer.png", "tinycd_masks.npz"):
            if not (DEMO_DIR / demo["dir"] / name).exists():
                raise ValueError(f"Missing demo asset: {demo['dir']}/{name}")
    local_test_list = DATA_DIR / "list" / "test.txt"
    if local_test_list.exists() and local_test_list.read_text().split() != common_names.tolist():
        raise ValueError("Local dataset test ordering differs from saved evidence")
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), benchmark_checks=checks,
                demo_pairs=len(demos), status="passed", scope="Saved evidence consistency, not a new full-dataset model evaluation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-weights", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "verification.json")
    args = ap.parse_args()
    report = verify(args.check_weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified three 2,048-pair results and {report['demo_pairs']} demo bundles; {args.out}")
