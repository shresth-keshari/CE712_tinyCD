"""Inference-time probes of the *trained* TinyCD -> results/probes.json

Each probe removes one skip mask from the decoder of the released, trained model (the mask is replaced by 1,
i.e. no attention). We then measure test F1.

IMPORTANT: this is **not** the paper's ablation. Table 5 retrains a network without skip masks, and that
network learns to cope without them (F1 90.38). A probe removes a part the trained network was built
around, so the drop shows how much *this trained model relies* on the part, not what the part
contributes when the architecture is designed and trained without it.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.env import environment, pick_device  # noqa: E402
from cdlib.evaluation import per_image_counts  # noqa: E402
from cdlib.metrics import scores  # noqa: E402
from cdlib.models.registry import load_model  # noqa: E402
from cdlib.paths import RESULTS_DIR  # noqa: E402

PROBES = {
    "none": ("Unmodified released model", {}),
    "no_M64": ("Mask M64 replaced by 1", {"M64": "ones"}),
    "no_M128": ("Mask M128 replaced by 1", {"M128": "ones"}),
    "no_M256": ("Mask M256 replaced by 1", {"M256": "ones"}),
    "no_masks": ("All three masks replaced by 1", {"M64": "ones", "M128": "ones", "M256": "ones"}),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = pick_device(args.device)
    model = load_model("tinycd", dev)
    res = {"environment": environment(dev), "split": "test", "probes": {}}
    for pid, (desc, override) in PROBES.items():
        c = per_image_counts(lambda a, b, o=override: model.predict(a, b, mask_override=o), device=dev, desc=pid)
        res["probes"][pid] = {"description": desc, "metrics": scores(c)}
        print(f"{pid:9s} F1={res['probes'][pid]['metrics']['f1']:.2f}")
    (RESULTS_DIR / "probes.json").write_text(json.dumps(res, indent=1))
