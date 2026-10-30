"""Run frozen checkpoints sequentially on a prepared Powai bundle and save evidence."""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.env import environment, pick_device
from cdlib.exports import evidence_zip, prediction_summary
from cdlib.inference import predict_pair
from cdlib.models.registry import MODEL_KEYS, load_model
from cdlib.paths import ROOT, OUTPUTS_DIR
from cdlib.usgs import DISCLAIMER, load_prepared, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", type=Path, default=ROOT / "assets" / "powai" / "pair")
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "powai")
    ap.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    if not 0 <= args.threshold <= 1:
        ap.error("threshold must be between 0 and 1")
    if not (args.pair / "manifest.json").exists():
        ap.exit(1, "Powai imagery is not prepared. Follow USGS_WORKFLOW.md, then run scripts/prepare_usgs.py.\n")
    a, b, valid, manifest = load_prepared(args.pair)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(disclaimer=DISCLAIMER, pair_manifest_sha256=sha256(args.pair / "manifest.json"),
                  environment=environment(device), threshold=args.threshold, models={})
    for key in args.models:
        model = load_model(key, device, check_hash=True)
        # Warm-up separately so the stored time does not include model loading or first CUDA setup.
        x = torch.zeros(1, 3, 256, 256, device=device)
        model.predict(x, x)
        prob, secs = predict_pair(model, a, b)
        meta = dict(model=key, checkpoint=model.meta, device=str(model.device), seconds=secs,
                    pair=manifest, evidence="exploratory inference; no ground truth", disclaimer=DISCLAIMER)
        (args.out / f"{key}_evidence.zip").write_bytes(evidence_zip(a, b, prob, args.threshold, meta, valid))
        path = args.out / f"prob_{key}.npz"
        np.savez_compressed(path, probability=prob.astype(np.float32))
        report["models"][key] = dict(meta, **prediction_summary(prob, args.threshold, valid),
                                    probability_sha256=sha256(path))
        print(key, f"{report['models'][key]['predicted_changed_percent']:.2f}% predicted response on valid pixels")
        del model, x
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.out}. These are model responses, not validated changes or accuracy scores.")


if __name__ == "__main__":
    main()
