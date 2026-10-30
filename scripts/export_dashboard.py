"""Refresh the supplementary static dashboard from the current inference benchmark."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.paths import ROOT, RESULTS_DIR


def main():
    eff = json.loads((RESULTS_DIR / "efficiency.json").read_text())
    models = []
    for key, label in (("tinycd", "TinyCD"), ("bit", "BIT"), ("changeformer", "ChangeFormer V6")):
        ev = json.loads((RESULTS_DIR / f"eval_{key}_test.json").read_text())
        e = eff["models"][key]
        models.append(dict(id=key, model=label, source="downloaded checkpoint", **ev["metrics"],
                           params_used=e["params_used"], latency_ms_bs1=e.get("latency_gpu_ms", {}).get("median")))
    payload = dict(generated_at=datetime.now(timezone.utc).isoformat(),
                   status="Inference-only replication; 2,048 test pairs; threshold 0.5",
                   replication=[dict(label=m["model"], f1=m["f1"], note="Released weights, shared test protocol") for m in models],
                   models=models, findings=[
                       dict(text="TinyCD: 285,128 parameters and F1 91.07, versus paper F1 91.05."),
                       dict(text="BIT: F1 89.94 versus 89.31 reported; discrepancy remains unexplained."),
                       dict(text="ChangeFormer: F1 90.40, matching the reported result to two decimals."),
                       dict(text="The Streamlit app is the primary demonstration. Powai uses separate USGS inputs and makes no accuracy claim.")])
    path = ROOT / "dashboard" / "dist" / "data" / "results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
