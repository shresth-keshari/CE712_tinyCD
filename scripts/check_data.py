"""Dataset integrity checks and summary statistics -> results/data_summary.json"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.data import integrity_report  # noqa: E402
from cdlib.paths import RESULTS_DIR  # noqa: E402

if __name__ == "__main__":
    rep = integrity_report()
    rep["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for s, d in rep["splits"].items():
        print(f"{s:5s} pairs={d['pairs']} (expected {d['expected_pairs']})  changed px={d['changed_pixel_pct']:.2f}%  "
              f"no-change pairs={d['no_change_pairs_pct']:.2f}%  stray mask px={d['stray_mask_pixels']}")
    print("problems:", rep["problems"] or "none")
    (RESULTS_DIR / "data_summary.json").write_text(json.dumps(rep, indent=1))
    sys.exit(1 if rep["problems"] else 0)
