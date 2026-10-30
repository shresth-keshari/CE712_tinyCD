"""Discover real USGS Powai scene metadata; image downloads require EarthExplorer login."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.paths import ROOT
from cdlib.usgs import CENTER, SEARCH_BBOX, search_scenes, scene_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", nargs=2, default=["2015-01-01", "2015-02-28"], metavar=("START", "END"))
    ap.add_argument("--after", nargs=2, default=["2025-01-01", "2025-02-28"], metavar=("START", "END"))
    ap.add_argument("--max-cloud", type=float, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "powai" / "selection.json")
    args = ap.parse_args()
    groups = {label: search_scenes(*dates, args.max_cloud) for label, dates in
              (("before", args.before), ("after", args.after))}
    if any(not fs for fs in groups.values()):
        raise SystemExit("No matching scenes in one date window; widen dates or cloud filter.")
    # Lowest scene cloud, then closest calendar day to 10 February (same-season comparison).
    def rank(f):
        dt = datetime.fromisoformat(f["properties"]["datetime"].replace("Z", "+00:00"))
        return f["properties"]["eo:cloud_cover"], abs((dt.replace(year=2000).date() - datetime(2000, 2, 10).date()).days), f["id"]
    selected = {label: scene_summary(min(fs, key=rank)) for label, fs in groups.items()}
    result = dict(status="awaiting_local_downloads", queried_utc=datetime.now(timezone.utc).isoformat(),
                  center_lon_lat=CENTER, search_bbox=SEARCH_BBOX, selected=selected,
                  candidates={k: [scene_summary(f) for f in fs] for k, fs in groups.items()},
                  selection_rule="Landsat 8 Tier 1, WRS 148/047; lowest scene cloud, then nearest 10 February",
                  note="Metadata only. Inspect local QA and alignment after download; scene cloud is not local cloud.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for label, s in selected.items():
        print(label, s["product_id"], s["acquired"], f"scene cloud {s['scene_cloud_percent']}%")
    print(f"Saved public catalog metadata to {args.out}; no imagery downloaded.")


if __name__ == "__main__":
    main()
