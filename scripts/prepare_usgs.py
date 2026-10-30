"""Convert two extracted USGS Landsat C2 L2 scenes into an aligned Powai demo pair."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.paths import ROOT
from cdlib.usgs import prepare_pair


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", type=Path, required=True, help="Extracted earlier scene folder, with MTL.json")
    ap.add_argument("--after", type=Path, required=True, help="Extracted later scene folder, with MTL.json")
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "powai" / "pair")
    ap.add_argument("--size", type=int, default=256, help="Square crop width in native 30 m pixels")
    ap.add_argument("--low", type=float, default=0.0, help="Shared reflectance display minimum")
    ap.add_argument("--high", type=float, default=0.3, help="Shared reflectance display maximum")
    ap.add_argument("--min-valid-percent", type=float, default=90.0)
    args = ap.parse_args()
    try:
        m = prepare_pair(args.before, args.after, args.out, args.size, args.low, args.high, args.min_valid_percent)
    except ImportError:
        ap.exit(1, "Install preparation dependencies: python -m pip install -r requirements-usgs.txt\n")
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as e:
        ap.exit(1, f"Preparation stopped: {e}\n")
    print(f"Prepared {args.out}: {m['common_valid_percent']:.2f}% common valid pixels. Inspect alignment before inference.")


if __name__ == "__main__":
    main()
