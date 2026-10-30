"""Project paths. Everything is relative to the repository root, so the project runs from any clone."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "LEVIR-CD256"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"          # small evidence files, intended for version control
DEMO_DIR = ROOT / "assets" / "demo"     # fixed pairs + fallback outputs, intended for version control
OUTPUTS_DIR = ROOT / "outputs" / "new"  # large, regenerable files (gitignored)

for _d in (RESULTS_DIR, DEMO_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
