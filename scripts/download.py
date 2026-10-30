"""Download LEVIR-CD256 and the three released LEVIR-CD checkpoints, then verify SHA-256 checksums.

Files that are already present are not downloaded again; their checksums are still verified.

    python scripts/download.py            # dataset + weights
    python scripts/download.py --weights  # weights only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.data import EXPECTED_SIZES, read_list  # noqa: E402
from cdlib.models.registry import WEIGHTS, sha256  # noqa: E402
from cdlib.paths import DATA_DIR, ROOT  # noqa: E402

DATASET_URL = "https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=1"   # link from the ChangeFormer README
DATASET_SHA256 = "2775377c7d7d60dc0f1408d11fd28bfd83933fc26f89c1fc677d4c81a0b52caa"
DOWNLOAD_DIR = ROOT / "data" / "_download"


def fetch(url: str, dest: Path):
    print(f"  downloading {url}\n    -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3',
                        '--retry-all-errors', '--connect-timeout', '30', '--max-time', '7200',
                        '--output', str(tmp), url], check=True)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Download produced an empty file: {url}")
        tmp.replace(dest)
    except (OSError, subprocess.CalledProcessError):
        tmp.unlink(missing_ok=True)
        raise


def extract_member(zip_path: Path, member_suffix: str, dest: Path):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(member_suffix)]
        if len(names) != 1:
            raise RuntimeError(f"Expected exactly one {member_suffix} in {zip_path}; found {len(names)}")
        with z.open(names[0]) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)


def verify_checksum(path, expected):
    got = sha256(path)
    if got != expected:
        raise ValueError(f"Checksum mismatch for {path}: expected {expected}, got {got}. Remove the corrupt download and retry.")


def safe_extract(z, destination):
    root = destination.resolve()
    for member in z.infolist():
        target = (root / member.filename).resolve()
        if not target.is_relative_to(root) or (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    z.extractall(root)


def dataset():
    print("LEVIR-CD256")
    if all((DATA_DIR / "list" / f"{s}.txt").exists() for s in EXPECTED_SIZES):
        print("  already extracted")
    else:
        zp = DOWNLOAD_DIR / "LEVIR-CD256.zip"
        if not zp.exists():
            fetch(DATASET_URL, zp)
        verify_checksum(zp, DATASET_SHA256)
        print("  extracting ...")
        with zipfile.ZipFile(zp) as z:
            safe_extract(z, DATA_DIR.parent)
    for s, n in EXPECTED_SIZES.items():
        names = read_list(s)
        got = len(names)
        print(f"  {s}: {got} pairs (expected {n}) {'OK' if got == n else 'MISMATCH'}")
        if got != n or len(set(names)) != n:
            raise ValueError(f"Invalid {s} dataset list")
        for name in names:
            for sub in ("A", "B", "label"):
                p = DATA_DIR / sub / name
                if not p.is_file() or p.stat().st_size == 0:
                    raise FileNotFoundError(f"Incomplete dataset: {p}")


def weights():
    for key, w in WEIGHTS.items():
        print(f"{key}: {w['source']}")
        p: Path = w["path"]
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            pending = p.with_suffix(p.suffix + ".pending")
            if "gdrive_id" in w:
                import gdown
                tmp = p.parent / "download.bin"
                if gdown.download(id=w["gdrive_id"], output=str(tmp), quiet=False) is None:
                    raise RuntimeError(f"Upstream Google Drive download failed for {key}; retry setup later")
                if zipfile.is_zipfile(tmp):
                    extract_member(tmp, p.name, pending)
                    tmp.unlink()
                else:
                    tmp.replace(pending)
            elif w["url"].endswith(".zip"):
                zp = p.parent / "download.zip"
                fetch(w["url"], zp)
                extract_member(zp, p.name, pending)
                zp.unlink()
            else:
                fetch(w["url"], pending)
            verify_checksum(pending, w["sha256"])
            pending.replace(p)
        got = sha256(p)
        ok = got == w["sha256"]
        print(f"  {p.relative_to(ROOT)} sha256 {got} {'OK' if ok else 'MISMATCH (expected ' + w['sha256'] + ')'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", action="store_true", help="weights only")
    args = ap.parse_args()
    if not args.weights:
        dataset()
    weights()
