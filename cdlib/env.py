"""Environment record stored alongside every result file."""
import platform
from datetime import datetime, timezone

import torch


def pick_device(pref: str = "auto") -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def environment(device: str) -> dict:
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cpu": platform.processor(),
        "os": platform.platform(),
    }
