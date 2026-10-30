"""Size and speed of each model on this machine -> results/efficiency.json

For one 256x256 image pair: parameters (stored and actually used in the forward pass), GFLOPs (fvcore),
latency at batch 1 on GPU and CPU (median of repeated runs), peak GPU memory.
The paper reports parameters and GFLOPs (Table 3) but no latency or memory.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib import efficiency as eff  # noqa: E402
from cdlib.env import environment  # noqa: E402
from cdlib.models.registry import MODEL_KEYS, load_model  # noqa: E402
from cdlib.paths import RESULTS_DIR  # noqa: E402


def profile(key: str, cpu_runs: int) -> dict:
    out = {"model": key}
    m = load_model(key, "cpu")
    a = torch.rand(1, 3, 256, 256)
    b = torch.rand(1, 3, 256, 256)
    na, nb = m.normalise(a), m.normalise(b)
    out["params_total"] = eff.params_total(m.net)
    out["params_used"], out["params_unused"] = eff.params_used(m.net, na, nb)
    out["gflops"], out["flops_unsupported_ops"] = eff.gflops(m.net, na, nb)
    out["latency_cpu_ms"] = eff.latency_ms(m.net, na, nb, warmup=3, runs=cpu_runs)
    out["cpu_threads"] = torch.get_num_threads()
    if torch.cuda.is_available():
        m.to("cuda")
        ga, gb = na.cuda(), nb.cuda()
        out["latency_gpu_ms"] = eff.latency_ms(m.net, ga, gb, warmup=10, runs=100)
        out["peak_gpu_mem_mib"] = eff.peak_memory_mb(m.net, ga, gb)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-runs", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"environment": environment(dev), "input": "one pair of 3x256x256 images, batch 1", "models": {}}
    for k in MODEL_KEYS:
        r = profile(k, args.cpu_runs)
        res["models"][k] = r
        print(k, {x: r[x] for x in ("params_total", "params_used", "gflops")},
              "gpu ms", r.get("latency_gpu_ms", {}).get("median"), "cpu ms", r["latency_cpu_ms"]["median"])
    (RESULTS_DIR / "efficiency.json").write_text(json.dumps(res, indent=1))
