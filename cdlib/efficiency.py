"""Size and speed: parameters (total and actually used in the forward pass), FLOPs, latency, peak GPU memory."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def params_total(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


@torch.no_grad()
def params_used(net: nn.Module, *inputs) -> tuple[int, int]:
    """(used, unused) parameter counts. A module's own parameters count as unused if the module is never called
    during a forward pass. Example: BIT builds a full ResNet-18 but its forward never calls `layer4` or `fc`."""
    called = set()
    hooks = [m.register_forward_pre_hook(lambda mod, inp: called.add(id(mod))) for m in net.modules()]
    try:
        net.eval()
        net(*inputs)
    finally:
        for h in hooks:
            h.remove()
    unused = sum(p.numel() for m in net.modules() if id(m) not in called for p in m.parameters(recurse=False))
    return params_total(net) - unused, unused


@torch.no_grad()
def gflops(net: nn.Module, *inputs) -> tuple[float, list[str]]:
    """fvcore FLOP count for one forward pass (fvcore counts one multiply-add as one FLOP, the usual convention
    in change-detection papers). Returns (GFLOPs, list of operators fvcore could not count)."""
    from fvcore.nn import FlopCountAnalysis
    net.eval()
    fca = FlopCountAnalysis(net, inputs)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    total = fca.total() / 1e9
    return total, sorted(fca.unsupported_ops().keys())


@torch.no_grad()
def latency_ms(net: nn.Module, *inputs, warmup: int = 10, runs: int = 50) -> dict:
    """Median and interquartile range of wall-clock latency per forward pass, in milliseconds."""
    net.eval()
    cuda = inputs[0].is_cuda
    for _ in range(warmup):
        net(*inputs)
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        net(*inputs)
        if cuda:
            torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    ts = np.array(ts)
    return {"median": float(np.median(ts)), "p25": float(np.percentile(ts, 25)),
            "p75": float(np.percentile(ts, 75)), "runs": runs}


@torch.no_grad()
def peak_memory_mb(net: nn.Module, *inputs) -> float:
    """Peak CUDA memory allocated by PyTorch (weights + activations) during one forward pass, in MiB."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net(*inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20
