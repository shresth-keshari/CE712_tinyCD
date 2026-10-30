import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib import paper  # noqa: E402

C.page_header(
    "7 · Size and speed",
    "TinyCD has 0.28 M parameters and 1.45 GFLOPs. BIT has 3.55 M (12.67x more) and 4.35 GFLOPs; ChangeFormer has "
    "41.02 M (146.50x more), FLOPs not reported. The abstract claims TinyCD is '13 to 140 times smaller' and "
    "'faster'. No latency or memory is reported.",
    "Table 3, Abstract",
)

eff = C.require(C.load_json("efficiency.json"), "scripts/profile_models.py")
ev = {k: C.load_json(f"eval_{k}_test.json") for k in C.MODEL_LABELS}
PK = {"tinycd": "TinyCD", "bit": "BIT", "changeformer": "ChangeFormer"}
m = eff["models"]
tiny_p = m["tinycd"]["params_used"]

rows = []
for k, lab in C.MODEL_LABELS.items():
    r, rep = m[k], paper.TABLE3_SIZE[PK[k]]
    rows.append({
        "model": lab,
        "params stored": r["params_total"],
        "params used in forward": r["params_used"],
        "params reported (M)": rep["params_m"],
        "ratio vs TinyCD (used)": round(r["params_used"] / tiny_p, 2),
        "ratio reported": rep["ratio"],
        "GFLOPs measured": round(r["gflops"], 3),
        "GFLOPs reported": rep["gflops"],
        "GPU latency ms (median)": round(r["latency_gpu_ms"]["median"], 1) if "latency_gpu_ms" in r else None,
        "CPU latency ms (median)": round(r["latency_cpu_ms"]["median"], 1),
        "peak GPU memory MiB": round(r["peak_gpu_mem_mib"], 1) if "peak_gpu_mem_mib" in r else None,
        "test F1 measured": round(ev[k]["metrics"]["f1"], 2) if ev[k] else None,
    })
df = pd.DataFrame(rows)
st.dataframe(df, hide_index=True)
env = eff["environment"]
st.caption(f"{C.badge('Measured', 'scripts/profile_models.py')} One pair of 256x256 images, batch 1, FP32, eager "
           f"PyTorch {env['torch']}. GPU: {env.get('gpu')}; CPU: {env.get('cpu')} "
           f"({m['tinycd'].get('cpu_threads')} threads). Latency = median of {m['tinycd']['latency_gpu_ms']['runs']} "
           f"(GPU) / {m['tinycd']['latency_cpu_ms']['runs']} (CPU) runs after warm-up. "
           f"{C.badge('Reported', 'Table 3')} columns are copied from the paper.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Accuracy vs size** (log scale)")
    ch = alt.Chart(df).mark_circle(size=160).encode(
        x=alt.X("params used in forward:Q", scale=alt.Scale(type="log"), title="parameters used (log)"),
        y=alt.Y("test F1 measured:Q", scale=alt.Scale(zero=False), title="test F1 (%)"),
        color=alt.Color("model:N", legend=None), tooltip=list(df.columns))
    st.altair_chart(ch + ch.mark_text(dy=-14).encode(text="model:N"), width="stretch")
with c2:
    st.markdown("**Accuracy vs GPU latency**")
    ch2 = alt.Chart(df).mark_circle(size=160).encode(
        x=alt.X("GPU latency ms (median):Q", title="GPU latency per pair (ms)"),
        y=alt.Y("test F1 measured:Q", scale=alt.Scale(zero=False), title="test F1 (%)"),
        color=alt.Color("model:N", legend=None), tooltip=list(df.columns))
    st.altair_chart(ch2 + ch2.mark_text(dy=-14).encode(text="model:N"), width="stretch")

st.subheader("Reading the numbers")
b, cf, t = m["bit"], m["changeformer"], m["tinycd"]
st.markdown(f"""
- **TinyCD parameters match exactly; FLOPs approximately match:** {t['params_used']:,} parameters (paper 285,128) and
  {t['gflops']:.3f} GFLOPs (paper 1.45).
- **"13-140x smaller" holds approximately.** Counting the parameters each model actually uses, BIT is
  **{b['params_used'] / tiny_p:.1f}x** and ChangeFormer **{cf['params_used'] / tiny_p:.1f}x** larger
  {C.badge('Derived')}. BIT's released code builds a full ResNet-18 ({b['params_total']:,} parameters stored) but its
  forward pass never calls ResNet `layer4` and `fc`. The {b['params_used']:,} we count as used is below the reported
  3.55 M; we could not determine how the BIT paper counted.
- **BIT FLOPs differ from the paper:** we measure {b['gflops']:.2f} GFLOPs vs 4.35 reported. We count with fvcore,
  where one multiply-add = 1 FLOP. That is the same convention that reproduces TinyCD's 1.45, but the reported
  BIT number may come from a different tool or convention. ChangeFormer, which the paper lists as "N.D", measures
  **{cf['gflops']:.1f} GFLOPs**.
- **Fewer FLOPs is not proportionally faster.** TinyCD needs {b['gflops'] / t['gflops']:.1f}x fewer FLOPs than BIT,
  but on our GPU it is only {b['latency_gpu_ms']['median'] / t['latency_gpu_ms']['median']:.1f}x faster at batch 1
  ({t['latency_gpu_ms']['median']:.1f} vs {b['latency_gpu_ms']['median']:.1f} ms). TinyCD is dominated by many small,
  memory-bound operations (depth-wise convolutions, instance normalisation, element-wise ops), so the GPU is not
  limited by arithmetic. Against ChangeFormer the gap is larger:
  {cf['latency_gpu_ms']['median'] / t['latency_gpu_ms']['median']:.1f}x on GPU and
  {cf['latency_cpu_ms']['median'] / t['latency_cpu_ms']['median']:.1f}x on CPU.
- **Memory:** peak GPU memory for one pair is {t['peak_gpu_mem_mib']:.0f} MiB (TinyCD) vs
  {cf['peak_gpu_mem_mib']:.0f} MiB (ChangeFormer).
""")
st.caption("Operators that fvcore cannot count (element-wise activations, softmax, etc.) are excluded for every model. "
           "That is the usual convention and is why FLOP counts are approximate: "
           + "; ".join(f"{C.MODEL_LABELS[k]}: {', '.join(v['flops_unsupported_ops'])}" for k, v in m.items()))

st.subheader("Time it yourself")
key = st.selectbox("Model", list(C.MODEL_LABELS), format_func=lambda k: C.MODEL_LABELS[k])
runs = st.slider("Runs", 5, 50, 20)
if st.button("Measure on the current device"):
    model = C.get_model(key)
    dev = model.device
    a = torch.rand(1, 3, 256, 256, device=dev)
    with torch.no_grad():
        for _ in range(3):
            model.predict(a, a)
        ts = []
        for _ in range(runs):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(a, a)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t0))
    st.metric(f"{C.MODEL_LABELS[key]} on {dev}", f"{np.median(ts):.1f} ms (median of {runs})")
    st.caption("Live timings depend on what else the machine is doing; the table above was measured on an idle machine.")

C.shows(
    "TinyCD's parameter count is exact and its FLOP count approximately matches; its size advantage over BIT and ChangeFormer is real; and its "
    "practical speed advantage is real but much smaller than the FLOP ratio suggests.",
    "Speed on other hardware (edge devices, other GPUs), with batching, or with optimised runtimes (TensorRT, "
    "ONNX). All timings are eager PyTorch on one laptop.",
)
C.footer()
