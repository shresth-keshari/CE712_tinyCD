import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib.viz import colormap, upscale  # noqa: E402

C.page_header(
    "2 · TinyCD architecture, stage by stage",
    "A Siamese U-Net. A shared, ImageNet-pre-trained encoder (only the **first 4 blocks** of EfficientNet-B4, i.e. "
    "low-level features) processes both images. At each scale a **MAMB** (Mix + Attention Mask Block) compares t1 "
    "and t2 and produces a one-channel spatial mask. The deepest features are fused by a **Mix** block. The decoder "
    "upsamples x2 three times, multiplying by the masks, and a per-pixel **PW-MLP** classifier outputs the change "
    "probability (Sec. 3, Fig. 2).",
    "Sec. 3",
)

st.subheader("The data flow")
st.graphviz_chart("""
digraph G {
  rankdir=TB; nodesep=0.35; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#f2f4f8", fontname="Helvetica", fontsize=12];
  edge [fontname="Helvetica", fontsize=10];
  {rank=same; I1; I2; M256; U3; CL}
  {rank=same; E1; M128; U2}
  {rank=same; E2; M64; U1}
  {rank=same; E3; MIX}
  I1 [label="t1 image\\n3x256x256", fillcolor="#dbeafe"]; I2 [label="t2 image\\n3x256x256", fillcolor="#dbeafe"];
  E0 [label="EffNet-B4 block 0\\n(stem, shared)"]; E1 [label="block 1\\n24x128x128"];
  E2 [label="block 2\\n32x64x64"]; E3 [label="block 3\\n56x32x32"];
  M256 [label="MAMB on raw RGB\\n-> mask M256 1x256x256", fillcolor="#fef3c7"];
  M128 [label="MAMB\\n-> mask M128 1x128x128", fillcolor="#fef3c7"];
  M64 [label="MAMB\\n-> mask M64 1x64x64", fillcolor="#fef3c7"];
  MIX [label="Mix (bottleneck)\\n56x32x32", fillcolor="#fde68a"];
  U1 [label="Up x2, * M64\\n64x64x64", fillcolor="#dcfce7"]; U2 [label="Up x2, * M128\\n64x128x128", fillcolor="#dcfce7"];
  U3 [label="Up x2, * M256\\n32x256x256", fillcolor="#dcfce7"];
  CL [label="PW-MLP + sigmoid\\nP(change) 1x256x256", fillcolor="#fecaca"];
  I1 -> E0; I2 -> E0 [label="same weights"]; E0 -> E1 -> E2 -> E3;
  I1 -> M256; I2 -> M256; E1 -> M128 [label="t1,t2 feats"]; E2 -> M64; E3 -> MIX;
  MIX -> U1 -> U2 -> U3 -> CL; M64 -> U1; M128 -> U2; M256 -> U3;
}
""", width="stretch")
st.caption("Each encoder box is applied to both images with the *same* weights (Siamese). MAMB = Mix + PW-MLP down "
           "to one channel. Up = bilinear x2 upsampling, multiplication by the mask, depth-wise separable conv.")

st.subheader("Where the 285,128 parameters are")
net = C.get_model("tinycd").net
groups = {
    "Encoder: EfficientNet-B4 blocks 0-3 (ImageNet init, fine-tuned)": net._backbone,
    "MAMB @256 (raw RGB)": net._first_mix,
    "MAMB @128": net._mixing_mask[0],
    "MAMB @64": net._mixing_mask[1],
    "Mix bottleneck @32": net._mixing_mask[2],
    "Up block 32->64": net._up[0],
    "Up block 64->128": net._up[1],
    "Up block 128->256": net._up[2],
    "PW-MLP classifier": net._classify,
}
rows = [{"block": k, "parameters": sum(p.numel() for p in m.parameters())} for k, m in groups.items()]
df = pd.DataFrame(rows)
df["share %"] = (100 * df["parameters"] / df["parameters"].sum()).round(2)
c1, c2 = st.columns([3, 2])
c1.dataframe(df, hide_index=True)
c2.metric("Total (counted from our model)", f"{df['parameters'].sum():,}")
c2.markdown(f"{C.badge('Measured')} Paper: 285,128 {C.badge('Reported', 'Table 4')}")
enc = int(df["parameters"].iloc[0])
c2.markdown(f"**{100 * enc / df['parameters'].sum():.1f} %** of the parameters sit in the pre-trained encoder slice. "
            f"The change-specific part (MAMBs, Mix, decoder, classifier) has only "
            f"**{df['parameters'].sum() - enc:,}** parameters {C.badge('Derived', 'from the table')}.")

st.subheader("Look inside, on a real pair")
p = C.pair_picker("p02")
C.show_pair(p)
t = C.tinycd_trace(p["a"], p["b"], C.device())

stage = st.segmented_control("Encoder output to inspect", ["block 1 (128x128)", "block 2 (64x64)", "block 3 (32x32)"],
                             default="block 2 (64x64)")
si = {"block 1 (128x128)": 1, "block 2 (64x64)": 2, "block 3 (32x32)": 3}[stage or "block 2 (64x64)"]
f1, f2 = t[f"F1_{si}"], t[f"F2_{si}"]
st.markdown(f"Feature tensors have shape **{f1.shape[0]} x {f1.shape[1]} x {f1.shape[2]}** (channels x H x W) "
            f"for each image. Below: the 6 channels whose t1/t2 difference is largest for this pair.")
diff = np.abs(f1 - f2).mean(axis=(1, 2))
top = np.argsort(-diff)[:6]
scale = 256 // f1.shape[1]
for row_name, arr in (("t1", f1), ("t2", f2), ("|t1 - t2|", np.abs(f1 - f2))):
    cols = st.columns(7)
    cols[0].markdown(f"**{row_name}**")
    for c, ch in zip(cols[1:], top):
        vmin, vmax = (min(f1[ch].min(), f2[ch].min()), max(f1[ch].max(), f2[ch].max())) if row_name != "|t1 - t2|" \
            else (None, None)
        c.image(upscale(colormap(arr[ch], "viridis", vmin, vmax), scale), caption=f"ch {ch}", width="stretch")
st.caption("Same colour scale for t1 and t2 of each channel (viridis: dark = low, yellow = high). Where t1 and t2 "
           "differ (bottom row), the encoder responds differently to the two dates. Mix and MAMB blocks compare "
           "exactly these channel pairs.")

C.shows(
    "The structure of TinyCD as implemented here (identical to the authors' code, verified by strict weight "
    "loading), where its parameters are, and what its low-level features look like on real data.",
    "Why these design choices are better than alternatives. That is the ablation question (page 8).",
)
C.footer()
