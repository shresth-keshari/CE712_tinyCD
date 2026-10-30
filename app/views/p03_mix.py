import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib.models.tinycd import interleave  # noqa: E402
from cdlib.viz import colormap, upscale  # noqa: E402

C.page_header(
    "3 · The Mix block: space-time fusion",
    "Given features X (t1) and Y (t2) with C channels each, Mix **interleaves** their channels into "
    "[X0, Y0, X1, Y1, ...] (Eq. 1) and applies a **grouped 3x3 convolution with C groups** (Eq. 2). Each kernel "
    "therefore sees one channel at t1 and *the same* channel at t2: a local space-time comparison. The paper argues "
    "that this (i) generalises pixel-wise subtraction (footnote 6) and (ii) needs only C(2·kh·kw) weights, against "
    "C(2C·kh·kw) for concatenation + convolution (Sec. 4.5.3).",
    "Sec. 3.3.1, 4.5.3",
)

# --------------------------------------------------------------------------------------------------------------
st.subheader("Step 1: interleave the channels")
c = st.slider("Number of channels C (toy example)", 2, 8, 4)
xs = [f"X{i}" for i in range(c)]
ys = [f"Y{i}" for i in range(c)]
inter = [v for pair in zip(xs, ys) for v in pair]
st.markdown(f"t1 features: `{' '.join(xs)}`  \nt2 features: `{' '.join(ys)}`  \n"
            f"interleaved (2C = {2 * c} channels): `{' '.join(inter)}`")
st.markdown(f"A grouped convolution with **groups = C = {c}** splits these {2 * c} channels into {c} groups of 2 "
            "consecutive channels. Group *g* contains exactly (X*g*, Y*g*). Its kernel has shape 2x3x3: one 3x3 "
            "slice for t1 and one for t2.")
xt = torch.arange(c).float().view(1, c, 1, 1)
yt = (100 + torch.arange(c)).float().view(1, c, 1, 1)
st.caption(f"Check with the real code (`cdlib.models.tinycd.interleave`): X = 0..{c - 1}, Y = 100..{99 + c} -> "
           f"{interleave(xt, yt).flatten().int().tolist()}")

# --------------------------------------------------------------------------------------------------------------
st.subheader("Step 2: why it is cheap")
k = st.select_slider("Kernel size k (kh = kw = k)", [1, 3, 5, 7], value=3)
cc = st.number_input("Channels C", 1, 512, 56)
mix_w = cc * 2 * k * k
cat_w = cc * 2 * cc * k * k
m1, m2, m3 = st.columns(3)
m1.metric("Mix weights  C·(2·k·k)", f"{mix_w:,}")
m2.metric("Concat + conv weights  C·(2C·k·k)", f"{cat_w:,}")
m3.metric("Ratio", f"{cat_w / mix_w:.0f}x  (= C)")
st.caption(f"{C.badge('Derived', 'formulas from Sec. 4.5.3')} Both also have C biases. The ratio of the weight "
           "counts is exactly C, so the saving grows with the channel width.")

var = C.load_json("variants.json")
if var:
    by = {v["id"]: v for v in var}
    rows = []
    for ch, where in ((3, "MAMB @256 (RGB)"), (24, "MAMB @128"), (32, "MAMB @64"), (56, "bottleneck @32")):
        rows.append({"location": where, "C": ch, "Mix: C(2·9)+C+1": ch * 18 + ch + 1,
                     "Concat+conv: C(2C·9)+C+1": ch * 2 * ch * 9 + ch + 1})
    df = pd.DataFrame(rows)
    st.markdown("The four fusion blocks in TinyCD (+1 = the PReLU slope that follows each conv):")
    st.dataframe(df, hide_index=True)
    d = int(df.iloc[:, 3].sum() - df.iloc[:, 2].sum())
    st.markdown(
        f"Swapping all four Mix blocks for concat+conv adds **{d:,}** parameters {C.badge('Derived')}: "
        f"285,128 + {d:,} = **{285_128 + d:,}**. Building that variant with our code gives "
        f"**{by['cat_all']['params']:,}** {C.badge('Measured', 'scripts/variants.py')}, and the paper reports "
        f"**368,468** {C.badge('Reported', 'Table 6')}. All three agree exactly."
    )

# --------------------------------------------------------------------------------------------------------------
st.subheader("Step 3: proof that Mix generalises subtraction (footnote 6)")
st.markdown("Set every 2x3x3 kernel to zero except its two centre weights, (+1 for t1, -1 for t2), with zero bias. "
            "Then the Mix convolution computes **exactly X - Y**. Try other centre weights: (0.5, 0.5) gives the "
            "average, (1, 0) copies t1. A trained Mix learns a full 3x3 space-time filter per channel instead.")
p = C.pair_picker("p03")
t = C.tinycd_trace(p["a"], p["b"], C.device())
lvl = st.segmented_control("Features to fuse", ["block 1 (24 ch, 128x128)", "block 2 (32 ch, 64x64)",
                                                "block 3 (56 ch, 32x32)"], default="block 2 (32 ch, 64x64)")
li = {"block 1 (24 ch, 128x128)": 1, "block 2 (32 ch, 64x64)": 2, "block 3 (56 ch, 32x32)": 3}[
    lvl or "block 2 (32 ch, 64x64)"]
X, Y = torch.from_numpy(t[f"F1_{li}"])[None], torch.from_numpy(t[f"F2_{li}"])[None]
ch = X.shape[1]
w1 = st.number_input("centre weight for t1", -2.0, 2.0, 1.0, 0.5)
w2 = st.number_input("centre weight for t2", -2.0, 2.0, -1.0, 0.5)
conv = torch.nn.Conv2d(2 * ch, ch, 3, padding=1, groups=ch, bias=False)
with torch.no_grad():
    conv.weight.zero_()
    conv.weight[:, 0, 1, 1] = w1
    conv.weight[:, 1, 1, 1] = w2
    out = conv(interleave(X, Y))
    target = w1 * X + w2 * Y
err = (out - target).abs().max().item()
st.markdown(f"Max |Mix(X, Y) - ({w1:g}·X + {w2:g}·Y)| over all {out.numel():,} values = **{err:.2e}** "
            f"{C.badge('Measured')} (float32 rounding level, i.e. identical).")
chan = int(np.argmax(np.abs(t[f"F1_{li}"] - t[f"F2_{li}"]).mean(axis=(1, 2))))
s = 256 // X.shape[-1]
cols = st.columns(4)
lo_in = float(min(X[0, chan].min(), Y[0, chan].min()))
hi_in = float(max(X[0, chan].max(), Y[0, chan].max()))
v = float(max(out[0, chan].abs().max(), target[0, chan].abs().max(), 1e-6))
panels = (("X (t1), ch %d" % chan, X[0, chan], "viridis", lo_in, hi_in),
          ("Y (t2)", Y[0, chan], "viridis", lo_in, hi_in),
          ("Mix output", out[0, chan], "RdBu_r", -v, v),
          (f"{w1:g}·X + {w2:g}·Y", target[0, chan], "RdBu_r", -v, v))
for col, (cap, arr, cm, vmin, vmax) in zip(cols, panels):
    col.image(upscale(colormap(arr.numpy(), cm, vmin, vmax), s), caption=cap, width="stretch")
st.caption("X and Y share one colour scale (viridis). The two right panels share a symmetric red/blue scale "
           "(white = 0), so identical images mean identical values.")
st.caption("In the real block this convolution is followed by PReLU and InstanceNorm. The equality above concerns "
           "the convolution, which is where the fusion happens.")

# --------------------------------------------------------------------------------------------------------------
st.subheader("What did the trained Mix kernels learn?")
net = C.get_model("tinycd").net
blocks = {"bottleneck @32 (56 kernels)": net._mixing_mask[2]._convmix[0],
          "MAMB @64 (32 kernels)": net._mixing_mask[1]._mixing._convmix[0],
          "MAMB @128 (24 kernels)": net._mixing_mask[0]._mixing._convmix[0],
          "MAMB @256 on RGB (3 kernels)": net._first_mix._mixing._convmix[0]}
which = st.selectbox("Mix block", list(blocks))
W = blocks[which].weight.detach().cpu().numpy()          # (C, 2, 3, 3)
s1, s2 = W[:, 0].sum((1, 2)), W[:, 1].sum((1, 2))
opp = float(np.mean(s1 * s2 < 0))
st.markdown(f"Kernels whose t1 and t2 halves have **opposite total sign** (subtraction-like): "
            f"**{100 * opp:.0f} %** ({int(round(opp * len(W)))} of {len(W)}) {C.badge('Measured', 'from the released weights')}. "
            "The others combine t1 and t2 with the same sign, so the network does not only learn differences.")
n_show = min(8, len(W))
idx = np.argsort(-np.abs(W).sum((1, 2, 3)))[:n_show]
vmax = float(np.abs(W[idx]).max())
cols = st.columns(n_show)
for c_, i in zip(cols, idx):
    pair_img = np.concatenate([W[i, 0], np.full((3, 1), np.nan), W[i, 1]], axis=1)
    rgb = colormap(np.nan_to_num(pair_img), "RdBu_r", -vmax, vmax)
    rgb[:, 3] = 255
    c_.image(upscale(rgb, 24), caption=f"kernel {i}: t1 | t2", width="stretch")
st.caption("Red = positive, blue = negative weight, same scale for all kernels shown (the largest by magnitude).")

C.shows(
    "Exactly how Mix works, that its cost formula matches the paper's parameter counts, and a numerical proof "
    "that subtraction is the special case the paper says it is.",
    "That Mix gives better *accuracy* than subtraction. The paper reports 91.05 vs 90.75 (Table 6), but that needs "
    "two trained networks and we did not retrain (see page 8).",
)
C.footer()
