import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib.viz import colormap, error_map, upscale  # noqa: E402

C.page_header(
    "4 · Attention masks (paper Fig. 4)",
    "Each MAMB turns the Mix output into a one-channel mask with a small per-pixel MLP. The masks multiply the "
    "decoder features (skip connections). The paper's reading of Fig. 4: **M64** highlights the objects to track, "
    "**M128** acts more like an edge detector, and **M256** (computed directly on the RGB images) separates "
    "object classes such as buildings/streets, vegetation and shadows (Sec. 4.5.2).",
    "Sec. 4.5.2, Fig. 4",
)

p = C.pair_picker("p04")
C.show_pair(p)
t = C.tinycd_trace(p["a"], p["b"], C.device())

cmap = st.segmented_control("Colour map", ["jet", "viridis", "gray"], default="jet")
cols = st.columns(5)
for col, k in zip(cols, ("M256", "M128", "M64")):
    m = t[k][0]
    col.image(upscale(colormap(m, cmap or "jet"), 256 // m.shape[0]), caption=f"{k}  ({m.shape[0]}x{m.shape[1]})",
              width="stretch")
prob = t["P"][0]
cols[3].image(colormap(prob, cmap or "jet", 0, 1), caption="Output P(change)", width="stretch")
if p["mask"] is not None:
    cols[4].image(error_map(prob > 0.5, p["mask"]), caption="Prediction vs GT", width="stretch")
st.caption(f"{C.badge('Qualitative')} Each mask is min-max scaled for display. Error map: white = correct change, "
           "red = false alarm, blue = missed change, black = correct no-change.")

st.markdown("**Value ranges of the raw masks** " + C.badge("Measured"))
st.table({k: {"min": f"{t[k].min():.3f}", "max": f"{t[k].max():.3f}", "mean": f"{t[k].mean():.3f}"}
          for k in ("M256", "M128", "M64")})
st.markdown("The masks are *not* probabilities. In the authors' code each mask's last layer is a 1x1 conv "
            "followed by PReLU, so values can be negative or exceed 1. They act as learned per-pixel gains on the "
            "decoder features.")

C.shows(
    "The masks of the released model on any pair, so the paper's qualitative interpretation can be checked by eye. "
    "The paper's own evidence for this claim is also qualitative (one example in Fig. 4).",
    "A quantitative proof that the masks *mean* objects/edges/classes. Visual interpretation of internal "
    "activations is suggestive, not proof. Whether the masks help accuracy is the ablation question (page 8).",
)
C.footer()
