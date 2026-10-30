import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib import paper  # noqa: E402

st.title("TinyCD, replicated and explained")
st.markdown(
    f"**Paper:** {paper.CITATION}  \n"
    "**Course:** CE712 mini project. Everything here (models, metrics, figures, this app) is written in Python. "
    "No GIS or remote-sensing software is used."
)

with st.container(border=True):
    st.markdown(
        "### The claim we are testing\n"
        "TinyCD detects building changes between two co-registered aerial images. The paper claims it reaches "
        f"**F1 = 91.05 %** on the LEVIR-CD test set {C.badge('Reported', 'Table 1')} with only **285,128 parameters** "
        f"and **1.45 GFLOPs** {C.badge('Reported', 'Table 3')}. That beats BIT (89.31 %) and ChangeFormer (90.40 %), which are "
        "about 12.7x and 146.5x larger."
    )

st.markdown("### What we measured")
ev = {k: C.load_json(f"eval_{k}_test.json") for k in C.MODEL_LABELS}
cols = st.columns(3)
for col, (k, lab) in zip(cols, C.MODEL_LABELS.items()):
    r = ev[k]
    rep = paper.TABLE1_LEVIR[{"tinycd": "TinyCD", "bit": "BIT", "changeformer": "ChangeFormer"}[k]]["f1"]
    with col:
        if r is None:
            st.metric(f"{lab} test F1", "not run yet")
            continue
        f1 = r["metrics"]["f1"]
        st.metric(f"{lab} · test F1 (measured)", f"{f1:.2f} %", delta=f"{f1 - rep:+.2f} vs reported {rep:.2f}",
                  delta_color="off", border=True)
st.caption("Measured on all 2,048 LEVIR-CD test pairs, one metric implementation, threshold 0.5, using each "
           "model's officially released LEVIR-CD weights. TinyCD runs in *our own* implementation of the architecture.")

st.markdown("### How to read this app")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        "**Understand the model**\n"
        "1. *The problem*: change detection, the dataset, and why accuracy (OA) is misleading.\n"
        "2. *Architecture*: TinyCD stage by stage, on a real image pair.\n"
        "3. *Mix block*: the paper's fusion idea, with a numerical proof that it generalises subtraction.\n"
        "4. *Attention masks*: the intermediate masks of paper Fig. 4.\n"
    )
with c2:
    st.markdown(
        "**Replicate the findings**\n"
        "5. *Live inference*: run all three models on any pair.\n"
        "6. *Results*: reported vs measured, with confidence intervals.\n"
        "7. *Size & speed*: parameters, FLOPs, latency, memory.\n"
        "8. *Variants & probes*: architecture sizes, published variant scores and inference-time interventions.\n"
        "9. *Failures*: object-level errors and a co-registration stress test.\n"
        "10. *IIT Bombay / Powai*: exploratory inference on USGS satellite imagery.\n"
        "11. *Limitations & honesty*: evidence, provenance and scope.\n"
        "12. *Demo guide*: setup, every control, verification and troubleshooting."
    )

st.markdown("### Scope, stated precisely")
st.markdown(
    "- We **re-implemented TinyCD** and verified it by loading the authors' released weights: every tensor name and "
    "shape matches, the parameter count is exactly 285,128, and the test score matches the paper.\n"
    "- We **re-measured** TinyCD, BIT and ChangeFormer under one protocol, using the authors' released weights.\n"
    "- This project is **inference-only replication** using frozen released checkpoints. There is no model "
    "training, fine-tuning or optimizer workflow.\n"
    "- The Powai demonstration uses real USGS imagery when downloaded and prepared. It is an exploratory "
    "application at a different resolution, with no labelled reference mask or accuracy claim.\n"
    "- We did **not** reproduce the ablation accuracies. They need one full training run per variant and no "
    "ablation weights were released. Page 8 shows the reported numbers and the parameter counts we *can* verify."
)
C.footer()
