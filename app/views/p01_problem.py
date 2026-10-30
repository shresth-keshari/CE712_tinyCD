import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib.metrics import scores  # noqa: E402
from cdlib.viz import outline  # noqa: E402

C.page_header(
    "1 · The problem: bi-temporal change detection",
    "Change detection (CD) compares two co-registered images I1 and I2 of the same place, taken at times t1 and t2, "
    "and marks the pixels where a *relevant* change happened (here: buildings built or demolished). Irrelevant "
    "changes such as lighting, shadows and seasons must be ignored. TinyCD treats this as per-pixel binary "
    "classification (Sec. 1, 3.5, 4.2).",
    "Sec. 1, 4.1-4.2",
)

st.subheader("See it")
p = C.pair_picker("p01", allow_upload=False)
fade = st.slider("Fade from t1 (0) to t2 (1)", 0.0, 1.0, 0.0, 0.05)
blend = ((1 - fade) * p["a"].astype(np.float32) + fade * p["b"].astype(np.float32)).astype(np.uint8)
show_gt = st.toggle("Outline the ground-truth change (yellow)", value=True)
c1, c2 = st.columns([2, 1])
c1.image(outline(blend, p["mask"]) if show_gt else blend, width="stretch",
         caption=f"{p['name']}: blend of t1 and t2")
with c2:
    st.image((p["mask"] * 255).astype(np.uint8), caption="Ground-truth change mask (white = change)", width="stretch")
    st.markdown(f"Changed pixels in this pair: **{100 * p['mask'].mean():.2f} %** {C.badge('Measured')}")

st.subheader("The dataset: LEVIR-CD")
ds = C.load_json("data_summary.json")
st.markdown(
    f"LEVIR-CD (Chen & Shi, 2020) contains 637 pairs of 1024x1024 Google Earth images at 0.5 m/px, from cities in "
    f"Texas, USA, with annotated building changes. Like the paper (Sec. 4.1), we use the standard non-overlapping "
    f"256x256 crops: 7,120 / 1,024 / 2,048 train / val / test pairs {C.badge('Reported', 'Sec. 4.1')}. The BIT and "
    f"ChangeFormer checkpoints were trained on this same split."
)
if ds:
    rows = []
    for s, d in ds["splits"].items():
        rows.append({"split": s, "pairs": d["pairs"], "expected": d["expected_pairs"],
                     "changed pixels %": round(d["changed_pixel_pct"], 2),
                     "pairs with no change %": round(d["no_change_pairs_pct"], 2),
                     "median change % (changed pairs)": round(d["median_change_pct_changed_pairs"], 2),
                     "stray mask pixels": d["stray_mask_pixels"]})
    st.markdown(f"Integrity check and class balance {C.badge('Measured', 'scripts/check_data.py')}")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Integrity checks passed: split sizes match, every file exists in A/, B/ and label/, splits are "
               "disjoint, all images are 256x256. A few masks contain stray values (156 or 254) besides 0/255; "
               "like the official TinyCD loader we treat any non-zero value as change. In the test set this "
               "affects 69 of ~6.8 million changed pixels.")
    hist = ds["splits"]["test"]["change_fraction_hist"]
    edges = np.linspace(0, 80, 21)
    st.markdown("Test pairs with some change: distribution of the changed-pixel fraction")
    bins = [f"{edges[i]:.0f}-{edges[i + 1]:.0f}%" for i in range(len(hist))]
    # Interval labels are text; specify numeric bin order instead of alphabetical sorting.
    chart = alt.Chart(pd.DataFrame({"interval": bins, "pairs": hist})).mark_bar().encode(
        x=alt.X("interval:O", sort=bins, title="Changed pixels (%)"),
        y=alt.Y("pairs:Q", title="Test pairs"),
        tooltip=[alt.Tooltip("interval:O", title="Changed pixels"),
                 alt.Tooltip("pairs:Q", title="Test pairs", format="d")],
    )
    st.altair_chart(chart, width="stretch")

st.subheader("Why overall accuracy (OA) is misleading here")
ev = C.load_json("eval_tinycd_test.json")
if ev:
    c = ev["counts"]
    pos = c["tp"] + c["fn"]
    neg = c["fp"] + c["tn"]
    allneg = scores(np.array([0, 0, pos, neg]))
    st.markdown(
        f"Only **{100 * pos / (pos + neg):.2f} %** of test pixels are changed. A useless model that predicts "
        f"*no change everywhere* gets {C.badge('Derived', 'from the test-set pixel counts')}:"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall accuracy (OA)", f"{allneg['oa']:.2f} %")
    m2.metric("F1 of the change class", f"{allneg['f1']:.2f} %")
    m3.metric("IoU of the change class", f"{allneg['iou']:.2f} %")
    st.markdown(
        "That is why the paper, and this app, report **Precision, Recall, F1 and IoU of the *change* class**, "
        "computed from one confusion matrix summed over all test pixels (Sec. 4.2):\n\n"
        r"$\mathrm{Pr}=\frac{TP}{TP+FP}\quad \mathrm{Rc}=\frac{TP}{TP+FN}\quad "
        r"\mathrm{F1}=\frac{2\,\mathrm{Pr}\,\mathrm{Rc}}{\mathrm{Pr}+\mathrm{Rc}}\quad "
        r"\mathrm{IoU}=\frac{TP}{TP+FP+FN}\quad \mathrm{OA}=\frac{TP+TN}{TP+TN+FP+FN}$"
    )

C.shows(
    "What the task looks like, the dataset split we use (identical to the paper's), the class imbalance, and "
    "why F1/IoU of the change class are the meaningful metrics.",
    "Anything about model quality yet. That starts on page 5.",
)
C.footer()
