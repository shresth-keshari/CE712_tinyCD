import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib.inference import predict_pair  # noqa: E402
from cdlib.metrics import per_image_f1  # noqa: E402
from cdlib.viz import error_map  # noqa: E402

C.page_header(
    "9 · Where it fails",
    "The authors conjecture that the gaps between models are 'more related to missing or hallucinated change "
    "regions, than region quality issues', and that both TinyCD and BIT sometimes fail in the same regions "
    "(Sec. 4.4, Fig. 3). As a limitation they note that TinyCD relies on co-registered images and 'can show worse "
    "performances' than global, transformer-based models when the images are spatially shifted (Sec. 6). "
    "This limitation is stated but not tested in the paper.",
    "Sec. 4.4, Sec. 6",
)

ev = {k: C.load_json(f"eval_{k}_test.json") for k in C.MODEL_LABELS}

# --------------------------------------------------------------------------------------------------------------
st.subheader("Object-level errors on the full test set")
st.markdown("A **changed building** = a connected region of the ground-truth change mask (8-connectivity). It counts "
            "as **detected** if at least 50% of its pixels are predicted as change. A **hallucinated region** is a "
            "predicted region with no overlap with any ground-truth change. A **false-alarm pair** is a test pair "
            "with no ground-truth change where the model predicts at least 50 changed pixels.")
rows, bins = [], []
for k, lab in C.MODEL_LABELS.items():
    if not ev[k]:
        continue
    o = ev[k]["objects"]
    rows.append({"model": lab, "GT objects": o["objects"], "detected": o["detected"],
                 "object recall %": round(o["object_recall_pct"], 2), "predicted regions": o["predicted_objects"],
                 "hallucinated regions": o["hallucinated_objects"], "hallucinated %": round(o["hallucinated_pct"], 2),
                 "false-alarm pairs": f"{o['false_alarm_pairs']} / {o['no_change_pairs']}",
                 "false-alarm rate %": round(o["false_alarm_rate_pct"], 2)})
    for bn in o["bins"]:
        bins.append({"model": lab, "size": bn["label"], "recall %": bn["recall_pct"], "n": bn["n"]})
st.dataframe(pd.DataFrame(rows), hide_index=True)
bdf = pd.DataFrame(bins)
order = list(dict.fromkeys(bdf["size"]))
ch = alt.Chart(bdf).mark_bar().encode(
    x=alt.X("size:N", sort=order, title="size of the changed building (pixels; 1 px = 0.25 m²)",
            axis=alt.Axis(labelAngle=0, labelLimit=260)),
    xOffset="model:N", y=alt.Y("recall %:Q", title="buildings detected (%)"), color="model:N",
    tooltip=["model", "size", alt.Tooltip("recall %:Q", format=".1f"), "n"])
st.altair_chart(ch.properties(height=320), width="stretch")
st.caption(f"{C.badge('Measured', 'scripts/evaluate.py')} All 2,048 test pairs, threshold 0.5.")
obs = []
for k, lab in C.MODEL_LABELS.items():
    if ev[k]:
        b0 = ev[k]["objects"]["bins"][0]
        missed = ev[k]["objects"]["objects"] - ev[k]["objects"]["detected"]
        obs.append(f"{lab}: {b0['recall_pct']:.0f}% of buildings < 100 px detected; "
                   f"{100 * (b0['n'] - b0['detected']) / missed:.0f}% of all its missed buildings are < 100 px")
st.markdown("**Observation** " + C.badge("Derived", "from the table") + "  \n" + "  \n".join(obs))
st.markdown("Small changes are where all three models lose most buildings. This is consistent with the paper's view "
            "that errors are mainly *missing (or hallucinated) regions*. We did not compute a boundary-quality "
            "metric, so we cannot weigh this against region quality directly.")

# --------------------------------------------------------------------------------------------------------------
st.subheader("Hardest test pairs for TinyCD")
pi = {k: C.load_npz(f"per_image_{k}_test.npz") for k in C.MODEL_LABELS}
if pi["tinycd"] is not None:
    names, cnt = pi["tinycd"]["names"], pi["tinycd"]["counts"]
    gt_pct = 100 * (cnt[:, 0] + cnt[:, 2]) / cnt.sum(1)
    f1s = {k: per_image_f1(v["counts"]) for k, v in pi.items() if v is not None}
    sel = gt_pct > 1.0
    order = np.argsort(f1s["tinycd"][sel])[:15]
    idx = np.flatnonzero(sel)[order]
    hard = pd.DataFrame({"pair": names[idx], "GT change %": gt_pct[idx].round(2),
                         **{f"F1 {C.MODEL_LABELS[k]}": v[idx].round(1) for k, v in f1s.items()}})
    st.markdown("Test pairs with more than 1% changed pixels and the lowest TinyCD F1 " + C.badge("Measured"))
    st.dataframe(hard, hide_index=True)
    both = np.mean((f1s["tinycd"][sel] < 50) & (f1s["bit"][sel] < 50)) * 100 if "bit" in f1s else None
    if both is not None:
        st.caption(f"Among pairs with > 1% change, {both:.1f}% have F1 < 50 for *both* TinyCD and BIT, i.e. failures "
                   "the two models share. The paper notes such shared failures qualitatively in Fig. 3. Some of the "
                   "hardest pairs have very small or ambiguous labelled changes; inspect them before drawing conclusions.")
    if C.dataset_available():
        pick = st.selectbox("Inspect a pair", hard["pair"].tolist())
        a, b, m = C.dataset_pair(pick)
        cols = st.columns(5)
        cols[0].image(a, caption="t1", width="stretch")
        cols[1].image(b, caption="t2", width="stretch")
        for col, key in zip(cols[2:], C.MODEL_LABELS):
            prob, _ = predict_pair(C.get_model(key), a, b)
            col.image(error_map(prob > 0.5, m), caption=C.MODEL_LABELS[key], width="stretch")
        st.caption("White = TP, red = FP (false alarm), blue = FN (missed), black = TN.")
    else:
        st.info("The LEVIR-CD dataset is not present on this machine; only the fixed demo pairs can be viewed.")

# --------------------------------------------------------------------------------------------------------------
st.subheader("Stress test: what if the images are not perfectly co-registered?")
sh = C.load_json("shift.json")
if sh:
    srows = [{"model": C.MODEL_LABELS[k], "shift (px)": int(s), "F1": v["f1"]}
             for k, d in sh["results"].items() for s, v in d.items()]
    sdf = pd.DataFrame(srows)
    ch = alt.Chart(sdf).mark_line(point=True).encode(
        x=alt.X("shift (px):Q", title="shift of the t2 image in x and y (pixels; 1 px = 0.5 m)"),
        y=alt.Y("F1:Q", title="test F1 (%), central 224x224 region"), color="model:N",
        tooltip=["model", "shift (px)", alt.Tooltip("F1:Q", format=".2f")])
    st.altair_chart(ch.properties(height=320), width="stretch")
    wide = sdf.pivot(index="model", columns="shift (px)", values="F1").round(2)
    st.dataframe(wide)
    st.markdown(
        f"{C.badge('Measured', 'scripts/shift_robustness.py')} *Our own experiment, not in the paper.* The t2 image "
        f"is shifted by k pixels right and down with `torch.roll`, and the ground truth is left unchanged. Only the "
        f"central 224x224 pixels are scored for every k, so no wrapped-around content is counted. Column 0 is "
        f"therefore slightly different from page 6 (border excluded).")
    if len(sh["results"]) == 3:
        lo = {k: v[str(max(sh["shifts_px"]))]["f1"] for k, v in sh["results"].items()}
        zero = {k: v["0"]["f1"] for k, v in sh["results"].items()}
        keep = {k: 100 * lo[k] / zero[k] for k in lo}
        spread = max(keep.values()) - min(keep.values())
        verdict = (
            f"The three curves are close: retained F1 differs by only {spread:.0f} percentage points between models. "
            "**In this test the transformer-based models are not clearly more robust than TinyCD.** The paper's "
            "concern (Sec. 6) is not confirmed for shifts up to 16 px on this dataset. It may still hold for other "
            "kinds of misregistration (rotation, scale, larger shifts), which we did not test."
            if spread < 5 else
            f"Retained F1 differs by {spread:.0f} percentage points between models; compare the curves above.")
        st.markdown(
            "**Reading:** misregistration hurts every model a lot, because a shifted unchanged building looks like "
            "change. At a 16 px (8 m) shift TinyCD keeps "
            f"{keep['tinycd']:.0f}% of its F1, BIT {keep['bit']:.0f}% and ChangeFormer {keep['changeformer']:.0f}%. "
            + verdict)
else:
    st.warning("Run `python scripts/shift_robustness.py` to generate this experiment.")

C.shows(
    "Which kinds of change the models miss (mostly small buildings), how often they hallucinate, shared failure "
    "cases, and a direct test of the paper's co-registration limitation.",
    "The causes of individual failures (some may be label errors), or performance on other datasets.",
)
C.footer()
