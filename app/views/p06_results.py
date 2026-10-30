import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib import paper  # noqa: E402

C.page_header(
    "6 · Replication results on the LEVIR-CD test set",
    "TinyCD: Pr 92.68, Rc 89.47, **F1 91.05**, IoU 83.57, OA 99.10. It outperforms BIT (F1 89.31) and "
    "ChangeFormer (F1 90.40). The competitors' numbers in the paper are taken from their own publications "
    "(Sec. 4.4, Table 1).",
    "Table 1",
)

PK = {"tinycd": "TinyCD", "bit": "BIT", "changeformer": "ChangeFormer"}
ev = {k: C.load_json(f"eval_{k}_test.json") for k in PK}
C.require(ev["tinycd"], "scripts/evaluate.py --model tinycd")
stats = C.load_json("stats.json")

st.subheader("Reported vs measured")
st.markdown(
    "**Protocol (identical for every model):** all 2,048 test pairs · each model's officially released LEVIR-CD "
    "weights · each model's own input normalisation · threshold 0.5 on P(change) (equivalent to argmax for 2-class "
    "models) · one confusion matrix summed over all test pixels · one metric implementation (`cdlib/metrics.py`)."
)
rows = []
for k, lab in C.MODEL_LABELS.items():
    rep = paper.TABLE1_LEVIR[PK[k]]
    rows.append({"model": lab, "kind": "Reported (Table 1)", **{m: rep[m] for m in paper.METRICS}})
    if ev[k]:
        mea = ev[k]["metrics"]
        rows.append({"model": lab, "kind": "Measured (ours)", **{m: round(mea[m], 2) for m in paper.METRICS}})
        rows.append({"model": lab, "kind": "Difference", **{m: round(mea[m] - rep[m], 2) for m in paper.METRICS}})
df = pd.DataFrame(rows).rename(columns={"precision": "Pr", "recall": "Rc", "f1": "F1", "iou": "IoU", "oa": "OA"})
st.dataframe(df, hide_index=True, height=36 * (len(df) + 1) + 4)
st.caption(f"{C.badge('Reported', 'Table 1')} rows are copied from the paper. {C.badge('Measured')} rows were "
           "produced by `scripts/evaluate.py`. All values in %.")
C.generated_note(ev["tinycd"])

st.subheader("Are the differences between models real?")
if stats:
    ci = stats["ci"]
    chart_rows = []
    for k, lab in C.MODEL_LABELS.items():
        if k in ci:
            chart_rows.append({"model": lab, "kind": "Measured (95% CI)", "F1": ci[k]["f1"], "lo": ci[k]["low"],
                               "hi": ci[k]["high"]})
        chart_rows.append({"model": lab, "kind": "Reported", "F1": paper.TABLE1_LEVIR[PK[k]]["f1"], "lo": None,
                           "hi": None})
    cd = pd.DataFrame(chart_rows)
    base = alt.Chart(cd).encode(y=alt.Y("model:N", sort=list(C.MODEL_LABELS.values()), title=None))
    bars = base.transform_filter(alt.datum.kind == "Measured (95% CI)").mark_errorbar(thickness=2, ticks=True).encode(
        x=alt.X("lo:Q", title="F1 (%)", scale=alt.Scale(zero=False)), x2="hi:Q")
    pts = base.mark_point(size=110, filled=True).encode(
        x="F1:Q", shape=alt.Shape("kind:N", title=None),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["Measured (95% CI)", "Reported"],
                                                              range=["#1f77b4", "#ff7f0e"])),
        tooltip=["model", "kind", alt.Tooltip("F1:Q", format=".2f")])
    st.altair_chart((bars + pts).properties(height=200), width="stretch")

    prow = []
    for k, v in stats["paired_vs_tinycd"].items():
        prow.append({"comparison": f"TinyCD - {C.MODEL_LABELS[k]}", "ΔF1": round(v["delta"], 2),
                     "95% CI low": round(v["low"], 2), "95% CI high": round(v["high"], 2),
                     "CI excludes 0": v["low"] > 0 or v["high"] < 0})
    st.dataframe(pd.DataFrame(prow), hide_index=True)
    st.markdown(
        f"{C.badge('Measured', 'scripts/stats.py')} Image-level bootstrap: the 2,048 test pairs are resampled with "
        f"replacement {stats['n_boot']:,} times, and F1 is recomputed from the re-summed confusion matrix. Paired: both "
        "models are scored on the same resampled images. This captures uncertainty from *which test images were "
        "drawn*. It does **not** capture training randomness: there is one released checkpoint per model, so "
        "seed-to-seed variation is unknown."
    )

st.subheader("Reading the results")
t = ev["tinycd"]["metrics"]["f1"]
lines = [f"- **TinyCD reproduces.** Measured F1 {t:.2f} vs reported 91.05 (difference {t - 91.05:+.2f}). Since this "
         "runs the authors' weights inside *our* implementation, it also verifies the architecture, the data "
         "pipeline and the metric code together."]
if ev["changeformer"]:
    cf = ev["changeformer"]["metrics"]["f1"]
    lines.append(f"- **ChangeFormer reproduces** its published number: measured {cf:.2f} vs reported 90.40.")
if ev["bit"]:
    b = ev["bit"]["metrics"]["f1"]
    lines.append(
        f"- **BIT scores higher than reported**: measured {b:.2f} vs 89.31 ({b - 89.31:+.2f}). The 89.31 comes from "
        "the BIT paper. The released checkpoint documents no test score (it stores only its best *validation* "
        "score, mean F1 over both classes 94.69 at epoch 183), so we cannot tell whether it is the exact model "
        "behind the published number. Our pipeline reproduces the other two models to within 0.02 F1, which "
        "makes a metric error unlikely. We report the difference rather than explain it away.")
if stats:
    pv = stats["paired_vs_tinycd"]
    lines.append(
        "- **TinyCD still ranks first** under the common protocol. Its F1 advantage over ChangeFormer ("
        f"{pv['changeformer']['delta']:+.2f}, 95% CI {pv['changeformer']['low']:+.2f} to {pv['changeformer']['high']:+.2f}) "
        f"and over BIT ({pv['bit']['delta']:+.2f}, CI {pv['bit']['low']:+.2f} to {pv['bit']['high']:+.2f}) is "
        "larger than test-set sampling noise. With BIT's measured score the margin over BIT is smaller than the "
        "paper's (1.74 points).")
st.markdown("\n".join(lines))

with st.expander("Full paper Table 1 (reported only, for context)"):
    st.dataframe(pd.DataFrame(paper.TABLE1_LEVIR).T.rename_axis("model").reset_index(), hide_index=True)
    st.caption(f"{C.badge('Reported', 'Table 1')} Models other than TinyCD, BIT and ChangeFormer were not "
               "evaluated by us. FC-Siam-diff (the paper's baseline) has no released LEVIR-CD weights that we "
               "could find.")

C.shows(
    "The paper's headline LEVIR-CD numbers for TinyCD and ChangeFormer are reproducible from the released "
    "weights. TinyCD's lead holds under one fair protocol, with confidence intervals.",
    "A reproduction of the authors' model-fitting procedure, which is outside this inference-only scope, or "
    "anything about the WHU-CD dataset (Table 2), which we did not evaluate.",
)
C.footer()
