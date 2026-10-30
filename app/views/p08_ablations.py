import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib import paper  # noqa: E402

C.page_header(
    "8 · Architecture variants and inference probes",
    "Removing or replacing TinyCD's components lowers F1 on LEVIR-CD: no PW-MLP classifier 62.02 (Table 4), no "
    "skip masks 90.38 (Table 5), subtraction instead of Mix 90.75 (Table 6). Concatenation + conv matches Mix "
    "(91.06) but costs 29% more parameters (Table 6). EfficientNet-B4 is the best backbone tested (Table 9).",
    "Sec. 4.5, Tables 4-9",
)

st.info(
    "**Why we do not measure ablation accuracy.** Each row of the paper's ablation tables is a *separately trained* "
    "network. The authors released weights only for the final model. This project is inference-only, "
    "so every ablation F1 below is "
    f"{C.badge('Reported')}. What we *can* check without training is that each variant, built with our code, has "
    f"exactly the architecture the paper describes: its parameter count {C.badge('Measured')}."
)

var = C.require(C.load_json("variants.json"), "scripts/variants.py")
V = {v["id"]: v for v in var}


def row(vid, name, f1_rep, table):
    v = V[vid]
    return {"variant": name, "reported F1": f1_rep, "table": table, "params reported": v["paper_params"],
            "params (our build)": v["params"],
            "params match": "—" if v["match"] is None else ("✅ exact" if v["match"] else f"❌ {v['diff']:+,}"),
            "GFLOPs (our build)": round(v["gflops"], 3)}


st.subheader("Fusion strategy (Tables 6 and 7)")
t7 = paper.TABLE7_LEVIR
fusion = pd.DataFrame([
    row("paper", "Mix in skips + bottleneck (TinyCD)", t7["paper"]["f1"], "6/7"),
    row("sub_all", "Subtraction in skips + bottleneck", t7["sub_all"]["f1"], "6/7a"),
    row("sub_skips", "Subtraction in skips only", t7["sub_skips"]["f1"], "7a"),
    row("sub_bottleneck", "Subtraction in bottleneck only", t7["sub_bottleneck"]["f1"], "7a"),
    row("cat_all", "Concat+conv in skips + bottleneck", t7["cat_all"]["f1"], "6/7b"),
    row("cat_skips", "Concat+conv in skips only", t7["cat_skips"]["f1"], "7b"),
    row("cat_bottleneck", "Concat+conv in bottleneck only", t7["cat_bottleneck"]["f1"], "7b"),
])
st.dataframe(fusion, hide_index=True)
st.caption("Table 7 marks each location with a check or a cross. Its caption says the check marks a replaced "
           "location, but the all-check row has 285,128 parameters, i.e. the unmodified model. We therefore matched "
           "rows to variants by parameter count, which our builds reproduce exactly. That makes the mapping above "
           "unambiguous.")

st.subheader("Classifier, skip masks and backbone (Tables 4, 5, 9)")
t9 = paper.TABLE9_LEVIR
other = pd.DataFrame([
    row("no_classifier", "No final PW-MLP classifier (sliced w/o MLP)", paper.TABLE4_LEVIR["Sliced w/o MLP"]["f1"], "4"),
    row("no_skip", "No MAMB skip masks", paper.TABLE5_LEVIR["No skip"]["f1"], "5"),
    row("b0", "EfficientNet-B0 backbone", t9["efficientnetb0"]["f1"], "9"),
    row("b1", "EfficientNet-B1 backbone", t9["efficientnetb1"]["f1"], "9"),
    row("b2", "EfficientNet-B2 backbone", t9["efficientnetb2"]["f1"], "9"),
    row("b3", "EfficientNet-B3 backbone", t9["efficientnetb3"]["f1"], "9"),
])
st.dataframe(other, hide_index=True)
st.markdown(f"""
- **No skip masks:** the paper gives no parameter count. Ours ({V['no_skip']['params']:,}) comes from removing the three
  MAMBs and the mask multiplications. This is our reading of "a model without them", so treat it as {C.badge('Derived')}.
- **Backbones B0-B2** are each exactly {-V['b0']['diff']} parameters below Table 9, while **B3 matches exactly**.
  B0-B2 have 16 and 24 channels at the 128 px and 64 px stages; B3 and B4 have 24 and 32. The paper does not
  publish the PW-MLP widths of the MAMBs, so the authors probably used different widths for those narrower stages
  than our rule [C, C/2, C/4]. Note that the B0-B3 builds only check sizes; this project never uses them otherwise.
- **Full (un-sliced) EfficientNet-B4 variants** of Table 4 (17.7 M parameters) were not built: the paper does not
  describe how the deeper stages connect to the MAMBs.
""")

st.subheader("Reported F1 of every variant")
plot = pd.concat([fusion, other])[["variant", "reported F1", "params (our build)"]]
ch = alt.Chart(plot).mark_bar(clip=True).encode(
    y=alt.Y("variant:N", sort="-x", title=None, axis=alt.Axis(labelLimit=360)),
    x=alt.X("reported F1:Q", scale=alt.Scale(domain=[60, 92]), title="reported test F1 on LEVIR-CD (%)"),
    color=alt.condition(alt.datum.variant == "Mix in skips + bottleneck (TinyCD)", alt.value("#e45756"),
                        alt.value("#4c78a8")),
    tooltip=list(plot.columns))
st.altair_chart(ch.properties(height=360), width="stretch")
stats = C.load_json("stats.json")
hw = ""
if stats:
    halves = [(v["high"] - v["low"]) / 2 for v in stats["ci"].values()]
    hw = f" (95% CI half-widths of {min(halves):.2f}-{max(halves):.2f} F1 points, page 6)"
st.caption(f"{C.badge('Reported', 'Tables 4-7, 9')} Single training runs; the paper gives no error bars. Several "
           "differences (e.g. 91.05 vs 91.06, 90.75 vs 90.71) are far smaller than the test-sampling uncertainty "
           f"we measure for a single model{hw}, so on their own they do not rank the variants.")

st.subheader("Probes on the trained model (not an ablation)")
pr = C.load_json("probes.json")
if pr:
    base = pr["probes"]["none"]["metrics"]["f1"]
    prow = [{"probe": v["description"], "test F1": round(v["metrics"]["f1"], 2),
             "change vs unmodified": round(v["metrics"]["f1"] - base, 2),
             "precision": round(v["metrics"]["precision"], 2), "recall": round(v["metrics"]["recall"], 2)}
            for v in pr["probes"].values()]
    st.dataframe(pd.DataFrame(prow), hide_index=True)
    st.markdown(
        f"{C.badge('Probe')} {C.badge('Measured', 'scripts/probes.py')} We take the *released, trained* TinyCD and "
        "replace a skip mask by 1 (the decoder features pass through unweighted), then re-score the full test "
        "set. **This answers a different question from the paper's Table 5.** A network trained *without* masks "
        "learns to cope (paper: F1 90.38). Removing a mask after training breaks something the network was built "
        "around. A large drop here shows the trained model *relies* on its masks. It does not show how much the "
        "masks are *worth* as a design choice.\n\n"
        "Why M256 is so critical: on all six demo pairs the raw M256 mask lies between about -0.17 and -0.07 "
        "(page 4), i.e. it is negative everywhere. Replacing it by +1 flips the sign of the features entering the "
        "last Up block. That block applies PReLU, which treats positive and negative inputs differently, so the "
        "flip is not undone by the InstanceNorm that follows, and the classifier receives features unlike any "
        "it saw in training. (Observed on the six demo pairs; not measured over the whole test set.)")
else:
    st.warning("Run `python scripts/probes.py` to generate the probe results.")

C.shows(
    "That the variants the paper describes are exactly the architectures we build (parameter counts match to the "
    "unit for all fusion and classifier variants), the paper's reported effect sizes, and how much the trained "
    "model depends on its skip masks.",
    "That the ablation *accuracies* are reproducible. We did not retrain any variant, and several reported "
    "differences are smaller than the measurement uncertainty.",
)
C.footer()
