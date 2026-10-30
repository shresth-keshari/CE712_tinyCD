import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib import paper  # noqa: E402
from cdlib.models.registry import WEIGHTS  # noqa: E402

st.title("11 · Limitations & honesty")
eff = C.load_json("efficiency.json")
if eff:
    _m = eff["models"]
    _t = _m["tinycd"]["params_used"]
    ratio_txt = (f"~{_m['bit']['params_used'] / _t:.1f}x (BIT, params used) and "
                 f"~{_m['changeformer']['params_used'] / _t:.0f}x (ChangeFormer) (page 7)")
else:
    ratio_txt = "see page 7"

st.subheader("What was reproduced, and how")
st.dataframe(pd.DataFrame([
    {"paper finding": "TinyCD F1 91.05 on LEVIR-CD (Table 1)", "status": "✅ reproduced",
     "evidence": "authors' weights in our implementation, full test set (page 6)"},
    {"paper finding": "TinyCD has 285,128 params, 1.45 GFLOPs (Tables 3-4)", "status": "✅ params exact; FLOPs approximately match",
     "evidence": "our build, strict weight loading, fvcore (pages 2, 7)"},
    {"paper finding": "ChangeFormer F1 90.40 (Table 1)", "status": "✅ reproduced", "evidence": "official weights (page 6)"},
    {"paper finding": "BIT F1 89.31 (Table 1)", "status": "⚠️ differs: we measure higher",
     "evidence": "official released checkpoint (page 6); cause unknown"},
    {"paper finding": "TinyCD beats BIT and ChangeFormer", "status": "✅ holds under one protocol",
     "evidence": "paired bootstrap CIs exclude 0 (page 6)"},
    {"paper finding": "13-140x fewer parameters", "status": "✅ approximately",
     "evidence": ratio_txt},
    {"paper finding": "'faster'", "status": "✅ but smaller than FLOPs suggest",
     "evidence": "batch-1 latency on our GPU/CPU (page 7)"},
    {"paper finding": "Mix generalises subtraction; cheaper than concat+conv", "status": "✅ shown",
     "evidence": "numerical proof + parameter counts (page 3)"},
    {"paper finding": "Ablation variant sizes (Tables 4, 6, 7)", "status": "✅ exact match",
     "evidence": "built with our code (page 8)"},
    {"paper finding": "Ablation accuracies (Tables 4-9)", "status": "❌ not tested",
     "evidence": "needs retraining each variant; no weights released"},
    {"paper finding": "Attention masks' meaning (Fig. 4)", "status": "👁 qualitative only", "evidence": "page 4"},
    {"paper finding": "WHU-CD results (Table 2)", "status": "❌ not tested", "evidence": "out of scope"},
    {"paper finding": "Co-registration limitation (Sec. 6)", "status": "🧪 tested by us (not in paper)",
     "evidence": "shift stress test (page 9)"},
    {"paper finding": "Inference on IIT Bombay / Powai imagery", "status": "Exploratory application, not a paper finding",
     "evidence": "USGS scene provenance and preparation status on page 10; no reference labels or accuracy claim"},
]), hide_index=True)

st.subheader("Deviations and implementation notes")
st.markdown("""
- **Inference-only scope.** Released checkpoints are frozen. Training, fine-tuning and optimizer workflows
  have been removed. Dataset split names describe the source dataset, not work performed here.
- **Paper text vs authors' code.** The paper writes the Mix and Up
  blocks as PReLU(IN(Conv(·))) (Eq. 2, Sec. 3.4), but the code applies Conv → PReLU → InstanceNorm. We follow the
  code everywhere, because it produced the released weights (and they load into our implementation only that way).
- **Mask values.** A few LEVIR-CD256 masks contain values 156/254 besides 0/255 (69 pixels in the test set). Like the
  official TinyCD loader we treat every non-zero value as change. BIT's and ChangeFormer's own loaders would map
  these few pixels to "no change". The effect on any metric is below 0.01 points.
- **One metric implementation for all models.** For BIT and ChangeFormer we threshold softmax P(change) at 0.5.
  For two classes this is the same decision as the argmax their evaluators use.
- **Vendored code.** BIT and ChangeFormer run on their official model code with import-only patches, plus one
  initialisation patch (no ImageNet download for BIT's ResNet, whose weights the checkpoint overwrites). See
  `cdlib/models/vendor/README.md`.
- **Local demonstration.** Landsat RGB pixels are 30 m, versus LEVIR-CD's 0.5 m. Common-grid resampling
  does not recover building detail or prove perfect co-registration. Clouds, haze, season, shadows and
  sensor differences can produce responses. Page 10 is qualitative; its changed-pixel fraction is not accuracy.
- **Uncertainty.** Confidence intervals reflect test-image sampling only. There is one released checkpoint per
  model, so training randomness is not captured.
- **Hardware.** All timings come from one laptop (RTX 3050 6 GB laptop GPU), eager FP32 PyTorch, batch 1.
""")

st.subheader("Data and weights: sources, checksums, licences")
st.dataframe(pd.DataFrame([{"model": C.MODEL_LABELS[k], "source": w["source"], "SHA-256": w["sha256"],
                            "licence": {"tinycd": "no licence file in the repository; academic evaluation only",
                                        "bit": "no licence file in the repository; academic evaluation only",
                                        "changeformer": "MIT"}[k]}
                           for k, w in WEIGHTS.items()]), hide_index=True)
st.markdown("**Dataset:** LEVIR-CD (Chen & Shi, *Remote Sensing* 12(10):1662, 2020), 256x256 crop release "
            "distributed via the ChangeFormer repository. Academic use. Neither the weights nor the dataset are "
            "distributed as full datasets by this project; six demonstration crops are included in assets/demo/. "
            "`scripts/download.py` retrieves the full dataset and checkpoints from their original sources. "
            "USGS scene records and their download workflow are on page 10 and in USGS_WORKFLOW.md.")

st.subheader("Course compliance (CE712)")
st.markdown("""
- Everything is done in a programming environment (Python): data loading, models, metrics, statistics,
  visualisation and this app. **No ERDAS, ENVI, QGIS, ArcGIS or other image-processing software is used.**
- Libraries used: PyTorch / torchvision, NumPy, SciPy (connected components), Pillow (reading PNG files),
  Matplotlib (colour maps), pandas, Altair, Streamlit, fvcore (FLOP counting), timm and einops (dependencies of
  ChangeFormer / BIT code).
- The paper is a journal article (Neural Computing and Applications, Springer). Whether it qualifies as "Q1"
  under the course rules must be checked against SJR/JCR by the group.
""")

st.subheader("Paper")
st.markdown(f"{paper.CITATION}  \n[{paper.DOI_URL}]({paper.DOI_URL}) · authors' code: [{paper.CODE_URL}]({paper.CODE_URL})")
