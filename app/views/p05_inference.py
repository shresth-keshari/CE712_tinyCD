import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from cdlib.inference import predict_pair  # noqa: E402
from cdlib.exports import evidence_zip  # noqa: E402
from cdlib.models.registry import WEIGHTS  # noqa: E402
from cdlib.metrics import confusion, scores  # noqa: E402
from cdlib.paths import DEMO_DIR  # noqa: E402
from cdlib.viz import colormap, error_map  # noqa: E402

C.page_header(
    "5 · Live inference: TinyCD vs BIT vs ChangeFormer",
    "TinyCD (0.28 M parameters) produces change masks of comparable quality to much larger models. The paper "
    "conjectures that their differences are 'more related to missing or hallucinated change regions, than region "
    "quality issues' (Sec. 4.4, Fig. 3).",
    "Sec. 4.4",
)

p = C.pair_picker("p05", allow_upload=True)
C.show_pair(p)

c1, c2 = st.columns([2, 1])
models = c1.pills("Models", list(C.MODEL_LABELS), default=list(C.MODEL_LABELS), selection_mode="multi",
                  format_func=lambda k: C.MODEL_LABELS[k], key="p05_models")
thr = c2.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05,
                help="The paper and all reported results use 0.5. Changing it here is exploration only.", key="p05_threshold")
if thr != 0.5:
    st.warning(f"Threshold {thr:.2f} is exploration only. All results elsewhere in this app use 0.5, as the paper does.")


@st.cache_data(show_spinner=False, max_entries=48)
def run(key: str, a: np.ndarray, b: np.ndarray, dev: str):
    model = C.get_model(key)
    prob, secs = predict_pair(model, a, b)
    return prob, secs, str(model.device)


if not models:
    st.stop()
cols = st.columns(len(models))
rows = []
for col, key in zip(cols, models):
    with col:
        st.markdown(f"#### {C.MODEL_LABELS[key]}")
        source = "inference (cached on repeat)"
        try:
            with st.spinner(f"Running {C.MODEL_LABELS[key]} on {C.device()} ..."):
                prob, secs, actual_device = run(key, p["a"], p["b"], C.device())
        except Exception as e:  # keep the demo alive: fall back to the pre-rendered output for demo pairs
            if p["demo_dir"] is None:
                st.error(f"Inference failed: {e}")
                continue
            prob = np.array(Image.open(DEMO_DIR / p["demo_dir"] / f"prob_{key}.png")).astype(np.float32) / 255
            secs, source = None, "pre-rendered"
            actual_device = "not run; saved 8-bit probability"
            st.warning(f"Live inference failed ({e.__class__.__name__}); showing the pre-rendered output.")
        pred = prob > thr
        st.image(colormap(prob, "magma", 0, 1), caption="P(change)  (black 0 -> yellow 1)", width="stretch")
        st.image((pred * 255).astype(np.uint8), caption=f"Binary mask (P > {thr:.2f})", width="stretch")
        row = {"model": C.MODEL_LABELS[key], "source": source, "device": actual_device,
               "time (ms)": None if secs is None else round(1000 * secs, 1),
               "changed px %": round(100 * pred.mean(), 2)}
        if p["mask"] is not None:
            st.image(error_map(pred, p["mask"]), caption="vs ground truth", width="stretch")
            s = scores(confusion(pred, p["mask"]))
            if not p["mask"].any():
                # No true positives are possible. Recall is undefined; false-positive F1/IoU are zero.
                s["recall"] = None
                if not pred.any():
                    s.update(precision=None, f1=None, iou=None)
            row.update({k: None if v is None else round(v, 2) for k, v in s.items()})
        rows.append(row)
        meta = dict(model=key, pair=p["name"], input_source=p["source"], prediction_source=source,
                    device=actual_device, seconds=secs, checkpoint_sha256=WEIGHTS[key]["sha256"],
                    per_pair_metrics=row, probability_precision="8-bit fallback" if secs is None else "float32",
                    note="Single-pair inference; benchmark results remain fixed at threshold 0.5.")
        st.download_button(f"Download {C.MODEL_LABELS[key]} evidence ZIP",
                           evidence_zip(p["a"], p["b"], prob, thr, meta, ground_truth=p["mask"]),
                           file_name=f"{key}_inference_evidence.zip", mime="application/zip", key=f"p05_export_{key}")

st.markdown("**Error map colours:** white = correct change (TP), red = false alarm (FP), blue = missed change "
            "(FN), black = correct no-change (TN).")
st.markdown(f"**Per-pair numbers** {C.badge('Measured', 'this pair only')}")
st.dataframe(pd.DataFrame(rows), hide_index=True)
if p["mask"] is not None and not p["mask"].any():
    st.info("This pair has no ground-truth change. Recall is undefined. If the prediction is also empty, precision, "
            "F1 and IoU are blank; otherwise they are zero. Every predicted change pixel is a false alarm.")
st.caption("Time = forward time of the stored inference result on the device shown in the table, including tiling. "
           "Loading and warm-up are excluded; cached reruns reuse that original time. Page 7 has repeated timings. "
           "Metrics for a single pair vary a lot; the test-set results are on page 6.")

C.shows(
    "All three released models running live on the same input, with their errors made visible.",
    "Which model is better overall. One pair is anecdotal; see page 6 for all 2,048 test pairs with "
    "confidence intervals. Uploaded images from other sensors or resolutions are outside the training "
    "distribution, so results on them say little about the paper's claims.",
)
C.footer()
