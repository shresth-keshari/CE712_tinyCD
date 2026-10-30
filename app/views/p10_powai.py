"""Real USGS imagery is user supplied; never substitute example imagery for Powai."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C
from cdlib.exports import change_overlay, evidence_zip, prediction_summary
from cdlib.inference import predict_pair
from cdlib.paths import ROOT, OUTPUTS_DIR
from cdlib.usgs import DISCLAIMER, load_prepared, sha256
from cdlib.viz import colormap

st.title("10 · IIT Bombay / Powai: USGS demonstration")
st.markdown("Apply the released checkpoints to two real satellite acquisitions of the campus and surrounding Powai area.")
st.warning(DISCLAIMER)
selection_path = ROOT / "assets" / "powai" / "selection.json"
if selection_path.exists():
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    st.subheader("Selected USGS acquisitions")
    st.dataframe(pd.DataFrame([
        {"date role": role, "USGS product": s["product_id"], "acquired": s["acquired"][:10],
         "scene cloud %": s["scene_cloud_percent"], "pixel size (m)": s["pixel_size_m"]}
        for role, s in selection["selected"].items()]), hide_index=True)
    st.caption("Catalog selection is provisional until local quality bands and alignment are checked. "
               "This table describes the suggested downloads; the prepared bundle below records the actual inputs.")
    with st.expander("USGS source records and download links"):
        st.link_button("Open USGS EarthExplorer", "https://earthexplorer.usgs.gov/")
        for role, scene in selection["selected"].items():
            st.markdown(f"**{role.capitalize()}** — [USGS catalog record]({scene['stac_url']})")
            for band, url in scene["assets"].items():
                st.markdown(f"[{band}]({url})")
        st.download_button("Download scene selection JSON", selection_path.read_bytes(),
                           file_name="powai_scene_selection.json", mime="application/json")

with st.expander("Preparation steps and commands"):
    st.markdown("Sign in to EarthExplorer in your own browser. Download and extract both Level-2 scenes, "
                "including SR_B2, SR_B3, SR_B4, QA_PIXEL, QA_RADSAT and MTL.json. "
                "Run the commands from the project folder after replacing the two local paths. "
                "The full procedure is in USGS_WORKFLOW.md and page 12.")
    st.caption("This clone already includes the prepared demonstration. Use a new output folder for another pair.")
    st.code('./python scripts/prepare_usgs.py --before data/usgs/before --after data/usgs/after '
            '--out assets/powai/pair_v2\n'
            './python scripts/infer_usgs.py --pair assets/powai/pair_v2 --out outputs/new/powai_v2', language="bash")

pair_dir = ROOT / "assets" / "powai" / "pair"
if not (pair_dir / "manifest.json").exists():
    st.info("Imagery not prepared. The USGS scenes have been identified, but local image files are still required. "
            "No Powai predictions or accuracy results have been generated.")
    C.footer()
    st.stop()
try:
    a, b, valid, manifest = load_prepared(pair_dir)
except (OSError, ValueError, KeyError) as e:
    st.error(f"Prepared imagery could not be verified: {e}")
    st.stop()

st.subheader("Prepared inputs and quality")
st.caption(f"Actual dates: {manifest['scenes'][0]['acquired']} → {manifest['scenes'][1]['acquired']}. "
           f"{manifest['size'][1]} × {manifest['size'][0]} pixels at 30 m; "
           f"{manifest['common_valid_percent']:.2f}% common valid pixels. North is up.")
cols = st.columns(3)
cols[0].image(a, caption="Earlier acquisition", width="stretch")
cols[1].image(b, caption="Later acquisition", width="stretch")
cols[2].image(valid.astype(np.uint8) * 255, caption="Quality mask: white = valid on both dates", width="stretch")
fade = st.slider("Blend dates to inspect alignment", 0.0, 1.0, 0.5, 0.05, key="powai_fade")
st.image(np.rint((1 - fade) * a + fade * b).astype(np.uint8),
         caption="Inspect lake shores, major roads and fixed landmarks for double edges", width="content")
with st.expander("Actual input provenance and processing"):
    st.json(manifest)
    st.download_button("Download preparation manifest", (pair_dir / "manifest.json").read_bytes(),
                       file_name="powai_manifest.json", mime="application/json")
st.caption("Common-grid reprojection is geometric preparation, not a guarantee of subpixel registration. "
           "The 256-pixel default covers 7.68 km, including surrounding neighbourhoods. "
           "Invalid pixels are filled black on both dates; their neighbours can still influence model responses.")

models = st.pills("Models", list(C.MODEL_LABELS), default=["tinycd"], selection_mode="multi",
                  format_func=lambda k: C.MODEL_LABELS[k], key="powai_models")
threshold = st.slider("Exploratory decision threshold", 0.05, 0.95, 0.5, 0.05, key="powai_threshold")
mode = st.segmented_control("Prediction source", ["Live inference", "Saved predictions"],
                            default="Live inference", key="powai_mode") or "Live inference"
identity = sha256(pair_dir / "manifest.json")
signature = (identity, C.device())
if st.session_state.get("powai_signature") != signature:
    st.session_state.powai_signature = signature
    st.session_state.powai_predictions = {}
predictions = {}
if mode == "Live inference":
    reviewed = st.checkbox("I inspected the two dates and their quality mask", key="powai_reviewed")
    if st.button("Run selected models", disabled=not (reviewed and models), key="powai_run"):
        for key in models:
            try:
                with st.spinner(f"Running {C.MODEL_LABELS[key]} ..."):
                    model = C.get_model(key)
                    prob, secs = predict_pair(model, a, b)
                    st.session_state.powai_predictions[key] = (prob, dict(
                        model=key, checkpoint=model.meta, device=str(model.device), seconds=secs,
                        source="live inference", pair=manifest, disclaimer=DISCLAIMER))
            except Exception as e:
                st.session_state.powai_predictions.pop(key, None)
                st.error(f"{C.MODEL_LABELS[key]} inference failed: {e}")
    predictions = st.session_state.powai_predictions
    st.caption("Results remain available while adjusting the threshold; click Run selected models to compute again.")
else:
    out_dir = OUTPUTS_DIR / "powai"
    if not (out_dir / "report.json").exists():
        out_dir = ROOT / "assets" / "powai" / "predictions"
    try:
        report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        if report["pair_manifest_sha256"] != identity:
            raise ValueError("Saved predictions belong to a different prepared pair")
        for key in models:
            if key not in report["models"]:
                st.info(f"No saved output for {C.MODEL_LABELS[key]}; run scripts/infer_usgs.py for that model.")
                continue
            path = out_dir / f"prob_{key}.npz"
            meta = report["models"][key]
            if sha256(path) != meta["probability_sha256"]:
                raise ValueError(f"Saved prediction checksum mismatch for {key}")
            with np.load(path, allow_pickle=False) as d:
                predictions[key] = (d["probability"], dict(meta, source="saved inference"))
    except (OSError, ValueError, KeyError) as e:
        predictions = {}
        st.info(f"Saved predictions unavailable: {e}. Generate them with scripts/infer_usgs.py.")

rows = []
for key in models:
    if key not in predictions:
        continue
    prob, meta = predictions[key]
    summary = prediction_summary(prob, threshold, valid)
    pred = (prob > threshold) & valid
    st.subheader(C.MODEL_LABELS[key])
    heat = colormap(prob, "magma", 0, 1)
    heat[~valid] = 100
    cols = st.columns(3)
    cols[0].image(heat, caption="Model probability; grey = excluded", width="stretch")
    cols[1].image(pred.astype(np.uint8) * 255, caption="Predicted response mask", width="stretch")
    cols[2].image(change_overlay(b, pred, valid), caption="Pink = predicted response; grey = excluded", width="stretch")
    rows.append({"model": C.MODEL_LABELS[key], "source": meta["source"], "device": meta["device"],
                 "time of stored run (ms)": round(meta["seconds"] * 1000, 1), **summary})
    st.download_button(f"Download {C.MODEL_LABELS[key]} evidence ZIP",
                       evidence_zip(a, b, prob, threshold, meta, valid),
                       file_name=f"powai_{key}_evidence.zip", mime="application/zip", key=f"powai_export_{key}")
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.info("These percentages describe model responses among valid pixels. No F1, IoU, precision or recall "
            "is computed because the Powai pair has no ground-truth change mask.")
C.footer()
