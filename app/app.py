"""TinyCD replication & explanation app.   Run:  streamlit run app/app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

st.set_page_config(page_title="TinyCD, replicated", page_icon=":material/compare:", layout="wide")

V = Path(__file__).resolve().parent / "views"
pages = {
    "Overview": [
        st.Page(V / "home.py", title="Home", icon=":material/home:", default=True),
    ],
    "Understand the model": [
        st.Page(V / "p01_problem.py", title="1 · The problem", icon=":material/satellite_alt:"),
        st.Page(V / "p02_architecture.py", title="2 · Architecture walk-through", icon=":material/account_tree:"),
        st.Page(V / "p03_mix.py", title="3 · The Mix block", icon=":material/merge:"),
        st.Page(V / "p04_masks.py", title="4 · Attention masks", icon=":material/layers:"),
    ],
    "Replicate the findings": [
        st.Page(V / "p05_inference.py", title="5 · Live inference", icon=":material/play_circle:"),
        st.Page(V / "p06_results.py", title="6 · Replication results", icon=":material/fact_check:"),
        st.Page(V / "p07_efficiency.py", title="7 · Size and speed", icon=":material/speed:"),
        st.Page(V / "p08_ablations.py", title="8 · Variants & probes", icon=":material/science:"),
        st.Page(V / "p09_failures.py", title="9 · Where it fails", icon=":material/report:"),
    ],
    "Local demonstration and scope": [
        st.Page(V / "p10_powai.py", title="10 · IIT Bombay / Powai", icon=":material/location_on:"),
        st.Page(V / "p11_limitations.py", title="11 · Limitations & honesty", icon=":material/policy:"),
        st.Page(V / "p12_guide.py", title="12 · Demo guide", icon=":material/menu_book:"),
    ],
}
nav = st.navigation(pages)
common.sidebar_settings()
with st.sidebar:
    st.markdown("**Evidence labels**")
    for k, v in common.LABEL_HELP.items():
        st.markdown(f"{common.badge(k)} {v}")
nav.run()
