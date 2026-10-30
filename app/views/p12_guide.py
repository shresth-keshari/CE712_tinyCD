from pathlib import Path

import streamlit as st

st.title("12 · Demo guide")
guide = Path(__file__).resolve().parents[2] / "demo_guide.md"
if guide.exists():
    text = guide.read_text(encoding="utf-8")
    st.download_button("Download the complete demo guide", text, file_name="demo_guide.md", mime="text/markdown")
    st.markdown(text)
else:
    st.error("demo_guide.md is missing from the project root.")
