"""Shared helpers for the Streamlit pages: cached models and results, the pair picker, evidence labels."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdlib import paper  # noqa: E402
from cdlib.data import load_pair, read_list  # noqa: E402
from cdlib.paths import DATA_DIR, DEMO_DIR, RESULTS_DIR  # noqa: E402

MODEL_LABELS = {"tinycd": "TinyCD", "bit": "BIT", "changeformer": "ChangeFormer V6"}

# ---------------------------------------------------------------------------------------------------------------
# Evidence labels. Every number on every page is one of these three kinds.
# ---------------------------------------------------------------------------------------------------------------
LABEL_HELP = {
    "Measured": "We ran it ourselves in this project (script and environment recorded in results/).",
    "Reported": "Copied from a publication; the table number is given. Not verified by us unless stated.",
    "Derived": "Computed from a formula or from other numbers on the page; the formula is shown.",
}
_COLOURS = {"Measured": "green", "Reported": "orange", "Derived": "blue", "Qualitative": "violet",
            "Probe": "red", "Demo": "gray"}


def badge(kind: str, extra: str = "") -> str:
    """Markdown badge, e.g. badge('Reported', 'Table 1')."""
    text = kind + (f" · {extra}" if extra else "")
    return f":{_COLOURS.get(kind, 'gray')}-badge[{text}]"


def page_header(title: str, paper_says: str, cite: str = ""):
    st.title(title)
    with st.container(border=True):
        st.markdown(f"**What the paper says** {badge('Reported', cite) if cite else ''}\n\n{paper_says}")


def shows(does: str, does_not: str):
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What this shows**")
        st.markdown(does)
    with c2:
        st.markdown("**What this does *not* show**")
        st.markdown(does_not)


def footer():
    st.caption(f"Paper: {paper.CITATION}. Code and weights of each model: see *Limitations & honesty*.")


# ---------------------------------------------------------------------------------------------------------------
# Device and models
# ---------------------------------------------------------------------------------------------------------------
@st.cache_resource
def gpu_name() -> str | None:
    """Name of a *usable* CUDA GPU, or None. Checks with a real allocation, not only is_available()."""
    import torch
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.zeros(1, device="cuda")
            return torch.cuda.get_device_name(0)
    except Exception:  # driver problems, hidden devices, ...: fall back to CPU
        pass
    return None


def device() -> str:
    if st.session_state.get("force_cpu") or gpu_name() is None:
        return "cpu"
    return "cuda"


def sidebar_settings():
    with st.sidebar:
        st.markdown("**Compute**")
        if gpu_name():
            st.toggle("Force CPU", key="force_cpu",
                      help="Run models on the CPU. Use this if the GPU misbehaves during a demo.")
            st.caption(f"Device: `{device()}` ({gpu_name()})")
        else:
            st.caption("Device: `cpu` (no usable CUDA GPU found)")


@st.cache_resource(show_spinner="Loading model weights ...")
def _load(key: str, dev: str):
    import torch
    from cdlib.models.registry import load_model
    m = load_model(key, dev, check_hash=True)
    x = torch.rand(1, 3, 256, 256, device=dev)
    m.predict(x, x)  # warm-up, so the first timed run on a page does not include CUDA initialisation
    return m


def get_model(key: str):
    """Load (once) and return a model; falls back to CPU if CUDA loading fails."""
    dev = device()
    try:
        return _load(key, dev)
    except RuntimeError as e:  # e.g. CUDA out of memory
        if dev == "cuda":
            st.warning(f"GPU failed for {MODEL_LABELS[key]} ({e.__class__.__name__}); using CPU instead.")
            return _load(key, "cpu")
        raise


# ---------------------------------------------------------------------------------------------------------------
# Results files
# ---------------------------------------------------------------------------------------------------------------
@st.cache_data
def load_json(name: str):
    p = RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_npz(name: str):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


def require(obj, script: str):
    if obj is None:
        st.warning(f"Result file not found. Generate it with `python {script}`.")
        st.stop()
    return obj


def generated_note(res: dict):
    env = res.get("environment", {})
    if env:
        st.caption(f"Generated {env.get('generated', '?')} on {env.get('gpu') or env.get('cpu') or env.get('device')}"
                   f" · PyTorch {env.get('torch')} · Python {env.get('python')}")


# ---------------------------------------------------------------------------------------------------------------
# Image pairs
# ---------------------------------------------------------------------------------------------------------------
@st.cache_data
def demo_index() -> list[dict]:
    p = DEMO_DIR / "index.json"
    return json.loads(p.read_text()) if p.exists() else []


def dataset_available() -> bool:
    return (DATA_DIR / "list" / "test.txt").exists()


@st.cache_data
def test_names() -> list[str]:
    return read_list("test") if dataset_available() else []


@st.cache_data
def demo_pair(dirname: str):
    d = DEMO_DIR / dirname
    a = np.array(Image.open(d / "A.png").convert("RGB"))
    b = np.array(Image.open(d / "B.png").convert("RGB"))
    m = np.array(Image.open(d / "label.png")) > 0
    return a, b, (m.any(2) if m.ndim == 3 else m)


@st.cache_data
def dataset_pair(name: str):
    return load_pair(name)


def _read_upload(f) -> np.ndarray:
    img = Image.open(io.BytesIO(f.getvalue()))
    if img.mode not in ("RGB", "RGBA", "L", "P"):
        raise ValueError("Use an 8-bit display-ready RGB image. Prepare raw USGS bands with scripts/prepare_usgs.py.")
    if max(img.size) > 2048:
        raise ValueError("Please upload images no larger than 2048 px on each side.")
    return np.array(img.convert("RGB"))


def pair_picker(key: str, allow_upload: bool = False, default_dir: str | None = None):
    """Sidebar-free picker. Returns dict(a, b, mask or None, name, source, demo_dir or None)."""
    idx = demo_index()
    sources = ["Demo pairs (fixed set)"]
    if dataset_available():
        sources.append("Any LEVIR-CD test pair")
    if allow_upload:
        sources.append("Upload your own pair")
    src = st.radio("Image pair source", sources, horizontal=True, key=f"{key}_src")

    if src.startswith("Demo"):
        if not idx:
            st.error("No demo pairs found. Run `python scripts/select_demo.py`.")
            st.stop()
        titles = [f"{e['title']}  ({e['name']})" for e in idx]
        start = next((i for i, e in enumerate(idx) if e["dir"] == default_dir), 0)
        i = st.selectbox("Demo pair", range(len(idx)), format_func=lambda j: titles[j], index=start, key=f"{key}_demo")
        e = idx[i]
        a, b, m = demo_pair(e["dir"])
        st.caption(f"Selection rule: {e['rule']}")
        return dict(a=a, b=b, mask=m, name=e["name"], source="demo", demo_dir=e["dir"])

    if src.startswith("Any"):
        names = test_names()
        name = st.selectbox("Test pair", names, key=f"{key}_name",
                            index=names.index(st.session_state.get(f"{key}_goto", names[0]))
                            if st.session_state.get(f"{key}_goto") in names else 0)
        a, b, m = dataset_pair(name)
        return dict(a=a, b=b, mask=m, name=name, source="dataset", demo_dir=None)

    st.info("Upload two co-registered RGB images of the same area and size (t1 = before, t2 = after). "
            "Images larger than 256x256 are tiled at native resolution. The models were trained on 0.5 m/px "
            "aerial imagery of buildings in Texas; results on other imagery are outside what they were trained for.")
    c1, c2, c3 = st.columns(3)
    fa = c1.file_uploader("t1 image (before)", type=["png", "jpg", "jpeg", "tif", "tiff"], key=f"{key}_ua")
    fb = c2.file_uploader("t2 image (after)", type=["png", "jpg", "jpeg", "tif", "tiff"], key=f"{key}_ub")
    fm = c3.file_uploader("Ground-truth mask (optional)", type=["png", "jpg", "tif", "tiff"], key=f"{key}_um")
    if not (fa and fb):
        st.stop()
    try:
        a, b = _read_upload(fa), _read_upload(fb)
    except (OSError, ValueError) as e:
        st.error(f"Cannot read the image pair: {e}")
        st.stop()
    if a.shape != b.shape:
        st.error(f"The two images must have the same size (got {a.shape[1]}x{a.shape[0]} and {b.shape[1]}x{b.shape[0]}).")
        st.stop()
    if max(a.shape[:2]) > 2048:
        st.error("Please upload images no larger than 2048 px on each side.")
        st.stop()
    m = None
    if fm:
        try:
            with Image.open(io.BytesIO(fm.getvalue())) as mask_image:
                if mask_image.size != (a.shape[1], a.shape[0]):
                    raise ValueError("The mask must have the same size as the images.")
                mm = np.array(mask_image)
        except (OSError, ValueError) as e:
            st.error(f"Cannot read the reference mask: {e}")
            st.stop()
        m = (mm.any(2) if mm.ndim == 3 else mm > 0)
        if m.shape != a.shape[:2]:
            st.error("The mask must have the same size as the images.")
            st.stop()
    return dict(a=a, b=b, mask=m, name=f"{fa.name} / {fb.name}", source="upload", demo_dir=None)


def show_pair(p: dict, outline_gt: bool = True):
    from cdlib.viz import outline
    cols = st.columns(3)
    cols[0].image(p["a"], caption="t1 (before)", width="stretch")
    cols[1].image(p["b"], caption="t2 (after)", width="stretch")
    if p["mask"] is not None:
        img = outline(p["b"], p["mask"]) if outline_gt else (p["mask"] * 255).astype(np.uint8)
        cols[2].image(img, caption="Ground truth (yellow outline on t2)" if outline_gt else "Ground truth",
                      width="stretch")
    else:
        cols[2].info("No ground truth for this pair.")


@st.cache_data(show_spinner="Running TinyCD ...", max_entries=16)
def tinycd_trace(a: np.ndarray, b: np.ndarray, dev: str) -> dict:
    """Every intermediate tensor of TinyCD for one 256x256 pair (batch dimension removed, float32 NumPy)."""
    import torch
    from cdlib.data import to_tensor
    model = get_model("tinycd")
    net = model.net
    with torch.no_grad():
        x1 = model.normalise(to_tensor(a)[None].to(model.device))
        x2 = model.normalise(to_tensor(b)[None].to(model.device))
        t = {"M256": net._first_mix(x1, x2)}
        f1, f2 = [], []
        for layer in net._backbone:
            x1, x2 = layer(x1), layer(x2)
            f1.append(x1)
            f2.append(x2)
        t["mix128"] = net._mixing_mask[0]._mixing(f1[1], f2[1])
        t["M128"] = net._mixing_mask[0](f1[1], f2[1])
        t["mix64"] = net._mixing_mask[1]._mixing(f1[2], f2[2])
        t["M64"] = net._mixing_mask[1](f1[2], f2[2])
        t["U32"] = net._mixing_mask[2](f1[3], f2[3])
        t["D64"] = net._up[0](t["U32"], t["M64"])
        t["D128"] = net._up[1](t["D64"], t["M128"])
        t["D256"] = net._up[2](t["D128"], t["M256"])
        t["P"] = net._classify(t["D256"])
        for i in range(4):
            t[f"F1_{i}"], t[f"F2_{i}"] = f1[i], f2[i]
    return {k: v[0].float().cpu().numpy() for k, v in t.items()}


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
