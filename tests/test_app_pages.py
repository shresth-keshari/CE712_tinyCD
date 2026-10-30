"""Smoke test: every app page runs without an exception (uses the demo pairs and precomputed results).

    python -m pytest -q tests/test_app_pages.py
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEWS = sorted(p.name for p in (ROOT / "app" / "views").glob("*.py"))


@pytest.mark.parametrize("view", VIEWS)
def test_page_runs(view):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=600)
    at.run()
    at.switch_page(f"views/{view}")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    title = at.title[0].value
    assert title.startswith("TinyCD") if view == "home.py" else title.startswith(view[1:3].lstrip("0") + " ·"), title
