"""Exercise frozen-checkpoint regression and interactive inference controls."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdlib.data import to_tensor
from cdlib.models.registry import WEIGHTS, load_model


@pytest.mark.parametrize("key", list(WEIGHTS))
def test_frozen_model_matches_preserved_demo_prediction(key):
    if not WEIGHTS[key]["path"].exists():
        pytest.skip("checkpoint not downloaded")
    demo = json.loads((ROOT / "assets/demo/index.json").read_text())[0]
    folder = ROOT / "assets/demo" / demo["dir"]
    a, b = [np.array(Image.open(folder / name).convert("RGB")) for name in ("A.png", "B.png")]
    saved = np.array(Image.open(folder / f"prob_{key}.png")).astype(float) / 255
    m = load_model(key, "cpu", check_hash=True)
    p = m.predict(to_tensor(a)[None], to_tensor(b)[None])[0, 0].numpy()
    # Saved previews came from GPU and were rounded to 8 bits. CPU/GPU kernel
    # differences can exceed half a quantization bin at individual pixels.
    # Bound the worst difference by one bin and mean error by half a bin.
    error = np.abs(p - saved)
    assert error.max() < 1 / 255
    assert error.mean() < 0.5 / 255


def app_page(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=600).run()
    at.switch_page(f"views/{page}").run()
    assert not at.exception
    return at


def test_live_threshold_exploration_does_not_change_benchmark():
    original = (ROOT / "results/eval_tinycd_test.json").read_bytes()
    at = app_page("p05_inference.py")
    at.slider(key="p05_threshold").set_value(0.75).run()
    assert not at.exception
    assert any("exploration only" in x.value for x in at.warning)
    assert (ROOT / "results/eval_tinycd_test.json").read_bytes() == original
    assert len(at.get("download_button")) >= 3


def test_powai_missing_data_state_does_not_offer_fake_inference(tmp_path, monkeypatch):
    import cdlib.paths
    monkeypatch.setattr(cdlib.paths, "ROOT", tmp_path)
    at = app_page("p10_powai.py")
    assert any("Imagery not prepared" in x.value for x in at.info)
    assert not at.button


def test_fresh_clone_uses_bundled_powai_predictions(tmp_path, monkeypatch):
    import cdlib.paths
    monkeypatch.setattr(cdlib.paths, 'OUTPUTS_DIR', tmp_path / 'empty_outputs')
    at = app_page('p10_powai.py')
    at.get('pills')[0].set_value(['tinycd', 'bit', 'changeformer']).run()
    at.get('segmented_control')[0].set_value('Saved predictions').run()
    assert not at.exception and not at.error
    table = at.dataframe[-1].value
    assert len(table) == 3
    assert set(table['source']) == {'saved inference'}
    assert (table['predicted_changed_pixels'] == 0).all()
    assert not any('Saved predictions unavailable' in x.value for x in at.info)


def test_powai_prepared_fixture_live_export_and_saved_failure(tmp_path, monkeypatch):
    # A synthetic fixture lives ONLY in pytest's temp folder and is explicitly labelled.
    import cdlib.paths
    from cdlib.usgs import sha256
    folder = tmp_path / "assets/powai/pair"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(12)
    a = rng.integers(0, 255, (256, 256, 3), np.uint8)
    for name in ("A.png", "B.png"):
        Image.fromarray(a).save(folder / name)
    Image.fromarray(np.full((256, 256), 255, np.uint8)).save(folder / "valid.png")
    manifest = dict(status="prepared", ground_truth=None, size=[256,256], common_valid_percent=100,
                    scenes=[dict(acquired="2015-02-10"), dict(acquired="2025-02-05")],
                    note="Synthetic software test fixture, not satellite imagery",
                    files={n: sha256(folder / n) for n in ("A.png", "B.png", "valid.png")})
    (folder / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cdlib.paths, "ROOT", tmp_path)
    monkeypatch.setattr(cdlib.paths, "OUTPUTS_DIR", tmp_path / "outputs")
    at = app_page("p10_powai.py")
    assert at.button(key="powai_run").disabled
    at.checkbox(key="powai_reviewed").check().run()
    at.button(key="powai_run").click().run()
    assert not at.exception and not at.error
    assert len(at.get("download_button")) >= 2
    assert "predicted_changed_percent" in at.dataframe[-1].value.columns
    assert "f1" not in at.dataframe[-1].value.columns
    at.slider(key="powai_threshold").set_value(0.7).run()
    assert not at.exception
    # Saved mode must accept a matching report and reject a tampered probability file.
    out = tmp_path / "outputs/powai"
    out.mkdir(parents=True)
    prob, meta = at.session_state.powai_predictions["tinycd"]
    probability_path = out / "prob_tinycd.npz"
    np.savez_compressed(probability_path, probability=prob)
    report = dict(pair_manifest_sha256=sha256(folder / "manifest.json"),
                  models={"tinycd": dict(meta, probability_sha256=sha256(probability_path))})
    (out / "report.json").write_text(json.dumps(report))
    at.get("segmented_control")[0].set_value("Saved predictions").run()
    assert not at.exception
    assert at.dataframe[-1].value.iloc[0]["source"] == "saved inference"
    probability_path.write_bytes(b"tampered test data")
    at.run()
    assert not at.exception
    assert any("Saved predictions unavailable" in x.value for x in at.info)
