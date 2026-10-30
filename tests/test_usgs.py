"""Synthetic GeoTIFF fixtures validate software, not actual Powai imagery or model accuracy."""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.exports import evidence_zip, prediction_summary
from cdlib.usgs import CENTER, load_prepared, prepare_pair, quality_valid, reflectance_rgb, sha256


def write_scene(folder, date, xshift=0, cloudy=False):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from rasterio.warp import transform
    import math
    folder.mkdir(parents=True)
    x, y = transform("EPSG:4326", "EPSG:32643", [CENTER[0]], [CENTER[1]])
    left = math.floor((x[0] - 256 * 15) / 30) * 30 - 32 * 30 + xshift * 30
    top = math.ceil((y[0] + 256 * 15) / 30) * 30 + 32 * 30
    grid = from_origin(left, top, 30, 30)
    pid = f"LC08_L2SP_148047_{date.replace('-', '')}_20200909_02_T1"
    ycoord, xcoord = np.indices((320, 320))
    # A geographic gradient with one known region, invariant to a one-pixel source-grid shift.
    dn = (10000 + (xcoord + xshift) * 10 + ycoord * 5).astype(np.uint16)
    for suffix in ("SR_B4", "SR_B3", "SR_B2", "QA_PIXEL", "QA_RADSAT"):
        values = dn if suffix.startswith("SR_") else np.zeros((320, 320), np.uint16)
        if cloudy and suffix == "QA_PIXEL":
            values[:] = 1 << 3
        with rasterio.open(folder / f"{pid}_{suffix}.TIF", "w", driver="GTiff", count=1,
                           width=320, height=320, dtype="uint16", crs="EPSG:32643", transform=grid) as dst:
            dst.write(values, 1)
    m = {"LANDSAT_METADATA_FILE": {"PRODUCT_CONTENTS": {"LANDSAT_PRODUCT_ID": pid},
         "IMAGE_ATTRIBUTES": {"DATE_ACQUIRED": date, "SPACECRAFT_ID": "LANDSAT_8",
                              "WRS_PATH": 148, "WRS_ROW": 47, "CLOUD_COVER": 0}}}
    (folder / f"{pid}_MTL.json").write_text(json.dumps(m))
    return folder


def test_quality_bits_and_water():
    flags = np.array([0, 64, 128, 1, 2, 4, 8, 16, 32, 0], np.uint16)
    sat = np.zeros_like(flags)
    sat[-1] = 1
    assert quality_valid(flags, sat).tolist() == [True, True, True, False, False, False, False, False, False, False]


def test_usgs_scaling_and_shared_stretch():
    # USGS example: DN 18639 -> reflectance 0.313, clipped at default upper 0.3.
    assert reflectance_rgb(np.array([18639], np.uint16))[0] == 255
    assert reflectance_rgb(np.array([7273], np.uint16))[0] == 0
    expected = round(((12000 * 0.0000275 - 0.2) / 0.3) * 255)
    assert reflectance_rgb(np.array([12000], np.uint16))[0] == expected
    with pytest.raises(ValueError):
        reflectance_rgb(np.ones(1), high=0)


def test_preparation_aligns_native_grids_and_records_provenance(tmp_path):
    a = write_scene(tmp_path / "before", "2015-02-10")
    b = write_scene(tmp_path / "after", "2025-02-05", xshift=1)
    out = tmp_path / "prepared"
    m = prepare_pair(a, b, out)
    first, second, valid, loaded = load_prepared(out)
    assert first.shape == (256, 256, 3)
    assert np.array_equal(first, second), "Common-grid alignment must remove the one-pixel source-grid offset"
    assert valid.all() and m["common_valid_percent"] == 100
    assert loaded["ground_truth"] is None and loaded["pixel_size_m"] == 30
    assert loaded["scenes"][0]["acquired"] == "2015-02-10"
    for name in ("A.png", "B.png", "valid.png"):
        assert loaded["files"][name] == sha256(out / name)
    with pytest.raises(FileExistsError):
        prepare_pair(a, b, out)
    (out / "A.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(out)


def test_reject_cloudy_pair_and_reverse_dates_without_writing(tmp_path):
    a = write_scene(tmp_path / "before", "2015-02-10")
    b = write_scene(tmp_path / "after", "2025-02-05", cloudy=True)
    with pytest.raises(ValueError, match="valid pixels"):
        prepare_pair(a, b, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="earlier"):
        prepare_pair(b, a, tmp_path / "reversed")


def test_evidence_export_excludes_invalid_pixels_and_preserves_probabilities():
    a = np.zeros((2, 2, 3), np.uint8)
    p = np.array([[0.8, 0.9], [0.1, 0.4]], np.float32)
    valid = np.array([[True, False], [True, True]])
    summary = prediction_summary(p, 0.5, valid)
    assert summary["predicted_changed_pixels"] == 1 and summary["valid_pixels"] == 3
    assert summary["predicted_changed_percent"] == pytest.approx(100 / 3)
    data = evidence_zip(a, a, p, 0.5, {"evidence": "synthetic software test, not Powai"}, valid)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert "ground_truth.png" not in z.namelist()
        assert np.array_equal(np.load(io.BytesIO(z.read("probability.npy")), allow_pickle=False), p)
        mask = np.array(Image.open(io.BytesIO(z.read("mask.png"))))
        assert mask.tolist() == [[255, 0], [0, 0]]
        meta = json.loads(z.read("metadata.json"))
        assert meta["prediction_summary"]["excluded_pixels"] == 1
        assert not any(k in meta for k in ("f1", "iou", "precision", "recall"))


def test_summary_rejects_invalid_probability_or_empty_coverage():
    with pytest.raises(ValueError):
        prediction_summary(np.array([[np.nan]]), 0.5)
    with pytest.raises(ValueError):
        prediction_summary(np.array([[0.5]]), 0.5, np.array([[False]]))
    with pytest.raises(ValueError):
        prediction_summary(np.array([[0.5]]), 1.1)
