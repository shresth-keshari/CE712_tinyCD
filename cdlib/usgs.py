"""USGS Landsat discovery and local preparation for exploratory inference.

No credentials, automatic downloads, model fitting or accuracy claims. Rasterio is
an optional dependency used only by preparation/export, not by the Streamlit app.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image

STAC_URL = "https://landsatlook.usgs.gov/stac-server"
COLLECTION = "landsat-c2l2-sr"
CENTER = (72.9133, 19.1334)  # Approximate IIT Bombay campus centre, lon/lat.
SEARCH_BBOX = (72.875, 19.095, 72.95, 19.17)
ASSETS = ("red", "green", "blue", "qa_pixel", "qa_radsat", "MTL.json")
SR_SCALE, SR_OFFSET = 0.0000275, -0.2
DISCLAIMER = (
    "Exploratory inference on 30 m USGS Landsat imagery. The released models were developed for "
    "0.5 m LEVIR-CD building imagery. Predicted responses are not verified building changes. "
    "No ground-truth mask is available; changed-pixel percentage is not accuracy."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def search_scenes(start: str, end: str, max_cloud: float = 5.0) -> list[dict]:
    """Query public USGS metadata; select Landsat 8 Tier 1, WRS 148/047.

    Both endpoints are dates, inclusive. Scene-wide cloud cover is a search filter,
    not a substitute for checking the local QA bands or inspecting the images.
    """
    first, last = datetime.fromisoformat(start), datetime.fromisoformat(end)
    if first > last:
        raise ValueError("Start date must precede end date")
    params = dict(collections=COLLECTION, bbox=",".join(map(str, SEARCH_BBOX)),
                  datetime=f"{start}T00:00:00Z/{end}T23:59:59Z", limit=100)
    url = f"{STAC_URL}/search?{urlencode(params)}"
    features, seen = [], set()
    while url:
        if url in seen or not url.startswith(STAC_URL + "/"):
            raise ValueError("Unexpected STAC pagination URL")
        seen.add(url)
        with urlopen(url, timeout=45) as response:
            data = json.load(response)
        features.extend(data.get("features", []))
        url = next((x["href"] for x in data.get("links", []) if x["rel"] == "next"), None)
    selected = []
    for f in features:
        p = f["properties"]
        if (p.get("platform") == "LANDSAT_8" and p.get("landsat:collection_category") == "T1"
                and int(p.get("landsat:wrs_path", -1)) == 148
                and int(p.get("landsat:wrs_row", -1)) == 47
                and 0 <= p.get("eo:cloud_cover", 101) <= max_cloud):
            selected.append(f)
    return sorted(selected, key=lambda f: (f["properties"]["eo:cloud_cover"], f["id"]))


def scene_summary(feature: dict) -> dict:
    p = feature["properties"]
    return dict(product_id=feature["id"].removesuffix("_SR"), stac_id=feature["id"],
                acquired=p["datetime"], scene_cloud_percent=p["eo:cloud_cover"],
                platform=p["platform"], path=148, row=47, pixel_size_m=30,
                stac_url=f"{STAC_URL}/collections/{COLLECTION}/items/{feature['id']}",
                assets={k: feature["assets"][k]["href"] for k in ASSETS})


def quality_valid(qa_pixel: np.ndarray, qa_radsat: np.ndarray) -> np.ndarray:
    """Reject QA_PIXEL bits 0-5 and any QA_RADSAT flags; water is retained."""
    return ((qa_pixel.astype(np.uint16) & 0b111111) == 0) & (qa_radsat == 0)


def reflectance_rgb(dn: np.ndarray, low: float = 0.0, high: float = 0.3) -> np.ndarray:
    """One fixed linear RGB stretch, identical for both dates; never per-image equalization."""
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        raise ValueError("Reflectance stretch must have finite low < high")
    sr = dn.astype(np.float32) * SR_SCALE + SR_OFFSET
    return np.rint(np.clip((sr - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def local_scene(folder: Path) -> tuple[dict, dict[str, Path]]:
    """Require one extracted Landsat 8/9 C2 L2 scene with unambiguous native filenames."""
    # A single scan avoids counting the same path twice on Windows' case-insensitive filesystem.
    mtls = [p for p in folder.rglob("*") if p.is_file() and p.name.lower().endswith("_mtl.json")]
    if len(mtls) != 1:
        raise ValueError(f"Expected exactly one *_MTL.json under {folder}; found {len(mtls)}")
    metadata = json.loads(mtls[0].read_text(encoding="utf-8"))
    m = metadata["LANDSAT_METADATA_FILE"]
    contents, attrs = m["PRODUCT_CONTENTS"], m["IMAGE_ATTRIBUTES"]
    product_id = contents["LANDSAT_PRODUCT_ID"]
    parts = product_id.split("_")
    if (len(parts) != 7 or parts[0] not in ("LC08", "LC09") or parts[1] not in ("L2SP", "L2SR")
            or parts[-2:] != ["02", "T1"]):
        raise ValueError("Use Landsat 8/9 Collection 2 Level-2 Tier 1 data")
    paths = {"metadata": mtls[0]}
    for key, suffix in (("red", "SR_B4.TIF"), ("green", "SR_B3.TIF"), ("blue", "SR_B2.TIF"),
                        ("qa_pixel", "QA_PIXEL.TIF"), ("qa_radsat", "QA_RADSAT.TIF")):
        matches = [p for p in folder.rglob("*") if p.is_file() and p.name.upper() == f"{product_id}_{suffix}".upper()]
        if len(matches) != 1:
            raise ValueError(f"Expected one {product_id}_{suffix}; found {len(matches)}")
        paths[key] = matches[0]
    info = dict(product_id=product_id, acquired=attrs["DATE_ACQUIRED"],
                platform=attrs["SPACECRAFT_ID"], scene_cloud_percent=attrs.get("CLOUD_COVER"),
                path=int(attrs["WRS_PATH"]), row=int(attrs["WRS_ROW"]),
                stac_url=f"{STAC_URL}/collections/{COLLECTION}/items/{product_id}_SR")
    return info, paths


def prepare_pair(before: Path, after: Path, output: Path, size: int = 256,
                 low: float = 0.0, high: float = 0.3, min_valid_percent: float = 90.0) -> dict:
    """Prepare aligned RGB + common quality mask from local GeoTIFFs, at native 30 m.

    Writes a new bundle only. A manifest is written last so interrupted preparation
    cannot be mistaken for a usable pair. Original downloaded scenes are untouched.
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling, transform

    if not 32 <= size <= 1024 or not 0 < min_valid_percent <= 100:
        raise ValueError("size must be 32..1024 and min-valid-percent must be >0 and <=100")
    reflectance_rgb(np.zeros((1,), dtype=np.uint16), low, high)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --out directory.")
    infos_paths = [local_scene(Path(before)), local_scene(Path(after))]
    ainfo, binfo = [x[0] for x in infos_paths]
    if ainfo["acquired"] >= binfo["acquired"]:
        raise ValueError("Before acquisition date must be strictly earlier than after")
    for info in (ainfo, binfo):
        if (info["path"], info["row"]) != (148, 47):
            raise ValueError("Powai preparation expects WRS path 148, row 047")
    if ainfo["platform"] != binfo["platform"]:
        raise ValueError("Use the same satellite for both dates to limit sensor differences")
    x, y = transform("EPSG:4326", "EPSG:32643", [CENTER[0]], [CENTER[1]])
    # Snap once to a metric grid; both scenes use exactly this transform and extent.
    west = math.floor((x[0] - size * 15) / 30) * 30
    north = math.ceil((y[0] + size * 15) / 30) * 30
    grid = from_origin(west, north, 30, 30)
    rgbs, valid_masks, scenes = [], [], []
    for info, paths in infos_paths:
        bands = {}
        for key in ("red", "green", "blue", "qa_pixel", "qa_radsat"):
            with rasterio.open(paths[key]) as src:
                if src.crs is None or src.count != 1 or src.dtypes[0] != "uint16":
                    raise ValueError(f"Expected one georeferenced uint16 band: {paths[key].name}")
                if not src.crs.is_projected or not np.allclose(src.res, (30, 30), atol=0.01):
                    raise ValueError(f"Expected native 30 m projected bands: {paths[key].name}")
                is_qa = key.startswith("qa_")
                # Out-of-coverage QA=65535 rejects the pixel; RGB=0 is fill.
                with WarpedVRT(src, crs="EPSG:32643", transform=grid, width=size, height=size,
                               src_nodata=src.nodata if not is_qa else None,
                               nodata=65535 if is_qa else 0,
                               resampling=Resampling.nearest if is_qa else Resampling.bilinear) as vrt:
                    bands[key] = vrt.read(1)
        dn = np.stack([bands[k] for k in ("red", "green", "blue")], -1)
        valid = quality_valid(bands["qa_pixel"], bands["qa_radsat"]) & (dn > 0).all(-1)
        rgbs.append(reflectance_rgb(dn, low, high))
        valid_masks.append(valid)
        scenes.append(dict(info, local_clear_percent=float(100 * valid.mean()),
                           files={k: dict(name=p.name, sha256=sha256(p)) for k, p in paths.items()}))
    common = valid_masks[0] & valid_masks[1]
    clear_pct = float(100 * common.mean())
    if clear_pct < min_valid_percent:
        raise ValueError(f"Only {clear_pct:.2f}% common valid pixels; require {min_valid_percent:g}%. "
                         "Choose clearer scenes or inspect coverage/quality before changing the limit.")
    output.mkdir(parents=True)
    for name, rgb in zip(("A.png", "B.png"), rgbs):
        rgb[~common] = 0  # Identical fill in both dates; these pixels are excluded from summaries.
        Image.fromarray(rgb).save(output / name)
    Image.fromarray(common.astype(np.uint8) * 255).save(output / "valid.png")
    manifest = dict(schema_version=1, status="prepared", area="IIT Bombay / Powai, Mumbai",
                    generated_utc=datetime.now(timezone.utc).isoformat(), center_lon_lat=list(CENTER),
                    size=[size, size], crs="EPSG:32643", transform=list(grid)[:6], pixel_size_m=30,
                    scenes=scenes, common_valid_percent=clear_pct, ground_truth=None,
                    processing=dict(rgb_bands=[4, 3, 2], reflectance_scale=SR_SCALE,
                                    reflectance_offset=SR_OFFSET, stretch=[low, high],
                                    resampling="RGB bilinear; QA nearest", qa="QA_PIXEL bits 0-5 unset; QA_RADSAT=0",
                                    invalid_fill="black on both dates; excluded from summaries",
                                    registration="common geospatial grid; residual alignment requires visual review"),
                    disclaimer=DISCLAIMER)
    manifest["files"] = {n: sha256(output / n) for n in ("A.png", "B.png", "valid.png")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_prepared(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Verify a prepared bundle before showing or evaluating it (no network access)."""
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if m.get("status") != "prepared" or m.get("ground_truth") is not None:
        raise ValueError("Expected a prepared, unlabelled USGS demonstration bundle")
    for name in ("A.png", "B.png", "valid.png"):
        if sha256(folder / name) != m["files"][name]:
            raise ValueError(f"Prepared file checksum mismatch: {name}")
    a, b = [np.array(Image.open(folder / n).convert("RGB")) for n in ("A.png", "B.png")]
    valid = np.array(Image.open(folder / "valid.png").convert("L")) > 0
    if a.shape != b.shape or a.shape[:2] != valid.shape or list(valid.shape) != m["size"] or not valid.any():
        raise ValueError("Prepared pair dimensions or valid mask do not agree")
    return a, b, valid, m
