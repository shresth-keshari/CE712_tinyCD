"""Download native 10 m true-colour campus previews, for inspection only.

These provider-rendered TCI crops are not shared-stretch surface-reflectance
inputs and are not wired into the Landsat inference demo. Original STAC items
and georeferencing are retained. No account or paid service is used.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.warp import transform
from rasterio.windows import Window

OUT = Path(__file__).resolve().parent
ITEMS = {
    "before": "S2A_43QBB_20170202_0_L2A",
    "after": "S2A_43QBB_20250121_0_L2A",
}
BASE = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/"


def main():
    record = {
        "purpose": "Native-resolution visual inspection only; no inference or accuracy claim",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "center_lon_lat": [72.9133, 19.1334],
        "pixel_size_m": 10,
        "size": [256, 256],
        "width_km": 2.56,
        "rendering": "Provider TCI; no local enhancement, resizing or super-resolution",
        "quality": "Cloud-filtered catalog candidates; local SCL screening and quantitative alignment not performed",
        "source": "Copernicus Sentinel-2 data, via Element 84 Earth Search / AWS Open Data",
        "scenes": {},
    }
    grid = None
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="45",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        for role, item_id in ITEMS.items():
            response = requests.get(BASE + item_id, timeout=40)
            response.raise_for_status()
            item = response.json()
            href = item["assets"]["visual"]["href"]
            with rasterio.open(href) as src:
                assert src.count == 3 and src.dtypes == ("uint8",) * 3
                assert src.crs.to_epsg() == 32643 and src.res == (10, 10)
                x, y = transform("EPSG:4326", src.crs, [72.9133], [19.1334])
                row, col = src.index(x[0], y[0])
                window = Window(col - 128, row - 128, 256, 256)
                rgb = src.read(window=window)
                current_grid = src.window_transform(window)
                assert grid is None or grid == current_grid
                grid = current_grid
                profile = dict(driver="GTiff", width=256, height=256, count=3,
                               dtype="uint8", crs=src.crs, transform=grid, compress="deflate")
                with rasterio.open(OUT / f"{role}.tif", "w", **profile) as dst:
                    dst.write(rgb)
            Image.fromarray(np.moveaxis(rgb, 0, -1)).save(OUT / f"{role}.png")
            (OUT / f"{role}_stac.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
            record["scenes"][role] = dict(
                item_id=item_id, acquired=item["properties"]["datetime"],
                cloud_percent=item["properties"]["eo:cloud_cover"], asset_url=href,
                stac_url=BASE + item_id,
                files={f"{role}.{ext}": hashlib.sha256((OUT / f"{role}.{ext}").read_bytes()).hexdigest()
                       for ext in ("tif", "png")})
            print(role, item_id, rgb.shape, flush=True)
    record["crs"] = "EPSG:32643"
    record["transform"] = list(grid)[:6]
    (OUT / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
