# USGS imagery for IIT Bombay / Powai

This is a separate exploratory application of frozen checkpoints. LEVIR-CD remains the source of benchmark accuracy. No Powai F1/IoU is available without independently annotated reference masks.

## 1. What has actually been obtained

The public USGS LandsatLook STAC catalog was queried on 29 October 2026. Metadata and direct asset links are saved in `assets/powai/selection.json`. The author downloaded and extracted both original scenes into `data/usgs/before/` and `data/usgs/after/`. Fresh clones include the small prepared pair and saved predictions; full original scenes are not bundled. All six required files per scene passed preparation checks. Never paste passwords, session cookies or API tokens into this repository or chat.

Actual preparation and inference completed on 29 October 2026. The bundle in `assets/powai/pair/` has 65,406 common valid pixels out of 65,536 (99.80%), with 130 excluded pixels. Visual review of both RGB previews found broadly corresponding lake and road features; this is not a quantitative registration assessment. All three frozen models ran on CUDA, and probabilities, evidence ZIPs and `report.json` are saved under `outputs/new/powai/`. At threshold 0.5 each model predicts zero changed pixels. This does not establish that no real changes occurred. The preparation manifest and saved probability hashes were verified, as were ZIP integrity and agreement between exported and saved probabilities.

Suggested same-sensor, same-season pair, WRS path 148 / row 047:

| Role | Acquisition | USGS product ID | Whole-scene cloud |
|---|---|---|---:|
| Before | 2015-02-10 | `LC08_L2SP_148047_20150210_20200909_02_T1` | 0.01% |
| After | 2025-02-05 | `LC08_L2SP_148047_20250205_20250208_02_T1` | 0% |

These are genuine Landsat 8 Collection 2 Level-2 Tier 1 acquisitions. Selection prioritizes low scene cloud and similar season; it does not select images for attractive model predictions. Whole-scene cloud percentage does not prove that the campus crop is clear. Local QA and visual inspection are still required.

## 2. Download in your browser

1. Open [USGS EarthExplorer](https://earthexplorer.usgs.gov/) and sign in with your own account.
2. Set a point near latitude **19.1334**, longitude **72.9133** (approximate IIT Bombay campus centre), or use the search area in `selection.json`.
3. Select the **Landsat 8-9 OLI/TIRS Collection 2 Level-2** dataset. Use the product IDs above to identify the acquisitions, or search the two date windows January-February 2015 and January-February 2025. Match path/row **148/047** and Tier 1. Portal wording can change; the product ID is the definitive check.
4. Use product/band download options, or download the full Level-2 bundle. Do **not** use a thumbnail, screenshot, map basemap, or reduced-resolution browse image as model input.
5. For each date, retain these six files with their original names:
   - `<product_id>_SR_B4.TIF` (red)
   - `<product_id>_SR_B3.TIF` (green)
   - `<product_id>_SR_B2.TIF` (blue)
   - `<product_id>_QA_PIXEL.TIF`
   - `<product_id>_QA_RADSAT.TIF`
   - `<product_id>_MTL.json`
6. Extract the two acquisitions into separate directories, for example `data/usgs/before/` and `data/usgs/after/`. Windows extraction or Python's standard archive tools can be used; no desktop GIS is required. `prepare_usgs.py` takes extracted directories, not `.tar` files. Keep exactly one metadata JSON per directory tree.

Page 10 exposes the official catalog and individual asset links. An asset URL that returns an HTML login page has **not** downloaded a TIFF. Sign in and use EarthExplorer's download controls. The preparation code rejects such files.

## 3. Prepare the local pair

For additional imagery, from the project root after `bash setup.sh` (the original pair is already bundled):

```bash
./python -m pip install -r requirements-usgs.txt
./python scripts/prepare_usgs.py --before "data/usgs/before" --after "data/usgs/after" --out assets/powai/pair_v2
```

Use absolute paths if the scenes are elsewhere. No network is used by preparation. The original downloads remain untouched.

Preparation performs the following:

1. Checks MTL metadata for Landsat 8/9 Collection 2 Level-2 Tier 1, the same satellite, WRS 148/047, and strictly increasing dates. Matches every required band's filename to that product ID. Requires one georeferenced uint16 band per file at native 30 m resolution.
2. Crops/reprojects both dates to the **same 256 × 256 grid in EPSG:32643**, with 30 m pixels centred near IIT Bombay. This covers **7.68 × 7.68 km**, including surrounding neighbourhoods. North is up. It does not enlarge the imagery to pretend that 0.5 m detail exists.
3. Uses bilinear resampling for RGB and nearest-neighbour for QA. Converts digital numbers to reflectance using `DN × 0.0000275 − 0.2`.
4. Uses **the same fixed linear stretch, reflectance 0.0–0.3**, on both dates to make 8-bit RGB inputs. There is no independent per-image contrast equalization. This display-to-model conversion is a documented domain-transfer choice, not the paper's original input protocol.
5. Rejects fill, dilated cloud, cirrus, cloud, cloud shadow and snow flags (QA_PIXEL bits 0–5), any QA_RADSAT flags, and RGB fill. Water is retained. Only pixels valid on **both** dates enter summaries.
6. Requires at least **90% common valid pixels** by default. A cloudy/out-of-coverage pair fails instead of generating a misleading demo. High-confidence QA screening is not a guarantee against haze or missed clouds.
7. Fills invalid pixels black on both dates, saves `A.png`, `B.png`, `valid.png`, and writes `manifest.json` last. The manifest contains dates, IDs, input-file SHA-256 hashes, processing choices, CRS/transform, and prepared-file hashes. There is **no ground-truth label**.

The default output is `assets/powai/pair/`. Existing output directories are refused to prevent mixing old and new imagery. For another preparation, use `--out assets/powai/pair_v2`; inspect it before choosing which bundle belongs in the default app location.

Optional preparation controls:

| Option | Default | Effect |
|---|---:|---|
| `--size` | 256 | Square width in native 30 m pixels, allowed 32–1024. Larger crops are tiled by inference. |
| `--low`, `--high` | 0.0, 0.3 | Shared reflectance stretch. Record any change; it changes model inputs. |
| `--min-valid-percent` | 90 | Minimum overlap passing QA. Do not lower merely to obtain a visually impressive result. |
| `--out` | `assets/powai/pair` | New output directory. The app reads the default location. |

Common-grid reprojection does not establish perfect co-registration. Inspect fixed roads, lake boundaries and large landmarks using page 10's blend slider. Double edges indicate residual mismatch. Differences in season, haze, water level and shadows can cause apparent change even without new buildings. Invalid pixels can also influence nearby network predictions despite being excluded from percentage calculations.

## 4. Run inference and keep evidence

In Streamlit, open page 10, inspect the inputs and valid mask, select models, check the inspection box, then click **Run selected models**. Start with TinyCD. Add BIT and ChangeFormer for comparison. The threshold slider changes the binary mask; it does not update model weights.

Alternatively, precompute all three outputs sequentially:

```bash
./python scripts/infer_usgs.py
# CPU-only example:
./python scripts/infer_usgs.py --models tinycd --device cpu
```

CLI options: `--pair` chooses a prepared bundle, `--out` chooses an output directory, `--models` accepts one or more of `tinycd bit changeformer`, `--device` accepts `auto cpu cuda`, and `--threshold` defaults to 0.5. `--threshold` must be in [0,1]. The UI's exploration slider uses 0.05–0.95 in 0.05 steps.

Outputs go to `outputs/new/powai/`: float32 `prob_<model>.npz`, one evidence ZIP per model, and `report.json`. The report links predictions to the exact preparation manifest and stores checkpoint provenance and hashes. Page 10's **Saved predictions** mode uses local outputs when their report exists, otherwise the bundled `assets/powai/predictions/`, and verifies these hashes before displaying them. Recomputing may replace outputs in the selected output directory; use a new `--out` to preserve another run. The UI reads the default output directory.

The evidence ZIP contains `before.png`, `after.png`, `probability.npy`, `mask.png`, `valid.png`, `overlay.png`, and `metadata.json`. The manifest retains the geospatial transform; the PNGs themselves are not GeoTIFFs. Grey in overlays means excluded QA pixels, pink means predicted model response. Invalid probabilities are retained in the raw array but must be masked with `valid.png` for analysis.

Do not report F1/IoU or verified building-area change from this unlabelled pair. A plausible image is not proof of accuracy. A small or zero predicted response is also a legitimate result; do not tune the dates or threshold to manufacture a success.

## 5. Refresh the catalog search

```bash
./python scripts/search_usgs.py
./python scripts/search_usgs.py --before 2015-01-01 2015-02-28 --after 2025-01-01 2025-02-28 --max-cloud 5 --out assets/powai/selection.json
```

This downloads **metadata only**, filters to Landsat 8 Tier 1 and path/row 148/047, and records all matching candidates. Changing the selection JSON does not change a prepared pair; actual input provenance comes from its manifest. The prepared pair is hash-verified when loaded.

## 6. Sources and interpretation

### Freely downloadable closer-view alternative

Following review of the Landsat scale limitation, native 10 m Sentinel-2A true-colour previews for 2017-02-02 and 2025-01-21 were downloaded without credentials from the public Earth Search/AWS archive. They are saved separately in `assets/powai/sentinel2_preview/`, with georeferenced TIFFs, PNGs, catalog records and provenance. Each covers 2.56 km across. These are inspection candidates only: local quality screening, consistent raw-band reflectance preparation and quantitative registration assessment have not been performed, and no inference is claimed for them. See that folder's README. The existing Landsat demonstration and results remain unchanged. Sentinel-2 is a Copernicus source; these downloads are not USGS acquisitions.

- [USGS STAC catalog documentation](https://www.usgs.gov/landsat-missions/spatiotemporal-asset-catalog-stac) and the exact catalog records linked in `selection.json` document discovery and provenance.
- [USGS Surface Reflectance](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance) documents 30 m resolution and reflectance scaling.
- [USGS QA band definitions](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands) document pixel-quality flags.
- [Rasterio reprojection](https://rasterio.readthedocs.io/en/stable/topics/reproject.html) documents the Python reprojection operation.

USGS Landsat 8-9 Level-2 Collection 2 dataset citation: Earth Resources Observation and Science (EROS) Center (2020), DOI [10.5066/P9OGBGM6](https://doi.org/10.5066/P9OGBGM6). These are real satellite observations rendered as natural-colour RGB, not ground-level photographs. A 30 m pixel has 60 times the linear extent and 3,600 times the area of a 0.5 m LEVIR-CD pixel; campus buildings are generally not individually resolved. For a meaningful building-scale transfer study, a separately sourced, well-registered high-resolution image pair and reference annotations would be needed.
