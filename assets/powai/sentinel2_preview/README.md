# Free Sentinel-2 campus previews

Downloaded 29 October 2026 using anonymous HTTP range requests to the AWS Open Data Sentinel-2 COG mirror, discovered through Element 84 Earth Search. No paid access or account was used.

| Role | Acquisition | Item |
|---|---|---|
| Before | 2017-02-02 | S2A_43QBB_20170202_0_L2A |
| After | 2025-01-21 | S2A_43QBB_20250121_0_L2A |

Both are Sentinel-2A, tile 43QBB. Each crop is 256 by 256 pixels at native 10 m spacing, covering 2.56 by 2.56 km around approximate campus centre 19.1334 N, 72.9133 E. They use identical EPSG:32643 grids. PNGs are for visual inspection; corresponding TIFFs retain georeferencing. `provenance.json` records source URLs, dates, grid and output hashes; full catalog items are retained alongside them. `download_preview.py` reproduces these crops and replaces files in this preview directory when rerun.

These are provider-rendered true-colour (TCI) products. No local sharpening, super-resolution or resizing was applied. Visual inspection shows more campus detail than Landsat, but no quantitative alignment assessment or local scene-classification screening has been performed. The historical processing baselines differ, and matching display rendering is not established. These previews have NOT been run through the models or substituted into the Streamlit USGS page. Before quantitative comparison, retrieve and consistently scale the original RGB bands, check the provider's offset conventions, and apply local quality masks. A 10 m pixel still spans 20 times the linear extent of a 0.5 m benchmark pixel.

Source documentation:
- https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2 (free data access and sensor resolution)
- https://github.com/Element84/earth-search (public catalog and open-data assets)

The original Landsat images and completed inference remain in `assets/powai/pair/` and `outputs/new/powai/` respectively.
