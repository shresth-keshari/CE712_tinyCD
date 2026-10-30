# Validation record — inference-only scope change

Date: 29 October 2026. Environment: existing `.venv_win`, Python 3.12, PyTorch 2.7.0 + CUDA 12.6. Optional USGS preparation dependency Rasterio 1.4.4 was installed in this environment.

## Checks performed

| Check | Result and scope |
|---|---|
| `python -m pytest -q -p no:cacheprovider` | **46 passed**, with dependency deprecation warnings. Includes all 13 app views (Home + pages 1–12). |
| Strict checkpoint loading | TinyCD, BIT and ChangeFormer load; evaluation mode and frozen parameters asserted. |
| Preserved demo regression | CPU inference for all three models agrees with the existing GPU-generated 8-bit previews to within one 8-bit bin at every pixel, and half a bin in mean absolute error. This permits quantization/device arithmetic differences; it is not a fresh full-dataset evaluation. |
| `scripts/verify_artifacts.py --check-weights` | All three 2,048-pair per-image arrays reproduce saved aggregate counts and metrics, agree on test ordering, and match local checkpoint SHA-256 values. Six demo bundles are present. Record: `results/verification.json`. |
| App interaction checks | Page 5 threshold exploration and export controls; page 10 missing-data state, inspected-input run gate, synthetic prepared-pair inference/export, threshold changes, saved-output handling and hash rejection. |
| Geospatial preparation tests | Synthetic, georeferenced uint16 GeoTIFFs verify native-grid alignment, fixed reflectance scaling, QA bit rules, cloudy-pair rejection, date ordering, refusal to overwrite an existing bundle, and input/output provenance checks. |
| Evidence export tests | Original float probabilities preserved; invalid pixels excluded from masks and denominators; no accuracy fields invented for unlabelled imagery. |
| Source/notebook syntax | All first-party Python sources and the current notebook code cells parse successfully. Not all optional notebooks were executed end-to-end. |
| Actual Streamlit server | Started locally on port 8512; both `/` and `/_stcore/health` returned HTTP 200. Headless AppTest supplied page/interaction validation; no browser screenshot or recorded rehearsal is claimed. |
| USGS discovery | Live public catalog query succeeded. Selected genuine 2015-02-10 and 2025-02-05 Landsat 8 scene records, with direct asset URLs, are saved in `assets/powai/selection.json`. |
| USGS download access | An unauthenticated asset request redirected to the USGS login page. No pixel files or credentials were obtained. |

The new tests found and resolved Windows case-insensitive duplicate metadata discovery. Regression tolerance accounts for saved preview quantization and CPU/GPU arithmetic. Warnings originate from Streamlit/pandas, timm, Rasterio/NumPy, and Rasterio/Affine compatibility; no failing tests remain in the recorded checks.

## Preserved scientific evidence

The benchmark evaluation JSON/NPZ files, statistics, efficiency measurements, architecture-variant counts, probes, shift tests and six fixed demo bundles are retained. Their original measurements date to 26 October 2026. No full benchmark was regenerated merely to change the app's scope. Reported F1 remains TinyCD 91.07, BIT 89.94 and ChangeFormer 90.40.

The former training material and original mixed-purpose notebooks were moved to the sibling `../Final_mini_project_training_archive_20261029/` after automatic review blocked the bulk deletion command. No runtime imports or commands depend on that archive. Non-training source data, released weights, upstream references and scientific evidence remain in the project.

## Explicitly not yet performed

- Quantitative registration assessment or independently labelled accuracy evaluation of the Powai imagery. Actual download, preparation, visual preview review and inference are now complete; see the real-data record below.
- New full-test model evaluations, fresh-machine installation/download reproduction, or exhaustive execution of every possible widget-value combination.
- A physical offline presentation rehearsal, presentation slides, or backup screen recording.
- Any model training or fine-tuning.

For a live demonstration, follow `demo_guide.md` and retain the actual evidence ZIPs. After real USGS preparation/inference, update this record with the input manifest and output report rather than reusing the software-test claims as imagery validation.

## Actual Powai run, 29 October 2026

The user supplied extracted original 2015-02-10 and 2025-02-05 scenes in `data/usgs/before/` and `data/usgs/after/`. The preparation command completed successfully using the default shared stretch and grid. `assets/powai/pair/manifest.json` records original-file hashes, the 256 by 256 EPSG:32643 grid at 30 m, and 65,406 common valid pixels (99.8016357421875%); 130 pixels are excluded. Both RGB previews were inspected: lake and major road features broadly correspond. This is a qualitative check, not a subpixel registration measurement.

`scripts/infer_usgs.py` completed for all three released checkpoints on `cuda:0`. Each model predicts exactly zero changed pixels at threshold 0.5. Maximum probabilities among valid pixels were TinyCD 0.007676, BIT 0.009377 and ChangeFormer 0.001056. These low responses do not establish an absence of real change or validate transfer from the 0.5 m benchmark domain.

Evidence is in `outputs/new/powai/report.json`, `prob_<model>.npz` and `<model>_evidence.zip`. A separate verification loaded the prepared bundle with hash checks, matched the report's manifest and prediction hashes, recomputed every prediction summary, checked ZIP CRC integrity, and confirmed exact equality of exported versus saved float probabilities for all three models. No benchmark was rerun, and no model weights or thresholds were changed.
