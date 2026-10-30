# TinyCD inference-only demo guide

This guide covers setup, every application page and control, evidence export, reproducibility, and failure recovery. The app evaluates **frozen released checkpoints**. There is no model training, fine-tuning, optimizer, epoch loop or training demonstration.

The benchmark asks whether the released TinyCD, BIT and ChangeFormer checkpoints reproduce the recorded LEVIR-CD findings under one evaluation protocol. The Powai demonstration separately asks how those same checkpoints respond to real USGS satellite imagery from a new location and resolution. Its predictions are exploratory, not benchmark evidence.

## 1. Start from the project folder

Use Bash (Linux/WSL, Apple Silicon macOS, or Windows Git Bash):

```bash
git clone https://github.com/shresth-keshari/CE712_tinyCD.git
cd CE712_tinyCD
bash setup.sh
./streamlit run app/app.py
```

Setup supplies a local Python 3.12 runtime, isolated `.venv`, CPU PyTorch, pinned dependencies including Rasterio, upstream BIT source, released weights and the complete LEVIR-CD256 dataset. It checks hashes and runs each frozen model once. No manual path changes, activation, system Python, CUDA or USGS login are required. Internet, Bash, curl, tar and approximately 15 GB free disk are needed for the default full installation. Allow time for the 2.48 GB dataset archive and checkpoints. Rerun the same setup command after an interrupted download; errors are not treated as successful installation.

For a smaller install, `bash setup.sh --demo-only` omits the full dataset. Fixed pairs, uploads, tables, prepared Powai inputs and saved Powai predictions still work; arbitrary test-pair selection and fresh full-dataset evaluation need the full setup. For an NVIDIA driver compatible with CUDA 12.6, `bash setup.sh --cuda` opts into the CUDA build. CPU is the default. Use separate Windows and WSL clones because virtual environments are OS-specific.

The repository-local `./streamlit` and `./python` launchers select the environment automatically. You can also activate `.venv/bin/activate` (Git Bash on Windows: `.venv/Scripts/activate`) and use normal `python` / `streamlit` commands. Existing development installations in `.venv_win` remain usable independently.

Open the local URL printed by Streamlit, normally `http://localhost:8501`. Leave the terminal running; Ctrl+C stops the server. Append `--server.port 8502` to use another port. After setup, the app's local inference, fixed examples and included evidence work offline; external source links and new imagery downloads need internet.

Validation commands:

```bash
./python -m pytest -q
./python scripts/check_install.py
```

The original Powai RGB/QA pair is bundled, so page 10 is ready immediately. Follow `USGS_WORKFLOW.md` only if acquiring or preparing additional scenes. The setup script performs acquisition; the app itself does not automatically download imagery or weights.

## 2. Shared controls and evidence language

Use the left navigation to open Home and pages 1–12. Navigation reruns a page; cached models and data avoid repeated loading. Large models can take longer on first use. Model caches are separate for CPU and GPU.

**Compute:** the sidebar identifies a usable CUDA GPU, otherwise it selects CPU. When a GPU is available, **Force CPU** switches inference to CPU. If model loading runs out of GPU memory, the helper attempts CPU fallback. Inference tables display the actual device used. Changing devices causes a separate model load. Saved benchmark numbers never change when this switch changes.

**Evidence labels:** Measured means a project computation; Reported means a published value; Derived means calculated from other values or formulas. Qualitative marks visual interpretation; Probe marks an intervention on a released model. None of these labels turns a Powai prediction into verified ground truth.

**Shared pair picker** on pages 1–5:

| Source/control | What to do | Expected behavior |
|---|---|---|
| Demo pairs (fixed set) | Select one of six named pairs | Loads local input images and reference labels. The selection rule is shown. |
| Any LEVIR-CD test pair | Search/select a filename | Available only when the dataset test list exists. Loads that test pair. |
| Upload your own pair | Available on page 5 | Upload before/after RGB images and optionally a reference mask. |

The six fixed cases are large development, a few buildings, model disagreement, a TinyCD failure, clean no-change and a TinyCD false alarm. They were selected by explicit rules over saved test predictions. They are illustrative examples, not a statistically representative sample.

Streamlit data tables can be scrolled, sorted by column headers, searched or downloaded through their built-in toolbar when available. Sorting a displayed table does not change the evaluation. Charts expose their plotted values on hover where tooltips are configured. These are viewing controls, not experiments.

## 3. Home — establish scope and headline evidence

Read the paper citation and the claim being checked. The three measured F1 cards should show:

| Model | Paper F1 | Measured F1 |
|---|---:|---:|
| TinyCD | 91.05 | 91.07 |
| BIT | 89.31 | 89.94 |
| ChangeFormer V6 | 90.40 | 90.40 |

All measured scores use 2,048 test pairs and threshold 0.5, with each model's released LEVIR-CD checkpoint. TinyCD uses this project's implementation; the competitors use vendored official implementations. The numbers are loaded from `results/eval_*_test.json`. They are not computed by opening Home.

Say: **“We replicate inference with released checkpoints, verify the implementation, and compare the models under the same test protocol.”** Do not describe the current deliverable as an independent model-training reproduction.

## 4. Page 1 — the problem

1. Select a pair. Move **Fade from t1 (0) to t2 (1)** from 0 to 1 in 0.05 increments. The before/after blend changes; labels do not.
2. Toggle **Outline the ground-truth change (yellow)**. The outline identifies the reference annotation, not a model prediction.
3. Inspect the separate black/white label and changed-pixel percentage.
4. Read the split table: 7,120 / 1,024 / 2,048 describes the source dataset's train/validation/test partition. This project performs no training on those splits.
5. Inspect the test-set class-balance histogram and the all-negative predictor example. A model that predicts no change scores approximately 94.91% overall accuracy because only about 5.09% of pixels are changed.

Proof: the all-negative predictor has zero change-class F1 despite high overall accuracy. Explain why precision, recall, F1 and IoU are used. Global metrics sum TP/FP/FN/TN over pixels before computing a score; they do not average per-image F1.

## 5. Page 2 — architecture

Read the diagram: shared EfficientNet-B4 blocks 0–3 → Mix/MAMB → masked upsampling decoder → pixelwise classifier. Check the parameter table totals **285,128**.

Choose a pair, then use **Encoder output to inspect**:

- Block 1: 24 channels at 128 × 128.
- Block 2: 32 channels at 64 × 64 (default).
- Block 3: 56 channels at 32 × 32.

The viewer displays six channels with the largest mean absolute before/after difference for that pair. The t1 and t2 channel images share a scale. Their absolute-difference row shows where the encoder responds differently. Changing the displayed stage does not change checkpoint weights.

Proof: parameter count plus strict checkpoint loading and forward-shape tests support architecture compatibility. Feature pictures provide interpretation, not accuracy measurements.

## 6. Page 3 — every Mix control

| Control | Range/options | Expected observation |
|---|---|---|
| Number of channels C (toy example) | 2–8; default 4 | Interleaved order is X0,Y0,X1,Y1,…; each group sees matching channels at two dates. |
| Kernel size k | 1,3,5,7; default 3 | Both cost formulas grow with k². |
| Channels C | 1–512; default 56 | Concat/Mix weight-count ratio equals C. Biases are discussed separately. |
| Image pair source / pair | Shared picker | Changes the actual feature tensors used in the numerical example. |
| Features to fuse | Blocks 1,2,3; default block 2 | Changes the channel width and spatial resolution. |
| centre weight for t1 | −2 to 2, step 0.5; default +1 | Controls the first centre coefficient in a temporary demonstration convolution. |
| centre weight for t2 | −2 to 2, step 0.5; default −1 | Controls the second coefficient. |
| Mix block | Bottleneck @32; MAMB @64, @128, @256 | Selects released kernels to inspect. |

At (+1,−1), the example convolution equals X−Y to floating-point precision. At (0.5,0.5), it computes an average. At (1,0), it copies X. The equality concerns the convolution **before PReLU and InstanceNorm**, not the entire nonlinear Mix block. These controls build a temporary toy convolution; they do not alter the loaded model.

The final kernel viewer reports how many released kernels have opposite total signs for the two date halves and plots up to eight kernels by magnitude. This visual inspection does not prove that every kernel is simple subtraction.

## 7. Page 4 — attention masks

Choose a pair and use **Colour map**: `jet`, `viridis` or `gray`. Inspect M256, M128 and M64 together with prediction/error displays and their raw value ranges. Colours are scaled for visualization. These intermediate masks can be negative; they are not calibrated change probabilities. Changing the colour map changes presentation only.

Proof: the app exposes intermediate tensors from the released model. Their suggested interpretation remains qualitative.

## 8. Page 5 — live inference, uploads and exports

1. Select a fixed pair, arbitrary test pair, or **Upload your own pair**.
2. Use the **Models** pills to select TinyCD, BIT and/or ChangeFormer V6. All three are selected by default. Selecting none suppresses inference output. TinyCD alone is useful on a slow CPU.
3. The **Decision threshold** slider ranges from 0.05 to 0.95, step 0.05, default 0.5. A response is change when P > threshold. Any non-default threshold is explicitly marked exploratory. This never rewrites full-test results.
4. Inference runs automatically for the selected input/model/device combination and is cached on repeats. Observe probability heatmap, binary mask and, when reference labels exist, TP/FP/FN overlay.
5. Compare per-pair metrics and predicted changed-pixel percentage. White = true positive, red = false positive, blue = false negative, black = true negative.
6. Click each **Download … evidence ZIP** button. Retain the ZIP as a record of the exact input and threshold.

**Uploads:** before and after images must have identical dimensions, at most 2048 pixels on each side. Supported extensions are PNG/JPG/JPEG/TIF/TIFF; supply display-ready RGB images. The optional mask is a same-size PNG/JPG/TIF/TIFF with zero = no change and any nonzero value = change. Prefer lossless PNG for masks because JPEG compression can create nonzero artifacts. A satellite's raw 16-bit single-band GeoTIFF is not display-ready RGB: use the USGS preparation script instead. The generic uploader does not georeference or align images and does not preserve geospatial metadata.

Larger images are padded and split into **non-overlapping 256 × 256 tiles** at their supplied resolution. Predictions are stitched and cropped back. There is no overlap averaging, and tile boundaries may be visible. Same dimensions alone do not establish alignment or matching resolution.

**Timing/source:** the table reports the actual device and time of the stored forward result. Loading and warm-up are excluded; cached reruns reuse that time. If live inference fails for a fixed demo pair, the app can show its saved **8-bit probability** image with source `pre-rendered` and no timing. Quantization can change threshold-edge pixels, so use full-precision saved benchmark counts for published metrics. Uploaded/dataset pairs have no such fallback.

**No-change labels:** recall is undefined and blank. If both reference and prediction are empty, precision/F1/IoU are blank too. If the model predicts change despite an empty reference, precision/F1/IoU are zero; every predicted change pixel is a false alarm.

**ZIP contents:** before/after PNGs, float32 `probability.npy`, thresholded `mask.png`, `valid.png`, `overlay.png`, `metadata.json`, plus `ground_truth.png` when supplied. The metadata identifies model, checkpoint hash, input source, actual device, threshold summary, timing, metrics and fallback precision. `probability.npy` stores the original full probability array; fallback values remain quantized even though stored as float32. ZIP overlays use pink for predicted change. Source validity is all pixels on page 5; the QA-screened validity mask is specific to page 10.

## 9. Page 6 — replication results

This page has tables and charts, with no model-execution button. Compare **Reported** and **Measured** rows and their deltas. Read the environment/date and notes explaining the BIT discrepancy.

Expand **Full paper Table 1 (reported only, for context)** to see the other published baselines. Those rows are publication context, not additional models evaluated in this inference-only comparison.

The 95% TinyCD F1 interval is approximately **90.63–91.45**. Paired bootstrap differences, TinyCD minus comparator, are approximately **+1.12 points vs BIT** and **+0.67 vs ChangeFormer**; their saved intervals exclude zero. There are 2,000 bootstrap resamples of test images, with fixed seed 0. This measures test-image sampling uncertainty, not variation from independently fitted models or spatially independent sites.

Proof: `scripts/verify_artifacts.py --check-weights` recomputes metrics from saved counts, verifies aligned test lists, checks demo-file presence and verifies downloaded checkpoint hashes. A fresh full-model evaluation is a separate command in section 16. Do not call an artifact consistency check a new evaluation.

## 10. Page 7 — size and speed

Inspect stored vs used parameter counts, approximate GFLOPs, GPU/CPU latency, memory and accuracy/size plots. TinyCD: **285,128 parameters**, about **1.456 GFLOPs**, saved median about **24.2 ms GPU / 194.4 ms CPU** per 256 × 256 pair. Those timings come from the recorded RTX 3050 laptop environment, not every user's machine.

For **Time it yourself**, select **Model**, choose **Runs** (5–50; default 20), and click **Measure on the current device**. The app performs three warm-up calls and reports median forward latency. GPU synchronization brackets timed operations. This optional live check does not replace `results/efficiency.json`. ChangeFormer CPU timing can take substantially longer than TinyCD.

BIT stores unused ResNet layers, so its stored and used parameter counts differ. FLOP counting excludes unsupported operators and is approximate; do not describe 1.45556 and 1.45 as numerically exact. Fewer FLOPs need not yield proportionally lower latency.

## 11. Page 8 — variants and inference probes

Read fusion/classifier/backbone tables and the reported-F1 chart. There are no local training or variant-fitting controls. Variant parameter counts are measured by constructing architectures; their accuracy values are labelled **Reported**.

Fusion/classifier counts match the corresponding paper counts; B0–B2 differ by 272 parameters. No-skip parameter count is an interpretation where the paper gives no count. Do not call every architecture variant an exact reproduction.

The probe table comes from the released TinyCD with one or all attention masks replaced by ones at inference. This asks how the existing checkpoint depends on those masks. It is different from evaluating a separately fitted no-mask model. The probe operation leaves the original checkpoint file unchanged.

## 12. Page 9 — failure analysis

Read object recall by size, hallucinated regions and false-alarm pairs. Definitions are explicit: an object is an 8-connected reference component, detected if at least 50% of its pixels are predicted; a no-change pair becomes a false-alarm pair at 50 or more predicted pixels.

The hardest-pair table restricts to more than 1% reference change and ranks the worst TinyCD F1. If the dataset exists, **Inspect a pair** selects among the listed examples and shows all three error maps. Without it, the table remains available but the image viewer explains the missing dataset.

The shift experiment is precomputed. It shifts t2 right and down by 0,2,4,8,16 pixels and scores the central 224 × 224 area. Even the zero-shift score differs slightly from page 6 because its border is excluded. This shows sensitivity to misregistration, not proven robustness to every kind of sensor/geometric change. No shift recomputation runs when opening this page.

## 13. Page 10 — IIT Bombay / Powai

Current local state (29 October 2026): both original scenes are prepared, with 99.80% common valid pixels. All three models have saved predictions and evidence ZIPs in `outputs/new/powai/`. Choose **Saved predictions** to inspect this completed run. At threshold 0.5 all three masks are empty (zero predicted changed pixels); this is the actual model response, not proof that the region has not changed. Input previews have been visually reviewed, but no quantitative registration or labelled accuracy assessment is available.

Before downloads, the page shows the two real USGS acquisitions, catalog/asset links, downloadable selection JSON and preparation instructions. **“Imagery not prepared” is the correct state.** It must not show synthetic inputs or invented predictions.

Follow `USGS_WORKFLOW.md`. With `assets/powai/pair/manifest.json` and its verified PNGs present, page 10 adds:

| Control | Use and expected result |
|---|---|
| Blend dates to inspect alignment | 0–1, step 0.05; compare unchanged landmarks for double edges. |
| Actual input provenance and processing | Expand for actual acquisition IDs, dates, hashes, grid, scaling and local QA fraction; download the manifest. |
| Models | TinyCD by default; add BIT/ChangeFormer or deselect all. |
| Exploratory decision threshold | 0.05–0.95, step 0.05; changes masks and percentages, not weights. |
| Prediction source: Live inference | Inspect images, check the inspection box and click Run selected models. Results persist while adjusting the threshold. |
| Prediction source: Saved predictions | Reads local `outputs/new/powai/` results when a report exists, otherwise bundled `assets/powai/predictions/`; verifies manifest and prediction hashes. Missing/stale outputs are explained rather than silently reused. |
| I inspected the two dates and their quality mask | Enables the live run button when at least one model is selected. It records a UI acknowledgement, not a scientific alignment certificate. |
| Run selected models | Computes selected models, with failures shown explicitly. Run again to refresh. |
| Download … evidence ZIP | Captures inputs, validity, float probabilities, response mask, overlay, threshold and provenance. |

Pink overlay = predicted response; grey = excluded quality pixels. The table reports predicted-change percentage **among valid pixels only**, not an accuracy metric. No F1/IoU/precision/recall is displayed because no reference mask is available.

The default crop is 7.68 km wide at 30 m/pixel. It includes the campus and surrounding Powai region. These are natural-colour satellite images, not individually resolved building photographs. The models' original imagery was 0.5 m/pixel; resizing cannot recover that missing detail. Say: **“This demonstrates inference on a different real-world input domain. These responses need independent interpretation and validation.”**

## 14. Page 11 — limitations and honesty

Read the claim/evidence table, implementation deviations, checkpoint sources/hashes, source licences and course notes. No controls trigger computation beyond table loading. Distinguish exact parameter matching from approximate FLOPs, reported ablations from inference probes, test-sampling intervals from model variability, and benchmark accuracy from the unlabelled USGS demonstration. WHU-CD remains outside scope.

## 15. Page 12 — this guide

Read the full guide in the app or click **Download the complete demo guide** to save Markdown. It is the same `demo_guide.md` in the project root. The guide describes expected behavior; successful tests and exported outputs supply evidence that the corresponding paths were exercised.

## 16. Reproduce the evidence, without training

Run expensive scripts sequentially, one model per process. These commands regenerate named output files; preserve a copy first if comparing environments. Do not run the full benchmark during a live presentation.

```bash
./python scripts/check_data.py
./python scripts/evaluate.py --model tinycd
./python scripts/evaluate.py --model bit
./python scripts/evaluate.py --model changeformer --batch 4
./python scripts/stats.py
./python scripts/profile_models.py
./python scripts/variants.py
./python scripts/probes.py
./python scripts/shift_robustness.py
./python scripts/select_demo.py
./python scripts/export_dashboard.py
./python scripts/verify_artifacts.py --check-weights
```

`evaluate.py --model` is required (`tinycd`, `bit`, `changeformer`); `--split` defaults to `test` (train/val are available only as dataset partitions), `--batch` defaults to 8, `--device` defaults to auto and `--workers` to 2. Use `--device cpu --workers 0` for a simple CPU evaluation. A non-test split does not replace the app's test results. Full scoring requires the dataset. Check each script's `--help` where it has an argument parser. `profile_models.py --cpu-runs` controls CPU timing repetitions; probe/shift scripts offer device/batch options, and shift also accepts a model list. Use the full default protocol to regenerate all app tables. `select_demo.py` regenerates the six demo bundles, so it changes assets rather than merely viewing them.

Outputs map to `results/data_summary.json`, `eval_*.json`, `per_image_*.npz`, `stats.json`, `efficiency.json`, `variants.json`, `probes.json`, `shift.json`, and `assets/demo/`. USGS outputs are separate. Source notebooks are optional walkthroughs of the same inference library; the old notebook-training queue has been removed from the active project.

## 17. Verification and evidence checklist

1. Run `python -m pytest -q`. Core tests cover metrics, architecture, checkpoint loading and tiled inference. App tests load every registered view. Added USGS tests use labelled synthetic fixtures solely to verify processing, quality rejection and provenance; these fixtures are never placed in the real Powai asset directory or presented as satellite observations.
2. Run `scripts/verify_artifacts.py --check-weights`. Read `results/verification.json`. This checks preserved benchmark consistency and provenance without overwriting benchmark scores.
3. Exercise pages 1–5 with a success, failure and no-change example. Export page 5 ZIPs. Change threshold then reset to 0.5.
4. Exercise CPU mode and the timing button. On page 10 inspect the actual prepared dates and quality mask, select Saved predictions, compare all three models and download evidence. All three currently predict zero changed pixels at threshold 0.5. Use automated fixtures to check the missing-data state without moving the actual imagery. A passing synthetic preprocessing test does not substitute for the real-data validation recorded in VALIDATION.md.
5. Open the app offline after setup. External source links/catalog search require connectivity; benchmark displays and local inference do not. Save evidence ZIPs and a screen recording of the actual demonstration if required.
6. Record actual results in `VALIDATION.md`; do not present an unperformed clean-machine installation, offline rehearsal, or Powai evaluation as completed.

## 18. Suggested 15-minute presentation

| Time | Action |
|---|---|
| 0–2 min | Home and page 1: task, labels, class imbalance and inference-only scope. |
| 2–5 min | Pages 2–4: architecture, Mix subtraction proof and attention masks. |
| 5–8 min | Page 5: a success, failure and no-change example; compare models; export one ZIP. |
| 8–10 min | Pages 6–7: measured vs reported results, uncertainty and size/speed. |
| 10–12 min | Pages 8–9: distinction between variants/probes; failure and misregistration. |
| 12–14 min | Page 10: actual USGS pair and results if prepared; otherwise explicitly show the ready workflow and pending download status. |
| 14–15 min | Page 11: scope and limitations, with a buffer. |

## 19. Troubleshooting

| Symptom | Action |
|---|---|
| Missing Python/Streamlit module | Rerun `bash setup.sh`; launch with `./streamlit` or `./python` so the correct environment is used. |
| Missing checkpoint | Run `download.py --weights` with internet access, then verify hashes. Do this before an offline demo. |
| GPU error or memory pressure | Enable Force CPU; select TinyCD alone. Restart the app to release cached models. |
| Missing full dataset | Use fixed demo pairs; download the dataset for arbitrary test-pair viewing/full evaluation. |
| Upload dimensions differ / exceed 2048 | Prepare matching aligned images or a smaller matching crop. Do not independently resize dates. |
| Raw USGS TIFF looks black/wrong | Use `prepare_usgs.py` on original bands and metadata; do not send individual bands through generic RGB upload. |
| USGS URL returns login HTML | Authenticate in your own browser and download via EarthExplorer. Renaming HTML to `.TIF` does not create imagery. |
| Multiple MTL files / missing band | Separate acquisitions and retain exactly one scene and the required original filenames per input tree. |
| Preparation has too few valid pixels | Inspect dates, cloud/QA, CRS and coverage. Prefer a clearer acquisition. |
| Preparation output already exists | Use a new `--out`; the script intentionally does not overwrite an existing pair. |
| Saved Powai outputs fail hashes | Re-run `infer_usgs.py` against the exact current bundle. Do not edit hashes to suppress the error. |
| A changed file is not visible in the app | Refresh/restart Streamlit to clear cached result data. |
| “Pre-rendered” instead of inference | A fixed-demo fallback was used. Its timing is blank; inspect the warning and fix the live-inference failure before claiming a live run. |
| Static dashboard cannot fetch JSON | Serve `dashboard/dist` with a local HTTP server; the Streamlit app is the primary demo. |

Scientific conclusion: the existing LEVIR-CD evidence supports inference replication of TinyCD and ChangeFormer and a higher-than-reported BIT result. A live UI, correct preprocessing, or visually plausible Powai mask alone does not establish accuracy on Mumbai imagery.
