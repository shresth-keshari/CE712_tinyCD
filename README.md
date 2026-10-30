# TinyCD: inference-only replication and explanation

CE712 mini project based on Codegoni, Lombardi & Ferrari, *TinyCD: A (Not So) Deep Learning Model for Change Detection*, Neural Computing and Applications 35:8471-8486 (2023), DOI 10.1007/s00521-022-08122-3.

We implement TinyCD, load frozen released checkpoints, and reproduce inference results on LEVIR-CD with official BIT and ChangeFormer comparators. The Streamlit application explains the architecture, measurements and limits. There is no training, fine-tuning, optimizer loop or training demo in the active project.

A separate IIT Bombay / Powai demonstration uses actual USGS Landsat downloads from 10 February 2015 and 5 February 2025. Both scenes are prepared locally, with 99.80% common valid pixels; saved inference and evidence ZIPs are available for all three models. At threshold 0.5 all three predict zero changed pixels. At 30 m resolution this is exploratory domain transfer, not verified building-change accuracy or proof of no actual change.

## Clone, install, run

Use Bash on Linux/WSL, macOS on Apple Silicon, or Windows x64 **Git Bash**:

```bash
git clone https://github.com/shresth-keshari/CE712_tinyCD.git
cd CE712_tinyCD
bash setup.sh
./streamlit run app/app.py
```

Open **http://localhost:8501**. No configuration edits, activation command, system Python installation, CUDA, administrator access or USGS account are needed. The `./streamlit` launcher uses this clone's `.venv` and passes through normal Streamlit arguments. Stop with Ctrl+C. Bash, curl, tar, internet access and enough free disk space are required; Git Bash provides these tools on Windows. Use separate clones for Windows and WSL.

Setup downloads a checksum-pinned Python 3.12 runtime into `.tools/`, creates `.venv`, installs pinned packages with CPU PyTorch, fetches the pinned upstream BIT source, downloads and verifies all three released checkpoints and LEVIR-CD256, then runs real CPU inference with each model. The dataset archive alone is **2.48 GB**; reserve approximately **15 GB** free for a full installation. The first setup can take a while. Reruns reuse the environment and verified dataset/weights. Upstream downloads must be reachable; failures stop setup with an error instead of silently disabling a model.

Optional lighter installation (fixed demo pairs, uploads, all saved result pages and Powai work; arbitrary LEVIR test-pair selection and fresh full-dataset evaluation require the default setup):

```bash
bash setup.sh --demo-only
./streamlit run app/app.py
```

For a compatible NVIDIA driver, `bash setup.sh --cuda` installs the CUDA 12.6 build instead. CPU is the portable default; recorded GPU timing numbers are not timing promises for another machine. Intel macOS is outside this pinned PyTorch configuration. Linux ARM64 and Apple Silicon support are included in the bootstrap; see VALIDATION.md for platforms actually tested.

After setup, project scripts can use `./python scripts/<name>.py`; tests use `./python -m pytest -q`. Standard `streamlit run app/app.py` also works after activating `.venv/bin/activate` (or `.venv/Scripts/activate` in Git Bash). Existing `.venv_win` environments are left untouched.

The repository includes six labelled LEVIR demo pairs, saved benchmark evidence, the prepared Landsat Powai images and saved Powai predictions. Large original satellite scenes, dataset archives and checkpoints are not committed. No fitting or training runs during setup.

## Read these documents

- [demo_guide.md](demo_guide.md): exhaustive startup, page/control walkthrough, expected evidence, exports, reproducibility and troubleshooting. Also available on app page 12.
- [USGS_WORKFLOW.md](USGS_WORKFLOW.md): exact USGS acquisitions, browser download instructions, native-resolution alignment/QA preparation, inference and provenance.
- [PROJECT_STATUS.md](PROJECT_STATUS.md): completed work and explicit pending items.
- [PLAN.md](PLAN.md): inference-only scope, acceptance criteria and presentation plan.
- [VALIDATION.md](VALIDATION.md): checks performed during the scope change and their limits.

## Preserved benchmark

All 2,048 LEVIR-CD test pairs, shared metrics, threshold 0.5, released checkpoints:

| Model | Paper F1 | Measured F1 |
|---|---:|---:|
| TinyCD, our implementation | 91.05 | 91.07 |
| BIT, official implementation | 89.31 | 89.94 |
| ChangeFormer V6, official implementation | 90.40 | 90.40 |

TinyCD: 285,128 parameters and about 1.456 GFLOPs. BIT's score differs from the paper; the cause remains unknown. Saved uncertainty, efficiency, object-level analysis, architecture counts, mask probes and shift experiments are retained. These scores do not represent locally fitted models or the Powai scene pair.

## Layout

| Path | Purpose |
|---|---|
| `app/app.py`, `app/common.py`, `app/views/` | Streamlit navigation, cached helpers, Home and pages 1-12 |
| `cdlib/` | Inference-only data loading, models, metrics, analysis, visualization, USGS preparation and evidence exports |
| `cdlib/models/vendor/` | Official comparator model implementations and provenance; upstream compatibility utilities may remain unchanged |
| `scripts/` | Dataset/checkpoint acquisition, evaluation, statistics, profiling, probes, scene discovery/preparation, inference and verification |
| `results/` | Preserved full-test evidence and separate artifact verification record |
| `assets/demo/` | Six fixed LEVIR-CD examples with labels and saved fallback predictions |
| `assets/powai/selection.json` | Genuine USGS catalog records and selected downloads; not pixel data |
| `assets/powai/pair/` | Bundled prepared USGS RGB/QA pair and hash-preserved provenance |
| `assets/powai/predictions/` | Bundled saved Powai probabilities, report and evidence ZIPs; used when no local recomputation exists |
| `data/`, `weights/` | Original dataset and downloaded checkpoints; no local training checkpoints are required |
| `outputs/new/` | Regenerable runtime outputs, including Powai predictions and evidence ZIPs |
| `notebooks/` | Optional inference walkthroughs using the same library; no separate experimental pipeline |
| `dashboard/dist/` | Supplementary static benchmark display; refresh with `scripts/export_dashboard.py` |
| `tests/` | Metrics, architecture, inference, USGS processing/export tests and app smoke/interaction checks |
| `THIRD_PARTY.md` | Source publications, upstream references and asset attribution |
| `requirements-usgs.txt` | Optional Rasterio dependency for original USGS GeoTIFF preparation |

Training artifacts were removed from the active workspace to the sibling recovery archive `../Final_mini_project_training_archive_20261029/`. The project does not import or depend on it. Raw LEVIR-CD split names and references to how the authors produced their checkpoints remain scientifically relevant. Upstream reference repositories and papers are preserved as sources, not edited to rewrite their history.

## Regenerate results

The full sequential commands and options are in [demo_guide.md](demo_guide.md#16-reproduce-the-evidence-without-training). Each model is evaluated in a separate process. Evaluation overwrites the named result files; the current scope changes preserve the existing benchmark files.

The bundled Powai demonstration is ready immediately. To prepare a different pair, follow [USGS_WORKFLOW.md](USGS_WORKFLOW.md) and choose a new output directory:

```bash
./python scripts/prepare_usgs.py --before data/usgs/before --after data/usgs/after --out assets/powai/pair_v2
./python scripts/infer_usgs.py --pair assets/powai/pair_v2 --out outputs/new/powai_v2
```

The app shows the bundled prepared pair immediately; a missing-data state is retained for installations without that bundle. No USGS password or token is stored in project code.

## Sources and redistribution

This is an academic replication. See [THIRD_PARTY.md](THIRD_PARTY.md) for source attribution and asset terms. BIT's unlicensed source and all model weights are fetched directly from upstream rather than republished. The paper and private course documents are not included in the public repository; the paper is linked by DOI above. ChangeFormer's MIT licence is retained with its code.
