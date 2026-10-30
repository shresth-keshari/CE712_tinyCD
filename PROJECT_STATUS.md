# Project status ? inference-only scope

Updated 29 October 2026. This file supersedes the earlier notebook-training tracker. The active project uses frozen released checkpoints; training artifacts and original mixed-purpose notebooks have been moved outside this workspace to `../Final_mini_project_training_archive_20261029/`.

## Completed evidence retained

- Full LEVIR-CD256 dataset and released checkpoints are present locally.
- TinyCD implementation strictly loads the official checkpoint: 285,128 parameters, approximately 1.456 GFLOPs.
- Full-test F1, 2,048 pairs at threshold 0.5: TinyCD 91.06578, BIT 89.94153, ChangeFormer 90.40072.
- Per-image counts, bootstrap confidence intervals, paired differences, profiling, object-level errors, architecture variant sizes, inference-time mask probes and shift robustness are retained.
- Six fixed demonstration pairs and saved fallback outputs are retained.

## Scope changes implemented

- Training page/scripts/results, optimizer/augmentation workflow, training-run checkpoints and orchestration removed from active use.
- Model loaders freeze released parameters; inference never downloads backbone weights.
- App has Home + 12 pages, with a Powai page and full demo guide replacing the former training emphasis.
- Evidence ZIP downloads include inputs, float probabilities, masks, validity, overlays and metadata.
- Public USGS catalog search identifies Landsat 8 acquisitions on 2015-02-10 and 2025-02-05, WRS 148/047.
- Local USGS preparation and inference commands, quality checks, provenance and honest missing-data states are implemented.
- Both original USGS scenes are downloaded and prepared in `assets/powai/pair/`: 99.80% common valid pixels. Actual CUDA inference for all three models and verified evidence exports are saved in `outputs/new/powai/`. Each model predicts zero changed pixels at threshold 0.5; this is not proof of no real change.
- Maintained notebooks are inference walkthroughs; the supplementary static dashboard uses current released-checkpoint results.

## Pending external input / work

1. Quantitative registration assessment and independently annotated Powai reference masks are unavailable. The actual RGB previews have been visually reviewed, but no Powai accuracy claim is supported.
2. Clean-machine install/download/evaluation reproduction has not been performed during this change.
3. Slides, offline rehearsals, backup recording and administrative course checks remain pending.
4. Most current project files were already untracked at the start; no Git commit/push is performed by this change.

## Validation

See VALIDATION.md for executed checks and results, and results/verification.json for machine-readable saved-artifact/hash verification. The preserved full-test results were originally generated on 26 October 2026; a source refactor is not a new full-test evaluation.

## Sources of truth

README.md: setup/layout. PLAN.md: scope and gates. demo_guide.md: exhaustive use of the app. USGS_WORKFLOW.md: acquisition/preparation. results/: benchmark evidence. assets/powai/selection.json: real public catalog metadata, not downloaded pixel data.
