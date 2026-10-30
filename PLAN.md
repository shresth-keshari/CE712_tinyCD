# CE712 plan: inference-only replication

Updated scope: 29 October 2026. Source paper: TinyCD, Neural Computing and Applications 35 (2023), DOI 10.1007/s00521-022-08122-3.

## Deliverable

A Python implementation verified against released TinyCD weights, a common inference comparison against BIT and ChangeFormer, an explanatory Streamlit app, a separate USGS Powai demonstration, and an exhaustive demo guide. No model training, fine-tuning, optimizer, training curve or independent training-reproduction claim is included.

## Evidence contract

- **Measured:** current inference runs or preserved results with recorded protocol and environment.
- **Reported:** paper numbers with their table citations; in particular variant accuracy is not re-measured.
- **Derived:** formulas, parameter ratios and calculations from saved counts.
- **Probe:** an inference-time intervention on a released checkpoint, not a separately fitted variant.
- **Qualitative:** image/feature interpretation and the unlabelled Powai application.

Benchmark: LEVIR-CD256 test partition, 2,048 pairs, threshold 0.5, global pixel confusion matrix. The dataset's train/validation files may remain for integrity checking; their presence does not imply local training. Preserve the original benchmark artifacts during this scope change.

## Work and gates

| Work | Gate/status |
|---|---|
| TinyCD architecture | Released weights load strictly; 285,128 parameters; approximate FLOP agreement |
| Full-test comparison | Preserved TinyCD 91.07, BIT 89.94, ChangeFormer 90.40 F1; BIT discrepancy explicitly recorded |
| Statistical/efficiency/failure analyses | Retain bootstrap intervals, profiling, object recall, variants, mask probes and shift test |
| Remove training workload | Remove active training scripts, notebooks/cells, local checkpoints, augmentation API and training page; frozen checkpoints only |
| App | Home + 12 pages; page 10 USGS demo, page 11 limitations, page 12 exhaustive guide |
| USGS discovery | Real Landsat 8 metadata for 2015-02-10 and 2025-02-05 over path 148/047 obtained |
| USGS inputs | Both original scenes downloaded and extracted locally; required RGB, QA and JSON metadata validated |
| USGS preparation | Actual 256 by 256 common 30 m grid prepared; 99.80% common valid pixels; previews visually reviewed; hashes/provenance retained |
| USGS inference | All three frozen models completed on CUDA; verified probabilities and evidence ZIPs saved; zero predicted changed pixels at threshold 0.5, no accuracy without labels |
| Guide and documentation | Every app option, expected result, failure state and export covered; current scope reflected in all maintained first-party docs |
| Verification | Core, app and geospatial tests; consistency/hash checks; document actual checks and unperformed ones |
| Presentation | Slides, offline rehearsal, timed demo and backup recording remain to be prepared by the group |

## Powai interpretation

Landsat Collection 2 Level-2 RGB is 30 m per pixel, versus 0.5 m LEVIR-CD. Use native-resolution inputs and disclose the mismatch. The suggested acquisitions are same sensor and season, low whole-scene cloud, but local cloud/registration still need checking. Predicted changed-pixel percentage is a model-response summary, not accuracy, verified construction or building area. Keep this evidence separate from LEVIR-CD. See USGS_WORKFLOW.md.

## Presentation and course preparation

Use the 15-minute walkthrough in demo_guide.md. Explain the problem, architecture/Mix, fixed-protocol evidence, efficiency, failures and limitations, then the separate Powai application if ready. Both group members should be able to explain co-registration, class imbalance, metrics, shared encoders, inference normalization, frozen weights and evidence provenance. The source course instructions are in paper/. Confirm paper eligibility under the course's Q1 clause and presentation logistics with the group; this implementation does not claim those administrative checks are complete.

## Remaining deliverables

A clean-machine install and download test; actual offline rehearsals and backup; presentation slides. Real USGS download, preparation and inference are complete; independent reference labels and quantitative registration validation remain outside the current demonstration. A written report is optional under the recorded project plan. Do not revive the archived training queue as a prerequisite for completing this scope.
