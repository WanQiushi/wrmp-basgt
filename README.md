# WRMP–BASGT experiment code

Code accompanying the wavelength-resolved bridge track-geometry reconstruction study. This repository contains the four-band BASGT model, its K=1/2/3 graph-depth comparison, the K=3 component ablations used in the manuscript, the data-only five-band comparison, and the model definitions required by those experiments. The dated directories preserve the model lineage used for the reported runs.

The data and trained weights are **not included**. The study uses a private 25-date bridge dataset (`PAPER2_V4_STATIC25_20260906`): 16 fitting dates and nine development dates, excluding 2024-08-28. The nine dates are a development set, not an untouched external test set. The repository starts from prepared wavelength-band arrays; raw monitoring ingestion and construction of the WRMP prior are outside this code snapshot.

## Contents

| Path | Purpose |
| --- | --- |
| `src/paper2_final/` | Shared graph blocks, data-only baseline bodies, metrics, and dataset verification |
| `experiments/v3_graph_five_band_7m_center_20260919/segments.py` | Fit adaptive boundaries from fitting dates |
| `experiments/v3_graph_four_band_ablation_20260919/` | K=1 four-band model and structural controls |
| `experiments/v3_graph_depth_ablation_20260919/` | K=2/K=3 training and K=1/2/3 evaluation |
| `experiments/v3_graph_k3_component_ablation_20260922/` | Three-layer component ablations matching the manuscript |
| `experiments/pure_blackbox_full_spectrum_20260921/` | Five-band direct-prediction BASGT and data-only baselines |
| `experiments/small_sample_reporting_20261029/` | Reuse the main-table 16-date endpoint in a separate reporting view |
| `reference_results/k3_component_and_full_budget.json` | Audited per-seed metrics, checkpoint depths, parameters, and epochs |
| `experiments/v3_graph_{soft_boundary24,peak_auto_count,all_band_auto_count}_*/` | Model ancestors imported by the dated experiments |

The model ancestors are source dependencies, not additional result sets. The vendored Time-Series-Library subset retains its upstream MIT license and provenance notice in `src/paper2_final/models/vendor/tslib_vendor/`.

## Install

Python 3.11 or newer and a PyTorch build suitable for the target machine are required. From the repository root:

```bash
python -m pip install -e .
```

Before running, set `PAPER2_DATA_ROOT` to a private directory containing `DATASET_LOCK.json`, `DATASET_BASELINE.json`, and an `inputs/` directory. Set `PAPER2_AUTHORITATIVE_DATASET_ROOT` to the root of the original locked dataset when running the data-only audit. `PAPER2_INPUT_DIR` can override the `inputs/` location. `PAPER2_OUTPUT_DIR` can override the data-only output location. No private path is built into the release code.

The input contract is described in [DATA_CONTRACT.md](DATA_CONTRACT.md). The package's `src/paper2_final/configs/dataset.json` records only the baseline identifier, split size, excluded date, and lock digest; it contains no profiles or sensor measurements. The existing verification scripts compare the private input files with that lock before training.

## Four-band BASGT and depth comparison

The following commands assume the environment variables above are set. Run each command from the repository root:

```bash
python -m paper2_final.experiments.verify
python experiments/v3_graph_five_band_7m_center_20260919/segments.py --stage selection
python experiments/v3_graph_five_band_7m_center_20260919/segments.py --stage final
python experiments/v3_graph_four_band_ablation_20260919/run.py --variant full4 --seed 20260901 --device cuda:0
python experiments/v3_graph_depth_ablation_20260919/run.py --variant graph_k2 --seed 20260901 --device cuda:0
python experiments/v3_graph_depth_ablation_20260919/run.py --variant graph_k3 --seed 20260901 --device cuda:0
```

Repeat the three training commands for seeds `20260902`–`20260905`, then run `python experiments/v3_graph_depth_ablation_20260919/evaluate.py`. K=1 is the `full4` result; K=2 and K=3 use the same prepared inputs and segment boundaries. The `no_graph`, `no_seg`, and `no_local_direct` K=1 controls are available through the four-band `run.py` entry point. Their evaluation script reads all four controls plus `full4`; they are **not** K=3 ablations.

## Manuscript component ablations: K=3

Use [the K=3 component entry points](experiments/v3_graph_k3_component_ablation_20260922/README.md) for the manuscript component-ablation table. The full model and `k3_no_graph`, `k3_no_seg`, and `k3_no_local_direct` all retain three graph layers and 4,230,604 registered trainable parameters. This was checked against all 20 archived checkpoints and their run records; the table does not compare a three-layer full model with one-layer controls. Equal registered parameter counts do not imply identical effective capacity after disabling a route.

These source files were added to the public release on 29 October 2026 from the completed 22 September experiment. Historical K=1 controls remain available with their original labels; they have not been renamed or substituted for K=3 results.

## Small-sample full-budget endpoint

The manuscript small-sample plot reuses the frozen main-table WRMP–BASGT 16-date result: **33.762450 ± 0.056735 mm** full-profile RMSE over five seeds. The 12/8/4-date runs and baseline runs retain their original results. The [report exporter](experiments/small_sample_reporting_20261029/README.md) records each point's experiment source and leaves all original training outputs untouched.

The independent n16 small-sample run remains **33.779141 ± 0.056331 mm** in the archive. It rebuilt prepared inputs and retrained the model; its selected epochs differ from the frozen main run. The main-table endpoint is therefore an explicit reused reference, not a rewritten result of that independent run. See the linked reporting protocol and audited reference JSON for the comparison conditions. The original private small-sample preparation/training pipeline is not included in this release.

## Five-band data-only comparison

```bash
python experiments/pure_blackbox_full_spectrum_20260921/audit.py
python experiments/pure_blackbox_full_spectrum_20260921/prepare_segments.py
python experiments/pure_blackbox_full_spectrum_20260921/worker.py --worker 0 --device cuda:0
python experiments/pure_blackbox_full_spectrum_20260921/worker.py --worker 1 --device cuda:0
python experiments/pure_blackbox_full_spectrum_20260921/aggregate.py
```

The two workers partition the fixed model/seed matrix and can be run sequentially on one GPU, as shown, or concurrently on separate GPUs. The audit creates the output directory and verifies the frozen lock, split, feature routing, and absence of white-box channels. This comparison predicts all five absolute bands directly; it has no WRMP bypass and no post-prediction smoothing.

The scripts retain the original fixed experiment settings. Do not interpret the development-set metrics as external generalization estimates.
