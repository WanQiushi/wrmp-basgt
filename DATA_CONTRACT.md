# Private prepared-input contract

The publication package contains code and configuration metadata only. Put the private arrays outside the repository and point `PAPER2_DATA_ROOT` to their directory. The locked baseline is `PAPER2_V4_STATIC25_20260906`: exactly 25 ordered static-survey dates, 16 fitting and nine development dates, with 2024-08-28 excluded.

`inputs/target.npz` supplies `target` with shape `(25, 1433, 5)` in millimetres, `absolute`, `dates`, `idx_train`, `idx_dev`, and `mileage_m`. The five bands are `<42`, `42–70`, `70–120`, `120–200`, and `>200` m on a 1 m longitudinal grid.

`inputs/selection_FULL.npz` and `inputs/final_FULL.npz` supply `features`, `prior`, `fit`, `val`, and `segments`. The model reads the first four bands and the first 83 feature channels for its learned residual. The fifth-band output is copied from `prior`. Selection fits 12 dates and chooses epochs with four held-out fitting dates; final training fits all 16 fitting dates.

The data-only comparison reads `selection_NO_WB_ALL.npz`, `final_NO_WB_ALL.npz`, and the corresponding `selection_DLinear_DATA.npz` and `final_DLinear_DATA.npz` files. Its audit expects `features` of shape `(25, 1433, 5, 88)`, a zero `prior`, and the first 83 channels as model input. It fits its own segment boundaries from the absolute training targets. The existing audits use the private `DATASET_LOCK.json` and `DATASET_BASELINE.json`; neither file nor any `.npz` data is distributed here.

The prepared inputs and WRMP construction are required to reproduce numerical results. This code release does not claim raw-data-to-paper reproducibility without them.

The shared `paper2_final.experiments.verify` command checks the complete locked prepared-input bundle, including historical source-removal variants beyond the files named above. Its full verification therefore requires that original private bundle. The data-only `audit.py` checks the narrower full-spectrum comparison inputs and the authoritative locked dataset.
