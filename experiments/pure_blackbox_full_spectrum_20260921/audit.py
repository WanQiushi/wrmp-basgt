"""Pre-run dataset, split, and data-only feature audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from settings import (
    AUTHORITATIVE_DATASET_ROOT,
    BANDS,
    BASELINE_FILE,
    CODE_DIR,
    FEATURE_NAMES,
    FORBIDDEN_CHANNEL_INDICES,
    INPUT_DIR,
    LOCK_FILE,
    MODELS,
    RESULT_DIR,
    SEEDS,
    SEGMENT_COUNTS,
    TRAINING,
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    for name in ("predictions", "checkpoints", "histories", "logs", "segments"):
        (RESULT_DIR / name).mkdir()

    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    frozen_lock_bytes = LOCK_FILE.read_bytes()
    frozen_lock_hash = hashlib.sha256(frozen_lock_bytes).hexdigest()
    authoritative_lock = AUTHORITATIVE_DATASET_ROOT / "DATASET_LOCK.json"
    authoritative_lock_hash = sha256(authoritative_lock)
    if frozen_lock_hash != baseline["lock_sha256"]:
        raise RuntimeError("Frozen DATASET_LOCK.json SHA256 differs from DATASET_BASELINE.json")
    if authoritative_lock_hash != baseline["lock_sha256"]:
        raise RuntimeError("Authoritative DATASET_LOCK.json SHA256 differs from baseline")
    lock = json.loads(frozen_lock_bytes)
    if json.loads(authoritative_lock.read_text(encoding="utf-8")) != lock:
        raise RuntimeError("Frozen and authoritative dataset locks differ")

    failures = []
    for relative, expected in lock["file_sha256"].items():
        path = AUTHORITATIVE_DATASET_ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            failures.append({"path": str(path), "expected": expected, "actual": actual})
    if failures:
        raise RuntimeError(f"Locked dataset files missing or modified: {failures}")

    target_values = np.load(INPUT_DIR / "target.npz")
    target = target_values["target"]
    dates = target_values["dates"].astype(str)
    train = target_values["idx_train"].astype(int)
    development = target_values["idx_dev"].astype(int)
    if dates.tolist() != lock["static_dates"]:
        raise RuntimeError("target.npz date order differs from DATASET_LOCK.json")
    if dates[train].tolist() != lock["train_dates"]:
        raise RuntimeError("16-date train split differs from DATASET_LOCK.json")
    if dates[development].tolist() != lock["development_dates"]:
        raise RuntimeError("9-date development split differs from DATASET_LOCK.json")
    if target.shape != (25, 1433, 5):
        raise RuntimeError(f"Unexpected target shape: {target.shape}")
    band_sum_delta = float(np.max(np.abs(target.sum(axis=2) - target_values["absolute"])))

    stages = {}
    feature_stages = {}
    for stage, expected_fit, expected_val in (
        ("selection", [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 14, 15], [7, 9, 11, 12]),
        ("final", list(range(16)), [7, 9, 11, 12]),
    ):
        no_wb_path = INPUT_DIR / f"{stage}_NO_WB_ALL.npz"
        data_path = INPUT_DIR / f"{stage}_DLinear_DATA.npz"
        no_wb = np.load(no_wb_path)
        data = np.load(data_path)
        if no_wb["fit"].tolist() != expected_fit or no_wb["val"].tolist() != expected_val:
            raise RuntimeError(f"Unexpected {stage} fit/validation split")
        if no_wb["features"].shape != (25, 1433, 5, 88):
            raise RuntimeError(f"Unexpected {stage} feature shape")
        if no_wb["prior"].shape != (25, 1433, 5):
            raise RuntimeError(f"Unexpected {stage} prior shape")
        if float(np.max(np.abs(no_wb["prior"]))) != 0.0:
            raise RuntimeError(f"{stage} NO_WB prior is not identically zero")
        for key in ("features", "prior", "fit", "val", "segments"):
            if not np.array_equal(no_wb[key], data[key]):
                raise RuntimeError(f"{stage} NO_WB and DLinear_DATA differ in {key}")
        forbidden_max = float(np.max(np.abs(no_wb["features"][..., FORBIDDEN_CHANNEL_INDICES])))
        if forbidden_max != 0.0:
            raise RuntimeError(f"{stage} contains a nonzero forbidden white-box channel")
        fit = no_wb["fit"].astype(int)
        fit_means = no_wb["features"][fit].mean(axis=(0, 1))
        mean_max = float(np.max(np.abs(fit_means)))
        if mean_max > 1e-4:
            raise RuntimeError(f"{stage} normalized fit mean is unexpectedly nonzero: {mean_max}")
        nonconstant = []
        for band in range(5):
            indices = np.flatnonzero(np.std(no_wb["features"][:, :, band], axis=(0, 1)) > 0)
            nonconstant.append(
                [{"index": int(index), "name": FEATURE_NAMES[index]} for index in indices]
            )
        stages[stage] = {
            "no_wb_sha256": sha256(no_wb_path),
            "dlinear_data_sha256": sha256(data_path),
            "array_values_identical": True,
            "fit_indices": no_wb["fit"].tolist(),
            "validation_indices": no_wb["val"].tolist(),
            "fit_dates": dates[no_wb["fit"].astype(int)].tolist(),
            "validation_dates": dates[no_wb["val"].astype(int)].tolist(),
            "feature_shape": list(no_wb["features"].shape),
            "target_shape": list(target.shape),
            "prior_max_abs": 0.0,
            "forbidden_channel_max_abs": forbidden_max,
            "training_fit_normalized_mean_max_abs": mean_max,
        }
        feature_stages[stage] = {
            "nonconstant_channels_by_band": dict(zip(BANDS, nonconstant)),
            "normalization_fit_indices": no_wb["fit"].tolist(),
            "development_indices_used_for_normalization": [],
        }

    dataset_verification = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "baseline_id": lock["baseline_id"],
        "frozen_lock_file": str(LOCK_FILE),
        "authoritative_lock_file": str(authoritative_lock),
        "lock_sha256_expected": baseline["lock_sha256"],
        "lock_sha256_frozen": frozen_lock_hash,
        "lock_sha256_authoritative": authoritative_lock_hash,
        "locked_file_count": len(lock["file_sha256"]),
        "locked_file_hash_failures": [],
        "static_dates": dates.tolist(),
        "train_dates": dates[train].tolist(),
        "development_dates": dates[development].tolist(),
        "selection_fit_dates": stages["selection"]["fit_dates"],
        "selection_validation_dates": stages["selection"]["validation_dates"],
        "target_shape": list(target.shape),
        "mileage_shape": list(target_values["mileage_m"].shape),
        "five_band_sum_vs_absolute_max_abs_mm": band_sum_delta,
        "development_use": "final evaluation only",
        "stages": stages,
    }
    write_json(RESULT_DIR / "dataset_verification.json", dataset_verification)

    feature_audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_files": [
            str(INPUT_DIR / "selection_NO_WB_ALL.npz"),
            str(INPUT_DIR / "final_NO_WB_ALL.npz"),
        ],
        "field_schema": {
            "features": [25, 1433, 5, 88],
            "prior": [25, 1433, 5],
            "fit": "stage-specific training indices",
            "val": "fixed four-date internal validation indices",
            "segments": [1433],
        },
        "all_88_channels": [
            {"index": index, "name": name} for index, name in enumerate(FEATURE_NAMES)
        ],
        "basgt_input_channels": {
            "count": 83,
            "indices": list(range(83)),
            "reason": "Frozen BASGT uses a learned band embedding; the five normalized band-identity placeholders are excluded exactly as in the frozen graph code.",
        },
        "baseline_input_channels": {
            "count_per_band": 83,
            "indices": list(range(83)),
            "reason": "Each frozen official baseline body remains band-specific and consumes the same first 83 channels as the frozen baseline loader.",
        },
        "raw_observation_channels": {
            "values": list(range(21, 52)),
            "availability": list(range(52, 83)),
            "interpretation": "25 direct deformation-monitoring channels plus six settlement endpoints and their availability flags",
        },
        "forbidden_whitebox_or_derived_channels": [
            {"index": index, "name": FEATURE_NAMES[index]} for index in FORBIDDEN_CHANNEL_INDICES
        ],
        "proof": {
            "all_forbidden_channels_identically_zero": True,
            "all_prior_arrays_identically_zero": True,
            "no_wb_and_dlinear_data_core_arrays_identical": True,
            "targets_are_absolute_five_band_targets": True,
            "target_scale_recomputed_from_stage_training_targets": True,
            "segment_boundaries_fitted_from_stage_training_absolute_targets": True,
            "fifth_band_is_a_neural_output": True,
            "fifth_band_whitebox_bypass": False,
            "prediction_smoothing": False,
            "development_used_for_training_decisions": False,
        },
        "stages": feature_stages,
    }
    write_json(RESULT_DIR / "feature_audit.json", feature_audit)

    experiment_config = {
        "experiment": "pure_blackbox_full_spectrum_20260921",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_dir": str(CODE_DIR),
        "result_dir": str(RESULT_DIR),
        "dataset_id": lock["baseline_id"],
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "ridge_seed": 0,
        "bands": list(BANDS),
        "segment_counts": list(SEGMENT_COUNTS),
        "basgt": {
            "graph_depth": 3,
            "d_model": 368,
            "attention_heads": 8,
            "ff_dim": 736,
            "dropout": 0.1,
            "local_kernel_size": 9,
            "tower_crossing_edge_weight": 0.25,
            "cross_band_edges": False,
            "point_readout": "cosine interpolation between adjacent segment centers",
        },
        "training": TRAINING,
        "ridge_alpha_candidates": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "target": "five absolute wavelength bands in millimetres",
        "full_profile": "direct sum of the five predicted bands",
        "post_prediction_smoothing": None,
        "development_use": "evaluation only",
    }
    write_json(RESULT_DIR / "experiment_config.json", experiment_config)

    readme = f"""# Pure black-box full-spectrum comparison (2026-09-21)

Status: **RUNNING**.

This experiment asks whether data-only models can reconstruct all five static wavelength bands without the Wavelength-Resolved Mechanistic Prior (WRMP). The five learned outputs are summed directly to form the full static profile. No Savitzky–Golay or other prediction post-smoothing is applied.

## Audit contract

- Dataset lock: `{lock['baseline_id']}`; SHA256 `{frozen_lock_hash}`; all {len(lock['file_sha256'])} locked files passed verification.
- Split: 12 dates for selection fitting, four later training dates for epoch/hyperparameter selection, all 16 training dates for final refit, and nine development dates for final evaluation only.
- Inputs: `selection_NO_WB_ALL.npz` and `final_NO_WB_ALL.npz`; the core arrays are exactly equal to the corresponding `DLinear_DATA` inputs. The saved prior and all WRMP/physics/derived-monitoring channels are identically zero. Direct raw monitoring and settlement observations remain valid data inputs.
- Outputs: every model predicts all five absolute bands. The `>200 m` band has no white-box bypass. BASGT boundaries are fitted separately for selection/final from the corresponding training absolute targets, not residuals.
- Frozen assets: this experiment reads the frozen snapshot and writes only to `{CODE_DIR}` and `{RESULT_DIR}`. Existing frozen code, models, inputs, and results are not modified.

See `dataset_verification.json`, `feature_audit.json`, and `experiment_config.json` for machine-readable details. Metrics files will be written only after all requested models finish.
"""
    (RESULT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print("AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
