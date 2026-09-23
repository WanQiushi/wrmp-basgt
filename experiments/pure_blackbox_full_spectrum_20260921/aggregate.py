"""Aggregate completed model predictions into the requested result tables."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from settings import BANDS, INPUT_DIR, MODELS, RESULT_DIR, SEEDS


def measures(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray) -> dict:
    error = prediction[indices] - target[indices]
    row = {}
    for band, name in enumerate(BANDS):
        band_error = error[:, :, band]
        row[f"{name}_rmse"] = float(np.sqrt(np.mean(band_error**2)))
        row[f"{name}_mae"] = float(np.mean(np.abs(band_error)))
    first4 = error[:, :, :4]
    row["first4_pooled_rmse"] = float(np.sqrt(np.mean(first4**2)))
    row["first4_pooled_mae"] = float(np.mean(np.abs(first4)))
    row["five_band_pooled_rmse"] = float(np.sqrt(np.mean(error**2)))
    row["five_band_pooled_mae"] = float(np.mean(np.abs(error)))
    profile_error = error.sum(axis=2)
    row["full_profile_rmse"] = float(np.sqrt(np.mean(profile_error**2)))
    row["full_profile_mae"] = float(np.mean(np.abs(profile_error)))
    return row


def summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    values = [column for column in frame.columns if column not in {*groups, "seed"}]
    rows = []
    for keys, group in frame.groupby(groups, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["seed_count"] = int(group["seed"].nunique())
        for column in values:
            row[f"{column}_mean"] = float(group[column].mean())
            deviation = group[column].std(ddof=1)
            row[f"{column}_std"] = 0.0 if len(group) == 1 else float(deviation)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    target_values = np.load(INPUT_DIR / "target.npz")
    target = target_values["target"]
    development = target_values["idx_dev"].astype(int)
    dates = target_values["dates"].astype(str)
    metric_rows = []
    date_rows = []
    parameter_rows = []
    epoch_rows = []
    for model in MODELS:
        model_seeds = (0,) if model == "Ridge-DATA" else SEEDS
        for seed in model_seeds:
            prediction_file = RESULT_DIR / "predictions" / model / str(seed) / "predictions.npz"
            checkpoint_file = RESULT_DIR / "checkpoints" / model / str(seed) / "checkpoint.pt"
            completion_file = RESULT_DIR / "checkpoints" / model / str(seed) / "COMPLETE.json"
            if (
                not prediction_file.is_file()
                or not checkpoint_file.is_file()
                or not completion_file.is_file()
            ):
                raise RuntimeError(f"Missing completed result for {model} seed={seed}")
            values = np.load(prediction_file)
            prediction = values["predicted_bands"]
            if not np.array_equal(values["true_bands"], target):
                raise RuntimeError(f"Target mismatch for {model} seed={seed}")
            completion = json.loads(completion_file.read_text(encoding="utf-8"))
            metric_rows.append(
                {"model": model, "seed": seed, **measures(prediction, target, development)}
            )
            for index in development:
                error = prediction[index].sum(axis=1) - target[index].sum(axis=1)
                date_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "date": dates[index],
                        "full_profile_rmse": float(np.sqrt(np.mean(error**2))),
                        "full_profile_mae": float(np.mean(np.abs(error))),
                    }
                )
            parameter_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "parameter_count": int(completion["parameter_count"]),
                    "definition": (
                        "dual coefficients plus fitted target intercepts"
                        if model == "Ridge-DATA"
                        else "trainable torch parameters across five learned outputs"
                    ),
                }
            )
            if model == "Ridge-DATA":
                for band, choice in zip(BANDS, completion["selected"]):
                    epoch_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "band": band,
                            "selected_epoch": 0,
                            "selected_learning_rate": np.nan,
                            "selected_alpha": choice["alpha"],
                            "selection_validation_loss": choice["validation_loss"],
                        }
                    )
            else:
                choice = completion["selected"]
                epoch_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "band": "all_five_joint",
                        "selected_epoch": choice["epochs"],
                        "selected_learning_rate": choice["learning_rate"],
                        "selected_alpha": np.nan,
                        "selection_validation_loss": choice["validation_loss"],
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    dates_frame = pd.DataFrame(date_rows)
    metrics.to_csv(RESULT_DIR / "metrics_by_seed.csv", index=False)
    summary(metrics, ["model"]).to_csv(RESULT_DIR / "metrics_summary.csv", index=False)
    dates_frame.to_csv(RESULT_DIR / "date_metrics_by_seed.csv", index=False)
    summary(dates_frame, ["model", "date"]).to_csv(
        RESULT_DIR / "date_metrics_summary.csv", index=False
    )
    pd.DataFrame(parameter_rows).to_csv(RESULT_DIR / "parameter_counts.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(RESULT_DIR / "selected_epochs.csv", index=False)

    table = summary(metrics, ["model"])[
        [
            "model",
            "full_profile_rmse_mean",
            "full_profile_rmse_std",
            "full_profile_mae_mean",
            "full_profile_mae_std",
            "gt200_rmse_mean",
            "gt200_rmse_std",
        ]
    ]
    table_lines = [
        "| Model | Full profile RMSE | RMSE SD | Full profile MAE | MAE SD | >200 m RMSE | >200 m SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        table_lines.append(
            f"| {row.model} | {row.full_profile_rmse_mean:.6f} | "
            f"{row.full_profile_rmse_std:.6f} | {row.full_profile_mae_mean:.6f} | "
            f"{row.full_profile_mae_std:.6f} | {row.gt200_rmse_mean:.6f} | "
            f"{row.gt200_rmse_std:.6f} |"
        )
    lines = [
        "# Pure black-box full-spectrum comparison (2026-09-21)",
        "",
        "Status: **COMPLETE**.",
        "",
        "All six requested model families completed. Every output is a direct prediction of all five absolute wavelength bands; the full profile is their direct sum. No WRMP input, residual target, long-wave white-box bypass, or prediction post-smoothing was used.",
        "",
        "## Primary development metrics (mm)",
        "",
        *table_lines,
        "",
        "The table reports the completed numerical results without predeclaring whether removing WRMP should improve or reduce accuracy. Detailed band, seed, and date results are in the CSV files.",
        "",
        "## Audit",
        "",
        "- `dataset_verification.json`: locked dates, SHA256 verification, 12/4 and 16/9 split.",
        "- `feature_audit.json`: actual input fields/channels and proof of zero WRMP/derived channels.",
        "- `segments/`: 204/102/51/34/17 boundaries fitted from selection/final training absolute targets.",
        "- `predictions/`, `checkpoints/`, and `histories/`: per-model, per-seed artifacts.",
        "- Existing frozen code, models, inputs, and results were read-only and not overwritten.",
        "",
        f"Completed UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    (RESULT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("AGGREGATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
