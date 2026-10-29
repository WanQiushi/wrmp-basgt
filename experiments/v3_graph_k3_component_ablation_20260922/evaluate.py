"""Evaluate frozen K=3 full BASGT and the new K=3 component ablations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper2_final.metrics import calculate_metrics
from paper2_final.paths import INPUT_DIR


ROOT = Path(__file__).resolve().parent
DEPTH_ROOT = ROOT.parent / "v3_graph_depth_ablation_20260919"
INPUT = INPUT_DIR
SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
VARIANTS = ("k3_full", "k3_no_graph", "k3_no_seg", "k3_no_local_direct")
BANDS = ("lt42", "w42_70", "w70_120", "w120_200", "gt200")


def measures(prediction, target, indices):
    output = calculate_metrics(prediction, target, indices)
    error = prediction[indices] - target[indices]
    for band, name in enumerate(BANDS):
        output[f"{name}_mae"] = float(np.mean(np.abs(error[:, :, band])))
    output["first4_pooled_mae"] = float(np.mean(np.abs(error[:, :, :4])))
    output["reconstructed_profile_mae"] = float(np.mean(np.abs(error.sum(axis=2))))
    return output


def prediction_path(variant, seed):
    if variant == "k3_full":
        return (
            DEPTH_ROOT
            / "factorial_runs"
            / "graph_k3"
            / f"seed_{seed}"
            / "predictions.npz"
        )
    return ROOT / "factorial_runs" / variant / f"seed_{seed}" / "predictions.npz"


def complete_path(variant, seed):
    if variant == "k3_full":
        return (
            DEPTH_ROOT
            / "factorial_runs"
            / "graph_k3"
            / f"seed_{seed}"
            / "COMPLETE.json"
        )
    return ROOT / "factorial_runs" / variant / f"seed_{seed}" / "COMPLETE.json"


def main():
    data = np.load(INPUT / "target.npz")
    target, dates = data["target"], data["dates"].astype(str)
    dev = data["idx_dev"].astype(int)
    scopes = {
        "all9": dev,
        "observed2023": dev[np.char.startswith(dates[dev], "2023-")],
        "forecast2024": dev[np.char.startswith(dates[dev], "2024-")],
    }
    prior = np.load(INPUT / "final_FULL.npz")["prior"]
    rows = []
    parameter_rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            values = np.load(prediction_path(variant, seed))
            prediction = np.concatenate(
                (values["prediction"], prior[:, :, 4:5]), axis=2
            )
            for scope, indices in scopes.items():
                rows.append(
                    {
                        "model": variant,
                        "seed": seed,
                        "scope": scope,
                        **measures(prediction, target, indices),
                    }
                )
            complete = json.loads(complete_path(variant, seed).read_text())
            parameter_rows.append(
                {
                    "model": variant,
                    "seed": seed,
                    "trainable_parameters": complete["trainable_parameters"],
                    "selected_epochs": complete["epochs"],
                }
            )
        for scope, indices in scopes.items():
            rows.append(
                {
                    "model": "whitebox_only",
                    "seed": seed,
                    "scope": scope,
                    **measures(prior, target, indices),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "metrics_by_seed.csv", index=False)
    metric_columns = [
        column for column in frame.columns if column not in ("model", "seed", "scope")
    ]
    summary = frame.groupby(["model", "scope"], sort=False)[metric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{key}_{stat}" for key, stat in summary.columns]
    summary.reset_index().to_csv(ROOT / "metrics_summary.csv", index=False)
    all9 = frame[(frame["scope"] == "all9") & frame["model"].isin(VARIANTS)]
    pairs = all9.pivot(index="seed", columns="model", values="first4_pooled_rmse")
    pairs.subtract(pairs["k3_full"], axis=0).to_csv(
        ROOT / "paired_pooled_rmse_change.csv"
    )
    pd.DataFrame(parameter_rows).to_csv(ROOT / "parameter_counts_epochs.csv", index=False)
    print(summary.loc[("k3_full", "all9")], flush=True)


if __name__ == "__main__":
    main()
