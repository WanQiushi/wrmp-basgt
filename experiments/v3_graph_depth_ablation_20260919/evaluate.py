"""Evaluate K=1,2,3 graph depths with the same nine-date publication metrics."""

from pathlib import Path

import numpy as np
import pandas as pd

from paper2_final.metrics import calculate_metrics
from paper2_final.paths import INPUT_DIR

ROOT = Path(__file__).resolve().parent
INPUT = INPUT_DIR
K1 = ROOT.parent / "v3_graph_four_band_ablation_20260919" / "factorial_runs" / "full4"
SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
MODELS = ("graph_k1", "graph_k2", "graph_k3")
BANDS = ("lt42", "w42_70", "w70_120", "w120_200", "gt200")


def measures(prediction, target, indices):
    output = calculate_metrics(prediction, target, indices)
    error = prediction[indices] - target[indices]
    for band, name in enumerate(BANDS):
        output[f"{name}_mae"] = float(np.mean(np.abs(error[:, :, band])))
    output["first4_pooled_mae"] = float(np.mean(np.abs(error[:, :, :4])))
    output["reconstructed_profile_mae"] = float(np.mean(np.abs(error.sum(axis=2))))
    return output


def prediction_path(model, seed):
    if model == "graph_k1":
        return K1 / f"seed_{seed}" / "predictions.npz"
    return ROOT / "factorial_runs" / model / f"seed_{seed}" / "predictions.npz"


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
    for seed in SEEDS:
        for model in MODELS:
            values = np.load(prediction_path(model, seed))
            prediction = np.concatenate((values["prediction"], prior[:, :, 4:5]), axis=2)
            for scope, indices in scopes.items():
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "scope": scope,
                        **measures(prediction, target, indices),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "metrics_by_seed.csv", index=False)
    metric_columns = [
        column for column in frame.columns if column not in ("model", "seed", "scope")
    ]
    summary = frame.groupby(["model", "scope"], sort=False)[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(ROOT / "metrics_summary.csv", index=False)
    pooled = frame[frame["scope"] == "all9"].pivot(
        index="seed", columns="model", values="first4_pooled_rmse"
    )
    pooled.subtract(pooled["graph_k1"], axis=0).to_csv(ROOT / "paired_pooled_rmse_change.csv")
    print(summary.loc[("graph_k1", "all9")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
