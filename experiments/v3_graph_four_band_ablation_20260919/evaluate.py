"""Evaluate all four-band arms and the frozen official baselines."""

from pathlib import Path

import numpy as np
import pandas as pd

from paper2_final.metrics import calculate_metrics
from paper2_final.paths import INPUT_DIR

ROOT = Path(__file__).resolve().parent
INPUT = INPUT_DIR
SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
VARIANTS = ("full4", "no_graph", "no_seg", "no_local_direct")
BANDS = ("lt42", "w42_70", "w70_120", "w120_200", "gt200")


def measures(prediction, target, indices):
    output = calculate_metrics(prediction, target, indices)
    error = prediction[indices] - target[indices]
    for band, name in enumerate(BANDS):
        output[f"{name}_mae"] = float(np.mean(np.abs(error[:, :, band])))
    output["first4_pooled_mae"] = float(np.mean(np.abs(error[:, :, :4])))
    output["reconstructed_profile_mae"] = float(np.mean(np.abs(error.sum(axis=2))))
    return output


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
        for variant in VARIANTS:
            values = np.load(ROOT / "factorial_runs" / variant / f"seed_{seed}" / "predictions.npz")
            prediction = np.concatenate((values["prediction"], prior[:, :, 4:5]), axis=2)
            for scope, indices in scopes.items():
                rows.append(
                    {
                        "model": variant,
                        "seed": seed,
                        "scope": scope,
                        **measures(prediction, target, indices),
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
    metric_columns = [item for item in frame.columns if item not in ("model", "seed", "scope")]
    summary = frame.groupby(["model", "scope"], sort=False)[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{key}_{stat}" for key, stat in summary.columns]
    summary.reset_index().to_csv(ROOT / "metrics_summary.csv", index=False)
    all9 = frame[frame["scope"] == "all9"]
    pairs = all9[all9["model"].isin(VARIANTS)].pivot(
        index="seed", columns="model", values="first4_pooled_rmse"
    )
    (pairs.subtract(pairs["full4"], axis=0).to_csv(ROOT / "paired_pooled_rmse_change.csv"))
    print(summary.loc[("full4", "all9")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
