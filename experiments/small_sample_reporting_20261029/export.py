"""Build a reporting view with the frozen main-table full-budget endpoint.

This does not modify training outputs or relabel the independent 16-date run.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[2]
REFERENCE = REPOSITORY / "reference_results/k3_component_and_full_budget.json"
MODELS = (
    "WRMP-BASGT-K3",
    "LSTM-DATA",
    "DLinear-DATA",
    "PatchTST-DATA",
    "iTransformer-DATA",
    "TimeXer-DATA",
)
KEYS = ["n_train", "model", "policy", "seed"]


def export(small_sample_metrics, output):
    original = pd.read_csv(small_sample_metrics)
    original = original[
        original.model.isin(MODELS) & (original.policy == "native")
    ][KEYS + ["full_profile_rmse"]].copy()
    endpoint = (original.n_train == 16) & (original.model == "WRMP-BASGT-K3")
    independent = original.loc[endpoint].copy()
    original = original.loc[~endpoint].copy()
    original["source_experiment"] = "small_sample_learning_curve_20260928"

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    canonical = pd.DataFrame(
        [row for row in reference["metrics_by_seed"] if row["model"] == "k3_full"]
    )[KEYS + ["full_profile_rmse"]]
    canonical["model"] = "WRMP-BASGT-K3"
    canonical["source_experiment"] = "v3_graph_depth_ablation_20260919/graph_k3"
    report = pd.concat([original, canonical], ignore_index=True)
    report = report.sort_values(["model", "n_train", "seed"])
    summary = report.groupby(["n_train", "model", "policy"], sort=False).agg(
        seed_count=("seed", "size"),
        full_profile_rmse_mean=("full_profile_rmse", "mean"),
        full_profile_rmse_std=("full_profile_rmse", "std"),
        source_experiment=("source_experiment", "first"),
    ).reset_index()

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / "metrics_by_seed.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    independent.to_csv(output / "independent_n16_reference.csv", index=False)
    manifest = {
        "input_metrics": str(small_sample_metrics),
        "canonical_reference": "reference_results/k3_component_and_full_budget.json",
        "endpoint_policy": reference["small_sample_reporting"]["policy"],
        "independent_n16_run_preserved": True,
        "interpretation": (
            "The 16-date proposed-model point is a reused main-table reference, "
            "not the independently prepared n16 training result."
        ),
        "metric": "Mean per-seed full-profile RMSE and sample standard deviation (mm).",
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-sample-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.small_sample_metrics, args.output)
