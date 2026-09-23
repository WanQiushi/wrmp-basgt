from __future__ import annotations

import json

import numpy as np

from paper2_final.config import input_name, load_config
from paper2_final.io import sha256, write_json
from paper2_final.paths import DATA_ROOT, INPUT_DIR

TRAINED_VARIANTS = [
    "FULL",
    "NO_WB_ALL",
    "NO_DYN_ALL",
    "NO_GNSS_ALL",
    "NO_DEFLECTION_ALL",
    "NO_SETTLE_ALL",
    "NO_TEMP_ALL",
]


def main() -> None:
    expected = load_config("dataset")
    lock_path = DATA_ROOT / "DATASET_LOCK.json"
    if sha256(lock_path) != expected["lock_sha256"]:
        raise RuntimeError("DATASET_LOCK.json hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["baseline_id"] != expected["baseline_id"]:
        raise RuntimeError("Unexpected dataset baseline")

    target = np.load(INPUT_DIR / "target.npz")
    dates = target["dates"].astype(str)
    if len(dates) != expected["sample_count"]:
        raise RuntimeError("Unexpected sample count")
    if len(target["idx_train"]) != expected["train_count"]:
        raise RuntimeError("Unexpected training count")
    if len(target["idx_dev"]) != expected["development_count"]:
        raise RuntimeError("Unexpected development count")
    if any(date in dates for date in expected["excluded_dates"]):
        raise RuntimeError("Excluded date found in target.npz")
    if dates.tolist() != lock["static_dates"]:
        raise RuntimeError("Date order differs from DATASET_LOCK.json")
    if target["mileage_m"].shape != (1433,) or not np.allclose(np.diff(target["mileage_m"]), 1.0):
        raise RuntimeError("Unexpected mileage grid")

    required = [
        *(
            f"{stage}_DLinear_{mode}.npz"
            for stage in ["selection", "final"]
            for mode in ["DATA", "HYBRID"]
        ),
        *(
            input_name(stage, variant)
            for stage in ["selection", "final"]
            for variant in TRAINED_VARIANTS
        ),
    ]
    rows = []
    for name in sorted(set(required)):
        path = INPUT_DIR / name
        values = np.load(path)
        if values["features"].shape[:3] != (25, 1433, 5):
            raise RuntimeError(f"Unexpected feature shape: {name}")
        if values["prior"].shape != (25, 1433, 5):
            raise RuntimeError(f"Unexpected prior shape: {name}")
        if not np.isfinite(values["features"]).all() or not np.isfinite(values["prior"]).all():
            raise RuntimeError(f"Nonfinite input: {name}")
        rows.append(
            {
                "path": f"data/inputs/{name}",
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    target_path = INPUT_DIR / "target.npz"
    rows.append(
        {
            "path": "data/inputs/target.npz",
            "sha256": sha256(target_path),
            "bytes": target_path.stat().st_size,
        }
    )
    write_json(DATA_ROOT / "manifest.json", {"dataset": expected, "files": rows})
    print(f"Verified {len(rows)} locked input files")


if __name__ == "__main__":
    main()
