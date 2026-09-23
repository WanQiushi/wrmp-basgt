"""Fit all five BASGT segmentations from training absolute targets only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from settings import INPUT_DIR, RESULT_DIR, SEGMENT_COUNTS

ROOT = Path(__file__).resolve().parent
FROZEN_SEGMENTS = ROOT.parent / "v3_graph_five_band_7m_center_20260919" / "segments.py"
spec = importlib.util.spec_from_file_location("frozen_segments", FROZEN_SEGMENTS)
frozen_segments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen_segments)


def prepare(stage: str) -> None:
    output = RESULT_DIR / "segments"
    output.mkdir(parents=True, exist_ok=True)
    target = np.load(INPUT_DIR / "target.npz")["target"]
    prepared = np.load(INPUT_DIR / f"{stage}_NO_WB_ALL.npz")
    fit = prepared["fit"].astype(int)
    rows = []
    for band, count in enumerate(SEGMENT_COUNTS):
        bounds, density = frozen_segments.fit_boundaries(target[fit, :, band], band, count)
        bounds_m = bounds + frozen_segments.MILEAGE_START
        path = output / f"{stage}_b{band}_k{count}_segments.npz"
        np.savez_compressed(
            path,
            bounds_idx=bounds,
            bounds_m=bounds_m,
            peak_density=density,
            fit_rows=fit,
            target_kind=np.asarray("absolute_band_target"),
        )
        lengths = np.diff(bounds)
        rows.append(
            {
                "stage": stage,
                "band": band,
                "count": count,
                "fit_rows": fit.tolist(),
                "target_kind": "absolute_band_target",
                "min_length_m": int(lengths.min()),
                "median_length_m": float(np.median(lengths)),
                "max_length_m": int(lengths.max()),
                "boundaries_m": bounds_m.tolist(),
                "file": str(path),
            }
        )
    (output / f"{stage}_segment_geometry.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    prepare("selection")
    prepare("final")
