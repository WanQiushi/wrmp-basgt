"""Fit fixed-count, adaptive boundaries for all five bands from training residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper2_final.paths import INPUT_DIR

ROOT = Path(__file__).resolve().parent
INPUT = INPUT_DIR
BANDS = (0, 1, 2, 3, 4)
COUNTS = (204, 102, 51, 34, 17)
MIN_LENGTH = {band: 7 for band in BANDS}
SMOOTH_SIGMA = {0: 3, 1: 5, 2: 7, 3: 7, 4: 14}
PROMINENCE_WINDOW = {0: 20, 1: 35, 2: 60, 3: 60, 4: 100}
PEAK_DISTANCE = {0: 10, 1: 15, 2: 25, 3: 35, 4: 50}
MILEAGE_START = 318206


def extrema_density(curves: np.ndarray, band: int) -> np.ndarray:
    sigma = SMOOTH_SIGMA[band]
    radius = round(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    radius_weight = PROMINENCE_WINDOW[band]
    wx = np.arange(-radius_weight, radius_weight + 1)
    weight_kernel = np.exp(-0.5 * (wx / (radius_weight / 3)) ** 2)
    density = np.zeros(curves.shape[1], dtype=float)
    for curve in curves:
        scaled = curve / max(float(np.sqrt(np.mean(curve**2))), 1e-6)
        smoothed = np.convolve(np.pad(scaled, radius, mode="edge"), kernel, mode="valid")
        for direction in (1, -1):
            wave = smoothed * direction
            potential = np.flatnonzero((wave[1:-1] > wave[:-2]) & (wave[1:-1] >= wave[2:])) + 1
            ranked = []
            for point in potential:
                left = wave[max(0, point - radius_weight) : point]
                right = wave[point + 1 : min(len(wave), point + radius_weight + 1)]
                if not len(left) or not len(right):
                    continue
                prominence = min(wave[point] - left.min(), wave[point] - right.min())
                if prominence >= 0.35:
                    ranked.append((float(prominence), int(point)))
            kept = []
            for prominence, point in sorted(ranked, reverse=True):
                if all(abs(point - other) >= PEAK_DISTANCE[band] for other in kept):
                    kept.append(point)
                    lo = max(0, point - radius_weight)
                    hi = min(len(density), point + radius_weight + 1)
                    density[lo:hi] += (
                        prominence
                        * weight_kernel[lo - point + radius_weight : hi - point + radius_weight]
                    )
    return density


def optimal_region(
    curves: np.ndarray, weights: np.ndarray, segments: int, min_length: int
) -> list[int]:
    dates, points = curves.shape
    if points == segments * min_length:
        return list(range(0, points + 1, min_length))
    weighted = curves * weights
    sums = np.concatenate((np.zeros((dates, 1)), np.cumsum(weighted, axis=1)), axis=1)
    squares = np.r_[0.0, np.cumsum(np.sum(curves**2 * weights, axis=0))]
    weight_sums = np.r_[0.0, np.cumsum(weights)]
    costs = np.full((points + 1, points + 1), np.inf)
    for stop in range(min_length, points + 1):
        starts = np.arange(stop - min_length + 1)
        totals = sums[:, stop, None] - sums[:, starts]
        costs[starts, stop] = (
            squares[stop]
            - squares[starts]
            - np.sum(totals**2, axis=0) / (weight_sums[stop] - weight_sums[starts])
        )
    best = np.full((segments + 1, points + 1), np.inf)
    previous = np.full((segments + 1, points + 1), -1, dtype=int)
    best[0, 0] = 0.0
    for count in range(1, segments + 1):
        for stop in range(count * min_length, points + 1):
            starts = np.arange((count - 1) * min_length, stop - min_length + 1)
            candidates = best[count - 1, starts] + costs[starts, stop]
            chosen = int(np.argmin(candidates))
            best[count, stop] = candidates[chosen]
            previous[count, stop] = starts[chosen]
    bounds = [points]
    stop = points
    for count in range(segments, 0, -1):
        stop = int(previous[count, stop])
        bounds.append(stop)
    return bounds[::-1]


def fit_boundaries(residual: np.ndarray, band: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    scaled = residual / np.maximum(np.sqrt(np.mean(residual**2, axis=1, keepdims=True)), 1e-6)
    density = extrema_density(residual, band)
    weights = 1.0 + density / max(float(density.mean()), 1e-6)
    side = max(1, round(count * 2 / 17))
    regions = ((0, 170, side), (170, 1262, count - 2 * side), (1262, 1433, side))
    bounds = [0]
    for start, stop, region_count in regions:
        local = optimal_region(
            scaled[:, start:stop], weights[start:stop], region_count, MIN_LENGTH[band]
        )
        bounds.extend(start + value for value in local[1:])
    return np.asarray(bounds, dtype=int), density


def prepare(stage: str) -> None:
    target = np.load(INPUT / "target.npz")["target"]
    values = np.load(INPUT / f"{stage}_FULL.npz")
    fit = values["fit"].astype(int)
    residual = target - values["prior"]
    rows = []
    for band in BANDS:
        for count in (COUNTS[band],):
            bounds, density = fit_boundaries(residual[fit, :, band], band, count)
            path = ROOT / f"{stage}_b{band}_k{count}_segments.npz"
            np.savez_compressed(
                path,
                bounds_idx=bounds,
                bounds_m=bounds + MILEAGE_START,
                peak_density=density,
                fit_rows=fit,
            )
            lengths = np.diff(bounds)
            rows.append(
                {
                    "stage": stage,
                    "band": band,
                    "count": count,
                    "min_length_m": int(lengths.min()),
                    "median_length_m": float(np.median(lengths)),
                    "max_length_m": int(lengths.max()),
                    "mean_length_m": float(lengths.mean()),
                    "lengths_m": lengths.tolist(),
                    "boundaries_m": (bounds + MILEAGE_START).tolist(),
                    "file": str(path),
                }
            )
    (ROOT / f"{stage}_segment_geometry.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(
        json.dumps(
            [
                {k: v for k, v in row.items() if k not in ("lengths_m", "boundaries_m")}
                for row in rows
            ],
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("selection", "final"), required=True)
    args = parser.parse_args()
    prepare(args.stage)
