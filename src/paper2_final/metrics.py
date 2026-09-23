"""Metrics used by the four-band and graph-depth comparisons."""

from __future__ import annotations

import numpy as np

from paper2_final.config import bands


def calculate_metrics(
    prediction: np.ndarray, target: np.ndarray, development: np.ndarray
) -> dict[str, float]:
    error = prediction[development] - target[development]
    band_values = {
        band: float(np.sqrt(np.mean(error[:, :, index] ** 2))) for index, band in enumerate(bands())
    }
    first_four_error = error[:, :, :4]
    reconstructed_error = error.sum(axis=2)
    return {
        **band_values,
        "first4_rmse_sum": float(sum(band_values[band] for band in bands()[:4])),
        "first4_pooled_rmse": float(np.sqrt(np.mean(first_four_error**2))),
        "five_band_pooled_rmse": float(np.sqrt(np.mean(error**2))),
        "reconstructed_profile_rmse": float(np.sqrt(np.mean(reconstructed_error**2))),
    }
