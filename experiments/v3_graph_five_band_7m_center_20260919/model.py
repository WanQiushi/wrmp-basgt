"""Adaptive graph boundaries with cosine interpolation between neighboring centers."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

SOURCE = (
    Path(__file__).resolve().parent.parent / "v3_graph_all_band_auto_count_20260919" / "model.py"
)
spec = importlib.util.spec_from_file_location("previous_all_band_model", SOURCE)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
graph_topology = previous.graph_topology
AdaptiveBandsGT = previous.AdaptiveBandsGT


class CenterBandsGT(AdaptiveBandsGT):
    def __init__(self, feature_dim: int, boundaries: list[np.ndarray]):
        super().__init__(feature_dim, boundaries, soft_bands=())
        mileage = np.arange(318206, 319639, dtype=float)
        maximum = max(len(bounds) - 1 for bounds in boundaries)
        fill = np.zeros((5, len(mileage), maximum), dtype=np.float32)
        points = np.arange(len(mileage))
        for band, bounds in enumerate(boundaries):
            centers = (bounds[:-1] + bounds[1:] - 1) / 2.0
            insertion = np.searchsorted(centers, mileage, side="right")
            left = np.clip(insertion - 1, 0, len(centers) - 1)
            right = np.clip(insertion, 0, len(centers) - 1)
            phase = np.zeros(len(mileage), dtype=float)
            between = left != right
            phase[between] = (mileage[between] - centers[left[between]]) / (
                centers[right[between]] - centers[left[between]]
            )
            right_weight = 0.5 - 0.5 * np.cos(np.pi * phase)
            fill[band, points, left] += (1.0 - right_weight).astype(np.float32)
            fill[band, points, right] += right_weight.astype(np.float32)
        self.point_fill = torch.from_numpy(fill)
