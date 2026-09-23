"""Per-band variable graph nodes with continuous point fill near boundaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from paper2_final.models.graph_transformer import band_layout

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "v3_graph_peak_auto_count_20260918" / "model.py"
spec = importlib.util.spec_from_file_location("previous_auto_model", SOURCE)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
SoftBoundary24GT = previous.SoftBoundary24GT
MILEAGE = np.arange(318206, 319639)
TRANSITION_M = (8.0, 14.0, 20.0, 24.0, 24.0)


def graph_topology(boundaries: list[np.ndarray], device: str):
    base = band_layout((17,) * 5)
    counts = [len(b) - 1 for b in boundaries]
    maximum = max(counts)
    point_ids = np.zeros((5, len(MILEAGE)), dtype=np.int64)
    edge_type = np.zeros((5, maximum, maximum), dtype=np.int64)
    edge_weight = np.zeros((5, maximum, maximum), dtype=np.float32)
    for band, bounds in enumerate(boundaries):
        count = counts[band]
        point_ids[band] = np.searchsorted(bounds[1:-1], MILEAGE, side="right")
        starts, ends = bounds[:-1], bounds[1:]
        zones = np.where(ends <= 318376, 0, np.where(starts >= 319468, 2, 1))
        for node in range(count - 1):
            crossing = zones[node] != zones[node + 1]
            edge_type[band, node, node + 1] = edge_type[band, node + 1, node] = 2 if crossing else 1
            edge_weight[band, node, node + 1] = edge_weight[band, node + 1, node] = (
                0.25 if crossing else 1.0
            )
        np.fill_diagonal(edge_weight[band], 1.0)
    return (
        torch.as_tensor(base["physical_ids"].copy(), dtype=torch.long, device=device),
        torch.as_tensor(point_ids, dtype=torch.long, device=device),
        torch.as_tensor(edge_type, dtype=torch.long, device=device),
        torch.as_tensor(edge_weight, dtype=torch.float32, device=device),
    )


class AdaptiveBandsGT(SoftBoundary24GT):
    def __init__(self, feature_dim: int, boundaries: list[np.ndarray], soft_bands: tuple[int, ...]):
        super().__init__(feature_dim)
        base = band_layout((17,) * 5)
        counts = [len(b) - 1 for b in boundaries]
        maximum = max(counts)
        pooling = np.zeros((5, maximum, len(MILEAGE)), dtype=np.float32)
        physical_pool = np.zeros((5, maximum, 17), dtype=np.float32)
        fill = np.zeros((5, len(MILEAGE), maximum), dtype=np.float32)
        for band, bounds in enumerate(boundaries):
            point_ids = np.searchsorted(bounds[1:-1], MILEAGE, side="right")
            for node in range(counts[band]):
                keep = point_ids == node
                pooling[band, node, keep] = 1.0 / keep.sum()
                physical_pool[band, node] = (
                    np.bincount(base["physical_ids"][keep], minlength=17) / keep.sum()
                )
            fill[band, :, : counts[band]] = np.eye(counts[band], dtype=np.float32)[point_ids]
            if band in soft_bands:
                width = TRANSITION_M[band]
                for node, boundary in enumerate(bounds[1:-1]):
                    nearby = (MILEAGE >= boundary - width / 2) & (MILEAGE <= boundary + width / 2)
                    phase = (MILEAGE[nearby] - (boundary - width / 2)) / width
                    right = 0.5 - 0.5 * np.cos(np.pi * phase)
                    fill[band, nearby] = 0.0
                    fill[band, nearby, node] = 1.0 - right
                    fill[band, nearby, node + 1] = right
        self.band_pooling = torch.from_numpy(pooling)
        self.physical_embedding_pool = torch.from_numpy(physical_pool)
        self.point_fill = torch.from_numpy(fill)
