"""Variable-count peak-aware graph, preserving the 24 m soft point fill."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from paper2_final.models.graph_transformer import band_layout

SOFT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "v3_graph_soft_boundary24_20260918" / "model.py"
)
spec = importlib.util.spec_from_file_location("soft_boundary24_model", SOFT_MODEL_PATH)
soft_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soft_model)
SoftBoundary24GT = soft_model.SoftBoundary24GT


def topology(boundaries_m, device):
    base = band_layout((17,) * 5)
    count = len(boundaries_m) - 1
    maximum = max(17, count)
    point_ids = base["point_ids"].copy()
    point_ids[3] = np.searchsorted(boundaries_m[1:-1], np.arange(318206, 319639), side="right")
    edge_type = np.zeros((5, maximum, maximum), dtype=np.int64)
    edge_weight = np.zeros((5, maximum, maximum), dtype=np.float32)
    for band in range(5):
        nodes = count if band == 3 else 17
        if band == 3:
            starts, ends = boundaries_m[:-1], boundaries_m[1:]
            zones = np.where(ends <= 318376, 0, np.where(starts >= 319468, 2, 1))
            for left in range(nodes - 1):
                cross_tower = zones[left] != zones[left + 1]
                edge_type[band, left, left + 1] = edge_type[band, left + 1, left] = (
                    2 if cross_tower else 1
                )
                edge_weight[band, left, left + 1] = edge_weight[band, left + 1, left] = (
                    0.25 if cross_tower else 1.0
                )
        else:
            for left in range(nodes - 1):
                edge_type[band, left, left + 1] = base["edge_type"][band, left, left + 1]
                edge_type[band, left + 1, left] = base["edge_type"][band, left + 1, left]
                edge_weight[band, left, left + 1] = base["edge_weight"][band, left, left + 1]
                edge_weight[band, left + 1, left] = base["edge_weight"][band, left + 1, left]
        np.fill_diagonal(edge_weight[band], 1.0)
    return (
        torch.as_tensor(base["physical_ids"].copy(), dtype=torch.long, device=device),
        torch.as_tensor(point_ids, dtype=torch.long, device=device),
        torch.as_tensor(edge_type, dtype=torch.long, device=device),
        torch.as_tensor(edge_weight, dtype=torch.float32, device=device),
    )


class AutoCountSoft24GT(SoftBoundary24GT):
    def __init__(self, feature_dim, boundaries_m):
        super().__init__(feature_dim)
        base = band_layout((17,) * 5)
        count = len(boundaries_m) - 1
        maximum = max(17, count)
        mileage = np.arange(318206.0, 319639.0)
        pooling = np.zeros((5, maximum, len(mileage)), dtype=np.float32)
        physical_pool = np.zeros((5, maximum, 17), dtype=np.float32)
        fill = np.zeros((5, len(mileage), maximum), dtype=np.float32)
        for band in (0, 1, 2, 4):
            pooling[band, :17] = base["pooling"][band]
            physical_pool[band, :17] = base["embedding_pool"][band]
            fill[band, :, :17] = np.eye(17, dtype=np.float32)[base["point_ids"][band]]
        point_ids = np.searchsorted(boundaries_m[1:-1], mileage, side="right")
        for node in range(count):
            keep = point_ids == node
            pooling[3, node, keep] = 1.0 / keep.sum()
            physical_pool[3, node] = (
                np.bincount(base["physical_ids"][keep], minlength=17) / keep.sum()
            )
        fill[3, :, :count] = np.eye(count, dtype=np.float32)[point_ids]
        for node, boundary in enumerate(boundaries_m[1:-1]):
            nearby = (mileage >= boundary - 12) & (mileage <= boundary + 12)
            phase = (mileage[nearby] - (boundary - 12)) / 24.0
            right = 0.5 - 0.5 * np.cos(np.pi * phase)
            fill[3, nearby] = 0.0
            fill[3, nearby, node] = 1.0 - right
            fill[3, nearby, node + 1] = right
        self.band_pooling = torch.from_numpy(pooling)
        self.physical_embedding_pool = torch.from_numpy(physical_pool)
        self.point_fill = torch.from_numpy(fill)
