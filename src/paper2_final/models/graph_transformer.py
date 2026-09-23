from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from paper2_final.models.legacy_segments import (
    observation_segments_by_band,
    physical_stiffness_segments,
    segment_index,
)
from paper2_final.models.point_token_graph import RoutedPointTokenResidualGT


@lru_cache(maxsize=None)
def band_layout(counts: tuple[int, ...], tower_weight: float = 0.25):
    if len(counts) != 5:
        raise ValueError("One segment count is required for each wavelength band")
    mileage = np.arange(318206.0, 319639.0)
    physical = physical_stiffness_segments()
    physical_ids = segment_index(mileage, physical)
    observed = observation_segments_by_band()
    existing = {
        102: observed["lt42"],
        51: observed["w42_70"],
        34: observed["w70_120"],
        17: physical,
    }
    merges = {9: (2, 2, 2, 2, 1, 2, 2, 2, 2), 5: (2, 4, 5, 4, 2), 3: (2, 13, 2)}
    maximum = max(counts)
    ids = np.zeros((5, len(mileage)), dtype=np.int64)
    pooling = np.zeros((5, maximum, len(mileage)), dtype=np.float32)
    embedding_pool = np.zeros((5, maximum, 17), dtype=np.float32)
    types = np.zeros((5, maximum, maximum), dtype=np.int64)
    weights = np.zeros((5, maximum, maximum), dtype=np.float32)
    all_bounds = []
    for band, count in enumerate(counts):
        if count in existing:
            bounds = [existing[count][0].start_m, *[s.end_m for s in existing[count]]]
        else:
            cursor, bounds = 0, [physical[0].start_m]
            for group in merges[count]:
                cursor += group
                bounds.append(physical[cursor - 1].end_m)
            assert cursor == 17
        assert len(bounds) == count + 1 and 318376 in bounds and 319468 in bounds
        point_ids = np.searchsorted(np.asarray(bounds[1:-1]), mileage, side="right")
        ids[band] = point_ids
        for node in range(count):
            keep = point_ids == node
            pooling[band, node, keep] = 1 / int(keep.sum())
            embedding_pool[band, node] = np.bincount(physical_ids[keep], minlength=17) / keep.sum()
        starts, ends = np.asarray(bounds[:-1]), np.asarray(bounds[1:])
        zones = np.where(ends <= 318376, 0, np.where(starts >= 319468, 2, 1))
        cross = zones[:, None] != zones[None, :]
        types[band, :count, :count] = np.where(cross, 2, 1)
        weights[band, :count, :count] = np.where(cross, tower_weight, 1)
        # Padded queries only attend to their own dummy self; actual queries
        # cannot attend to padded keys. This prevents all-masked softmax rows.
        np.fill_diagonal(types[band], 0)
        np.fill_diagonal(weights[band], 1)
        all_bounds.append(bounds)
    return dict(
        counts=counts,
        physical_ids=physical_ids,
        point_ids=ids,
        pooling=pooling,
        embedding_pool=embedding_pool,
        edge_type=types,
        edge_weight=weights,
        boundaries_m=all_bounds,
    )


def multires_graph_tensors(physical_ids, counts, tower_weight, device):
    layout = band_layout(tuple(counts), float(tower_weight))
    if not np.array_equal(physical_ids, layout["physical_ids"]):
        raise ValueError("Unexpected physical segment mapping or mileage grid")
    return (
        torch.as_tensor(physical_ids, dtype=torch.long, device=device),
        torch.as_tensor(layout["point_ids"].copy(), dtype=torch.long, device=device),
        torch.as_tensor(layout["edge_type"].copy(), dtype=torch.long, device=device),
        torch.as_tensor(layout["edge_weight"].copy(), dtype=torch.float32, device=device),
    )


class BandMultiResolutionFullGT(RoutedPointTokenResidualGT):
    """Parameter-matched per-band complete graphs with different node counts.

    Pooling and physical-embedding maps are fixed geometry, not parameters.
    Band graphs are separate batch entries; no cross-band messages are possible.
    """

    def __init__(self, feature_dim, *, band_segment_counts, **kwargs):
        super().__init__(feature_dim, token_mode="segment17_full", **kwargs)
        self.token_mode = "band_multires_full"
        self.band_segment_counts = tuple(map(int, band_segment_counts))
        layout = band_layout(self.band_segment_counts)
        self.register_buffer("band_pooling", torch.tensor(layout["pooling"]), persistent=False)
        self.register_buffer(
            "physical_embedding_pool",
            torch.tensor(layout["embedding_pool"]),
            persistent=False,
        )

    def _attend(self, h, edge_type, edge_weight):
        block = self.token_graph
        batch, nodes, width = h.shape
        q = block.q(h).view(batch, nodes, block.heads, block.head_dim).transpose(1, 2)
        k = block.k(h).view(batch, nodes, block.heads, block.head_dim).transpose(1, 2)
        v = block.v(h).view(batch, nodes, block.heads, block.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(block.head_dim)
        logits = logits + block.edge_bias(edge_type).permute(0, 3, 1, 2)
        logits = logits + torch.log(edge_weight.clamp_min(1e-6))[:, None]
        logits = logits.masked_fill(edge_weight[:, None] <= 0, -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        messages = torch.matmul(attention * edge_weight[:, None], v)
        update = block.o(messages.transpose(1, 2).reshape(batch, nodes, width))
        h = block.norm1(h + block.drop(update))
        return block.norm2(h + block.drop(block.ff(h)))

    def forward(
        self,
        point_features,
        segment_index,
        band_point_ids,
        edge_type,
        edge_weight,
        *,
        graph_enabled=True,
    ):
        if not graph_enabled:
            raise ValueError("The ABC experiment requires within-band complete graphs")
        events, points, bands, features = point_features.shape
        if points != 1433 or bands != 5 or segment_index.shape != (points,):
            raise ValueError("Expected five bands on the canonical 1433-point grid")
        flattened = point_features.permute(0, 2, 3, 1).reshape(events * bands, features, points)
        local = self.local_encoder(flattened).transpose(1, 2)
        band_ids = torch.arange(bands, device=local.device).repeat(events)
        local = self.local_norm(local + self.band_embedding(band_ids).unsqueeze(1))
        local_by_band = local.reshape(events, bands, points, self.d_model)
        tokens = torch.matmul(self.band_pooling.unsqueeze(0), local_by_band)
        structural_embedding = torch.matmul(
            self.physical_embedding_pool, self.segment_embedding.weight
        )
        tokens = self.token_pre(tokens + structural_embedding.unsqueeze(0))
        maximum = self.band_pooling.shape[1]
        tokens = tokens.reshape(events * bands, maximum, self.d_model)
        tokens = self._attend(
            tokens,
            edge_type.repeat(events, 1, 1),
            edge_weight.repeat(events, 1, 1),
        )
        tokens = tokens.reshape(events, bands, maximum, self.d_model)
        gather = band_point_ids[None, :, :, None].expand(events, bands, points, self.d_model)
        context = torch.gather(tokens, 2, gather).reshape(events * bands, points, self.d_model)
        fused = self.fusion(torch.cat([local, context], dim=-1).transpose(1, 2))
        fused = fused.reshape(events, bands, self.d_model, points)
        return torch.stack([self.heads[b](fused[:, b]).squeeze(1) for b in range(bands)], dim=2)
