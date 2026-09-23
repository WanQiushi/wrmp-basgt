from __future__ import annotations

import torch
from torch import nn

from paper2_final.models.graph_blocks import DensePointGraphBlock, GraphBlock


class RoutedPointTokenResidualGT(nn.Module):
    """Five routed wavelength streams with shared structural graph reasoning.

    Input has shape ``[event, point, band, feature]``.  Each wavelength is
    encoded independently, so a source masked from one band cannot reach that
    band's output through a shared multi-output head.  Parameters are shared
    across bands for small-sample regularization; a band embedding is added
    exactly once before segment pooling.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        n_segments: int = 17,
        n_bands: int = 5,
        d_model: int = 32,
        heads: int = 4,
        ff_dim: int = 64,
        local_kernel_size: int = 9,
        dropout: float = 0.1,
        n_edge_types: int = 3,
        token_mode: str = "segment17",
    ) -> None:
        super().__init__()
        if local_kernel_size < 3 or local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be an odd integer of at least three")
        if token_mode not in ("segment17", "segment17_full", "point1433_full"):
            raise ValueError(token_mode)
        self.token_mode = token_mode
        self.n_segments = int(n_segments)
        self.n_bands = int(n_bands)
        self.d_model = int(d_model)
        padding = local_kernel_size // 2
        self.local_encoder = nn.Sequential(
            nn.Conv1d(feature_dim, d_model, local_kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(
                d_model,
                d_model,
                local_kernel_size,
                padding=padding,
                groups=d_model,
            ),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1),
        )
        self.local_norm = nn.LayerNorm(d_model)
        self.band_embedding = nn.Embedding(n_bands, d_model)
        self.segment_embedding = nn.Embedding(n_segments, d_model)
        self.token_pre = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU()
        )
        block = (
            DensePointGraphBlock
            if token_mode in ("segment17_full", "point1433_full")
            else GraphBlock
        )
        self.token_graph = block(d_model, heads, ff_dim, n_edge_types, dropout)
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * d_model, d_model, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Conv1d(d_model, 1, 1) for _ in range(n_bands)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _pool_tokens(self, point_state: torch.Tensor, segment_index: torch.Tensor) -> torch.Tensor:
        tokens = []
        for segment in range(self.n_segments):
            keep = segment_index == segment
            if not torch.any(keep):
                raise ValueError(f"segment {segment} has no point on the supplied grid")
            tokens.append(point_state[:, keep].mean(dim=1))
        return torch.stack(tokens, dim=1)

    def forward(
        self,
        point_features: torch.Tensor,
        segment_index: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
        *,
        graph_enabled: bool = True,
    ) -> torch.Tensor:
        if point_features.ndim != 4:
            raise ValueError("point_features must have shape [event, point, band, feature]")
        events, points, bands, features = point_features.shape
        if bands != self.n_bands:
            raise ValueError(f"expected {self.n_bands} wavelength bands, got {bands}")
        if segment_index.shape != (points,):
            raise ValueError("segment_index must have one entry per point")

        flattened = point_features.permute(0, 2, 3, 1).reshape(events * bands, features, points)
        local = self.local_encoder(flattened).transpose(1, 2)
        band_ids = torch.arange(bands, device=local.device).repeat(events)
        local = self.local_norm(local + self.band_embedding(band_ids).unsqueeze(1))
        if self.token_mode == "point1433_full":
            tokens = local
            segment_ids = segment_index
        else:
            tokens = self._pool_tokens(local, segment_index)
            segment_ids = torch.arange(self.n_segments, device=local.device)
        tokens = self.token_pre(tokens + self.segment_embedding(segment_ids).unsqueeze(0))
        tokens = self.token_graph(
            tokens,
            edge_index,
            edge_type,
            edge_weight,
            graph_enabled=graph_enabled,
        )
        point_context = tokens if self.token_mode == "point1433_full" else tokens[:, segment_index]
        fused = self.fusion(torch.cat([local, point_context], dim=-1).transpose(1, 2))
        fused = fused.reshape(events, bands, self.d_model, points)
        outputs = [self.heads[band](fused[:, band]).squeeze(1) for band in range(bands)]
        return torch.stack(outputs, dim=2)
