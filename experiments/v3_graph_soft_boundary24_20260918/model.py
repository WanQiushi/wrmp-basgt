"""17-node adjacent graph with a 24 m soft fill for the 120-200 m band."""

import numpy as np
import torch

from paper2_final.models.graph_transformer import BandMultiResolutionFullGT, band_layout

COUNTS = (17, 17, 17, 17, 17)
TRANSITION_M = 24.0


class SoftBoundary24GT(BandMultiResolutionFullGT):
    def __init__(self, feature_dim):
        super().__init__(
            feature_dim,
            band_segment_counts=COUNTS,
            d_model=368,
            heads=8,
            ff_dim=736,
            local_kernel_size=9,
            dropout=0.1,
        )
        layout = band_layout(COUNTS)
        fill = np.eye(17, dtype=np.float32)[layout["point_ids"]]
        mileage = np.arange(318206.0, 319639.0)
        for j, boundary in enumerate(layout["boundaries_m"][3][1:-1]):
            in_transition = (mileage >= boundary - TRANSITION_M / 2) & (
                mileage <= boundary + TRANSITION_M / 2
            )
            phase = (mileage[in_transition] - (boundary - TRANSITION_M / 2)) / TRANSITION_M
            right = 0.5 - 0.5 * np.cos(np.pi * phase)
            fill[3, in_transition] = 0.0
            fill[3, in_transition, j] = 1.0 - right
            fill[3, in_transition, j + 1] = right
        self.register_buffer("point_fill", torch.from_numpy(fill), persistent=False)

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
        events, points, bands, features = point_features.shape
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
        context = torch.matmul(self.point_fill.unsqueeze(0), tokens)
        context = context.reshape(events * bands, points, self.d_model)
        fused = self.fusion(torch.cat([local, context], dim=-1).transpose(1, 2))
        fused = fused.reshape(events, bands, self.d_model, points)
        return torch.stack([self.heads[b](fused[:, b]).squeeze(1) for b in range(bands)], dim=2)
