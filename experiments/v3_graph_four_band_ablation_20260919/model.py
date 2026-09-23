"""Four learned bands, with the fifth band supplied only by the white-box prior."""

import importlib.util
import math
from pathlib import Path

import torch
from torch import nn

SOURCE = (
    Path(__file__).resolve().parent.parent / "v3_graph_five_band_7m_center_20260919" / "model.py"
)
spec = importlib.util.spec_from_file_location("five_band_center_model", SOURCE)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
CenterBandsGT = previous.CenterBandsGT


class FourBandGT(CenterBandsGT):
    def __init__(self, feature_dim, boundaries, *, no_local_direct=False):
        super().__init__(feature_dim, boundaries)
        self.band_pooling = self.band_pooling[:4]
        self.physical_embedding_pool = self.physical_embedding_pool[:4]
        self.point_fill = self.point_fill[:4]
        self.band_embedding = nn.Embedding(4, self.d_model)
        self.heads = nn.ModuleList(list(self.heads)[:4])
        self.n_bands = 4
        self.no_local_direct = no_local_direct

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
        direct = torch.zeros_like(local) if self.no_local_direct else local
        fused = self.fusion(torch.cat([direct, context], dim=-1).transpose(1, 2))
        fused = fused.reshape(events, bands, self.d_model, points)
        return torch.stack(
            [self.heads[band](fused[:, band]).squeeze(1) for band in range(bands)],
            dim=2,
        )


class PointGraphGT(FourBandGT):
    """No segment pooling: 1433 point nodes per band with adjacent edges."""

    def _point_attend(self, h):
        block = self.token_graph
        batch, nodes, width = h.shape
        q = block.q(h).view(batch, nodes, block.heads, block.head_dim)
        k = block.k(h).view(batch, nodes, block.heads, block.head_dim)
        v = block.v(h).view(batch, nodes, block.heads, block.head_dim)
        index = torch.arange(nodes, device=h.device)
        neighbors = torch.stack(
            ((index - 1).clamp_min(0), index, (index + 1).clamp_max(nodes - 1)), dim=1
        )
        weight = torch.ones((nodes, 3), device=h.device, dtype=h.dtype)
        weight[0, 0] = 0
        weight[-1, 2] = 0
        weight[170, 0] = weight[169, 2] = 0.25
        weight[1262, 0] = weight[1261, 2] = 0.25
        edge_type = torch.ones((nodes, 3), device=h.device, dtype=torch.long)
        edge_type[:, 1] = 0
        edge_type[170, 0] = edge_type[169, 2] = 2
        edge_type[1262, 0] = edge_type[1261, 2] = 2
        logits = (q.unsqueeze(2) * k[:, neighbors]).sum(-1) / math.sqrt(block.head_dim)
        logits = logits + block.edge_bias(edge_type).unsqueeze(0)
        logits = logits + weight.clamp_min(1e-6).log()[None, :, :, None]
        logits = logits.masked_fill(weight[None, :, :, None] == 0, -torch.inf)
        attention = torch.softmax(logits, dim=2)
        messages = (attention.unsqueeze(-1) * weight[None, :, :, None, None] * v[:, neighbors]).sum(
            2
        )
        update = block.o(messages.reshape(batch, nodes, width))
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
        events, points, bands, features = point_features.shape
        flattened = point_features.permute(0, 2, 3, 1).reshape(events * bands, features, points)
        local = self.local_encoder(flattened).transpose(1, 2)
        band_ids = torch.arange(bands, device=local.device).repeat(events)
        local = self.local_norm(local + self.band_embedding(band_ids).unsqueeze(1))
        physical = self.segment_embedding(segment_index)
        tokens = self.token_pre(local + physical.unsqueeze(0))
        context = self._point_attend(tokens)
        fused = self.fusion(torch.cat([local, context], dim=-1).transpose(1, 2))
        fused = fused.reshape(events, bands, self.d_model, points)
        return torch.stack(
            [self.heads[band](fused[:, band]).squeeze(1) for band in range(bands)],
            dim=2,
        )
