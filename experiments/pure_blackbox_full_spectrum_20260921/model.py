"""Models for the five-band, data-only full-spectrum comparison."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

FIVE_BAND_MODEL = ROOT.parent / "v3_graph_five_band_7m_center_20260919" / "model.py"
spec = importlib.util.spec_from_file_location("frozen_five_band_model", FIVE_BAND_MODEL)
frozen_five_band = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen_five_band)
CenterBandsGT = frozen_five_band.CenterBandsGT

from paper2_final.reference_models import Model as FrozenBaselineModel


class BASGTDataK3(CenterBandsGT):
    """Frozen five-band BASGT geometry with three graph Transformer layers."""

    def __init__(self, feature_dim: int, boundaries: list):
        super().__init__(feature_dim, boundaries)
        first = self._modules.pop("token_graph")
        self.graph_layers = nn.ModuleList([first] + [copy.deepcopy(first) for _ in range(2)])
        self.graph_depth = 3

    @staticmethod
    def _attend_layer(h, edge_type, edge_weight, block):
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
        repeated_type = edge_type.repeat(events, 1, 1)
        repeated_weight = edge_weight.repeat(events, 1, 1)
        for block in self.graph_layers:
            tokens = self._attend_layer(tokens, repeated_type, repeated_weight, block)
        tokens = tokens.reshape(events, bands, maximum, self.d_model)
        context = torch.matmul(self.point_fill.unsqueeze(0), tokens)
        context = context.reshape(events * bands, points, self.d_model)
        fused = self.fusion(torch.cat([local, context], dim=-1).transpose(1, 2))
        fused = fused.reshape(events, bands, self.d_model, points)
        return torch.stack(
            [self.heads[band](fused[:, band]).squeeze(1) for band in range(bands)],
            dim=2,
        )


class FiveBandBaseline(nn.Module):
    """Five independent frozen baseline bodies, one for each predicted band."""

    def __init__(self, name: str, input_dim: int):
        super().__init__()
        self.name = name
        self.bodies = nn.ModuleList([FrozenBaselineModel(name, input_dim) for _ in range(5)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.bodies[band](features[:, :, band]) for band in range(5)],
            dim=2,
        )
