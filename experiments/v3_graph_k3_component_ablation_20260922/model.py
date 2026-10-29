"""K=3 component-ablation models for the frozen four-band BASGT."""

import copy
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent

DEPTH_SOURCE = ROOT.parent / "v3_graph_depth_ablation_20260919" / "model.py"
depth_spec = importlib.util.spec_from_file_location("frozen_k3_depth_model", DEPTH_SOURCE)
depth_model = importlib.util.module_from_spec(depth_spec)
depth_spec.loader.exec_module(depth_model)
DeepFourBandGT = depth_model.DeepFourBandGT

COMPONENT_SOURCE = ROOT.parent / "v3_graph_four_band_ablation_20260919" / "model.py"
component_spec = importlib.util.spec_from_file_location(
    "frozen_k1_component_model", COMPONENT_SOURCE
)
component_model = importlib.util.module_from_spec(component_spec)
component_spec.loader.exec_module(component_model)
PointGraphGT = component_model.PointGraphGT


class K3FourBandGT(DeepFourBandGT):
    """Three graph layers with an optional removal of the direct local route."""

    def __init__(self, feature_dim, boundaries, *, no_local_direct=False):
        super().__init__(feature_dim, boundaries, graph_depth=3)
        self.no_local_direct = bool(no_local_direct)

    def forward(self, point_features, segment_index, band_point_ids,
                edge_type, edge_weight, *, graph_enabled=True):
        events, points, bands, features = point_features.shape
        flattened = point_features.permute(0, 2, 3, 1).reshape(
            events * bands, features, points
        )
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
            tokens = self._attend_layer(
                tokens, repeated_type, repeated_weight, block
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


class K3PointGraphGT(PointGraphGT):
    """Three-layer 1433-point adjacent graph without segment pooling."""

    def __init__(self, feature_dim, boundaries):
        super().__init__(feature_dim, boundaries)
        first = self._modules.pop("token_graph")
        self.graph_layers = nn.ModuleList([first, copy.deepcopy(first), copy.deepcopy(first)])
        self.graph_depth = 3

    @staticmethod
    def _point_attend_layer(h, block):
        batch, nodes, width = h.shape
        q = block.q(h).view(batch, nodes, block.heads, block.head_dim)
        k = block.k(h).view(batch, nodes, block.heads, block.head_dim)
        v = block.v(h).view(batch, nodes, block.heads, block.head_dim)
        index = torch.arange(nodes, device=h.device)
        neighbors = torch.stack(
            ((index - 1).clamp_min(0), index, (index + 1).clamp_max(nodes - 1)),
            dim=1,
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
        messages = (
            attention.unsqueeze(-1)
            * weight[None, :, :, None, None]
            * v[:, neighbors]
        ).sum(2)
        update = block.o(messages.reshape(batch, nodes, width))
        h = block.norm1(h + block.drop(update))
        return block.norm2(h + block.drop(block.ff(h)))

    def _point_attend(self, h):
        for block in self.graph_layers:
            h = self._point_attend_layer(h, block)
        return h
