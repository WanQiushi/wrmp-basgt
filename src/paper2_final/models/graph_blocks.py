from __future__ import annotations

import math

import torch
from torch import nn


class GraphBlock(nn.Module):
    """Small edge-aware attention block migrated from the paper2 PIGT design."""

    def __init__(self, d_model: int, heads: int, ff_dim: int, n_edge_types: int, dropout: float):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Embedding(n_edge_types, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_type, edge_weight, *, graph_enabled: bool = True):
        batch, nodes, d_model = h.shape
        q = self.q(h).view(batch, nodes, self.heads, self.head_dim)
        k = self.k(h).view(batch, nodes, self.heads, self.head_dim)
        v = self.v(h).view(batch, nodes, self.heads, self.head_dim)
        if not graph_enabled:
            # Capacity-matched self-token control: Q, K, V, O, edge bias,
            # normalization and FFN all remain active, but no node can receive
            # information from another node.
            bias = self.edge_bias.weight[0].view(1, 1, self.heads)
            gate = torch.sigmoid((q * k).sum(-1) / math.sqrt(self.head_dim) + bias)
            messages = gate.unsqueeze(-1) * v
            update = self.o(messages.reshape(batch, nodes, d_model))
            h = self.norm1(h + self.drop(update))
            return self.norm2(h + self.drop(self.ff(h)))
        src, dst = edge_index[0], edge_index[1]
        logits = (q[:, dst] * k[:, src]).sum(-1) / math.sqrt(self.head_dim)
        logits = logits + self.edge_bias(edge_type).unsqueeze(0)
        logits = logits + torch.log(edge_weight.clamp_min(1e-6)).view(1, -1, 1)
        messages = torch.zeros(
            batch, nodes, self.heads, self.head_dim, device=h.device, dtype=h.dtype
        )
        for node in range(nodes):
            ids = torch.nonzero(dst == node, as_tuple=False).flatten()
            if ids.numel() == 0:
                continue
            attention = torch.softmax(logits[:, ids], dim=1)
            weighted = attention.unsqueeze(-1) * edge_weight[ids].view(1, -1, 1, 1) * v[:, src[ids]]
            messages[:, node] = weighted.sum(1)
        update = self.o(messages.reshape(batch, nodes, d_model))
        h = self.norm1(h + self.drop(update))
        return self.norm2(h + self.drop(self.ff(h)))


class DensePointGraphBlock(GraphBlock):
    """Exact all-pairs version of GraphBlock with the same learned parameters.

    Matrix axes are [destination, source]. All weights are strictly positive;
    tower weights enter both logits and messages, exactly as in GraphBlock.
    No attention dropout is added (the baseline also has none).
    """

    def forward(self, h, edge_index, edge_type, edge_weight, *, graph_enabled=True):
        if not graph_enabled:
            return super().forward(h, edge_index, edge_type, edge_weight, graph_enabled=False)
        batch, nodes, width = h.shape
        q = self.q(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + self.edge_bias(edge_type).permute(2, 0, 1).unsqueeze(0)
        logits = logits + torch.log(edge_weight.clamp_min(1e-6))[None, None]
        attention = torch.softmax(logits, dim=-1)
        messages = torch.matmul(attention * edge_weight[None, None], v)
        update = self.o(messages.transpose(1, 2).reshape(batch, nodes, width))
        h = self.norm1(h + self.drop(update))
        return self.norm2(h + self.drop(self.ff(h)))
