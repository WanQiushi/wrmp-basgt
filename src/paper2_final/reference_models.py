from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from paper2_final.models.vendor.tslib_vendor import DLinear, TimeXer, iTransformer


class Model(nn.Module):
    """Common adapter for the data-only neural baseline families."""

    def __init__(
        self,
        name: str,
        input_dim: int = 83,
    ) -> None:
        super().__init__()
        self.name = name
        self.input_projection = nn.Linear(input_dim, 4)
        if name == "LSTM":
            self.body = nn.LSTM(4, 32, num_layers=1, batch_first=True)
            self.output_projection = nn.Linear(32, 1)
            return

        self.output_projection = nn.Linear(4, 1)
        config = SimpleNamespace(
            task_name="long_term_forecast",
            features="M",
            seq_len=1440,
            pred_len=1440,
            label_len=0,
            enc_in=4,
            dec_in=4,
            c_out=4,
            d_model=32,
            n_heads=4,
            e_layers=1,
            d_layers=1,
            d_ff=64,
            factor=1,
            dropout=0.1,
            activation="gelu",
            embed="timeF",
            freq="d",
            moving_avg=25,
            patch_len=16,
            use_norm=False,
            output_attention=False,
        )
        modules = {
            "DLinear": DLinear,
            "iTransformer": iTransformer,
            "TimeXer": TimeXer,
        }
        module = modules[name]
        self.body = module.Model(config)

    @property
    def inp(self) -> nn.Module:
        return self.input_projection

    @property
    def out(self) -> nn.Module:
        return self.output_projection

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.input_projection(features)
        point_count = values.shape[1]
        if self.name == "LSTM":
            encoded, _ = self.body(values)
            return self.output_projection(encoded)[:, :, 0]
        values = torch.cat([values, values[:, -1:].expand(-1, 1440 - point_count, -1)], dim=1)
        return self.output_projection(self.body(values, None, None, None))[:, :point_count, 0]
