#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mamba-ssm is required. Install with: pip install mamba-ssm causal-conv1d"
    ) from exc


class MambaFrictionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 21,
        n_joints: int = 7,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
        head_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(d_model),
                        "mamba": Mamba(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        ),
                        "drop": nn.Dropout(dropout),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, head_hidden),
                    nn.GELU(),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(n_joints)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, D)
        y = self.proj(x)
        for layer in self.layers:
            residual = y
            y = layer["norm"](y)
            y = layer["mamba"](y)
            y = layer["drop"](y) + residual
        feat = y[:, -1, :]
        return torch.cat([h(feat) for h in self.heads], dim=1)
