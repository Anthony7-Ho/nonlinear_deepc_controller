#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalDilatedBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = F.pad(x, (self.left_pad, 0))
        y = self.conv(y)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.drop(self.act(y))
        return x + y


class TCNFrictionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 21,
        n_joints: int = 7,
        channels: int = 64,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        head_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, channels)
        self.blocks = nn.ModuleList(
            [
                CausalDilatedBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels, head_hidden),
                    nn.GELU(),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(n_joints)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, D)
        y = self.proj(x)
        y = y.transpose(1, 2)  # (B, C, W)
        for block in self.blocks:
            y = block(y)
        feat = y[:, :, -1]
        return torch.cat([h(feat) for h in self.heads], dim=1)
