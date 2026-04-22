#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


class TCNPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        window: int,
        input_dim: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        device: str = "cpu",
    ) -> None:
        self.model = model.eval().to(device)
        self.device = torch.device(device)
        self.window = int(window)
        self.buffer = torch.zeros(self.window, input_dim, dtype=torch.float32)
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = np.where(y_std < 1e-8, 1.0, y_std).astype(np.float32)

    @torch.no_grad()
    def step(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        x_raw = np.concatenate([q, dq, np.sign(dq)]).astype(np.float32)
        x_norm = (x_raw - self.x_mean.reshape(-1)) / self.x_std.reshape(-1)

        self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
        self.buffer[-1] = torch.from_numpy(x_norm)

        pred_n = self.model(self.buffer.unsqueeze(0).to(self.device))
        pred = pred_n.cpu().numpy() * self.y_std + self.y_mean
        return pred.squeeze(0)


class MambaPredictor:
    """Stateful single-step inference for Mamba models.

    Requires a mamba_ssm version where each block exposes step().
    """

    def __init__(
        self,
        model: torch.nn.Module,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        device: str = "cpu",
    ) -> None:
        self.model = model.eval().to(device)
        self.device = torch.device(device)
        self.cache: Optional[List[object]] = None
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = np.where(y_std < 1e-8, 1.0, y_std).astype(np.float32)

    @torch.no_grad()
    def step(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        x_raw = np.concatenate([q, dq, np.sign(dq)]).astype(np.float32)
        x_norm = (x_raw - self.x_mean.reshape(-1)) / self.x_std.reshape(-1)
        x_t = torch.from_numpy(x_norm).unsqueeze(0).to(self.device)

        feat = self.model.proj(x_t)
        new_cache: List[object] = []

        for i, layer in enumerate(self.model.layers):
            feat_norm = layer["norm"](feat)
            mamba_block = layer["mamba"]
            if not hasattr(mamba_block, "step"):
                raise RuntimeError(
                    "Current mamba_ssm build has no step() API. "
                    "Use full-window forward deployment or update mamba_ssm."
                )

            cache_i = self.cache[i] if self.cache is not None else None
            out, new_cache_i = mamba_block.step(feat_norm, cache_i)
            feat = layer["drop"](out) + feat
            new_cache.append(new_cache_i)

        self.cache = new_cache
        pred_n = torch.cat([head(feat) for head in self.model.heads], dim=1)
        pred = pred_n.cpu().numpy() * self.y_std + self.y_mean
        return pred.squeeze(0)

    def reset(self) -> None:
        self.cache = None
