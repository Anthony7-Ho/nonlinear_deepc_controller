#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedFrictionDataset(Dataset):
    """Windowed time-series dataset for friction prediction.

    Returns (x_window, y_target), where x_window has shape (W, D)
    and y_target is the target vector at the last timestep of that window.
    """

    def __init__(self, x_step: np.ndarray, y_step: np.ndarray, window: int = 15):
        if window < 2:
            raise ValueError("window must be >= 2")
        if x_step.shape[0] != y_step.shape[0]:
            raise ValueError("x_step and y_step must have same number of rows")
        if x_step.shape[0] < window:
            raise ValueError("x_step length must be >= window")

        self.x = torch.from_numpy(x_step.astype(np.float32))
        self.y = torch.from_numpy(y_step.astype(np.float32))
        self.window = int(window)
        self.n_valid = int(self.x.shape[0] - self.window + 1)

    def __len__(self) -> int:
        return self.n_valid

    def __getitem__(self, idx: int):
        x_win = self.x[idx : idx + self.window]
        y_tgt = self.y[idx + self.window - 1]
        return x_win, y_tgt
