# TCN and Mamba Implementation Guide (for nonlinear_deepc_controller)

This guide is adapted to your current pipeline in `data_processing/train_nn_friction_predictor.py`.
It follows the path you proposed:

1. Phase 1: Causal TCN first (low-risk, fast integration)
2. Phase 2: Mamba if TCN plateaus
3. Skip Transformer / JEPA / LSTM / GRU for now

---

## 1) Current baseline in your repo

Your NN trainer currently does this:

- Baseline model: shared-trunk MLP on one timestep `[q, dq]`
- Physics model: LuGre analytical prefit + per-joint residual MLP on one timestep
- Targets: `tau_ext_0..6`
- Outputs:
  - plots to `data_processing/plots`
  - model checkpoints to `data_processing/model`
  - summary json to `data_processing/nn_results`

The key limitation is that both NN branches are memoryless (single-step input).

---

## 2) Shared sequence data pipeline

### Goal
Convert feature construction from per-step to windowed sequence:

- input shape: `(B, W, D)`
- target shape: `(B, 7)`
- target at the last index of each window

### Recommended feature set
Use per-timestep feature vector:

- `q_t` (7)
- `dq_t` (7)
- `sign(dq_t)` (7)

Total `D = 21`.

### Important anti-leak rule
Do not shuffle timesteps before creating windows.
Your train/validation split is already file-based (`tau_log_train.csv`, `tau_log_validation.csv`), so this is good.
Only shuffle windows at DataLoader level.

### Minimal dataset class
Create a new file: `data_processing/sequence_dataset.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedFrictionDataset(Dataset):
    """
    x_step: (N, D)
    y_step: (N, 7)
    returns:
      x_win: (W, D)
      y_tgt: (7,) at the last step of the window
    """

    def __init__(self, x_step: np.ndarray, y_step: np.ndarray, window: int = 15):
        if window < 2:
            raise ValueError("window must be >= 2")
        if x_step.shape[0] != y_step.shape[0]:
            raise ValueError("x_step and y_step must have the same length")
        if x_step.shape[0] < window:
            raise ValueError("number of samples must be >= window")

        self.x = torch.from_numpy(x_step.astype(np.float32))
        self.y = torch.from_numpy(y_step.astype(np.float32))
        self.window = int(window)
        self.n_valid = self.x.shape[0] - self.window + 1

    def __len__(self) -> int:
        return self.n_valid

    def __getitem__(self, idx: int):
        x_win = self.x[idx : idx + self.window]           # (W, D)
        y_tgt = self.y[idx + self.window - 1]             # (7,)
        return x_win, y_tgt
```

---

## 3) Phase 1: Causal TCN integration

### Architecture target

- Window: `W = 10..15` (start with 15)
- 3-4 causal dilated Conv1d blocks
- channels: `64` or `96`
- per-joint heads kept as in current code

### Receptive field
For kernel `k=3`, dilations `[1,2,4,8]`:

- receptive field = `1 + (k-1)*(1+2+4+8) = 31` timesteps

Usually enough for mild hysteresis.

### New model file
Create: `data_processing/model_tcn.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalDilatedBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0)
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
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, channels)
        self.blocks = nn.ModuleList(
            [
                CausalDilatedBlock(channels, kernel_size, 2 ** i, dropout)
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
        y = self.proj(x).transpose(1, 2)  # (B, C, W)
        for blk in self.blocks:
            y = blk(y)
        feat = y[:, :, -1]                # (B, C)
        out = torch.cat([h(feat) for h in self.heads], dim=1)
        return out
```

### Integrate into your current trainer
In `train_nn_friction_predictor.py`:

1. Add new args:
   - `--arch {mlp,tcn,mamba}` default `tcn`
   - `--window` default `15`
   - TCN args: `--tcn-channels`, `--tcn-layers`, `--tcn-kernel-size`, `--tcn-dropout`
2. Build step features with temporal context:
   - `x_step = [q, dq, sign(dq)]` -> `(N,21)`
3. Use `WindowedFrictionDataset` and DataLoader to emit `(B,W,D)`
4. Train model to predict `tau` directly first (simplest)
5. Save checkpoint to `data_processing/model/friction_tcn.pt`
6. Save plots to `data_processing/plots` and summary to `data_processing/nn_results`

### Why direct prediction first
You can combine LuGre+TCN later, but first compare pure TCN vs your current MLP baseline quickly and cleanly.

---

## 4) Phase 2: Mamba integration (only if TCN plateaus)

### Install

GPU path:

```bash
pip install mamba-ssm causal-conv1d
```

If fallback is needed:

```bash
pip install "mamba-ssm[torch]"
```

### Model file
Create `data_processing/model_mamba.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm") from exc


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
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(d_model),
                        "mamba": Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
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
            r = y
            y = layer["norm"](y)
            y = layer["mamba"](y)
            y = layer["drop"](y) + r
        feat = y[:, -1, :]  # last step
        return torch.cat([h(feat) for h in self.heads], dim=1)
```

### Integrate in trainer
Add Mamba args:

- `--mamba-d-model` default 64
- `--mamba-d-state` default 16
- `--mamba-d-conv` default 4
- `--mamba-expand` default 2
- `--mamba-layers` default 2
- `--mamba-dropout` default 0.1

Set `--arch mamba` to instantiate this model and reuse the same windowed DataLoader.

---

## 5) Training recipe (works for both TCN and Mamba)

### Start values

- window `W=15`
- lr `3e-4`
- weight decay `1e-5`
- batch size `256` or `512`
- epochs `200`
- early stop patience `20` eval checks
- gradient clip `1.0` (especially useful for Mamba)

### Add in your training step

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## 6) Evaluation protocol (keep it fair)

Compare these on the same validation file:

1. current MLP baseline
2. TCN direct
3. Mamba direct

Track:

- global MSE
- per-joint MSE
- per-joint correlation
- improvement by joint (some joints likely gain more due to stronger hysteresis)

If TCN already gives strong gains and stable latency, Mamba is optional.
If gains are uneven by joint and dynamics are selective, Mamba is worth the next step.

---

## 7) Deployment notes

### TCN
Maintain a rolling buffer of the last `W` feature vectors.
Each control tick:

1. append current `[q,dq,sign(dq)]`
2. run model on `(1,W,D)`
3. output `(7,)`

### Mamba
Use stateful step inference only after offline validation is complete.
If step-cache API is used, call reset at trajectory boundaries.

---

## 8) Suggested execution plan

### Day 1

- Implement `WindowedFrictionDataset`
- Implement `TCNFrictionModel`
- Add `--arch tcn` path in `train_nn_friction_predictor.py`
- Train + validate + save `friction_tcn.pt`

### Day 2

- Tune TCN (`W`, channels, layers)
- If no further gain, implement Mamba path
- Compare TCN vs Mamba on same metrics

---

## 9) What to skip for now

- Full Transformer (overkill and latency-heavy for this task)
- JEPA pretraining (complexity not justified yet)
- LSTM/GRU (usually weaker/less parallel than TCN here)

---

## 10) Final recommendation

Start with TCN now. It is the best complexity/performance trade for your current codebase.
If TCN saturates and hysteresis remains joint-selective, add Mamba as the principled upgrade.
