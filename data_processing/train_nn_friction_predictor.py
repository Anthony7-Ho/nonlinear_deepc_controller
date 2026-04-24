#!/usr/bin/env python3
"""Train baseline and physics-informed neural networks for friction prediction.

Baseline model (old approach):
  - shared-trunk multi-head MLP
  - inputs: [q_state_0..6, dq_state_0..6]
  - target: [tau_ext_0..6]

Physics-informed model (additional approach):
  - per-joint independent MLPs
  - explicit non-smooth feature sign(dq)
  - LuGre/Stribeck-inspired analytical prefit
  - NN learns residual after analytical fit
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from model_tcn import TCNFrictionModel
from sequence_dataset import WindowedFrictionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train friction predictors: legacy MLP/LuGre, temporal TCN, or temporal Mamba."
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="tcn",
        choices=["legacy", "tcn", "mamba"],
        help="Model architecture: legacy reproduces existing MLP+LuGre pipeline.",
    )
    parser.add_argument("--train-csv", type=str, default="tau_log_train.csv")
    parser.add_argument("--valid-csv", type=str, default="tau_log_validation.csv")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="nn_results",
        help="Directory for non-plot run artifacts (e.g., summary JSON).",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory for saved plots.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="model",
        help="Directory for NN checkpoint files (.pt).",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="nn_model_config.json",
        help="JSON file with model architecture settings.",
    )

    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size. <=0 means full-batch training.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Eval batch size. <=0 means one full validation batch.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--window",
        type=int,
        default=15,
        help="Sliding window length for temporal models (tcn/mamba).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Evaluate validation loss every N epochs for temporal models.",
    )
    parser.add_argument(
        "--early-stop-patience-checks",
        type=int,
        default=20,
        help="Early stop patience measured in validation checks for temporal models.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for temporal models.",
    )

    parser.add_argument("--tcn-channels", type=int, default=64)
    parser.add_argument("--tcn-layers", type=int, default=4)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--tcn-dropout", type=float, default=0.1)
    parser.add_argument("--head-hidden", type=int, default=32)

    parser.add_argument("--mamba-d-model", type=int, default=64)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-layers", type=int, default=2)
    parser.add_argument("--mamba-dropout", type=float, default=0.1)

    parser.add_argument(
        "--physics-include-q",
        action="store_true",
        help="Include q_j in physics-informed features (default: only dq_j and sign(dq_j)).",
    )
    parser.add_argument(
        "--lugre-vs-grid-size",
        type=int,
        default=33,
        help="Number of log-spaced vs candidates for LuGre prefit per joint.",
    )

    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Use torch.compile when available (can reduce training time after warmup).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ordered_cols(columns: pd.Index, pattern: str) -> list[str]:
    cols = [c for c in columns if c.startswith(pattern)]
    cols.sort(key=lambda c: int(c.split("_")[-1]))
    return cols


def load_state_to_friction_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv(csv_path)

    q_cols = _ordered_cols(data.columns, "q_state_")
    dq_cols = _ordered_cols(data.columns, "dq_state_")
    tau_cols = _ordered_cols(data.columns, "tau_ext_")

    q_full = data[q_cols].to_numpy(dtype=np.float64)
    dq_full = data[dq_cols].to_numpy(dtype=np.float64)
    tau_ext = data[tau_cols].to_numpy(dtype=np.float64)

    if q_full.size == 0 or dq_full.size == 0 or tau_ext.size == 0:
        raise ValueError(
            f"Missing q_state_ / dq_state_ / tau_ext_ columns in {csv_path}."
        )
    if not (q_full.shape == dq_full.shape == tau_ext.shape):
        raise ValueError(
            f"Inconsistent shapes in {csv_path}: "
            f"q={q_full.shape}, dq={dq_full.shape}, tau_ext={tau_ext.shape}"
        )

    return q_full.astype(np.float32), dq_full.astype(np.float32), tau_ext.astype(np.float32)


def drop_nonfinite_rows(
    q: np.ndarray,
    dq: np.ndarray,
    tau: np.ndarray,
    dataset_name: str,
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_q = np.isfinite(q).all(axis=1)
    mask_dq = np.isfinite(dq).all(axis=1)
    mask_tau = np.isfinite(tau).all(axis=1)
    valid_mask = mask_q & mask_dq & mask_tau

    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(
            f"Warning: dropped {dropped} non-finite rows from {dataset_name} set "
            f"({csv_path})."
        )

    q_clean = q[valid_mask]
    dq_clean = dq[valid_mask]
    tau_clean = tau[valid_mask]

    if q_clean.shape[0] == 0:
        raise ValueError(
            f"No finite rows remaining in {dataset_name} dataset after filtering: {csv_path}"
        )

    return q_clean, dq_clean, tau_clean


def build_baseline_features(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    return np.hstack([q, dq]).astype(np.float32)


def build_physics_features(
    q: np.ndarray,
    dq: np.ndarray,
    include_q: bool,
) -> np.ndarray:
    sign_dq = np.sign(dq)
    if include_q:
        per_joint = np.stack([q, dq, sign_dq], axis=2)
    else:
        per_joint = np.stack([dq, sign_dq], axis=2)
    return per_joint.reshape(per_joint.shape[0], -1).astype(np.float32)


def print_distribution_diagnostics(
    dq_train: np.ndarray,
    dq_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
) -> None:
    """Print per-feature train/val mean and std to flag distribution shift."""
    print("\n--- Distribution diagnostics (train vs val) ---")
    print(f"{'Feature':<18} {'train_mean':>12} {'val_mean':>12} {'train_std':>12} {'val_std':>12} {'mean_shift':>12}")
    n_joints = dq_train.shape[1]
    for j in range(n_joints):
        tm, vm = float(dq_train[:, j].mean()), float(dq_valid[:, j].mean())
        ts, vs_ = float(dq_train[:, j].std()), float(dq_valid[:, j].std())
        shift = abs(tm - vm) / max(ts, 1e-8)
        flag = " <-- SHIFT" if shift > 0.5 else ""
        print(f"  dq[{j}]           {tm:>12.4f} {vm:>12.4f} {ts:>12.4f} {vs_:>12.4f} {shift:>12.3f}{flag}")
    for j in range(n_joints):
        tm, vm = float(y_train[:, j].mean()), float(y_valid[:, j].mean())
        ts, vs_ = float(y_train[:, j].std()), float(y_valid[:, j].std())
        shift = abs(tm - vm) / max(ts, 1e-8)
        flag = " <-- SHIFT" if shift > 0.5 else ""
        print(f"  tau_ext[{j}]      {tm:>12.4f} {vm:>12.4f} {ts:>12.4f} {vs_:>12.4f} {shift:>12.3f}{flag}")
    print("--- End diagnostics ---\n")


def build_temporal_features(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    sign_dq = np.sign(dq)
    return np.hstack([q, dq, sign_dq]).astype(np.float32)


def compute_normalization(
    x_train: np.ndarray, y_train: np.ndarray
) -> Dict[str, np.ndarray]:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)

    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }


def normalize(
    x: np.ndarray, y: np.ndarray, stats: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    x_n = (x - stats["x_mean"]) / stats["x_std"]
    y_n = (y - stats["y_mean"]) / stats["y_std"]
    return x_n.astype(np.float32), y_n.astype(np.float32)


def load_model_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    hidden_dims = cfg.get("hidden_dims", [512, 256, 128, 64])
    head_hidden_dims = cfg.get("head_hidden_dims", [64])
    physics_hidden_dims = cfg.get("physics_hidden_dims", [64, 32])
    activation = str(cfg.get("activation", "tanh")).lower()

    if not isinstance(hidden_dims, list) or len(hidden_dims) == 0:
        raise ValueError("model config must contain non-empty list 'hidden_dims'.")
    if not all(isinstance(d, int) and d > 0 for d in hidden_dims):
        raise ValueError("'hidden_dims' entries must be positive integers.")
    if not isinstance(head_hidden_dims, list):
        raise ValueError("'head_hidden_dims' must be a list of integers.")
    if not all(isinstance(d, int) and d > 0 for d in head_hidden_dims):
        raise ValueError("'head_hidden_dims' entries must be positive integers.")
    if not isinstance(physics_hidden_dims, list) or len(physics_hidden_dims) == 0:
        raise ValueError("'physics_hidden_dims' must be a non-empty list of integers.")
    if not all(isinstance(d, int) and d > 0 for d in physics_hidden_dims):
        raise ValueError("'physics_hidden_dims' entries must be positive integers.")
    if activation not in {"tanh", "relu", "gelu"}:
        raise ValueError("'activation' must be one of: tanh, relu, gelu.")

    return {
        "hidden_dims": hidden_dims,
        "head_hidden_dims": head_hidden_dims,
        "physics_hidden_dims": physics_hidden_dims,
        "activation": activation,
    }


def resolve_batch_size(requested_batch_size: int, n_samples: int) -> int:
    if requested_batch_size <= 0:
        return n_samples
    return min(requested_batch_size, n_samples)


def build_data_loaders(
    x_train_n: np.ndarray,
    y_train_n: np.ndarray,
    x_valid_n: np.ndarray,
    y_valid_n: np.ndarray,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train_n))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid_n), torch.from_numpy(y_valid_n))

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, valid_loader


def build_windowed_loaders(
    x_train_n: np.ndarray,
    y_train_n: np.ndarray,
    x_valid_n: np.ndarray,
    y_valid_n: np.ndarray,
    window: int,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = WindowedFrictionDataset(x_train_n, y_train_n, window=window)
    valid_dataset = WindowedFrictionDataset(x_valid_n, y_valid_n, window=window)

    effective_train_bs = resolve_batch_size(train_batch_size, len(train_dataset))
    effective_eval_bs = resolve_batch_size(eval_batch_size, len(valid_dataset))

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=effective_eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, valid_loader


def _activation_layer(activation: str) -> type[nn.Module]:
    if activation == "tanh":
        return nn.Tanh
    if activation == "relu":
        return nn.ReLU
    return nn.GELU


class FrictionMLP(nn.Module):
    """Shared-trunk MLP with one prediction head per joint."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        head_hidden_dims: List[int],
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        act_layer = _activation_layer(activation)

        trunk_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(act_layer())
            prev = h

        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.ModuleList()
        for _ in range(output_dim):
            head_layers: List[nn.Module] = []
            head_prev = prev
            for h in head_hidden_dims:
                head_layers.append(nn.Linear(head_prev, h))
                head_layers.append(act_layer())
                head_prev = h
            head_layers.append(nn.Linear(head_prev, 1))
            self.heads.append(nn.Sequential(*head_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        outs = [head(feat) for head in self.heads]
        return torch.cat(outs, dim=1)


class IndependentJointMLP(nn.Module):
    """Per-joint MLPs to model mostly decoupled friction dynamics."""

    def __init__(
        self,
        num_joints: int,
        features_per_joint: int,
        hidden_dims: List[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.features_per_joint = features_per_joint
        act_layer = _activation_layer(activation)

        self.joint_nets = nn.ModuleList()
        for _ in range(num_joints):
            layers: List[nn.Module] = []
            prev = features_per_joint
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(act_layer())
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.joint_nets.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xj = x.view(-1, self.num_joints, self.features_per_joint)
        outs = [self.joint_nets[j](xj[:, j, :]) for j in range(self.num_joints)]
        return torch.cat(outs, dim=1)


def build_temporal_model(args: argparse.Namespace, input_dim: int, output_dim: int) -> nn.Module:
    if args.arch == "tcn":
        return TCNFrictionModel(
            input_dim=input_dim,
            n_joints=output_dim,
            channels=args.tcn_channels,
            n_layers=args.tcn_layers,
            kernel_size=args.tcn_kernel_size,
            dropout=args.tcn_dropout,
            head_hidden=args.head_hidden,
        )

    if args.arch == "mamba":
        from model_mamba import MambaFrictionModel

        return MambaFrictionModel(
            input_dim=input_dim,
            n_joints=output_dim,
            d_model=args.mamba_d_model,
            d_state=args.mamba_d_state,
            d_conv=args.mamba_d_conv,
            expand=args.mamba_expand,
            n_layers=args.mamba_layers,
            dropout=args.mamba_dropout,
            head_hidden=args.head_hidden,
        )

    raise ValueError(f"Unsupported temporal architecture: {args.arch}")


def train_temporal_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    eval_every: int,
    early_stop_patience_checks: int,
    tag: str,
) -> Tuple[Dict[str, list], float, int, int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history: Dict[str, list] = {
        "train_epochs": [],
        "train_loss": [],
        "valid_epochs": [],
        "valid_loss": [],
    }

    best_valid = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    checks_without_improve = 0
    last_epoch = epochs

    use_amp = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            bs = xb.shape[0]
            running += float(loss.item()) * bs
            count += bs

        scheduler.step()

        train_loss = running / max(count, 1)
        history["train_epochs"].append(epoch)
        history["train_loss"].append(train_loss)

        should_eval = epoch == 1 or epoch % max(eval_every, 1) == 0 or epoch == epochs
        if not should_eval:
            continue

        valid_loss = evaluate_loss(model, valid_loader, criterion, device)
        history["valid_epochs"].append(epoch)
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            checks_without_improve = 0
        else:
            checks_without_improve += 1

        print(
            f"[{tag}] Epoch {epoch:4d}/{epochs} | "
            f"train_loss={train_loss:.6e} | valid_loss={valid_loss:.6e}"
        )

        if early_stop_patience_checks > 0 and checks_without_improve >= early_stop_patience_checks:
            print(
                f"[{tag}] Early stop at epoch {epoch} "
                f"(best valid {best_valid:.6e} at epoch {best_epoch})"
            )
            last_epoch = epoch
            break

    model.load_state_dict(best_state)
    return history, best_valid, best_epoch, last_epoch


@torch.no_grad()
def predict_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    use_amp = device.type == "cuda"
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pb = model(xb)
        preds.append(pb.float().cpu())
        targets.append(yb.float().cpu())

    pred_n = torch.cat(preds, dim=0).numpy()
    true_n = torch.cat(targets, dim=0).numpy()
    return pred_n, true_n


def fit_lugre_single_joint(
    velocity: np.ndarray,
    torque: np.ndarray,
    vs_grid_size: int,
) -> Dict[str, float]:
    v = velocity.astype(np.float64)
    y = torque.astype(np.float64)
    eps = 1e-6

    abs_v = np.abs(v)
    nz = abs_v[abs_v > eps]
    if nz.size == 0:
        return {"Fc": 0.0, "Fv": 0.0, "Fs": 0.0, "vs": 1.0}

    v_min = max(float(np.percentile(nz, 10)), eps)
    v_max = max(float(np.percentile(nz, 95)), v_min * 1.01)
    vs_grid = np.geomspace(v_min, v_max, max(vs_grid_size, 5))

    s = np.sign(v)
    best = {
        "mse": float("inf"),
        "Fc": 0.0,
        "Fv": 0.0,
        "Fs": 0.0,
        "vs": float(vs_grid[0]),
    }

    for vs in vs_grid:
        phi = np.column_stack([
            s,
            v,
            np.exp(-((v / vs) ** 2)) * s,
        ])

        coeffs, _, _, _ = np.linalg.lstsq(phi, y, rcond=None)
        pred = phi @ coeffs
        mse = float(np.mean((y - pred) ** 2))

        if mse < best["mse"]:
            best.update(
                {
                    "mse": mse,
                    "Fc": float(coeffs[0]),
                    "Fv": float(coeffs[1]),
                    "Fs": float(coeffs[2]),
                    "vs": float(vs),
                }
            )

    return {
        "Fc": best["Fc"],
        "Fv": best["Fv"],
        "Fs": best["Fs"],
        "vs": best["vs"],
    }


def fit_lugre_bundle(
    dq_train: np.ndarray,
    tau_train: np.ndarray,
    vs_grid_size: int,
) -> Dict[str, np.ndarray]:
    n_joints = dq_train.shape[1]
    fc = np.zeros(n_joints, dtype=np.float64)
    fv = np.zeros(n_joints, dtype=np.float64)
    fs = np.zeros(n_joints, dtype=np.float64)
    vs = np.ones(n_joints, dtype=np.float64)

    for j in range(n_joints):
        params = fit_lugre_single_joint(dq_train[:, j], tau_train[:, j], vs_grid_size)
        fc[j] = params["Fc"]
        fv[j] = params["Fv"]
        fs[j] = params["Fs"]
        vs[j] = params["vs"]

    return {
        "Fc": fc,
        "Fv": fv,
        "Fs": fs,
        "vs": vs,
    }


def predict_lugre(dq: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    dq64 = dq.astype(np.float64)
    sign_dq = np.sign(dq64)
    fc = params["Fc"][None, :]
    fv = params["Fv"][None, :]
    fs = params["Fs"][None, :]
    vs = params["vs"][None, :]

    return (
        fc * sign_dq
        + fv * dq64
        + fs * np.exp(-((dq64 / np.maximum(vs, 1e-9)) ** 2)) * sign_dq
    ).astype(np.float32)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0

    use_amp = device.type == "cuda"
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(xb)
            loss = criterion(pred, yb)

        bs = xb.shape[0]
        total += float(loss.item()) * bs
        count += bs

    return total / max(count, 1)


@torch.no_grad()
def batched_predict(
    model: nn.Module,
    x_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds = []
    use_amp = device.type == "cuda"

    for start in range(0, x_tensor.shape[0], batch_size):
        xb = x_tensor[start : start + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pb = model(xb)
        preds.append(pb.float().cpu())

    return torch.cat(preds, dim=0).numpy()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    tag: str,
) -> Dict[str, list]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = {"train_loss": [], "valid_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        num_samples = 0

        use_amp = device.type == "cuda"
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = xb.shape[0]
            running += float(loss.item()) * bs
            num_samples += bs

        train_loss = running / max(num_samples, 1)
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[{tag}] Epoch {epoch:4d}/{epochs} | "
                f"train_loss={train_loss:.6e} | valid_loss={valid_loss:.6e}"
            )

    return history


def compute_joint_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    for j in range(y_true.shape[1]):
        y_true_j = y_true[:, j]
        y_pred_j = y_pred[:, j]
        mse = float(np.mean((y_true_j - y_pred_j) ** 2))
        std_true = float(np.std(y_true_j))
        std_pred = float(np.std(y_pred_j))
        corr = (
            float(np.corrcoef(y_true_j, y_pred_j)[0, 1])
            if std_true > 0.0 and std_pred > 0.0
            else float("nan")
        )
        metrics.append(
            {
                "joint": float(j),
                "mse": mse,
                "std_true": std_true,
                "std_pred": std_pred,
                "corr": corr,
            }
        )
    return metrics


def print_metrics(title: str, metrics: List[Dict[str, float]]) -> None:
    print(title)
    for m in metrics:
        j = int(m["joint"])
        print(
            f"  Joint {j}: MSE={m['mse']:.6e}, std(true)={m['std_true']:.6e}, "
            f"std(pred)={m['std_pred']:.6e}, corr={m['corr']:.4f}"
        )


def plot_prediction_comparison(
    y_valid_full: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_physics: np.ndarray,
    out_path: Path,
    show_plot: bool,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    axes = axes.flatten()
    t = np.arange(y_valid_full.shape[0])
    n_joints = y_valid_full.shape[1]

    for j in range(n_joints):
        ax = axes[j]
        ax.plot(t, y_valid_full[:, j], label="true y", linewidth=1.0)
        ax.plot(t, y_pred_baseline[:, j], label="baseline (shared q+dq)", linewidth=1.0)
        ax.plot(t, y_pred_physics[:, j], label="physics (LuGre + residual NN)", linewidth=1.0)
        ax.set_title(f"Joint {j}")
        ax.grid(True)
        ax.legend(fontsize=8)

    for j in range(n_joints, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_learning_curve_comparison(
    baseline_history: Dict[str, list],
    physics_history: Dict[str, list],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(9, 5))
    plt.plot(baseline_history["train_loss"], label="baseline train")
    plt.plot(baseline_history["valid_loss"], label="baseline valid")
    plt.plot(physics_history["train_loss"], label="physics train (residual)")
    plt.plot(physics_history["valid_loss"], label="physics valid (residual)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized target space)")
    plt.title("Learning Curves: Baseline vs Physics-Informed")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_temporal_predictions(
    y_valid: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    show_plot: bool,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    axes = axes.flatten()
    t = np.arange(y_valid.shape[0])
    n_joints = y_valid.shape[1]

    for j in range(n_joints):
        ax = axes[j]
        ax.plot(t, y_valid[:, j], label="true y", linewidth=1.0)
        ax.plot(t, y_pred[:, j], label="pred y", linewidth=1.0)
        ax.set_title(f"Joint {j}")
        ax.grid(True)
        ax.legend(fontsize=8)

    for j in range(n_joints, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(out_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_temporal_training_history(
    history: Dict[str, list],
    out_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(9, 5))
    plt.plot(history["train_epochs"], history["train_loss"], label="train")
    plt.plot(history["valid_epochs"], history["valid_loss"], label="valid")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized target space)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_temporal_training(
    args: argparse.Namespace,
    q_train: np.ndarray,
    dq_train: np.ndarray,
    y_train: np.ndarray,
    q_valid: np.ndarray,
    dq_valid: np.ndarray,
    y_valid: np.ndarray,
    train_batch_size: int,
    eval_batch_size: int,
    device: torch.device,
    output_dir: Path,
    plots_dir: Path,
    model_dir: Path,
) -> None:
    x_train_step = build_temporal_features(q_train, dq_train)
    x_valid_step = build_temporal_features(q_valid, dq_valid)

    temporal_stats = compute_normalization(x_train_step, y_train)
    x_train_n, y_train_n = normalize(x_train_step, y_train, temporal_stats)
    x_valid_n, y_valid_n = normalize(x_valid_step, y_valid, temporal_stats)

    temporal_train_loader, temporal_valid_loader = build_windowed_loaders(
        x_train_n=x_train_n,
        y_train_n=y_train_n,
        x_valid_n=x_valid_n,
        y_valid_n=y_valid_n,
        window=args.window,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    temporal_model = build_temporal_model(
        args=args,
        input_dim=x_train_step.shape[1],
        output_dim=y_train.shape[1],
    ).to(device)

    if args.compile_model and hasattr(torch, "compile"):
        temporal_model = torch.compile(temporal_model)
        print(f"{args.arch.upper()} model compiled with torch.compile")

    history, best_valid, best_epoch, stop_epoch = train_temporal_model(
        model=temporal_model,
        train_loader=temporal_train_loader,
        valid_loader=temporal_valid_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        eval_every=args.eval_every,
        early_stop_patience_checks=args.early_stop_patience_checks,
        tag=args.arch,
    )

    y_pred_n, y_true_n = predict_from_loader(
        model=temporal_model,
        loader=temporal_valid_loader,
        device=device,
    )
    y_pred = (y_pred_n * temporal_stats["y_std"] + temporal_stats["y_mean"]).astype(np.float64)
    y_true = (y_true_n * temporal_stats["y_std"] + temporal_stats["y_mean"]).astype(np.float64)

    metrics = compute_joint_metrics(y_true, y_pred)
    print_metrics(f"Validation metrics per joint ({args.arch.upper()}):", metrics)
    global_mse = float(np.mean((y_true - y_pred) ** 2))
    print(f"Global validation MSE {args.arch.upper()}: {global_mse:.6e}")

    pred_plot_path = plots_dir / f"{args.arch}_validation_predictions.png"
    history_plot_path = plots_dir / f"{args.arch}_training_history.png"
    checkpoint_path = model_dir / f"friction_{args.arch}.pt"
    summary_path = output_dir / f"{args.arch}_run_summary.json"

    plot_temporal_predictions(
        y_valid=y_true,
        y_pred=y_pred,
        out_path=pred_plot_path,
        show_plot=args.show_plot,
        title=f"Validation Predictions ({args.arch.upper()})",
    )
    plot_temporal_training_history(
        history=history,
        out_path=history_plot_path,
        title=f"Learning Curve ({args.arch.upper()})",
    )

    torch.save(
        {
            "model_state_dict": temporal_model.state_dict(),
            "architecture": args.arch,
            "inputs": ["q_state_0..6", "dq_state_0..6", "sign(dq_state_0..6)"],
            "target": "tau_ext_0..6",
            "window": args.window,
            "x_mean": temporal_stats["x_mean"],
            "x_std": temporal_stats["x_std"],
            "y_mean": temporal_stats["y_mean"],
            "y_std": temporal_stats["y_std"],
            "train_csv": args.train_csv,
            "valid_csv": args.valid_csv,
            "model_args": {
                "tcn_channels": args.tcn_channels,
                "tcn_layers": args.tcn_layers,
                "tcn_kernel_size": args.tcn_kernel_size,
                "tcn_dropout": args.tcn_dropout,
                "head_hidden": args.head_hidden,
                "mamba_d_model": args.mamba_d_model,
                "mamba_d_state": args.mamba_d_state,
                "mamba_d_conv": args.mamba_d_conv,
                "mamba_expand": args.mamba_expand,
                "mamba_layers": args.mamba_layers,
                "mamba_dropout": args.mamba_dropout,
            },
        },
        checkpoint_path,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "architecture": args.arch,
                "window": args.window,
                "global_mse": global_mse,
                "metrics": metrics,
                "epochs_requested": args.epochs,
                "best_valid_loss_norm": best_valid,
                "best_epoch": best_epoch,
                "stop_epoch": stop_epoch,
                "eval_every": args.eval_every,
                "early_stop_patience_checks": args.early_stop_patience_checks,
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "checkpoint": str(checkpoint_path),
                "predictions_plot": str(pred_plot_path),
                "training_history_plot": str(history_plot_path),
            },
            f,
            indent=2,
        )

    print(f"Saved validation prediction plot: {pred_plot_path}")
    print(f"Saved training history plot: {history_plot_path}")
    print(f"Saved {args.arch.upper()} checkpoint: {checkpoint_path}")
    print(f"Saved {args.arch.upper()} run summary: {summary_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    train_csv = Path(args.train_csv)
    valid_csv = Path(args.valid_csv)

    if not train_csv.is_absolute():
        train_csv = script_dir / train_csv
    if not valid_csv.is_absolute():
        valid_csv = script_dir / valid_csv

    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    model_dir = Path(args.model_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    if not plots_dir.is_absolute():
        plots_dir = script_dir / plots_dir
    if not model_dir.is_absolute():
        model_dir = script_dir / model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    torch.set_float32_matmul_precision("high")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    q_train, dq_train, y_train = load_state_to_friction_data(str(train_csv))
    q_valid, dq_valid, y_valid = load_state_to_friction_data(str(valid_csv))

    q_train, dq_train, y_train = drop_nonfinite_rows(
        q=q_train,
        dq=dq_train,
        tau=y_train,
        dataset_name="train",
        csv_path=train_csv,
    )
    q_valid, dq_valid, y_valid = drop_nonfinite_rows(
        q=q_valid,
        dq=dq_valid,
        tau=y_valid,
        dataset_name="validation",
        csv_path=valid_csv,
    )

    print(f"q_train shape: {q_train.shape}, dq_train shape: {dq_train.shape}, y_train: {y_train.shape}")
    print(f"q_valid shape: {q_valid.shape}, dq_valid shape: {dq_valid.shape}, y_valid: {y_valid.shape}")

    print_distribution_diagnostics(dq_train, dq_valid, y_train, y_valid)

    n_train = q_train.shape[0]
    n_valid = q_valid.shape[0]
    train_batch_size = resolve_batch_size(args.batch_size, n_train)
    eval_batch_size = resolve_batch_size(args.eval_batch_size, n_valid)
    print(f"Training batch size: {train_batch_size} (requested={args.batch_size})")
    print(f"Eval batch size: {eval_batch_size} (requested={args.eval_batch_size})")

    if args.arch in {"tcn", "mamba"}:
        print(f"Architecture: {args.arch.upper()} | window={args.window}")
        run_temporal_training(
            args=args,
            q_train=q_train,
            dq_train=dq_train,
            y_train=y_train,
            q_valid=q_valid,
            dq_valid=dq_valid,
            y_valid=y_valid,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            device=device,
            output_dir=output_dir,
            plots_dir=plots_dir,
            model_dir=model_dir,
        )
        return

    model_config_path = Path(args.model_config)
    if not model_config_path.is_absolute():
        model_config_path = script_dir / model_config_path
    model_cfg = load_model_config(model_config_path)
    print(
        "Model config: "
        f"hidden_dims={model_cfg['hidden_dims']}, "
        f"head_hidden_dims={model_cfg['head_hidden_dims']}, "
        f"physics_hidden_dims={model_cfg['physics_hidden_dims']}, "
        f"activation={model_cfg['activation']}"
    )

    x_train_baseline = build_baseline_features(q_train, dq_train)
    x_valid_baseline = build_baseline_features(q_valid, dq_valid)

    baseline_stats = compute_normalization(x_train_baseline, y_train)
    x_train_baseline_n, y_train_baseline_n = normalize(
        x_train_baseline,
        y_train,
        baseline_stats,
    )
    x_valid_baseline_n, y_valid_baseline_n = normalize(
        x_valid_baseline,
        y_valid,
        baseline_stats,
    )

    baseline_train_loader, baseline_valid_loader = build_data_loaders(
        x_train_n=x_train_baseline_n,
        y_train_n=y_train_baseline_n,
        x_valid_n=x_valid_baseline_n,
        y_valid_n=y_valid_baseline_n,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    baseline_model = FrictionMLP(
        input_dim=x_train_baseline.shape[1],
        hidden_dims=model_cfg["hidden_dims"],
        head_hidden_dims=model_cfg["head_hidden_dims"],
        output_dim=y_train.shape[1],
        activation=model_cfg["activation"],
    ).to(device)

    if args.compile_model and hasattr(torch, "compile"):
        baseline_model = torch.compile(baseline_model)
        print("Baseline model compiled with torch.compile")

    baseline_history = train_model(
        model=baseline_model,
        train_loader=baseline_train_loader,
        valid_loader=baseline_valid_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tag="baseline",
    )

    y_pred_baseline_n = batched_predict(
        baseline_model,
        torch.from_numpy(x_valid_baseline_n),
        device=device,
        batch_size=eval_batch_size,
    )
    y_pred_baseline = (
        y_pred_baseline_n * baseline_stats["y_std"] + baseline_stats["y_mean"]
    ).astype(np.float64)

    print("Fitting LuGre/Stribeck analytical pre-model per joint...")
    lugre_params = fit_lugre_bundle(
        dq_train=dq_train,
        tau_train=y_train,
        vs_grid_size=args.lugre_vs_grid_size,
    )
    y_train_lugre = predict_lugre(dq_train, lugre_params)
    y_valid_lugre = predict_lugre(dq_valid, lugre_params)

    y_train_residual = y_train - y_train_lugre
    y_valid_residual = y_valid - y_valid_lugre

    x_train_physics = build_physics_features(
        q=q_train,
        dq=dq_train,
        include_q=args.physics_include_q,
    )
    x_valid_physics = build_physics_features(
        q=q_valid,
        dq=dq_valid,
        include_q=args.physics_include_q,
    )

    physics_stats = compute_normalization(x_train_physics, y_train_residual)
    x_train_physics_n, y_train_residual_n = normalize(
        x_train_physics,
        y_train_residual,
        physics_stats,
    )
    x_valid_physics_n, y_valid_residual_n = normalize(
        x_valid_physics,
        y_valid_residual,
        physics_stats,
    )

    physics_train_loader, physics_valid_loader = build_data_loaders(
        x_train_n=x_train_physics_n,
        y_train_n=y_train_residual_n,
        x_valid_n=x_valid_physics_n,
        y_valid_n=y_valid_residual_n,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    features_per_joint = 3 if args.physics_include_q else 2
    physics_model = IndependentJointMLP(
        num_joints=y_train.shape[1],
        features_per_joint=features_per_joint,
        hidden_dims=model_cfg["physics_hidden_dims"],
        activation=model_cfg["activation"],
    ).to(device)

    if args.compile_model and hasattr(torch, "compile"):
        physics_model = torch.compile(physics_model)
        print("Physics model compiled with torch.compile")

    physics_history = train_model(
        model=physics_model,
        train_loader=physics_train_loader,
        valid_loader=physics_valid_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tag="physics",
    )

    y_pred_residual_n = batched_predict(
        physics_model,
        torch.from_numpy(x_valid_physics_n),
        device=device,
        batch_size=eval_batch_size,
    )
    y_pred_residual = (
        y_pred_residual_n * physics_stats["y_std"] + physics_stats["y_mean"]
    ).astype(np.float64)

    y_pred_physics = (y_valid_lugre.astype(np.float64) + y_pred_residual).astype(np.float64)

    baseline_metrics = compute_joint_metrics(y_valid.astype(np.float64), y_pred_baseline)
    physics_metrics = compute_joint_metrics(y_valid.astype(np.float64), y_pred_physics)

    print_metrics("Validation metrics per joint (baseline shared-trunk):", baseline_metrics)
    print_metrics("Validation metrics per joint (physics-informed residual):", physics_metrics)

    baseline_mse = float(np.mean((y_valid.astype(np.float64) - y_pred_baseline) ** 2))
    physics_mse = float(np.mean((y_valid.astype(np.float64) - y_pred_physics) ** 2))
    print(f"Global validation MSE baseline: {baseline_mse:.6e}")
    print(f"Global validation MSE physics : {physics_mse:.6e}")

    pred_plot_path = plots_dir / "nn_validation_predictions_comparison.png"
    history_plot_path = plots_dir / "nn_training_history_comparison.png"
    baseline_model_path = model_dir / "friction_mlp.pt"
    physics_model_path = model_dir / "friction_mlp_physics_residual.pt"
    summary_path = output_dir / "nn_comparison_summary.json"

    plot_prediction_comparison(
        y_valid_full=y_valid.astype(np.float64),
        y_pred_baseline=y_pred_baseline,
        y_pred_physics=y_pred_physics,
        out_path=pred_plot_path,
        show_plot=args.show_plot,
    )
    plot_learning_curve_comparison(
        baseline_history=baseline_history,
        physics_history=physics_history,
        out_path=history_plot_path,
    )

    torch.save(
        {
            "model_state_dict": baseline_model.state_dict(),
            "inputs": ["q_state_0..6", "dq_state_0..6"],
            "target": "tau_ext_0..6",
            "architecture": "shared_trunk_multi_head",
            "hidden_dims": model_cfg["hidden_dims"],
            "head_hidden_dims": model_cfg["head_hidden_dims"],
            "activation": model_cfg["activation"],
            "x_mean": baseline_stats["x_mean"],
            "x_std": baseline_stats["x_std"],
            "y_mean": baseline_stats["y_mean"],
            "y_std": baseline_stats["y_std"],
            "train_csv": str(train_csv),
            "valid_csv": str(valid_csv),
            "model_config": str(model_config_path),
        },
        baseline_model_path,
    )

    torch.save(
        {
            "model_state_dict": physics_model.state_dict(),
            "inputs": ["dq_state_0..6", "sign(dq_state_0..6)"]
            if not args.physics_include_q
            else ["q_state_0..6", "dq_state_0..6", "sign(dq_state_0..6)"],
            "target": "tau_ext_0..6",
            "target_mode": "residual_over_lugre",
            "architecture": "independent_joint_mlp",
            "physics_hidden_dims": model_cfg["physics_hidden_dims"],
            "features_per_joint": features_per_joint,
            "activation": model_cfg["activation"],
            "x_mean": physics_stats["x_mean"],
            "x_std": physics_stats["x_std"],
            "y_mean": physics_stats["y_mean"],
            "y_std": physics_stats["y_std"],
            "lugre_params": {
                "Fc": lugre_params["Fc"],
                "Fv": lugre_params["Fv"],
                "Fs": lugre_params["Fs"],
                "vs": lugre_params["vs"],
            },
            "train_csv": str(train_csv),
            "valid_csv": str(valid_csv),
            "model_config": str(model_config_path),
            "physics_include_q": args.physics_include_q,
        },
        physics_model_path,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_global_mse": baseline_mse,
                "physics_global_mse": physics_mse,
                "baseline_metrics": baseline_metrics,
                "physics_metrics": physics_metrics,
                "batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "epochs": args.epochs,
                "physics_include_q": args.physics_include_q,
                "lugre_vs_grid_size": args.lugre_vs_grid_size,
                "baseline_checkpoint": str(baseline_model_path),
                "physics_checkpoint": str(physics_model_path),
                "predictions_plot": str(pred_plot_path),
                "learning_curves_plot": str(history_plot_path),
            },
            f,
            indent=2,
        )

    print(f"Saved prediction comparison plot: {pred_plot_path}")
    print(f"Saved learning-curve comparison plot: {history_plot_path}")
    print(f"Saved baseline model checkpoint: {baseline_model_path}")
    print(f"Saved physics model checkpoint: {physics_model_path}")
    print(f"Saved comparison summary: {summary_path}")


if __name__ == "__main__":
    main()