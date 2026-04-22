#!/usr/bin/env python3
"""Train a sparse variational multitask GP friction predictor (GPyTorch).

This script implements an SVGP pipeline tailored to friction torque prediction:
  - optional physics-informed feature augmentation (including sign(dq))
  - k-means inducing-point initialization
  - independent multitask GP (7 joints)
  - Matérn-5/2 kernel with ARD
  - ELBO training with minibatches
  - uncertainty calibration checks and +-2sigma plots
  - deployment-friendly prediction path with fast_pred_var()
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset

try:
    import gpytorch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gpytorch is required. Install with: pip install gpytorch"
    ) from exc

try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tqdm is required for progress bars. Install with: pip install tqdm"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SVGP multitask friction model with uncertainty estimates."
    )
    parser.add_argument("--train-csv", type=str, default="tau_log_train.csv")
    parser.add_argument("--valid-csv", type=str, default="tau_log_validation.csv")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gpr_results",
        help="Directory for non-plot GPR artifacts (checkpoint, metadata, exports).",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory for saved plots.",
    )

    parser.add_argument(
        "--feature-mode",
        type=str,
        default="sign_augmented",
        choices=["sign_augmented", "dq_warped"],
        help=(
            "Feature construction mode: sign_augmented -> [q, dq, sign(dq)] (21 dims), "
            "dq_warped -> [q, tanh(dq/scale)] (14 dims)."
        ),
    )
    parser.add_argument(
        "--dq-warp-scale",
        type=float,
        default=0.5,
        help="Scale parameter for tanh(dq / dq_warp_scale) when feature-mode=dq_warped.",
    )

    parser.add_argument(
        "--num-inducing",
        type=int,
        default=500,
        help="Number of inducing points (k in k-means and m in SVGP).",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--noise-lower-bound",
        type=float,
        default=1e-5,
        help="Lower bound for likelihood noise in normalized target space.",
    )
    parser.add_argument(
        "--noise-upper-bound",
        type=float,
        default=1.0,
        help="Upper bound for likelihood noise in normalized target space.",
    )

    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ordered_cols(columns: pd.Index, pattern: str) -> List[str]:
    cols = [c for c in columns if c.startswith(pattern)]
    cols.sort(key=lambda c: int(c.split("_")[-1]))
    return cols


def load_state_to_friction_data(
    csv_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return (
        q_full.astype(np.float32),
        dq_full.astype(np.float32),
        tau_ext.astype(np.float32),
    )


def build_features(
    q: np.ndarray,
    dq: np.ndarray,
    feature_mode: str,
    dq_warp_scale: float,
) -> np.ndarray:
    if feature_mode == "sign_augmented":
        sign_dq = np.sign(dq)
        x = np.hstack([q, dq, sign_dq]).astype(np.float32)
    else:
        if dq_warp_scale <= 0.0:
            raise ValueError("dq_warp_scale must be > 0.")
        dq_warped = np.tanh(dq / float(dq_warp_scale))
        x = np.hstack([q, dq_warped]).astype(np.float32)
    return x


def compute_normalization(
    x_train: np.ndarray,
    y_train: np.ndarray,
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
    x: np.ndarray,
    y: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    x_n = (x - stats["x_mean"]) / stats["x_std"]
    y_n = (y - stats["y_mean"]) / stats["y_std"]
    return x_n.astype(np.float32), y_n.astype(np.float32)


def resolve_batch_size(requested: int, n_samples: int) -> int:
    if requested <= 0:
        return n_samples
    return min(requested, n_samples)


def initialize_inducing_points(
    x_train_n: np.ndarray,
    num_inducing: int,
    num_tasks: int,
    seed: int,
) -> torch.Tensor:
    if num_inducing <= 0:
        raise ValueError("num_inducing must be > 0")

    n_samples = x_train_n.shape[0]
    k = min(num_inducing, n_samples)
    if k < num_inducing:
        print(
            f"Requested {num_inducing} inducing points but only {n_samples} samples exist; "
            f"using {k}."
        )

    # MiniBatchKMeans is robust and efficient for ~O(1e5) sample scales.
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(8192, max(512, n_samples // 8)),
        n_init=10,
    )
    kmeans.fit(x_train_n)
    centers = kmeans.cluster_centers_.astype(np.float32)

    inducing = np.repeat(centers[None, :, :], repeats=num_tasks, axis=0)
    return torch.from_numpy(inducing)


class IndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
    """Independent SVGP heads, one per joint, with shared feature space."""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        input_dim: int,
        num_tasks: int,
    ) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=torch.Size([num_tasks]),
        )
        base_variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base_variational_strategy,
            num_tasks=num_tasks,
        )
        super().__init__(variational_strategy)

        batch_shape = torch.Size([num_tasks])
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=input_dim,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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


@torch.no_grad()
def predict_distribution(
    model: IndependentMultitaskGPModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    x_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
    use_fast_pred_var: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    likelihood.eval()

    means: List[torch.Tensor] = []
    stds: List[torch.Tensor] = []

    fast_pred_ctx = gpytorch.settings.fast_pred_var() if use_fast_pred_var else None
    if fast_pred_ctx is None:
        context = torch.no_grad()
    else:
        context = fast_pred_ctx

    with context:
        for start in range(0, x_tensor.shape[0], batch_size):
            xb = x_tensor[start : start + batch_size].to(device, non_blocking=True)
            pred = likelihood(model(xb))
            means.append(pred.mean.cpu())
            stds.append(pred.stddev.cpu())

    mean_all = torch.cat(means, dim=0).numpy()
    std_all = torch.cat(stds, dim=0).numpy()
    return mean_all, std_all


@torch.no_grad()
def evaluate_elbo(
    model: IndependentMultitaskGPModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    loader: DataLoader,
    mll: gpytorch.mlls.VariationalELBO,
    device: torch.device,
) -> float:
    model.eval()
    likelihood.eval()

    total = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        output = model(xb)
        loss = -mll(output, yb)

        bs = xb.shape[0]
        total += float(loss.item()) * bs
        count += bs

    return total / max(count, 1)


def train_svgp(
    model: IndependentMultitaskGPModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    num_data: int,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=lr,
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)

    history: Dict[str, List[float]] = {"train_loss": [], "valid_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        likelihood.train()

        running = 0.0
        n_seen = 0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for xb, yb in batch_bar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            running += float(loss.item()) * bs
            n_seen += bs
            batch_bar.set_postfix(loss=f"{loss.item():.4e}")

        train_loss = running / max(n_seen, 1)
        valid_loss = evaluate_elbo(model, likelihood, valid_loader, mll, device)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train_neg_elbo={train_loss:.6e} | valid_neg_elbo={valid_loss:.6e}"
            )

    return history


def compute_metrics_and_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> Tuple[List[Dict[str, float]], float]:
    metrics: List[Dict[str, float]] = []

    inside = np.logical_and(
        y_true >= (y_pred - 2.0 * y_std),
        y_true <= (y_pred + 2.0 * y_std),
    )
    global_coverage = float(np.mean(inside))

    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        ys = y_std[:, j]

        mse = float(np.mean((yt - yp) ** 2))
        std_true = float(np.std(yt))
        std_pred = float(np.std(yp))
        corr = (
            float(np.corrcoef(yt, yp)[0, 1])
            if std_true > 0.0 and std_pred > 0.0
            else float("nan")
        )
        cov_2sigma = float(np.mean(np.logical_and(yt >= yp - 2.0 * ys, yt <= yp + 2.0 * ys)))
        metrics.append(
            {
                "joint": float(j),
                "mse": mse,
                "std_true": std_true,
                "std_pred": std_pred,
                "corr": corr,
                "coverage_2sigma": cov_2sigma,
            }
        )

    return metrics, global_coverage


def print_metrics(metrics: List[Dict[str, float]], global_coverage: float) -> None:
    print("Validation metrics per joint:")
    for m in metrics:
        j = int(m["joint"])
        print(
            f"  Joint {j}: MSE={m['mse']:.6e}, std(true)={m['std_true']:.6e}, "
            f"std(pred)={m['std_pred']:.6e}, corr={m['corr']:.4f}, "
            f"coverage(+-2sigma)={m['coverage_2sigma']:.3f}"
        )

    print(f"Global coverage (+-2sigma): {global_coverage:.3f}")
    if global_coverage < 0.90:
        print(
            "Calibration note: uncertainty bands are too narrow. "
            "Try increasing --noise-upper-bound or increasing --num-inducing."
        )
    elif global_coverage > 0.98:
        print(
            "Calibration note: uncertainty bands may be too wide. "
            "Try reducing --noise-upper-bound or tightening --noise-lower-bound."
        )
    else:
        print("Calibration note: coverage is in a good range near the 95% target.")


def plot_predictions_with_uncertainty(
    y_valid: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    out_path: Path,
    show_plot: bool,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    axes = axes.flatten()
    t = np.arange(y_valid.shape[0])
    n_joints = y_valid.shape[1]

    for j in range(n_joints):
        ax = axes[j]
        mean_j = y_pred[:, j]
        std_j = y_std[:, j]
        lower = mean_j - 2.0 * std_j
        upper = mean_j + 2.0 * std_j

        ax.plot(t, y_valid[:, j], label="true y", linewidth=1.0)
        ax.plot(t, mean_j, label="pred mean", linewidth=1.0)
        ax.fill_between(
            t,
            lower,
            upper,
            alpha=0.2,
            label="pred +-2sigma",
        )
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


def plot_training_history(history: Dict[str, List[float]], out_path: Path) -> None:
    fig = plt.figure(figsize=(9, 5))
    plt.plot(history["train_loss"], label="train neg-ELBO")
    plt.plot(history["valid_loss"], label="valid neg-ELBO")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SVGP Training History")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def extract_variational_export(
    model: IndependentMultitaskGPModel,
) -> Dict[str, np.ndarray]:
    state = model.state_dict()
    export_dict: Dict[str, np.ndarray] = {}

    for key, val in state.items():
        if (
            "variational" in key
            or "inducing_points" in key
            or "covar_module" in key
            or "mean_module" in key
        ):
            export_dict[key] = val.detach().cpu().numpy()

    return export_dict


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    train_csv = Path(args.train_csv)
    valid_csv = Path(args.valid_csv)
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)

    if not train_csv.is_absolute():
        train_csv = script_dir / train_csv
    if not valid_csv.is_absolute():
        valid_csv = script_dir / valid_csv
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    if not plots_dir.is_absolute():
        plots_dir = script_dir / plots_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    q_train, dq_train, y_train = load_state_to_friction_data(str(train_csv))
    q_valid, dq_valid, y_valid = load_state_to_friction_data(str(valid_csv))

    x_train = build_features(
        q=q_train,
        dq=dq_train,
        feature_mode=args.feature_mode,
        dq_warp_scale=args.dq_warp_scale,
    )
    x_valid = build_features(
        q=q_valid,
        dq=dq_valid,
        feature_mode=args.feature_mode,
        dq_warp_scale=args.dq_warp_scale,
    )

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")

    stats = compute_normalization(x_train, y_train)
    x_train_n, y_train_n = normalize(x_train, y_train, stats)
    x_valid_n, y_valid_n = normalize(x_valid, y_valid, stats)

    n_train, input_dim = x_train_n.shape
    num_tasks = y_train_n.shape[1]
    train_batch_size = resolve_batch_size(args.batch_size, n_train)
    eval_batch_size = resolve_batch_size(args.eval_batch_size, x_valid_n.shape[0])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(
        f"Feature mode: {args.feature_mode}, input_dim={input_dim}, "
        f"num_tasks={num_tasks}, num_inducing={args.num_inducing}"
    )

    inducing_points = initialize_inducing_points(
        x_train_n=x_train_n,
        num_inducing=args.num_inducing,
        num_tasks=num_tasks,
        seed=args.seed,
    ).to(device)

    model = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        input_dim=input_dim,
        num_tasks=num_tasks,
    ).to(device)

    noise_constraint = gpytorch.constraints.Interval(
        lower_bound=args.noise_lower_bound,
        upper_bound=args.noise_upper_bound,
    )
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        noise_constraint=noise_constraint,
    ).to(device)

    train_loader, valid_loader = build_data_loaders(
        x_train_n=x_train_n,
        y_train_n=y_train_n,
        x_valid_n=x_valid_n,
        y_valid_n=y_valid_n,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    history = train_svgp(
        model=model,
        likelihood=likelihood,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        num_data=n_train,
    )

    mean_valid_n, std_valid_n = predict_distribution(
        model=model,
        likelihood=likelihood,
        x_tensor=torch.from_numpy(x_valid_n),
        device=device,
        batch_size=eval_batch_size,
        use_fast_pred_var=True,
    )

    y_pred = (mean_valid_n * stats["y_std"] + stats["y_mean"]).astype(np.float64)
    y_std = (std_valid_n * stats["y_std"]).astype(np.float64)

    metrics, global_coverage = compute_metrics_and_coverage(
        y_true=y_valid.astype(np.float64),
        y_pred=y_pred,
        y_std=y_std,
    )
    print_metrics(metrics, global_coverage)

    pred_plot_path = plots_dir / "gpr_svgp_validation_predictions_uncertainty.png"
    history_plot_path = plots_dir / "gpr_svgp_training_history.png"
    checkpoint_path = output_dir / "gpr_svgp_matern52_ard.pt"
    variational_export_path = output_dir / "gpr_svgp_variational_export.joblib"
    summary_path = output_dir / "gpr_svgp_run_meta.json"

    plot_predictions_with_uncertainty(
        y_valid=y_valid.astype(np.float64),
        y_pred=y_pred,
        y_std=y_std,
        out_path=pred_plot_path,
        show_plot=args.show_plot,
    )
    plot_training_history(history, history_plot_path)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "inputs": (
                ["q_state_0..6", "dq_state_0..6", "sign(dq_state_0..6)"]
                if args.feature_mode == "sign_augmented"
                else ["q_state_0..6", "tanh(dq_state_0..6 / dq_warp_scale)"]
            ),
            "target": "tau_ext_0..6",
            "architecture": "svgp_independent_multitask",
            "kernel": "ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=input_dim))",
            "num_inducing": int(inducing_points.shape[1]),
            "x_mean": stats["x_mean"],
            "x_std": stats["x_std"],
            "y_mean": stats["y_mean"],
            "y_std": stats["y_std"],
            "feature_mode": args.feature_mode,
            "dq_warp_scale": float(args.dq_warp_scale),
            "train_csv": str(train_csv),
            "valid_csv": str(valid_csv),
        },
        checkpoint_path,
    )

    variational_export = extract_variational_export(model)
    joblib.dump(variational_export, variational_export_path)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_csv": str(train_csv),
                "valid_csv": str(valid_csv),
                "feature_mode": args.feature_mode,
                "dq_warp_scale": float(args.dq_warp_scale),
                "input_dim": int(input_dim),
                "num_tasks": int(num_tasks),
                "num_inducing": int(inducing_points.shape[1]),
                "epochs": int(args.epochs),
                "batch_size": int(train_batch_size),
                "eval_batch_size": int(eval_batch_size),
                "lr": float(args.lr),
                "noise_lower_bound": float(args.noise_lower_bound),
                "noise_upper_bound": float(args.noise_upper_bound),
                "global_coverage_2sigma": float(global_coverage),
                "metrics": metrics,
                "checkpoint": str(checkpoint_path),
                "variational_export": str(variational_export_path),
                "predictions_plot": str(pred_plot_path),
                "training_history_plot": str(history_plot_path),
            },
            f,
            indent=2,
        )

    print("Saved artifacts:")
    print(f"  Prediction +-2sigma plot: {pred_plot_path}")
    print(f"  Training history plot: {history_plot_path}")
    print(f"  SVGP checkpoint: {checkpoint_path}")
    print(f"  Variational export: {variational_export_path}")
    print(f"  Run metadata: {summary_path}")
    print(
        "Deployment tip: use model.eval(), likelihood.eval(), torch.no_grad(), and "
        "gpytorch.settings.fast_pred_var() for low-latency inference."
    )


if __name__ == "__main__":
    main()
