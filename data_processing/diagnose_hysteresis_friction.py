#!/usr/bin/env python3
"""Quick hysteresis diagnostic for friction data.

This script plots tau_ext_j vs dq_j for one joint and colors points by sign(ddq_j),
where ddq_j is approximated with a numerical gradient.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tau_ext_j vs dq_j colored by sign(ddq_j) to diagnose hysteresis."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="tau_log_train.csv",
        help="CSV with dq_state_* and tau_ext_* columns.",
    )
    parser.add_argument(
        "--joint",
        type=int,
        default=-1,
        help="Joint index to inspect. Use -1 to plot all joints in one figure (default: -1).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Sample period used in np.gradient for ddq estimation (default: 1.0).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Scatter marker size (default: 1.0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Scatter transparency (default: 0.3).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of dq bins used for hysteresis score estimation (default: 40).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path. If omitted, saves to data_processing/plots/.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="",
        help="Optional JSON path for per-joint hysteresis summary.",
    )
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def _ordered_cols(columns: pd.Index, prefix: str) -> list[str]:
    cols = [c for c in columns if c.startswith(prefix)]
    cols.sort(key=lambda c: int(c.split("_")[-1]))
    return cols


def load_dq_tau(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(csv_path)

    dq_cols = _ordered_cols(data.columns, "dq_state_")
    tau_cols = _ordered_cols(data.columns, "tau_ext_")
    if len(dq_cols) == 0 or len(tau_cols) == 0:
        raise ValueError("CSV must contain dq_state_* and tau_ext_* columns.")

    dq = data[dq_cols].to_numpy(dtype=np.float64)
    tau = data[tau_cols].to_numpy(dtype=np.float64)

    if dq.shape != tau.shape:
        raise ValueError(f"Inconsistent shapes: dq={dq.shape}, tau={tau.shape}")

    return dq, tau


def resolve_csv_path(csv_arg: str, script_dir: Path) -> Path:
    csv_path = Path(csv_arg)
    if csv_path.is_absolute():
        return csv_path

    # First respect the caller's working directory, then fall back to script directory.
    cwd_candidate = csv_path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (script_dir / csv_path).resolve()


def compute_hysteresis_score(
    dq_j: np.ndarray,
    tau_j: np.ndarray,
    ddq_sign_j: np.ndarray,
    bins: int,
) -> Tuple[float, int]:
    """Compute a normalized loop-separation score for one joint.

    Score definition (heuristic):
      mean_bin(|mean(tau | ddq>0) - mean(tau | ddq<0)|) / std(tau)
    over dq bins that contain enough points in both acceleration directions.
    """
    if bins < 5:
        bins = 5

    dq_min = float(np.min(dq_j))
    dq_max = float(np.max(dq_j))
    if not np.isfinite(dq_min) or not np.isfinite(dq_max) or dq_max <= dq_min:
        return float("nan"), 0

    edges = np.linspace(dq_min, dq_max, bins + 1)
    bin_idx = np.digitize(dq_j, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)

    separations: List[float] = []
    for b in range(bins):
        m_acc = (bin_idx == b) & (ddq_sign_j > 0.0)
        m_dec = (bin_idx == b) & (ddq_sign_j < 0.0)
        if int(m_acc.sum()) < 5 or int(m_dec.sum()) < 5:
            continue
        sep = abs(float(np.mean(tau_j[m_acc])) - float(np.mean(tau_j[m_dec])))
        separations.append(sep)

    if len(separations) == 0:
        return float("nan"), 0

    tau_scale = float(np.std(tau_j))
    if tau_scale < 1e-9:
        return 0.0, len(separations)

    return float(np.mean(separations) / tau_scale), len(separations)


def classify_hysteresis(score: float) -> str:
    if not np.isfinite(score):
        return "insufficient overlap"
    if score < 0.10:
        return "low evidence"
    if score < 0.25:
        return "moderate evidence"
    return "strong evidence"


def resolve_output_paths(args: argparse.Namespace, script_dir: Path, all_joints: bool) -> Tuple[Path, Path]:
    if args.output.strip():
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / output_path
    else:
        output_dir = script_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        name = "hysteresis_diagnostic_all_joints.png" if all_joints else f"hysteresis_diagnostic_joint_{args.joint}.png"
        output_path = output_dir / name

    if args.summary_output.strip():
        summary_path = Path(args.summary_output)
        if not summary_path.is_absolute():
            summary_path = script_dir / summary_path
    else:
        summary_path = output_path.with_suffix(".json")

    return output_path, summary_path


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path = resolve_csv_path(args.csv, script_dir)

    dq, tau = load_dq_tau(csv_path)

    n_joints = dq.shape[1]
    j = int(args.joint)
    if j < -1 or j >= n_joints:
        raise ValueError(f"joint index out of range: {j}. Valid range is [-1, {n_joints - 1}].")
    if args.dt <= 0.0:
        raise ValueError("dt must be > 0.")
    if args.bins <= 1:
        raise ValueError("bins must be > 1.")

    all_joints = j == -1
    output_path, summary_path = resolve_output_paths(args, script_dir, all_joints)

    joints = list(range(n_joints)) if all_joints else [j]
    results: List[Dict[str, object]] = []

    if all_joints:
        cols = 4
        rows = int(math.ceil(n_joints / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.0 * rows))
        axes_flat = np.array(axes).reshape(-1)
    else:
        fig, ax_single = plt.subplots(figsize=(7.5, 5.5))
        axes_flat = np.array([ax_single])

    scatter_for_colorbar = None
    for idx, joint_id in enumerate(joints):
        ax = axes_flat[idx]
        ddq_j = np.gradient(dq[:, joint_id], args.dt)
        ddq_sign_j = np.sign(ddq_j)

        score, used_bins = compute_hysteresis_score(
            dq_j=dq[:, joint_id],
            tau_j=tau[:, joint_id],
            ddq_sign_j=ddq_sign_j,
            bins=args.bins,
        )
        label = classify_hysteresis(score)
        results.append(
            {
                "joint": joint_id,
                "score": None if not np.isfinite(score) else float(score),
                "classification": label,
                "bins_used": int(used_bins),
                "bins_requested": int(args.bins),
            }
        )

        sc = ax.scatter(
            dq[:, joint_id],
            tau[:, joint_id],
            c=ddq_sign_j,
            cmap="bwr",
            vmin=-1.0,
            vmax=1.0,
            s=float(args.point_size),
            alpha=float(args.alpha),
            linewidths=0.0,
        )
        if scatter_for_colorbar is None:
            scatter_for_colorbar = sc

        score_text = "n/a" if not np.isfinite(score) else f"{score:.3f}"
        ax.set_title(f"Joint {joint_id} | score={score_text} | {label}")
        ax.set_xlabel(f"dq_state_{joint_id}")
        ax.set_ylabel(f"tau_ext_{joint_id}")
        ax.grid(True, alpha=0.25)

    for idx in range(len(joints), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    if scatter_for_colorbar is not None:
        cbar = fig.colorbar(scatter_for_colorbar, ax=fig.axes, ticks=[-1, 0, 1], shrink=0.95)
        cbar.set_label("sign(ddq)")
        cbar.set_ticklabels(["-1 decelerating", "0", "+1 accelerating"])

    guide = (
        "How to read: strong overlap of red/blue clouds implies mostly memoryless friction; "
        "clear split/looping implies hysteresis (temporal effects). "
        "Score is normalized branch separation in dq bins."
    )
    fig.suptitle("Friction Hysteresis Diagnostic: tau_ext vs dq colored by sign(ddq)", y=0.98)
    fig.text(0.5, 0.01, guide, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.96])

    fig.savefig(output_path, dpi=180)
    print(f"Saved hysteresis diagnostic plot: {output_path}")

    summary = {
        "csv": str(csv_path),
        "dt": float(args.dt),
        "bins": int(args.bins),
        "score_definition": "mean_bin(|mean(tau|ddq>0)-mean(tau|ddq<0)|)/std(tau)",
        "interpretation": {
            "low_evidence": "score < 0.10",
            "moderate_evidence": "0.10 <= score < 0.25",
            "strong_evidence": "score >= 0.25",
            "insufficient_overlap": "not enough bins populated in both acceleration directions",
        },
        "joint_results": results,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved hysteresis summary: {summary_path}")

    print("\nInterpretation guide:")
    print("  score < 0.10  -> low evidence of hysteresis")
    print("  0.10-0.25     -> moderate evidence")
    print("  >= 0.25       -> strong evidence")
    print("  n/a           -> insufficient overlap for reliable estimate")
    print("\nPer-joint results:")
    for r in results:
        score = r["score"]
        score_text = "n/a" if score is None else f"{float(score):.3f}"
        print(
            f"  joint {int(r['joint'])}: score={score_text}, "
            f"classification={r['classification']}, bins_used={int(r['bins_used'])}/{int(r['bins_requested'])}"
        )

    if args.show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
