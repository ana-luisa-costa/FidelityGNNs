"""Plot Fidelity+ and 1-Fidelity- together for one dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/fidelity_prob_5pairs_3seeds_20260717_115219")
DEFAULT_OUTPUT = Path("output/cross_datasets/5pair")
METHODS = ("gradient", "ig", "lime", "shap", "gnnexplainer", "pgmexplainer")
METHOD_LABELS = {
    "gradient": "Gradient",
    "ig": "IG",
    "lime": "LIME",
    "shap": "SHAP",
    "gnnexplainer": "GNNExplainer",
    "pgmexplainer": "PGMExplainer",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Probability Fidelity+ and 1 - Probability Fidelity- on the same small-multiple panels."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Run output folder.")
    parser.add_argument("--dataset", default="BPIC13_O", help="Dataset folder to plot.")
    parser.add_argument(
        "--task",
        default="activity",
        help="Task to plot: activity, resource, role, lifecycle, or a canonical task.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output folder.")
    parser.add_argument("--formats", default="png", help="Comma-separated output formats, e.g. png,pdf.")
    return parser.parse_args()


def _resolve_task(tasks: Sequence[str], requested: str) -> str | None:
    requested = str(requested).strip()
    if requested in tasks:
        return requested
    needles = {
        "resource": "org_resource",
        "role": "org_role",
        "lifecycle": "lifecycle_transition",
        "lifecycle_transition": "lifecycle_transition",
    }.get(requested.lower())
    if needles is None:
        return None
    matches = [task for task in tasks if needles in task.lower()]
    return sorted(matches)[0] if matches else None


def _read_summary(path: Path, task: str) -> pd.DataFrame:
    required = {
        "explainer",
        "task",
        "top_k",
        "mean_fid_prob_plus",
        "mean_fid_prob_minus",
    }
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    available_tasks = sorted(df["task"].astype(str).unique())
    resolved_task = _resolve_task(available_tasks, task)
    if resolved_task is None:
        raise ValueError(f"{path} has no rows for task {task!r}; available={available_tasks}")
    df = df[df["task"].astype(str) == resolved_task].copy()
    task_label = resolved_task
    if df.empty:
        raise ValueError(f"{path} has no rows for task {task!r}")

    for column in (
        "top_k",
        "mean_fid_prob_plus",
        "mean_fid_prob_minus",
        "sem_fid_prob_plus",
        "sem_fid_prob_minus",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if not np.isfinite(df[["top_k", "mean_fid_prob_plus", "mean_fid_prob_minus"]]).all().all():
        raise ValueError(f"{path} contains non-finite fidelity values")
    df["mean_one_minus_fid_prob_minus"] = 1.0 - df["mean_fid_prob_minus"]
    print(f"Plotting task -> {task_label}")
    return df.sort_values(["task", "explainer", "top_k"])


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#344054",
            "axes.grid": True,
            "grid.color": "#e4e7ec",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def _ordered_methods(df: pd.DataFrame) -> Sequence[str]:
    available = set(df["explainer"].astype(str))
    ordered = [method for method in METHODS if method in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def plot_crossing(df: pd.DataFrame, dataset: str, out_dir: Path, formats: Sequence[str]) -> None:
    methods = _ordered_methods(df)
    cols = 3
    rows = math.ceil(len(methods) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15.5, 4.2 * rows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    top_k = np.sort(df["top_k"].unique())

    for ax, method in zip(axes_flat, methods):
        part = df[df["explainer"] == method].sort_values("top_k")
        x = part["top_k"].to_numpy(dtype=float)
        y_plus = part["mean_fid_prob_plus"].to_numpy(dtype=float)
        y_sufficiency = part["mean_one_minus_fid_prob_minus"].to_numpy(dtype=float)

        ax.plot(
            x,
            y_plus,
            color="#1f77b4",
            marker="o",
            linewidth=2.2,
            label="Probability Fidelity+",
        )
        ax.plot(
            x,
            y_sufficiency,
            color="#2ca02c",
            marker="s",
            linewidth=2.2,
            linestyle="--",
            label="1 - Probability Fidelity-",
        )

        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xticks(top_k)

    plotted = np.concatenate([
        df["mean_fid_prob_plus"].to_numpy(dtype=float),
        df["mean_one_minus_fid_prob_minus"].to_numpy(dtype=float),
    ])
    lo = min(0.0, float(np.nanmin(plotted)))
    hi = max(0.0, float(np.nanmax(plotted)))
    pad = max(0.02, 0.08 * (hi - lo))
    for ax in axes_flat[:len(methods)]:
        ax.set_ylim(lo - pad, hi + pad)

    for ax in axes_flat[len(methods) :]:
        ax.axis("off")
    for ax in axes_flat[-cols:]:
        if ax.has_data():
            ax.set_xlabel("Top-k nodes")
    for ax in axes_flat[::cols]:
        ax.set_ylabel("Mean signed probability fidelity")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))
    task_suffix = "all tasks" if df["task"].nunique() > 1 else str(df["task"].iloc[0])
    fig.suptitle(
        f"{dataset} ({task_suffix}): Probability Fidelity+ and 1 - Probability Fidelity-",
        fontsize=18,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(
        fig,
        out_dir,
        f"{dataset}_{task_suffix.replace('/', '_').replace(':', '_')}_prob_fidelity_plus_one_minus_small_multiples",
        formats,
    )


def main() -> None:
    args = _parse_args()
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    if not formats:
        raise ValueError("--formats must contain at least one format")

    summary_path = args.input / args.dataset / "merged" / "fidelity_curves_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    args.out.mkdir(parents=True, exist_ok=True)
    _setup_style()
    df = _read_summary(summary_path, args.task)
    plot_crossing(df, args.dataset, args.out, formats)


if __name__ == "__main__":
    main()
