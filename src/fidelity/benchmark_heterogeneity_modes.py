"""Benchmark fidelity metrics across heterogeneity modes (full, middle, homogeneous).

This script loads AUC (Area Under Curve) data from fidelity evaluations run with
different heterogeneity modes and compares:
  - Necessity (AUC+ normalized): How much do predictions drop when removing important nodes?
  - Sufficiency (1 - |AUC-| normalized): How stable are predictions when removing unimportant nodes?
  - Overall Score: Geometric mean of necessity and sufficiency.

Usage:
    python benchmark_heterogeneity_modes.py \\
        --full data/processed/BPIC13_O/fidelity_curves_full \\
        --middle data/processed/BPIC13_O/fidelity_curves_middle \\
        --homogeneous data/processed/BPIC13_O/fidelity_curves_homogeneous \\
        --task activity \\
        --out results/heterogeneity_benchmark
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODES = ("full", "middle", "homogeneous")
DEFAULT_METHODS = ("gradient", "ig", "lime", "shap", "gnnexplainer", "pgmexplainer")
METRICS = ("prob", "acc")

MODE_COLORS = {
    "full": "#1f77b4",
    "middle": "#ff7f0e", 
    "homogeneous": "#2ca02c",
}
MODE_ORDER = {"full": 0, "middle": 1, "homogeneous": 2}

METHOD_LABELS = {
    "gradient": "Gradient",
    "ig": "IG",
    "lime": "LIME",
    "shap": "SHAP",
    "gnnexplainer": "GNNExplainer",
    "pgmexplainer": "PGMExplainer",
    "baseline": "Baseline",
    "random": "Random",
    "degree": "Degree",
    "fox": "FOX",
    "prophet": "Prophet",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fidelity metrics across heterogeneity modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--full",
        type=Path,
        required=True,
        help="Folder with full-mode fidelity_curves_auc.csv",
    )
    parser.add_argument(
        "--middle",
        type=Path,
        required=True,
        help="Folder with middle-mode fidelity_curves_auc.csv",
    )
    parser.add_argument(
        "--homogeneous",
        type=Path,
        required=True,
        help="Folder with homogeneous-mode fidelity_curves_auc.csv",
    )
    parser.add_argument(
        "--task",
        default="activity",
        help="Task to benchmark (activity, org_resource, lifecycle_transition, etc.)",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="prob",
        help="Fidelity metric to use (prob for probability, acc for accuracy)",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated explainer methods to include",
    )
    parser.add_argument(
        "--dataset",
        default="BPIC",
        help="Dataset name for plot titles",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder for plots and tables",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _setup_style() -> None:
    """Configure matplotlib styling for publication-quality plots."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
    })
    sns.set_palette("husl")


def _load_mode_auc(
    folder: Path,
    mode: str,
    task: str,
    metric: str,
) -> pd.DataFrame:
    """Load and validate AUC data for a heterogeneity mode."""
    auc_file = folder / "fidelity_curves_auc.csv"
    if not auc_file.exists():
        raise FileNotFoundError(f"Missing {auc_file}")
    
    df = pd.read_csv(auc_file)
    
    # Filter to requested task
    available_tasks = sorted(df["task"].astype(str).unique())
    task_filtered = [t for t in available_tasks if task.lower() in t.lower()]
    if not task_filtered:
        raise ValueError(
            f"Task {task!r} not found in {mode} mode. Available: {available_tasks}"
        )
    target_task = task_filtered[0]
    df = df[df["task"].astype(str) == target_task].copy()
    
    # Extract relevant columns
    required_cols = [
        f"auc_fid_{metric}_plus_normalized",
        f"auc_one_minus_fid_{metric}_minus_normalized",
        f"score_{metric}",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {auc_file}: {missing}")
    
    df["heterogeneity_mode"] = mode
    df["task_resolved"] = target_task
    df["necessity"] = df[f"auc_fid_{metric}_plus_normalized"]
    df["sufficiency"] = df[f"auc_one_minus_fid_{metric}_minus_normalized"]
    df["overall_score"] = df[f"score_{metric}"]
    
    return df[["explainer", "task_resolved", "heterogeneity_mode", 
               "necessity", "sufficiency", "overall_score"]]


def _combine_modes(
    full_df: pd.DataFrame,
    middle_df: pd.DataFrame,
    homogeneous_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine AUC data from all three heterogeneity modes."""
    df = pd.concat([full_df, middle_df, homogeneous_df], ignore_index=True)
    df["necessity"] = pd.to_numeric(df["necessity"], errors="coerce")
    df["sufficiency"] = pd.to_numeric(df["sufficiency"], errors="coerce")
    df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce")
    return df


def _save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> None:
    """Save figure in multiple formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def _plot_bar_comparison(
    df: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    methods: Sequence[str],
    dataset: str,
    task: str,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    """Create bar plot comparing metric across modes and methods."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Prepare data
    plot_data = []
    for method in methods:
        for mode in MODES:
            subset = df[(df["explainer"] == method) & (df["heterogeneity_mode"] == mode)]
            if not subset.empty:
                value = subset[metric_col].iloc[0]
                plot_data.append({
                    "method": METHOD_LABELS.get(method, method),
                    "mode": mode.capitalize(),
                    "value": value,
                })
    
    if not plot_data:
        print(f"No data for {metric_name} plot")
        return
    
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values(["method", "mode"])
    
    # Create grouped bar plot
    x = np.arange(len(methods))
    width = 0.25
    for i, mode in enumerate(MODES):
        mode_data = plot_df[plot_df["mode"] == mode.capitalize()]
        mode_values = []
        for method in methods:
            method_subset = mode_data[mode_data["method"] == METHOD_LABELS.get(method, method)]
            if not method_subset.empty:
                mode_values.append(method_subset["value"].iloc[0])
            else:
                mode_values.append(np.nan)
        
        ax.bar(
            x + i * width,
            mode_values,
            width,
            label=mode.capitalize(),
            color=MODE_COLORS[mode],
            alpha=0.8,
        )
    
    ax.set_xlabel("Explainer Method", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
    ax.set_title(
        f"{dataset} ({task}): {metric_name} Across Heterogeneity Modes",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=45, ha="right")
    ax.legend(title="Heterogeneity Mode", loc="best", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    
    _save_figure(fig, out_dir, f"bar_{metric_col}", formats, dpi)


def _plot_heatmaps(
    df: pd.DataFrame,
    methods: Sequence[str],
    dataset: str,
    task: str,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    """Create heatmaps for necessity, sufficiency, and overall score."""
    for metric_col, metric_name in [
        ("necessity", "Necessity (AUC+)"),
        ("sufficiency", "Sufficiency (1-|AUC-|)"),
        ("overall_score", "Overall Score (S)"),
    ]:
        # Prepare data for heatmap
        heatmap_data = []
        for mode in MODES:
            row = []
            for method in methods:
                subset = df[(df["explainer"] == method) & (df["heterogeneity_mode"] == mode)]
                if not subset.empty:
                    row.append(subset[metric_col].iloc[0])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[m.capitalize() for m in MODES],
            columns=[METHOD_LABELS.get(m, m) for m in methods],
        )
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": metric_name},
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_title(
            f"{dataset} ({task}): {metric_name} Heatmap",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Explainer Method", fontsize=12, fontweight="bold")
        ax.set_ylabel("Heterogeneity Mode", fontsize=12, fontweight="bold")
        
        _save_figure(fig, out_dir, f"heatmap_{metric_col}", formats, dpi)


def _generate_summary_statistics(
    df: pd.DataFrame,
    methods: Sequence[str],
    out_dir: Path,
) -> None:
    """Generate and save summary statistics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary by mode
    summary_by_mode = df.groupby("heterogeneity_mode").agg({
        "necessity": ["mean", "std", "min", "max"],
        "sufficiency": ["mean", "std", "min", "max"],
        "overall_score": ["mean", "std", "min", "max"],
    }).round(4)
    print("\n=== Summary by Heterogeneity Mode ===")
    print(summary_by_mode)
    summary_by_mode.to_csv(out_dir / "summary_by_mode.csv")
    
    # Summary by method
    summary_by_method = df.groupby("explainer").agg({
        "necessity": ["mean", "std"],
        "sufficiency": ["mean", "std"],
        "overall_score": ["mean", "std"],
    }).round(4)
    print("\n=== Summary by Explainer Method ===")
    print(summary_by_method)
    summary_by_method.to_csv(out_dir / "summary_by_method.csv")
    
    # Rankings
    for mode in MODES:
        mode_df = df[df["heterogeneity_mode"] == mode]
        ranking = mode_df.groupby("explainer")["overall_score"].mean().sort_values(ascending=False)
        print(f"\n=== Ranking in {mode.upper()} mode ===")
        for rank, (method, score) in enumerate(ranking.items(), 1):
            print(f"  {rank}. {METHOD_LABELS.get(method, method)}: {score:.4f}")
    
    # Full summary table
    full_summary = []
    for method in methods:
        for mode in MODES:
            subset = df[(df["explainer"] == method) & (df["heterogeneity_mode"] == mode)]
            if not subset.empty:
                full_summary.append({
                    "Method": METHOD_LABELS.get(method, method),
                    "Mode": mode.capitalize(),
                    "Necessity": subset["necessity"].iloc[0],
                    "Sufficiency": subset["sufficiency"].iloc[0],
                    "Overall Score": subset["overall_score"].iloc[0],
                })
    
    summary_df = pd.DataFrame(full_summary)
    summary_df.to_csv(out_dir / "benchmark_summary.csv", index=False)
    print("\n=== Full Summary Table ===")
    print(summary_df.to_string(index=False))


def main() -> None:
    args = _parse_args()
    methods = _parse_csv_list(args.methods)
    formats = tuple(_parse_csv_list(args.formats))
    
    if not methods:
        raise ValueError("--methods must contain at least one method")
    if not formats:
        raise ValueError("--formats must contain at least one format")
    
    print(f"Loading fidelity data for task={args.task}, metric={args.metric}...")
    
    # Load data from all modes
    full_df = _load_mode_auc(args.full, "full", args.task, args.metric)
    middle_df = _load_mode_auc(args.middle, "middle", args.task, args.metric)
    homogeneous_df = _load_mode_auc(args.homogeneous, "homogeneous", args.task, args.metric)
    
    # Combine data
    df = _combine_modes(full_df, middle_df, homogeneous_df)
    task_name = df["task_resolved"].iloc[0]
    
    print(f"Loaded {len(df)} records")
    print(f"Methods: {methods}")
    print(f"Modes: {MODES}")
    
    # Setup plotting
    _setup_style()
    
    # Generate plots
    print("\nGenerating plots...")
    _plot_bar_comparison(
        df, "necessity", "Necessity (AUC+)",
        methods, args.dataset, task_name, args.out, formats, args.dpi
    )
    _plot_bar_comparison(
        df, "sufficiency", "Sufficiency (1-|AUC-|)",
        methods, args.dataset, task_name, args.out, formats, args.dpi
    )
    _plot_bar_comparison(
        df, "overall_score", "Overall Score (S)",
        methods, args.dataset, task_name, args.out, formats, args.dpi
    )
    _plot_heatmaps(
        df, methods, args.dataset, task_name, args.out, formats, args.dpi
    )
    
    # Generate statistics
    print("\nGenerating summary statistics...")
    _generate_summary_statistics(df, methods, args.out)
    
    # Save full dataframe
    df.to_csv(args.out / "benchmark_full_data.csv", index=False)
    print(f"Saved benchmark results to {args.out}")


if __name__ == "__main__":
    main()
