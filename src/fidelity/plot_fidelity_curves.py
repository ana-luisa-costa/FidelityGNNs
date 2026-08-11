"""Plot probability and accuracy fidelity curves from an aggregated CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fidelity.config import load_config_defaults


DEFAULT_SUMMARY = Path("data/processed/BPIC13_O/fidelity_curves/fidelity_curves_summary.csv")
DEFAULT_OUT = Path("data/processed/BPIC13_O/fidelity_curves/plots")

EXPLAINER_ORDER = [
    "gradient",
    "ig",
    "lime",
    "shap",
    "gnnexplainer",
    "pgmexplainer",
    "random",
    "degree",
    "fox",
    "prophet",
]

COLOR_MAP: Dict[str, str] = {
    "gradient": "#1f77b4",
    "ig": "#17becf",
    "fox": "#ff7f0e",
    "lime": "#2ca02c",
    "shap": "#9467bd",
    "random": "#8a8f98",
    "degree": "#3f454d",
    "prophet": "#7f7f7f",
    "gnnexplainer": "#d62728",
    "pgmexplainer": "#8c564b",
}

DISPLAY_NAMES = {
    "gnnexplainer": "GNNExplainer",
    "pgmexplainer": "PGMExplainer",
    "prophet": "PROPHET",
}

MARKER_MAP: Dict[str, str] = {
    "gradient": "o",
    "ig": "s",
    "fox": "X",
    "lime": "^",
    "shap": "D",
    "random": "x",
    "degree": "v",
    "prophet": "h",
    "gnnexplainer": "*",
    "pgmexplainer": "P",
}

LINESTYLE_MAP: Dict[str, str] = {
    "gradient": "-",
    "ig": "--",
    "fox": "-",
    "lime": ":",
    "shap": "-.",
    "random": ":",
    "degree": "--",
    "prophet": ":",
    "gnnexplainer": "-.",
    "pgmexplainer": "--",
}


def _build_arg_parser(config_defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    defaults = config_defaults or {}
    parser = argparse.ArgumentParser(
        description="Plot PyG model-fidelity curves from fidelity_curves_summary.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(defaults.get("summary", DEFAULT_SUMMARY)),
        help="Path to fidelity_curves_summary.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(defaults.get("out", DEFAULT_OUT)),
        help="Directory where figures will be written.",
    )
    parser.add_argument(
        "--title-suffix",
        default=defaults.get("title_suffix", "BPIC13_O"),
        help="Dataset suffix used in plot titles.",
    )
    parser.add_argument(
        "--formats",
        default=defaults.get("formats", "png,pdf"),
        help="Comma-separated output formats, e.g. png,pdf,svg.",
    )
    return parser


def _read_summary(path: Path) -> pd.DataFrame:
    required = {"explainer", "task", "top_k"}
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    available = [
        metric
        for metric in ("prob", "acc")
        if {f"mean_fid_{metric}_plus", f"mean_fid_{metric}_minus"}.issubset(df.columns)
    ]
    if not available:
        raise ValueError(f"{path} contains no probability or accuracy fidelity columns")

    df = df.copy()
    df["explainer"] = df["explainer"].astype(str)
    df["top_k"] = df["top_k"].astype(int)
    for col in df.columns:
        if col.startswith(("mean_fid_", "mean_one_minus_fid_", "sem_fid_", "sem_one_minus_fid_")):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["task", "explainer", "top_k"])


def _ordered_explainers(names: Iterable[str]) -> List[str]:
    available = set(names)
    ordered = [name for name in EXPLAINER_ORDER if name in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _parse_formats(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return [fmt.strip() for fmt in raw.split(",") if fmt.strip()]
    if isinstance(raw, Sequence):
        return [str(fmt).strip() for fmt in raw if str(fmt).strip()]
    raise TypeError(f"Expected comma-separated string or sequence, got {type(raw).__name__}")


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#243447",
            "axes.linewidth": 1.1,
            "axes.grid": True,
            "grid.color": "#e4e7eb",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def _set_top_k_xlim(ax: plt.Axes, df: pd.DataFrame) -> None:
    lo = float(df["top_k"].min())
    hi = float(df["top_k"].max())
    if lo == hi:
        pad = max(1.0, lo * 0.1)
        ax.set_xlim(max(0.0, lo - pad), hi + pad)
    else:
        pad = max(1.0, (hi - lo) * 0.05)
        ax.set_xlim(max(0.0, lo - pad), hi + pad)


def _has_variability_bands(df: pd.DataFrame) -> bool:
    return any(col.startswith("sem_") for col in df.columns)


def _title_with_variability_note(title: str, df: pd.DataFrame) -> str:
    if not _has_variability_bands(df):
        return title
    return f"{title}\n(bands: SEM across pair-seed runs)"


def _draw_variability_band(
    ax: plt.Axes,
    sub: pd.DataFrame,
    metric_col: str,
    color: Optional[str],
) -> None:
    sem_col = metric_col.replace("mean_", "sem_", 1)
    if sem_col not in sub.columns:
        return
    x = sub["top_k"].to_numpy(dtype=float)
    y = sub[metric_col].to_numpy(dtype=float)
    sem = sub[sem_col].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sem)
    if not valid.any():
        return
    ax.fill_between(
        x[valid],
        y[valid] - sem[valid],
        y[valid] + sem[valid],
        color=color,
        alpha=0.13,
        linewidth=0,
    )


def _draw_metric_lines(ax: plt.Axes, df: pd.DataFrame, metric_col: str, marker_size: float) -> List[str]:
    labels: List[str] = []
    for explainer in _ordered_explainers(df["explainer"]):
        sub = df[df["explainer"] == explainer].sort_values("top_k")
        if sub.empty:
            continue
        label = DISPLAY_NAMES.get(explainer, explainer)
        color = COLOR_MAP.get(explainer)
        labels.append(label)
        _draw_variability_band(ax, sub, metric_col, color)
        ax.plot(
            sub["top_k"], sub[metric_col],
            marker=MARKER_MAP.get(explainer, "o"),
            linestyle=LINESTYLE_MAP.get(explainer, "-"),
            linewidth=2.4,
            markersize=marker_size,
            color=color, label=label,
        )
    return labels


def _metric_ylim(df: pd.DataFrame, metric_col: str) -> Tuple[float, float]:
    vals = df[metric_col].dropna()
    if vals.empty:
        return (-0.05, 0.05)
    lo = min(0.0, float(vals.min()))
    hi = max(0.0, float(vals.max()))
    pad = max((hi - lo) * 0.12, 0.02)
    return lo - pad, hi + pad


def _finish_metric_axis(ax: plt.Axes, df: pd.DataFrame, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color="#667085", linestyle=":", linewidth=1.2)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    _set_top_k_xlim(ax, df)


def _plot_metric(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    labels = _draw_metric_lines(ax, df, metric_col, marker_size=6.5)
    ax.set_title(_title_with_variability_note(title, df), pad=16)
    ax.set_xlabel("top-k nodes")
    _finish_metric_axis(ax, df, ylabel)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=min(4, len(labels)),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return fig


def _plot_combined(
    df: pd.DataFrame,
    plus_col: str,
    second_col: str,
    plus_ylabel: str,
    second_ylabel: str,
    second_subtitle: str,
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 9.2), sharex=True)
    metric_specs = [
        (plus_col, plus_ylabel, "Fidelity+: removing explanation nodes"),
        (second_col, second_ylabel, second_subtitle),
    ]

    for ax, (metric_col, ylabel, subtitle) in zip(axes, metric_specs):
        _draw_metric_lines(ax, df, metric_col, marker_size=6.2)
        ax.set_title(subtitle, pad=10, fontsize=15)
        _finish_metric_axis(ax, df, ylabel)

    axes[-1].set_xlabel("top-k nodes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(4, len(labels)),
        frameon=False,
    )
    fig.suptitle(_title_with_variability_note(title, df), fontsize=19, y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def _plot_small_multiples(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
) -> plt.Figure:
    explainers = _ordered_explainers(df["explainer"])
    n = len(explainers)
    ncols = 3 if n > 4 else 2
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.35 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    ylim = _metric_ylim(df, metric_col)

    for ax, explainer in zip(axes_flat, explainers):
        sub = df[df["explainer"] == explainer].sort_values("top_k")
        label = DISPLAY_NAMES.get(explainer, explainer)
        color = COLOR_MAP.get(explainer)
        _draw_variability_band(ax, sub, metric_col, color)
        ax.plot(
            sub["top_k"],
            sub[metric_col],
            marker=MARKER_MAP.get(explainer, "o"),
            linestyle=LINESTYLE_MAP.get(explainer, "-"),
            linewidth=2.2,
            markersize=6.2,
            color=color,
        )
        ax.set_title(label, fontsize=13, pad=8)
        ax.axhline(0.0, color="#667085", linestyle=":", linewidth=1.0)
        ax.set_ylim(*ylim)
        _set_top_k_xlim(ax, df)

    for ax in axes_flat[n:]:
        ax.axis("off")

    for row_idx in range(nrows):
        axes_flat[row_idx * ncols].set_ylabel(ylabel)
    for ax in axes_flat[max(0, (nrows - 1) * ncols): nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel("top-k nodes")

    fig.suptitle(_title_with_variability_note(title, df), fontsize=18, y=1.01)
    fig.tight_layout()
    return fig


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def _safe_stem(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def main() -> None:
    config_defaults, config_path = load_config_defaults()
    args = _build_arg_parser(config_defaults).parse_args()
    if config_path is not None:
        print(f"Loaded config -> {config_path}")
    formats = _parse_formats(args.formats)
    _setup_style()
    df = _read_summary(args.summary)
    task_values = sorted(df["task"].astype(str).unique())
    multi_task = len(task_values) > 1
    for task in task_values:
        task_df = df[df["task"].astype(str) == task].copy()
        task_prefix = f"{_safe_stem(task)}_" if multi_task else ""
        title_suffix = f"{args.title_suffix} - {task}" if multi_task else args.title_suffix
        for metric in ("prob", "acc"):
            plus_col = f"mean_fid_{metric}_plus"
            minus_col = f"mean_fid_{metric}_minus"
            if plus_col not in task_df or not task_df[plus_col].notna().any():
                continue
            one_minus_col = f"mean_one_minus_fid_{metric}_minus"
            if one_minus_col not in task_df:
                task_df[one_minus_col] = 1.0 - task_df[minus_col]
            ylabel = "mean signed probability drop" if metric == "prob" else "mean class-change rate"
            sufficiency_ylabel = (
                "mean 1 - signed probability drop" if metric == "prob" else "mean prediction agreement"
            )
            label = "Probability" if metric == "prob" else "Accuracy"

            views = [
                (plus_col, "fid_plus_curve", f"{label} Fidelity+: Removing Explanation", ylabel),
                (minus_col, "fid_minus_curve", f"{label} Fidelity-: Keeping Explanation", ylabel),
                (
                    one_minus_col,
                    "one_minus_fid_minus_curve",
                    f"{label} 1-Fidelity-: Sufficiency",
                    sufficiency_ylabel,
                ),
            ]
            for column, stem, title, view_ylabel in views:
                figure = _plot_metric(
                    task_df, column, view_ylabel, f"{title} ({title_suffix})"
                )
                _save_figure(figure, args.out, f"{task_prefix}{metric}_{stem}", formats)

            raw_combined = _plot_combined(
                task_df,
                plus_col,
                minus_col,
                ylabel,
                ylabel,
                "Fidelity-: keeping explanation nodes with protected context",
                f"{label} Fidelity Curves ({title_suffix})",
            )
            _save_figure(raw_combined, args.out, f"{task_prefix}{metric}_fidelity_raw_combined", formats)
            sufficient_combined = _plot_combined(
                task_df,
                plus_col,
                one_minus_col,
                ylabel,
                sufficiency_ylabel,
                "1-Fidelity-: sufficiency",
                f"{label} Necessity and Sufficiency Curves ({title_suffix})",
            )
            _save_figure(
                sufficient_combined, args.out, f"{task_prefix}{metric}_fidelity_sufficiency_combined", formats
            )

            for column, stem, title, view_ylabel in views:
                figure = _plot_small_multiples(
                    task_df, column, view_ylabel, f"{title} Small Multiples ({title_suffix})"
                )
                _save_figure(figure, args.out, f"{task_prefix}{metric}_{stem.replace('_curve', '')}_small_multiples", formats)


if __name__ == "__main__":
    main()
