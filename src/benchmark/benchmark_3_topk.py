
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_4_targets import METHOD_COLORS

DEFAULT_INPUT = Path("data/processed")
DEFAULT_OUTPUT = Path("results/benchmark_3")
DATASETS = (
    "BPIC12_A", "BPIC12_W", "BPIC12_WC", "BPIC13_I",
    "BPIC13_O", "BPIC17_O", "BPIC20_P", "BPIC20_R",
)
CANONICAL_TASKS = ("activity", "lifecycle", "resource", "role")
METHODS = ("gradient", "ig", "lime", "shap", "gnnexplainer", "pgmexplainer")
METHOD_LABELS = {
    "gradient": "Gradient",
    "ig": "IG",
    "lime": "LIME",
    "shap": "SHAP",
    "gnnexplainer": "GNNExplainer",
    "pgmexplainer": "PGMExplainer",
}
DATASET_STYLES = {
    "BPIC12_A": ("#1f77b4", "o", "-"),
    "BPIC12_W": ("#ff7f0e", "s", "--"),
    "BPIC12_WC": ("#2ca02c", "^", "-."),
    "BPIC13_I": ("#bcbd22", "v", ":"),
    "BPIC13_O": ("#17becf", "X", "--"),
    "BPIC17_O": ("#8c564b", "*", "-"),
    "BPIC20_P": ("#d62728", "D", ":"),
    "BPIC20_R": ("#9467bd", "P", "-"),
}
EXPLAINER_MARKERS = {
    "gradient":     "s",
    "ig":           "o",
    "lime":         "^",
    "shap":         "P",
    "gnnexplainer": "*",
    "pgmexplainer": "D",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot probability fidelity across datasets with one panel per explainer."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formats", default="png", help="Comma-separated formats, e.g. png,pdf.")
    parser.add_argument(
        "--task",
        default="activity",
        help=(
            "Task to plot: activity, resource, role, lifecycle, a canonical "
            "task name, or 'all' to produce one output set per target."
        ),
    )
    parser.add_argument("--expected-pairs", type=int, default=5)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument(
        "--use-full-mode",
        action="store_true",
        default=False,
        help="Read from <input>/<dataset>/fidelity_curves_full/ instead of <input>/<dataset>/merged/",
    )
    parser.add_argument(
        "--dir-suffix", default="",
        help="Suffix appended to the fidelity_curves_full/ (or merged/) folder name, "
             "e.g. '_n20' to read fidelity_curves_full_n20/.",
    )
    parser.add_argument(
        "--saturation-dataset", default=None,
        help=(
            "If set, skip the normal cross-dataset plots and instead produce a "
            "single necessity-vs-k saturation plot for this one dataset, reading "
            "a fine-grained top_k sweep (e.g. k=1..25) from "
            "<input>/<dataset>/<saturation-dir>/fidelity_curves_summary.csv."
        ),
    )
    parser.add_argument(
        "--saturation-dir", default="fidelity_ks",
        help="Subfolder under <input>/<saturation-dataset>/ holding the fine-grained "
             "fidelity_curves_summary.csv (default: fidelity_ks).",
    )
    parser.add_argument(
        "--saturation-task", default="activity",
        help="Task to plot in the saturation figure (default: activity).",
    )
    parser.add_argument(
        "--saturation-tol", type=float, default=0.05,
        help=(
            "Saturation tolerance, as a fraction of each explainer's observed "
            "necessity range. The saturation point is the smallest k after which "
            "the curve stays within this band of its value at the largest k."
        ),
    )
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


def _load_outputs(input_dir: Path, expected_pairs: int, expected_seeds: int, task: str, use_full_mode: bool = False, dir_suffix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    aucs = []
    required_summary_base = {
        "explainer", "task", "top_k", "mean_fid_prob_plus", "mean_fid_prob_minus",
        "mean_one_minus_fid_prob_minus",
    }
    required_summary_merged = required_summary_base | {
        "sem_fid_prob_plus", "sem_fid_prob_minus", "sem_one_minus_fid_prob_minus",
        "n_pairs", "n_seeds",
    }
    required_summary = required_summary_base if use_full_mode else required_summary_merged
    required_auc = {
        "explainer", "task", "auc_fid_prob_plus_raw", "auc_fid_prob_plus_normalized",
        "auc_fid_prob_minus_raw", "auc_fid_prob_minus_normalized",
        "auc_one_minus_fid_prob_minus_raw",
        "auc_one_minus_fid_prob_minus_normalized", "score_prob",
    }

    for dataset in DATASETS:
        subdir_name = f"fidelity_curves_full{dir_suffix}" if use_full_mode else f"merged{dir_suffix}"
        fidelity_dir = input_dir / dataset / subdir_name
        summary_path = fidelity_dir / "fidelity_curves_summary.csv"
        auc_path = fidelity_dir / "fidelity_curves_auc.csv"
        if not summary_path.is_file() or not auc_path.is_file():
            warnings.warn(f"Missing fidelity files for {dataset}: {fidelity_dir}; skipping")
            continue

        summary = pd.read_csv(summary_path)
        auc = pd.read_csv(auc_path)
        missing_summary = required_summary - set(summary.columns)
        missing_auc = required_auc - set(auc.columns)
        if missing_summary or missing_auc:
            raise ValueError(
                f"{dataset} has incompatible columns; "
                f"summary missing={sorted(missing_summary)}, AUC missing={sorted(missing_auc)}"
            )

        available_summary_tasks = sorted(summary["task"].astype(str).unique())
        available_auc_tasks = sorted(auc["task"].astype(str).unique())
        resolved_task = _resolve_task(available_summary_tasks, task)
        if resolved_task is None or resolved_task not in available_auc_tasks:
            warnings.warn(
                f"Skipping {dataset}: no rows for task {task!r}; "
                f"available={available_summary_tasks}"
            )
            continue
        summary = summary[summary["task"].astype(str) == resolved_task].copy()
        auc = auc[auc["task"].astype(str) == resolved_task].copy()
        missing_methods = set(METHODS) - set(summary["explainer"]) | set(METHODS) - set(auc["explainer"])
        if missing_methods:
            warnings.warn(
                f"Skipping {dataset}/{resolved_task}: missing explainers {sorted(missing_methods)}"
            )
            continue
        if not use_full_mode:
            if set(summary["n_pairs"].astype(int)) != {expected_pairs}:
                raise ValueError(f"{dataset} does not contain exactly {expected_pairs} pairs")
            if set(summary["n_seeds"].astype(int)) != {expected_seeds}:
                raise ValueError(f"{dataset} does not contain exactly {expected_seeds} seeds")

        summary["dataset"] = dataset
        auc["dataset"] = dataset
        summaries.append(summary)
        aucs.append(auc)

    if not summaries:
        raise ValueError(f"No datasets contain task {task!r}")

    summary_all = pd.concat(summaries, ignore_index=True)
    auc_all = pd.concat(aucs, ignore_index=True)
    numeric_summary = [c for c in required_summary if c not in {"explainer", "task"}]
    numeric_auc = [c for c in required_auc if c not in {"explainer", "task"}]
    bad_summary = ~np.isfinite(summary_all[numeric_summary].to_numpy(dtype=float)).all(axis=1)
    if bad_summary.any():
        dropped = summary_all.loc[bad_summary, ["dataset", "explainer"]].drop_duplicates()
        warnings.warn(
            "Dropping non-finite summary rows for: "
            + ", ".join(f"{r.dataset}/{r.explainer}" for r in dropped.itertuples())
        )
        summary_all = summary_all.loc[~bad_summary].copy()
    bad_auc = ~np.isfinite(auc_all[numeric_auc].to_numpy(dtype=float)).all(axis=1)
    if bad_auc.any():
        dropped = auc_all.loc[bad_auc, ["dataset", "explainer"]].drop_duplicates()
        warnings.warn(
            "Dropping non-finite AUC rows for: "
            + ", ".join(f"{r.dataset}/{r.explainer}" for r in dropped.itertuples())
        )
        auc_all = auc_all.loc[~bad_auc].copy()
    if summary_all.empty or auc_all.empty:
        raise ValueError("All rows contained non-finite values")

    top_k = np.sort(summary_all["top_k"].unique().astype(float))
    span = float(top_k[-1] - top_k[0])
    for raw, normalized in (
        ("auc_fid_prob_plus_raw", "auc_fid_prob_plus_normalized"),
        ("auc_fid_prob_minus_raw", "auc_fid_prob_minus_normalized"),
        ("auc_one_minus_fid_prob_minus_raw", "auc_one_minus_fid_prob_minus_normalized"),
    ):
        if not np.allclose(auc_all[raw] / span, auc_all[normalized], atol=1e-9):
            raise ValueError(f"Normalized AUC values do not match {raw} / {span:g}")
    return summary_all, auc_all


def _setup_style() -> None:
    plt.rcParams.update({
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
    })


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def _safe_stem(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def _plot_fidelity_combined(
    summary: pd.DataFrame,
    task_label: str,
    stem: str,
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    """Fidelity+ and Fidelity− in one figure: one row per metric, one panel per explainer."""
    datasets = tuple(d for d in DATASETS if d in set(summary["dataset"]))
    top_k = np.sort(summary["top_k"].unique())

    row_specs = [
        ("mean_fid_prob_plus",  "sem_fid_prob_plus",  "Fidelity+",      "#2171b5"),
        ("mean_fid_prob_minus", "sem_fid_prob_minus", "Fidelity\u2212", "#c0392b"),
    ]

    # 6-method cols split into two groups of 3 with a narrow spacer column between them
    _SPACER = 3
    _METHOD_COLS = [0, 1, 2, 4, 5, 6]  # indices in the 7-column grid
    fig, all_axes = plt.subplots(
        len(row_specs), 7,
        figsize=(25, 8.5),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.08, 1, 1, 1], "hspace": 0.12, "wspace": 0.05},
    )
    # Hide spacer columns
    for row_idx in range(len(row_specs)):
        all_axes[row_idx, _SPACER].set_visible(False)
    axes = all_axes[:, _METHOD_COLS]  # shape (2, 6)

    # Manually share x and share y within each row
    for col_idx in range(len(METHODS)):
        for row_idx in range(1, len(row_specs)):
            axes[row_idx, col_idx].sharex(axes[0, col_idx])
    for row_idx in range(len(row_specs)):
        for col_idx in range(1, len(METHODS)):
            axes[row_idx, col_idx].sharey(axes[row_idx, 0])

    for row_idx, (value_col, sem_col, row_label, row_color) in enumerate(row_specs):
        vals = summary[value_col].to_numpy(dtype=float)
        lo = min(0.0, float(np.nanmin(vals)))
        hi = max(0.0, float(np.nanmax(vals)))
        pad = max(0.02, 0.08 * (hi - lo))

        for col_idx, method in enumerate(METHODS):
            ax = axes[row_idx, col_idx]
            for dataset in datasets:
                part = summary[
                    (summary["explainer"] == method) & (summary["dataset"] == dataset)
                ].sort_values("top_k")
                color, marker, linestyle = DATASET_STYLES[dataset]
                x = part["top_k"].to_numpy(dtype=float)
                y = part[value_col].to_numpy(dtype=float)
                ax.plot(x, y, color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.8, markersize=5.0)
                if sem_col in part.columns:
                    sem = part[sem_col].to_numpy(dtype=float)
                    ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.10, linewidth=0)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks(top_k)
            ax.grid(linewidth=0.5)
            # Titles only on top row
            if row_idx == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=12)
            # x-label only on bottom row
            if row_idx == len(row_specs) - 1:
                ax.set_xlabel("Top-k nodes")
            else:
                ax.tick_params(labelbottom=False)
            # y-tick labels only on left column
            if col_idx != 0:
                ax.tick_params(labelleft=False)

        # Shared y-axis label and coloured row badge on the first panel
        axes[row_idx, 0].set_ylabel("Mean signed probability drop")
        axes[row_idx, 0].annotate(
            row_label,
            xy=(-0.28, 0.5),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            rotation=90,
            annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=row_color, edgecolor="none"),
        )

    legend_handles = [
        plt.Line2D(
            [0], [0],
            color=DATASET_STYLES[d][0],
            marker=DATASET_STYLES[d][1],
            linestyle=DATASET_STYLES[d][2],
            linewidth=1.8,
            markersize=6,
            label=d,
        )
        for d in datasets
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(datasets),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0, 1, 0.94))
    _save(fig, out_dir, stem, formats)


def _plot_curves(
    summary: pd.DataFrame,
    value_col: str,
    sem_col: str,
    title: str,
    ylabel: str,
    stem: str,
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    datasets = tuple(dataset for dataset in DATASETS if dataset in set(summary["dataset"]))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), sharex=True, sharey=True)
    top_k = np.sort(summary["top_k"].unique())

    for ax, method in zip(axes.flat, METHODS):
        for dataset in datasets:
            part = summary[
                (summary["explainer"] == method) & (summary["dataset"] == dataset)
            ].sort_values("top_k")
            color, marker, linestyle = DATASET_STYLES[dataset]
            x = part["top_k"].to_numpy(dtype=float)
            y = part[value_col].to_numpy(dtype=float)
            ax.plot(
                x, y, label=dataset, color=color, marker=marker, linestyle=linestyle,
                linewidth=2.0, markersize=5.5,
            )
            if sem_col in part.columns:
                sem = part[sem_col].to_numpy(dtype=float)
                ax.fill_between(
                    x, np.clip(y - sem, 0.0, 1.0), np.clip(y + sem, 0.0, 1.0),
                    color=color, alpha=0.10, linewidth=0,
                )
        ax.set_title(METHOD_LABELS[method])
        ax.set_xticks(top_k)

    values = summary[value_col].to_numpy(dtype=float)
    lo = min(0.0, float(np.nanmin(values)))
    hi = max(0.0, float(np.nanmax(values)))
    pad = max(0.02, 0.08 * (hi - lo))
    for ax in axes.flat:
        ax.set_ylim(lo - pad, hi + pad)

    for ax in axes[-1]:
        ax.set_xlabel("Top-k nodes")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(datasets), frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, out_dir, stem, formats)


def _plot_auc(
    auc: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    stem: str,
    normalized: bool,
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    datasets = tuple(dataset for dataset in DATASETS if dataset in set(auc["dataset"]))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), sharey=True)
    positions = np.arange(len(datasets))
    colors = [DATASET_STYLES[dataset][0] for dataset in datasets]

    for ax, method in zip(axes.flat, METHODS):
        method_rows = auc[auc["explainer"] == method].set_index("dataset")
        values = np.array(
            [
                float(method_rows.loc[dataset, value_col])
                if dataset in method_rows.index
                else np.nan
                for dataset in datasets
            ],
            dtype=float,
        )
        bars = ax.bar(positions, np.nan_to_num(values, nan=0.0), color=colors, width=0.72)
        ax.bar_label(
            bars,
            labels=["" if np.isnan(v) else f"{v:.3f}" for v in values],
            fontsize=7,
            padding=2,
        )
        ax.set_title(METHOD_LABELS[method])
        ax.set_xticks(positions, datasets, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="x", visible=False)

    values = auc[value_col].to_numpy(dtype=float)
    lo = min(0.0, float(np.nanmin(values)))
    hi = max(0.0, float(np.nanmax(values)))
    pad = max(0.02 if normalized else 0.1, 0.10 * (hi - lo))
    for ax in axes.flat:
        ax.set_ylim(lo - pad, hi + pad)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, stem, formats)


def _plot_score_lollipop(
    auc: pd.DataFrame,
    title: str,
    stem: str,
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    """Lollipop chart of score_prob per explainer, one panel per dataset."""
    datasets = tuple(d for d in DATASETS if d in set(auc["dataset"]))
    n = len(datasets)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 4.5 * nrows), sharey=True)
    axes_flat = list(np.array(axes).flat)

    x_pos = np.arange(len(METHODS))
    x_labels = [METHOD_LABELS[m] for m in METHODS]

    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        color = DATASET_STYLES[dataset][0]
        rows = auc[auc["dataset"] == dataset].set_index("explainer")
        for j, method in enumerate(METHODS):
            if method not in rows.index:
                continue
            s = float(rows.loc[method, "score_prob"])
            ax.vlines(j, 0, s, color=color, linewidth=1.6, alpha=0.55)
            ax.plot(j, s, marker=EXPLAINER_MARKERS.get(method, "o"),
                    color=color, markersize=9, zorder=3)
        ax.set_title(dataset)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks(np.arange(0.0, 1.1, 0.2))
        ax.grid(axis="y", linewidth=0.6)
        ax.grid(axis="x", visible=False)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for ax in np.array(axes).reshape(nrows, ncols)[:, 0]:
        ax.set_ylabel(r"$S_{Eq}$")

    legend_handles = [
        plt.Line2D([0], [0], marker=EXPLAINER_MARKERS.get(m, "o"), color="grey",
                   linestyle="None", markersize=8, label=METHOD_LABELS[m])
        for m in METHODS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(METHODS),
               frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=10)
    fig.suptitle(title, fontsize=16, y=1.01)
    fig.tight_layout(rect=(0, 0.07, 1, 1.0))
    _save(fig, out_dir, stem, formats)


def _plot_fidelity_comparison_topk(
    summary: pd.DataFrame,
    title: str,
    stem: str,
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    """Scatter plot of Fidelity+ vs 1-Fidelity- per top_k, one panel per k value."""
    _SHOW_K = {5, 15, 25}
    top_k_values = sorted(
        k for k in summary["top_k"].unique().astype(int).tolist() if k in _SHOW_K
    )
    n = len(top_k_values)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    # Compute figsize so each panel is naturally square given the fixed margins,
    # avoiding the white gaps that set_aspect("equal", adjustable="box") produces.
    _L, _R, _B, _T = 0.08, 0.97, 0.18, 0.90
    _panel_side = 4.5  # inches per square panel
    fig_w = _panel_side * ncols / (_R - _L)
    fig_h = _panel_side * nrows / (_T - _B)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.1, "hspace": 0.0},
    )
    axes_flat = list(np.array(axes).flat)
    datasets = tuple(d for d in DATASETS if d in set(summary["dataset"]))

    for i, k in enumerate(top_k_values):
        ax = axes_flat[i]
        subset = summary[summary["top_k"] == k]
        for dataset in datasets:
            for method in METHODS:
                row = subset[(subset["dataset"] == dataset) & (subset["explainer"] == method)]
                if row.empty:
                    continue
                x = float(row["mean_fid_prob_plus"].iloc[0])
                y = float(row["mean_one_minus_fid_prob_minus"].iloc[0])
                ax.scatter(
                    x, y,
                    color=DATASET_STYLES[dataset][0],
                    marker=EXPLAINER_MARKERS.get(method, "o"),
                    s=85, alpha=0.9, zorder=3,
                )
        ax.set_title(f"top_k = {k}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.arange(0.0, 1.1, 0.2))
        ax.set_yticks(np.arange(0.0, 1.1, 0.2))
        ax.grid(linewidth=0.6)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    axes_grid = np.array(axes).reshape(nrows, ncols)
    for ax in axes_grid[:, 0]:
        ax.set_ylabel("1 \u2212 Fidelity\u2212")
    for ax in axes_grid[-1]:
        ax.set_xlabel("Fidelity+")
    for ax in axes_flat[:n]:
        ax.tick_params(labelbottom=False, labelleft=False)
    for ax in axes_grid[-1]:
        ax.tick_params(labelbottom=True)
    for ax in axes_grid[:, 0]:
        ax.tick_params(labelleft=True)

    dataset_handles = [
        plt.Line2D([0], [0], marker="o", color=DATASET_STYLES[d][0],
                   linestyle="None", markersize=8, label=d)
        for d in datasets
    ]
    explainer_handles = [
        plt.Line2D([0], [0], marker=EXPLAINER_MARKERS.get(m, "o"), color="grey",
                   linestyle="None", markersize=8, label=METHOD_LABELS[m])
        for m in METHODS
    ]
    leg1 = fig.legend(handles=dataset_handles, title="Dataset", loc="lower left",
                      bbox_to_anchor=(0.01, 0.0), frameon=True, fontsize=9)
    fig.legend(handles=explainer_handles, title="Explainer", loc="lower right",
               bbox_to_anchor=(0.99, 0.0), frameon=True, fontsize=9)
    fig.add_artist(leg1)
    fig.suptitle(title, fontsize=16, y=0.99)
    fig.subplots_adjust(left=_L, right=_R, bottom=_B, top=_T, wspace=0.0, hspace=0.0)
    _save(fig, out_dir, stem, formats)


def _saturation_point(k_values: np.ndarray, y_values: np.ndarray, tol: float) -> int:
    """Smallest k after which the curve stays within `tol` (a fraction of the
    curve's own observed range) of its value at the largest k. This is a
    settling-time style definition: an early value close to the final one
    doesn't count unless the curve stays there for the rest of the sweep.
    Falls back to the last k if the curve never settles within tolerance.
    """
    if len(k_values) == 0:
        raise ValueError("empty curve")
    final = y_values[-1]
    y_range = float(y_values.max() - y_values.min())
    band = tol * y_range if y_range > 0 else tol
    for i in range(len(k_values)):
        if np.all(np.abs(y_values[i:] - final) <= band):
            return int(k_values[i])
    return int(k_values[-1])


def _load_ks_summary(input_dir: Path, dataset: str, saturation_dir: str, task: str) -> pd.DataFrame:
    summary_path = input_dir / dataset / saturation_dir / "fidelity_curves_summary.csv"
    if not summary_path.is_file():
        raise ValueError(f"Missing {summary_path}")
    summary = pd.read_csv(summary_path)
    resolved = _resolve_task(sorted(summary["task"].astype(str).unique()), task)
    if resolved is None:
        raise ValueError(
            f"{summary_path}: no rows for task {task!r}; "
            f"available={sorted(summary['task'].astype(str).unique())}"
        )
    summary = summary[summary["task"].astype(str) == resolved].copy()
    if summary.empty:
        raise ValueError(f"{summary_path}: empty after filtering to task {resolved!r}")
    return summary


def _plot_necessity_saturation(
    summary: pd.DataFrame,
    dataset: str,
    task_label: str,
    tol: float,
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    """Mean necessity (Fidelity+) vs explanation size k for one dataset, one
    panel, one line per explainer, with each curve's saturation point
    highlighted (larger marker + k label)."""
    fig, ax = plt.subplots(figsize=(9, 6.5))

    methods = [m for m in METHODS if m in set(summary["explainer"])]
    for method in methods:
        rows = summary[summary["explainer"] == method].sort_values("top_k")
        k = rows["top_k"].to_numpy(dtype=int)
        y = rows["mean_fid_prob_plus"].to_numpy(dtype=float)
        color = METHOD_COLORS.get(method, "0.3")

        ax.plot(k, y, color=color, linewidth=1.6, marker="o", markersize=5, zorder=2)

        sat_k = _saturation_point(k, y, tol)
        sat_idx = int(np.searchsorted(k, sat_k))
        ax.scatter(
            k[sat_idx], y[sat_idx],
            color=color, s=170, zorder=4, edgecolor="white", linewidth=1.0,
        )
        ax.annotate(
            str(sat_k), (k[sat_idx], y[sat_idx]),
            xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold", color="0.15",
        )

    ax.set_xlabel("Explanation size k")
    ax.set_ylabel("Mean necessity component")
    ax.set_xlim(float(summary["top_k"].min()), float(summary["top_k"].max()))
    ax.grid(True, linewidth=0.6)

    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS.get(m, "0.3"), marker="o",
                   markersize=6, linewidth=1.6, label=METHOD_LABELS[m])
        for m in methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=11)
    fig.suptitle(
        f"{dataset} — necessity saturation vs. explanation size ({task_label})",
        fontsize=13, y=1.14,
    )
    fig.tight_layout()
    stem = f"{_safe_stem(dataset)}_{_safe_stem(task_label)}_necessity_saturation"
    _save(fig, out_dir, stem, formats)


def _run_saturation(args: argparse.Namespace, formats: Sequence[str]) -> None:
    dataset = args.saturation_dataset
    summary = _load_ks_summary(args.input, dataset, args.saturation_dir, args.saturation_task)
    _setup_style()
    task_label = str(summary["task"].iloc[0])
    _plot_necessity_saturation(summary, dataset, task_label, args.saturation_tol, args.out, formats)


def main() -> None:
    args = _parse_args()
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    if not formats:
        raise ValueError("--formats must contain at least one format")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.saturation_dataset:
        _run_saturation(args, formats)
        return

    tasks = CANONICAL_TASKS if str(args.task).strip().lower() == "all" else (args.task,)
    for task in tasks:
        try:
            _run_task(args, task, formats)
        except ValueError as exc:
            warnings.warn(f"Skipping task {task!r}: {exc}")


def _run_task(args: argparse.Namespace, task: str, formats: Sequence[str]) -> None:
    summary, auc = _load_outputs(args.input, args.expected_pairs, args.expected_seeds, task, use_full_mode=args.use_full_mode, dir_suffix=args.dir_suffix)
    _setup_style()
    task_label = str(summary["task"].iloc[0])
    task_stem = _safe_stem(task_label)
    summary.to_csv(args.out / f"{task_stem}_probability_fidelity_summary_cross_datasets.csv", index=False)
    auc.to_csv(args.out / f"{task_stem}_probability_fidelity_auc_cross_datasets.csv", index=False)

    curve_specs = (
        ("mean_fid_prob_plus", "sem_fid_prob_plus", f"Probability Fidelity+ ({task_label})", "Mean signed probability drop", f"{task_stem}_prob_fidelity_plus_cross_datasets"),
        ("mean_fid_prob_minus", "sem_fid_prob_minus", f"Probability Fidelity- ({task_label})", "Mean signed probability drop", f"{task_stem}_prob_fidelity_minus_cross_datasets"),
        ("mean_one_minus_fid_prob_minus", "sem_one_minus_fid_prob_minus", f"1 - Probability Fidelity- ({task_label})", "Mean 1 - signed probability drop", f"{task_stem}_prob_one_minus_fidelity_minus_cross_datasets"),
    )
    for spec in curve_specs:
        _plot_curves(summary, *spec, args.out, formats)

    auc_specs = (
        ("auc_fid_prob_plus_raw", f"Raw AUC of Probability Fidelity+ ({task_label})", "Raw AUC", f"{task_stem}_prob_auc_fidelity_plus_raw_cross_datasets", False),
        ("auc_fid_prob_minus_raw", f"Raw AUC of Probability Fidelity- ({task_label})", "Raw AUC", f"{task_stem}_prob_auc_fidelity_minus_raw_cross_datasets", False),
        ("auc_one_minus_fid_prob_minus_raw", f"Raw AUC of 1 - Probability Fidelity- ({task_label})", "Raw AUC", f"{task_stem}_prob_auc_one_minus_fidelity_minus_raw_cross_datasets", False),
        ("auc_fid_prob_plus_normalized", f"Normalized AUC of Probability Fidelity+ ({task_label})", "Normalized AUC", f"{task_stem}_prob_auc_fidelity_plus_normalized_cross_datasets", True),
        ("auc_fid_prob_minus_normalized", f"Normalized AUC of Probability Fidelity- ({task_label})", "Normalized AUC", f"{task_stem}_prob_auc_fidelity_minus_normalized_cross_datasets", True),
        ("auc_one_minus_fid_prob_minus_normalized", f"Normalized AUC of 1 - Probability Fidelity- ({task_label})", "Normalized AUC", f"{task_stem}_prob_auc_one_minus_fidelity_minus_normalized_cross_datasets", True),
        ("score_prob", f"Geometric Probability-Fidelity Score ({task_label})", "Geometric score", f"{task_stem}_prob_geometric_score_cross_datasets", True),
    )
    for spec in auc_specs:
        _plot_auc(auc, *spec, args.out, formats)

    _plot_score_lollipop(
        auc,
        title=f"Combined Fidelity Score ({task_label})",
        stem=f"{task_stem}_prob_score_lollipop_cross_datasets",
        out_dir=args.out,
        formats=formats,
    )

    _plot_fidelity_combined(
        summary,
        task_label=task_label,
        stem=f"{task_stem}_prob_fidelity_combined_cross_datasets",
        out_dir=args.out,
        formats=formats,
    )

    _plot_fidelity_comparison_topk(
        summary,
        title=f"Fidelity+ vs 1\u2212Fidelity\u2212 by Top-k ({task_label})",
        stem=f"{task_stem}_prob_fidelity_comparison_topk_cross_datasets",
        out_dir=args.out,
        formats=formats,
    )

    print(f"Cross-dataset outputs -> {args.out}")


if __name__ == "__main__":
    main()
