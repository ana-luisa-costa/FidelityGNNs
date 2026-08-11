"""Create cross-dataset probability-fidelity plots from merged experiment outputs."""

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


DEFAULT_INPUT = Path("outputs/fidelity_prob_5pairs_3seeds_20260717_115219")
DEFAULT_OUTPUT = Path("output/cross_datasets/5pair")
DATASETS = ("BPIC12_A", "BPIC12_W", "BPIC12_WC", "BPIC20_P", "BPIC20_R", "BPIC13_O")
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
    "BPIC20_P": ("#d62728", "D", ":"),
    "BPIC20_R": ("#9467bd", "P", "-"),
    "BPIC13_O": ("#17becf", "X", "--"),
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
        help="Task to plot: activity, resource, role, lifecycle, or a canonical task.",
    )
    parser.add_argument("--expected-pairs", type=int, default=5)
    parser.add_argument("--expected-seeds", type=int, default=3)
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


def _load_outputs(input_dir: Path, expected_pairs: int, expected_seeds: int, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    aucs = []
    required_summary = {
        "explainer", "task", "top_k", "mean_fid_prob_plus", "mean_fid_prob_minus",
        "mean_one_minus_fid_prob_minus", "sem_fid_prob_plus", "sem_fid_prob_minus",
        "sem_one_minus_fid_prob_minus", "n_pairs", "n_seeds",
    }
    required_auc = {
        "explainer", "task", "auc_fid_prob_plus_raw", "auc_fid_prob_plus_normalized",
        "auc_fid_prob_minus_raw", "auc_fid_prob_minus_normalized",
        "auc_one_minus_fid_prob_minus_raw",
        "auc_one_minus_fid_prob_minus_normalized", "score_prob",
    }

    for dataset in DATASETS:
        merged = input_dir / dataset / "merged"
        summary_path = merged / "fidelity_curves_summary.csv"
        auc_path = merged / "fidelity_curves_auc.csv"
        if not summary_path.is_file() or not auc_path.is_file():
            raise FileNotFoundError(f"Missing merged fidelity files for {dataset}: {merged}")

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
        if set(summary["explainer"]) != set(METHODS) or set(auc["explainer"]) != set(METHODS):
            raise ValueError(f"{dataset} does not contain exactly the six expected explainers")
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
    numeric_summary = [column for column in required_summary if column not in {"explainer", "task"}]
    numeric_auc = [column for column in required_auc if column not in {"explainer", "task"}]
    if not np.isfinite(summary_all[numeric_summary].to_numpy(dtype=float)).all():
        raise ValueError("Summary data contains non-finite probability-fidelity values")
    if not np.isfinite(auc_all[numeric_auc].to_numpy(dtype=float)).all():
        raise ValueError("AUC data contains non-finite values")

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
            # sem = part[sem_col].to_numpy(dtype=float)
            ax.plot(
                x, y, label=dataset, color=color, marker=marker, linestyle=linestyle,
                linewidth=2.0, markersize=5.5,
            )
            # ax.fill_between(
            #     x, np.clip(y - sem, 0.0, 1.0), np.clip(y + sem, 0.0, 1.0),
            #     color=color, alpha=0.10, linewidth=0,
            # )
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
        values = np.array([method_rows.loc[dataset, value_col] for dataset in datasets], dtype=float)
        bars = ax.bar(positions, values, color=colors, width=0.72)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
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


def main() -> None:
    args = _parse_args()
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    if not formats:
        raise ValueError("--formats must contain at least one format")
    args.out.mkdir(parents=True, exist_ok=True)
    summary, auc = _load_outputs(args.input, args.expected_pairs, args.expected_seeds, args.task)
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

    print(f"Cross-dataset outputs -> {args.out}")


if __name__ == "__main__":
    main()
