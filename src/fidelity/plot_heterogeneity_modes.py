"""Plot probability-fidelity curves across heterogeneity modes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODES = ("full", "middle", "homogeneous")
DEFAULT_METHODS = ("gradient", "ig", "lime", "shap", "gnnexplainer", "pgmexplainer")
METHOD_LABELS = {
    "gradient": "Gradient",
    "ig": "IG",
    "lime": "LIME",
    "shap": "SHAP",
    "gnnexplainer": "GNNExplainer",
    "pgmexplainer": "PGMExplainer",
}
MODE_STYLES: Dict[str, tuple[str, str, str]] = {
    "full": ("#1f77b4", "o", "-"),
    "middle": ("#ff7f0e", "s", "--"),
    "homogeneous": ("#2ca02c", "^", "-."),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one BPIC/dataset task across full, middle, and homogeneous "
            "heterogeneity-mode fidelity summaries."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Folder containing full/, middle/, homogeneous/ subfolders. Each mode "
            "may contain fidelity_curves_summary.csv directly or under merged/."
        ),
    )
    parser.add_argument("--full", type=Path, default=None, help="Full-mode fidelity_curves_summary.csv.")
    parser.add_argument("--middle", type=Path, default=None, help="Middle-mode fidelity_curves_summary.csv.")
    parser.add_argument(
        "--homogeneous",
        type=Path,
        default=None,
        help="Homogeneous-mode fidelity_curves_summary.csv.",
    )
    parser.add_argument("--task", default="activity", help="Task to plot.")
    parser.add_argument("--dataset", default="BPIC13_O", help="Dataset label for titles.")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated explainability methods to plot.",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=None,
        help="Optional pair-count check. Uses n_pairs or n_samples when present.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output folder.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated formats, e.g. png,pdf.")
    return parser.parse_args()


def _parse_csv(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, Sequence):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise TypeError(f"Expected comma-separated string or sequence, got {type(raw).__name__}")


def _safe_stem(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def _summary_path_for_mode(args: argparse.Namespace, mode: str) -> Path:
    explicit = getattr(args, mode)
    if explicit is not None:
        return explicit
    if args.input_root is None:
        raise ValueError("Pass --input-root or explicit --full/--middle/--homogeneous summary paths")

    direct = args.input_root / mode / "fidelity_curves_summary.csv"
    merged = args.input_root / mode / "merged" / "fidelity_curves_summary.csv"
    if direct.is_file():
        return direct
    if merged.is_file():
        return merged
    raise FileNotFoundError(
        f"Missing summary for {mode}: expected {direct} or {merged}"
    )


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


def _pair_count_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in ("n_pairs", "n_samples") if column in df.columns]


def _validate_pair_count(df: pd.DataFrame, expected_pairs: int | None, mode: str) -> None:
    pair_columns = _pair_count_columns(df)
    if not pair_columns:
        print(f"{mode}: pair count not available in summary")
        return

    observed = sorted(
        {
            int(value)
            for column in pair_columns
            for value in pd.to_numeric(df[column], errors="coerce").dropna().unique()
        }
    )
    print(f"{mode}: pair count {observed}")
    if expected_pairs is not None and observed != [int(expected_pairs)]:
        raise ValueError(
            f"{mode} has pair count {observed}, expected [{int(expected_pairs)}]"
        )


def _read_mode_summary(
    path: Path,
    mode: str,
    task: str,
    methods: Sequence[str],
    expected_pairs: int | None,
) -> pd.DataFrame:
    required = {"explainer", "task", "top_k", "mean_fid_prob_plus", "mean_fid_prob_minus"}
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    available_tasks = sorted(df["task"].astype(str).unique())
    resolved_task = _resolve_task(available_tasks, task)
    if resolved_task is None:
        raise ValueError(f"{path} has no rows for task {task!r}; available={available_tasks}")

    df = df[df["task"].astype(str) == resolved_task].copy()
    df["explainer"] = df["explainer"].astype(str)
    missing_methods = [method for method in methods if method not in set(df["explainer"])]
    if missing_methods:
        raise ValueError(f"{path} is missing methods for task {resolved_task}: {missing_methods}")
    df = df[df["explainer"].isin(methods)].copy()

    for column in ("top_k", "mean_fid_prob_plus", "mean_fid_prob_minus"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if not np.isfinite(df[["top_k", "mean_fid_prob_plus", "mean_fid_prob_minus"]]).all().all():
        raise ValueError(f"{path} contains non-finite probability-fidelity values")

    if "mean_one_minus_fid_prob_minus" not in df.columns:
        df["mean_one_minus_fid_prob_minus"] = 1.0 - df["mean_fid_prob_minus"]
    else:
        df["mean_one_minus_fid_prob_minus"] = pd.to_numeric(
            df["mean_one_minus_fid_prob_minus"], errors="coerce"
        )
    if not np.isfinite(df["mean_one_minus_fid_prob_minus"]).all():
        raise ValueError(f"{path} contains non-finite sufficiency values")

    _validate_pair_count(df, expected_pairs, mode)
    df["heterogeneity_mode"] = mode
    df["task"] = resolved_task
    print(f"{mode}: {path} task={resolved_task}")
    return df.sort_values(["explainer", "top_k"])


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
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def _plot_metric(
    df: pd.DataFrame,
    methods: Sequence[str],
    value_col: str,
    title: str,
    ylabel: str,
    stem: str,
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    cols = 3 if len(methods) > 4 else 2
    rows = math.ceil(len(methods) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15.5, 4.2 * rows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    top_k = np.sort(df["top_k"].unique())

    for ax, method in zip(axes_flat, methods):
        for mode in MODES:
            part = df[
                (df["explainer"] == method) & (df["heterogeneity_mode"] == mode)
            ].sort_values("top_k")
            color, marker, linestyle = MODE_STYLES[mode]
            ax.plot(
                part["top_k"].to_numpy(dtype=float),
                part[value_col].to_numpy(dtype=float),
                label=mode,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.2,
                markersize=5.5,
            )
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xticks(top_k)

    values = df[value_col].to_numpy(dtype=float)
    lo = min(0.0, float(np.nanmin(values)))
    hi = max(0.0, float(np.nanmax(values)))
    pad = max(0.02, 0.08 * (hi - lo))
    for ax in axes_flat[:len(methods)]:
        ax.set_ylim(lo - pad, hi + pad)

    for ax in axes_flat[len(methods):]:
        ax.axis("off")
    for ax in axes_flat[-cols:]:
        if ax.has_data():
            ax.set_xlabel("Top-k nodes")
    for ax in axes_flat[::cols]:
        ax.set_ylabel(ylabel)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODES), frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, out_dir, stem, formats)


def main() -> None:
    args = _parse_args()
    methods = _parse_csv(args.methods)
    formats = tuple(fmt.lower() for fmt in _parse_csv(args.formats))
    if not methods:
        raise ValueError("--methods must contain at least one method")
    if not formats:
        raise ValueError("--formats must contain at least one format")

    frames = [
        _read_mode_summary(
            _summary_path_for_mode(args, mode),
            mode,
            args.task,
            methods,
            args.expected_pairs,
        )
        for mode in MODES
    ]
    df = pd.concat(frames, ignore_index=True)
    _setup_style()
    task_label = str(df["task"].iloc[0])
    task_stem = _safe_stem(task_label)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"{task_stem}_probability_fidelity_heterogeneity_modes.csv", index=False)

    specs = (
        (
            "mean_fid_prob_plus",
            f"{args.dataset} ({task_label}): Probability Fidelity+ by Heterogeneity Mode",
            "Mean signed probability drop",
            f"{task_stem}_prob_fidelity_plus_heterogeneity_modes",
        ),
        (
            "mean_fid_prob_minus",
            f"{args.dataset} ({task_label}): Probability Fidelity- by Heterogeneity Mode",
            "Mean signed probability drop",
            f"{task_stem}_prob_fidelity_minus_heterogeneity_modes",
        ),
        (
            "mean_one_minus_fid_prob_minus",
            f"{args.dataset} ({task_label}): 1 - Probability Fidelity- by Heterogeneity Mode",
            "Mean 1 - signed probability drop",
            f"{task_stem}_prob_one_minus_fidelity_minus_heterogeneity_modes",
        ),
    )
    for spec in specs:
        _plot_metric(df, methods, *spec, args.out, formats)


if __name__ == "__main__":
    main()
