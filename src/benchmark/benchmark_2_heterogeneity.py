from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from benchmark_4_targets import METHOD_COLORS as COLORS

LEVELS = ("Homogeneous", "Middle", "Full")
LEVEL_MARKERS = {"Homogeneous": "s", "Middle": "D", "Full": "o"}

EXPLAINER_ORDER = (
    "gradient", "ig", "lime", "shap", "pgmexplainer", "gnnexplainer",
)
DISPLAY_NAMES = {
    "gradient": "Gradient",
    "ig": "Integrated Gradients",
    "lime": "LIME",
    "shap": "SHAP",
    "pgmexplainer": "PGMExplainer",
    "gnnexplainer": "GNNExplainer",
}

MODE_DIR_BASE = {
    "Homogeneous": "fidelity_curves_homogeneous",
    "Middle": "fidelity_curves_middle",
    "Full": "fidelity_curves_full",
}


def _mode_dir(level: str, dir_suffix: str = "") -> str:
    """Name of the fidelity_curves_<mode> folder, e.g. 'fidelity_curves_full_n20'."""
    return f"{MODE_DIR_BASE[level]}{dir_suffix}"

AUC_REQUIRED = {
    "explainer", "task",
    "auc_fid_prob_plus_raw", "auc_fid_prob_plus_normalized",
    "auc_fid_prob_minus_raw", "auc_fid_prob_minus_normalized",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined necessity-sufficiency plot for all available datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed"),
        help="Root folder containing one subdirectory per dataset.",
    )
    parser.add_argument(
        "--datasets", default=None,
        help="Comma-separated dataset names to include. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/benchmark_2/all_datasets_benchmark.png"),
        help="Output path (.png, .pdf, or .svg).",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Resolution for raster output.",
    )
    parser.add_argument(
        "--axis-mode", choices=("data", "unit"), default="data",
        help="'data' zooms to observed range; 'unit' fixes axes to [0,1].",
    )
    parser.add_argument(
        "--ncols", type=int, default=4,
        help="Number of columns in the subplot grid.",
    )
    parser.add_argument(
        "--tasks", default=None,
        help=(
            "Comma-separated task names to plot (e.g. 'activity,org_resource'). "
            "Use 'all' to auto-discover and produce one figure per task. "
            "If omitted, macro-averages over all tasks (default)."
        ),
    )
    parser.add_argument(
        "--list-tasks", action="store_true", default=False,
        help="List all available prediction targets and exit (no plots generated).",
    )
    parser.add_argument(
        "--dir-suffix", default="",
        help="Suffix appended to each fidelity_curves_<mode> folder name, "
             "e.g. '_n20' to read fidelity_curves_full_n20/ instead of fidelity_curves_full/.",
    )
    return parser.parse_args()


def _ordered_explainers(values: Iterable[str]) -> list[str]:
    found = set(values)
    ordered = [n for n in EXPLAINER_ORDER if n in found]
    return ordered + sorted(found - set(ordered))


def _discover_datasets(data_dir: Path, dir_suffix: str = "") -> list[str]:
    """Keep every dataset that provides at least two heterogeneity levels."""
    datasets = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        n_modes = sum(
            (d / _mode_dir(level, dir_suffix) / "fidelity_curves_auc.csv").exists()
            for level in LEVELS
        )
        if n_modes >= 2:
            datasets.append(d.name)
    return datasets


def _load_auc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(AUC_REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def _validate_and_load_condition(dataset_dir: Path, level: str, dir_suffix: str = "") -> pd.DataFrame | None:
    mode_dir = dataset_dir / _mode_dir(level, dir_suffix)
    auc_path = mode_dir / "fidelity_curves_auc.csv"
    summary_path = mode_dir / "fidelity_curves_summary.csv"
    checkpoint_path = mode_dir / "curve_checkpoint.csv"

    # summary/checkpoint aren't read here, just required to exist as a signal
    # that the compute_fidelity_curves run for this mode actually finished.
    if not all(p.exists() for p in (auc_path, summary_path, checkpoint_path)):
        return None

    auc = _load_auc(auc_path)

    result = auc.copy()
    result["heterogeneity"] = level
    result["necessity"] = result["auc_fid_prob_plus_normalized"].astype(float).clip(0.0, 1.0)
    result["sufficiency"] = (
        1.0 - result["auc_fid_prob_minus_normalized"].astype(float).abs()
    ).clip(0.0, 1.0)
    return result


def _discover_tasks(data_dir: Path, dataset_names: list[str], dir_suffix: str = "") -> list[str]:
    """Collect all unique task names across all datasets and modes."""
    tasks: list[str] = []
    seen: set[str] = set()
    for name in dataset_names:
        dataset_dir = data_dir / name
        for level in LEVELS:
            auc_path = dataset_dir / _mode_dir(level, dir_suffix) / "fidelity_curves_auc.csv"
            if not auc_path.exists():
                continue
            df = pd.read_csv(auc_path, usecols=["task"])
            for t in df["task"].astype(str).unique():
                if t not in seen:
                    tasks.append(t)
                    seen.add(t)
    return tasks


def _load_dataset(
    data_dir: Path,
    name: str,
    task: str | None = None,
    dir_suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load one dataset and return (plot_frame, score_frame).

    plot_frame:  necessity/sufficiency per (heterogeneity, explainer), averaged
                 over targets when task is None.
    score_frame: per-target S_{E,q} = sqrt(a_{E,q} * b_{E,q}) plus the overall
                 S_E = sum_q w_q S_{E,q} with uniform weights (paper Sec. 4.4).
    """
    dataset_dir = data_dir / name
    frames = []
    for level in LEVELS:
        df = _validate_and_load_condition(dataset_dir, level, dir_suffix)
        if df is not None:
            frames.append(df)
    if len(frames) < 2:
        print(f"Skipping {name}: fewer than 2 modes available")
        return None
    combined = pd.concat(frames, ignore_index=True)
    if task is not None:
        # Filter to the requested task (substring match like benchmark_plot_1.py)
        available = combined["task"].astype(str).unique().tolist()
        matched = [t for t in available if task.lower() in t.lower()]
        if not matched:
            return None
        combined = combined[combined["task"].astype(str) == matched[0]].copy()

    # Per-target score S_{E,q} following the paper's geometric mean.
    per_task = combined[["heterogeneity", "explainer", "task",
                         "necessity", "sufficiency"]].copy()
    per_task["score"] = np.sqrt(per_task["necessity"] * per_task["sufficiency"])

    # Overall S_E: uniform weights w_q = 1/|Q| over the available targets.
    overall = (
        per_task.groupby(["heterogeneity", "explainer"], as_index=False)
        .agg(necessity=("necessity", "mean"), sufficiency=("sufficiency", "mean"),
             score=("score", "mean"))
    )
    score_frame = pd.concat(
        [per_task, overall.assign(task="overall")], ignore_index=True
    )[["heterogeneity", "explainer", "task", "necessity", "sufficiency", "score"]]

    # Plot in the (necessity, sufficiency) plane; macro-average over targets.
    plot_frame = overall[["heterogeneity", "explainer", "necessity", "sufficiency"]].copy()
    return plot_frame, score_frame


def _axis_limits(values: pd.Series, mode: str) -> tuple[float, float]:
    if mode == "unit":
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    pad = max(0.02, (hi - lo) * 0.18)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def _add_score_contours(ax: plt.Axes, xlim: tuple, ylim: tuple) -> None:
    x = np.linspace(max(xlim[0], 1e-6), xlim[1], 300)
    y = np.linspace(max(ylim[0], 1e-6), ylim[1], 300)
    xx, yy = np.meshgrid(x, y)
    score = np.sqrt(xx * yy)
    lo, hi = float(score.min()), float(score.max())
    candidates = np.arange(np.ceil(lo / 0.05) * 0.05, np.floor(hi / 0.05) * 0.05 + 0.001, 0.05)
    levels = candidates[(candidates > lo) & (candidates < hi)]
    if len(levels) == 0:
        levels = np.linspace(lo, hi, 4)[1:-1]
    contours = ax.contour(xx, yy, score, levels=levels, colors="#98A2B3",
                          linewidths=0.6, linestyles="--", alpha=0.7, zorder=0)
    ax.clabel(contours, inline=True, fontsize=7, fmt=lambda v: f"S={v:.2f}", colors="#667085")


def _plot_dataset(ax: plt.Axes, frame: pd.DataFrame, title: str, axis_mode: str) -> None:
    present_levels = [l for l in LEVELS if l in frame["heterogeneity"].values]

    xlim = _axis_limits(frame["necessity"], axis_mode)
    ylim = _axis_limits(frame["sufficiency"], axis_mode)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _add_score_contours(ax, xlim, ylim)

    explainers = _ordered_explainers(frame["explainer"])
    for idx, explainer in enumerate(explainers):
        trajectory = (
            frame[frame["explainer"] == explainer]
            .assign(heterogeneity=lambda d: pd.Categorical(
                d["heterogeneity"], categories=LEVELS, ordered=True))
            .sort_values("heterogeneity")
        )
        color = COLORS.get(explainer, plt.cm.tab10(idx % 10))
        x = trajectory["necessity"].to_numpy(dtype=float)
        y = trajectory["sufficiency"].to_numpy(dtype=float)

        ax.plot(x, y, color=color, linewidth=1.4, alpha=0.9, zorder=2)
        for s in range(len(x) - 1):
            ax.annotate("", xy=(x[s + 1], y[s + 1]), xytext=(x[s], y[s]),
                        arrowprops={"arrowstyle": "-|>", "color": color,
                                    "linewidth": 1.4, "mutation_scale": 9,
                                    "shrinkA": 5, "shrinkB": 5}, zorder=3)
        for row in trajectory.itertuples(index=False):
            ax.scatter(row.necessity, row.sufficiency, s=50,
                       marker=LEVEL_MARKERS[str(row.heterogeneity)],
                       facecolor=color, edgecolor="white", linewidth=0.7, zorder=4)

    ax.grid(True, color="#EAECF0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, fontweight="semibold", pad=5)
    ax.set_xlabel("Necessity", fontsize=10)
    ax.set_ylabel("Sufficiency", fontsize=10)
    ax.tick_params(labelsize=9)


def _build_legend_handles(all_explainers: list[str], present_levels: list[str]) -> tuple:
    explainer_handles = [
        Line2D([0], [0], color=COLORS.get(n, "gray"), linewidth=2.0,
               label=DISPLAY_NAMES.get(n, n))
        for n in all_explainers
    ]
    level_handles = [
        Line2D([0], [0], linestyle="none", marker=LEVEL_MARKERS[l],
               markerfacecolor="#667085", markeredgecolor="white",
               markersize=7, label=l)
        for l in present_levels
    ]
    return explainer_handles, level_handles


def _render_figure(
    data: dict[str, pd.DataFrame],
    title: str,
    output: Path,
    ncols: int,
    axis_mode: str,
    dpi: int
) -> None:
    n = len(data)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": "#344054",
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.4, nrows * 3.6),
                             squeeze=False)

    all_explainers: list[str] = []
    all_levels: list[str] = []
    for df in data.values():
        for e in _ordered_explainers(df["explainer"]):
            if e not in all_explainers:
                all_explainers.append(e)
        for lv in LEVELS:
            if lv in df["heterogeneity"].values and lv not in all_levels:
                all_levels.append(lv)

    for i, (name, df) in enumerate(data.items()):
        row, col = divmod(i, ncols)
        _plot_dataset(axes[row][col], df, name, axis_mode)

    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    exp_handles, lvl_handles = _build_legend_handles(all_explainers, all_levels)
    fig.legend(handles=exp_handles, title="Explainer", loc="lower center",
               ncol=len(all_explainers), bbox_to_anchor=(0.40, 0.02),
               frameon=False, fontsize=10, title_fontsize=11)
    fig.legend(handles=lvl_handles, title="Heterogeneity Level",
               loc="lower center", ncol=len(all_levels),
               bbox_to_anchor=(0.85, 0.02), frameon=False,
               fontsize=10, title_fontsize=11)

#    fig.suptitle(title, fontsize=12, fontweight="semibold", y=1.01)
    fig.subplots_adjust(
    left=0.055,
    right=0.99,
    top=0.97,
    bottom=0.16,   # reserve space for one row of side-by-side legends
    wspace=0.28,
    hspace=0.38,   # important: separates the two rows
)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"Saved → {output}")


def main() -> None:
    args = _parse_args()

    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        dataset_names = _discover_datasets(args.data_dir, args.dir_suffix)

    if not dataset_names:
        raise SystemExit(f"No datasets found under {args.data_dir}")

    print(f"Datasets: {dataset_names}")

    # Handle --list-tasks
    if args.list_tasks:
        all_tasks = _discover_tasks(args.data_dir, dataset_names, args.dir_suffix)
        print(f"\nAvailable prediction targets ({len(all_tasks)}):")
        for i, task in enumerate(all_tasks, 1):
            print(f"  {i}. {task}")
        print("\nUsage examples:")
        print("  PYTHONPATH=. python src/benchmark/benchmark_2_heterogeneity.py --tasks all")
        print(f"  PYTHONPATH=. python src/benchmark/benchmark_2_heterogeneity.py --tasks {','.join(all_tasks[:2])}")
        return

    # Resolve which tasks to plot
    if args.tasks is None:
        task_list = [None]  # None = macro-average
    elif args.tasks.strip().lower() == "all":
        task_list = _discover_tasks(args.data_dir, dataset_names, args.dir_suffix)
        print(f"Auto-discovered tasks: {task_list}")
    else:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    stem = args.output.stem
    suffix = args.output.suffix

    for task in task_list:
        label = task if task is not None else "macro_avg"
        print(f"\n--- Task: {label} ---")

        data: dict[str, pd.DataFrame] = {}
        score_frames: list[pd.DataFrame] = []
        for name in dataset_names:
            loaded = _load_dataset(args.data_dir, name, task, args.dir_suffix)
            if loaded is not None:
                plot_frame, score_frame = loaded
                data[name] = plot_frame
                score_frames.append(score_frame.assign(dataset=name))
            else:
                print(f"  Skipping {name} (task '{label}' not found)")

        if not data:
            print(f"  No datasets available for task '{label}', skipping.")
            continue

        if task is None:
            out_path = args.output
            title = "Fidelity under graph heterogeneity — all datasets\n(macro-average over tasks)"
        else:
            out_path = args.output.parent / f"{stem}_{label}{suffix}"
            title = f"Fidelity under graph heterogeneity — all datasets\nTask: {label}"

        # Export the paper-formalism scores: a, b, S_{E,q} per target and
        # overall S_E per (dataset, heterogeneity, explainer).
        scores = pd.concat(score_frames, ignore_index=True)
        scores = scores[["dataset", "heterogeneity", "explainer", "task",
                         "necessity", "sufficiency", "score"]]
        csv_path = out_path.parent / f"{out_path.stem}_scores.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(csv_path, index=False)
        print(f"Scores → {csv_path}")

        _render_figure(data, title, out_path, args.ncols, args.axis_mode, args.dpi)


if __name__ == "__main__":
    main()
