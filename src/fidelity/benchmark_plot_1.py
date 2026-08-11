from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


LEVELS = ("Homogeneous", "Middle", "Full")
LEVEL_MARKERS = {"Homogeneous": "s", "Middle": "D", "Full": "o"}

EXPLAINER_ORDER = (
    "gradient",
    "ig",
    "lime",
    "shap",
    "pgmexplainer",
    "gnnexplainer",
)

DISPLAY_NAMES = {
    "gradient": "Gradient",
    "ig": "Integrated Gradients",
    "lime": "LIME",
    "shap": "SHAP",
    "pgmexplainer": "PGMExplainer",
    "gnnexplainer": "GNNExplainer",
}

COLORS = {
    "gradient": "#0072B2",
    "ig": "#56B4E9",
    "lime": "#009E73",
    "shap": "#CC79A7",
    "pgmexplainer": "#E69F00",
    "gnnexplainer": "#D55E00",
}

AUC_REQUIRED = {
    "explainer",
    "task",
    "auc_fid_prob_plus_raw",
    "auc_fid_prob_plus_normalized",
    "auc_fid_prob_minus_raw",
    "auc_fid_prob_minus_normalized",
}
SUMMARY_REQUIRED = {
    "explainer",
    "task",
    "top_k",
    "mean_fid_prob_plus",
    "mean_fid_prob_minus",
}
CHECKPOINT_REQUIRED = {
    "explainer",
    "task",
    "top_k",
    "fid_prob_plus",
    "fid_prob_minus",
}


@dataclass(frozen=True)
class ConditionPaths:
    level: str
    auc: Path
    summary: Path
    checkpoint: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot necessity--sufficiency trajectories for one dataset using "
            "homogeneous, middle, and full fidelity result triplets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset value to filter when the CSVs contain a 'dataset' column; "
            "otherwise this is used as the plot title."
        ),
    )

    for prefix in ("homogeneous", "middle", "full"):
        required = prefix != "homogeneous"
        parser.add_argument(
            f"--{prefix}-auc",
            type=Path,
            required=required,
            default=None,
            help=f"{prefix.capitalize()} fidelity_curves_auc CSV.",
        )
        parser.add_argument(
            f"--{prefix}-summary",
            type=Path,
            required=required,
            default=None,
            help=f"{prefix.capitalize()} fidelity_curves_summary CSV.",
        )
        parser.add_argument(
            f"--{prefix}-checkpoint",
            type=Path,
            required=required,
            default=None,
            help=f"{prefix.capitalize()} curve_checkpoint CSV.",
        )

    parser.add_argument(
        "--target",
        default=None,
        help=(
            "Optional exact task/target value. If omitted, coordinates are "
            "macro-averaged across all targets."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("heterogeneity_necessity_sufficiency.png"),
        help="Output path; use .png, .pdf, or .svg.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for raster output.",
    )
    parser.add_argument(
        "--axis-mode",
        choices=("data", "unit"),
        default="data",
        help=(
            "'data' zooms to the observed results; 'unit' fixes both axes to "
            "[0, 1]."
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path, required: set[str], dataset: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if "dataset" in frame.columns:
        available = sorted(frame["dataset"].dropna().astype(str).unique())
        frame = frame[frame["dataset"].astype(str) == dataset].copy()
        if frame.empty:
            raise ValueError(
                f"{path} has no rows for dataset {dataset!r}. "
                f"Available values: {available}"
            )
    else:
        frame = frame.copy()
    return frame


def _assert_unique(frame: pd.DataFrame, keys: list[str], path: Path) -> None:
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, keys].head().to_dict("records")
        raise ValueError(f"{path} has duplicate rows for {keys}: {examples}")


def _validate_triplet(
    auc: pd.DataFrame,
    summary: pd.DataFrame,
    checkpoint: pd.DataFrame,
    paths: ConditionPaths,
    tolerance: float = 1e-9,
) -> None:
    _assert_unique(auc, ["explainer", "task"], paths.auc)
    _assert_unique(summary, ["explainer", "task", "top_k"], paths.summary)

    auc_pairs = set(map(tuple, auc[["explainer", "task"]].astype(str).to_numpy()))
    summary_pairs = set(
        map(tuple, summary[["explainer", "task"]].astype(str).to_numpy())
    )
    checkpoint_pairs = set(
        map(tuple, checkpoint[["explainer", "task"]].astype(str).to_numpy())
    )
    if not (auc_pairs == summary_pairs == checkpoint_pairs):
        raise ValueError(
            f"{paths.level}: explainer-target coverage differs between the "
            "AUC, summary, and checkpoint files."
        )

    checkpoint_means = (
        checkpoint.groupby(["explainer", "task", "top_k"], as_index=False)
        .agg(
            checkpoint_fid_plus=("fid_prob_plus", "mean"),
            checkpoint_fid_minus=("fid_prob_minus", "mean"),
        )
    )
    checked = summary.merge(
        checkpoint_means,
        on=["explainer", "task", "top_k"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (checked["_merge"] == "both").all():
        raise ValueError(
            f"{paths.level}: summary and checkpoint have different "
            "explainer-target-k coverage."
        )

    plus_error = (
        checked["mean_fid_prob_plus"] - checked["checkpoint_fid_plus"]
    ).abs().max()
    minus_error = (
        checked["mean_fid_prob_minus"] - checked["checkpoint_fid_minus"]
    ).abs().max()
    if plus_error > tolerance or minus_error > tolerance:
        raise ValueError(
            f"{paths.level}: checkpoint means do not reproduce the summary "
            f"(max errors: Fidelity+={plus_error:g}, Fidelity-={minus_error:g})."
        )

    integrations: list[dict[str, object]] = []
    for (explainer, task), group in summary.groupby(["explainer", "task"]):
        group = group.sort_values("top_k")
        k = group["top_k"].to_numpy(dtype=float)
        if len(k) < 2 or np.any(np.diff(k) <= 0):
            raise ValueError(
                f"{paths.level}: {explainer}/{task} needs at least two unique, "
                "increasing top_k values."
            )
        integrations.append(
            {
                "explainer": explainer,
                "task": task,
                "recomputed_plus": np.trapz(
                    group["mean_fid_prob_plus"].to_numpy(dtype=float), k
                ),
                "recomputed_minus": np.trapz(
                    group["mean_fid_prob_minus"].to_numpy(dtype=float), k
                ),
            }
        )

    integrated = auc.merge(
        pd.DataFrame(integrations),
        on=["explainer", "task"],
        how="left",
        validate="one_to_one",
    )
    plus_auc_error = (
        integrated["auc_fid_prob_plus_raw"] - integrated["recomputed_plus"]
    ).abs().max()
    minus_auc_error = (
        integrated["auc_fid_prob_minus_raw"] - integrated["recomputed_minus"]
    ).abs().max()
    if plus_auc_error > tolerance or minus_auc_error > tolerance:
        raise ValueError(
            f"{paths.level}: summary curves do not reproduce the raw AUC file "
            f"(max errors: Fidelity+={plus_auc_error:g}, "
            f"Fidelity-={minus_auc_error:g})."
        )


def load_condition(paths: ConditionPaths, dataset: str) -> pd.DataFrame:
    auc = _read_csv(paths.auc, AUC_REQUIRED, dataset)
    summary = _read_csv(paths.summary, SUMMARY_REQUIRED, dataset)
    checkpoint = _read_csv(paths.checkpoint, CHECKPOINT_REQUIRED, dataset)
    _validate_triplet(auc, summary, checkpoint, paths)

    result = auc.copy()
    result["explainer"] = result["explainer"].astype(str)
    result["task"] = result["task"].astype(str)
    result["heterogeneity"] = paths.level

    # These are the two non-negative components used by the combined score:
    # necessity = max(AUC+, 0)
    # sufficiency = max(1 - |AUC-|, 0)
    result["necessity"] = result[
        "auc_fid_prob_plus_normalized"
    ].astype(float).clip(lower=0.0, upper=1.0)
    result["sufficiency"] = (
        1.0
        - result["auc_fid_prob_minus_normalized"].astype(float).abs()
    ).clip(lower=0.0, upper=1.0)
    result["target_score"] = np.sqrt(
        result["necessity"] * result["sufficiency"]
    )
    return result


def _ordered_explainers(values: Iterable[str]) -> list[str]:
    found = set(values)
    ordered = [name for name in EXPLAINER_ORDER if name in found]
    return ordered + sorted(found - set(ordered))


def prepare_plot_data(
    frame: pd.DataFrame, target: str | None
) -> tuple[pd.DataFrame, str]:
    if target is not None:
        available = sorted(frame["task"].unique())
        frame = frame[frame["task"] == target].copy()
        if frame.empty:
            raise ValueError(
                f"Target {target!r} was not found. Available targets: {available}"
            )
        subtitle = f"Target: {target}"
    else:
        frame = (
            frame.groupby(["heterogeneity", "explainer"], as_index=False)
            .agg(
                necessity=("necessity", "mean"),
                sufficiency=("sufficiency", "mean"),
                n_targets=("task", "nunique"),
            )
        )
        counts = sorted(frame["n_targets"].unique())
        if len(counts) != 1:
            raise ValueError(
                "The number of targets differs across explainers or "
                f"heterogeneity levels: {counts}"
            )
        subtitle = f"Macro-average across {counts[0]} prediction targets"

    # This score corresponds exactly to the plotted x/y coordinates and hence
    # to the iso-score contours.
    frame["plotted_score"] = np.sqrt(
        frame["necessity"] * frame["sufficiency"]
    )
    return frame, subtitle


def _axis_limits(values: pd.Series, mode: str) -> tuple[float, float]:
    if mode == "unit":
        return 0.0, 1.0
    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo
    pad = max(0.015, span * 0.18)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def _add_score_contours(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    x = np.linspace(max(xlim[0], 1e-6), xlim[1], 350)
    y = np.linspace(max(ylim[0], 1e-6), ylim[1], 350)
    xx, yy = np.meshgrid(x, y)
    score = np.sqrt(xx * yy)

    low = float(score.min())
    high = float(score.max())
    candidate_levels = np.arange(
        np.ceil(low / 0.05) * 0.05,
        np.floor(high / 0.05) * 0.05 + 0.001,
        0.05,
    )
    levels = candidate_levels[(candidate_levels > low) & (candidate_levels < high)]
    if len(levels) == 0:
        levels = np.linspace(low, high, 4)[1:-1]
    contours = ax.contour(
        xx,
        yy,
        score,
        levels=levels,
        colors="#98A2B3",
        linewidths=0.85,
        linestyles="--",
        alpha=0.8,
        zorder=0,
    )
    ax.clabel(
        contours,
        inline=True,
        fontsize=8,
        fmt=lambda value: f"S={value:.2f}",
        colors="#667085",
    )


def plot_trajectories(
    frame: pd.DataFrame,
    dataset: str,
    subtitle: str,
    output: Path,
    dpi: int,
    axis_mode: str,
) -> None:
    present_levels = [l for l in LEVELS if l in frame["heterogeneity"].values]
    expected = {
        (level, explainer)
        for level in present_levels
        for explainer in frame["explainer"].unique()
    }
    actual = set(zip(frame["heterogeneity"], frame["explainer"]))
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"Missing plotted condition/explainer combinations: {missing}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.edgecolor": "#344054",
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    xlim = _axis_limits(frame["necessity"], axis_mode)
    ylim = _axis_limits(frame["sufficiency"], axis_mode)
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _add_score_contours(ax, xlim, ylim)

    explainers = _ordered_explainers(frame["explainer"])
    for index, explainer in enumerate(explainers):
        trajectory = (
            frame[frame["explainer"] == explainer]
            .assign(
                heterogeneity=lambda d: pd.Categorical(
                    d["heterogeneity"], categories=LEVELS, ordered=True
                )
            )
            .sort_values("heterogeneity")
        )
        color = COLORS.get(explainer, plt.cm.tab10(index % 10))
        x = trajectory["necessity"].to_numpy(dtype=float)
        y = trajectory["sufficiency"].to_numpy(dtype=float)

        ax.plot(x, y, color=color, linewidth=1.8, alpha=0.9, zorder=2)
        for start in range(len(x) - 1):
            ax.annotate(
                "",
                xy=(x[start + 1], y[start + 1]),
                xytext=(x[start], y[start]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.8,
                    "mutation_scale": 11,
                    "shrinkA": 7,
                    "shrinkB": 7,
                },
                zorder=3,
            )
        for row in trajectory.itertuples(index=False):
            ax.scatter(
                row.necessity,
                row.sufficiency,
                s=78,
                marker=LEVEL_MARKERS[str(row.heterogeneity)],
                facecolor=color,
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )

    ax.grid(True, color="#EAECF0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Necessity component: $\max(\mathrm{AUC}^{+}, 0)$")
    ax.set_ylabel(
        r"Sufficiency component: $\max(1-|\mathrm{AUC}^{-}|, 0)$"
    )
    ax.set_title(
        f"{dataset}: fidelity under graph heterogeneity\n{subtitle}",
        loc="left",
        pad=14,
        fontweight="semibold",
    )

    explainer_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS.get(name, plt.cm.tab10(i % 10)),
            linewidth=2.2,
            label=DISPLAY_NAMES.get(name, name),
        )
        for i, name in enumerate(explainers)
    ]
    level_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=LEVEL_MARKERS[level],
            markerfacecolor="#667085",
            markeredgecolor="white",
            markersize=8,
            label=level,
        )
        for level in present_levels
    ]
    first_legend = ax.legend(
        handles=explainer_handles,
        title="Explainer",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.33),
        frameon=False,
        borderaxespad=0,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=level_handles,
        title="Graph representation",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.03),
        frameon=False,
        borderaxespad=0,
    )
    ax.text(
        1.02,
        0.98,
        "Arrows:\n" + " → ".join(present_levels) + "\n\nHigher is better on both axes.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#475467",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    suffix = args.output.suffix.lower()
    if suffix not in {".png", ".pdf", ".svg"}:
        raise ValueError("--output must end in .png, .pdf, or .svg")

    conditions = []
    if args.homogeneous_auc is not None:
        conditions.append(ConditionPaths(
            "Homogeneous",
            args.homogeneous_auc,
            args.homogeneous_summary,
            args.homogeneous_checkpoint,
        ))
    conditions += [
        ConditionPaths(
            "Middle",
            args.middle_auc,
            args.middle_summary,
            args.middle_checkpoint,
        ),
        ConditionPaths(
            "Full",
            args.full_auc,
            args.full_summary,
            args.full_checkpoint,
        ),
    ]
    combined = pd.concat(
        [load_condition(paths, args.dataset) for paths in conditions],
        ignore_index=True,
    )
    plot_data, subtitle = prepare_plot_data(combined, args.target)
    plot_trajectories(
        plot_data,
        dataset=args.dataset,
        subtitle=subtitle,
        output=args.output,
        dpi=args.dpi,
        axis_mode=args.axis_mode,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
