"""Benchmark 1: overall fidelity score S_E per explainer.

Consumes results/benchmark_4/target_profile_scores.csv (produced by
benchmark_4_targets.py) plus data/processed/<dataset>/evaluation.txt and the
raw curve checkpoints (data/processed/<dataset>/fidelity_curves_full/curve_checkpoint.csv),
and emits everything related to the overall score S_E and prediction
performance:

  - explainer_summary.csv        mean score, mean rank, wins per explainer
  - overall_heatmap.{fmt}        explainer x dataset heatmap of Overall S_E
  - explainer_boxplot.{fmt}      distribution of Overall S_E per explainer
  - per_target_heatmap.{fmt}     explainer x target mean S_{E,q}
  - prediction_performance.csv   model accuracy/F1 per dataset and target
  - prediction_performance.{fmt} heatmap of model accuracy per dataset/target
  - prediction_vs_fidelity.{fmt} scatter of model accuracy vs S_{E,q}
  - explainer_rank_heatmap.{fmt}          per-dataset rank of each explainer
  - explainer_pairwise_wins.{fmt}         head-to-head win counts across datasets
  - explainer_necessity_sufficiency.{fmt} decomposition of S_E into its two factors
  - explainer_necessity_sufficiency.csv

Run benchmark_4_targets.py first to produce target_profile_scores.csv.
"""

from __future__ import annotations

import argparse
import ast
import re
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_4_targets import (
    DEFAULT_DATASETS,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT as TARGETS_OUTPUT,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    METHODS,
    TARGET_LABELS,
    TARGETS,
    _AUC_FN,
    _canonical_target,
    _pair_matrices,
    checkpoint_subpath,
)

DEFAULT_OUTPUT = Path("results/benchmark_1")
TARGET_MARKERS = {"activity": "o", "resource": "s", "role": "^"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scores",
        type=Path,
        default=TARGETS_OUTPUT / "target_profile_scores.csv",
        help="CSV produced by benchmark_4_targets.py.",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--explainers", default=",".join(METHODS))
    parser.add_argument("--formats", default="png")
    parser.add_argument(
        "--dir-suffix", default="",
        help="Suffix appended to the fidelity_curves_full/ folder name, "
             "e.g. '_n20' to read fidelity_curves_full_n20/curve_checkpoint.csv.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prediction performance from evaluation.txt
# ---------------------------------------------------------------------------

_ATTR_RE = re.compile(r"^\s+(\S+):\s*(\{.*\})\s*$")


def _parse_evaluation(path: Path) -> pd.DataFrame:
    """Extract per-target classification metrics from one evaluation.txt."""
    rows: List[Dict[str, object]] = []
    pending: str | None = None
    for line in path.read_text().splitlines():
        if line.startswith("== Activity"):
            pending = "activity"
            continue
        if line.startswith("=="):
            pending = None
        if pending and line.strip().startswith("{"):
            metrics = ast.literal_eval(line.strip())
            rows.append({"task": pending, **metrics})
            pending = None
            continue
        m = _ATTR_RE.match(line)
        if m:
            rows.append({"task": m.group(1), **ast.literal_eval(m.group(2))})
    df = pd.DataFrame(rows)
    df["target"] = df["task"].map(_canonical_target)
    return df.dropna(subset=["target"])


def _load_prediction_performance(
    input_dir: Path, datasets: Sequence[str]
) -> pd.DataFrame:
    frames = []
    for dataset in datasets:
        path = input_dir / dataset / "evaluation.txt"
        if not path.is_file():
            warnings.warn(f"Missing evaluation.txt for {dataset}; skipping")
            continue
        df = _parse_evaluation(path)
        # If several raw tasks map to one target, keep the best-predicted one
        # (the fidelity pipeline explains one head per target).
        df = df.sort_values("acc", ascending=False).drop_duplicates("target")
        df.insert(0, "dataset", dataset)
        frames.append(df[["dataset", "target", "task", "n", "acc", "precision", "recall", "f1"]])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str]) -> None:
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Plot    -> {path}")
    plt.close(fig)


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    fmt: str = "{:.2f}",
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> None:
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    threshold = np.nanmin(data) + 0.55 * (np.nanmax(data) - np.nanmin(data))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=7, color="0.4")
                continue
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if v < threshold else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)


def _plot_overall_heatmap(
    scores: pd.DataFrame,
    datasets: Sequence[str],
    explainers: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    overall = scores[scores["target"] == "overall"]
    matrix = overall.pivot_table(index="explainer", columns="dataset", values="score")
    matrix = matrix.reindex(index=list(explainers), columns=list(datasets))
    matrix["Mean"] = matrix.mean(axis=1)
    matrix = matrix.sort_values("Mean", ascending=False)
    matrix.index = [METHOD_LABELS.get(e, e) for e in matrix.index]

    fig, ax = plt.subplots(figsize=(0.95 * matrix.shape[1] + 2.2, 0.5 * matrix.shape[0] + 1.6))
    _annotated_heatmap(ax, matrix, vmax=float(np.nanmax(matrix.to_numpy())))
    ax.set_title("Overall fidelity score $S_E$ per explainer and dataset", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "overall_heatmap", formats)


def _plot_explainer_boxplot(
    scores: pd.DataFrame,
    explainers: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    overall = scores[scores["target"] == "overall"]
    order = (
        overall.groupby("explainer")["score"].mean().reindex(list(explainers)).sort_values(ascending=False)
    )
    ranks = overall.pivot_table(index="dataset", columns="explainer", values="score").rank(
        axis=1, ascending=False
    )
    mean_rank = ranks.mean()

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, explainer in enumerate(order.index):
        vals = overall.loc[overall["explainer"] == explainer, "score"].to_numpy()
        color = METHOD_COLORS.get(explainer, "0.3")
        bp = ax.boxplot(
            vals,
            positions=[i],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black"},
        )
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.35)
        jitter = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.18
        ax.scatter(
            np.full(len(vals), float(i)) + jitter,
            vals,
            s=22,
            color=color,
            marker=METHOD_MARKERS.get(explainer, "o"),
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.text(
            i,
            ax.get_ylim()[0],
            f"rank {mean_rank.get(explainer, np.nan):.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.25",
        )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([METHOD_LABELS.get(e, e) for e in order.index], rotation=20, ha="right")
    ax.set_ylabel("Overall $S_E$ across datasets")
    ax.set_title("Distribution of Overall $S_E$ per explainer (label: mean rank, 1=best)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    _save(fig, out_dir, "explainer_boxplot", formats)


def _plot_per_target_heatmap(
    scores: pd.DataFrame,
    explainers: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    per_target = scores[scores["target"].isin(TARGETS)]
    matrix = per_target.pivot_table(index="explainer", columns="target", values="score")
    matrix = matrix.reindex(index=list(explainers), columns=list(TARGETS))
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    matrix.index = [METHOD_LABELS.get(e, e) for e in matrix.index]
    matrix.columns = [TARGET_LABELS.get(t, t) for t in matrix.columns]

    fig, ax = plt.subplots(figsize=(4.6, 0.5 * matrix.shape[0] + 1.6))
    _annotated_heatmap(ax, matrix, vmax=float(np.nanmax(matrix.to_numpy())))
    ax.set_title("Mean $S_{E,q}$ per target (averaged over datasets)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "per_target_heatmap", formats)


def _plot_prediction_performance(
    perf: pd.DataFrame,
    datasets: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    matrix = perf.pivot_table(index="target", columns="dataset", values="acc")
    matrix = matrix.reindex(index=list(TARGETS), columns=list(datasets))
    matrix.index = [TARGET_LABELS.get(t, t) for t in matrix.index]

    fig, ax = plt.subplots(figsize=(0.95 * matrix.shape[1] + 2.2, 0.5 * matrix.shape[0] + 1.6))
    _annotated_heatmap(ax, matrix, cmap="cividis", vmax=1.0)
    ax.set_title("GNN prediction accuracy per dataset and target", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "prediction_performance", formats)


def _plot_prediction_vs_fidelity(
    scores: pd.DataFrame,
    perf: pd.DataFrame,
    explainers: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    per_target = scores[scores["target"].isin(TARGETS)]
    merged = per_target.merge(perf[["dataset", "target", "acc"]], on=["dataset", "target"])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for explainer in explainers:
        sub = merged[merged["explainer"] == explainer]
        if sub.empty:
            continue
        for target in TARGETS:
            st = sub[sub["target"] == target]
            if st.empty:
                continue
            ax.scatter(
                st["acc"],
                st["score"],
                color=METHOD_COLORS.get(explainer, "0.3"),
                marker=TARGET_MARKERS.get(target, "o"),
                s=34,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.3,
            )
    rho = merged["acc"].corr(merged["score"], method="spearman")
    # Least-squares trend line over all points.
    coef = np.polyfit(merged["acc"], merged["score"], 1)
    xs = np.linspace(float(merged["acc"].min()), float(merged["acc"].max()), 50)
    ax.plot(xs, np.polyval(coef, xs), color="0.3", linestyle="--", linewidth=1.0)

    method_handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker="o", linestyle="none",
                   markersize=6, label=METHOD_LABELS[m])
        for m in explainers if m in METHOD_COLORS
    ]
    target_handles = [
        plt.Line2D([], [], color="0.4", marker=TARGET_MARKERS[t], linestyle="none",
                   markersize=6, label=TARGET_LABELS[t])
        for t in TARGETS
    ]
    leg1 = ax.legend(handles=method_handles, loc="upper left", fontsize=7, frameon=False, title="Explainer", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=target_handles, loc="lower right", fontsize=7, frameon=False, title="Target", title_fontsize=8)

    ax.set_xlabel("Model prediction accuracy")
    ax.set_ylabel(r"Fidelity score $S_{E,q}$")
    ax.set_title(
        f"Prediction quality vs. explanation fidelity (Spearman $\\rho$ = {rho:.2f})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    _save(fig, out_dir, "prediction_vs_fidelity", formats)


# ---------------------------------------------------------------------------
# Figure: per-dataset rank heatmap
# ---------------------------------------------------------------------------


def _plot_rank_heatmap(
    scores: pd.DataFrame,
    datasets: Sequence[str],
    explainers: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    overall = scores[scores["target"] == "overall"]
    pivot = overall.pivot_table(index="dataset", columns="explainer", values="score")
    ranks = pivot.rank(axis=1, ascending=False).T
    ranks = ranks.reindex(index=list(explainers), columns=list(datasets))
    ranks["Mean"] = ranks.mean(axis=1)
    ranks = ranks.sort_values("Mean")
    ranks.index = [METHOD_LABELS.get(e, e) for e in ranks.index]

    data = ranks.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.85 * ranks.shape[1] + 2.0, 0.5 * ranks.shape[0] + 1.4))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(explainers))
    ax.set_xticks(range(ranks.shape[1]))
    ax.set_xticklabels(ranks.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(ranks.shape[0]))
    ax.set_yticklabels(ranks.index, fontsize=8)
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            v = data[i, j]
            txt = f"{v:.1f}" if ranks.columns[j] == "Mean" else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    fontweight="bold" if ranks.columns[j] == "Mean" else "normal")
    ax.axvline(ranks.shape[1] - 1.5, color="black", linewidth=1.0)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Rank (1 = best)")
    ax.set_title("Per-dataset rank of Overall $S_E$ (1 = best)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "explainer_rank_heatmap", formats)


# ---------------------------------------------------------------------------
# Figure: pairwise win matrix
# ---------------------------------------------------------------------------


def _plot_pairwise_wins(
    scores: pd.DataFrame,
    explainers: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    overall = scores[scores["target"] == "overall"]
    pivot = overall.pivot_table(index="dataset", columns="explainer", values="score")
    order = pivot.mean().reindex(list(explainers)).sort_values(ascending=False).index
    n = len(order)
    wins = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if a == b:
                continue
            both = pivot[[a, b]].dropna()
            wins[i, j] = (both[a] > both[b]).sum()
    total = int(pivot.dropna().shape[0])

    fig, ax = plt.subplots(figsize=(0.85 * n + 2.2, 0.6 * n + 1.4))
    im = ax.imshow(wins, cmap="RdYlGn", aspect="auto", vmin=0, vmax=total)
    ax.set_xticks(range(n))
    ax.set_xticklabels([METHOD_LABELS.get(e, e) for e in order], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([METHOD_LABELS.get(e, e) for e in order], fontsize=8)
    for i in range(n):
        for j in range(n):
            if np.isnan(wins[i, j]):
                ax.text(j, i, "-", ha="center", va="center", fontsize=8, color="0.4")
            else:
                ax.text(j, i, f"{wins[i, j]:.0f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=f"Wins (out of {total} datasets)")
    ax.set_xlabel("... beats column explainer", fontsize=8)
    ax.set_title(f"Head-to-head wins on Overall $S_E$ across {total} datasets", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "explainer_pairwise_wins", formats)


# ---------------------------------------------------------------------------
# Figure: necessity vs. sufficiency decomposition
# ---------------------------------------------------------------------------


def _decompose_dataset(
    checkpoint_path: Path, explainers: Sequence[str]
) -> pd.DataFrame:
    """Per (explainer, target): a = nAUC(fid+) [necessity], b = 1-|nAUC(fid-)| [sufficiency]."""
    df = pd.read_csv(checkpoint_path)
    df["target"] = df["task"].map(_canonical_target)
    df = df.dropna(subset=["target"])
    df = df[df["explainer"].isin(explainers)]
    rows: List[Dict[str, object]] = []
    for (explainer, target), group in df.groupby(["explainer", "target"]):
        plus, minus, x = _pair_matrices(group)
        if plus.shape[0] == 0 or len(x) < 2:
            continue
        span = float(x.max() - x.min())
        auc_plus = float(_AUC_FN(plus.mean(axis=0), x)) / span
        auc_minus = float(_AUC_FN(minus.mean(axis=0), x)) / span
        rows.append(
            {
                "explainer": explainer,
                "target": target,
                "necessity": float(np.clip(auc_plus, 0.0, 1.0)),
                "sufficiency": 1.0 - min(1.0, abs(auc_minus)),
            }
        )
    return pd.DataFrame(rows)


def _load_decomposition(
    input_dir: Path, datasets: Sequence[str], explainers: Sequence[str], dir_suffix: str = ""
) -> pd.DataFrame:
    frames = []
    for dataset in datasets:
        path = input_dir / dataset / checkpoint_subpath(dir_suffix)
        if not path.is_file():
            warnings.warn(f"Missing checkpoint for {dataset}; skipping")
            continue
        df = _decompose_dataset(path, explainers)
        df.insert(0, "dataset", dataset)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _plot_necessity_sufficiency(
    decomp: pd.DataFrame,
    explainers: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    # Overall per dataset: uniform mean over targets (mirrors Overall S_E weighting).
    overall = (
        decomp.groupby(["dataset", "explainer"])[["necessity", "sufficiency"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    # Iso-S_E contours: sqrt(a * b) = const.
    a_grid = np.linspace(0.005, 1.0, 200)
    b_grid = np.linspace(0.005, 1.0, 200)
    A, B = np.meshgrid(a_grid, b_grid)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    cs = ax.contour(A, B, np.sqrt(A * B), levels=levels, colors="0.75", linewidths=0.7)
    ax.clabel(cs, fmt=lambda v: f"$S_E$={v:.1f}", fontsize=6, colors="0.5")

    for explainer in explainers:
        sub = overall[overall["explainer"] == explainer]
        if sub.empty:
            continue
        color = METHOD_COLORS.get(explainer, "0.3")
        marker = METHOD_MARKERS.get(explainer, "o")
        ax.scatter(sub["necessity"], sub["sufficiency"], s=26, color=color,
                   marker=marker, alpha=0.45, edgecolor="none", zorder=2)
        ax.scatter(
            sub["necessity"].mean(),
            sub["sufficiency"].mean(),
            s=170,
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
            label=METHOD_LABELS.get(explainer, explainer),
        )
    ax.set_xlabel("Necessity  $a$ = nAUC of Fid$^+$ (removing explanation breaks prediction)")
    ax.set_ylabel("Sufficiency  $b$ = 1 - |nAUC of Fid$^-$| (explanation alone preserves it)")
    ax.set_title("Why explainers rank as they do: $S_E = \\sqrt{a \\cdot b}$\n"
                 "(small: per dataset, large: mean across datasets)", fontsize=10)
    ax.set_xlim(0, max(0.65, overall["necessity"].max() + 0.05))
    ax.set_ylim(min(0.5, overall["sufficiency"].min() - 0.05), 1.0)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    _save(fig, out_dir, "explainer_necessity_sufficiency", formats)

    csv_path = out_dir / "explainer_necessity_sufficiency.csv"
    overall.to_csv(csv_path, index=False)
    print(f"CSV     -> {csv_path}")
    print(
        overall.groupby("explainer")[["necessity", "sufficiency"]]
        .mean()
        .sort_values("necessity", ascending=False)
        .to_string(float_format=lambda v: f"{v:.3f}")
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _summary_table(scores: pd.DataFrame, explainers: Sequence[str]) -> pd.DataFrame:
    overall = scores[scores["target"] == "overall"]
    pivot = overall.pivot_table(index="dataset", columns="explainer", values="score")
    ranks = pivot.rank(axis=1, ascending=False)
    rows = []
    for explainer in explainers:
        if explainer not in pivot.columns:
            continue
        row: Dict[str, object] = {
            "explainer": explainer,
            "mean_overall_S": pivot[explainer].mean(),
            "std_overall_S": pivot[explainer].std(),
            "mean_rank": ranks[explainer].mean(),
            "n_wins": int((ranks[explainer] == 1.0).sum()),
            "n_datasets": int(pivot[explainer].notna().sum()),
        }
        for target in TARGETS:
            sub = scores[(scores["target"] == target) & (scores["explainer"] == explainer)]
            row[f"mean_S_{target}"] = sub["score"].mean()
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("mean_overall_S", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    args = _parse_args()
    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    explainers = [e.strip() for e in str(args.explainers).split(",") if e.strip()]
    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]

    if not args.scores.is_file():
        raise SystemExit(f"Scores file not found: {args.scores}. Run benchmark_4_targets.py first.")
    scores = pd.read_csv(args.scores)
    scores = scores[scores["dataset"].isin(datasets) & scores["explainer"].isin(explainers)]

    perf = _load_prediction_performance(args.input, datasets)

    args.out.mkdir(parents=True, exist_ok=True)

    perf_path = args.out / "prediction_performance.csv"
    perf.to_csv(perf_path, index=False)
    print(f"CSV     -> {perf_path}")

    summary = _summary_table(scores, explainers)
    summary_path = args.out / "explainer_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"CSV     -> {summary_path}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    _plot_overall_heatmap(scores, datasets, explainers, args.out, formats)
    _plot_explainer_boxplot(scores, explainers, args.out, formats)
    _plot_per_target_heatmap(scores, explainers, args.out, formats)
    _plot_prediction_performance(perf, datasets, args.out, formats)
    _plot_prediction_vs_fidelity(scores, perf, explainers, args.out, formats)

    _plot_rank_heatmap(scores, datasets, explainers, args.out, formats)
    _plot_pairwise_wins(scores, explainers, args.out, formats)

    decomp = _load_decomposition(args.input, datasets, explainers, args.dir_suffix)
    if decomp.empty:
        warnings.warn("No checkpoint data found; skipping necessity/sufficiency plot")
    else:
        _plot_necessity_sufficiency(decomp, explainers, args.out, formats)


if __name__ == "__main__":
    main()
