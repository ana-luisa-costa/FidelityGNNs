
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("data/processed")
DEFAULT_OUTPUT = Path("results/benchmark_4")
DEFAULT_DATASETS = (
    "BPIC12_A",
    "BPIC12_W",
    "BPIC12_WC",
    "BPIC13_I",
    "BPIC13_O",
    "BPIC17_O",
    "BPIC20_P",
    "BPIC20_R",
)
def checkpoint_subpath(dir_suffix: str = "") -> Path:
    """Path (relative to a dataset dir) of the raw checkpoint CSV to read.

    dir_suffix lets callers point at a differently-named fidelity_curves_full*
    run, e.g. checkpoint_subpath("_n20") -> fidelity_curves_full_n20/curve_checkpoint.csv.
    """
    return Path(f"fidelity_curves_full{dir_suffix}") / "curve_checkpoint.csv"


CHECKPOINT_SUBPATH = checkpoint_subpath()

# Canonical target order on the x-axis.
TARGETS = ("activity", "resource", "role")
TARGET_LABELS = {
    "activity": "Activity",
    "resource": "Resource",
    "role": "Role",
}
# Substring needles used to map raw task names onto canonical targets.
TARGET_NEEDLES = {
    "resource": "org_resource",
    "role": "org_role",
}

METHODS = ("gradient", "ig", "lime", "shap", "gnnexplainer", "pgmexplainer")
METHOD_LABELS = {
    "gradient": "Gradient",
    "ig": "IG",
    "lime": "LIME",
    "shap": "SHAP",
    "gnnexplainer": "GNNExplainer",
    "pgmexplainer": "PGMExplainer",
}
METHOD_COLORS = {
    "gradient": "#1f77b4",
    "ig": "#ff7f0e",
    "lime": "#2ca02c",
    "shap": "#d62728",
    "gnnexplainer": "#9467bd",
    "pgmexplainer": "#8c564b",
}
METHOD_MARKERS = {
    "gradient": "s",
    "ig": "o",
    "lime": "^",
    "shap": "P",
    "gnnexplainer": "*",
    "pgmexplainer": "X",
}

_AUC_FN = getattr(np, "trapezoid", np.trapz)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Faceted target-profile plot of explainer fidelity scores S_{E,q} "
            "across prediction targets, with a separated Overall S_E column."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--explainers",
        default=",".join(METHODS),
        help="Comma-separated explainer names.",
    )
    parser.add_argument("--formats", default="png", help="Comma-separated formats, e.g. png,pdf.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=95.0, help="Confidence level in percent.")
    parser.add_argument("--n-cols", type=int, default=4, help="Panels per row of small multiples.")
    parser.add_argument(
        "--dir-suffix", default="",
        help="Suffix appended to the fidelity_curves_full/ folder name, "
             "e.g. '_n20' to read fidelity_curves_full_n20/curve_checkpoint.csv.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help=(
            "Shared y-axis upper limit. S theoretically ranges from 0 to 1; a "
            "zoomed range such as 0.35 is acceptable when observed scores stay "
            "below it. Default: 0.35, expanded automatically if scores exceed it."
        ),
    )
    return parser.parse_args()


def _canonical_target(task: str) -> str | None:
    task_l = str(task).lower()
    if task_l == "activity":
        return "activity"
    for target, needle in TARGET_NEEDLES.items():
        if needle in task_l:
            return target
    return None


def _score_from_curves(
    mean_plus: np.ndarray, mean_minus: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Compute S = sqrt(a * b) from mean fidelity curves.

    Accepts arrays whose last axis is the top-k axis; leading axes (e.g. the
    bootstrap axis) are preserved.
    """
    span = float(x.max() - x.min())
    if span <= 0:
        return np.full(mean_plus.shape[:-1], np.nan)
    auc_plus = _AUC_FN(mean_plus, x, axis=-1) / span
    auc_minus = _AUC_FN(mean_minus, x, axis=-1) / span
    a = np.clip(auc_plus, 0.0, 1.0)
    b = 1.0 - np.minimum(1.0, np.abs(auc_minus))
    return np.sqrt(a * b)


def _pair_matrices(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a (explainer, task) group to per-pair curve matrices.

    Returns (plus, minus, x) with plus/minus of shape (n_pairs, n_topk).
    """
    plus = group.pivot_table(index="pair_idx", columns="top_k", values="fid_prob_plus")
    minus = group.pivot_table(index="pair_idx", columns="top_k", values="fid_prob_minus")
    plus, minus = plus.align(minus, join="inner")
    plus = plus.dropna()
    minus = minus.loc[plus.index].dropna()
    plus = plus.loc[minus.index]
    x = plus.columns.to_numpy(dtype=float)
    order = np.argsort(x)
    return (
        plus.to_numpy(dtype=float)[:, order],
        minus.to_numpy(dtype=float)[:, order],
        x[order],
    )


def _compute_dataset_scores(
    checkpoint_path: Path,
    explainers: Sequence[str],
    n_bootstrap: int,
    ci: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Point estimates and bootstrap CIs for S_{E,q} and Overall S_E."""
    df = pd.read_csv(checkpoint_path)
    df["target"] = df["task"].map(_canonical_target)
    unknown = df.loc[df["target"].isna(), "task"].unique()
    if len(unknown):
        warnings.warn(f"{checkpoint_path}: skipping unmapped tasks {list(unknown)}")
    df = df.dropna(subset=["target"])
    df = df[df["explainer"].isin(explainers)]

    alpha = (100.0 - ci) / 2.0
    rows: List[Dict[str, object]] = []
    for explainer, exp_group in df.groupby("explainer"):
        per_target_point: Dict[str, float] = {}
        per_target_boot: Dict[str, np.ndarray] = {}
        for target, group in exp_group.groupby("target"):
            plus, minus, x = _pair_matrices(group)
            n_pairs = plus.shape[0]
            if n_pairs == 0 or len(x) < 2:
                continue
            # Point estimate: mean curve over all pairs -> AUC -> a, b -> S.
            point = float(_score_from_curves(plus.mean(axis=0), minus.mean(axis=0), x))
            # Bootstrap prediction pairs and recompute the complete sequence.
            idx = rng.integers(0, n_pairs, size=(n_bootstrap, n_pairs))
            boot = _score_from_curves(plus[idx].mean(axis=1), minus[idx].mean(axis=1), x)
            per_target_point[target] = point
            per_target_boot[target] = boot
            rows.append(
                {
                    "explainer": explainer,
                    "target": target,
                    "score": point,
                    "ci_low": float(np.nanpercentile(boot, alpha)),
                    "ci_high": float(np.nanpercentile(boot, 100.0 - alpha)),
                    "n_pairs": n_pairs,
                }
            )
        if not per_target_point:
            continue
        # Overall S_E = sum_q w_q S_{E,q} with uniform weights over targets.
        targets_present = sorted(per_target_point)
        w = 1.0 / len(targets_present)
        overall_point = float(sum(w * per_target_point[t] for t in targets_present))
        overall_boot = sum(w * per_target_boot[t] for t in targets_present)
        rows.append(
            {
                "explainer": explainer,
                "target": "overall",
                "score": overall_point,
                "ci_low": float(np.nanpercentile(overall_boot, alpha)),
                "ci_high": float(np.nanpercentile(overall_boot, 100.0 - alpha)),
                "n_pairs": int(exp_group["pair_idx"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _plot(
    scores: Dict[str, pd.DataFrame],
    datasets: Sequence[str],
    explainers: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
    n_cols: int,
    y_max: float | None,
    ci: float,
) -> None:
    if y_max is None:
        observed = [
            float(np.nanmax(t["ci_high"].to_numpy()))
            for t in scores.values()
            if t is not None and not t.empty
        ]
        y_max = 0.35
        if observed and max(observed) > y_max:
            y_max = min(1.0, float(np.ceil((max(observed) + 0.02) * 20.0) / 20.0))

    n = len(datasets)
    n_cols = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 2.9 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x_targets = np.arange(len(TARGETS), dtype=float)
    x_overall = float(len(TARGETS)) + 0.6
    divider_x = float(len(TARGETS)) - 0.5 + 0.35

    # Fixed per-explainer x-offsets within the Overall column so that
    # diamonds never overlap even when scores are very similar.
    n_exp = len(explainers)
    _offsets = np.linspace(-0.22, 0.22, n_exp) if n_exp > 1 else np.array([0.0])
    exp_x_offset: dict[str, float] = {
        e: float(_offsets[i]) for i, e in enumerate(explainers)
    }

    for i, dataset in enumerate(datasets):
        ax = axes[i // n_cols][i % n_cols]
        data = scores.get(dataset)
        ax.set_title(dataset, fontsize=10)
        # Shaded Overall column behind everything.
        ax.axvspan(divider_x, x_overall + 0.55, color="0.92", zorder=0)
        ax.axvline(divider_x, color="0.4", linewidth=0.8, zorder=1)
        if data is None or data.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            continue
        for explainer in explainers:
            sub = data[data["explainer"] == explainer]
            if sub.empty:
                continue
            color = METHOD_COLORS.get(explainer, "0.3")
            marker = METHOD_MARKERS.get(explainer, "o")
            by_target = sub.set_index("target")
            ys = np.array(
                [by_target["score"].get(t, np.nan) for t in TARGETS], dtype=float
            )
            lo = np.array(
                [by_target["ci_low"].get(t, np.nan) for t in TARGETS], dtype=float
            )
            hi = np.array(
                [by_target["ci_high"].get(t, np.nan) for t in TARGETS], dtype=float
            )
            yerr = np.vstack([ys - lo, hi - ys])
            ax.errorbar(
                x_targets,
                ys,
                yerr=np.nan_to_num(yerr, nan=0.0),
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.4,
                capsize=2,
                elinewidth=0.8,
                zorder=3,
            )
            # Overall S_E: diamond at explainer-specific x-offset so markers
            # never overlap; annotate with rank after all explainers are drawn.
            if "overall" in by_target.index:
                s = float(by_target.loc["overall", "score"])
                s_lo = float(by_target.loc["overall", "ci_low"])
                s_hi = float(by_target.loc["overall", "ci_high"])
                xd = x_overall + exp_x_offset.get(explainer, 0.0)
                ax.errorbar(
                    [xd],
                    [s],
                    yerr=[[max(s - s_lo, 0.0)], [max(s_hi - s, 0.0)]],
                    fmt="D",
                    color=color,
                    markersize=5,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    capsize=2,
                    elinewidth=0.8,
                    zorder=4,
                )

        # Rank annotations: number each diamond 1=best inside the shaded column.
        if data is not None and not data.empty:
            overall_rows = data[data["target"] == "overall"].copy()
            overall_rows = overall_rows[overall_rows["explainer"].isin(explainers)]
            overall_rows = overall_rows.sort_values("score", ascending=False).reset_index(drop=True)
            for rank, row in overall_rows.iterrows():
                xd = x_overall + exp_x_offset.get(row["explainer"], 0.0)
                ax.text(
                    xd, float(row["score"]) + y_max * 0.025,
                    str(rank + 1),
                    ha="center", va="bottom",
                    fontsize=6, color="0.2", zorder=5,
                )

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    tick_positions = list(x_targets) + [x_overall]
    tick_labels = [TARGET_LABELS[t] for t in TARGETS] + ["Overall $S_E$"]
    for row in axes:
        for ax in row:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
            ax.set_xlim(-0.5, x_overall + 0.5)
            ax.set_ylim(0.0, y_max)
            ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    for r in range(n_rows):
        axes[r][0].set_ylabel(r"$S_{E,q} = \sqrt{a_{E,q}\,b_{E,q}}$", fontsize=9)

    handles = [
        plt.Line2D(
            [],
            [],
            color=METHOD_COLORS[m],
            marker=METHOD_MARKERS[m],
            linewidth=1.4,
            markersize=6,
            label=METHOD_LABELS[m],
        )
        for m in explainers
        if m in METHOD_COLORS
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            color="0.3",
            marker="D",
            linestyle="none",
            markersize=5,
            markeredgecolor="black",
            label=r"Overall $S_E$",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 7),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Explainer fidelity score $S_{E,q}$ per prediction target "
        f"(error bars: {ci:g}% bootstrap CI; $S \\in [0, 1]$, axis zoomed to "
        f"$[0, {y_max:g}]$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"target_profiles.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Plot    -> {path}")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    explainers = [e.strip() for e in str(args.explainers).split(",") if e.strip()]
    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]
    rng = np.random.default_rng(args.seed)

    scores: Dict[str, pd.DataFrame] = {}
    all_rows = []
    for dataset in datasets:
        checkpoint_path = args.input / dataset / checkpoint_subpath(args.dir_suffix)
        if not checkpoint_path.is_file():
            warnings.warn(f"Missing checkpoint for {dataset}: {checkpoint_path}; skipping")
            continue
        table = _compute_dataset_scores(
            checkpoint_path, explainers, args.n_bootstrap, args.ci, rng
        )
        scores[dataset] = table
        table = table.copy()
        table.insert(0, "dataset", dataset)
        all_rows.append(table)
        print(f"Scores  -> {dataset}: {len(table)} rows")

    if not all_rows:
        raise SystemExit("No datasets with usable checkpoint files were found.")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "target_profile_scores.csv"
    pd.concat(all_rows, ignore_index=True).to_csv(csv_path, index=False)
    print(f"CSV     -> {csv_path}")

    _plot(scores, datasets, explainers, args.out, formats, args.n_cols, args.y_max, args.ci)


if __name__ == "__main__":
    main()
