#!/usr/bin/env python3
"""Merge fidelity outputs and summarize pair-seed row variability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_EXPLAINERS = "gradient,ig,lime,shap,gnnexplainer,pgmexplainer"


def _parse_names(raw: str) -> List[str]:
    return [name.strip() for name in raw.split(",") if name.strip()]


def _parse_seeds(raw: Optional[str]) -> List[int]:
    if raw is None or not str(raw).strip():
        return []
    return [int(seed.strip()) for seed in str(raw).split(",") if seed.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge parallel fidelity curve outputs. Multi-seed summaries report "
            "variability across evaluated pair-seed rows."
        )
    )
    parser.add_argument("--base", type=Path, default=Path("plots_parallel"))
    parser.add_argument("--out", type=Path, default=Path("plots_parallel/merged"))
    parser.add_argument("--explainers", default=DEFAULT_EXPLAINERS)
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Optional comma-separated seeds. If provided, reads "
            "base/seed_<seed>/<explainer>/ and summarizes pair-seed rows."
        ),
    )
    return parser


def _checkpoint_path(base: Path, explainer: str, seed: Optional[int]) -> Path:
    if seed is None:
        return base / explainer / "curve_checkpoint.csv"
    return base / f"seed_{seed}" / explainer / "curve_checkpoint.csv"


def main() -> None:
    args = _build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    seed_values: List[Optional[int]] = seeds if seeds else [None]

    frames = []
    for seed in seed_values:
        for explainer in _parse_names(args.explainers):
            path = _checkpoint_path(args.base, explainer, seed)
            if not path.exists():
                raise FileNotFoundError(f"Missing expected fidelity output: {path}")
            frame = pd.read_csv(path)
            if seed is not None:
                frame["seed"] = int(seed)
            frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out / "curve_checkpoint.csv", index=False)
    df["top_k"] = df["top_k"].astype(int)
    metrics = [
        metric
        for metric in ("prob", "acc")
        if f"fid_{metric}_plus" in df.columns
        and df[f"fid_{metric}_plus"].notna().any()
    ]
    if not metrics:
        raise ValueError("No finite probability or accuracy fidelity values found")

    aggregations: Dict[str, Any] = {}
    for metric in metrics:
        aggregations[f"mean_fid_{metric}_plus"] = (f"fid_{metric}_plus", "mean")
        aggregations[f"mean_fid_{metric}_minus"] = (f"fid_{metric}_minus", "mean")
        if seeds:
            aggregations[f"std_fid_{metric}_plus"] = (f"fid_{metric}_plus", "std")
            aggregations[f"std_fid_{metric}_minus"] = (f"fid_{metric}_minus", "std")

    if seeds:
        aggregations.update({
            "n_pairs": ("pair_idx", "nunique"),
            "n_seeds": ("seed", "nunique"),
            "n_rows": ("pair_idx", "size"),
        })
    else:
        aggregations["n_samples"] = ("pair_idx", "nunique")
    summary = (
        df.groupby(["explainer", "task", "top_k"], as_index=False)
        .agg(**aggregations)
        .sort_values(["explainer", "task", "top_k"])
    )
    for metric in metrics:
        summary[f"mean_one_minus_fid_{metric}_minus"] = (
            1.0 - summary[f"mean_fid_{metric}_minus"]
        )
        if seeds:
            summary[f"sem_fid_{metric}_plus"] = (
                summary[f"std_fid_{metric}_plus"] / np.sqrt(summary["n_rows"])
            )
            summary[f"sem_fid_{metric}_minus"] = (
                summary[f"std_fid_{metric}_minus"] / np.sqrt(summary["n_rows"])
            )
            summary[f"sem_one_minus_fid_{metric}_minus"] = summary[
                f"sem_fid_{metric}_minus"
            ]
    summary.to_csv(args.out / "fidelity_curves_summary.csv", index=False)

    auc_fn = getattr(np, "trapezoid", np.trapz)
    rows = []
    for (explainer, task), group in summary.groupby(["explainer", "task"]):
        group = group.sort_values("top_k")
        x = group["top_k"].to_numpy(dtype=float)
        span = float(x.max() - x.min()) if len(x) else 0.0
        row: Dict[str, Any] = {"explainer": explainer, "task": task}
        for metric in metrics:
            y_plus = group[f"mean_fid_{metric}_plus"].to_numpy(dtype=float)
            y_minus = group[f"mean_fid_{metric}_minus"].to_numpy(dtype=float)
            auc_plus = float(auc_fn(y_plus, x))
            auc_minus = float(auc_fn(y_minus, x))
            auc_one_minus = float(auc_fn(1.0 - y_minus, x))
            norm_plus = auc_plus / span if span > 0 else float("nan")
            norm_minus = auc_minus / span if span > 0 else float("nan")
            norm_one_minus = auc_one_minus / span if span > 0 else float("nan")
            score = float("nan")
            if np.isfinite(norm_plus) and np.isfinite(norm_minus):
                a = min(1.0, max(0.0, norm_plus))
                # Sufficiency rewards an AUC- close to zero in either direction.
                b = 1.0 - min(1.0, abs(norm_minus))
                score = float(np.sqrt(a * b))
            row.update({
                f"auc_fid_{metric}_plus_raw": auc_plus,
                f"auc_fid_{metric}_plus_normalized": norm_plus,
                f"auc_fid_{metric}_minus_raw": auc_minus,
                f"auc_fid_{metric}_minus_normalized": norm_minus,
                f"auc_one_minus_fid_{metric}_minus_raw": auc_one_minus,
                f"auc_one_minus_fid_{metric}_minus_normalized": norm_one_minus,
                f"score_{metric}": score,
            })
        rows.append(row)

    pd.DataFrame(rows).sort_values(["task", "explainer"]).to_csv(
        args.out / "fidelity_curves_auc.csv", index=False
    )
    print(f"Merged fidelity outputs -> {args.out}")


if __name__ == "__main__":
    main()
