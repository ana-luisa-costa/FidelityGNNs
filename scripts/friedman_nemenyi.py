import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Scores CSV from benchmark_4_targets.py (default: sibling benchmark_4/target_profile_scores.csv).",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    scores_path = args.scores or args.benchmark_dir.parent / "benchmark_4" / "target_profile_scores.csv"
    df = pd.read_csv(scores_path)
    df = df[df["target"] == "overall"]
    scores = df.pivot_table(index="dataset", columns="explainer", values="score")
    scores = scores.dropna()

    n, k = scores.shape
    # Higher score is better, so rank 1 = best explainer on each dataset.
    ranks = scores.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean().sort_values()

    print(f"Datasets (N) = {n}, explainers (k) = {k}\n")
    print("Mean scores per explainer (sqrt(a*b), mean over datasets):")
    print(scores.mean().sort_values(ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nMean ranks (lower is better):")
    print(mean_ranks.to_string(float_format=lambda v: f"{v:.3f}"))

    chi2, p_friedman = stats.friedmanchisquare(*[scores[c].to_numpy() for c in scores.columns])
    print(f"\nFriedman chi2 = {chi2:.4f}, df = {k - 1}, p = {p_friedman:.4g}")

    # Nemenyi: CD = q_alpha * sqrt(k(k+1) / (6N)), q_alpha from studentized range / sqrt(2).
    q_alpha = stats.studentized_range.ppf(1 - args.alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    print(f"Nemenyi critical difference (alpha={args.alpha}) = {cd:.4f}")

    rows = []
    for a, b in combinations(mean_ranks.index, 2):
        diff = abs(mean_ranks[a] - mean_ranks[b])
        q_stat = diff / np.sqrt(k * (k + 1) / (6 * n))
        p_nem = stats.studentized_range.sf(q_stat * np.sqrt(2), k, np.inf)
        rows.append(
            {
                "explainer_a": a,
                "explainer_b": b,
                "rank_diff": diff,
                "p_nemenyi": p_nem,
                "significant": bool(diff > cd),
            }
        )
    pairwise = pd.DataFrame(rows).sort_values("p_nemenyi").reset_index(drop=True)
    print("\nPairwise Nemenyi comparisons:")
    print(pairwise.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out_path = args.benchmark_dir / "friedman_nemenyi_pairwise.csv"
    pairwise.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
