"""Standalone node-level PGMExplainer for R-GCN event-pair predictions."""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import asdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from src.explainers._shared import (
    ExplainerContext,
    _entity_short_label,
    _k_hop_subgraph,
    _pair_side_caption,
    add_common_explainer_args,
    load_explainer_context,
    setup_matplotlib_style,
    write_summary_json,
)
from src.explainers.pgmexplainer_core import (
    PGMExplanation,
    PGMExplanationError,
    PGMExplainerConfig,
    explain_pgmexplainer_nodes,
    pgmexplainer_seed,
    validate_classification_task,
)
from src.fidelity.fidelity_utils import task_target_value_nodes
from src.train import _sanitize, get_event_attrs
from src.utils.io_helpers import load_vocabs

setup_matplotlib_style()


NODE_COLUMNS = [
    "rank", "local_id", "entity_id", "uri", "label", "p_value",
    "importance_1_minus_p", "chi_square", "degrees_of_freedom",
    "is_significant", "is_protected", "protection_semantics",
    "perturbation_count", "pred_class_id", "pred_class_label",
    "original_probability",
]
SUMMARY_COLUMNS = [
    "pair_idx", "task", "method", "src_id", "dst_id", "dst_activity",
    "pred_class_id", "pred_class_label", "original_logit", "original_probability",
    "num_nodes_sub", "num_edges_sub", "num_selectable_nodes", "num_protected_nodes",
    "num_significant_nodes", "prediction_changed_count", "prediction_unchanged_count",
    "prediction_changed_rate", "num_samples", "perturb_probability", "response_mode",
    "response_top_fraction",
    "significance_threshold", "baseline", "seed", "elapsed_s", "node_csv", "plot",
]
SKIPPED_COLUMNS = [
    "pair_idx", "task", "src_id", "dst_id", "seed", "failure_code",
    "failure_message", "prediction_changed_count", "prediction_unchanged_count",
    "num_samples", "perturb_probability", "response_mode", "response_top_fraction",
    "significance_threshold",
    "elapsed_s",
]


def _otherc_id2label_by_key(vocabs: Dict[str, object]) -> Dict[str, Dict[int, str]]:
    result: Dict[str, Dict[int, str]] = {}
    for raw_key, label_to_id in vocabs.get("c_vocabs", {}).items():
        if isinstance(label_to_id, dict):
            result[_sanitize(str(raw_key))] = {
                int(class_id): str(label) for label, class_id in label_to_id.items()
            }
    return result


def _predicted_class_label(
    task: str,
    class_id: int,
    id2act: Dict[int, str],
    otherc_id2label: Dict[str, Dict[int, str]],
) -> str:
    if task == "activity":
        return id2act.get(class_id, str(class_id))
    key = task[len("otherC_") :]
    return otherc_id2label.get(key, {}).get(class_id, str(class_id))


def _node_rows(
    explanation: PGMExplanation,
    nodes_global: Tensor,
    graph,
    id2ent: Dict[int, str],
) -> List[dict]:
    protected = set(explanation.protected_nodes)
    ranked = sorted(
        explanation.selectable_nodes,
        key=lambda node: (-float(explanation.node_mask[node].item()), node),
    )
    ranks = {node: rank + 1 for rank, node in enumerate(ranked)}
    rows: List[dict] = []
    for local_id in range(nodes_global.numel()):
        global_id = int(nodes_global[local_id].item())
        uri = id2ent.get(global_id, "?")
        is_protected = local_id in protected
        rows.append(
            {
                "rank": ranks.get(local_id, ""),
                "local_id": local_id,
                "entity_id": global_id,
                "uri": uri,
                "label": _entity_short_label(graph, uri),
                "p_value": (
                    "" if is_protected else float(explanation.p_values[local_id].item())
                ),
                "importance_1_minus_p": float(explanation.node_mask[local_id].item()),
                "chi_square": (
                    ""
                    if is_protected
                    else float(explanation.chi_square_statistics[local_id].item())
                ),
                "degrees_of_freedom": (
                    ""
                    if is_protected
                    else int(explanation.degrees_of_freedom[local_id].item())
                ),
                "is_significant": (
                    False
                    if is_protected
                    else bool(explanation.significant_mask[local_id].item())
                ),
                "is_protected": is_protected,
                "protection_semantics": (
                    "mandatory_context_fixed_to_1" if is_protected else ""
                ),
                "perturbation_count": int(
                    explanation.perturbation_counts[local_id].item()
                ),
                "pred_class_id": explanation.pred_class_id,
                "pred_class_label": explanation.pred_class_label,
                "original_probability": explanation.original_probability,
            }
        )
    return rows


def _plot_top_nodes(
    rows: List[dict],
    top_nodes: int,
    title: str,
    out_path: str,
    pair_side_text: str,
) -> None:
    selected = sorted(
        (row for row in rows if not row["is_protected"]),
        key=lambda row: int(row["rank"]),
    )[:top_nodes]
    scores = np.asarray([row["importance_1_minus_p"] for row in selected])
    labels = [str(row["label"]) for row in selected]
    height = max(4.0, len(selected) * 0.28)
    fig = plt.figure(figsize=(13.0, height))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.3], wspace=0.12)
    axis = fig.add_subplot(grid[0, 0])
    side = fig.add_subplot(grid[0, 1])
    side.axis("off")
    side.text(0, 1, pair_side_text, va="top", fontsize=7.5, linespacing=1.22)
    positions = np.arange(len(scores))
    colors = ["#2878b5" if row["is_significant"] else "#9aa0a6" for row in selected]
    axis.barh(positions, scores, color=colors, alpha=0.88)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Node importance (1 - p-value)")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.9, bottom=0.14)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Node-level PGMExplainer for R-GCN event-pair predictions."
    )
    add_common_explainer_args(parser)
    parser.set_defaults(tasks="activity")
    parser.add_argument(
        "--pgm-subgraph-hop", "--num-hops", dest="pgm_subgraph_hop", type=int, default=3
    )
    parser.add_argument("--pgm-num-samples", type=int, default=1000)
    parser.add_argument("--pgm-perturb-probability", type=float, default=0.5)
    parser.add_argument("--pgm-response-top-fraction", type=float, default=0.125)
    parser.add_argument("--pgm-significance-threshold", type=float, default=0.05)
    parser.add_argument("--top-nodes", type=int, default=25)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    requested_tasks = [part.strip() for part in args.tasks.split(",") if part.strip()]
    normalized_tasks = [validate_classification_task(task) for task in requested_tasks]
    args.tasks = ",".join(normalized_tasks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context: ExplainerContext = load_explainer_context(args, device)
    vocabs = load_vocabs(args.vocabs) if args.vocabs else {}
    otherc_id2label = _otherc_id2label_by_key(vocabs)
    global_mean = context.x.mean(dim=0).detach()

    pair_dir = os.path.join(args.out, "per_pair")
    plot_dir = os.path.join(args.out, "pgmexplainer")
    os.makedirs(pair_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    summary_rows: List[dict] = []
    skipped_rows: List[dict] = []

    for pair_idx in tqdm(context.pair_indices, desc="PGMExplainer pairs"):
        src_global, dst_global = context.val_pairs[pair_idx]
        nodes, edge_index, edge_type, global_to_local = _k_hop_subgraph(
            seeds=[src_global, dst_global],
            edge_index=context.ei_val,
            edge_type=context.et_val,
            num_nodes=context.x.size(0),
            k=args.pgm_subgraph_hop,
        )
        x_sub = context.x[nodes.to(device)].detach()
        edge_index, edge_type = edge_index.to(device), edge_type.to(device)
        src_local = global_to_local[src_global]
        dst_local = global_to_local[dst_global]

        for task in context.tasks:
            protected = {src_local, dst_local}
            protected.update(
                task_target_value_nodes(
                    context.g,
                    context.id2ent,
                    nodes,
                    global_to_local,
                    dst_global,
                    task,
                )
            )
            config = PGMExplainerConfig(
                num_samples=args.pgm_num_samples,
                perturb_probability=args.pgm_perturb_probability,
                response_top_fraction=args.pgm_response_top_fraction,
                significance_threshold=args.pgm_significance_threshold,
                seed=pgmexplainer_seed(args.seed, pair_idx, task),
            )
            started = time.perf_counter()
            try:
                explanation = explain_pgmexplainer_nodes(
                    context.encoder,
                    context.head,
                    x_sub,
                    edge_index,
                    edge_type,
                    src_local,
                    task,
                    sorted(protected),
                    global_mean,
                    config,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                code = exc.code if isinstance(exc, PGMExplanationError) else "failed"
                diagnostics = (
                    exc.diagnostics if isinstance(exc, PGMExplanationError) else {}
                )
                warnings.warn(
                    f"PGMExplainer pair={pair_idx} task={task} skipped ({code}): {exc}"
                )
                skipped_rows.append(
                    {
                        "pair_idx": pair_idx,
                        "task": task,
                        "src_id": src_global,
                        "dst_id": dst_global,
                        "seed": config.seed,
                        "failure_code": code,
                        "failure_message": str(exc),
                        "prediction_changed_count": diagnostics.get(
                            "prediction_changed_count", ""
                        ),
                        "prediction_unchanged_count": diagnostics.get(
                            "prediction_unchanged_count", ""
                        ),
                        "num_samples": config.num_samples,
                        "perturb_probability": config.perturb_probability,
                        "response_mode": "top_probability_drop_fraction",
                        "response_top_fraction": config.response_top_fraction,
                        "significance_threshold": config.significance_threshold,
                        "elapsed_s": round(elapsed, 3),
                    }
                )
                continue

            elapsed = time.perf_counter() - started
            explanation.pred_class_label = _predicted_class_label(
                task,
                explanation.pred_class_id,
                context.id2act,
                otherc_id2label,
            )
            stem = f"pair{pair_idx:05d}_{task}_pgmexplainer"
            node_csv = f"{stem}_nodes.csv"
            plot_file = f"{stem}_nodes.png"
            rows = _node_rows(explanation, nodes, context.g, context.id2ent)
            pd.DataFrame(rows, columns=NODE_COLUMNS).to_csv(
                os.path.join(pair_dir, node_csv), index=False
            )
            _plot_top_nodes(
                rows,
                max(args.top_nodes, 0),
                f"PGMExplainer node importance (pair {pair_idx}, {task})",
                os.path.join(plot_dir, plot_file),
                _pair_side_caption(
                    context.g, context.id2ent, src_global, dst_global, pair_idx
                ),
            )
            dst_activity, _, _, _ = get_event_attrs(
                context.g, context.id2ent.get(dst_global, "?")
            )
            summary_rows.append(
                {
                    "pair_idx": pair_idx,
                    "task": task,
                    "method": "pgmexplainer",
                    "src_id": src_global,
                    "dst_id": dst_global,
                    "dst_activity": dst_activity or "",
                    "pred_class_id": explanation.pred_class_id,
                    "pred_class_label": explanation.pred_class_label,
                    "original_logit": explanation.original_logit,
                    "original_probability": explanation.original_probability,
                    "num_nodes_sub": x_sub.size(0),
                    "num_edges_sub": edge_type.numel(),
                    "num_selectable_nodes": len(explanation.selectable_nodes),
                    "num_protected_nodes": len(explanation.protected_nodes),
                    "num_significant_nodes": int(
                        explanation.significant_mask.sum().item()
                    ),
                    "prediction_changed_count": explanation.prediction_changed_count,
                    "prediction_unchanged_count": explanation.prediction_unchanged_count,
                    "prediction_changed_rate": (
                        explanation.prediction_changed_count / config.num_samples
                    ),
                    "num_samples": config.num_samples,
                    "perturb_probability": config.perturb_probability,
                    "response_mode": "top_probability_drop_fraction",
                    "response_top_fraction": config.response_top_fraction,
                    "significance_threshold": config.significance_threshold,
                    "baseline": "full_graph_mean",
                    "seed": explanation.seed,
                    "elapsed_s": round(elapsed, 3),
                    "node_csv": node_csv,
                    "plot": plot_file,
                }
            )

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(
        os.path.join(args.out, "summary.csv"), index=False
    )
    pd.DataFrame(skipped_rows, columns=SKIPPED_COLUMNS).to_csv(
        os.path.join(args.out, "skipped_explanations.csv"), index=False
    )
    write_summary_json(
        os.path.join(args.out, "meta.json"),
        args,
        context.n_val,
        context.pair_indices,
        context.tasks,
        extra_fields={
            "method": "pgmexplainer",
            "target_semantics": "original_model_predicted_class_at_src_local",
            "primary_mask": "node",
            "importance_semantics": "1_minus_chi_square_p_value",
            "baseline": "full_graph_mean",
            "protected_mask_semantics": "mandatory_context_fixed_to_1",
            "config": asdict(
                PGMExplainerConfig(
                    num_samples=args.pgm_num_samples,
                    perturb_probability=args.pgm_perturb_probability,
                    response_top_fraction=args.pgm_response_top_fraction,
                    significance_threshold=args.pgm_significance_threshold,
                    seed=args.seed,
                )
            ),
            "successful_explanations": len(summary_rows),
            "skipped_explanations": len(skipped_rows),
            "skipped_csv": "skipped_explanations.csv",
            "node_csv_columns": NODE_COLUMNS,
            "summary_columns": SUMMARY_COLUMNS,
            "skipped_columns": SKIPPED_COLUMNS,
        },
        rows=summary_rows,
    )
    print(f"Done. PGMExplainer results -> {args.out}")


if __name__ == "__main__":
    main()
