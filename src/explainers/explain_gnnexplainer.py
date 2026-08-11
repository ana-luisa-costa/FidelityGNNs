"""Standalone stock-PyG GNNExplainer for the R-GCN event-pair model."""

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
from src.explainers.gnnexplainer_core import (
    GNNExplainerConfig,
    GNNExplainerExplanation,
    UnusableGNNExplanation,
    explain_gnnexplainer_masks,
    gnnexplainer_seed,
    validate_classification_task,
)
from src.fidelity.fidelity_utils import task_target_value_nodes
from src.train import _sanitize, get_event_attrs
from src.utils.io_helpers import load_vocabs

setup_matplotlib_style()


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
    explanation: GNNExplainerExplanation,
    nodes_global: Tensor,
    graph,
    id2ent: Dict[int, str],
) -> List[dict]:
    protected = set(explanation.protected_nodes)
    selectable = [node for node in range(nodes_global.numel()) if node not in protected]
    ranked = sorted(
        selectable,
        key=lambda node: (-float(explanation.node_mask[node].item()), node),
    )
    ranks = {node: rank + 1 for rank, node in enumerate(ranked)}
    rows = []
    for local_id in ranked + sorted(protected):
        global_id = int(nodes_global[local_id].item())
        uri = id2ent.get(global_id, "?")
        rows.append(
            {
                "rank": ranks.get(local_id, ""),
                "local_id": local_id,
                "entity_id": global_id,
                "uri": uri,
                "label": _entity_short_label(graph, uri),
                "node_mask": float(explanation.node_mask[local_id].item()),
                "is_protected": local_id in protected,
                "protection_semantics": (
                    "mandatory_context_fixed_to_1" if local_id in protected else ""
                ),
                "pred_class_id": explanation.pred_class_id,
                "pred_class_label": explanation.pred_class_label,
                "original_probability": explanation.original_probability,
            }
        )
    return rows


def _edge_rows(
    explanation: GNNExplainerExplanation,
    nodes_global: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    num_rel: int,
    id2rel: Dict[int, str],
    graph,
    id2ent: Dict[int, str],
) -> List[dict]:
    rows = []
    for rank, edge_id in enumerate(
        torch.argsort(explanation.edge_mask, descending=True).tolist(), start=1
    ):
        source_local = int(edge_index[0, edge_id].item())
        target_local = int(edge_index[1, edge_id].item())
        source_global = int(nodes_global[source_local].item())
        target_global = int(nodes_global[target_local].item())
        relation_id = int(edge_type[edge_id].item())
        base_relation_id = relation_id % num_rel
        rows.append(
            {
                "rank": rank,
                "edge_id": edge_id,
                "edge_mask": float(explanation.edge_mask[edge_id].item()),
                "src_local": source_local,
                "dst_local": target_local,
                "src_global": source_global,
                "dst_global": target_global,
                "src_label": _entity_short_label(graph, id2ent.get(source_global, "?")),
                "dst_label": _entity_short_label(graph, id2ent.get(target_global, "?")),
                "relation_id": relation_id,
                "base_relation_id": base_relation_id,
                "relation_uri": id2rel.get(base_relation_id, str(base_relation_id)),
                "direction": "inverse" if relation_id >= num_rel else "forward",
            }
        )
    return rows


def _plot_importance(
    scores: np.ndarray,
    labels: List[str],
    title: str,
    xlabel: str,
    out_path: str,
    pair_side_text: str | None = None,
) -> None:
    height = max(4.0, len(scores) * 0.28)
    if pair_side_text:
        fig = plt.figure(figsize=(13.0, height))
        grid = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.3], wspace=0.12)
        axis = fig.add_subplot(grid[0, 0])
        side = fig.add_subplot(grid[0, 1])
        side.axis("off")
        side.text(0, 1, pair_side_text, va="top", fontsize=7.5, linespacing=1.22)
    else:
        fig, axis = plt.subplots(figsize=(10.5, height))
    positions = np.arange(len(scores))
    axis.barh(positions, scores, color="#2878b5", alpha=0.88)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    if pair_side_text:
        fig.subplots_adjust(left=0.25, right=0.98, top=0.9, bottom=0.14)
    else:
        fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stock PyG GNNExplainer for R-GCN node-class predictions."
    )
    add_common_explainer_args(parser)
    parser.set_defaults(tasks="activity")
    parser.add_argument(
        "--gnn-subgraph-hop",
        "--num-hops",
        dest="gnn_subgraph_hop",
        type=int,
        default=3,
    )
    parser.add_argument("--top-nodes", type=int, default=25)
    parser.add_argument("--top-edges", type=int, default=20)
    parser.add_argument("--gnn-epochs", type=int, default=100)
    parser.add_argument("--gnn-lr", type=float, default=0.01)
    parser.add_argument("--edge-size-reg", type=float, default=0.005)
    parser.add_argument("--edge-ent-reg", type=float, default=1.0)
    parser.add_argument("--node-size-reg", type=float, default=1.0)
    parser.add_argument("--node-ent-reg", type=float, default=0.1)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    requested_tasks = [part.strip() for part in args.tasks.split(",") if part.strip()]
    normalized_tasks = [validate_classification_task(task) for task in requested_tasks]
    args.tasks = ",".join(normalized_tasks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context: ExplainerContext = load_explainer_context(args, device)
    id2rel = {relation_id: uri for uri, relation_id in context.rel2id.items()}
    vocabs = load_vocabs(args.vocabs) if args.vocabs else {}
    otherc_id2label = _otherc_id2label_by_key(vocabs)

    pair_dir = os.path.join(args.out, "per_pair")
    plot_dir = os.path.join(args.out, "gnnexplainer")
    os.makedirs(pair_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    summary_rows: List[dict] = []

    for pair_idx in tqdm(context.pair_indices, desc="GNNExplainer pairs"):
        src_global, dst_global = context.val_pairs[pair_idx]
        if args.gnn_subgraph_hop > 0:
            nodes, edge_index, edge_type, global_to_local = _k_hop_subgraph(
                seeds=[src_global, dst_global],
                edge_index=context.ei_val,
                edge_type=context.et_val,
                num_nodes=context.x.size(0),
                k=args.gnn_subgraph_hop,
            )
            x_sub = context.x[nodes.to(device)].detach()
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
        else:
            nodes = torch.arange(context.x.size(0))
            x_sub = context.x.detach()
            edge_index, edge_type = context.ei_val, context.et_val
            global_to_local = {node: node for node in range(context.x.size(0))}

        src_local = global_to_local[src_global]
        dst_local = global_to_local.get(dst_global, -1)
        for task in context.tasks:
            protected = {
                node for node in (src_local, dst_local) if 0 <= node < x_sub.size(0)
            }
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
            config = GNNExplainerConfig(
                epochs=args.gnn_epochs,
                lr=args.gnn_lr,
                edge_size=args.edge_size_reg,
                edge_ent=args.edge_ent_reg,
                node_size=args.node_size_reg,
                node_ent=args.node_ent_reg,
                seed=gnnexplainer_seed(args.seed, pair_idx, task),
            )
            started = time.perf_counter()
            try:
                explanation = explain_gnnexplainer_masks(
                    context.encoder,
                    context.head,
                    x_sub,
                    edge_index,
                    edge_type,
                    src_local,
                    task,
                    sorted(protected),
                    config,
                )
            except Exception as exc:
                status = "unusable" if isinstance(exc, UnusableGNNExplanation) else "failed"
                warnings.warn(
                    f"GNNExplainer pair={pair_idx} task={task} {status}: {exc}"
                )
                summary_rows.append(
                    {
                        "pair_idx": pair_idx,
                        "task": task,
                        "method": "gnnexplainer",
                        "status": status,
                        "failure_reason": str(exc),
                        "seed": config.seed,
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

            stem = f"pair{pair_idx:05d}_{task}_gnnexplainer"
            node_rows = _node_rows(explanation, nodes, context.g, context.id2ent)
            edge_rows = _edge_rows(
                explanation,
                nodes,
                edge_index,
                edge_type,
                context.num_rel,
                id2rel,
                context.g,
                context.id2ent,
            )
            node_csv, edge_csv = f"{stem}_nodes.csv", f"{stem}_edges.csv"
            pd.DataFrame(node_rows).to_csv(os.path.join(pair_dir, node_csv), index=False)
            pd.DataFrame(edge_rows).to_csv(os.path.join(pair_dir, edge_csv), index=False)

            side_text = _pair_side_caption(
                context.g, context.id2ent, src_global, dst_global, pair_idx
            )
            top_nodes = [row for row in node_rows if not row["is_protected"]][
                : args.top_nodes
            ]
            top_edges = edge_rows[: args.top_edges]
            _plot_importance(
                np.asarray([row["node_mask"] for row in top_nodes]),
                [str(row["label"]) for row in top_nodes],
                f"GNNExplainer node importance (pair {pair_idx}, {task})",
                "Native PyG node mask",
                os.path.join(plot_dir, f"{stem}_nodes.png"),
                side_text,
            )
            _plot_importance(
                np.asarray([row["edge_mask"] for row in top_edges]),
                [
                    f"{row['src_label']} -[{str(row['relation_uri']).rsplit('/', 1)[-1]}]-> {row['dst_label']}"
                    for row in top_edges
                ],
                f"GNNExplainer edge importance (pair {pair_idx}, {task})",
                "Native PyG edge mask",
                os.path.join(plot_dir, f"{stem}_edges.png"),
            )

            dst_activity, _, _, _ = get_event_attrs(
                context.g, context.id2ent.get(dst_global, "?")
            )
            summary_rows.append(
                {
                    "pair_idx": pair_idx,
                    "task": task,
                    "method": "gnnexplainer",
                    "method_detail": "stock PyG GNNExplainer adapted to basis R-GCN",
                    "status": "ok",
                    "failure_reason": "",
                    "src_id": src_global,
                    "dst_id": dst_global,
                    "dst_activity": dst_activity or "",
                    "pred_class_id": explanation.pred_class_id,
                    "pred_class_label": explanation.pred_class_label,
                    "original_logit": explanation.original_logit,
                    "original_probability": explanation.original_probability,
                    "masked_logit": explanation.masked_logit,
                    "masked_probability": explanation.masked_probability,
                    "masked_class_id": explanation.masked_class_id,
                    "prediction_agreement": explanation.prediction_agreement,
                    "num_nodes_sub": x_sub.size(0),
                    "num_edges_sub": edge_type.numel(),
                    "protected_local_ids": list(explanation.protected_nodes),
                    "seed": explanation.seed,
                    "elapsed_s": round(elapsed, 3),
                    "encoder_parity_max_abs_diff": explanation.encoder_parity_max_abs_diff,
                    "all_one_edge_parity_max_abs_diff": explanation.all_one_edge_parity_max_abs_diff,
                    "pyg_version": explanation.pyg_version,
                    "node_csv": node_csv,
                    "edge_csv": edge_csv,
                }
            )

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    write_summary_json(
        os.path.join(args.out, "meta.json"),
        args,
        context.n_val,
        context.pair_indices,
        context.tasks,
        extra_fields={
            "method": "gnnexplainer",
            "method_detail": "stock PyG GNNExplainer adapted to basis R-GCN",
            "target_semantics": "original_model_predicted_class",
            "primary_mask": "node",
            "supplementary_mask": "edge",
            "protected_mask_semantics": "mandatory_context_fixed_to_1",
            "config": asdict(
                GNNExplainerConfig(
                    epochs=args.gnn_epochs,
                    lr=args.gnn_lr,
                    edge_size=args.edge_size_reg,
                    edge_ent=args.edge_ent_reg,
                    node_size=args.node_size_reg,
                    node_ent=args.node_ent_reg,
                    seed=args.seed,
                )
            ),
        },
        rows=summary_rows,
    )
    print(f"Done. GNNExplainer results -> {args.out}")


if __name__ == "__main__":
    main()
