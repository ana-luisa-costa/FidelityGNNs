
from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from src.explainers._shared import (
    ExplainerContext,
    _entity_short_label,
    _k_hop_subgraph,
    add_common_explainer_args,
    load_explainer_context,
    setup_matplotlib_style,
    write_summary_json,
)
from src.explainers.prophet_core import (
    ProphetConfig,
    ProphetExplanation,
    explain_prophet_masks,
    prophet_seed,
    validate_classification_task,
)
from src.fidelity.fidelity_utils import task_target_value_nodes
from src.gnn4ppm.train import _sanitize, get_event_attrs
from src.utils.io_helpers import load_vocabs

setup_matplotlib_style()

__all__ = [
    "ProphetConfig",
    "ProphetExplanation",
    "explain_prophet_masks",
    "prophet_seed",
    "validate_classification_task",
]


def _otherc_id2label_by_key(vocabs: Dict[str, object]) -> Dict[str, Dict[int, str]]:
    result: Dict[str, Dict[int, str]] = {}
    for raw_key, label_to_id in vocabs.get("c_vocabs", {}).items():
        if isinstance(label_to_id, dict):
            result[_sanitize(str(raw_key))] = {
                int(class_id): str(label)
                for label, class_id in label_to_id.items()
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
    explanation: ProphetExplanation,
    nodes_global: Tensor,
    g_rdf,
    id2ent: Dict[int, str],
) -> List[dict]:
    protected = set(explanation.protected_nodes)
    selectable = [
        local_id for local_id in range(nodes_global.numel()) if local_id not in protected
    ]
    ranked = sorted(
        selectable,
        key=lambda local_id: (-float(explanation.node_mask[local_id].item()), local_id),
    )
    rank_by_local = {local_id: rank + 1 for rank, local_id in enumerate(ranked)}
    ordered = ranked + sorted(protected)
    rows: List[dict] = []
    for local_id in ordered:
        global_id = int(nodes_global[local_id].item())
        uri = id2ent.get(global_id, "?")
        rows.append(
            {
                "rank": rank_by_local.get(local_id, ""),
                "local_id": local_id,
                "entity_id": global_id,
                "uri": uri,
                "label": _entity_short_label(g_rdf, uri),
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
    explanation: ProphetExplanation,
    nodes_global: Tensor,
    ei_sub: Tensor,
    et_sub: Tensor,
    num_rel: int,
    id2rel: Dict[int, str],
    g_rdf,
    id2ent: Dict[int, str],
) -> List[dict]:
    order = torch.argsort(explanation.edge_mask, descending=True).tolist()
    rows: List[dict] = []
    for rank, edge_id in enumerate(order, start=1):
        src_local = int(ei_sub[0, edge_id].item())
        dst_local = int(ei_sub[1, edge_id].item())
        src_global = int(nodes_global[src_local].item())
        dst_global = int(nodes_global[dst_local].item())
        relation_id = int(et_sub[edge_id].item())
        base_relation_id = relation_id % num_rel
        rows.append(
            {
                "rank": rank,
                "edge_id": edge_id,
                "edge_mask": float(explanation.edge_mask[edge_id].item()),
                "src_local": src_local,
                "dst_local": dst_local,
                "src_global": src_global,
                "dst_global": dst_global,
                "src_label": _entity_short_label(g_rdf, id2ent.get(src_global, "?")),
                "dst_label": _entity_short_label(g_rdf, id2ent.get(dst_global, "?")),
                "relation_id": relation_id,
                "base_relation_id": base_relation_id,
                "relation_uri": id2rel.get(base_relation_id, str(base_relation_id)),
                "direction": "inverse" if relation_id >= num_rel else "forward",
            }
        )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PROPHET node/edge-mask explainer for the R-GCN event-pair model."
    )
    add_common_explainer_args(parser)
    parser.set_defaults(tasks="activity")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument(
        "--prophet-subgraph-hop",
        "--num-hops",
        dest="prophet_subgraph_hop",
        type=int,
        default=3,
        help="Pair-centered subgraph depth; default 3 matches the R-GCN receptive field.",
    )
    parser.add_argument("--top-nodes", type=int, default=25)
    parser.add_argument("--top-edges", type=int, default=15)
    parser.add_argument("--gnn-epochs", type=int, default=100)
    parser.add_argument("--gnn-lr", type=float, default=0.01)
    parser.add_argument("--edge-size-reg", type=float, default=0.005)
    parser.add_argument("--edge-ent-reg", type=float, default=1.0)
    parser.add_argument(
        "--node-size-reg",
        "--feat-size-reg",
        dest="node_size_reg",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--node-ent-reg",
        "--feat-ent-reg",
        dest="node_ent_reg",
        type=float,
        default=0.1,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    requested_tasks = [part.strip() for part in args.tasks.split(",") if part.strip()]
    normalized_tasks = [validate_classification_task(task) for task in requested_tasks]
    args.tasks = ",".join(normalized_tasks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context: ExplainerContext = load_explainer_context(args, device)
    encoder, head = context.encoder, context.head
    x, graph, id2ent = context.x, context.g, context.id2ent
    ei_test, et_test = context.ei_test, context.et_test
    test_pairs = context.test_pairs
    num_rel, rel2id = context.num_rel, context.rel2id
    id2rel = {value: key for key, value in rel2id.items()}
    tasks, pair_indices = context.tasks, context.pair_indices

    vocabs = load_vocabs(args.vocabs) if args.vocabs else {}
    otherc_id2label = _otherc_id2label_by_key(vocabs)

    pair_dir = os.path.join(args.out, "per_pair")
    os.makedirs(pair_dir, exist_ok=True)

    summary_rows: List[dict] = []
    for pair_idx in tqdm(pair_indices, desc="PROPHET pairs"):
        src_global, dst_global = test_pairs[pair_idx]
        if args.prophet_subgraph_hop > 0:
            nodes, ei_sub, et_sub, g2l = _k_hop_subgraph(
                seeds=[src_global, dst_global],
                edge_index=ei_test,
                edge_type=et_test,
                num_nodes=x.size(0),
                k=args.prophet_subgraph_hop,
            )
            x_sub = x[nodes.to(device)].detach()
            ei_sub, et_sub = ei_sub.to(device), et_sub.to(device)
        else:
            nodes = torch.arange(x.size(0))
            x_sub, ei_sub, et_sub = x.detach(), ei_test, et_test
            g2l = {index: index for index in range(x.size(0))}

        src_local = g2l[src_global]
        dst_local = g2l.get(dst_global, -1)
        if et_sub.numel() == 0:
            warnings.warn(f"PROPHET pair={pair_idx}: empty subgraph; skipping")
            continue

        for task in tasks:
            protected = {
                node
                for node in (src_local, dst_local)
                if 0 <= int(node) < x_sub.size(0)
            }
            protected.update(
                task_target_value_nodes(graph, id2ent, nodes, g2l, dst_global, task)
            )
            config = ProphetConfig(
                epochs=args.gnn_epochs,
                lr=args.gnn_lr,
                edge_size_reg=args.edge_size_reg,
                edge_ent_reg=args.edge_ent_reg,
                node_size_reg=args.node_size_reg,
                node_ent_reg=args.node_ent_reg,
                seed=prophet_seed(args.seed, pair_idx, task),
            )

            started = time.perf_counter()
            try:
                explanation = explain_prophet_masks(
                    encoder=encoder,
                    head=head,
                    x_sub=x_sub,
                    ei_sub=ei_sub,
                    et_sub=et_sub,
                    src_local=src_local,
                    task=task,
                    protected_nodes=sorted(protected),
                    config=config,
                )
            except Exception as exc:
                warnings.warn(f"PROPHET failed pair={pair_idx} task={task}: {exc}")
                continue
            elapsed = time.perf_counter() - started
            explanation.pred_class_label = _predicted_class_label(
                task, explanation.pred_class_id, context.id2act, otherc_id2label
            )

            protected_result = set(explanation.protected_nodes)
            selectable = [
                local_id
                for local_id in range(x_sub.size(0))
                if local_id not in protected_result
            ]
            selectable_scores = explanation.node_mask[selectable]
            degenerate = bool(
                selectable_scores.numel() == 0
                or float(selectable_scores.max() - selectable_scores.min()) < 1e-8
            )
            if degenerate:
                warnings.warn(f"PROPHET pair={pair_idx} task={task}: degenerate node mask")
            if not explanation.prediction_agreement:
                warnings.warn(
                    f"PROPHET pair={pair_idx} task={task}: masked prediction changed "
                    f"from class {explanation.pred_class_id} to {explanation.masked_class_id}"
                )

            stem = f"pair{pair_idx:05d}_{task}_prophet"
            node_rows = _node_rows(explanation, nodes, graph, id2ent)
            edge_rows = _edge_rows(
                explanation,
                nodes,
                ei_sub,
                et_sub,
                num_rel,
                id2rel,
                graph,
                id2ent,
            )
            node_csv = f"{stem}_nodes.csv"
            edge_csv = f"{stem}_edges.csv"
            trace_csv = f"{stem}_trace.csv"
            pd.DataFrame(node_rows).to_csv(os.path.join(pair_dir, node_csv), index=False)
            pd.DataFrame(edge_rows).to_csv(os.path.join(pair_dir, edge_csv), index=False)
            pd.DataFrame(explanation.trace).to_csv(
                os.path.join(pair_dir, trace_csv), index=False
            )

            dst_activity, _, _, _ = get_event_attrs(
                graph, id2ent.get(dst_global, "?")
            )
            summary_rows.append(
                {
                    "pair_idx": pair_idx,
                    "task": task,
                    "method": "prophet",
                    "method_detail": "PROPHET adapted to R-GCN event-pair prediction",
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
                    "num_edges_sub": et_sub.size(0),
                    "num_selectable_nodes": len(selectable),
                    "protected_local_ids": list(explanation.protected_nodes),
                    "protected_mask_semantics": "mandatory_context_fixed_to_1",
                    "mean_selectable_node_mask": (
                        float(selectable_scores.mean().item())
                        if selectable_scores.numel()
                        else float("nan")
                    ),
                    "mean_edge_mask": float(explanation.edge_mask.mean().item()),
                    "degenerate_node_mask": degenerate,
                    "seed": explanation.seed,
                    "epochs": config.epochs,
                    "elapsed_s": round(elapsed, 3),
                    "encoder_parity_max_abs_diff": explanation.encoder_parity_max_abs_diff,
                    "node_grad_finite": explanation.node_grad_finite,
                    "edge_grad_finite": explanation.edge_grad_finite,
                    "node_csv": node_csv,
                    "edge_csv": edge_csv,
                    "trace_csv": trace_csv,
                }
            )

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    write_summary_json(
        os.path.join(args.out, "meta.json"),
        args,
        context.n_test,
        pair_indices,
        tasks,
        extra_fields={
            "method": "prophet",
            "method_detail": "PROPHET adapted to R-GCN event-pair prediction",
            "target_semantics": "original_model_predicted_class",
            "primary_mask": "node",
            "supplementary_mask": "edge",
            "protected_mask_semantics": "mandatory_context_fixed_to_1",
            "prophet_subgraph_hop": args.prophet_subgraph_hop,
            "config": asdict(
                ProphetConfig(
                    epochs=args.gnn_epochs,
                    lr=args.gnn_lr,
                    edge_size_reg=args.edge_size_reg,
                    edge_ent_reg=args.edge_ent_reg,
                    node_size_reg=args.node_size_reg,
                    node_ent_reg=args.node_ent_reg,
                    seed=args.seed,
                )
            ),
        },
        rows=summary_rows,
    )
    print(f"Done. PROPHET results -> {args.out}")


if __name__ == "__main__":
    main()
