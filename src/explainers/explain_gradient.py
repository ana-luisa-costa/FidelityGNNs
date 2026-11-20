
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx

from src.explainers._shared import (
    ExplainerContext,
    _enriched_label,
    _entity_short_label,
    _pair_side_caption,
    _plot_top_entities,
    _plot_top_entities_signed,
    _plot_top_entities_signed_noreorder,
    add_common_explainer_args,
    load_explainer_context,
    setup_matplotlib_style,
    write_summary_json,
)
from gnn4ppm.train import get_event_attrs

setup_matplotlib_style()

PLOT_TOP_K = 20


def _model_prediction_scalar(
    out: dict, task: str
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if task == "activity":
        logits = out["act"][0]
        pred_id = int(logits.argmax().item())
        return logits[pred_id], {"pred_class_id": pred_id}
    if task == "time":
        v = out["time"][0]
        scalar = v if v.numel() == 1 else v.sum()
        return scalar, {"pred_value": float(scalar.detach().item())}
    if task.startswith("otherC_"):
        k = task[len("otherC_"):]
        if k not in out["otherC"]:
            raise KeyError(f"otherC head missing key (sanitized): {k}")
        logits = out["otherC"][k][0]
        pred_id = int(logits.argmax().item())
        return logits[pred_id], {
            "pred_class_id": pred_id,
            "pred_class_label": str(pred_id),
        }
    if task.startswith("otherN_"):
        k = task[len("otherN_"):]
        if k not in out["otherN"]:
            raise KeyError(f"otherN head missing key: {k}")
        v = out["otherN"][k][0]
        scalar = v if v.numel() == 1 else v.sum()
        return scalar, {"pred_value": float(scalar.detach().item())}
    raise ValueError(f"Unknown task: {task}")


def _gradient_saliency(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    x: torch.Tensor,
    ei: torch.Tensor,
    et: torch.Tensor,
    src_id: int,
    task: str,
    y_act: Optional[torch.Tensor],
    y_time: Optional[torch.Tensor],
    pair_test_idx: int,
    device: torch.device,
    use_model_target: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    x_req = x.detach().clone().requires_grad_(True)
    z = encoder(x_req, ei, et)
    out = head(z[src_id : src_id + 1])
    scalar, pred_meta = _model_prediction_scalar(out, task)
    encoder.zero_grad()
    head.zero_grad()
    scalar.backward()
    g = x_req.grad
    if g is None:
        n = x.size(0)
        return (
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            pred_meta,
        )
    grad_x_input_signed = (g * x_req).sum(dim=1).detach().cpu().numpy()
    grad_x_input_abs = (g * x_req).abs().sum(dim=1).detach().cpu().numpy()
    return grad_x_input_signed, grad_x_input_abs, pred_meta


def _save_pair_plots(
    g,
    id2ent: Dict[int, str],
    plot_dir: str,
    pair_idx: int,
    task: str,
    src_id: int,
    dst_id: int,
    top_local_ids: List[int],
    top_global_ids: List[int],
    gxi_abs: np.ndarray,
    gxi_signed: np.ndarray,
    noreorder_only: bool,
) -> None:
    local_ids = top_local_ids[:PLOT_TOP_K]
    labels = [_enriched_label(g, id2ent, nid) for nid in top_global_ids[:PLOT_TOP_K]]
    signed_scores = np.array([float(gxi_signed[i]) for i in local_ids])
    side_text = _pair_side_caption(g, id2ent, src_id, dst_id, pair_idx)
    base = os.path.join(plot_dir, f"pair{pair_idx:05d}_{task}")

    if not noreorder_only:
        abs_scores = np.array([float(gxi_abs[i]) for i in local_ids])
        _plot_top_entities(
            abs_scores,
            labels,
            f"Gradient saliency (|Input*Gradient|) — pair {pair_idx}, {task}",
            f"{base}.png",
            top_k=PLOT_TOP_K,
            xlabel="|grad · x|  (input * gradient magnitude)",
            pair_side_text=side_text,
        )
        _plot_top_entities_signed(
            signed_scores,
            labels,
            f"Signed gradient saliency (Input*Gradient) — pair {pair_idx}, {task}",
            f"{base}_signed.png",
            top_k=PLOT_TOP_K,
            xlabel="grad · x  (red = supports prediction, blue = opposes prediction)",
            pair_side_text=side_text,
        )

    _plot_top_entities_signed_noreorder(
        signed_scores,
        labels,
        f"Gradient saliency (signed Input*Gradient, ranked by |Input*Gradient|) - pair {pair_idx}, {task}",
        f"{base}_signed_noreorder.png",
        top_k=PLOT_TOP_K,
        xlabel="grad · x  (no reorder, red = supports prediction, blue = opposes prediction)",
        pair_side_text=side_text,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Input * Gradient saliency explainer for the R-GCN event-pair model."
    )
    add_common_explainer_args(ap)
    ap.add_argument("--top-candidates", type=int, default=30)
    ap.add_argument("--k-hop", type=int, default=0)
    ap.add_argument("--gradient-exclude-dst", action="store_true")
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument(
        "--noreorder-only",
        action="store_true",
        help="With --save-plots, write only the signed_noreorder plot.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx: ExplainerContext = load_explainer_context(args, device)
    encoder, head = ctx.encoder, ctx.head
    x, g, id2ent = ctx.x, ctx.g, ctx.id2ent
    test_pairs = ctx.test_pairs
    y_act_test, y_time_test = ctx.y_act_test, ctx.y_time_test
    act_vocab, id2act = ctx.act_vocab, ctx.id2act
    tasks, n_test, pair_indices = ctx.tasks, ctx.n_test, ctx.pair_indices

    os.makedirs(args.out, exist_ok=True)
    gradient_dir = os.path.join(args.out, "gradient")
    pair_dir = os.path.join(args.out, "per_pair")
    if args.save_plots:
        os.makedirs(gradient_dir, exist_ok=True)
    os.makedirs(pair_dir, exist_ok=True)

    summary_rows: List[dict] = []

    for pi in tqdm(pair_indices, desc="pairs"):
        src_id, dst_id = test_pairs[pi]

        # prefix graph of the pair (k_hop<=0 keeps the whole prefix); dst is not in it
        nodes_t, ei_work, et_work, g2l = ctx.pair_subgraph(pi, hops=max(0, int(args.k_hop)))
        x_work = x[nodes_t.to(device)]
        ei_work = ei_work.to(device)
        et_work = et_work.to(device)
        src_work = g2l[src_id]
        dst_work = g2l.get(dst_id, -1)
        local_to_global = [int(n) for n in nodes_t.tolist()]

        for task in tasks:
            try:
                gxi_signed, gxi_abs, pred_meta = _gradient_saliency(
                    encoder, head, x_work, ei_work, et_work,
                    src_work, task, y_act_test, y_time_test, pi, device,
                    use_model_target=True,
                )
            except Exception as exc:
                print(f"[warn] gradient failed pair={pi} task={task}: {exc}")
                continue

            act_name, _, _, _ = get_event_attrs(g, id2ent.get(dst_id, ""))
            act_label = act_name or ""
            if task == "activity" and act_vocab and act_name and act_name in act_vocab:
                act_label = f"{act_name} (id={act_vocab[act_name]})"
            if task == "activity" and "pred_class_id" in pred_meta:
                pred_meta = dict(pred_meta)
                pred_meta["pred_class_label"] = id2act.get(
                    int(pred_meta["pred_class_id"]), str(pred_meta["pred_class_id"])
                )

            all_local_ids = range(x_work.size(0))
            exclude = {src_work}
            if args.gradient_exclude_dst and dst_work >= 0:
                exclude.add(dst_work)
            candidate_ids = [int(i) for i in all_local_ids if int(i) not in exclude]
            sorted_idx = np.argsort(-gxi_abs[candidate_ids])
            top_local_ids = [candidate_ids[j] for j in sorted_idx[: args.top_candidates]]
            top_global_ids = [local_to_global[i] for i in top_local_ids]

            rows = []
            for rank, (local_id, global_id) in enumerate(
                zip(top_local_ids, top_global_ids)
            ):
                uri = id2ent.get(int(global_id), "?")
                row: dict = {
                    "rank": rank + 1,
                    "entity_id": int(global_id),
                    "uri": uri,
                    "label": _entity_short_label(g, uri),
                    "gradXinput_signed": float(gxi_signed[local_id]),
                    "gradXinput_abs": float(gxi_abs[local_id]),
                    "dst_activity": act_label,
                }
                row.update(pred_meta)
                rows.append(row)

            csv_name = f"pair{pi:05d}_{task}_gradient.csv"
            pd.DataFrame(rows).to_csv(os.path.join(pair_dir, csv_name), index=False)

            if args.save_plots:
                _save_pair_plots(
                    g, id2ent, gradient_dir, pi, task, src_id, dst_id,
                    top_local_ids, top_global_ids, gxi_abs, gxi_signed,
                    args.noreorder_only,
                )

            summary_row = {
                "pair_idx": pi,
                "task": task,
                "method": "gradient",
                "src_id": src_id,
                "dst_id": dst_id,
                "csv": csv_name,
            }
            summary_row.update(pred_meta)
            summary_rows.append(summary_row)

    write_summary_json(
        os.path.join(args.out, "summary_gradient.json"),
        args,
        n_test,
        pair_indices,
        tasks,
        extra_fields={
            "k_hop": args.k_hop,
            "gradient_exclude_dst": args.gradient_exclude_dst,
            "save_plots": args.save_plots,
            "noreorder_only": args.noreorder_only,
        },
        rows=summary_rows,
    )

    print(f"\nDone. Results -> {args.out}")


if __name__ == "__main__":
    main()
