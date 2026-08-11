"""
GraphLIME explainability for the R-GCN event-pair model.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from tqdm import tqdm

from collections import deque


from src.explainers._shared import (
    ExplainerContext,
    _enriched_label,
    _entity_short_label,
    _k_hop_subgraph,
    _pair_side_caption,
    _scalar_for_model_task,
    _scalar_for_task,
    add_common_explainer_args,
    load_explainer_context,
    setup_matplotlib_style,
    write_summary_json,
)
from src.train import RGCNEncoder, _sanitize, get_event_attrs
from src.utils.io_helpers import load_vocabs

setup_matplotlib_style()

_RESOURCE_OTHERC_RAW_KEY = "http://example.org/hasevent_otherC_org_resource"
_RESOURCE_OTHERC_KEY = _sanitize(_RESOURCE_OTHERC_RAW_KEY)
_LIME_TASK_ALIASES = {"resource": f"otherC_{_RESOURCE_OTHERC_KEY}"}


def _plot_graphlime_signed(
    scores: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
    top_k: int = 20,
    pair_side_text: Optional[str] = None,
) -> None:
    vals = np.asarray(scores, dtype=float)[:top_k]
    lab = list(labels)[:top_k]
    h = max(4.0, len(vals) * 0.25)

    if pair_side_text:
        fig = plt.figure(figsize=(12.8, h))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.32], wspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        ax_r = fig.add_subplot(gs[0, 1])
        ax_r.axis("off")
        ax_r.text(
            0.0,
            1.0,
            pair_side_text,
            transform=ax_r.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            linespacing=1.22,
        )
    else:
        fig, ax = plt.subplots(figsize=(10, h))

    y_pos = np.arange(len(vals))
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in vals]
    ax.barh(y_pos, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Ridge coefficient (red = supports target score, blue = suppresses)")
    ax.set_title(title)
    if pair_side_text:
        fig.subplots_adjust(left=0.26, right=0.98, top=0.9, bottom=0.14)
    else:
        fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _neighbor_mask_graphlime_ridge(
    encoder: RGCNEncoder,
    head: torch.nn.Module,
    x_base: torch.Tensor,
    ei: torch.Tensor,
    et: torch.Tensor,
    src_idx: int,
    feat_node_ids: List[int],
    task: str,
    y_act: Optional[torch.Tensor],
    y_time: Optional[torch.Tensor],
    pair_val_idx: int,
    device: torch.device,
    n_perturb: int,
    mask_frac: float,
    rng: np.random.Generator,
    baseline_row: torch.Tensor,
    use_model_target: bool = True,
    pred_meta: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    K = len(feat_node_ids)
    if K < 1:
        return np.zeros(0, dtype=np.float64)
    x_fix = x_base.detach()
    mask_prob = min(max(float(mask_frac), 1e-6), 1.0)
    rows: List[np.ndarray] = []
    targets: List[float] = []
    weights: List[float] = []
    encoder.eval()
    head.eval()

    def score_mask(keep: np.ndarray) -> float:
        x_pert = x_fix.clone()
        for j, kept in enumerate(keep):
            if kept < 0.5:
                x_pert[feat_node_ids[j]] = baseline_row # DEMO3 replace masked node with baseline embedding (e.g. global mean or zero)

        with torch.no_grad():
            z = encoder(x_pert, ei, et)
            out = head(z[src_idx : src_idx + 1]) #DEMO4 do the prediction with perturbed graph and ge the value
            if use_model_target and task == "activity" and pred_meta:
                t = out["act"][0, int(pred_meta["pred_class_id"])]
            elif use_model_target and pred_meta and pred_meta.get("target_kind") == "otherC":
                key = str(pred_meta["target_key"])
                if key not in out["otherC"]:
                    raise KeyError(f"otherC head missing key (sanitized): {key}")
                t = out["otherC"][key][0, int(pred_meta["pred_class_id"])]
            elif use_model_target:
                t = _scalar_for_model_task(out, task)
            else:
                t = _scalar_for_task(out, task, y_act, y_time, pair_val_idx, device)
            return float(t.item())

    keep_all = np.ones(K, dtype=np.float64)
    rows.append(keep_all.copy())
    targets.append(score_mask(keep_all))
    weights.append(1.0)

    denom = mask_prob ** 2
    for _ in range(n_perturb):
        keep = (rng.random(K) >= mask_prob).astype(np.float64)  # DEMO2 random mask. keeping only ~1-mask_prob fraction of nodes (0.7 by default)
        if K > 1:
            if keep.sum() == K:
                keep[int(rng.integers(K))] = 0.0
            elif keep.sum() == 0:
                keep[int(rng.integers(K))] = 1.0
        elif keep.sum() == K:
            keep[0] = 0.0

        targets.append(score_mask(keep))
        rows.append(keep.copy())
        masked_frac = float(1.0 - keep.mean())
        weights.append(float(np.exp(-((masked_frac ** 2) / denom))))

    X = np.stack(rows, axis=0)
    y = np.asarray(targets, dtype=np.float64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    if X.shape[0] < 2 or not np.isfinite(y).all():
        return np.zeros(K, dtype=np.float64)
    ridge = Ridge(alpha=1.0, fit_intercept=True) # DEMO6 fit ridge regression to the perturbation data (mask patterns -> target score) and return the coefficients as feature importance scores
    ridge.fit(X, y, sample_weight=sample_weight)
    return ridge.coef_.astype(np.float64)

def _structural_graphlime_candidates(
    ei_sub: torch.Tensor,
    num_nodes: int,
    src_local: int,
    dst_local: int,
    limit: int,
) -> List[int]:
    excluded = {int(src_local), int(dst_local)}
    if limit <= 0:
        return []

    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    degree = np.zeros(num_nodes, dtype=np.int64)

    ei_np = ei_sub.detach().cpu().numpy()
    for s_raw, d_raw in zip(ei_np[0], ei_np[1]):
        s, d = int(s_raw), int(d_raw)
        if 0 <= s < num_nodes and 0 <= d < num_nodes:
            adj[s].append(d)
            adj[d].append(s)
            degree[s] += 1
            degree[d] += 1

    def bfs(seed: int) -> np.ndarray:
        dist = np.full(num_nodes, np.iinfo(np.int32).max, dtype=np.int64)
        if not (0 <= seed < num_nodes):
            return dist
        dist[seed] = 0
        queue = deque([seed])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == np.iinfo(np.int32).max:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return dist

    dist_src = bfs(int(src_local))
    dist_dst = bfs(int(dst_local))

    candidates = [node for node in range(num_nodes) if node not in excluded]

    #Build adjacency from ei_sub
    #Compute shortest-path distance from src_local
    #Compute shortest-path distance from dst_local
    #Compute local degree for each node
    #Exclude src_local and dst_local

    #Sort candidates by: (dist_to_src, dist_to_dst, -degree, local_id)
    candidates.sort( # 
        key=lambda node: (
            int(dist_src[node]),
            int(dist_dst[node]),
            -int(degree[node]),
            int(node),
        )
    )
    return candidates[: min(limit, len(candidates))]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Explain R-GCN checkpoints (GraphLIME-style neighbor mask + Ridge only)."
    )
    add_common_explainer_args(ap)
    ap.add_argument("--top-k", type=int, default=30, help="Top nodes per pair in CSV / plot")
    ap.add_argument(
        "--k-hop",
        type=int,
        default=3,
        help="k-hop induced subgraph around pair endpoints before perturbation (default 3)",
    )
    ap.add_argument(
        "--graphlime-perturbations",
        type=int,
        default=200,
        help="Number of random mask samples",
    )
    ap.add_argument(
        "--graphlime-mask-fraction",
        type=float,
        default=0.3,
        help="Expected independent masking probability for each candidate entity",
    )
    ap.add_argument(
        "--graphlime-baseline",
        default="mean",
        choices=["mean", "zero"],
        help="Replacement embedding for masked entities: global mean (default) or zero",
    )
    ap.add_argument(
        "--graphlime-exclude-dst",
        action="store_true",
        help="Compatibility flag; destination is always excluded from maskable/ranked candidates",
    )
    ap.add_argument(
        "--graphlime-candidates",
        type=int,
        default=50,
        help="Maximum structural candidate nodes to mask/rank inside the k-hop subgraph",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx: ExplainerContext = load_explainer_context(args, device)
    encoder, head = ctx.encoder, ctx.head
    x, g, id2ent = ctx.x, ctx.g, ctx.id2ent
    ei_val, et_val = ctx.ei_val, ctx.et_val
    val_pairs = ctx.val_pairs
    y_act_val, y_time_val = ctx.y_act_val, ctx.y_time_val
    act_vocab, id2act = ctx.act_vocab, ctx.id2act
    tasks, n_val, pair_indices = ctx.tasks, ctx.n_val, ctx.pair_indices
    rng = ctx.rng
    vocabs = load_vocabs(args.vocabs) if args.vocabs else {}
    resource_vocab = vocabs.get("c_vocabs", {}).get(_RESOURCE_OTHERC_RAW_KEY, {})
    resource_id2label = {int(class_id): str(label) for label, class_id in resource_vocab.items()}

    os.makedirs(args.out, exist_ok=True)
    graphlime_dir = os.path.join(args.out, "graphlime")
    pair_dir = os.path.join(args.out, "per_pair")
    os.makedirs(graphlime_dir, exist_ok=True)
    os.makedirs(pair_dir, exist_ok=True)

    summary_rows = []
    baseline_row = (
        x.mean(dim=0).detach()
        if args.graphlime_baseline == "mean"
        else torch.zeros(x.size(1), device=device)
    )

    for pi in tqdm(pair_indices, desc="pairs"):
        src_id, dst_id = val_pairs[pi]
        pair_val_idx = pi

        nodes_t, ei_sub, et_sub, g2l = _k_hop_subgraph( #DEMO1 extract the k-hop subgraph around the source and destination entities of the pair
            seeds=[src_id, dst_id],
            edge_index=ei_val,
            edge_type=et_val,
            num_nodes=x.size(0),
            k=max(0, int(args.k_hop)),
        )
        x_sub = x[nodes_t.to(device)]
        ei_sub = ei_sub.to(device)
        et_sub = et_sub.to(device)
        src_local = g2l[src_id]
        dst_local = g2l[dst_id]
        l2g: Dict[int, int] = {ln: gn for gn, ln in g2l.items()}
        # DEMO2 exclude src and dst from candidates 
        feat_ids = _structural_graphlime_candidates(
            ei_sub=ei_sub,
            num_nodes=x_sub.size(0),
            src_local=src_local,
            dst_local=dst_local,
            limit=args.graphlime_candidates,
        )

        for task in tasks:
            model_task = _LIME_TASK_ALIASES.get(task, task)
            act_name, _ts, _, _ = get_event_attrs(g, id2ent[dst_id])
            act_label = act_name if act_name else ""
            if task == "activity" and act_vocab and act_name and act_name in act_vocab:
                act_label = f"{act_name} (id={act_vocab[act_name]})"
            pred_meta: Dict[str, object] = {}
            if task == "activity":
                with torch.no_grad():
                    z_pred = encoder(x_sub, ei_sub, et_sub)
                    out_pred = head(z_pred[src_local : src_local + 1])
                    logits = out_pred["act"][0]
                    pred_id = int(logits.argmax().item())
                pred_meta = {
                    "pred_class_id": pred_id,
                    "pred_class_label": id2act.get(pred_id, str(pred_id)),
                    "pred_score": float(logits[pred_id].item()),
                }
            elif model_task == _LIME_TASK_ALIASES["resource"]:
                with torch.no_grad():
                    z_pred = encoder(x_sub, ei_sub, et_sub)
                    out_pred = head(z_pred[src_local : src_local + 1])
                    if _RESOURCE_OTHERC_KEY not in out_pred["otherC"]:
                        raise KeyError(f"otherC head missing key (sanitized): {_RESOURCE_OTHERC_KEY}")
                    logits = out_pred["otherC"][_RESOURCE_OTHERC_KEY][0]
                    pred_id = int(logits.argmax().item())
                pred_meta = {
                    "target_kind": "otherC",
                    "target_key": _RESOURCE_OTHERC_KEY,
                    "pred_class_id": pred_id,
                    "pred_class_label": resource_id2label.get(pred_id, str(pred_id)),
                    "pred_score": float(logits[pred_id].item()),
                }

            if len(feat_ids) < 1:
                print(
                    f"[warn] graphlime skip pair={pi} task={task}: no context entities "
                    f"after excluding src and dst from the {args.k_hop}-hop subgraph"
                )
            else:
                coef_gl = _neighbor_mask_graphlime_ridge(
                    encoder,
                    head,
                    x_sub,
                    ei_sub,
                    et_sub,
                    src_local,
                    feat_ids,
                    model_task,
                    y_act_val,
                    y_time_val,
                    pair_val_idx,
                    device,
                    n_perturb=args.graphlime_perturbations,
                    mask_frac=float(args.graphlime_mask_fraction),
                    rng=rng,
                    baseline_row=baseline_row,
                    pred_meta=pred_meta,
                )
                #DEMO7 get absolute values of coefficients for ranking, and coef_signed for directionality (positive = supports target score, negative = suppresses target score)
                abs_coef = np.abs(coef_gl) 
                order_gl = np.argsort(-abs_coef)
                gl_rows = []
                for r, j in enumerate(order_gl[: args.top_k]):
                    local_id = int(feat_ids[int(j)])
                    nid = int(l2g.get(local_id, local_id))
                    u = id2ent.get(nid, "?")
                    row_gl = {
                        "rank": r + 1,
                        "entity_id": nid,
                        "local_id": local_id,
                        "uri": u,
                        "label": _entity_short_label(g, u),
                        "coef_signed": float(coef_gl[int(j)]),
                        "abs_coef": float(abs_coef[int(j)]),
                        "dst_activity": act_label,
                    }
                    row_gl.update(pred_meta)
                    gl_rows.append(row_gl)
                gl_csv = f"pair{pi:05d}_{task}_graphlime.csv"
                pd.DataFrame(gl_rows).to_csv(os.path.join(pair_dir, gl_csv), index=False)
                plot_order = order_gl[:20]
                gl_labels = [
                    _enriched_label(g, id2ent, int(l2g.get(int(feat_ids[int(j)]), int(feat_ids[int(j)]))))
                    for j in plot_order
                ]
                side_txt = _pair_side_caption(g, id2ent, src_id, dst_id, pi)
                _plot_graphlime_signed(
                    coef_gl[plot_order],
                    gl_labels,
                    f"GraphLIME-style signed node mask (pair {pi}, {task})",
                    os.path.join(graphlime_dir, f"pair{pi:05d}_{task}.png"),
                    top_k=20,
                    pair_side_text=side_txt,
                )
                summary_rows.append(
                    {
                        "pair_idx": pi,
                        "task": task,
                        "method": "graphlime",
                        "src_id": src_id,
                        "dst_id": dst_id,
                        "csv": gl_csv,
                        **pred_meta,
                        "sub_nodes": int(x_sub.size(0)),
                        "sub_edges": int(et_sub.size(0)),
                    }
                )

    write_summary_json(
        os.path.join(args.out, "summary.json"),
        args,
        n_val,
        pair_indices,
        tasks,
        extra_fields={
            "k_hop": args.k_hop,
            "graphlime_perturbations": args.graphlime_perturbations,
            "graphlime_mask_fraction": args.graphlime_mask_fraction,
            "graphlime_baseline": args.graphlime_baseline,
            "graphlime_exclude_dst": bool(args.graphlime_exclude_dst),
            "src_dst_excluded_from_candidates": True,
            "graphlime_candidates": args.graphlime_candidates,
        },
        rows=summary_rows,
    )

    print(f"Done. Wrote summary → {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
