
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from tqdm import tqdm

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
from src.gnn4ppm.train import _sanitize, get_event_attrs
from src.utils.io_helpers import load_vocabs

setup_matplotlib_style()

_RESOURCE_OTHERC_RAW_KEY = "http://example.org/hasevent_otherC_org_resource"
_RESOURCE_OTHERC_KEY = _sanitize(_RESOURCE_OTHERC_RAW_KEY)
_SHAP_TASK_ALIASES = {"resource": f"otherC_{_RESOURCE_OTHERC_KEY}"}


def _resolve_nsamples(nsamples, K: int) -> int:
    if nsamples == "auto":
        return min(2 * K + 256, 1024)
    return int(nsamples)


def _parse_shap_baselines(raw: str) -> List[str]:
    baselines = [part.strip() for part in raw.split(",") if part.strip()]
    if not baselines:
        baselines = ["mean"]
    allowed = {"zero", "mean"}
    invalid = [baseline for baseline in baselines if baseline not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            "--shap-baseline must contain only 'zero' and/or 'mean' "
            f"(got: {', '.join(invalid)})"
        )
    deduped: List[str] = []
    seen = set()
    for baseline in baselines:
        if baseline not in seen:
            deduped.append(baseline)
            seen.add(baseline)
    return deduped


def _structural_shap_candidates(
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
    candidates.sort(
        key=lambda node: (
            int(dist_src[node]),
            int(dist_dst[node]),
            -int(degree[node]),
            int(node),
        )
    )
    return candidates[: min(limit, len(candidates))]


def _build_predict_fn(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    x_sub: torch.Tensor,
    ei_sub: torch.Tensor,
    et_sub: torch.Tensor,
    src_local: int,
    feat_ids_local: List[int],
    task: str,
    y_act: Optional[torch.Tensor],
    y_time: Optional[torch.Tensor],
    pair_test_idx: int,
    device: torch.device,
    baseline_row: torch.Tensor,
    pbar: Optional[tqdm] = None,
    use_model_target: bool = True,
    pred_meta: Optional[Dict[str, object]] = None,
):
    encoder.eval()
    head.eval()
    x_sub_ref = x_sub.detach()
    x_pert = x_sub_ref.clone()
    prev_zeroed: List[int] = []

    def predict_fn(masks: np.ndarray) -> np.ndarray:
        out_vals = np.empty(masks.shape[0], dtype=np.float64)
        for r in range(masks.shape[0]):
            for j in prev_zeroed:
                x_pert[j] = x_sub_ref[j]
            prev_zeroed.clear()
            for j, kept in enumerate(masks[r]):
                if kept < 0.5:
                    x_pert[feat_ids_local[j]] = baseline_row
                    prev_zeroed.append(feat_ids_local[j])
            with torch.no_grad():
                z = encoder(x_pert, ei_sub, et_sub)
                out = head(z[src_local : src_local + 1])
                if use_model_target and pred_meta and pred_meta.get("target_kind") == "activity":
                    val = out["act"][0, int(pred_meta["pred_class_id"])]
                elif use_model_target and pred_meta and pred_meta.get("target_kind") == "otherC":
                    key = str(pred_meta["target_key"])
                    if key not in out["otherC"]:
                        raise KeyError(f"otherC head missing key (sanitized): {key}")
                    val = out["otherC"][key][0, int(pred_meta["pred_class_id"])]
                elif use_model_target:
                    val = _scalar_for_model_task(out, task)
                else:
                    val = _scalar_for_task(out, task, y_act, y_time, pair_test_idx, device)
                out_vals[r] = float(val.item())
        if pbar is not None:
            pbar.update(masks.shape[0])
        return out_vals

    return predict_fn


def _otherc_id2label_by_key(vocabs: Dict[str, object]) -> Dict[str, Dict[int, str]]:
    id2label_by_key: Dict[str, Dict[int, str]] = {}
    for raw_key, label_to_id in vocabs.get("c_vocabs", {}).items():
        if not isinstance(label_to_id, dict):
            continue
        id2label_by_key[_sanitize(str(raw_key))] = {
            int(class_id): str(label) for label, class_id in label_to_id.items()
        }
    return id2label_by_key


def _classification_prediction_metadata(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    x_sub: torch.Tensor,
    ei_sub: torch.Tensor,
    et_sub: torch.Tensor,
    src_local: int,
    task: str,
    id2act: Dict[int, str],
    otherc_id2label: Dict[str, Dict[int, str]],
) -> Dict[str, object]:
    with torch.no_grad():
        z = encoder(x_sub, ei_sub, et_sub)
        out = head(z[src_local : src_local + 1])
    if task == "activity":
        logits = out["act"][0]
        pred_id = int(logits.argmax().item())
        return {
            "target_kind": "activity",
            "pred_class_id": pred_id,
            "pred_class_label": id2act.get(pred_id, str(pred_id)),
            "pred_score": float(logits[pred_id].item()),
        }
    if task.startswith("otherC_"):
        key = task[len("otherC_"):]
        if key not in out["otherC"]:
            raise KeyError(f"otherC head missing key (sanitized): {key}")
        logits = out["otherC"][key][0]
        pred_id = int(logits.argmax().item())
        return {
            "target_kind": "otherC",
            "target_key": key,
            "pred_class_id": pred_id,
            "pred_class_label": otherc_id2label.get(key, {}).get(pred_id, str(pred_id)),
            "pred_score": float(logits[pred_id].item()),
        }
    return {}


def _run_kernel_shap(
    predict_fn,
    K: int,
    nsamples: int,
    l1_reg: str,
) -> np.ndarray:
    background = np.zeros((1, K), dtype=np.float64)
    instance = np.ones((1, K), dtype=np.float64)
    explainer = shap.KernelExplainer(predict_fn, background, link="identity")
    raw = explainer.shap_values(instance, nsamples=nsamples, l1_reg=l1_reg, silent=False)
    if isinstance(raw, list):
        raw = raw[0]
    return np.asarray(raw, dtype=np.float64).reshape(-1)


def _plot_top_entities_signed(
    phi: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
    top_k: int = 20,
    pair_side_text: Optional[str] = None,
) -> None:
    order = np.argsort(-np.abs(phi))[:top_k]
    vals = phi[order]
    lab = [labels[i] for i in order]

    h = max(4.0, top_k * 0.25)
    if pair_side_text:
        fig = plt.figure(figsize=(12.8, h))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.32], wspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        ax_r = fig.add_subplot(gs[0, 1])
        ax_r.axis("off")
        ax_r.text(0.0, 1.0, pair_side_text, transform=ax_r.transAxes,
                  va="top", ha="left", fontsize=7.5, linespacing=1.22)
    else:
        fig, ax = plt.subplots(figsize=(10, h))

    y_pos = np.arange(len(vals))
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in vals]
    ax.barh(y_pos, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Shapley value φ  (red = pushes prediction up, blue = down)")
    ax.set_title(title)
    if pair_side_text:
        fig.subplots_adjust(left=0.26, right=0.98, top=0.9, bottom=0.14)
    else:
        fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="KernelSHAP entity-level explainer for the R-GCN event-pair model."
    )
    add_common_explainer_args(ap)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--k-hop", type=int, default=0,
                    help="Deprecated; kept for compatibility.")
    ap.add_argument("--shap-candidates", type=int, default=25, dest="shap_candidates",
                    help="K: structural candidate entities fed to KernelSHAP per pair (default 25)")
    ap.add_argument("--shap-nsamples", default=300, dest="shap_nsamples",
                    help="Coalitions per KernelExplainer call: int or 'auto' → min(2K+256,1024)")
    ap.add_argument("--shap-l1-reg", default="aic", dest="shap_l1_reg",
                    help="L1 regularisation: 'aic', 'bic', 'auto', or float (default 'aic')")
    ap.add_argument("--shap-baseline", default="mean", dest="shap_baseline",
                    type=_parse_shap_baselines,
                    help="Ablation baseline: 'mean' (default), 'zero', or comma-separated 'zero,mean'")
    ap.add_argument("--shap-exclude-dst", action="store_true", dest="shap_exclude_dst",
                    help="Compatibility flag; source and destination are always excluded")
    ap.add_argument("--shap-subgraph-hop", type=int, default=3, dest="shap_subgraph_hop",
                    help="Subgraph depth for encoder forward pass; 3 = exact for 3-layer RGCN")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx: ExplainerContext = load_explainer_context(args, device)
    encoder, head = ctx.encoder, ctx.head
    x, g, id2ent = ctx.x, ctx.g, ctx.id2ent
    ei_test, et_test = ctx.ei_test, ctx.et_test
    test_pairs = ctx.test_pairs
    y_act_test, y_time_test = ctx.y_act_test, ctx.y_time_test
    act_vocab, id2act = ctx.act_vocab, ctx.id2act
    tasks, n_test, pair_indices = ctx.tasks, ctx.n_test, ctx.pair_indices
    vocabs = load_vocabs(args.vocabs) if args.vocabs else {}
    otherc_id2label = _otherc_id2label_by_key(vocabs)

    baseline_rows: Dict[str, torch.Tensor] = {
        "mean": x.mean(dim=0).detach(),
        "zero": torch.zeros(x.size(1), device=device),
    }

    os.makedirs(args.out, exist_ok=True)
    shap_dir = os.path.join(args.out, "shap")
    pair_dir = os.path.join(args.out, "per_pair")
    for d in (shap_dir, pair_dir):
        os.makedirs(d, exist_ok=True)

    summary_rows: List[dict] = []

    for pi in tqdm(pair_indices, desc="pairs"):
        src_id, dst_id = test_pairs[pi]

        if args.shap_subgraph_hop > 0:
            nodes_t, ei_sub, et_sub, g2l = _k_hop_subgraph(
                seeds=[src_id, dst_id],
                edge_index=ei_test,
                edge_type=et_test,
                num_nodes=x.size(0),
                k=args.shap_subgraph_hop,
            )
            x_sub = x[nodes_t.to(device)]
            ei_sub = ei_sub.to(device)
            et_sub = et_sub.to(device)
            src_local = g2l[src_id]
            dst_local = g2l[dst_id]
        else:
            nodes_t = torch.arange(x.size(0))
            ei_sub, et_sub, x_sub = ei_test, et_test, x
            src_local, dst_local = src_id, dst_id
            g2l = None

        N_sub = x_sub.size(0)
        E_sub = et_sub.size(0)
        feat_ids_local = _structural_shap_candidates(
            ei_sub=ei_sub,
            num_nodes=N_sub,
            src_local=src_local,
            dst_local=dst_local,
            limit=args.shap_candidates,
        )
        feat_ids_global = [int(nodes_t[i].item()) for i in feat_ids_local]

        for task in tasks:
            model_task = _SHAP_TASK_ALIASES.get(task, task)
            K = len(feat_ids_global)
            if K < 2:
                print(f"[warn] shap skip pair={pi} task={task}: only {K} candidate(s)")
                continue

            pred_meta = _classification_prediction_metadata(
                encoder,
                head,
                x_sub,
                ei_sub,
                et_sub,
                src_local,
                model_task,
                id2act,
                otherc_id2label,
            )
            nsamples_resolved = _resolve_nsamples(args.shap_nsamples, K)
            for shap_baseline in args.shap_baseline:
                t0 = time.time()
                pbar = tqdm(total=nsamples_resolved, desc=f"  pair{pi:05d}/{task}/{shap_baseline}",
                            leave=False, unit="coal")
                predict_fn = _build_predict_fn(
                    encoder=encoder, head=head,
                    x_sub=x_sub, ei_sub=ei_sub, et_sub=et_sub,
                    src_local=src_local, feat_ids_local=feat_ids_local,
                    task=model_task, y_act=y_act_test, y_time=y_time_test,
                    pair_test_idx=pi, device=device,
                    baseline_row=baseline_rows[shap_baseline], pbar=pbar,
                    use_model_target=True,
                    pred_meta=pred_meta,
                )

                try:
                    phi = _run_kernel_shap(predict_fn, K, nsamples_resolved, args.shap_l1_reg)
                except Exception as exc:
                    pbar.close()
                    print(f"[warn] KernelSHAP failed pair={pi} task={task} baseline={shap_baseline}: {exc}")
                    continue
                finally:
                    pbar.close()

                elapsed = time.time() - t0
                print(
                    f"  [pair{pi:05d}/{task}/{shap_baseline}] "
                    f"nodes={N_sub}  edges={E_sub}  K={K}  "
                    f"nsamples={nsamples_resolved}  {elapsed:.1f}s"
                )

                act_name, _, _, _ = get_event_attrs(g, id2ent.get(dst_id, ""))
                act_label = act_name or ""
                if task == "activity" and act_vocab and act_name and act_name in act_vocab:
                    act_label = f"{act_name} (id={act_vocab[act_name]})"

                abs_phi = np.abs(phi)
                order = np.argsort(-abs_phi)

                rows: List[dict] = []
                for rank, j in enumerate(order[: args.top_k]):
                    nid = feat_ids_global[int(j)]
                    uri = id2ent.get(int(nid), "?")
                    row: dict = {
                        "rank": rank + 1,
                        "entity_id": int(nid),
                        "uri": uri,
                        "label": _entity_short_label(g, uri),
                        "shap_baseline": shap_baseline,
                        "shap_value": float(phi[int(j)]),
                        "abs_shap": float(abs_phi[int(j)]),
                        "dst_activity": act_label,
                        **pred_meta,
                    }
                    rows.append(row)

                csv_name = f"pair{pi:05d}_{task}_{shap_baseline}_shap.csv"
                pd.DataFrame(rows).to_csv(os.path.join(pair_dir, csv_name), index=False)

                plot_n = min(20, K)
                plot_order = order[:plot_n]
                plot_labels = [
                    _enriched_label(g, id2ent, int(feat_ids_global[int(j)])) for j in plot_order
                ]
                side_txt = _pair_side_caption(g, id2ent, src_id, dst_id, pi)
                _plot_top_entities_signed(
                    phi=phi[plot_order], labels=plot_labels,
                    title=f"KernelSHAP entity importance (pair {pi}, {task}, {shap_baseline})",
                    out_path=os.path.join(shap_dir, f"pair{pi:05d}_{task}_{shap_baseline}.png"),
                    top_k=plot_n, pair_side_text=side_txt,
                )

                summary_rows.append({
                    "pair_idx": pi, "task": task, "method": "shap",
                    "src_id": src_id, "dst_id": dst_id, "csv": csv_name,
                    "shap_baseline": shap_baseline,
                    **pred_meta,
                    "K": K, "nsamples": nsamples_resolved,
                    "sub_nodes": N_sub, "sub_edges": E_sub, "elapsed_s": round(elapsed, 2),
                })

    write_summary_json(
        os.path.join(args.out, "summary_shap.json"),
        args,
        n_test,
        pair_indices,
        tasks,
        extra_fields={
            "shap_subgraph_hop": args.shap_subgraph_hop,
            "shap_candidates": args.shap_candidates,
            "shap_nsamples": str(args.shap_nsamples),
            "shap_l1_reg": args.shap_l1_reg,
            "shap_baseline": args.shap_baseline,
            "shap_exclude_dst": bool(args.shap_exclude_dst),
            "src_dst_excluded_from_candidates": True,
            "candidate_selection": "structural",
        },
        rows=summary_rows,
    )

    print(f"\nDone. Results -> {args.out}")
    print(f"Summary -> {os.path.join(args.out, 'summary_shap.json')}")


if __name__ == "__main__":
    main()
