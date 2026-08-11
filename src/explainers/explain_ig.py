"""
Integrated Gradients explainer for the R-GCN event-pair model.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm

from src.explainers._shared import (
    ExplainerContext,
    _enriched_label,
    _entity_category,
    _entity_short_label,
    _k_hop_subgraph,
    _pair_side_caption,
    _plot_global_category,
    _plot_top_entities,
    _plot_top_entities_signed,
    _plot_top_entities_signed_noreorder,
    _scalar_for_task,
    add_common_explainer_args,
    load_explainer_context,
    setup_matplotlib_style,
    write_summary_json,
)
from src.train import get_event_attrs

setup_matplotlib_style()


def _parse_ig_baselines(raw: str) -> List[str]:
    baselines = [part.strip() for part in raw.split(",") if part.strip()]
    if not baselines:
        baselines = ["zero"]
    allowed = {"zero", "mean"}
    invalid = [b for b in baselines if b not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            "--ig-baseline must contain only 'zero' and/or 'mean' "
            f"(got: {', '.join(invalid)})"
        )
    seen = set()
    deduped: List[str] = []
    for baseline in baselines:
        if baseline not in seen:
            deduped.append(baseline)
            seen.add(baseline)
    return deduped


def _resolve_fixed_model_target(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    x: torch.Tensor,
    ei: torch.Tensor,
    et: torch.Tensor,
    src_id: int,
    task: str,
    id2act: dict,
) -> dict:
    with torch.no_grad():
        z = encoder(x, ei, et)
        out = head(z[src_id : src_id + 1])
    if task == "activity":
        logits = out["act"][0]
        pred_id = int(logits.argmax().item())
        return {
            "target_kind": "activity",
            "pred_class_id": pred_id,
            "pred_class_label": id2act.get(pred_id, str(pred_id)),
            "pred_score": float(logits[pred_id].item()),
        }
    if task == "time":
        v = out["time"][0]
        scalar = v if v.numel() == 1 else v.sum()
        return {"target_kind": "time", "pred_value": float(scalar.item())}
    if task.startswith("otherC_"):
        k = task[len("otherC_"):]
        if k not in out["otherC"]:
            return {}
        logits = out["otherC"][k][0]
        pred_id = int(logits.argmax().item())
        return {
            "target_kind": "otherC",
            "target_key": k,
            "pred_class_id": pred_id,
            "pred_class_label": str(pred_id),
            "pred_score": float(logits[pred_id].item()),
        }
    if task.startswith("otherN_"):
        k = task[len("otherN_"):]
        if k not in out["otherN"]:
            return {}
        v = out["otherN"][k][0]
        scalar = v if v.numel() == 1 else v.sum()
        return {"target_kind": "otherN", "target_key": k, "pred_value": float(scalar.item())}
    return {}


def _scalar_for_fixed_model_target(out: dict, task: str, target_meta: dict) -> torch.Tensor:
    if not target_meta:
        raise ValueError(f"Missing fixed target metadata for task: {task}")
    if task == "activity" and target_meta.get("target_kind") == "activity":
        return out["act"][0, int(target_meta["pred_class_id"])]
    if task.startswith("otherC_") and target_meta.get("target_kind") == "otherC":
        k = str(target_meta["target_key"])
        if k not in out["otherC"]:
            raise KeyError(f"otherC head missing key (sanitized): {k}")
        return out["otherC"][k][0, int(target_meta["pred_class_id"])]
    if task == "time":
        v = out["time"][0]
        return v if v.numel() == 1 else v.sum()
    if task.startswith("otherN_"):
        k = task[len("otherN_"):]
        if k not in out["otherN"]:
            raise KeyError(f"otherN head missing key: {k}")
        v = out["otherN"][k][0]
        return v if v.numel() == 1 else v.sum()
    raise ValueError(f"Fixed target metadata does not match task: {task}")


def _public_target_metadata(target_meta: dict) -> dict:
    return {
        k: v
        for k, v in target_meta.items()
        if k in {"pred_class_id", "pred_class_label", "pred_score", "pred_value"}
    }


def _fixed_target_score(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    x: torch.Tensor,
    ei: torch.Tensor,
    et: torch.Tensor,
    src_id: int,
    task: str,
    target_meta: dict,
) -> float:
    with torch.no_grad():
        z = encoder(x, ei, et)
        out = head(z[src_id : src_id + 1])
        scalar = _scalar_for_fixed_model_target(out, task, target_meta)
    return float(scalar.item())


# ---------------------------------------------------------------------------
# Captum-compatible forward wrapper
# ---------------------------------------------------------------------------

def _make_forward_fn(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    ei: torch.Tensor,
    et: torch.Tensor,
    src_id: int,
    task: str,
    y_act: Optional[torch.Tensor],
    y_time: Optional[torch.Tensor],
    pair_val_idx: int,
    device: torch.device,
    use_model_target: bool = False,
    fixed_target: Optional[dict] = None,
):
    def forward_fn(x_batched: torch.Tensor) -> torch.Tensor:
        # x_batched: [1, N, in_dim]  (Captum adds batch dim of 1)
        x_in = x_batched.squeeze(0)           # [N, in_dim]
        z = encoder(x_in, ei, et)
        out = head(z[src_id : src_id + 1])
        scalar = (
            _scalar_for_fixed_model_target(out, task, fixed_target or {})
            if use_model_target
            else _scalar_for_task(out, task, y_act, y_time, pair_val_idx, device)
        )
        return scalar.unsqueeze(0)             # [1]

    return forward_fn


# Core IG attribution function
# ---------------------------------------------------------------------------

def _integrated_gradients(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    x: torch.Tensor,
    ei: torch.Tensor,
    et: torch.Tensor,
    src_id: int,
    task: str,
    y_act: Optional[torch.Tensor],
    y_time: Optional[torch.Tensor],
    pair_val_idx: int,
    device: torch.device,
    baseline: torch.Tensor,
    n_steps: int,
    method: str,
    use_model_target: bool = False,
    fixed_target: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    encoder.eval()
    head.eval()

    forward_fn = _make_forward_fn(
        encoder, head, ei, et, src_id,
        task, y_act, y_time, pair_val_idx, device, use_model_target,
        fixed_target=fixed_target,
    )

    ig = IntegratedGradients(forward_fn)

    x_in = x.unsqueeze(0)          # [1, N, in_dim]
    b_in = baseline.unsqueeze(0)   # [1, N, in_dim]

    with torch.enable_grad():
        attributions, delta_t = ig.attribute(
            inputs=x_in,
            baselines=b_in,
            method=method,
            n_steps=n_steps,
            internal_batch_size=1,
            return_convergence_delta=True,
        )

    # attributions: [1, N, in_dim]  →  squeeze to [N, in_dim]
    attr_np = attributions.squeeze(0).detach().cpu().float().numpy()

    ig_signed = attr_np.sum(axis=1).astype(np.float64)    # signed node score
    ig_abs    = np.abs(attr_np).sum(axis=1).astype(np.float64)  # magnitude

    delta_val = float(delta_t.item()) if delta_t.numel() == 1 else float(delta_t[0].item())

    # Clean up accumulated gradients on model params (not used, avoids mem build-up)
    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)

    return ig_signed, ig_abs, delta_val


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Integrated Gradients entity-level explainer for the R-GCN event-pair model."
    )
    add_common_explainer_args(ap)
    ap.add_argument(
        "--top-k", type=int, default=30,
        help="Top entities per pair in CSV / plot (default 30).",
    )
    ap.add_argument(
        "--ig-baseline", default="mean", dest="ig_baseline",
        type=_parse_ig_baselines,
        help="Reference baseline for the IG path: "
             "'zero' = all-zero embedding, "
             "'mean' = subgraph mean RDF2Vec embedding (default), "
             "or comma-separated 'zero,mean'.",
    )
    ap.add_argument(
        "--ig-n-steps", type=int, default=32, dest="ig_n_steps",
        help="Number of integral approximation steps (default 32). "
             "Increase to 100-200 if convergence delta is large.",
    )
    ap.add_argument(
        "--ig-method", default="gausslegendre", dest="ig_method",
        choices=[
            "gausslegendre", "riemann_trapezoid",
            "riemann_left", "riemann_right", "riemann_middle",
        ],
        help="Integral approximation method (default 'gausslegendre'). "
             "Gauss-Legendre converges fastest for smooth functions.",
    )
    ap.add_argument(
        "--ig-subgraph-hop", type=int, default=3, dest="ig_subgraph_hop",
        help="k-hop subgraph depth around the pair before running IG. "
             "3 = exact receptive field for a 3-layer R-GCN (default). "
             "0 = full graph (slow on large datasets).",
    )
    ap.add_argument(
        "--ig-exclude-dst", action="store_true", dest="ig_exclude_dst",
        help="Exclude the destination event node from the ranked output.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx: ExplainerContext = load_explainer_context(args, device)
    encoder, head    = ctx.encoder, ctx.head
    x, g, id2ent     = ctx.x, ctx.g, ctx.id2ent
    ei_val, et_val   = ctx.ei_val, ctx.et_val
    val_pairs        = ctx.val_pairs
    y_act_val, y_time_val = ctx.y_act_val, ctx.y_time_val
    act_vocab, id2act = ctx.act_vocab, ctx.id2act
    tasks, n_val, pair_indices = ctx.tasks, ctx.n_val, ctx.pair_indices

    os.makedirs(args.out, exist_ok=True)
    ig_dir   = os.path.join(args.out, "integrated_gradients")
    pair_dir = os.path.join(args.out, "per_pair")
    for d in (ig_dir, pair_dir):
        os.makedirs(d, exist_ok=True)

    summary_rows:    List[dict] = []
    global_cat_rows: List[dict] = []   # for per-category aggregation at the end

    for pi in tqdm(pair_indices, desc="pairs"):
        src_id, dst_id = val_pairs[pi]

        # ------------------------------------------------------------------
        # k-hop subgraph extraction
        # ------------------------------------------------------------------
        if args.ig_subgraph_hop > 0:
            nodes_t, ei_sub, et_sub, g2l = _k_hop_subgraph(
                seeds=[src_id, dst_id],
                edge_index=ei_val,
                edge_type=et_val,
                num_nodes=x.size(0),
                k=args.ig_subgraph_hop,
            )
            x_sub      = x[nodes_t.to(device)].detach()
            ei_sub     = ei_sub.to(device)
            et_sub     = et_sub.to(device)
            src_local  = g2l[src_id]
            dst_local  = g2l.get(dst_id, -1)

        else:
            nodes_t    = torch.arange(x.size(0))
            x_sub      = x.detach()
            ei_sub     = ei_val
            et_sub     = et_val
            src_local  = src_id
            dst_local  = dst_id
            g2l        = None

        N_sub      = x_sub.size(0)
        nodes_list = nodes_t.tolist()   # local index → global node id

        # Per-task, per-baseline IG loop
        for task in tasks:
            target_meta = _resolve_fixed_model_target(
                encoder, head, x_sub, ei_sub, et_sub, src_local, task, id2act
            )
            pred_meta = _public_target_metadata(target_meta)
            for ig_baseline in args.ig_baseline:
                baseline_sub = (
                    x_sub.mean(dim=0).unsqueeze(0).expand_as(x_sub).clone()
                    if ig_baseline == "mean"
                    else torch.zeros_like(x_sub)
                )
                t0 = time.time()

                try:
                    ig_signed, ig_abs, delta = _integrated_gradients(
                        encoder=encoder,
                        head=head,
                        x=x_sub,
                        ei=ei_sub,
                        et=et_sub,
                        src_id=src_local,
                        task=task,
                        y_act=y_act_val,
                        y_time=y_time_val,
                        pair_val_idx=pi,
                        device=device,
                        baseline=baseline_sub,
                        n_steps=args.ig_n_steps,
                        method=args.ig_method,
                        use_model_target=True,
                        fixed_target=target_meta,
                    )
                except Exception as exc:
                    print(f"[warn] IG failed pair={pi} task={task} baseline={ig_baseline}: {exc}")
                    continue

                elapsed = time.time() - t0

                f_x = _fixed_target_score(
                    encoder, head, x_sub, ei_sub, et_sub, src_local, task, target_meta
                )
                f_baseline = _fixed_target_score(
                    encoder, head, baseline_sub, ei_sub, et_sub, src_local, task, target_meta
                )
                f_x_minus_f_baseline = f_x - f_baseline

                # Completeness check: sum_attributions ≈ F(x) - F(baseline)
                sum_attributions = float(ig_signed.sum())
                if abs(sum_attributions) > 1e-8:
                    delta_frac = abs(delta) / abs(sum_attributions)
                    if delta_frac > 0.05:
                        print(
                            f"  [warn] pair={pi} task={task} baseline={ig_baseline}: "
                            f"convergence delta={delta:.4f} "
                            f"({delta_frac * 100:.1f}% of total attribution). "
                            f"Consider increasing --ig-n-steps."
                        )

                # candidate ranking
                exclude_local: set = {src_local}
                if args.ig_exclude_dst:
                    if dst_local >= 0:
                        exclude_local.add(dst_local)

                candidate_local = [i for i in range(N_sub) if i not in exclude_local]
                top_local = sorted(candidate_local, key=lambda i: -ig_abs[i])[: args.top_k]

                # ------------------------------------------------------------------
                # Activity label for annotation
                act_name, _, _, _ = get_event_attrs(g, id2ent.get(dst_id, ""))
                act_label = act_name or ""
                if task == "activity" and act_vocab and act_name and act_name in act_vocab:
                    act_label = f"{act_name} (id={act_vocab[act_name]})"

                # Build CSV rows
                rows: List[dict] = []
                for rank, loc_i in enumerate(top_local):
                    glob_i = int(nodes_list[loc_i])
                    uri    = id2ent.get(glob_i, "?")
                    row: dict = {
                        "rank":               rank + 1,
                        "entity_id":          glob_i,
                        "uri":                uri,
                        "label":              _entity_short_label(g, uri),
                        "ig_baseline":        ig_baseline,
                        "ig_attribution":     float(ig_signed[loc_i]),
                        "abs_ig_attribution": float(ig_abs[loc_i]),
                        "convergence_delta":  float(delta),
                        "F_x":                 f_x,
                        "F_baseline":          f_baseline,
                        "F_x_minus_F_baseline": f_x_minus_f_baseline,
                        "sum_attributions":    sum_attributions,
                        "dst_activity":       act_label,
                    }
                    if task == "activity" and y_act_val is not None:
                        ya = int(y_act_val[pi].item())
                        row["y_act_id"]   = ya
                        row["y_act_name"] = id2act.get(ya, "")
                    row.update(pred_meta)
                    rows.append(row)

                    global_cat_rows.append({
                        "task":        task,
                        "ig_baseline": ig_baseline,
                        "category":    _entity_category(uri),
                        "abs_score":   float(ig_abs[loc_i]),
                    })

                csv_name = f"pair{pi:05d}_{task}_{ig_baseline}.csv"
                pd.DataFrame(rows).to_csv(os.path.join(pair_dir, csv_name), index=False)

                plot_n     = min(20, len(top_local))
                plot_local = top_local[:plot_n]
                plot_labels = [
                    _enriched_label(g, id2ent, int(nodes_list[loc_i]))
                    for loc_i in plot_local
                ]
                side_txt = _pair_side_caption(g, id2ent, src_id, dst_id, pi)
                plot_abs = np.array([float(ig_abs[loc_i]) for loc_i in plot_local])
                plot_signed = np.array([float(ig_signed[loc_i]) for loc_i in plot_local])
                stem = f"pair{pi:05d}_{task}_{ig_baseline}"
                _plot_top_entities(
                    plot_abs,
                    plot_labels,
                    f"Integrated Gradients (|IG|) - pair {pi}, {task}, {ig_baseline}",
                    os.path.join(ig_dir, f"{stem}.png"),
                    top_k=plot_n,
                    xlabel="|IG attribution|",
                    pair_side_text=side_txt,
                )
                _plot_top_entities_signed(
                    plot_signed,
                    plot_labels,
                    f"Signed Integrated Gradients - pair {pi}, {task}, {ig_baseline}",
                    os.path.join(ig_dir, f"{stem}_signed.png"),
                    top_k=plot_n,
                    xlabel="IG attribution (red = supports prediction, blue = opposes prediction)",
                    pair_side_text=side_txt,
                )
                _plot_top_entities_signed_noreorder(
                    plot_signed,
                    plot_labels,
                    f"Integrated Gradients (signed, ranked by |IG|) - pair {pi}, {task}, {ig_baseline}",
                    os.path.join(ig_dir, f"{stem}_signed_noreorder.png"),
                    top_k=plot_n,
                    pair_side_text=side_txt,
                )

                print(
                    f"  [pair{pi:05d}/{task}/{ig_baseline}]  nodes={N_sub}  "
                    f"delta={delta:.4f}  {elapsed:.1f}s"
                )

                summary_rows.append({
                    "pair_idx":          pi,
                    "task":              task,
                    "method":            "integrated_gradients",
                    "ig_baseline":       ig_baseline,
                    "src_id":            src_id,
                    "dst_id":            dst_id,
                    "csv":               csv_name,
                    "sub_nodes":         N_sub,
                    "n_steps":           args.ig_n_steps,
                    "convergence_delta": float(delta),
                    "total_attribution": float(sum_attributions),
                    "sum_attributions":  float(sum_attributions),
                    "F_x":               f_x,
                    "F_baseline":        f_baseline,
                    "F_x_minus_F_baseline": f_x_minus_f_baseline,
                    "elapsed_s":         round(elapsed, 2),
                } | pred_meta)


    if global_cat_rows:
        global_dir = os.path.join(args.out, "global")
        os.makedirs(global_dir, exist_ok=True)
        cat_df_all = pd.DataFrame(global_cat_rows)

        for task in tasks:
            task_df = cat_df_all[cat_df_all["task"] == task]
            if task_df.empty:
                continue
            for ig_baseline in args.ig_baseline:
                base_df = task_df[task_df["ig_baseline"] == ig_baseline]
                if base_df.empty:
                    continue
                cat_agg = (
                    base_df.groupby("category")["abs_score"]
                    .agg(mean_score="mean", count="count")
                    .reset_index()
                )
                cat_csv = os.path.join(global_dir, f"category_ig_{task}_{ig_baseline}.csv")
                cat_agg.to_csv(cat_csv, index=False)
                _plot_global_category(
                    cat_df=cat_agg,
                    task=task,
                    out_path=os.path.join(global_dir, f"category_ig_{task}_{ig_baseline}.png"),
                    value_col="mean_score",
                    xlabel="Mean |IG attribution|",
                    title_template=f"Global entity-category IG attribution ({ig_baseline}) — {{task_lbl}}",
                )
                print(f"[OK] Global category plot -> {cat_csv.replace('.csv', '.png')}")


    write_summary_json(
        os.path.join(args.out, "summary_ig.json"),
        args,
        n_val,
        pair_indices,
        tasks,
        extra_fields={
            "ig_baseline":      args.ig_baseline,
            "ig_n_steps":       args.ig_n_steps,
            "ig_method":        args.ig_method,
            "ig_subgraph_hop":  args.ig_subgraph_hop,
            "ig_exclude_dst":   bool(args.ig_exclude_dst),
        },
        rows=summary_rows,
    )

    print(f"\nDone. Results -> {args.out}")
    print(f"Summary -> {os.path.join(args.out, 'summary_ig.json')}")


if __name__ == "__main__":
    main()
