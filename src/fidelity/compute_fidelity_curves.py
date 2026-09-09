from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainers.explainer_adapters import (
    BaselineAdapter,
    FOXAdapter,
    GNNExplainerAdapter,
    GradientAdapter,
    IGAdapter,
    LIMEAdapter,
    PGMExplainerAdapter,
    ProphetAdapter,
    ShapAdapter,
)
from src.explainers.explanation import ExplanationResult
from src.explainers._shared import add_common_explainer_args, load_explainer_context, _k_hop_subgraph
from src.fidelity.config import load_config_defaults
from src.fidelity.experiment_record import write_manifest, write_selected_pairs
from src.fidelity.fidelity_utils import (
    build_proxy,
    compute_original_prediction,
    compute_fidelity_details,
    exclude_protected_scores,
    node_scores_from_mask,
    selected_node_edge_counts,
    is_classification_task,
    task_target_value_nodes,
    threshold_node_scores_by_count,
    warn_if_degenerate_explanation,
)
from src.fidelity.reproducibility import set_deterministic_seed

DEFAULT_TOP_K_VALUES = "5,10,15,20,25"
HETEROGENEITY_MODES = ("full", "middle", "homogeneous")

ALL_EXPLAINERS = [
    "gradient", "ig", "lime", "shap", "random", "degree",
    "fox", "prophet", "gnnexplainer", "pgmexplainer",
]
DEFAULT_EXPLAINERS = [
    "gradient", "ig", "lime", "shap", "gnnexplainer", "pgmexplainer",
]
CHECKPOINT_FIELDS = [
    "explainer", "pair_idx", "task", "top_k",
    "fid_prob_plus", "fid_prob_minus", "fid_acc_plus", "fid_acc_minus",
    "p_orig", "p_complement", "p_explanation", "target_class",
    "num_nodes_sub", "num_edges_sub", "num_selected_nodes", "num_selected_edges",
    "selected_dst", "selected_target_value_nodes", "removed_target_value_nodes_fid_plus",
    "mask_type",
]
DEFAULTS = {
    "ttl": "data/raw/BPIC13_O/BPIC13_OpenProblems.ttl",
    "emb": "data/raw/BPIC13_O/entity_embeddings.npy",
    "entity2id": "data/raw/BPIC13_O/entity2id.json",
    "case_split": "data/raw/BPIC13_O/case_split.json",
    "model": "data/processed/BPIC13_O/best_val_model.pt",
    "vocabs": "data/processed/BPIC13_O/best_val_vocabs.json",
    "best_test": "data/processed/BPIC13_O/best_val.pt",
    "out": "data/processed/BPIC13_O/fidelity_curves",
    "tasks": "activity,resource,role,lifecycle",
    "num_samples": 20,
    "fidelity_metrics": "prob",
    "heterogeneity_mode": "full",
}
ADAPTER_OPTIONS = [
    ("--gnn-epochs", int, 100), ("--gnn-lr", float, 0.01),
    ("--gnn-edge-size-reg", float, 0.005),
    ("--gnn-edge-ent-reg", float, 1.0),
    ("--gnn-node-size-reg", float, 1.0),
    ("--gnn-node-ent-reg", float, 0.1),
    ("--prophet-epochs", int, 100), ("--prophet-lr", float, 0.01),
    ("--prophet-edge-size-reg", float, 0.005),
    ("--prophet-edge-ent-reg", float, 1.0),
    ("--prophet-node-size-reg", float, 1.0),
    ("--prophet-node-ent-reg", float, 0.1),
    ("--pgm-num-samples", int, 1000),
    ("--pgm-perturb-probability", float, 0.5),
    ("--pgm-response-top-fraction", float, 0.125),
    ("--pgm-significance-threshold", float, 0.05),
    ("--fox-m", int, 5), ("--fox-perturb", int, 200),
    ("--shap-candidates", int, 25), ("--shap-nsamples", int, 300),
    ("--lime-perturb", int, 200), ("--ig-steps", int, 50),
]

CompletedKey = Tuple[str, int, str, int]


def _build_arg_parser(config_defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    defaults = dict(DEFAULTS)
    defaults.update(config_defaults or {})
    # accept configs from before the val->test rename
    if "best_val" in defaults:
        defaults.setdefault("best_test", defaults.pop("best_val"))
    parser = argparse.ArgumentParser(
        description="Compute PyG/GraphFramEx model-fidelity curves for RGCN explainers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    add_common_explainer_args(parser)

    for action in parser._actions:
        if action.dest in defaults:
            action.default = defaults[action.dest]
            action.required = False

    parser.add_argument("--explainers", default=defaults.get("explainers", ",".join(DEFAULT_EXPLAINERS)),
                        help=f"Comma-separated explainers. Available: {', '.join(ALL_EXPLAINERS)}")
    parser.add_argument("--subgraph-hops", type=int, default=defaults.get("subgraph_hops", 3),
                        help="k-hop subgraph depth; 3 matches the trained 3-layer RGCN.")
    parser.add_argument("--top-k-values", default=defaults.get("top_k_values", DEFAULT_TOP_K_VALUES),
                        help="Comma-separated fixed node counts for fidelity curves.")
    parser.add_argument(
        "--fidelity-metrics",
        default=defaults.get("fidelity_metrics", "prob"),
        help="Comma-separated fidelity metrics: prob, acc, or acc,prob.",
    )
    parser.add_argument(
        "--heterogeneity-mode",
        choices=HETEROGENEITY_MODES,
        default=defaults.get("heterogeneity_mode", "full"),
        help="Filter the extracted subgraph before explanations/fidelity are computed.",
    )
    adapters = parser.add_argument_group("adapter options")
    for flag, typ, default in ADAPTER_OPTIONS:
        adapters.add_argument(flag, type=typ, default=defaults.get(flag.lstrip("-").replace("-", "_"), default))
    return parser


def _parse_csv_or_sequence(raw: Any) -> List[Any]:
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, Sequence):
        return list(raw)
    raise TypeError(f"Expected comma-separated string or sequence, got {type(raw).__name__}")


def _parse_top_k_values(raw: Any) -> List[int]:
    values: List[int] = []
    for item in _parse_csv_or_sequence(raw):
        value = int(item)
        if value <= 0:
            raise ValueError(f"top_k must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("--top-k-values must contain at least one value")
    return sorted(dict.fromkeys(values))


def _parse_fidelity_metrics(raw: Any) -> List[str]:
    metrics = [str(item).lower() for item in _parse_csv_or_sequence(raw)]
    invalid = [metric for metric in metrics if metric not in {"prob", "acc"}]
    if invalid or not metrics:
        raise ValueError(f"--fidelity-metrics must contain prob and/or acc; got {invalid or metrics}")
    return list(dict.fromkeys(metrics))


def _completed_key(
    explainer: str,
    pair_idx: int,
    task: str,
    top_k: int,
) -> CompletedKey:
    return (explainer, int(pair_idx), task, int(top_k))


def _build_adapters(args: argparse.Namespace, global_mean: Tensor) -> Dict[str, Any]:
    names = [str(name).strip().lower() for name in _parse_csv_or_sequence(args.explainers)]
    invalid = [name for name in names if name not in ALL_EXPLAINERS]
    if invalid:
        raise ValueError(f"Unknown explainers: {invalid}. Available: {ALL_EXPLAINERS}")

    factories = {
        "gradient": lambda: GradientAdapter(),
        "ig": lambda: IGAdapter(n_steps=args.ig_steps),
        "fox": lambda: FOXAdapter(fox_m=args.fox_m, n_perturb=args.fox_perturb, seed=args.seed),
        "lime": lambda: LIMEAdapter(
            n_perturb=args.lime_perturb, seed=args.seed, global_mean=global_mean
        ),
        "shap": lambda: ShapAdapter(
            shap_candidates=args.shap_candidates,
            shap_nsamples=args.shap_nsamples,
            seed=args.seed,
            global_mean=global_mean,
        ),
        "random": lambda: BaselineAdapter("random", seed=args.seed),
        "degree": lambda: BaselineAdapter("degree"),
        "prophet": lambda: ProphetAdapter(
            epochs=args.prophet_epochs,
            lr=args.prophet_lr,
            edge_size_reg=args.prophet_edge_size_reg,
            edge_ent_reg=args.prophet_edge_ent_reg,
            node_size_reg=args.prophet_node_size_reg,
            node_ent_reg=args.prophet_node_ent_reg,
            seed=args.seed,
        ),
        "gnnexplainer": lambda: GNNExplainerAdapter(
            epochs=args.gnn_epochs,
            lr=args.gnn_lr,
            edge_size=args.gnn_edge_size_reg,
            edge_ent=args.gnn_edge_ent_reg,
            node_size=args.gnn_node_size_reg,
            node_ent=args.gnn_node_ent_reg,
            seed=args.seed,
        ),
        "pgmexplainer": lambda: PGMExplainerAdapter(
            global_mean=global_mean,
            num_samples=args.pgm_num_samples,
            perturb_probability=args.pgm_perturb_probability,
            response_top_fraction=args.pgm_response_top_fraction,
            significance_threshold=args.pgm_significance_threshold,
            seed=args.seed,
        ),
    }
    return {name: factories[name]() for name in names}


def _load_completed(
    checkpoint_path: Path,
    allowed_explainers: Optional[Set[str]] = None,
) -> Set[CompletedKey]:
    completed: Set[CompletedKey] = set()
    if not checkpoint_path.exists():
        return completed
    with checkpoint_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CHECKPOINT_FIELDS:
            raise ValueError(
                f"Checkpoint schema mismatch at {checkpoint_path}. "
                "Use a fresh --out directory before rerunning."
            )
        for row in reader:
            explainer = str(row.get("explainer", ""))
            if allowed_explainers is not None and explainer not in allowed_explainers:
                raise ValueError(
                    f"Checkpoint contains explainer {explainer!r}, which is not in "
                    f"this run: {sorted(allowed_explainers)}. Use a fresh --out directory."
                )
            try:
                completed.add(
                    _completed_key(
                        explainer,
                        int(row["pair_idx"]),
                        str(row["task"]),
                        int(row["top_k"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                warnings.warn(f"Skipping malformed checkpoint row: {row}")
    return completed


def _append_checkpoint_row(checkpoint_path: Path, row: Dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    exists = checkpoint_path.exists() and checkpoint_path.stat().st_size > 0
    if exists:
        with checkpoint_path.open(newline="", encoding="utf-8") as existing:
            reader = csv.reader(existing)
            header = next(reader, [])
        if header != CHECKPOINT_FIELDS:
            raise ValueError(
                f"Checkpoint schema mismatch at {checkpoint_path}. "
                "Use a fresh --out directory or remove the old checkpoint before rerunning."
            )
    with checkpoint_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHECKPOINT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CHECKPOINT_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def _classification_tasks(tasks: Iterable[str]) -> List[str]:
    requested = list(tasks)
    cls_tasks = [task for task in requested if is_classification_task(task)]
    skipped = [task for task in requested if task not in cls_tasks]
    for task in skipped:
        warnings.warn(
            f"Skipping non-classification fidelity task {task!r}; "
            "this runner currently supports activity and otherC_* tasks."
        )
    if not cls_tasks:
        raise ValueError(
            "This fidelity experiment requires at least one classification task "
            "(activity or otherC_*)."
        )
    return cls_tasks


def _filter_subgraph_by_heterogeneity(
    mode: str,
    nodes_t: Tensor,
    ei_sub: Tensor,
    et_sub: Tensor,
    g2l: Dict[int, int],
    rel2id: Dict[str, int],
    num_rel: int,
    required_global_nodes: Sequence[int],
) -> Tuple[Tensor, Tensor, Tensor, Dict[int, int]]:
    if mode == "full":
        return nodes_t, ei_sub, et_sub, g2l
    if mode not in HETEROGENEITY_MODES:
        raise ValueError(f"Unknown heterogeneity mode: {mode}")

    def keep_relation(rel_uri: str) -> bool:
        rel = rel_uri.lower()
        is_activity = "hasevent_activity_concept_name" in rel
        is_directly_follows = "directlyfollows" in rel
        is_timestamp = "hasevent_timestamp_time_timestamp" in rel
        is_caseid = "hascaseID_case_concept_name" in rel
        
        if mode == "homogeneous":
            return is_directly_follows or is_activity or is_timestamp or is_caseid
        
        is_resource = "org_resource" in rel
        return is_directly_follows or is_activity or is_resource or is_timestamp or is_caseid

    id2rel = {int(idx): str(uri) for uri, idx in rel2id.items()}
    keep_edge_flags = [
        keep_relation(id2rel.get(int(edge_type) % num_rel, ""))
        for edge_type in et_sub.detach().cpu().tolist()
    ]
    keep_edges = torch.tensor(keep_edge_flags, dtype=torch.bool, device=ei_sub.device)
    filtered_ei = ei_sub[:, keep_edges]
    filtered_et = et_sub[keep_edges]

    kept_local_nodes = set()
    if filtered_ei.numel() > 0:
        kept_local_nodes.update(int(node) for node in filtered_ei.detach().cpu().reshape(-1).tolist())
    for global_node in required_global_nodes:
        local_node = g2l.get(int(global_node))
        if local_node is not None:
            kept_local_nodes.add(int(local_node))

    if not kept_local_nodes:
        empty_nodes = nodes_t.new_empty((0,), dtype=nodes_t.dtype)
        empty_ei = ei_sub.new_empty((2, 0), dtype=ei_sub.dtype)
        empty_et = et_sub.new_empty((0,), dtype=et_sub.dtype)
        return empty_nodes, empty_ei, empty_et, {}

    kept_local_sorted = sorted(kept_local_nodes)
    kept_local_t = torch.tensor(kept_local_sorted, dtype=torch.long, device=nodes_t.device)
    filtered_nodes = nodes_t[kept_local_t]
    old_to_new = {old_local: new_local for new_local, old_local in enumerate(kept_local_sorted)}

    if filtered_ei.numel() > 0:
        remapped_rows = [
            [old_to_new[int(node)] for node in filtered_ei[row].detach().cpu().tolist()]
            for row in range(2)
        ]
        filtered_ei = torch.tensor(remapped_rows, dtype=ei_sub.dtype, device=ei_sub.device)
    else:
        filtered_ei = ei_sub.new_empty((2, 0), dtype=ei_sub.dtype)

    filtered_g2l = {
        int(global_id): old_to_new[int(local_id)]
        for global_id, local_id in g2l.items()
        if int(local_id) in old_to_new
    }
    return filtered_nodes, filtered_ei, filtered_et, filtered_g2l


def _relation_hash(rel2id: Dict[str, int]) -> str:
    payload = "\n".join(f"{k}\t{v}" for k, v in sorted(rel2id.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _selection_diagnostics(
    binary_mask: Tensor,
    dst_local: int,
    target_nodes: Sequence[int],
    protected_nodes: Sequence[int],
) -> Dict[str, int]:
    selected = binary_mask.detach().reshape(-1).to(torch.bool).cpu()
    protected = {int(node) for node in protected_nodes}

    selected_dst = int(0 <= int(dst_local) < selected.numel() and bool(selected[int(dst_local)].item()))
    selected_target_nodes = 0
    removed_target_nodes = 0
    for node in target_nodes:
        node_i = int(node)
        if 0 <= node_i < selected.numel() and bool(selected[node_i].item()):
            selected_target_nodes += 1
            if node_i not in protected:
                removed_target_nodes += 1

    return {
        "selected_dst": selected_dst,
        "selected_target_value_nodes": selected_target_nodes,
        "removed_target_value_nodes_fid_plus": removed_target_nodes,
    }


def _write_aggregates(
    checkpoint_path: Path,
    summary_path: Path,
    auc_path: Path,
    top_k_values: Sequence[int],
    metrics: Sequence[str],
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}")

    df = pd.read_csv(checkpoint_path)
    df["top_k"] = df["top_k"].astype(int)

    aggregations: Dict[str, Any] = {"n_samples": ("pair_idx", "nunique")}
    for metric in metrics:
        aggregations[f"mean_fid_{metric}_plus"] = (f"fid_{metric}_plus", "mean")
        aggregations[f"mean_fid_{metric}_minus"] = (f"fid_{metric}_minus", "mean")
    summary = (
        df.groupby(["explainer", "task", "top_k"], as_index=False)
        .agg(**aggregations)
        .sort_values(["explainer", "task", "top_k"])
    )
    for metric in metrics:
        summary[f"mean_one_minus_fid_{metric}_minus"] = (
            1.0 - summary[f"mean_fid_{metric}_minus"]
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    auc_rows: List[Dict[str, Any]] = []
    x_expected = np.array(top_k_values, dtype=float)
    auc_fn = getattr(np, "trapezoid", np.trapz)
    for (explainer, task), group in summary.groupby(["explainer", "task"]):
        group = group.sort_values("top_k")
        x = group["top_k"].to_numpy(dtype=float)
        if len(x) != len(x_expected) or not np.allclose(x, x_expected):
            warnings.warn(
                f"AUC for {explainer}/{task} uses available top-k values only; "
                f"expected {x_expected.tolist()}, got {x.tolist()}"
            )
        span = float(x.max() - x.min()) if len(x) else 0.0
        row: Dict[str, Any] = {"explainer": explainer, "task": task}
        for metric in metrics:
            y_plus = group[f"mean_fid_{metric}_plus"].to_numpy(dtype=float)
            y_minus = group[f"mean_fid_{metric}_minus"].to_numpy(dtype=float)
            y_one_minus = 1.0 - y_minus
            auc_plus = float(y_plus.mean())
            auc_minus = float(y_minus.mean())
            auc_one_minus = float(y_one_minus.mean())
            norm_plus = auc_plus
            norm_minus = auc_minus
            norm_one_minus = auc_one_minus
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
        auc_rows.append(row)

    pd.DataFrame(auc_rows).sort_values(["task", "explainer"]).to_csv(auc_path, index=False)
    print(f"Summary -> {summary_path}")
    print(f"AUC     -> {auc_path}")


def main() -> None:
    mode_start = time.perf_counter()

    config_defaults, config_path = load_config_defaults()
    parser = _build_arg_parser(config_defaults)
    args = parser.parse_args()
    if config_path is not None:
        print(f"Loaded config -> {config_path}")
    print(f"Heterogeneity mode: {args.heterogeneity_mode}")

    top_k_values = _parse_top_k_values(args.top_k_values)
    fidelity_metrics = _parse_fidelity_metrics(args.fidelity_metrics)
    out_dir = Path(args.out)
    checkpoint_path = out_dir / "curve_checkpoint.csv"
    summary_path = out_dir / "fidelity_curves_summary.csv"
    auc_path = out_dir / "fidelity_curves_auc.csv"

    set_deterministic_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = load_explainer_context(args, device)
    encoder, head = ctx.encoder, ctx.head
    cls_tasks = _classification_tasks(ctx.tasks)
    global_mean = ctx.x.mean(dim=0).detach()
    adapters = _build_adapters(args, global_mean)
    context_payload = {
        "pair_indices": ctx.pair_indices,
        "task_aliases": ctx.task_aliases,
        "skipped_tasks": ctx.skipped_tasks,
        "relation_hash": _relation_hash(ctx.rel2id),
        "num_rel": ctx.num_rel,
        "mean_baseline_scope": "full_graph",
        "probability_fidelity_definition": "signed_probability_drop",
        "heterogeneity_mode": args.heterogeneity_mode,
    }

    completed = _load_completed(checkpoint_path, set(adapters))
    if completed:
        print(f"Loaded {len(completed)} completed rows from {checkpoint_path}")

    write_selected_pairs(out_dir / "selected_pairs.csv", ctx.pair_indices, ctx.test_pairs)
    write_manifest(
        out_dir / "experiment_manifest.json",
        args=args,
        config_path=config_path,
        config_defaults=config_defaults,
        top_k_values=top_k_values,
        cls_tasks=cls_tasks,
        context_payload=context_payload,
        workspace=ROOT.parent,
    )

    encoder.eval()
    head.eval()

    for pair_number, pair_idx in enumerate(ctx.pair_indices, start=1):
        src_id, dst_id = ctx.test_pairs[pair_idx]
        print(
            f"Pair {pair_number}/{len(ctx.pair_indices)} "
            f"[test_pairs[{pair_idx}]] src={src_id} dst={dst_id}"
        )

        try:
            nodes_t, ei_sub, et_sub, g2l = _k_hop_subgraph(
                seeds=[src_id, dst_id],
                edge_index=ctx.ei_test,
                edge_type=ctx.et_test,
                num_nodes=ctx.x.size(0),
                k=args.subgraph_hops,
            )
        except Exception as exc:
            warnings.warn(f"Subgraph extraction failed for pair={pair_idx}: {exc}")
            continue

        nodes_t, ei_sub, et_sub, g2l = _filter_subgraph_by_heterogeneity(
            args.heterogeneity_mode,
            nodes_t,
            ei_sub,
            et_sub,
            g2l,
            ctx.rel2id,
            ctx.num_rel,
            required_global_nodes=(src_id, dst_id),
        )
        if et_sub.numel() == 0:
            suffix = (
                ""
                if args.heterogeneity_mode == "full"
                else f" after {args.heterogeneity_mode} heterogeneity filter"
            )
            warnings.warn(f"Empty subgraph{suffix} for pair={pair_idx}; skipping pair")
            continue
        if src_id not in g2l:
            warnings.warn(
                f"src_id={src_id} not found in {args.heterogeneity_mode} "
                f"subgraph for pair={pair_idx}"
            )
            continue
        if dst_id not in g2l:
            warnings.warn(
                f"dst_id={dst_id} not found in {args.heterogeneity_mode} "
                f"subgraph for pair={pair_idx}"
            )
            continue

        x_sub = ctx.x[nodes_t.to(device)].detach()
        ei_sub = ei_sub.to(device)
        et_sub = et_sub.to(device)
        src_local = g2l[src_id]
        dst_local = g2l.get(dst_id, -1)
        base_protected_nodes = tuple(
            node for node in (src_local, dst_local) if 0 <= int(node) < x_sub.size(0)
        )

        for task in cls_tasks:
            target_nodes = task_target_value_nodes(
                ctx.g, ctx.id2ent, nodes_t, g2l, dst_id, task
            )
            protected_nodes = tuple(
                sorted({int(node) for node in (*base_protected_nodes, *target_nodes)})
            )
            proxy = build_proxy(encoder, head, et_sub, src_local, task)
            original_details = compute_original_prediction(proxy, x_sub, ei_sub, et_sub)

            for explainer_name, adapter in adapters.items():
                pending_k_values = [
                    top_k
                    for top_k in top_k_values
                    if _completed_key(explainer_name, pair_idx, task, top_k) not in completed
                ]
                if not pending_k_values:
                    continue

                explanation: Optional[ExplanationResult] = None
                node_scores: Optional[Tensor] = None
                try:
                    explanation = adapter.explain(
                        encoder,
                        head,
                        x_sub,
                        ei_sub,
                        et_sub,
                        src_local,
                        dst_local,
                        task,
                        ctx.y_act_test,
                        ctx.y_time_test,
                        pair_idx,
                        device,
                        use_model_target=True,
                        protected_nodes=protected_nodes,
                    )
                    if explanation is not None:
                        explanation.src_global = int(src_id)
                        explanation.dst_global = int(dst_id)
                        explanation.nodes_global = nodes_t.detach()
                        explanation.edge_index = ei_sub.detach()
                        explanation.edge_type = et_sub.detach()
                        node_scores = node_scores_from_mask(explanation, x_sub.size(0), ei_sub).to(device)
                    encoder.eval()
                    head.eval()
                except Exception as exc:
                    warnings.warn(
                        f"[{explainer_name}] explanation failed for "
                        f"pair={pair_idx}, task={task}: {exc}"
                    )

                if explanation is None or node_scores is None:
                    warnings.warn(
                        f"[{explainer_name}] no usable node scores for "
                        f"pair={pair_idx}, task={task}; recording NaN"
                    )

                for top_k in pending_k_values:
                    details: Dict[str, Any] = {
                        "fid_prob_plus": float("nan"),
                        "fid_prob_minus": float("nan"),
                        "fid_acc_plus": float("nan"),
                        "fid_acc_minus": float("nan"),
                        "p_orig": float("nan"),
                        "p_complement": float("nan"),
                        "p_explanation": float("nan"),
                        "target_class": "",
                    }
                    num_selected_nodes = ""
                    num_selected_edges = ""
                    selection_diag: Dict[str, Any] = {
                        "selected_dst": "",
                        "selected_target_value_nodes": "",
                        "removed_target_value_nodes_fid_plus": "",
                    }

                    if node_scores is not None:
                        try:
                            candidate_scores = exclude_protected_scores(node_scores, protected_nodes)
                            binary_mask = threshold_node_scores_by_count(candidate_scores, top_k)
                            num_selected_nodes, num_selected_edges = selected_node_edge_counts(
                                binary_mask, ei_sub
                            )
                            selection_diag = _selection_diagnostics(
                                binary_mask, dst_local, target_nodes, protected_nodes
                            )
                            warn_if_degenerate_explanation(
                                explainer_name,
                                pair_idx,
                                task,
                                top_k,
                                x_sub,
                                ei_sub,
                                binary_mask,
                            )
                            details = compute_fidelity_details(
                                proxy,
                                x_sub,
                                ei_sub,
                                et_sub,
                                binary_mask,
                                metrics=fidelity_metrics,
                                original=original_details,
                                protected_nodes=protected_nodes,
                            )
                        except Exception as exc:
                            warnings.warn(
                                f"[{explainer_name}] fidelity failed for "
                                f"pair={pair_idx}, task={task}, "
                                f"k={top_k}: {exc}"
                            )

                    row = {
                        "explainer": explainer_name,
                        "pair_idx": pair_idx,
                        "task": task,
                        "top_k": int(top_k),
                        "fid_prob_plus": details["fid_prob_plus"],
                        "fid_prob_minus": details["fid_prob_minus"],
                        "fid_acc_plus": details["fid_acc_plus"],
                        "fid_acc_minus": details["fid_acc_minus"],
                        "p_orig": details["p_orig"],
                        "p_complement": details["p_complement"],
                        "p_explanation": details["p_explanation"],
                        "target_class": details["target_class"],
                        "num_nodes_sub": int(x_sub.size(0)),
                        "num_edges_sub": int(ei_sub.size(1)),
                        "num_selected_nodes": num_selected_nodes,
                        "num_selected_edges": num_selected_edges,
                        **selection_diag,
                        "mask_type": explanation.primary_mask_type if explanation is not None else "",
                    }
                    _append_checkpoint_row(checkpoint_path, row)
                    completed.add(_completed_key(explainer_name, pair_idx, task, top_k))
                    values = " ".join(
                        f"{metric}:fid+={details[f'fid_{metric}_plus']:.3f} "
                        f"fid-={details[f'fid_{metric}_minus']:.3f}"
                        for metric in fidelity_metrics
                    )
                    print(f"[{explainer_name}] pair={pair_idx} k={top_k} {values}")

    _write_aggregates(
        checkpoint_path, summary_path, auc_path, top_k_values, fidelity_metrics
    )

    elapsed = time.perf_counter() - mode_start
    print(f"Heterogeneity mode '{args.heterogeneity_mode}' finished in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
