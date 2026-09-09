
from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdflib import Graph

from src.gnn4ppm.train import (
    MultiTaskHead,
    RGCNEncoder,
    _sanitize,
    build_graph,
    build_train_directly_follows,
    get_event_attrs,
)
from src.utils.io_helpers import (
    load_case_split,
    load_embeddings,
    load_entity2id,
    load_pt,
    load_vocabs,
)
from src.utils.uri_helpers import EVENT_RE


def setup_matplotlib_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


def _build_model_from_checkpoint(
    ckpt: dict, device: torch.device
) -> Tuple[RGCNEncoder, torch.nn.Module]:
    best = ckpt["best_params"]
    hid = best["hid_dim"]
    num_rel = ckpt["num_rel"]
    in_dim = ckpt["in_dim"]
    encoder = RGCNEncoder(
        in_dim,
        hid,
        hid,
        num_rel,
        num_bases=best["num_bases"],
        dropout=best["dropout"],
    ).to(device)
    n_act = ckpt["n_act"]
    otherC_num_classes = ckpt["otherC_num_classes"]
    c_vocabs_dummy = {k: {i: i for i in range(sz)} for k, sz in otherC_num_classes.items()}
    head = MultiTaskHead(
        hid,
        n_act,
        c_vocabs_dummy,
        list(ckpt["otherN_keys"]),
        head_hidden=best["head_hidden"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    encoder.eval()
    head.eval()
    return encoder, head


def _case_map_from_ent2id(
    id2ent: Dict[int, str],
) -> Dict[str, Dict[int, int]]:
    case_map: Dict[str, Dict[int, int]] = {}
    for nid, uri in id2ent.items():
        m = EVENT_RE.search(uri)
        if m:
            ev_num, cid = int(m.group(1)), m.group(2)
            case_map.setdefault(cid, {})[ev_num] = nid
    return case_map


def _build_train_test_pairs(
    case_map: Dict[str, Dict[int, int]],
    train_cases: Set[str],
    test_cases: Set[str],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    train_pairs: List[Tuple[int, int]] = []
    test_pairs: List[Tuple[int, int]] = []
    for cid, km in case_map.items():
        ks = sorted(km.keys())
        case_pairs = [
            (km[ks[i]], km[ks[i + 1]]) for i in range(len(ks) - 1) if ks[i] + 1 in km
        ]
        if cid in train_cases:
            train_pairs.extend(case_pairs)
        elif cid in test_cases:
            test_pairs.extend(case_pairs)
    return train_pairs, test_pairs


def _scalar_for_task(
    out: dict,
    task: str,
    y_act: Optional[torch.Tensor],
    y_time: Optional[torch.Tensor],
    pair_test_idx: int,
    device: torch.device,
) -> torch.Tensor:
    oa = out["act"][0]
    if task == "activity":
        if y_act is not None and y_act[pair_test_idx].item() >= 0:
            c = int(y_act[pair_test_idx].item())
            return oa[c]
        return oa.max()
    if task == "time":
        v = out["time"][0]
        return v if v.numel() == 1 else v.sum()
    if task.startswith("otherC_"):
        k = task[len("otherC_"):]
        if k not in out["otherC"]:
            raise KeyError(f"otherC head missing key (sanitized): {k}")
        return out["otherC"][k][0].max()
    if task.startswith("otherN_"):
        k = task[len("otherN_"):]
        if k not in out["otherN"]:
            raise KeyError(f"otherN head missing key: {k}")
        v = out["otherN"][k][0]
        return v if v.numel() == 1 else v.sum()
    raise ValueError(f"Unknown task: {task}")


def _scalar_for_model_task(
    out: dict,
    task: str,
) -> torch.Tensor:
    """Return the scalar for model-explanation attribution.

    This intentionally targets the model's own prediction, not the ground-truth
    label.  Use this for fidelity benchmarking where the metric compares
    perturbed predictions against the original model prediction.
    """
    if task == "activity":
        return out["act"][0].max()
    if task == "time":
        v = out["time"][0]
        return v if v.numel() == 1 else v.sum()
    if task.startswith("otherC_"):
        k = task[len("otherC_"):]
        if k not in out["otherC"]:
            raise KeyError(f"otherC head missing key (sanitized): {k}")
        return out["otherC"][k][0].max()
    if task.startswith("otherN_"):
        k = task[len("otherN_"):]
        if k not in out["otherN"]:
            raise KeyError(f"otherN head missing key: {k}")
        v = out["otherN"][k][0]
        return v if v.numel() == 1 else v.sum()
    raise ValueError(f"Unknown task: {task}")


def parse_task_list(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, Sequence):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise TypeError(f"Expected comma-separated task string or sequence, got {type(raw).__name__}")


def _task_from_otherc_key(key: str) -> str:
    return f"otherC_{key}"


def _task_from_othern_key(key: str) -> str:
    return f"otherN_{key}"


def _otherc_task_lookup(ckpt: dict, vocabs: Dict[str, Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    ckpt_keys = {str(key) for key in ckpt.get("otherC_num_classes", {})}
    for key in ckpt.get("otherC_num_classes", {}):
        lookup[str(key)] = _task_from_otherc_key(str(key))
    for raw_key in vocabs.get("c_vocabs", {}):
        sanitized = _sanitize(str(raw_key))
        if ckpt_keys and sanitized not in ckpt_keys:
            continue
        lookup.setdefault(sanitized, _task_from_otherc_key(sanitized))
        lookup.setdefault(str(raw_key), _task_from_otherc_key(sanitized))
    return lookup


def _othern_task_lookup(ckpt: dict, vocabs: Dict[str, Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    ckpt_keys = {str(key) for key in ckpt.get("otherN_keys", ())}
    for key in ckpt.get("otherN_keys", ()):
        lookup[str(key)] = _task_from_othern_key(str(key))
    for raw_key in vocabs.get("n_keys", ()):
        sanitized = _sanitize(str(raw_key))
        if ckpt_keys and sanitized not in ckpt_keys:
            continue
        lookup.setdefault(sanitized, _task_from_othern_key(sanitized))
        lookup.setdefault(str(raw_key), _task_from_othern_key(sanitized))
    return lookup


def _find_otherc_alias(alias: str, available_tasks: Sequence[str]) -> Optional[str]:
    needles = {
        "resource": ("org_resource",),
        "role": ("org_role",),
        "lifecycle": ("lifecycle_transition",),
        "lifecycle_transition": ("lifecycle_transition",),
    }.get(alias)
    if needles is None:
        return None
    matches = [
        task
        for task in available_tasks
        if all(needle in task.lower() for needle in needles)
    ]
    return sorted(matches)[0] if matches else None


def resolve_task_aliases(
    requested: Any,
    ckpt: dict,
    vocabs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, str], List[Dict[str, str]]]:
    vocabs = vocabs or {}
    otherc_lookup = _otherc_task_lookup(ckpt, vocabs)
    othern_lookup = _othern_task_lookup(ckpt, vocabs)
    available_otherc = sorted(set(otherc_lookup.values()))
    tasks: List[str] = []
    aliases: Dict[str, str] = {}
    skipped: List[Dict[str, str]] = []

    for raw_task in parse_task_list(requested):
        token = raw_task.strip()
        normalized = token.lower()
        resolved: Optional[str] = None
        reason = ""

        if normalized == "activity":
            resolved = "activity"
        elif normalized == "time":
            resolved = "time"
        elif normalized in {"resource", "role", "lifecycle", "lifecycle_transition"}:
            resolved = _find_otherc_alias(normalized, available_otherc)
            if resolved is None:
                reason = f"no available categorical head for alias '{token}'"
        elif token.startswith("otherC_"):
            key = token[len("otherC_") :]
            resolved = _task_from_otherc_key(key) if key in otherc_lookup else None
            if resolved is None:
                reason = f"otherC head not found for '{token}'"
        elif token.startswith("otherN_"):
            key = token[len("otherN_") :]
            resolved = _task_from_othern_key(key) if key in othern_lookup else None
            if resolved is None:
                reason = f"otherN head not found for '{token}'"
        elif token in otherc_lookup:
            resolved = otherc_lookup[token]
        elif token in othern_lookup:
            resolved = othern_lookup[token]
        else:
            sanitized = _sanitize(token)
            if sanitized in otherc_lookup:
                resolved = otherc_lookup[sanitized]
            elif sanitized in othern_lookup:
                resolved = othern_lookup[sanitized]
            else:
                reason = f"unknown task or unavailable alias '{token}'"

        if resolved is None:
            skipped.append({"requested": token, "reason": reason})
            print(f"[warn] Skipping task '{token}': {reason}")
            continue
        if resolved not in tasks:
            tasks.append(resolved)
        aliases[token] = resolved

    if not tasks:
        requested_text = ", ".join(parse_task_list(requested))
        available_text = ", ".join(["activity", "time", *available_otherc, *sorted(set(othern_lookup.values()))])
        raise ValueError(
            f"No requested tasks are available. Requested: {requested_text}. "
            f"Available: {available_text}"
        )
    return tasks, aliases, skipped


def _k_hop_nodes(
    edge_index: torch.Tensor,
    seeds: List[int],
    num_nodes: int,
    k: int,
) -> Set[int]:
    """BFS within k hops of seeds; k<=0 returns the full node set."""
    if k <= 0:
        return set(range(num_nodes))
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    ei = edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        s, t = int(ei[0, i]), int(ei[1, i])
        adj[s].append(t)
        adj[t].append(s)
    seen: Set[int] = set()
    frontier = set(seeds)
    for _ in range(k):
        seen |= frontier
        nxt: Set[int] = set()
        for u in frontier:
            for v in adj.get(u, []):
                nxt.add(v)
        frontier = nxt - seen
    return seen | frontier


def _k_hop_subgraph(
    seeds: List[int],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int]]:
    ei_np = edge_index.cpu().numpy()
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for idx in range(ei_np.shape[1]):
        adj[int(ei_np[0, idx])].append(int(ei_np[1, idx]))
        adj[int(ei_np[1, idx])].append(int(ei_np[0, idx]))

    seen: Set[int] = set(seeds)
    frontier = set(seeds)
    for _ in range(k):
        nxt: Set[int] = set()
        for u in frontier:
            for v in adj.get(u, []):
                if v not in seen:
                    nxt.add(v)
        seen |= nxt
        frontier = nxt

    nodes_sorted = sorted(seen)
    g2l: Dict[int, int] = {gn: ln for ln, gn in enumerate(nodes_sorted)}
    nodes_t = torch.tensor(nodes_sorted, dtype=torch.long)

    nodes_set = set(nodes_sorted)
    src_np, dst_np = ei_np[0], ei_np[1]
    keep = np.array(
        [s in nodes_set and d in nodes_set for s, d in zip(src_np, dst_np)]
    )
    ei_sub_global = edge_index[:, torch.from_numpy(keep)]
    et_sub = edge_type[torch.from_numpy(keep)]

    ei_sub_local = torch.stack(
        [
            torch.tensor([g2l[int(n)] for n in ei_sub_global[0].tolist()]),
            torch.tensor([g2l[int(n)] for n in ei_sub_global[1].tolist()]),
        ]
    )
    return nodes_t, ei_sub_local, et_sub, g2l


_CATEGORY_PREFIXES: List[Tuple[str, str]] = [
    ("http://eventActivity.org/", "eventActivity"),
    ("http://eventTimestamp.org/", "eventTimestamp"),
    ("http://eventOtherC.org/", "eventOtherC"),
    ("http://eventOtherN.org/", "eventOtherN"),
    ("http://eventID.org/", "eventID"),
    ("http://caseID.org/", "caseID"),
    ("http://caseOtherC.org/", "caseOtherC"),
    ("http://caseOtherN.org/", "caseOtherN"),
    ("http://event", "event"),
    ("http://case", "case"),
]


def _entity_category(uri: str) -> str:
    for prefix, label in _CATEGORY_PREFIXES:
        if uri.startswith(prefix):
            return label
    return "literal"


def _entity_short_label(
    g: Graph,
    uri: str,
    max_len: int = 56,
    separator: str = " | ",
) -> str:
    """Human-readable label: event -> activity+timestamp, otherwise last URI segment.
    """
    if EVENT_RE.search(uri):
        act, ts, _, _ = get_event_attrs(g, uri)
        parts: List[str] = []
        if act:
            leaf = act.rsplit("/", 1)[-1]
            parts.append(leaf[:40] + (".." if len(leaf) > 40 else ""))
        if ts:
            parts.append(f"t={str(ts)[:18]}")
        s = separator.join(parts) if parts else uri
        return s if len(s) <= max_len else s[: max_len - 2] + ".."
    leaf = str(uri).rstrip("/").rsplit("/", 1)[-1]
    if not leaf:
        leaf = uri
    s = leaf if len(leaf) <= max_len else leaf[: max_len - 2] + ".."
    return s


def _enriched_label(g: Graph, id2ent: Dict[int, str], nid: int) -> str:
    uri = id2ent.get(int(nid), "?")
    return f"#{int(nid)} [{_entity_category(uri)}] {_entity_short_label(g, uri)}"


def _pair_side_caption(
    g: Graph,
    id2ent: Dict[int, str],
    src_id: int,
    dst_id: int,
    pi: int,
    wrap_w: int = 46,
) -> str:
    su = str(id2ent.get(int(src_id), "?"))
    du = str(id2ent.get(int(dst_id), "?"))
    src_lab = _entity_short_label(g, su, max_len=86)
    dst_lab = _entity_short_label(g, du, max_len=86)
    su_wrapped = textwrap.fill(su, width=wrap_w, break_long_words=True)
    du_wrapped = textwrap.fill(du, width=wrap_w, break_long_words=True)
    return (
        f"src:\n{src_lab}\n(#{src_id})\n{su_wrapped}\n\n"
        f"dst:\n{dst_lab}\n(#{dst_id})\n{du_wrapped}\n\n"
        f"test_pairs[{pi}]"
    )


def _plot_top_entities(
    scores: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
    top_k: int = 20,
    xlabel: str = "|Ridge coef|",
    pair_side_text: Optional[str] = None,
) -> None:
    order = np.argsort(-scores)[:top_k]
    s = scores[order]
    lab = [labels[i] for i in order]
    h = max(4.0, top_k * 0.25)
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
    y_pos = np.arange(len(s))
    ax.barh(y_pos, s, color="steelblue", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_top_entities_signed(
    scores: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
    top_k: int = 20,
    xlabel: str = "Score  (red = positive, blue = negative)",
    pair_side_text: Optional[str] = None,
) -> None:
    order = np.argsort(-np.abs(scores))[:top_k]
    s = scores[order]
    lab = [labels[i] for i in order]
    h = max(4.0, top_k * 0.25)
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
    y_pos = np.arange(len(s))
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in s]
    ax.barh(y_pos, s, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _plot_top_entities_signed_noreorder(
    scores: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
    top_k: int = 20,
    xlabel: str = "|Ridge coef|",
    pair_side_text: Optional[str] = None,
) -> None:
    vals = np.asarray(scores, dtype=float)[:top_k]
    lab = list(labels)[:top_k]

    h = max(4.0, top_k * 0.25)
    if pair_side_text:
        fig = plt.figure(figsize=(12.8, h))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.32], wspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        ax_r = fig.add_subplot(gs[0, 1])
        ax_r.axis("off")
        ax_r.text(
            0.0, 1.0, pair_side_text,
            transform=ax_r.transAxes,
            va="top", ha="left", fontsize=7.5, linespacing=1.22,
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
    ax.set_xlabel("Input*Gradient attribution  (red = pushes prediction up, blue = down)")
    ax.set_title(title)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_global_category(
    cat_df: pd.DataFrame,
    task: str,
    out_path: str,
    value_col: str = "mean_score",
    xlabel: str = "Mean score",
    title_template: str = "Global entity-category saliency — {task_lbl}",
) -> None:
    df = cat_df.sort_values(value_col, ascending=True)
    h = max(4.0, len(df) * 0.45)
    fig, ax = plt.subplots(figsize=(10, h))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df[value_col].values, color="steelblue", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{c} (n={int(n)})" for c, n in zip(df["category"].values, df["count"].values)],
        fontsize=9,
    )
    ax.set_xlabel(xlabel)
    task_lbl = task.replace("otherC_", "").replace("otherN_", "").replace("_", " ")
    ax.set_title(title_template.format(task_lbl=task_lbl))
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_common_explainer_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--ttl", required=True, help="RDF/Turtle knowledge-graph file.")
    ap.add_argument("--emb", required=True, help="Entity embeddings (.npy).")
    ap.add_argument("--entity2id", required=True, help="entity2id.json mapping.")
    ap.add_argument("--case_split", required=True, help="case_split.json (train/test case IDs).")
    ap.add_argument("--model", required=True, help="*_model.pt checkpoint from train.py.")
    ap.add_argument("--vocabs", default=None, help="*_vocabs.json for activity names.")
    ap.add_argument(
        "--best-test",
        "--best-val",
        dest="best_test",
        default=None,
        help="best_test.pt with test labels. If omitted, activity uses max logit.",
    )
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument(
        "--tasks",
        default="activity,time",
        help=(
            "Comma-separated tasks. Supports aliases activity, resource, role, "
            "lifecycle, plus canonical activity, time, otherC_*, otherN_*."
        ),
    )
    ap.add_argument("--num-samples", type=int, default=10,
                    help="Number of test pairs to explain.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for test-pair sampling. Defaults to --seed.",
    )


@dataclass
class ExplainerContext:
    encoder: Any
    head: Any
    x: torch.Tensor
    g: Graph
    id2ent: Dict[int, str]
    ei_test: torch.Tensor
    et_test: torch.Tensor
    rel2id: Dict[str, int]
    num_rel: int
    test_pairs: List[Tuple[int, int]]
    y_act_test: Optional[torch.Tensor]
    y_time_test: Optional[torch.Tensor]
    act_vocab: Dict[str, int]
    id2act: Dict[int, str]
    tasks: List[str]
    task_aliases: Dict[str, str]
    skipped_tasks: List[Dict[str, str]]
    n_test: int
    pair_indices: List[int]
    rng: Any


def load_explainer_context(
    args: Any,
    device: torch.device,
) -> ExplainerContext:
    seed = int(args.seed)
    sample_seed_arg = getattr(args, "sample_seed", None)
    sample_seed = int(sample_seed_arg if sample_seed_arg is not None else seed)
    rng = np.random.default_rng(seed)
    sample_rng = np.random.default_rng(sample_seed)
    torch.manual_seed(seed)

    ckpt = load_pt(args.model)
    encoder, head = _build_model_from_checkpoint(ckpt, device)

    ent2id = load_entity2id(args.entity2id)
    id2ent: Dict[int, str] = {i: e for e, i in ent2id.items()}
    x = load_embeddings(args.emb).to(device)

    train_cases, test_cases = load_case_split(args.case_split)

    print("Parsing TTL …")
    g = Graph()
    g.parse(args.ttl, format="turtle")

    rel2id: Dict[str, int] = dict(ckpt["rel2id"])
    num_rel = len(rel2id)
    assert num_rel == ckpt["num_rel"], "TTL graph relation count mismatch with checkpoint"

    ei_test, et_test = build_graph(
        g,
        ent2id,
        rel2id,
        excluded_cases=train_cases,
        allow_new_relations=False,
    )
    # same train-only directlyFollows edges the model was trained with
    ei_df, et_df = build_train_directly_follows(
        g, ent2id, rel2id, train_cases, allow_new_relations=False,
    )
    ei_test = torch.cat([ei_test, ei_df], dim=1)
    et_test = torch.cat([et_test, et_df])
    ei_test, et_test = ei_test.to(device), et_test.to(device)

    case_map = _case_map_from_ent2id(id2ent)
    _, test_pairs = _build_train_test_pairs(case_map, train_cases, test_cases)

    y_act_test: Optional[torch.Tensor] = None
    y_time_test: Optional[torch.Tensor] = None
    if args.best_test:
        bv = load_pt(args.best_test)
        y_act_test = bv["y_act"].to(device)
        y_time_test = bv["y_time"].to(device)
        pv = bv["pairs_test"] 
        test_pairs_t = torch.tensor(test_pairs, dtype=torch.long)
        if pv.shape != test_pairs_t.shape or not torch.all(pv == test_pairs_t):
            print("[warn] best-test pairs_test differs from rebuilt test_pairs")
        if y_act_test.numel() != len(test_pairs):
            raise ValueError("best_test y_act length does not match number of test pairs")

    voc = load_vocabs(args.vocabs) if args.vocabs else {}
    act_vocab: Dict[str, int] = voc.get("act_vocab", {})
    id2act: Dict[int, str] = {v: k for k, v in act_vocab.items()}

    tasks, task_aliases, skipped_tasks = resolve_task_aliases(args.tasks, ckpt, voc)
    if skipped_tasks:
        skipped_text = ", ".join(item["requested"] for item in skipped_tasks)
        print(f"[warn] Skipped unavailable tasks: {skipped_text}")
    n_test = len(test_pairs)
    if n_test == 0:
        raise RuntimeError("No test pairs to explain.")

    sample_n = min(args.num_samples, n_test)
    pair_indices = sorted(sample_rng.choice(n_test, size=sample_n, replace=False).tolist())

    print(f"Test pairs: {n_test} total  ({len(pair_indices)} to explain)")

    return ExplainerContext(
        encoder=encoder,
        head=head,
        x=x,
        g=g,
        id2ent=id2ent,
        ei_test=ei_test,
        et_test=et_test,
        rel2id=rel2id,
        num_rel=num_rel,
        test_pairs=test_pairs,
        y_act_test=y_act_test,
        y_time_test=y_time_test,
        act_vocab=act_vocab,
        id2act=id2act,
        tasks=tasks,
        task_aliases=task_aliases,
        skipped_tasks=skipped_tasks,
        n_test=n_test,
        pair_indices=pair_indices,
        rng=rng,
    )


def get_classification_prediction(
    encoder: Any,
    head: Any,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    src_idx: int,
    task: str,
    device: torch.device,
) -> int:
    """
    Return the argmax class prediction for a classification task.

    Supports ``'activity'`` and ``'otherC_*'`` tasks.  Raises
    ``ValueError`` for regression tasks (``'time'``, ``'otherN_*'``).

    Parameters
    ----------
    encoder :
        RGCNEncoder instance (set to eval mode internally).
    head :
        MultiTaskHead instance (set to eval mode internally).
    x : Tensor [N, in_dim]
    edge_index : Tensor [2, E]
    edge_type : Tensor [E]
    src_idx : int
        Local index of the source event node whose prediction is returned.
    task : str
        ``'activity'`` or ``'otherC_<key>'``.
    device : torch.device

    Returns
    -------
    int
        Predicted class index.
    """
    encoder.eval()
    head.eval()
    with torch.no_grad():
        z = encoder(x, edge_index, edge_type)
        out = head(z[src_idx: src_idx + 1])

    if task == "activity":
        return int(out["act"].argmax(dim=-1).item())
    elif task.startswith("otherC_"):
        key = task[len("otherC_"):]
        if key not in out["otherC"]:
            available = list(out["otherC"].keys())
            raise KeyError(
                f"otherC head missing key '{key}'. Available: {available}"
            )
        return int(out["otherC"][key].argmax(dim=-1).item())
    else:
        raise ValueError(
            f"get_classification_prediction() only supports 'activity' and "
            f"'otherC_*' tasks. Got: '{task}'"
        )


def write_summary_json(
    path: str,
    args: Any,
    n_test: int,
    pair_indices: List[int],
    tasks: List[str],
    extra_fields: Optional[Dict[str, Any]] = None,
    rows: Optional[List[dict]] = None,
) -> None:
    data: Dict[str, Any] = {
        "ttl": os.path.abspath(args.ttl),
        "model": os.path.abspath(args.model),
        "vocabs": os.path.abspath(args.vocabs) if getattr(args, "vocabs", None) else None,
        "best_test": os.path.abspath(args.best_test) if getattr(args, "best_test", None) else None,
        "out": os.path.abspath(args.out),
        "num_test_pairs": n_test,
        "explained_pair_indices": pair_indices,
        "tasks": tasks,
        "seed": args.seed,
        "sample_seed": getattr(args, "sample_seed", args.seed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra_fields:
        data.update(extra_fields)
    if rows is not None:
        data["rows"] = rows
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Summary -> {path}")
