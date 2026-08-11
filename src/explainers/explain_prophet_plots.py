"""Node-primary plot helpers for the standalone PROPHET explainer."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from rdflib import Graph

from src.explainers._shared import _entity_short_label

_REL_VIEW_MAP: Dict[str, str] = {
    "hasevent_activity_concept_name": "activity",
    "hasevent_timestamp_time_timestamp": "timestamp",
    "hasevent_otherC_lifecycle_transition": "lifecycle",
    "hasevent_otherC_org_group": "org_group",
    "hasevent_otherC_org_resource": "org_resource",
    "hasevent_otherC_org_role": "org_role",
    "hasevent_otherC_resourcecountry": "resourcecountry",
    "hascase_otherC_impact": "case_impact",
    "hascase_otherC_organizationcountry": "case_country",
    "hascase_otherN_product": "case_product",
    "directlyFollows": "directlyFollows",
    "contains": "case_contains",
    "hascaseID_case_concept_name": "case_concept",
}

_REL_COLORS: Dict[str, str] = {
    "timestamp": "#d1495b",
    "activity": "#2878b5",
    "org_resource": "#e07a1f",
    "org_group": "#b58b00",
    "org_role": "#7851a9",
    "lifecycle": "#2a9d6f",
    "directlyFollows": "#5f6b72",
    "case_contains": "#8c969d",
    "default": "#aab2b8",
}


def rel_view(rel_uri: str) -> str:
    for fragment, view in _REL_VIEW_MAP.items():
        if fragment in rel_uri:
            return view
    leaf = rel_uri.rstrip("/").rsplit("/", 1)[-1]
    return leaf or rel_uri


def _rel_color(view: str) -> str:
    return _REL_COLORS.get(view, _REL_COLORS["default"])


def plot_node_importance(
    scores: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
    pair_side_text: Optional[str] = None,
) -> None:
    """Plot the already-ranked node-mask values."""

    height = max(4.0, len(scores) * 0.28)
    if pair_side_text:
        fig = plt.figure(figsize=(13.0, height))
        grid = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.3], wspace=0.12)
        ax = fig.add_subplot(grid[0, 0])
        side = fig.add_subplot(grid[0, 1])
        side.axis("off")
        side.text(
            0.0,
            1.0,
            pair_side_text,
            transform=side.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            linespacing=1.22,
        )
    else:
        fig, ax = plt.subplots(figsize=(10.5, height))

    positions = np.arange(len(scores))
    colors = plt.cm.viridis(np.clip(scores, 0.0, 1.0))
    ax.barh(positions, scores, color=colors, alpha=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("PROPHET node mask")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    if pair_side_text:
        fig.subplots_adjust(left=0.26, right=0.98, top=0.9, bottom=0.14)
    else:
        fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_local(
    g_rdf: Graph,
    id2ent: Dict[int, str],
    nodes_global: torch.Tensor,
    ei_sub_local: torch.Tensor,
    et_sub: torch.Tensor,
    rel2id: Dict[str, int],
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    protected_nodes: Sequence[int],
    src_local: int,
    dst_local: int,
    task: str,
    pair_idx: int,
    top_nodes: int,
    top_edges: int,
    out_path: str,
) -> None:
    """Plot node masks with supplementary edge-mask context."""

    id2rel = {value: key for key, value in rel2id.items()}
    num_rel = max(rel2id.values()) + 1
    protected = {int(node) for node in protected_nodes}
    selectable = [i for i in range(node_mask.numel()) if i not in protected]
    ranked_nodes = sorted(
        selectable,
        key=lambda node: (-float(node_mask[node].item()), node),
    )[:top_nodes]
    ranked_edges = torch.argsort(edge_mask, descending=True)[
        : min(top_edges, edge_mask.numel())
    ].tolist()

    included: Set[int] = set(ranked_nodes) | protected
    included.update(node for node in (src_local, dst_local) if node >= 0)
    for edge_id in ranked_edges:
        included.add(int(ei_sub_local[0, edge_id]))
        included.add(int(ei_sub_local[1, edge_id]))

    graph = nx.DiGraph()
    nodes_list = nodes_global.tolist()
    for local_id in included:
        global_id = int(nodes_list[local_id])
        uri = id2ent.get(global_id, "?")
        graph.add_node(
            local_id,
            label=_entity_short_label(g_rdf, uri, max_len=34, separator="\n"),
            importance=float(node_mask[local_id].item()),
            protected=local_id in protected,
        )

    for edge_id in ranked_edges:
        source = int(ei_sub_local[0, edge_id])
        destination = int(ei_sub_local[1, edge_id])
        if source not in included or destination not in included:
            continue
        relation_id = int(et_sub[edge_id].item())
        relation_uri = id2rel.get(relation_id % num_rel, str(relation_id))
        graph.add_edge(
            source,
            destination,
            weight=float(edge_mask[edge_id].item()),
            view=rel_view(relation_uri),
        )

    positions = nx.spring_layout(graph, seed=42, k=2.8)
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_facecolor("white")

    for source, destination, data in graph.edges(data=True):
        weight = data["weight"]
        color = _rel_color(data["view"])
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(source, destination)],
            ax=ax,
            edge_color=color,
            width=0.8 + 4.0 * weight,
            alpha=0.78,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            connectionstyle="arc3,rad=0.10",
        )

    node_list = list(graph.nodes())
    colors, sizes = [], []
    for local_id in node_list:
        importance = graph.nodes[local_id]["importance"]
        if local_id == src_local:
            colors.append("#d1495b")
            sizes.append(1850)
        elif local_id == dst_local:
            colors.append("#2a9d6f")
            sizes.append(1850)
        elif graph.nodes[local_id]["protected"]:
            colors.append("#f0c75e")
            sizes.append(1450)
        else:
            colors.append(plt.cm.viridis(importance))
            sizes.append(650 + 1250 * importance)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=node_list,
        ax=ax,
        node_color=colors,
        node_size=sizes,
        alpha=0.92,
    )

    for local_id in node_list:
        importance = graph.nodes[local_id]["importance"]
        x_pos, y_pos = positions[local_id]
        suffix = "\n[protected]" if graph.nodes[local_id]["protected"] else ""
        ax.text(
            x_pos,
            y_pos + 0.06,
            f"{graph.nodes[local_id]['label']}\nmask={importance:.2f}{suffix}",
            fontsize=7,
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.28",
                fc="white",
                ec="gray",
                lw=0.5,
                alpha=0.9,
            ),
        )

    legend: List[mpatches.Patch] = [
        mpatches.Patch(color="#d1495b", label="source event"),
        mpatches.Patch(color="#2a9d6f", label="destination event"),
        mpatches.Patch(color="#f0c75e", label="protected context"),
    ]
    seen_views: Set[str] = set()
    for _, _, data in graph.edges(data=True):
        view = data["view"]
        if view not in seen_views:
            seen_views.add(view)
            legend.append(mpatches.Patch(color=_rel_color(view), label=view))
    ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    task_label = task.replace("otherC_", "").replace("_", " ")
    ax.set_title(
        f"PROPHET local explanation - pair {pair_idx} - task: {task_label}\n"
        "Node color/size: node mask | Edge width: supplementary edge mask",
        fontsize=10,
        pad=14,
    )
    ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
