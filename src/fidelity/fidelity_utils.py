"""RGCN-safe fidelity utilities for the active curve pipeline."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from rdflib import URIRef
from torch import Tensor

P_ACT = "http://example.org/hasevent_activity_concept_name"
P_TIME = "http://example.org/hasevent_timestamp_time_timestamp"


def is_classification_task(task: str) -> bool:
    return task == "activity" or task.startswith("otherC_")


def is_regression_task(task: str) -> bool:
    return task == "time" or task.startswith("otherN_")


# ---------------------------------------------------------------------------
# Task-logit wrapper/proxy
# ---------------------------------------------------------------------------

class RGCNPerturbationWrapper(torch.nn.Module):
    """Expose one task prediction for one local source node."""

    def __init__(self, encoder: torch.nn.Module, head: torch.nn.Module, edge_type: Tensor, src_local: int, task: str = "activity") -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.edge_type = edge_type
        self.src_local = int(src_local)
        self.task = task

    def forward(self, x: Tensor, edge_index: Tensor, *, edge_type: Optional[Tensor] = None, src_local: Optional[int] = None) -> Tensor:
        et = self.edge_type if edge_type is None else edge_type
        src = self.src_local if src_local is None else int(src_local)
        z = self.encoder(x, edge_index, et)
        out = self.head(z[src : src + 1])
        if self.task == "activity":
            return out["act"]
        if self.task == "time":
            return out["time"].reshape(1, 1)
        if self.task.startswith("otherC_"):
            key = self.task[len("otherC_") :]
            if key not in out["otherC"]:
                raise KeyError(f"otherC head missing key '{key}'")
            return out["otherC"][key]
        if self.task.startswith("otherN_"):
            key = self.task[len("otherN_") :]
            if key not in out["otherN"]:
                raise KeyError(f"otherN head missing key '{key}'")
            return out["otherN"][key].reshape(1, 1)
        raise ValueError(
            "RGCNPerturbationWrapper supports 'activity', 'time', "
            f"'otherC_*', and 'otherN_*'; got {self.task!r}"
        )


class RGCNFidelityProxy:
    """Small proxy that owns the RGCN/process-mining perturbation semantics."""

    def __init__(self, model: RGCNPerturbationWrapper) -> None:
        self.model = model

    @torch.no_grad()
    def get_prediction(self, *args: Any, **kwargs: Any) -> Tensor:
        training = self.model.training
        self.model.eval()
        out = self.model(*args, **kwargs)
        self.model.train(training)
        return out

    def get_masked_prediction(self, x: Tensor, edge_index: Tensor, node_mask: Optional[Tensor] = None, edge_mask: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
        """Predict after applying keep masks.

        A node mask value of 1 keeps the node. Unkept nodes have their features
        zeroed, and incident edges are removed from the typed edge list.
        """

        edge_type = kwargs.pop("edge_type", self.model.edge_type).to(edge_index.device)
        x_perturbed = x.clone()
        keep_edges = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

        if node_mask is not None:
            node_keep = node_mask.reshape(-1).to(x.device) > 0.5
            x_perturbed = x_perturbed * node_keep.float().view(-1, 1)
            keep_edges &= node_keep.to(edge_index.device)[edge_index[0]]
            keep_edges &= node_keep.to(edge_index.device)[edge_index[1]]

        if edge_mask is not None:
            keep_edges &= edge_mask.reshape(-1).to(edge_index.device) > 0.5

        edge_index_perturbed = edge_index[:, keep_edges]
        edge_type_perturbed = edge_type[keep_edges]

        if edge_mask is not None:
            active = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            if edge_index_perturbed.numel() > 0:
                active[edge_index_perturbed.reshape(-1).to(x.device)] = True
            x_perturbed[~active] = 0.0

        return self.get_prediction(
            x_perturbed,
            edge_index_perturbed,
            edge_type=edge_type_perturbed,
            **kwargs,
        )


def build_proxy(encoder: torch.nn.Module, head: torch.nn.Module, edge_type: Tensor, src_local: int, task: str) -> RGCNFidelityProxy:
    wrapper = RGCNPerturbationWrapper(
        encoder=encoder, head=head, edge_type=edge_type, src_local=src_local, task=task
    ).to(edge_type.device)
    wrapper.eval()
    return RGCNFidelityProxy(wrapper)


# ---------------------------------------------------------------------------
# Explanation mask to node scores
# ---------------------------------------------------------------------------

def node_scores_from_mask(mask: Any, num_nodes: int, edge_index: Tensor) -> Tensor:
    """Return one node-importance score per local node."""

    node_mask = getattr(mask, "node_mask", None)
    if node_mask is not None:
        scores = node_mask.detach().float().reshape(-1)
        if scores.numel() != num_nodes:
            raise ValueError(f"node_mask length {scores.numel()} does not match num_nodes={num_nodes}")
        return scores

    edge_mask = getattr(mask, "edge_mask", None)
    if edge_mask is None:
        raise ValueError("Explanation mask contains neither node_mask nor edge_mask")

    edge_scores = edge_mask.detach().float().reshape(-1).to(edge_index.device)
    if edge_scores.numel() != edge_index.size(1):
        raise ValueError(f"edge_mask length {edge_scores.numel()} does not match edges={edge_index.size(1)}")

    scores = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    src, dst = edge_index[0], edge_index[1]
    for endpoint in (src, dst):
        scores.scatter_reduce_(0, endpoint, edge_scores, reduce="amax", include_self=True)
    return scores


# ---------------------------------------------------------------------------
# Top-k and protected-node masking
# ---------------------------------------------------------------------------

def threshold_node_scores_by_count(scores: Tensor, top_k: int) -> Tensor:
    """Return a binary node keep mask selecting the top-k finite scores."""

    scores = scores.detach().float().reshape(-1)
    n = scores.numel()
    if n == 0:
        raise ValueError("Cannot threshold an empty node-score tensor")
    k_requested = int(top_k)
    if k_requested <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    valid = torch.isfinite(scores)
    valid_indices = valid.nonzero(as_tuple=False).reshape(-1)
    if valid_indices.numel() == 0:
        raise ValueError("Cannot threshold node scores with no selectable nodes")
    k = min(valid_indices.numel(), k_requested)
    binary_mask = torch.zeros(n, dtype=torch.float32, device=scores.device)
    top_local = scores[valid_indices].topk(k).indices
    binary_mask[valid_indices[top_local]] = 1.0
    return binary_mask.view(-1, 1)


def exclude_protected_scores(scores: Tensor, protected_nodes: Sequence[int]) -> Tensor:
    """Return scores where protected nodes cannot be selected as evidence."""

    filtered = scores.detach().float().reshape(-1).clone()
    n = filtered.numel()
    for node in protected_nodes:
        if 0 <= int(node) < n:
            filtered[int(node)] = -torch.inf
    if torch.isneginf(filtered).all():
        raise ValueError("All node scores are protected; no context nodes remain")
    return filtered


def apply_protected_nodes(binary_mask: Tensor, protected_nodes: Sequence[int]) -> Tensor:
    """Force protected nodes to stay kept during masked prediction."""

    protected = binary_mask.detach().float().reshape(-1, 1).clone()
    n = protected.size(0)
    for node in protected_nodes:
        if 0 <= int(node) < n:
            protected[int(node)] = 1.0
    return protected


def selected_node_edge_counts(binary_mask: Tensor, edge_index: Tensor) -> Tuple[int, int]:
    node_keep = binary_mask.reshape(-1).to(edge_index.device) > 0.5
    keep_edges = node_keep[edge_index[0]] & node_keep[edge_index[1]]
    return int(node_keep.sum().item()), int(keep_edges.sum().item())


# ---------------------------------------------------------------------------
# Target-value-node diagnostics
# ---------------------------------------------------------------------------

def _sanitize_key(key: str) -> str:
    return key.replace(":", "_").replace("/", "_").replace(".", "_")


def task_target_value_nodes(g: Any, id2ent: Dict[int, str], nodes_global: Tensor, g2l: Dict[int, int], dst_global: int, task: str) -> Tuple[int, ...]:
    """Return local direct dst value nodes that reveal the target task."""

    dst_uri = id2ent.get(int(dst_global))
    if not dst_uri:
        return ()

    local_by_uri = {
        id2ent[int(global_id)]: g2l[int(global_id)]
        for global_id in nodes_global.detach().cpu().tolist()
        if int(global_id) in g2l and int(global_id) in id2ent
    }
    protected = set()
    otherc_key = task[len("otherC_") :] if task.startswith("otherC_") else None
    othern_key = task[len("otherN_") :] if task.startswith("otherN_") else None

    for pred, obj in g.predicate_objects(URIRef(dst_uri)):
        pred_s = str(pred)
        obj_local = local_by_uri.get(str(obj))
        if obj_local is None:
            continue
        if task == "activity" and pred_s == P_ACT:
            protected.add(int(obj_local))
        elif task == "time" and pred_s == P_TIME:
            protected.add(int(obj_local))
        elif otherc_key is not None and _sanitize_key(pred_s) == otherc_key:
            protected.add(int(obj_local))
        elif othern_key is not None and _sanitize_key(pred_s) == othern_key:
            protected.add(int(obj_local))
    return tuple(sorted(protected))


def warn_if_degenerate_explanation(explainer: str, pair_idx: int, task: str, top_k: int, x: Tensor, edge_index: Tensor, binary_mask: Tensor) -> None:
    node_keep = binary_mask.reshape(-1).to(x.device) > 0.5
    keep_edges = node_keep.to(edge_index.device)[edge_index[0]]
    keep_edges &= node_keep.to(edge_index.device)[edge_index[1]]
    nonzero_rows = int((x[node_keep].abs().sum(dim=1) > 0).sum().item())
    if int(keep_edges.sum().item()) == 0 and nonzero_rows == 0:
        warnings.warn(
            f"[{explainer}] pair={pair_idx} task={task} k={int(top_k)}: "
            "explanation perturbation has 0 edges and 0 non-zero feature rows; "
            "fidelity may be unreliable"
        )


# ---------------------------------------------------------------------------
# Prediction extraction and fidelity calculation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_original_prediction(proxy: RGCNFidelityProxy, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Dict[str, float]:
    original_prediction = proxy.get_prediction(x, edge_index, edge_type=edge_type)
    task = proxy.model.task
    if is_classification_task(task):
        target_class = int(original_prediction.argmax(dim=-1).item())
        p_orig = float(F.softmax(original_prediction, dim=-1)[0, target_class].item())
        return {
            "p_orig": p_orig,
            "target_class": float(target_class),
            "original_value": float(original_prediction[0, target_class].item()),
        }
    if is_regression_task(task):
        original_value = float(original_prediction.reshape(-1)[0].item())
        return {
            "p_orig": float("nan"),
            "target_class": float("nan"),
            "original_value": original_value,
        }
    raise ValueError(f"Unsupported fidelity task: {task!r}")


@torch.no_grad()
def compute_fidelity_details(
    proxy: RGCNFidelityProxy,
    x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    binary_mask: Tensor,
    metrics: Sequence[str] = ("prob",),
    original: Optional[Dict[str, float]] = None,
    protected_nodes: Sequence[int] = (),
) -> Dict[str, float]:
    """Compute classification and/or regression fidelity.

    For classification tasks, ``prob`` measures the signed probability drop for the
    original predicted class and ``acc`` measures whether the predicted class
    changes. For regression tasks, ``value`` measures absolute scalar prediction
    change.
    """

    selected = set(metrics)
    invalid = selected - {"prob", "acc", "value"}
    if invalid or not selected:
        raise ValueError(f"Unsupported fidelity metrics: {sorted(invalid or selected)}")
    original = original or compute_original_prediction(proxy, x, edge_index, edge_type)
    p_orig = float(original["p_orig"])
    target_class_f = float(original["target_class"])
    original_value = float(original["original_value"])
    task = proxy.model.task

    explanation_mask = apply_protected_nodes(binary_mask, protected_nodes)
    complement_mask = apply_protected_nodes(1.0 - binary_mask, protected_nodes)
    complement_logits = proxy.get_masked_prediction(
        x, edge_index, node_mask=complement_mask, edge_type=edge_type
    )
    explanation_logits = proxy.get_masked_prediction(
        x, edge_index, node_mask=explanation_mask, edge_type=edge_type
    )
    nan = float("nan")
    details = {
        "fid_prob_plus": nan,
        "fid_prob_minus": nan,
        "fid_acc_plus": nan,
        "fid_acc_minus": nan,
        "fid_value_plus": nan,
        "fid_value_minus": nan,
        "p_orig": p_orig,
        "p_complement": nan,
        "p_explanation": nan,
        "target_class": target_class_f,
        "original_value": original_value,
        "complement_value": nan,
        "explanation_value": nan,
    }

    if is_classification_task(task):
        target_class = int(target_class_f)
        p_complement = float(F.softmax(complement_logits, dim=-1)[0, target_class].item())
        p_explanation = float(F.softmax(explanation_logits, dim=-1)[0, target_class].item())
        complement_class = int(complement_logits.argmax(dim=-1).item())
        explanation_class = int(explanation_logits.argmax(dim=-1).item())
        details.update({
            "fid_prob_plus": p_orig - p_complement if "prob" in selected else nan,
            "fid_prob_minus": p_orig - p_explanation if "prob" in selected else nan,
            "fid_acc_plus": float(complement_class != target_class) if "acc" in selected else nan,
            "fid_acc_minus": float(explanation_class != target_class) if "acc" in selected else nan,
            "p_complement": p_complement,
            "p_explanation": p_explanation,
        })
        return details

    if is_regression_task(task):
        complement_value = float(complement_logits.reshape(-1)[0].item())
        explanation_value = float(explanation_logits.reshape(-1)[0].item())
        details.update({
            "fid_value_plus": abs(original_value - complement_value) if "value" in selected else nan,
            "fid_value_minus": abs(original_value - explanation_value) if "value" in selected else nan,
            "complement_value": complement_value,
            "explanation_value": explanation_value,
        })
        return details

    raise ValueError(f"Unsupported fidelity task: {task!r}")
