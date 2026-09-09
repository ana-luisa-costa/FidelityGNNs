
from __future__ import annotations

import contextlib
import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.explain.algorithm.utils import clear_masks
from src.gnn4ppm.train import MultiTaskHead, RGCNEncoder, _sanitize

_RESOURCE_OTHERC_RAW_KEY = "http://example.org/hasevent_otherC_org_resource"
_ROLE_OTHERC_RAW_KEY = "http://example.org/hasevent_otherC_org_role"
_LIFECYCLE_OTHERC_RAW_KEY = "http://example.org/hasevent_otherC_lifecycle_transition"
_TASK_ALIASES = {
    "resource": f"otherC_{_sanitize(_RESOURCE_OTHERC_RAW_KEY)}",
    "role": f"otherC_{_sanitize(_ROLE_OTHERC_RAW_KEY)}",
    "lifecycle": f"otherC_{_sanitize(_LIFECYCLE_OTHERC_RAW_KEY)}",
}


@dataclass(frozen=True)
class ProphetConfig:
    """Published DGL defaults plus local reproducibility settings."""

    epochs: int = 100
    lr: float = 0.01
    edge_size_reg: float = 0.005
    edge_ent_reg: float = 1.0
    node_size_reg: float = 1.0
    node_ent_reg: float = 0.1
    seed: int = 42
    eps: float = 1e-15
    parity_atol: float = 1e-5
    parity_rtol: float = 1e-4


@dataclass
class ProphetExplanation:
    """PROPHET masks and audit data for one pair and task."""

    node_mask: Tensor
    edge_mask: Tensor
    pred_class_id: int
    pred_class_label: str
    original_logit: float
    original_probability: float
    masked_logit: float
    masked_probability: float
    masked_class_id: int
    prediction_agreement: bool
    protected_nodes: Tuple[int, ...]
    seed: int
    encoder_parity_max_abs_diff: float
    node_grad_finite: bool
    edge_grad_finite: bool
    trace: List[Dict[str, float]]


def _normalize_task(task: str) -> str:
    return _TASK_ALIASES.get(task, task)


def validate_classification_task(task: str) -> str:
    normalized = _normalize_task(task)
    if normalized == "activity" or normalized.startswith("otherC_"):
        return normalized
    raise ValueError(
        "PROPHET currently supports classification tasks only: "
        f"'activity' and 'otherC_*'. Got: '{task}'"
    )


def _task_logits(out: dict, task: str) -> Tensor:
    if task == "activity":
        return out["act"]
    key = task[len("otherC_") :]
    if key not in out["otherC"]:
        available = sorted(out["otherC"].keys())
        raise KeyError(f"otherC head missing key '{key}'. Available: {available}")
    return out["otherC"][key]


def prophet_seed(global_seed: int, pair_idx: int, task: str) -> int:
    payload = f"{int(global_seed)}|{int(pair_idx)}|{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _canonical_edge_order(
    edge_index: Tensor,
    edge_type: Tensor,
    num_nodes: int,
) -> Tensor:
    """Return a stable typed-edge order independent of RDF iteration order."""

    relation_span = max(int(edge_type.max().item()) + 1, 1)
    keys = (
        (edge_index[0].long() * max(int(num_nodes), 1) + edge_index[1].long())
        * relation_span
        + edge_type.long()
    )
    return torch.argsort(keys, stable=True)


def _copy_encoder_for_edge_masks(encoder: RGCNEncoder) -> RGCNEncoder:
    """Keep the trained R-GCN aggregation path while isolating mask state."""

    copied = copy.deepcopy(encoder)
    copied.eval()
    return copied


@contextlib.contextmanager
def _relation_edge_masks(
    encoder: RGCNEncoder,
    edge_type: Tensor,
    edge_logits: Tensor,
    *,
    apply_sigmoid: bool,
):
    """Apply one mask per edge to each relation-specific RGCN propagation.

    Basis-decomposed RGCNConv executes one ``propagate`` call per relation.
    PyG's generic mask helper assumes one call over the complete edge list, so
    this hook supplies the matching relation slice immediately before every
    propagation call without changing the model's mean aggregation semantics.
    """

    states: List[Dict[str, int]] = []
    hooks = []
    convolutions = (encoder.conv1, encoder.conv2, encoder.conv3)

    for convolution in convolutions:
        state = {"relation": 0}
        states.append(state)
        convolution.explain = True
        convolution._apply_sigmoid = apply_sigmoid
        convolution._loop_mask = torch.ones(
            edge_logits.size(0), dtype=torch.bool, device=edge_logits.device
        )
        convolution._edge_mask = edge_logits[edge_type == 0]

        def pre_hook(module, _inputs, *, state=state):
            relation = state["relation"]
            state["relation"] += 1
            module._edge_mask = edge_logits[edge_type == relation]

        hooks.append(convolution.register_propagate_forward_pre_hook(pre_hook))

    def reset() -> None:
        for state in states:
            state["relation"] = 0

    try:
        yield reset
    finally:
        for hook in hooks:
            hook.remove()
        clear_masks(encoder)


def _binary_entropy(mask: Tensor, eps: float) -> Tensor:
    return -mask * torch.log(mask + eps) - (1.0 - mask) * torch.log(
        1.0 - mask + eps
    )


def _fixed_class_prediction_loss(logits: Tensor, pred_class_id: int) -> Tensor:
    """Return negative log probability for the original predicted class."""

    return -F.log_softmax(logits, dim=-1)[int(pred_class_id)]


def _mask_regularizers(
    node_mask: Tensor,
    edge_mask: Tensor,
    protected: Tensor,
    config: ProphetConfig,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    edge_size = config.edge_size_reg * edge_mask.sum()
    edge_ent = config.edge_ent_reg * _binary_entropy(edge_mask, config.eps).mean()
    selectable = node_mask[~protected]
    if selectable.numel() == 0:
        zero = node_mask.sum() * 0.0
        node_size, node_ent = zero, zero
    else:
        node_size = config.node_size_reg * selectable.mean()
        node_ent = config.node_ent_reg * _binary_entropy(
            selectable, config.eps
        ).mean()
    return edge_size, edge_ent, node_size, node_ent


def _encoder_parity(
    encoder: RGCNEncoder,
    masked_encoder: RGCNEncoder,
    x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    config: ProphetConfig,
) -> float:
    with torch.no_grad():
        original = encoder(x, edge_index, edge_type)
        copied = masked_encoder(x, edge_index, edge_type)
    max_diff = float((original - copied).abs().max().item())
    if not torch.allclose(
        original,
        copied,
        atol=config.parity_atol,
        rtol=config.parity_rtol,
    ):
        raise RuntimeError(
            "Copied R-GCN does not reproduce original embeddings: "
            f"max_abs_diff={max_diff:.6g}"
        )
    return max_diff


def _all_one_edge_mask_parity(
    encoder: RGCNEncoder,
    x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    config: ProphetConfig,
) -> None:
    with torch.no_grad():
        expected = encoder(x, edge_index, edge_type)
        ones = torch.ones(edge_index.size(1), device=x.device)
        with _relation_edge_masks(
            encoder, edge_type, ones, apply_sigmoid=False
        ) as reset_masks:
            reset_masks()
            actual = encoder(x, edge_index, edge_type)
    if not torch.allclose(
        expected,
        actual,
        atol=config.parity_atol,
        rtol=config.parity_rtol,
    ):
        max_diff = float((expected - actual).abs().max().item())
        raise RuntimeError(
            "All-one edge masks changed the R-GCN output: "
            f"max_abs_diff={max_diff:.6g}"
        )


def explain_prophet_masks(
    encoder: RGCNEncoder,
    head: MultiTaskHead,
    x_sub: Tensor,
    ei_sub: Tensor,
    et_sub: Tensor,
    src_local: int,
    task: str,
    protected_nodes: Sequence[int],
    config: ProphetConfig,
    *,
    pred_class_label: str | None = None,
) -> ProphetExplanation:
    """Learn individual-node and per-edge PROPHET masks."""

    task = validate_classification_task(task)
    if et_sub.numel() == 0:
        raise ValueError("PROPHET requires a subgraph containing at least one edge")
    if not 0 <= int(src_local) < x_sub.size(0):
        raise ValueError(f"src_local is outside the subgraph: {src_local}")

    device = x_sub.device
    ei_input, et_input = ei_sub.to(device), et_sub.to(device)
    edge_order = _canonical_edge_order(ei_input, et_input, x_sub.size(0))
    ei_work = ei_input[:, edge_order]
    et_work = et_input[edge_order]
    encoder.eval()
    head.eval()
    masked_encoder = _copy_encoder_for_edge_masks(encoder).to(device)

    parity_diff = _encoder_parity(
        encoder, masked_encoder, x_sub, ei_work, et_work, config
    )
    _all_one_edge_mask_parity(
        masked_encoder, x_sub, ei_work, et_work, config
    )

    with torch.no_grad():
        original_out = head(encoder(x_sub, ei_work, et_work)[src_local : src_local + 1])
        original_logits = _task_logits(original_out, task)[0]
        pred_class_id = int(original_logits.argmax().item())
        original_probs = F.softmax(original_logits, dim=-1)
        original_logit = float(original_logits[pred_class_id].item())
        original_probability = float(original_probs[pred_class_id].item())

    protected = torch.zeros(x_sub.size(0), dtype=torch.bool, device=device)
    valid_protected = tuple(
        sorted({int(node) for node in protected_nodes if 0 <= int(node) < x_sub.size(0)})
    )
    if valid_protected:
        protected[list(valid_protected)] = True

    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.seed))
    node_logits = torch.nn.Parameter(
        torch.randn(
            (x_sub.size(0), 1), generator=generator, device=device
        ) * 0.1
    )
    edge_std = torch.nn.init.calculate_gain("relu") * math.sqrt(
        2.0 / (2.0 * x_sub.size(0))
    )
    edge_logits = torch.nn.Parameter(
        torch.randn(
            et_work.size(0), generator=generator, device=device
        ) * edge_std
    )
    optimizer = torch.optim.Adam([node_logits, edge_logits], lr=config.lr)
    trace: List[Dict[str, float]] = []
    node_grad_finite = True
    edge_grad_finite = True

    modules = (masked_encoder, head)
    requires_grad_states = [
        [parameter.requires_grad for parameter in module.parameters()]
        for module in modules
    ]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    try:
        with _relation_edge_masks(
            masked_encoder, et_work, edge_logits, apply_sigmoid=True
        ) as reset_masks:
            for epoch in range(config.epochs):
                reset_masks()
                optimizer.zero_grad(set_to_none=True)
                learned_node_mask = torch.sigmoid(node_logits)
                node_mask = torch.where(
                    protected.view(-1, 1),
                    torch.ones_like(learned_node_mask),
                    learned_node_mask,
                )
                edge_mask = torch.sigmoid(edge_logits)
                masked_x = x_sub * node_mask
                masked_out = head(
                    masked_encoder(masked_x, ei_work, et_work)[src_local : src_local + 1]
                )
                masked_logits = _task_logits(masked_out, task)[0]
                prediction_loss = _fixed_class_prediction_loss(
                    masked_logits, pred_class_id
                )
                edge_size, edge_ent, node_size, node_ent = _mask_regularizers(
                    node_mask, edge_mask, protected, config
                )
                loss = prediction_loss + edge_size + edge_ent + node_size + node_ent
                loss.backward()

                if node_logits.grad is None or edge_logits.grad is None:
                    raise RuntimeError("PROPHET could not compute both node and edge gradients")
                node_grad_finite = node_grad_finite and bool(
                    torch.isfinite(node_logits.grad).all().item()
                )
                edge_grad_finite = edge_grad_finite and bool(
                    torch.isfinite(edge_logits.grad).all().item()
                )
                if not node_grad_finite or not edge_grad_finite:
                    raise FloatingPointError("PROPHET produced non-finite mask gradients")

                probability = F.softmax(masked_logits, dim=-1)[pred_class_id]
                trace.append(
                    {
                        "epoch": float(epoch + 1),
                        "prediction_loss": float(prediction_loss.detach().item()),
                        "edge_size_loss": float(edge_size.detach().item()),
                        "edge_entropy_loss": float(edge_ent.detach().item()),
                        "node_size_loss": float(node_size.detach().item()),
                        "node_entropy_loss": float(node_ent.detach().item()),
                        "total_loss": float(loss.detach().item()),
                        "fixed_class_probability": float(probability.detach().item()),
                    }
                )
                optimizer.step()

            with torch.no_grad():
                learned_node_mask = torch.sigmoid(node_logits)
                node_mask = torch.where(
                    protected.view(-1, 1),
                    torch.ones_like(learned_node_mask),
                    learned_node_mask,
                )
                edge_mask = torch.sigmoid(edge_logits)
                reset_masks()
                final_out = head(
                    masked_encoder(x_sub * node_mask, ei_work, et_work)[src_local : src_local + 1]
                )
                final_logits = _task_logits(final_out, task)[0]
                final_probs = F.softmax(final_logits, dim=-1)
                masked_class_id = int(final_logits.argmax().item())
                masked_logit = float(final_logits[pred_class_id].item())
                masked_probability = float(final_probs[pred_class_id].item())
    finally:
        for module, states in zip(modules, requires_grad_states):
            for parameter, state in zip(module.parameters(), states):
                parameter.requires_grad_(state)

    edge_mask_input_order = torch.empty_like(edge_mask)
    edge_mask_input_order[edge_order] = edge_mask
    return ProphetExplanation(
        node_mask=node_mask.detach().reshape(-1),
        edge_mask=edge_mask_input_order.detach(),
        pred_class_id=pred_class_id,
        pred_class_label=pred_class_label or str(pred_class_id),
        original_logit=original_logit,
        original_probability=original_probability,
        masked_logit=masked_logit,
        masked_probability=masked_probability,
        masked_class_id=masked_class_id,
        prediction_agreement=masked_class_id == pred_class_id,
        protected_nodes=valid_protected,
        seed=int(config.seed),
        encoder_parity_max_abs_diff=parity_diff,
        node_grad_finite=node_grad_finite,
        edge_grad_finite=edge_grad_finite,
        trace=trace,
    )
