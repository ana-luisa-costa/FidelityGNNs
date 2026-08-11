"""Stock PyG GNNExplainer integration for the basis-decomposed R-GCN."""

from __future__ import annotations

import contextlib
import copy
import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import RGCNConv

from src.train import MultiTaskHead, RGCNEncoder, _sanitize


_RESOURCE_OTHERC_RAW_KEY = "http://example.org/hasevent_otherC_org_resource"
_TASK_ALIASES = {"resource": f"otherC_{_sanitize(_RESOURCE_OTHERC_RAW_KEY)}"}


@dataclass(frozen=True)
class GNNExplainerConfig:
    epochs: int = 100
    lr: float = 0.01
    edge_size: float = 0.005
    edge_ent: float = 1.0
    node_size: float = 1.0
    node_ent: float = 0.1
    seed: int = 42
    parity_atol: float = 1e-5
    parity_rtol: float = 1e-4


@dataclass
class GNNExplainerExplanation:
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
    all_one_edge_parity_max_abs_diff: float
    pyg_version: str


class UnusableGNNExplanation(RuntimeError):
    """Raised when PyG cannot produce a rankable node mask."""


def validate_classification_task(task: str) -> str:
    normalized = _TASK_ALIASES.get(task, task)
    if normalized == "activity" or normalized.startswith("otherC_"):
        return normalized
    raise ValueError(
        "GNNExplainer supports classification tasks only: "
        f"'activity' and 'otherC_*'. Got: {task!r}"
    )


def gnnexplainer_seed(global_seed: int, pair_idx: int, task: str) -> int:
    payload = f"{int(global_seed)}|{int(pair_idx)}|{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _task_logits(out: Dict[str, object], task: str) -> Tensor:
    if task == "activity":
        return out["act"]  # type: ignore[return-value]
    key = task[len("otherC_") :]
    otherc = out["otherC"]
    if key not in otherc:  # type: ignore[operator]
        raise KeyError(f"otherC head missing key {key!r}")
    return otherc[key]  # type: ignore[index,return-value]


def _canonical_edge_order(edge_index: Tensor, edge_type: Tensor, num_nodes: int) -> Tensor:
    relation_span = max(int(edge_type.max().item()) + 1, 1)
    keys = (
        (edge_index[0].long() * max(int(num_nodes), 1) + edge_index[1].long())
        * relation_span
        + edge_type.long()
    )
    return torch.argsort(keys, stable=True)


class RGCNNodeClassificationWrapper(torch.nn.Module):
    """Return per-node class logits while restoring mandatory node context."""

    def __init__(
        self,
        encoder: RGCNEncoder,
        head: MultiTaskHead,
        task: str,
        original_x: Tensor,
        protected_nodes: Sequence[int],
        reset_relation_state: Callable[[], None],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.task = task
        self.protected_nodes = tuple(int(node) for node in protected_nodes)
        self.reset_relation_state = reset_relation_state
        self.register_buffer("original_x", original_x.detach().clone())

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        self.reset_relation_state()
        if self.protected_nodes:
            x = x.clone()
            x[list(self.protected_nodes)] = self.original_x[list(self.protected_nodes)]
        return _task_logits(self.head(self.encoder(x, edge_index, edge_type)), self.task)


def _remove_registered_edge_mask(module: RGCNConv) -> Tensor | None:
    if "_edge_mask" not in module._parameters:
        return getattr(module, "_edge_mask", None)
    mask = module._parameters.pop("_edge_mask")
    object.__setattr__(module, "_edge_mask", mask)
    return mask


@contextlib.contextmanager
def _relation_edge_masking(
    encoder: RGCNEncoder,
    edge_type: Tensor,
) -> Iterator[Callable[[], None]]:
    """Slice PyG's full edge mask for each relation-specific propagation."""

    edge_count = int(edge_type.numel())
    states = []
    handles = []
    convolutions = [module for module in encoder.modules() if isinstance(module, RGCNConv)]

    for convolution in convolutions:
        state: Dict[str, object] = {"relation": 0, "full_mask": None}
        states.append(state)

        def pre_hook(module, _inputs, *, state=state):
            relation = int(state["relation"])
            current = _remove_registered_edge_mask(module)
            if current is not None and current.numel() == edge_count:
                state["full_mask"] = current
            full_mask = state["full_mask"]
            if module.explain and full_mask is not None:
                module._edge_mask = full_mask[edge_type == relation]
            state["relation"] = relation + 1

        handles.append(convolution.register_propagate_forward_pre_hook(pre_hook))

    def reset() -> None:
        for state in states:
            state["relation"] = 0

    try:
        yield reset
    finally:
        for handle in handles:
            handle.remove()
        clear_masks(encoder)
        for convolution in convolutions:
            convolution._parameters.pop("_edge_mask", None)
            object.__setattr__(convolution, "_edge_mask", None)


def _max_parity_diff(expected: Tensor, actual: Tensor, config: GNNExplainerConfig) -> float:
    diff = float((expected - actual).abs().max().item())
    if not torch.allclose(
        expected,
        actual,
        atol=config.parity_atol,
        rtol=config.parity_rtol,
    ):
        raise RuntimeError(f"R-GCN parity check failed: max_abs_diff={diff:.6g}")
    return diff


def _build_explainer(
    model: torch.nn.Module,
    config: GNNExplainerConfig,
) -> Explainer:
    

    ## GNN Explainer Algorithm
    algorithm = GNNExplainer(
        epochs=config.epochs,
        lr=config.lr,
        edge_size=config.edge_size,
        edge_ent=config.edge_ent,
        node_feat_size=config.node_size,
        node_feat_ent=config.node_ent,
    )
    return Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )


def explain_gnnexplainer_masks(
    encoder: RGCNEncoder,
    head: MultiTaskHead,
    x_sub: Tensor,
    ei_sub: Tensor,
    et_sub: Tensor,
    src_local: int,
    task: str,
    protected_nodes: Sequence[int],
    config: GNNExplainerConfig,
    *,
    pred_class_label: str | None = None,
) -> GNNExplainerExplanation:
    """Learn PyG node and edge masks for one fixed model prediction."""

    task = validate_classification_task(task)
    if config.epochs <= 0:
        raise ValueError("GNNExplainer epochs must be positive")
    if x_sub.size(0) == 0 or et_sub.numel() == 0:
        raise UnusableGNNExplanation("GNNExplainer requires a non-empty subgraph")
    if not 0 <= int(src_local) < x_sub.size(0):
        raise ValueError(f"src_local is outside the subgraph: {src_local}")

    device = x_sub.device
    ei_input, et_input = ei_sub.to(device), et_sub.to(device)
    edge_order = _canonical_edge_order(ei_input, et_input, x_sub.size(0))
    ei_work, et_work = ei_input[:, edge_order], et_input[edge_order]
    valid_protected = tuple(
        sorted({int(node) for node in protected_nodes if 0 <= int(node) < x_sub.size(0)})
    )

    encoder.eval()
    head.eval()
    copied_encoder = copy.deepcopy(encoder).to(device).eval()
    copied_head = copy.deepcopy(head).to(device).eval()
    for parameter in (*copied_encoder.parameters(), *copied_head.parameters()):
        parameter.requires_grad_(False)

    with torch.no_grad():
        expected_embeddings = encoder(x_sub, ei_work, et_work)
        copied_embeddings = copied_encoder(x_sub, ei_work, et_work)
    encoder_parity = _max_parity_diff(expected_embeddings, copied_embeddings, config)

    with _relation_edge_masking(copied_encoder, et_work) as reset_relation_state:
        wrapper = RGCNNodeClassificationWrapper(
            copied_encoder,
            copied_head,
            task,
            x_sub,
            valid_protected,
            reset_relation_state,
        ).to(device)
        wrapper.eval()

        with torch.no_grad():
            original_logits = wrapper(x_sub, ei_work, et_work)[src_local]
            pred_class_id = int(original_logits.argmax().item())
            original_probs = F.softmax(original_logits, dim=-1)
            original_logit = float(original_logits[pred_class_id].item())
            original_probability = float(original_probs[pred_class_id].item())

            set_masks(wrapper, torch.ones(et_work.numel(), device=device), ei_work, apply_sigmoid=False)
            all_one_logits = wrapper(x_sub, ei_work, et_work)[src_local]
            clear_masks(wrapper)
        all_one_parity = _max_parity_diff(original_logits, all_one_logits, config)

        explainer = _build_explainer(wrapper, config)
        cuda_devices = [] if device.type != "cuda" else [device.index or 0]
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(int(config.seed))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(config.seed))
                explanation = explainer(
                    x=x_sub,
                    edge_index=ei_work,
                    edge_type=et_work,
                    index=src_local,
                )

            if explanation.node_mask is None or explanation.edge_mask is None:
                raise UnusableGNNExplanation("PyG did not return both masks")
            node_mask = explanation.node_mask.detach().reshape(-1).to(device)
            edge_mask = explanation.edge_mask.detach().reshape(-1).to(device)
            if node_mask.numel() != x_sub.size(0) or edge_mask.numel() != et_work.numel():
                raise UnusableGNNExplanation("PyG returned incorrectly shaped masks")
            if not torch.isfinite(node_mask).all() or not torch.isfinite(edge_mask).all():
                raise UnusableGNNExplanation("PyG returned non-finite masks")

            inferred_target = int(explanation.target.reshape(-1)[src_local].item())
            if inferred_target != pred_class_id:
                raise RuntimeError(
                    "PyG model target differs from the original prediction: "
                    f"{inferred_target} != {pred_class_id}"
                )

            if valid_protected:
                node_mask[list(valid_protected)] = 1.0
            selectable = [
                node for node in range(node_mask.numel()) if node not in set(valid_protected)
            ]
            if not selectable:
                raise UnusableGNNExplanation("All subgraph nodes are protected")
            selectable_scores = node_mask[selectable]
            if float((selectable_scores.max() - selectable_scores.min()).item()) < 1e-8:
                raise UnusableGNNExplanation("Selectable node mask is constant")

            set_masks(wrapper, edge_mask, ei_work, apply_sigmoid=False)
            with torch.no_grad():
                masked_logits = wrapper(
                    x_sub * node_mask.view(-1, 1), ei_work, et_work
                )[src_local]
            clear_masks(wrapper)
            masked_probs = F.softmax(masked_logits, dim=-1)
            masked_class_id = int(masked_logits.argmax().item())
            masked_logit = float(masked_logits[pred_class_id].item())
            masked_probability = float(masked_probs[pred_class_id].item())
        finally:
            clear_masks(wrapper)

    edge_mask_input_order = torch.empty_like(edge_mask)
    edge_mask_input_order[edge_order] = edge_mask
    return GNNExplainerExplanation(
        node_mask=node_mask,
        edge_mask=edge_mask_input_order,
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
        encoder_parity_max_abs_diff=encoder_parity,
        all_one_edge_parity_max_abs_diff=all_one_parity,
        pyg_version=torch_geometric.__version__,
    )
