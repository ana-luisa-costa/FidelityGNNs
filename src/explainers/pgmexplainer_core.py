"""Node-level PGMExplainer for fixed R-GCN event-pair predictions."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from pgmpy.estimators.CITests import chi_square

from src.train import _sanitize


_RESOURCE_OTHERC_RAW_KEY = "http://example.org/hasevent_otherC_org_resource"
_TASK_ALIASES = {"resource": f"otherC_{_sanitize(_RESOURCE_OTHERC_RAW_KEY)}"}


@dataclass(frozen=True)
class PGMExplainerConfig:
    num_samples: int = 1000
    perturb_probability: float = 0.5
    response_top_fraction: float = 0.125
    significance_threshold: float = 0.05
    seed: int = 42


@dataclass
class PGMExplanation:
    node_mask: Tensor
    p_values: Tensor
    chi_square_statistics: Tensor
    degrees_of_freedom: Tensor
    significant_mask: Tensor
    perturbation_counts: Tensor
    pred_class_id: int
    pred_class_label: str
    original_logit: float
    original_probability: float
    prediction_changed_count: int
    prediction_unchanged_count: int
    protected_nodes: Tuple[int, ...]
    selectable_nodes: Tuple[int, ...]
    seed: int


class PGMExplanationError(RuntimeError):
    """An explanation that cannot provide a defensible node ranking."""

    def __init__(self, code: str, message: str, **diagnostics: object) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics


def validate_classification_task(task: str) -> str:
    normalized = _TASK_ALIASES.get(task, task)
    if normalized == "activity" or normalized.startswith("otherC_"):
        return normalized
    raise ValueError(
        "PGMExplainer supports classification tasks only: "
        f"'activity' and 'otherC_*'. Got: {task!r}"
    )


def pgmexplainer_seed(global_seed: int, pair_idx: int, task: str) -> int:
    payload = f"{int(global_seed)}|{int(pair_idx)}|{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _task_logits(output: Dict[str, object], task: str) -> Tensor:
    if task == "activity":
        return output["act"]  # type: ignore[return-value]
    key = task[len("otherC_") :]
    otherc = output["otherC"]
    if key not in otherc:  # type: ignore[operator]
        raise KeyError(f"otherC head missing key {key!r}")
    return otherc[key]  # type: ignore[index,return-value]


def _validate_inputs(
    x_sub: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    src_local: int,
    global_mean: Tensor,
    config: PGMExplainerConfig,
) -> None:
    if config.num_samples < 2:
        raise ValueError("PGMExplainer num_samples must be at least 2")
    if not 0.0 < config.perturb_probability < 1.0:
        raise ValueError("PGMExplainer perturb_probability must be in (0, 1)")
    if not 0.0 < config.response_top_fraction < 1.0:
        raise ValueError("PGMExplainer response_top_fraction must be in (0, 1)")
    if not 0.0 < config.significance_threshold < 1.0:
        raise ValueError("PGMExplainer significance_threshold must be in (0, 1)")
    if x_sub.ndim != 2 or x_sub.size(0) == 0:
        raise ValueError("PGMExplainer requires a non-empty [N, F] node matrix")
    if edge_index.shape != (2, edge_type.numel()):
        raise ValueError("edge_index and edge_type shapes do not match")
    if not 0 <= int(src_local) < x_sub.size(0):
        raise ValueError(f"src_local is outside the subgraph: {src_local}")
    if global_mean.reshape(-1).numel() != x_sub.size(1):
        raise ValueError("global_mean feature dimension does not match x_sub")
    if not torch.isfinite(global_mean).all():
        raise ValueError("global_mean contains non-finite values")


def explain_pgmexplainer_nodes(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    x_sub: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    src_local: int,
    task: str,
    protected_nodes: Sequence[int],
    global_mean: Tensor,
    config: PGMExplainerConfig,
    *,
    pred_class_label: str | None = None,
) -> PGMExplanation:
    """Explain one fixed predicted class using node perturbation dependence."""

    task = validate_classification_task(task)
    _validate_inputs(x_sub, edge_index, edge_type, src_local, global_mean, config)
    device = x_sub.device
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    baseline = global_mean.detach().to(device=device, dtype=x_sub.dtype).reshape(-1)
    protected = tuple(
        sorted({int(node) for node in protected_nodes if 0 <= int(node) < x_sub.size(0)})
    )
    protected_set = set(protected)
    selectable = tuple(node for node in range(x_sub.size(0)) if node not in protected_set)
    if not selectable:
        raise PGMExplanationError("no_selectable_nodes", "All subgraph nodes are protected")

    encoder_training, head_training = encoder.training, head.training
    encoder.eval()
    head.eval()
    rng = np.random.default_rng(int(config.seed))

    def logits_for(x: Tensor) -> Tensor:
        embeddings = encoder(x, edge_index, edge_type)
        return _task_logits(head(embeddings[src_local : src_local + 1]), task)[0]

    try:
        with torch.no_grad():
            original_logits = logits_for(x_sub)
            pred_class_id = int(original_logits.argmax().item())
            original_probability = float(
                F.softmax(original_logits, dim=-1)[pred_class_id].item()
            )
            original_logit = float(original_logits[pred_class_id].item())

            perturbations = rng.binomial(
                1,
                config.perturb_probability,
                size=(config.num_samples, len(selectable)),
            ).astype(np.int8)
            probability_drops = np.zeros(config.num_samples, dtype=np.float64)

            for sample_idx, sample in enumerate(perturbations):
                selected_columns = np.flatnonzero(sample)
                x_perturbed = x_sub.clone()
                if selected_columns.size:
                    selected_nodes = [selectable[int(column)] for column in selected_columns]
                    x_perturbed[selected_nodes] = baseline
                perturbed_logits = logits_for(x_perturbed)
                perturbed_probability = float(
                    F.softmax(perturbed_logits, dim=-1)[pred_class_id].item()
                )
                probability_drops[sample_idx] = original_probability - perturbed_probability
    except PGMExplanationError:
        raise
    except Exception as exc:
        raise PGMExplanationError("model_inference_failed", str(exc)) from exc
    finally:
        encoder.train(encoder_training)
        head.train(head_training)

    changed_count = min(
        config.num_samples - 1,
        max(1, int(config.num_samples * config.response_top_fraction)),
    )
    changed = np.zeros(config.num_samples, dtype=np.int8)
    top_indices = np.argsort(-probability_drops, kind="stable")[:changed_count]
    changed[top_indices] = 1
    unchanged_count = int(config.num_samples - changed_count)
    response_diagnostics = {
        "prediction_changed_count": changed_count,
        "prediction_unchanged_count": unchanged_count,
    }
    p_values = torch.full((x_sub.size(0),), float("nan"), device=device)
    statistics = torch.full_like(p_values, float("nan"))
    dof = torch.full_like(p_values, float("nan"))
    significance = torch.zeros(x_sub.size(0), dtype=torch.bool, device=device)
    perturbation_counts = torch.zeros(x_sub.size(0), dtype=torch.long, device=device)
    frame = pd.DataFrame(perturbations, columns=list(selectable))
    response_column = "prediction_changed"
    frame[response_column] = changed

    for column_idx, node in enumerate(selectable):
        perturbation_counts[node] = int(perturbations[:, column_idx].sum())
        if frame[node].nunique(dropna=False) < 2:
            raise PGMExplanationError(
                "invalid_perturbation_column",
                f"Node {node} has a constant perturbation indicator",
                node=node,
                **response_diagnostics,
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                statistic, p_value, node_dof = chi_square(
                    node,
                    response_column,
                    [],
                    frame,
                    boolean=False,
                    significance_level=config.significance_threshold,
                )
        except Exception as exc:
            raise PGMExplanationError(
                "chi_square_failed",
                f"Chi-square test failed for node {node}: {exc}",
                node=node,
                **response_diagnostics,
            ) from exc
        values = np.asarray([statistic, p_value, node_dof], dtype=float)
        if not np.isfinite(values).all() or not 0.0 <= float(p_value) <= 1.0:
            raise PGMExplanationError(
                "invalid_chi_square_result",
                f"Chi-square test returned invalid values for node {node}",
                node=node,
                **response_diagnostics,
            )
        statistics[node] = float(statistic)
        p_values[node] = float(p_value)
        dof[node] = float(node_dof)
        significance[node] = float(p_value) < config.significance_threshold

    node_mask = torch.ones(x_sub.size(0), dtype=x_sub.dtype, device=device)
    selectable_tensor = torch.tensor(selectable, dtype=torch.long, device=device)
    node_mask[selectable_tensor] = 1.0 - p_values[selectable_tensor]
    selectable_scores = node_mask[selectable_tensor]
    if not torch.isfinite(selectable_scores).all():
        raise PGMExplanationError(
            "nonfinite_importance",
            "Selectable node importance contains non-finite values",
            **response_diagnostics,
        )
    if float((selectable_scores.max() - selectable_scores.min()).item()) < 1e-12:
        raise PGMExplanationError(
            "constant_importance",
            "Selectable node importance is constant",
            **response_diagnostics,
        )

    return PGMExplanation(
        node_mask=node_mask,
        p_values=p_values,
        chi_square_statistics=statistics,
        degrees_of_freedom=dof,
        significant_mask=significance,
        perturbation_counts=perturbation_counts,
        pred_class_id=pred_class_id,
        pred_class_label=pred_class_label or str(pred_class_id),
        original_logit=original_logit,
        original_probability=original_probability,
        prediction_changed_count=changed_count,
        prediction_unchanged_count=unchanged_count,
        protected_nodes=protected,
        selectable_nodes=selectable,
        seed=int(config.seed),
    )
