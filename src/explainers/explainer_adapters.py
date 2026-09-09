
from __future__ import annotations
import hashlib
import warnings
from collections import deque
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from torch import Tensor

# TODO: FOX is outside the current ready-method pass.



from src.explainers.explanation import ExplanationResult
ExplanationMask = ExplanationResult

_RESOURCE_OTHERC_RAW_KEY = "http://example.org/hasevent_otherC_org_resource"
_RESOURCE_OTHERC_KEY = (
    _RESOURCE_OTHERC_RAW_KEY.replace(":", "_").replace("/", "_").replace(".", "_")
)
_TASK_ALIASES = {"resource": f"otherC_{_RESOURCE_OTHERC_KEY}"}


def _model_task(task: str) -> str:
    return _TASK_ALIASES.get(task, task)


def _fixed_classification_metadata(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    x_sub: Tensor,
    ei_sub: Tensor,
    et_sub: Tensor,
    src_local: int,
    task: str,
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
            "pred_class_label": str(pred_id),
            "pred_score": float(logits[pred_id].item()),
        }
    if task.startswith("otherC_"):
        key = task[len("otherC_") :]
        if key not in out["otherC"]:
            raise KeyError(f"otherC head missing key (sanitized): {key}")
        logits = out["otherC"][key][0]
        pred_id = int(logits.argmax().item())
        return {
            "target_kind": "otherC",
            "target_key": key,
            "pred_class_id": pred_id,
            "pred_class_label": str(pred_id),
            "pred_score": float(logits[pred_id].item()),
        }
    return {}


def _normalize_scores(scores: np.ndarray) -> Optional[np.ndarray]:
    if scores.size == 0 or not np.isfinite(scores).all():
        return None
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        return None
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def _structural_candidate_ids(
    num_nodes: int,
    edge_index: Tensor,
    src_local: int,
    limit: int,
    exclude: Sequence[int] = (),
) -> List[int]:
    """Choose a method-neutral candidate pool from graph structure only."""

    excluded = {int(src_local), *(int(node) for node in exclude)}
    if limit <= 0:
        return []

    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    degree = np.zeros(num_nodes, dtype=np.int64)
    ei = edge_index.detach().cpu().numpy()
    for s_raw, d_raw in zip(ei[0], ei[1]):
        s, d = int(s_raw), int(d_raw)
        if 0 <= s < num_nodes and 0 <= d < num_nodes:
            adj[s].append(d)
            adj[d].append(s)
            degree[s] += 1
            degree[d] += 1

    dist = np.full(num_nodes, np.iinfo(np.int32).max, dtype=np.int64)
    if 0 <= int(src_local) < num_nodes:
        dist[int(src_local)] = 0
        queue = deque([int(src_local)])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == np.iinfo(np.int32).max:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)

    candidates = [node for node in range(num_nodes) if node not in excluded]
    candidates.sort(key=lambda node: (int(dist[node]), -int(degree[node]), int(node)))
    return candidates[: min(limit, len(candidates))]


def _result(
    name: str,
    pair_idx: int,
    task: str,
    src_local: int,
    dst_local: int,
    *,
    node_mask: Optional[Tensor] = None,
    edge_mask: Optional[Tensor] = None,
    feature_mask: Optional[Tensor] = None,
    primary_mask: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> ExplanationResult:
    return ExplanationResult(
        explainer=name,
        pair_idx=pair_idx,
        task=task,
        src_local=src_local,
        dst_local=dst_local,
        node_mask=node_mask,
        edge_mask=edge_mask,
        feature_mask=feature_mask,
        primary_mask=primary_mask,
        metadata=dict(metadata or {}),
    )


def _stable_uint32_seed(seed: int, pair_idx: int, task: str) -> int:
    payload = f"{int(seed)}:{int(pair_idx)}:{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class BaselineAdapter:
    def __init__(self, name: str, seed: int = 0) -> None:
        if name not in {"random", "degree"}:
            raise ValueError(f"Unknown baseline adapter: {name}")
        self.name = name
        self.seed = int(seed)

    def explain(self, encoder: torch.nn.Module, head: torch.nn.Module, x_sub: Tensor, ei_sub: Tensor, et_sub: Tensor, src_local: int, dst_local: int, task: str, y_act: Optional[Tensor], y_time: Optional[Tensor], pair_test_idx: int, device: torch.device, use_model_target: bool = True, protected_nodes: Sequence[int] = ()) -> Optional[ExplanationMask]:
        num_nodes = int(x_sub.size(0))
        if self.name == "random":
            rng = np.random.default_rng(_stable_uint32_seed(self.seed, pair_test_idx, task))
            scores = torch.tensor(rng.random(num_nodes).astype(np.float32), device=device)
            metadata = {"baseline": "random", "seed": self.seed}
        else:
            scores = torch.zeros(num_nodes, dtype=torch.float32, device=device)
            if ei_sub.numel() > 0:
                edge_index = ei_sub.to(device)
                ones = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
                scores.scatter_add_(0, edge_index[0], ones)
                scores.scatter_add_(0, edge_index[1], ones)
            metadata = {"baseline": "degree"}

        return _result(
            self.name,
            pair_test_idx,
            task,
            src_local,
            dst_local,
            node_mask=scores,
            metadata=metadata,
        )


class GradientAdapter:
    name = "gradient"
    def explain(self, encoder: torch.nn.Module, head: torch.nn.Module, x_sub: Tensor, ei_sub: Tensor, et_sub: Tensor, src_local: int, dst_local: int, task: str, y_act: Optional[Tensor], y_time: Optional[Tensor], pair_test_idx: int, device: torch.device, use_model_target: bool = True, protected_nodes: Sequence[int] = ()) -> Optional[ExplanationMask]:
        from src.explainers.explain_gradient import _gradient_saliency
        try:
            _, grad_x_input_abs, pred_meta = _gradient_saliency(
                encoder,
                head,
                x_sub,
                ei_sub,
                et_sub,
                src_local,
                task,
                y_act,
                y_time,
                pair_test_idx,
                device,
                use_model_target=use_model_target,
            )
            scores_np = _normalize_scores(grad_x_input_abs)
            if scores_np is None:
                warnings.warn(f"[GradientAdapter] task={task}: degenerate scores, skipping.")
                return None
            return _result(
                self.name,
                pair_test_idx,
                task,
                src_local,
                dst_local,
                node_mask=torch.tensor(scores_np, dtype=torch.float32, device=device),
                metadata={"target_metadata": pred_meta},
            )
        except Exception as exc:
            warnings.warn(f"[GradientAdapter] failed for task={task}: {exc}")
            return None
class LIMEAdapter:
    name = "lime"
    def __init__(
        self,
        n_perturb: int = 200,
        mask_frac: float = 0.3,
        top_k_candidates: int = 30,
        baseline: str = "mean",
        seed: int = 42,
        global_mean: Optional[Tensor] = None,
    ) -> None:
        if baseline not in {"mean", "zero"}:
            raise ValueError("LIMEAdapter baseline must be 'mean' or 'zero'")
        self.n_perturb = n_perturb
        self.mask_frac = mask_frac
        self.top_k_candidates = top_k_candidates
        self.baseline = baseline
        self.global_mean = global_mean.detach().clone() if global_mean is not None else None
        self.rng = np.random.default_rng(seed)
    def explain(self, encoder: torch.nn.Module, head: torch.nn.Module, x_sub: Tensor, ei_sub: Tensor, et_sub: Tensor, src_local: int, dst_local: int, task: str, y_act: Optional[Tensor], y_time: Optional[Tensor], pair_test_idx: int, device: torch.device, use_model_target: bool = True, protected_nodes: Sequence[int] = ()) -> Optional[ExplanationMask]:
        from src.explainers.explain_lime import (
            _neighbor_mask_graphlime_ridge,
            _structural_graphlime_candidates,
        )
        try:
            model_task = _model_task(task)
            protected = {int(node) for node in protected_nodes}
            feat_ids = _structural_graphlime_candidates(
                ei_sub=ei_sub,
                num_nodes=x_sub.size(0),
                src_local=src_local,
                dst_local=dst_local,
                limit=self.top_k_candidates + len(protected),
            )
            feat_ids = [
                nid for nid in feat_ids if int(nid) not in protected
            ][: self.top_k_candidates]
            if len(feat_ids) < 2:
                warnings.warn(f"[LIMEAdapter] task={task}: only {len(feat_ids)} candidate(s), skipping.")
                return None
            if self.baseline == "mean" and self.global_mean is None:
                raise ValueError("LIMEAdapter requires global_mean for baseline='mean'")
            baseline_row = (
                self.global_mean.to(device=device, dtype=x_sub.dtype)
                if self.baseline == "mean"
                else torch.zeros(x_sub.size(1), device=device)
            )
            pred_meta = (
                _fixed_classification_metadata(
                    encoder,
                    head,
                    x_sub,
                    ei_sub,
                    et_sub,
                    src_local,
                    model_task,
                )
                if use_model_target
                else {}
            )
            coef = _neighbor_mask_graphlime_ridge(
                encoder,
                head,
                x_sub,
                ei_sub,
                et_sub,
                src_local,
                feat_ids,
                model_task,
                y_act,
                y_time,
                pair_test_idx,
                device,
                self.n_perturb,
                self.mask_frac,
                self.rng,
                baseline_row=baseline_row,
                use_model_target=use_model_target,
                pred_meta=pred_meta,
            )
            candidate_scores = np.abs(np.asarray(coef, dtype=np.float32)).reshape(-1)
            if candidate_scores.size != len(feat_ids) or not np.isfinite(candidate_scores).all():
                warnings.warn(f"[LIMEAdapter] task={task}: invalid candidate scores, skipping.")
                return None
            if float(candidate_scores.max()) <= 0.0:
                warnings.warn(f"[LIMEAdapter] task={task}: degenerate scores, skipping.")
                return None
            full_scores = np.full(x_sub.size(0), -np.inf, dtype=np.float32)
            for j, nid in enumerate(feat_ids):
                full_scores[nid] = float(candidate_scores[j])
            return _result(
                self.name,
                pair_test_idx,
                task,
                src_local,
                dst_local,
                node_mask=torch.tensor(full_scores, dtype=torch.float32, device=device),
                metadata={
                    "model_task": model_task,
                    "baseline": "full_graph_mean" if self.baseline == "mean" else "zero",
                    "candidate_count": len(feat_ids),
                    "target_metadata": pred_meta,
                },
            )
        except Exception as exc:
            warnings.warn(f"[LIMEAdapter] failed for task={task}: {exc}")
            return None
class ShapAdapter:
    name = "shap"
    def __init__(
        self,
        shap_candidates: int = 25,
        shap_nsamples: int = 300,
        shap_l1_reg: str = "aic",
        baseline: str = "mean",
        seed: int = 42,
        global_mean: Optional[Tensor] = None,
    ) -> None:
        if baseline not in {"mean", "zero"}:
            raise ValueError("ShapAdapter baseline must be 'mean' or 'zero'")
        self.shap_candidates = shap_candidates
        self.shap_nsamples = shap_nsamples
        self.shap_l1_reg = shap_l1_reg
        self.baseline = baseline
        self.global_mean = global_mean.detach().clone() if global_mean is not None else None
        self.rng = np.random.default_rng(seed)
    def explain(self, encoder: torch.nn.Module, head: torch.nn.Module, x_sub: Tensor, ei_sub: Tensor, et_sub: Tensor, src_local: int, dst_local: int, task: str, y_act: Optional[Tensor], y_time: Optional[Tensor], pair_test_idx: int, device: torch.device, use_model_target: bool = True, protected_nodes: Sequence[int] = ()) -> Optional[ExplanationMask]:
        from src.explainers.explain_shap import (
            _build_predict_fn,
            _run_kernel_shap,
            _structural_shap_candidates,
        )
        try:
            model_task = _model_task(task)
            protected = {int(node) for node in protected_nodes}
            feat_ids_local = _structural_shap_candidates(
                ei_sub=ei_sub,
                num_nodes=x_sub.size(0),
                src_local=src_local,
                dst_local=dst_local,
                limit=self.shap_candidates + len(protected),
            )
            feat_ids_local = [
                nid for nid in feat_ids_local if int(nid) not in protected
            ][: self.shap_candidates]
            if len(feat_ids_local) < 2:
                warnings.warn(f"[ShapAdapter] task={task}: only {len(feat_ids_local)} candidate(s), skipping.")
                return None
            if self.baseline == "mean" and self.global_mean is None:
                raise ValueError("ShapAdapter requires global_mean for baseline='mean'")
            baseline_row = (
                self.global_mean.to(device=device, dtype=x_sub.dtype)
                if self.baseline == "mean"
                else torch.zeros(x_sub.size(1), device=device)
            )
            pred_meta = (
                _fixed_classification_metadata(
                    encoder,
                    head,
                    x_sub,
                    ei_sub,
                    et_sub,
                    src_local,
                    model_task,
                )
                if use_model_target
                else {}
            )
            nsamples = (
                min(2 * len(feat_ids_local) + 256, 1024)
                if str(self.shap_nsamples) == "auto"
                else int(self.shap_nsamples)
            )
            predict_fn = _build_predict_fn(
                encoder=encoder,
                head=head,
                x_sub=x_sub,
                ei_sub=ei_sub,
                et_sub=et_sub,
                src_local=src_local,
                feat_ids_local=feat_ids_local,
                task=model_task,
                y_act=y_act,
                y_time=y_time,
                pair_test_idx=pair_test_idx,
                device=device,
                baseline_row=baseline_row,
                pbar=None,
                use_model_target=use_model_target,
                pred_meta=pred_meta,
            )
            phi = _run_kernel_shap(
                predict_fn, len(feat_ids_local), nsamples, self.shap_l1_reg
            )
            candidate_scores = np.abs(np.asarray(phi, dtype=np.float32)).reshape(-1)
            if candidate_scores.size != len(feat_ids_local) or not np.isfinite(candidate_scores).all():
                warnings.warn(f"[ShapAdapter] task={task}: invalid candidate scores, skipping.")
                return None
            if float(candidate_scores.max()) <= 0.0:
                warnings.warn(f"[ShapAdapter] task={task}: degenerate scores, skipping.")
                return None
            full_scores = np.full(x_sub.size(0), -np.inf, dtype=np.float32)
            for j, local_id in enumerate(feat_ids_local):
                full_scores[local_id] = float(candidate_scores[j])
            return _result(
                self.name,
                pair_test_idx,
                task,
                src_local,
                dst_local,
                node_mask=torch.tensor(full_scores, dtype=torch.float32, device=device),
                metadata={
                    "model_task": model_task,
                    "baseline": "full_graph_mean" if self.baseline == "mean" else "zero",
                    "candidate_count": len(feat_ids_local),
                    "nsamples": nsamples,
                    "target_metadata": pred_meta,
                },
            )
        except Exception as exc:
            warnings.warn(f"[ShapAdapter] failed for task={task}: {exc}")
            return None
class IGAdapter:
    name = "ig"
    def __init__(
        self,
        n_steps: int = 32,
        method: str = "gausslegendre",
        baseline: str = "mean",
    ) -> None:
        self.n_steps = n_steps
        self.method = method
        self.baseline = baseline
    def explain(self, encoder: torch.nn.Module, head: torch.nn.Module, x_sub: Tensor, ei_sub: Tensor, et_sub: Tensor, src_local: int, dst_local: int, task: str, y_act: Optional[Tensor], y_time: Optional[Tensor], pair_test_idx: int, device: torch.device, use_model_target: bool = True, protected_nodes: Sequence[int] = ()) -> Optional[ExplanationMask]:
        from src.explainers.explain_ig import (
            _integrated_gradients,
            _resolve_fixed_model_target,
        )
        try:
            baseline_tensor = (
                x_sub.mean(dim=0).unsqueeze(0).expand_as(x_sub).clone()
                if self.baseline == "mean"
                else torch.zeros_like(x_sub)
            )
            fixed_target = (
                _resolve_fixed_model_target(
                    encoder, head, x_sub, ei_sub, et_sub, src_local, task, {}
                )
                if use_model_target
                else None
            )
            _, ig_abs, delta = _integrated_gradients(
                encoder,
                head,
                x_sub,
                ei_sub,
                et_sub,
                src_local,
                task,
                y_act,
                y_time,
                pair_test_idx,
                device,
                baseline_tensor,
                self.n_steps,
                self.method,
                use_model_target=bool(use_model_target),
                fixed_target=fixed_target,
            )
            scores_np = _normalize_scores(ig_abs)
            if scores_np is None:
                warnings.warn(f"[IGAdapter] task={task}: degenerate scores, skipping.")
                return None
            return _result(
                self.name,
                pair_test_idx,
                task,
                src_local,
                dst_local,
                node_mask=torch.tensor(scores_np, dtype=torch.float32, device=device),
                metadata={
                    "baseline": self.baseline,
                    "n_steps": self.n_steps,
                    "method": self.method,
                    "convergence_delta": float(delta),
                    "target_metadata": fixed_target or {},
                },
            )
        except Exception as exc:
            warnings.warn(f"[IGAdapter] failed for task={task}: {exc}")
            return None

class FOXAdapter:
    name = "fox"
    def __init__(
        self,
        fox_m: int = 5,
        n_perturb: int = 200,
        num_mfs: int = 3,
        epochs: int = 80,
        lr: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.fox_m = fox_m
        self.n_perturb = n_perturb
        self.num_mfs = num_mfs
        self.epochs = epochs
        self.lr = lr
        self.rng = np.random.default_rng(seed)
    def explain(self, encoder: torch.nn.Module, head: torch.nn.Module, x_sub: Tensor, ei_sub: Tensor, et_sub: Tensor, src_local: int, dst_local: int, task: str, y_act: Optional[Tensor], y_time: Optional[Tensor], pair_test_idx: int, device: torch.device, use_model_target: bool = True, protected_nodes: Sequence[int] = ()) -> Optional[ExplanationMask]:
        from src.explainers.explain_fox import _fox_explain
        try:
            feat_ids = _structural_candidate_ids(
                num_nodes=x_sub.size(0),
                edge_index=ei_sub,
                src_local=src_local,
                limit=self.fox_m,
            )
            if len(feat_ids) < 2:
                warnings.warn(f"[FOXAdapter] task={task}: only {len(feat_ids)} candidate(s), skipping.")
                return None
            _, importance = _fox_explain(
                encoder,
                head,
                x_sub,
                ei_sub,
                et_sub,
                src_local,
                feat_ids,
                task,
                y_act,
                y_time,
                pair_test_idx,
                device,
                n_perturb=self.n_perturb,
                num_mfs=self.num_mfs,
                epochs=self.epochs,
                lr=self.lr,
                rng=self.rng,
                use_model_target=use_model_target,
            )
            full_scores = np.zeros(x_sub.size(0), dtype=np.float32)
            for j, nid in enumerate(feat_ids):
                full_scores[nid] = float(importance[j])
            scores_np = _normalize_scores(full_scores)
            if scores_np is None:
                warnings.warn(f"[FOXAdapter] task={task}: degenerate scores, skipping.")
                return None
            return _result(
                self.name,
                pair_test_idx,
                task,
                src_local,
                dst_local,
                node_mask=torch.tensor(scores_np, dtype=torch.float32, device=device),
            )
        except Exception as exc:
            warnings.warn(f"[FOXAdapter] failed for task={task}: {exc}")
            return None
class GNNExplainerAdapter:
    name = "gnnexplainer"

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        edge_size: float = 0.005,
        edge_ent: float = 1.0,
        node_size: float = 1.0,
        node_ent: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.edge_size = edge_size
        self.edge_ent = edge_ent
        self.node_size = node_size
        self.node_ent = node_ent
        self.seed = seed

    def explain(self, encoder: torch.nn.Module, head: torch.nn.Module, x_sub: Tensor, ei_sub: Tensor, et_sub: Tensor, src_local: int, dst_local: int, task: str, y_act: Optional[Tensor], y_time: Optional[Tensor], pair_test_idx: int, device: torch.device, use_model_target: bool = True, protected_nodes: Sequence[int] = ()) -> Optional[ExplanationMask]:
        from src.explainers.gnnexplainer_core import (
            GNNExplainerConfig,
            explain_gnnexplainer_masks,
            gnnexplainer_seed,
        )

        try:
            if not use_model_target:
                raise ValueError("GNNExplainer explains the original model prediction only")
            if x_sub.size(0) == 0 or et_sub.numel() == 0:
                warnings.warn(f"[GNNExplainerAdapter] task={task}: empty subgraph, skipping.")
                return None

            model_task = _model_task(task)
            config = GNNExplainerConfig(
                epochs=self.epochs,
                lr=self.lr,
                edge_size=self.edge_size,
                edge_ent=self.edge_ent,
                node_size=self.node_size,
                node_ent=self.node_ent,
                seed=gnnexplainer_seed(self.seed, pair_test_idx, model_task),
            )
            explanation = explain_gnnexplainer_masks(
                encoder=encoder,
                head=head,
                x_sub=x_sub,
                ei_sub=ei_sub.to(device),
                et_sub=et_sub.to(device),
                src_local=src_local,
                task=model_task,
                protected_nodes=protected_nodes,
                config=config,
            )

            node_mask = explanation.node_mask.detach().to(device).float().reshape(-1)
            edge_mask = explanation.edge_mask.detach().to(device).float().reshape(-1)
            if node_mask.numel() != x_sub.size(0):
                raise ValueError(
                    f"node mask length {node_mask.numel()} does not match nodes={x_sub.size(0)}"
                )
            if edge_mask.numel() != et_sub.numel():
                raise ValueError(
                    f"edge mask length {edge_mask.numel()} does not match edges={et_sub.numel()}"
                )
            if not torch.isfinite(node_mask).all() or not torch.isfinite(edge_mask).all():
                warnings.warn(f"[GNNExplainerAdapter] task={task}: non-finite masks, skipping.")
                return None

            protected = {int(node) for node in protected_nodes}
            selectable = torch.tensor(
                [node for node in range(node_mask.numel()) if node not in protected],
                dtype=torch.long,
                device=device,
            )
            if selectable.numel() == 0:
                warnings.warn(
                    f"[GNNExplainerAdapter] task={task}: all nodes are protected, skipping."
                )
                return None
            selectable_scores = node_mask[selectable]
            if float((selectable_scores.max() - selectable_scores.min()).item()) < 1e-8:
                warnings.warn(
                    f"[GNNExplainerAdapter] task={task}: degenerate node scores, skipping."
                )
                return None

            return _result(
                self.name,
                pair_test_idx,
                model_task,
                src_local,
                dst_local,
                node_mask=node_mask,
                edge_mask=edge_mask,
                primary_mask="node",
                metadata={
                    "target_semantics": "original_model_predicted_class",
                    "pred_class_id": explanation.pred_class_id,
                    "pred_class_label": explanation.pred_class_label,
                    "original_logit": explanation.original_logit,
                    "original_probability": explanation.original_probability,
                    "masked_logit": explanation.masked_logit,
                    "masked_probability": explanation.masked_probability,
                    "masked_class_id": explanation.masked_class_id,
                    "prediction_agreement": explanation.prediction_agreement,
                    "protected_nodes": list(explanation.protected_nodes),
                    "seed": explanation.seed,
                    "epochs": config.epochs,
                    "lr": config.lr,
                    "edge_size": config.edge_size,
                    "edge_ent": config.edge_ent,
                    "node_size": config.node_size,
                    "node_ent": config.node_ent,
                    "config": {
                        "epochs": config.epochs,
                        "lr": config.lr,
                        "edge_size": config.edge_size,
                        "edge_ent": config.edge_ent,
                        "node_size": config.node_size,
                        "node_ent": config.node_ent,
                        "seed": config.seed,
                        "parity_atol": config.parity_atol,
                        "parity_rtol": config.parity_rtol,
                    },
                    "encoder_parity_max_abs_diff": explanation.encoder_parity_max_abs_diff,
                    "all_one_edge_parity_max_abs_diff": (
                        explanation.all_one_edge_parity_max_abs_diff
                    ),
                    "pyg_version": explanation.pyg_version,
                },
            )
        except Exception as exc:
            warnings.warn(f"[GNNExplainerAdapter] failed for task={task}: {exc}")
            return None


class PGMExplainerAdapter:
    name = "pgmexplainer"

    def __init__(
        self,
        global_mean: Tensor,
        num_samples: int = 1000,
        perturb_probability: float = 0.5,
        response_top_fraction: float = 0.125,
        significance_threshold: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.global_mean = global_mean.detach().clone()
        self.num_samples = num_samples
        self.perturb_probability = perturb_probability
        self.response_top_fraction = response_top_fraction
        self.significance_threshold = significance_threshold
        self.seed = seed

    def explain(self, encoder: torch.nn.Module, head: torch.nn.Module, x_sub: Tensor, ei_sub: Tensor, et_sub: Tensor, src_local: int, dst_local: int, task: str, y_act: Optional[Tensor], y_time: Optional[Tensor], pair_test_idx: int, device: torch.device, use_model_target: bool = True, protected_nodes: Sequence[int] = ()) -> Optional[ExplanationMask]:
        from src.explainers.pgmexplainer_core import (
            PGMExplainerConfig,
            explain_pgmexplainer_nodes,
            pgmexplainer_seed,
        )
        try:
            if not use_model_target:
                raise ValueError("PGMExplainer explains the original model prediction only")
            if x_sub.size(0) == 0 or et_sub.numel() == 0:
                warnings.warn(f"[PGMExplainerAdapter] task={task}: empty subgraph, skipping.")
                return None
            model_task = _model_task(task)
            config = PGMExplainerConfig(
                num_samples=self.num_samples,
                perturb_probability=self.perturb_probability,
                response_top_fraction=self.response_top_fraction,
                significance_threshold=self.significance_threshold,
                seed=pgmexplainer_seed(self.seed, pair_test_idx, model_task),
            )
            explanation = explain_pgmexplainer_nodes(
                encoder=encoder,
                head=head,
                x_sub=x_sub,
                edge_index=ei_sub.to(device),
                edge_type=et_sub.to(device),
                src_local=src_local,
                task=model_task,
                protected_nodes=protected_nodes,
                global_mean=self.global_mean.to(device=device, dtype=x_sub.dtype),
                config=config,
            )
            node_mask = explanation.node_mask.detach().to(device).float().reshape(-1)
            if node_mask.numel() != x_sub.size(0) or not torch.isfinite(node_mask).all():
                warnings.warn(f"[PGMExplainerAdapter] task={task}: invalid node mask, skipping.")
                return None
            protected = {int(node) for node in protected_nodes}
            selectable = torch.tensor(
                [node for node in range(node_mask.numel()) if node not in protected],
                dtype=torch.long,
                device=device,
            )
            if selectable.numel() == 0:
                warnings.warn(f"[PGMExplainerAdapter] task={task}: all nodes are protected, skipping.")
                return None
            selectable_scores = node_mask[selectable]
            if float((selectable_scores.max() - selectable_scores.min()).item()) < 1e-12:
                warnings.warn(f"[PGMExplainerAdapter] task={task}: constant node scores, skipping.")
                return None

            p_values = [
                float(value) if np.isfinite(value) else None
                for value in explanation.p_values.detach().cpu().tolist()
            ]
            significant_nodes = torch.nonzero(
                explanation.significant_mask, as_tuple=False
            ).reshape(-1).detach().cpu().tolist()
            return _result(
                self.name,
                pair_test_idx,
                model_task,
                src_local,
                dst_local,
                node_mask=node_mask,
                primary_mask="node",
                metadata={
                    "target_semantics": "original_model_predicted_class_at_src_local",
                    "pred_class_id": explanation.pred_class_id,
                    "pred_class_label": explanation.pred_class_label,
                    "original_logit": explanation.original_logit,
                    "original_probability": explanation.original_probability,
                    "p_values": p_values,
                    "significant_local_ids": [int(node) for node in significant_nodes],
                    "prediction_changed_count": explanation.prediction_changed_count,
                    "prediction_unchanged_count": explanation.prediction_unchanged_count,
                    "protected_nodes": list(explanation.protected_nodes),
                    "seed": explanation.seed,
                    "baseline": "full_graph_mean",
                    "response_mode": "top_probability_drop_fraction",
                    "config": {
                        "num_samples": config.num_samples,
                        "perturb_probability": config.perturb_probability,
                        "response_top_fraction": config.response_top_fraction,
                        "significance_threshold": config.significance_threshold,
                        "seed": config.seed,
                    },
                },
            )
        except Exception as exc:
            warnings.warn(f"[PGMExplainerAdapter] failed for task={task}: {exc}")
            return None


class ProphetAdapter:
    name = "prophet"

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        edge_size_reg: float = 0.005,
        edge_ent_reg: float = 1.0,
        node_size_reg: float = 1.0,
        node_ent_reg: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.edge_size_reg = edge_size_reg
        self.edge_ent_reg = edge_ent_reg
        self.node_size_reg = node_size_reg
        self.node_ent_reg = node_ent_reg
        self.seed = seed

    def explain(self, encoder: torch.nn.Module, head: torch.nn.Module, x_sub: Tensor, ei_sub: Tensor, et_sub: Tensor, src_local: int, dst_local: int, task: str, y_act: Optional[Tensor], y_time: Optional[Tensor], pair_test_idx: int, device: torch.device, use_model_target: bool = True, protected_nodes: Sequence[int] = ()) -> Optional[ExplanationMask]:
        from src.explainers.prophet_core import (
            ProphetConfig,
            explain_prophet_masks,
            prophet_seed,
        )
        try:
            if not use_model_target:
                raise ValueError("PROPHET explains the original model prediction only")
            if x_sub.size(0) == 0 or et_sub.size(0) == 0:
                warnings.warn(f"[ProphetAdapter] task={task}: empty subgraph, skipping.")
                return None
            model_task = _model_task(task)
            config = ProphetConfig(
                epochs=self.epochs,
                lr=self.lr,
                edge_size_reg=self.edge_size_reg,
                edge_ent_reg=self.edge_ent_reg,
                node_size_reg=self.node_size_reg,
                node_ent_reg=self.node_ent_reg,
                seed=prophet_seed(self.seed, pair_test_idx, model_task),
            )
            explanation = explain_prophet_masks(
                encoder=encoder,
                head=head,
                x_sub=x_sub,
                ei_sub=ei_sub.to(device),
                et_sub=et_sub.to(device),
                src_local=src_local,
                task=model_task,
                protected_nodes=protected_nodes,
                config=config,
            )

            node_mask = explanation.node_mask.detach().to(device).float().reshape(-1)
            edge_mask = explanation.edge_mask.detach().to(device).float().reshape(-1)
            if node_mask.numel() != x_sub.size(0):
                raise ValueError(
                    f"node mask length {node_mask.numel()} does not match nodes={x_sub.size(0)}"
                )
            if edge_mask.numel() != et_sub.size(0):
                raise ValueError(
                    f"edge mask length {edge_mask.numel()} does not match edges={et_sub.size(0)}"
                )
            if not torch.isfinite(node_mask).all() or not torch.isfinite(edge_mask).all():
                warnings.warn(f"[ProphetAdapter] task={task}: non-finite masks, skipping.")
                return None
            protected = {int(node) for node in protected_nodes}
            selectable = torch.tensor(
                [node for node in range(node_mask.numel()) if node not in protected],
                dtype=torch.long,
                device=device,
            )
            if selectable.numel() == 0:
                warnings.warn(f"[ProphetAdapter] task={task}: all nodes are protected, skipping.")
                return None
            selectable_scores = node_mask[selectable]
            if float((selectable_scores.max() - selectable_scores.min()).item()) < 1e-8:
                warnings.warn(f"[ProphetAdapter] task={task}: degenerate node scores, skipping.")
                return None
            return _result(
                self.name,
                pair_test_idx,
                model_task,
                src_local,
                dst_local,
                node_mask=node_mask,
                edge_mask=edge_mask,
                primary_mask="node",
                metadata={
                    "target_semantics": "original_model_predicted_class",
                    "pred_class_id": explanation.pred_class_id,
                    "pred_class_label": explanation.pred_class_label,
                    "original_logit": explanation.original_logit,
                    "original_probability": explanation.original_probability,
                    "masked_logit": explanation.masked_logit,
                    "masked_probability": explanation.masked_probability,
                    "masked_class_id": explanation.masked_class_id,
                    "prediction_agreement": explanation.prediction_agreement,
                    "protected_nodes": list(explanation.protected_nodes),
                    "seed": explanation.seed,
                    "epochs": config.epochs,
                    "lr": config.lr,
                    "edge_size_reg": config.edge_size_reg,
                    "edge_ent_reg": config.edge_ent_reg,
                    "node_size_reg": config.node_size_reg,
                    "node_ent_reg": config.node_ent_reg,
                    "config": {
                        "epochs": config.epochs,
                        "lr": config.lr,
                        "edge_size_reg": config.edge_size_reg,
                        "edge_ent_reg": config.edge_ent_reg,
                        "node_size_reg": config.node_size_reg,
                        "node_ent_reg": config.node_ent_reg,
                        "seed": config.seed,
                        "eps": config.eps,
                        "parity_atol": config.parity_atol,
                        "parity_rtol": config.parity_rtol,
                    },
                    "encoder_parity_max_abs_diff": explanation.encoder_parity_max_abs_diff,
                    "node_grad_finite": explanation.node_grad_finite,
                    "edge_grad_finite": explanation.edge_grad_finite,
                },
            )
        except Exception as exc:
            warnings.warn(f"[ProphetAdapter] failed for task={task}: {exc}")
            return None
