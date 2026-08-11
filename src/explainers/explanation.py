"""Shared explanation container used by explainers and fidelity evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor


@dataclass
class ExplanationResult:
    """Normalized explanation result for one explainer, pair, and task."""

    explainer: str
    pair_idx: int
    task: str
    src_global: int = -1
    dst_global: int = -1
    src_local: int = -1
    dst_local: int = -1
    node_mask: Optional[Tensor] = None
    edge_mask: Optional[Tensor] = None
    feature_mask: Optional[Tensor] = None
    primary_mask: Optional[str] = None
    nodes_global: Optional[Tensor] = None
    edge_index: Optional[Tensor] = None
    edge_type: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_mask is None and self.edge_mask is None and self.feature_mask is None:
            raise ValueError("ExplanationResult requires node_mask, edge_mask, or feature_mask.")
        if self.primary_mask not in {None, "node", "edge", "feature"}:
            raise ValueError(f"Unknown primary_mask: {self.primary_mask}")
        if self.primary_mask is not None and getattr(self, f"{self.primary_mask}_mask") is None:
            raise ValueError(
                f"primary_mask={self.primary_mask!r} requires a matching mask tensor"
            )

    @property
    def primary_mask_type(self) -> str:
        if self.primary_mask is not None:
            return self.primary_mask
        if self.edge_mask is not None:
            return "edge"
        if self.node_mask is not None:
            return "node"
        return "feature"

    def to_cache_dict(self) -> Dict[str, Any]:
        data = {
            "explainer": self.explainer,
            "pair_idx": int(self.pair_idx),
            "task": self.task,
            "src_global": int(self.src_global),
            "dst_global": int(self.dst_global),
            "src_local": int(self.src_local),
            "dst_local": int(self.dst_local),
            "primary_mask": self.primary_mask,
            "metadata": dict(self.metadata),
        }
        for name in ("node_mask", "edge_mask", "feature_mask", "nodes_global", "edge_index", "edge_type"):
            value = getattr(self, name)
            data[name] = value.detach().cpu() if isinstance(value, Tensor) else value
        return data

    @classmethod
    def from_cache_dict(cls, data: Dict[str, Any]) -> "ExplanationResult":
        return cls(**data)

    def to_device(self, device: torch.device) -> "ExplanationResult":
        for name in ("node_mask", "edge_mask", "feature_mask", "nodes_global", "edge_index", "edge_type"):
            value = getattr(self, name)
            if isinstance(value, Tensor):
                setattr(self, name, value.to(device))
        return self
