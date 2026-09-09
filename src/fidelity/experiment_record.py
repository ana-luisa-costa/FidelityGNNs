"""Experiment manifest and selected-pair recording."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def write_selected_pairs(path: Path, pair_indices: Sequence[int], test_pairs: Sequence[Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_idx", "src_id", "dst_id"])
        writer.writeheader()
        for pair_idx in pair_indices:
            src_id, dst_id = test_pairs[pair_idx]
            writer.writerow({"pair_idx": pair_idx, "src_id": src_id, "dst_id": dst_id})


def write_manifest(
    path: Path,
    *,
    args: Any,
    config_path: Any,
    config_defaults: Dict[str, Any],
    top_k_values: Iterable[int],
    cls_tasks: Sequence[str],
    context_payload: Dict[str, Any],
    workspace: Path,
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "args": _jsonable(vars(args)),
        "config_path": _jsonable(config_path),
        "config_defaults": _jsonable(config_defaults),
        "seed": getattr(args, "seed", None),
        "top_k_values": list(top_k_values),
        "tasks": list(cls_tasks),
        "task_aliases": context_payload.get("task_aliases", {}),
        "skipped_tasks": context_payload.get("skipped_tasks", []),
        "heterogeneity_mode": context_payload.get("heterogeneity_mode"),
        "selected_pair_indices": list(context_payload.get("pair_indices", [])),
        "input_fingerprints": context_payload.get("fingerprints", {}),
        "relation_hash": context_payload.get("relation_hash"),
        "num_rel": context_payload.get("num_rel"),
        "mean_baseline_scope": context_payload.get("mean_baseline_scope"),
        "probability_fidelity_definition": context_payload.get(
            "probability_fidelity_definition"
        ),
        "torch_version": torch.__version__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
