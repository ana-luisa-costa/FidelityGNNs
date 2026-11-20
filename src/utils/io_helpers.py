
import csv
import json
from typing import Dict, Tuple, Optional

import numpy as np
import torch

from src.utils.uri_helpers import parse_ts


def load_entity2id(path: str) -> Dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(path: str) -> torch.Tensor:
    return torch.tensor(np.load(path), dtype=torch.float)


def load_temporal_boundaries(split_path: str) -> Tuple[float, float]:
    with open(split_path, encoding="utf-8") as f:
        d = json.load(f)
    return d["t1"], d["t2"]


def load_temporal_boundaries_from_csv(csv_path: str, train_ratio: float = 0.8,
                                      val_ratio: float = 0.1) -> Tuple[float, float]:
    """t1 and t2: the case-start quantiles that end the train and val windows."""
    case_start: Dict[str, float] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cid = str(row["caseID_case_concept_name"]).strip()
            ts = parse_ts(row["event_timestamp_time_timestamp"])
            case_start[cid] = min(ts, case_start.get(cid, ts))

    starts = sorted(case_start.values())
    n = len(starts)
    t1 = starts[int(n * train_ratio)]
    t2 = starts[int(n * (train_ratio + val_ratio))]
    print(f"Temporal boundaries: t1={t1}  t2={t2}  (from {n} cases)")
    return t1, t2


def load_pt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_vocabs(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_id2ent(path: str) -> Optional[Dict[int, str]]:
    with open(path, encoding="utf-8") as f:
        ent2id = json.load(f)
    return {i: uri for uri, i in ent2id.items()}
