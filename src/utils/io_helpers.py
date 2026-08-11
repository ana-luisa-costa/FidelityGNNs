
import csv
import json
import random
from typing import Dict, Tuple, Optional

import numpy as np
import torch


def load_entity2id(path: str) -> Dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(path: str) -> torch.Tensor:
    return torch.tensor(np.load(path), dtype=torch.float)


def load_case_split(split_path: str) -> Tuple[set, set]:
    with open(split_path, encoding="utf-8") as f:
        d = json.load(f)
    return set(d["train"]), set(d["val"])


def load_case_split_from_csv(csv_path: str, train_ratio: float = 0.8, seed: int = 42, temporal: bool = False):
    cases = {}  # case_id -> earliest_timestamp (for temporal split)
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row["caseID_case_concept_name"]).strip()
            if temporal:
                timestamp_str = str(row["event_timestamp_time_timestamp"]).strip()
                if case_id not in cases:
                    cases[case_id] = timestamp_str
                else:
                    # Keep earliest timestamp for this case
                    if timestamp_str < cases[case_id]:
                        cases[case_id] = timestamp_str
            else:
                if case_id not in cases:
                    cases[case_id] = None

    if temporal:
        # Sort chronologically by earliest timestamp
        unique_cases = sorted(cases.keys(), key=lambda c: cases[c])
        split_mode = "temporal"
    else:
        # Random split
        unique_cases = sorted(cases.keys())
        rng = random.Random(seed)
        rng.shuffle(unique_cases)
        split_mode = "random"

    n = len(unique_cases)
    split = int(n * train_ratio)
    train_cases = set(unique_cases[:split])
    val_cases = set(unique_cases[split:])
    print(f"Case split ({split_mode}): {len(train_cases)} train / {len(val_cases)} val  (total {n})")
    return train_cases, val_cases, unique_cases


def load_pt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_vocabs(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_id2ent(path: str) -> Optional[Dict[int, str]]:
    with open(path, encoding="utf-8") as f:
        ent2id = json.load(f)
    return {i: uri for uri, i in ent2id.items()}
