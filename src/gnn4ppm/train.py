import re
import json
import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from datetime import datetime, timezone
from rdflib import Graph, URIRef
from torch_geometric.nn import RGCNConv

from src.utils.uri_helpers import get_case_of_entity, EVENT_RE, CASE_RE
from src.utils.io_helpers import load_entity2id, load_embeddings, load_case_split


P_ACT  = "http://example.org/hasevent_activity_concept_name"
P_TIME = "http://example.org/hasevent_timestamp_time_timestamp"


def parse_ts(ts: str) -> Optional[float]:
    ts = str(ts).strip()
    if len(ts) < 14 or not ts[:14].isdigit():
        return None
    try:
        dt = datetime.strptime(ts[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        frac = ts[14:]
        frac_sec = (int(frac) / (10 ** len(frac))) if (frac and frac.isdigit()) else 0.0
        return dt.timestamp() + frac_sec
    except Exception:
        return None


def get_event_attrs(g, event_uri: str):
    act, ts, otherC, otherN = None, None, {}, {}
    subj = URIRef(event_uri)
    for p, o in g.predicate_objects(subj):
        ps, os = str(p), str(o)
        pl = ps.lower()
        if ps == P_ACT:
            act = os
        elif ps == P_TIME:
            ts = os
        elif "hasevent_otherc" in pl or "org_resource" in pl:
            otherC[ps] = os
        elif "hasevent_othern" in pl:
            otherN[ps] = os
    return act, ts, otherC, otherN


def build_graph(g, ent2id: Dict[str, int], rel2id: Dict[str, int],
                excluded_cases: Optional[set] = None,
                allow_new_relations: bool = True):
    src_list, dst_list, rel_list = [], [], []
    for s, p, o in g:
        ps = str(p)
        if "rdf-schema" in ps or ps.endswith("#type"):
            continue
        if "directlyFollows" in ps:
            continue
        s_str, o_str = str(s), str(o)
        if s_str not in ent2id or o_str not in ent2id:
            continue
        if excluded_cases is not None:
            s_case = get_case_of_entity(s_str)
            o_case = get_case_of_entity(o_str)
            if ((s_case is not None and s_case in excluded_cases) or
                (o_case is not None and o_case in excluded_cases)):
                continue
        s_is_event = EVENT_RE.search(s_str) is not None
        o_is_event = EVENT_RE.search(o_str) is not None
        s_is_case = CASE_RE.search(s_str) is not None
        o_is_case = CASE_RE.search(o_str) is not None
        if s_is_event and o_is_case:
            continue  # event->case would let future events update the case node
        if o_is_event and not s_is_event:
            src, dst = s_str, o_str   # case->event: keep direction
        elif s_is_event or s_is_case:
            src, dst = o_str, s_str   # reverse to attribute->event / attribute->case
        else:
            src, dst = s_str, o_str
        if ps not in rel2id and not allow_new_relations:
            raise KeyError(f"Relation {ps!r} is missing from checkpoint rel2id")
        if ps not in rel2id:
            rel2id[ps] = len(rel2id)
        src_list.append(ent2id[src])
        dst_list.append(ent2id[dst])
        rel_list.append(rel2id[ps])

    ei = torch.tensor([src_list, dst_list], dtype=torch.long)
    et = torch.tensor(rel_list, dtype=torch.long)
    return ei, et


DF_REL = "http://example.org/directlyFollows"


def build_train_directly_follows(g, ent2id: Dict[str, int], rel2id: Dict[str, int],
                                 train_cases: set,
                                 allow_new_relations: bool = True):
    case_map: Dict[str, Dict[int, str]] = {}
    for uri in ent2id:
        m = EVENT_RE.search(uri)
        if m and m.group(2) in train_cases:
            case_map.setdefault(m.group(2), {})[int(m.group(1))] = uri

    pairs = set()
    for km in case_map.values():
        ks = sorted(km.keys())
        for a, b in zip(ks, ks[1:]):
            if b != a + 1:
                continue
            act_a, _, oc_a, _ = get_event_attrs(g, km[a])
            act_b, _, oc_b, _ = get_event_attrs(g, km[b])
            if act_a and act_b:
                pairs.add((act_b, act_a))
            for k, v_b in oc_b.items():
                v_a = oc_a.get(k)
                if v_a:
                    pairs.add((v_b, v_a))

    pairs = sorted((s, o) for s, o in pairs if s in ent2id and o in ent2id)
    if pairs and DF_REL not in rel2id:
        if not allow_new_relations:
            raise KeyError(f"Relation {DF_REL!r} is missing from checkpoint rel2id")
        rel2id[DF_REL] = len(rel2id)
    rel = rel2id.get(DF_REL, 0)
    src = [ent2id[s] for s, _ in pairs]
    dst = [ent2id[o] for _, o in pairs]
    ei = torch.tensor([src, dst], dtype=torch.long)
    et = torch.full((len(pairs),), rel, dtype=torch.long)
    return ei, et


def _split_fit_val_cases(
    g, case_map: Dict[str, Dict[int, int]], id2ent: Dict[int, str],
    train_cases: set, val_ratio: float = 0.1,
) -> Tuple[set, set]:
    """Chronological holdout carved out of train_cases only, used solely to
    score Optuna trials (never to backprop, never to touch test_cases).

    Mirrors the outer train/test split's own methodology (sort by earliest
    event time, hold out the latest slice) instead of a random draw, so no
    extra randomness/seed is introduced.
    """
    earliest: Dict[str, float] = {}
    for cid, km in case_map.items():
        if cid not in train_cases:
            continue
        best = None
        for nid in km.values():
            _, t_str, _, _ = get_event_attrs(g, id2ent[nid])
            ts = parse_ts(t_str)
            if ts is not None and (best is None or ts < best):
                best = ts
        # Cases with no parseable timestamp sort first (into fit), not last,
        # so they can't quietly dominate the small held-out val slice.
        earliest[cid] = best if best is not None else float("-inf")

    ordered = sorted(train_cases, key=lambda c: earliest.get(c, float("-inf")))
    n_total = len(ordered)
    n_val = int(round(n_total * val_ratio))
    n_val = max(0, min(n_val, n_total - 1)) if n_total > 1 else 0

    val_cases = set(ordered[n_total - n_val:]) if n_val else set()
    fit_cases = set(ordered) - val_cases
    return fit_cases, val_cases


# model

class RGCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_rel, num_bases=30, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid_dim) if in_dim != hid_dim else nn.Identity()
        self.conv1 = RGCNConv(in_dim,  hid_dim, num_rel, num_bases=num_bases)
        self.norm1 = nn.LayerNorm(hid_dim)
        self.conv2 = RGCNConv(hid_dim, hid_dim, num_rel, num_bases=num_bases)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.conv3 = RGCNConv(hid_dim, out_dim, num_rel, num_bases=num_bases)
        self.dropout_p = dropout

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.norm1(self.conv1(x, edge_index, edge_type)))
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        res = h
        h = F.relu(self.norm2(self.conv2(h, edge_index, edge_type)))
        h = F.dropout(h, p=self.dropout_p, training=self.training) + res
        return self.conv3(h, edge_index, edge_type)


class MultiTaskHead(nn.Module):
    def __init__(self, z_dim, n_act, otherC_vocabs, otherN_keys, head_hidden=256):
        super().__init__()
        self.act = nn.Sequential(
            nn.Linear(z_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(head_hidden, n_act)
        )
        self.time = nn.Sequential(
            nn.Linear(z_dim, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden // 2, 1)
        )
        self.otherC = nn.ModuleDict({
            _sanitize(k): nn.Sequential(
                nn.Linear(z_dim, head_hidden // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(head_hidden // 2, len(v))
            )
            for k, v in otherC_vocabs.items()
        })
        self.otherN = nn.ModuleDict({
            _sanitize(k): nn.Sequential(
                nn.Linear(z_dim, head_hidden // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(head_hidden // 2, 1)
            )
            for k in otherN_keys
        })

    def forward(self, z):
        return {
            "act":    self.act(z),
            "time":   self.time(z).squeeze(-1),
            "otherC": {k: ly(z) for k, ly in self.otherC.items()},
            "otherN": {k: ly(z).squeeze(-1) for k, ly in self.otherN.items()},
        }


def _sanitize(k: str) -> str:
    return k.replace(":", "_").replace("/", "_").replace(".", "_")


# loss

def compute_loss(out, y_act, y_time, y_c, y_n, idx, time_weight, act_weight=None, otherC_weights=None, otherC_weight_factor=0.5, otherN_weight_factor=0.3):
    mask_act = y_act[idx] >= 0
    loss = F.cross_entropy(out["act"][mask_act], y_act[idx][mask_act], weight=act_weight) if mask_act.any() else torch.tensor(0.0)

    m_t = torch.isfinite(y_time[idx])
    if m_t.any():
        loss = loss + time_weight * F.mse_loss(out["time"][m_t], y_time[idx][m_t])

    for k in y_c:
        m_c = y_c[k][idx] >= 0
        if m_c.any():
            c_weight = otherC_weights.get(k) if otherC_weights else None
            loss_c = F.cross_entropy(out["otherC"][k][m_c], y_c[k][idx][m_c], weight=c_weight)
            loss = loss + otherC_weight_factor * loss_c

    for k in y_n:
        m_n = torch.isfinite(y_n[k][idx])
        if m_n.any():
            loss_n = F.smooth_l1_loss(out["otherN"][k][m_n], y_n[k][idx][m_n])
            loss = loss + otherN_weight_factor * loss_n

    return loss


# optuna

def objective(trial, data, args):
    hid_dim    = trial.suggest_categorical("hid_dim", [64, 128, 256])
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    num_bases  = trial.suggest_int("num_bases", 10, 50)
    dropout    = trial.suggest_float("dropout", 0.2, 0.5)
    time_weight = trial.suggest_float("time_weight", 0.1, 1.0)
    head_hidden = trial.suggest_categorical("head_hidden", [64, 128, 256])
    otherC_weight_factor = trial.suggest_float("otherC_weight", 0.3, 1.0)
    otherN_weight_factor = trial.suggest_float("otherN_weight", 0.1, 0.5) 

    (x, ei_tr, et_tr, num_rel,
     train_pairs_t,
     y_act, y_time, y_c, y_n,
     n_train, device, act_weight, otherC_weights,
     fit_idx, val_idx) = data

    encoder = RGCNEncoder(x.size(1), hid_dim, hid_dim, num_rel,
                          num_bases=num_bases, dropout=dropout).to(device)
    head = _make_head(hid_dim, y_act[:n_train],
                      {k: v[:n_train] for k, v in y_c.items()},
                      y_n, device, head_hidden=head_hidden)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.optuna_epochs)

    # Backprop only on fit_idx; score trials on val_idx (both carved out of
    # train_cases only — see _split_fit_val_cases). Each is its own head()
    # call restricted to that pair subset's rows, so compute_loss (which
    # expects out's rows already aligned to idx) needs no changes.
    use_val = val_idx.numel() > 0
    best_score = float("inf")

    for epoch in range(args.optuna_epochs):
        encoder.train(); head.train()
        optimizer.zero_grad()
        z = encoder(x, ei_tr, et_tr)
        out_fit = head(z[train_pairs_t[fit_idx, 0]])
        fit_loss = compute_loss(out_fit, y_act, y_time, y_c, y_n, fit_idx, time_weight, act_weight, otherC_weights, otherC_weight_factor, otherN_weight_factor)
        fit_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if use_val:
            encoder.eval(); head.eval()
            with torch.no_grad():
                z_eval = encoder(x, ei_tr, et_tr)
                out_val = head(z_eval[train_pairs_t[val_idx, 0]])
                score = compute_loss(out_val, y_act, y_time, y_c, y_n, val_idx, time_weight, act_weight, otherC_weights, otherC_weight_factor, otherN_weight_factor).item()
        else:
            score = fit_loss.item()

        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_score = min(best_score, score)

    return best_score


def _make_head(hid_dim, y_act, y_c, y_n, device, head_hidden=256):
    n_act = int(y_act.max().item()) + 1
    c_vocabs_sizes = {k: int(y_c[k].max().item()) + 1 for k in y_c}
    c_vocabs_dummy = {k: {i: i for i in range(sz)} for k, sz in c_vocabs_sizes.items()}
    return MultiTaskHead(hid_dim, n_act, c_vocabs_dummy, list(y_n.keys()), head_hidden=head_hidden).to(device)


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttl",          required=True)
    ap.add_argument("--emb",          required=True, help="entity_embeddings.npy")
    ap.add_argument("--entity2id",    required=True, help="entity2id.json")
    ap.add_argument("--case_split",   required=True, help="case_split.json from build_rdf2vec_2.py")
    ap.add_argument("--save_best_test", "--save_best_val", dest="save_best_test", required=True)
    ap.add_argument(               ### added argument for saving model ###
        "--save_model",
        default=None,
        help="Path to save encoder+head checkpoint (default: same base as --save_best_test with _model.pt).",
    ) 
    ap.add_argument("--vocabs_out",   default=None,  help="Path to save vocabs.json (for evaluate).")
    ap.add_argument("--epochs",        type=int, default=30)
    ap.add_argument("--optuna_epochs", type=int, default=10,
                    help="Epochs per Optuna trial (shorter than final training).")
    ap.add_argument("--trials",        type=int, default=30)
    ap.add_argument(
        "--optuna_val_ratio", type=float, default=0.1,
        help="Fraction of train_cases (chronologically most recent) held out "
             "from Optuna's fit loss and used only to score trials. Never "
             "touches test_cases.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ent2id = load_entity2id(args.entity2id)
    id2ent = {i: e for e, i in ent2id.items()}
    x      = load_embeddings(args.emb).to(device)
    x_cpu  = x.cpu()

    train_cases, test_cases = load_case_split(args.case_split)
    print(f"Case split loaded: {len(train_cases)} train / {len(test_cases)} test")

    print("Parsing TTL …")
    g = Graph()
    g.parse(args.ttl, format="turtle")

    rel2id: Dict[str, int] = {}
    ei_tr, et_tr = build_graph(g, ent2id, rel2id, excluded_cases=test_cases)

    ei_df, et_df = build_train_directly_follows(g, ent2id, rel2id, train_cases)
    ei_tr = torch.cat([ei_tr, ei_df], dim=1)
    et_tr = torch.cat([et_tr, et_df])
    num_rel = len(rel2id)


    rel2id_test = dict(rel2id)
    ei_test, et_test = build_graph(g, ent2id, rel2id_test, excluded_cases=train_cases)
    keep = et_test < num_rel  
    ei_test, et_test = ei_test[:, keep], et_test[keep]

    ei_test = torch.cat([ei_test, ei_df], dim=1)
    et_test = torch.cat([et_test, et_df])

    ei_tr,  et_tr  = ei_tr.to(device),  et_tr.to(device)
    ei_test, et_test = ei_test.to(device), et_test.to(device)
    print(f"Train graph: {et_tr.size(0)} edges  |  Test graph: {et_test.size(0)} edges  "
          f"(incl. {et_df.size(0)} train-only directlyFollows)")

    case_map: Dict[str, Dict[int, int]] = {}   # case_id → {event_num → node_id}
    for nid, uri in id2ent.items():
        m = EVENT_RE.search(uri)
        if m:
            ev_num, cid = int(m.group(1)), m.group(2)
            case_map.setdefault(cid, {})[ev_num] = nid

    fit_cases, val_cases = _split_fit_val_cases(
        g, case_map, id2ent, train_cases, val_ratio=args.optuna_val_ratio)
    print(f"Optuna case split (out of train_cases only): "
          f"{len(fit_cases)} fit / {len(val_cases)} val")

    train_pairs: List[Tuple[int,int]] = []
    train_pair_cases: List[str] = []   # cid for each train_pairs entry, parallel list
    test_pairs:   List[Tuple[int,int]] = []

    for cid, km in case_map.items():
        ks = sorted(km.keys())
        case_pairs = [(km[ks[i]], km[ks[i+1]])
                      for i in range(len(ks)-1)
                      if ks[i]+1 in km]
        if cid in train_cases:
            train_pairs.extend(case_pairs)
            train_pair_cases.extend([cid] * len(case_pairs))
        else:
            test_pairs.extend(case_pairs)

    print(f"Pairs — train: {len(train_pairs)}  test: {len(test_pairs)}")

    fit_idx = torch.tensor(
        [i for i, cid in enumerate(train_pair_cases) if cid in fit_cases],
        dtype=torch.long, device=device)
    val_idx = torch.tensor(
        [i for i, cid in enumerate(train_pair_cases) if cid in val_cases],
        dtype=torch.long, device=device)
    print(f"Optuna pairs (out of train pairs only) — fit: {fit_idx.numel()}  val: {val_idx.numel()}")

    train_pairs_t = torch.tensor(train_pairs, dtype=torch.long, device=device)
    test_pairs_t   = torch.tensor(test_pairs,   dtype=torch.long, device=device)

    act_vocab: Dict[str, int] = {}
    c_vocabs:  Dict[str, Dict[str, int]] = {}
    n_keys:    set = set()

    for _, dst_id in train_pairs:
        a, _, oc, on = get_event_attrs(g, id2ent[dst_id])
        if a and a not in act_vocab:
            act_vocab[a] = len(act_vocab)
        for k, v in oc.items():
            if k not in c_vocabs:
                c_vocabs[k] = {}
            if v not in c_vocabs[k]:
                c_vocabs[k][v] = len(c_vocabs[k])
        n_keys.update(on.keys())

    MAX_CATEG_CLASSES = 500
    high_card = [k for k, v in c_vocabs.items() if len(v) > MAX_CATEG_CLASSES]
    for k in high_card:
        print(f"  Dropping categ attr '{k}' with {len(c_vocabs[k])} classes (>{MAX_CATEG_CLASSES})")
        del c_vocabs[k]

    n_keys = sorted(n_keys)
    print(f"Vocabs — activities: {len(act_vocab)}  "
          f"categ attrs: {len(c_vocabs)}  num attrs: {len(n_keys)}")

    N_tr = len(train_pairs)
    N_test = len(test_pairs)
    N = N_tr + N_test

    y_act  = torch.full((N,), -1, dtype=torch.long,  device=device)
    y_time = torch.full((N,), float("nan"),           device=device)
    y_c    = {_sanitize(k): torch.full((N,), -1, dtype=torch.long, device=device)
              for k in c_vocabs}
    y_n    = {_sanitize(k): torch.full((N,), float("nan"), device=device)
              for k in n_keys}

    _ts_cache: Dict[int, Optional[float]] = {}

    def _get_ts(eid: int) -> Optional[float]:
        if eid not in _ts_cache:
            _, t_str, _, _ = get_event_attrs(g, id2ent[eid])
            _ts_cache[eid] = parse_ts(t_str)
        return _ts_cache[eid]

    for i, (src_id, dst_id) in enumerate(train_pairs):
        a, t_str, oc, on = get_event_attrs(g, id2ent[dst_id])
        if a in act_vocab:
            y_act[i] = act_vocab[a]
        ts_src = _get_ts(src_id)
        ts_dst = parse_ts(t_str)
        if ts_src is not None and ts_dst is not None:
            y_time[i] = max(ts_dst - ts_src, 0.0)  
        for k, v in oc.items():
            sk = _sanitize(k)
            if sk in y_c and v in c_vocabs[k]:
                y_c[sk][i] = c_vocabs[k][v]
        for k, v in on.items():
            sk = _sanitize(k)
            if sk in y_n:
                try:
                    y_n[sk][i] = float(v)
                except (ValueError, TypeError):
                    pass

    for j, (src_id, dst_id) in enumerate(test_pairs):
        i = N_tr + j
        a, t_str, oc, on = get_event_attrs(g, id2ent[dst_id])
        if a in act_vocab:
            y_act[i] = act_vocab[a]
        ts_src = _get_ts(src_id)
        ts_dst = parse_ts(t_str)
        if ts_src is not None and ts_dst is not None:
            y_time[i] = max(ts_dst - ts_src, 0.0)   
        for k, v in oc.items():
            sk = _sanitize(k)
            if sk in y_c and v in c_vocabs.get(k, {}):
                y_c[sk][i] = c_vocabs[k][v]
        for k, v in on.items():
            sk = _sanitize(k)
            if sk in y_n:
                try:
                    y_n[sk][i] = float(v)
                except (ValueError, TypeError):
                    pass

    y_time = torch.log1p(torch.clamp(y_time, min=0.0))

    train_mask = torch.isfinite(y_time[:N_tr])
    if train_mask.any():
        mu_t  = y_time[:N_tr][train_mask].mean()
        std_t = y_time[:N_tr][train_mask].std(correction=0).clamp(min=1e-6)
    else:
        mu_t, std_t = torch.tensor(0.0), torch.tensor(1.0)
    y_time = (y_time - mu_t) / std_t

    norm_stats: Dict[str, Dict[str, float]] = {}
    for k in y_n:
        m_tr = torch.isfinite(y_n[k][:N_tr])
        if m_tr.any():
            mu  = y_n[k][:N_tr][m_tr].mean()
            std = y_n[k][:N_tr][m_tr].std(correction=0).clamp(min=1e-6)
        else:
            mu, std = torch.tensor(0.0), torch.tensor(1.0)
        y_n[k] = (y_n[k] - mu) / std
        norm_stats[k] = {"mu": float(mu), "std": float(std)}

    vocabs = {
        "act_vocab":   act_vocab,
        "c_vocabs":    c_vocabs,
        "n_keys":      n_keys,
        "mu_t":        float(mu_t),
        "std_t":       float(std_t),
        "norm_stats":  norm_stats,
    }
    vocabs_path = args.vocabs_out or os.path.splitext(args.save_best_test)[0] + "_vocabs.json"
    with open(vocabs_path, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, indent=2)
    print(f"Saved vocabs/norm → {vocabs_path}")


    def output_pairs_info(pairs, split_name, N_offset, out_path, edge_index, edge_type):
        edge_dict = {}  # {(src, dst): [rel_ids]} 
        for edge_idx, rel_id in enumerate(edge_type):
            src, dst = int(edge_index[0, edge_idx].item()), int(edge_index[1, edge_idx].item())
            if (src, dst) not in edge_dict:
                edge_dict[(src, dst)] = []
            edge_dict[(src, dst)].append(int(rel_id.item()))
        
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (src_id, dst_id) in enumerate(pairs):
                idx = N_offset + i
                src_id, dst_id = int(src_id), int(dst_id)
                src_uri = id2ent[src_id]
                dst_uri = id2ent[dst_id]
                a, t_str, oc, on = get_event_attrs(g, dst_uri)
                
                src_emb = x_cpu[src_id].tolist()
                dst_emb = x_cpu[dst_id].tolist()
                
                pair_edges = edge_dict.get((src_id, dst_id), [])
                
                entry = {
                    "pair_idx": i,
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "src_uri": src_uri,
                    "dst_uri": dst_uri,
                    "direct_edges_rel_ids": pair_edges,
                    "activity": a,
                    "timestamp_str": t_str,
                    "otherC": oc,
                    "otherN": on,
                    "y_act": int(y_act[idx].item()) if y_act[idx] >= 0 else None,
                    "y_time": float(y_time[idx].item()) if torch.isfinite(y_time[idx]) else None,
                    "y_c": {k: int(y_c[k][idx].item()) if y_c[k][idx] >= 0 else None for k in y_c},
                    "y_n": {k: float(y_n[k][idx].item()) if torch.isfinite(y_n[k][idx]) else None for k in y_n},
                }
                f.write(json.dumps(entry) + "\n")

    train_info_path = os.path.splitext(args.save_best_test)[0] + "_train_info.jsonl"
    test_info_path   = os.path.splitext(args.save_best_test)[0] + "_test_info.jsonl"
    output_pairs_info(train_pairs, "train", 0, train_info_path, ei_tr, et_tr)
    output_pairs_info(test_pairs, "test", N_tr, test_info_path, ei_test, et_test)
    print(f"Wrote all training info to {train_info_path}")
    print(f"Wrote all test info to {test_info_path}")


    train_act = y_act[:N_tr]
    train_act_valid = train_act[train_act >= 0]
    n_act = int(train_act_valid.max().item()) + 1
    counts = torch.bincount(train_act_valid, minlength=n_act).float().clamp(min=1.0)
    act_weight = (counts.sum() / (n_act * counts)).to(device)
    print(f"Activity class weights: {act_weight.tolist()}")


    otherC_weights = {}
    for k in y_c:
        train_c = y_c[k][:N_tr]
        train_c_valid = train_c[train_c >= 0]
        if train_c_valid.numel() > 0:
            n_c = int(train_c_valid.max().item()) + 1
            c_counts = torch.bincount(train_c_valid, minlength=n_c).float().clamp(min=1.0)
            otherC_weights[k] = (c_counts.sum() / (n_c * c_counts)).clamp(min=0.05, max=20.0).to(device)
        else:
            otherC_weights[k] = None
    print(f"Computed class weights for {len(otherC_weights)} categorical attributes")

    # optuna study
    data_bundle = (x, ei_tr, et_tr, num_rel,
                   train_pairs_t,
                   y_act, y_time, y_c, y_n,
                   N_tr, device, act_weight, otherC_weights,
                   fit_idx, val_idx)

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.PercentilePruner(
                                    25.0, n_startup_trials=3, n_warmup_steps=10))
    study.optimize(
        lambda trial: objective(trial, data_bundle, args),
        n_trials=args.trials,
    )
    score_label = "validation" if val_idx.numel() > 0 else "train"
    print(f"\nBest trial {score_label}-loss: {study.best_trial.value:.4f}")
    print(f"Best params:         {study.best_params}")

    best   = study.best_params
    hid    = best["hid_dim"]
    encoder = RGCNEncoder(x.size(1), hid, hid, num_rel,
                          num_bases=best["num_bases"],
                          dropout=best["dropout"]).to(device)
    head   = _make_head(hid, y_act[:N_tr],
                        {k: v[:N_tr] for k, v in y_c.items()},
                        y_n, device, head_hidden=best["head_hidden"])
    opt    = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=best["lr"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    time_weight = best["time_weight"]
    otherC_weight_factor = best["otherC_weight"]
    otherN_weight_factor = best["otherN_weight"]

    train_idx = torch.arange(N_tr, device=device)
    test_idx   = torch.arange(N_tr, N, device=device)

    for epoch in range(args.epochs):
        encoder.train(); head.train()
        opt.zero_grad()
        z   = encoder(x, ei_tr, et_tr)
        out = head(z[train_pairs_t[:, 0]])
        loss = compute_loss(out, y_act, y_time, y_c, y_n, train_idx, time_weight, act_weight, otherC_weights, otherC_weight_factor, otherN_weight_factor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), max_norm=1.0)
        opt.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:4d}  train_loss={loss.item():.4f}")

    # single inference pass on test set after training
    encoder.eval(); head.eval()
    with torch.no_grad():
        z_t   = encoder(x, ei_test, et_test)
        t_out = head(z_t[test_pairs_t[:, 0]])

    torch.save({
        "y_act":    y_act[test_idx],
        "y_time":   y_time[test_idx],
        "y_otherC": {k: y_c[k][test_idx] for k in y_c},
        "y_otherN": {k: y_n[k][test_idx] for k in y_n},
        "pred": {
            "act_logits":    t_out["act"],
            "time_pred_log": t_out["time"],
            "otherC_logits": t_out["otherC"],
            "otherN_pred":   t_out["otherN"],
        },
        "pairs_test": test_pairs_t,
    }, args.save_best_test)

    model_path = args.save_model or (os.path.splitext(args.save_best_test)[0] + "_model.pt")
    train_act_valid = y_act[:N_tr][y_act[:N_tr] >= 0]
    n_act_save = int(train_act_valid.max().item()) + 1 if train_act_valid.numel() > 0 else 1
    otherC_num_classes = {}
    for k in y_c:
        m = y_c[k][:N_tr] >= 0
        if m.any():
            otherC_num_classes[k] = int(y_c[k][:N_tr][m].max().item()) + 1
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "head_state_dict":    head.state_dict(),
        "best_params":          dict(best),
        "in_dim":               int(x.size(1)),
        "num_rel":              int(num_rel),
        "rel2id":               dict(rel2id),
        "n_act":                n_act_save,
        "otherC_num_classes":   otherC_num_classes,
        "otherN_keys":          sorted(y_n.keys()),
        "time_weight":          float(time_weight),
        "otherC_weight_factor": float(otherC_weight_factor),
        "otherN_weight_factor": float(otherN_weight_factor),
    }, model_path)
    print(f"Training complete. Saved model -> {model_path}")


if __name__ == "__main__":
    main()
