import json
import math
import os
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from rdflib import Graph, URIRef
from torch_geometric.nn import RGCNConv

from src.utils.uri_helpers import CASE_URI, EVENT_RE, P_TIME, parse_ts, order_case_events, usable_event_uris, event_timestamps
from src.utils.io_helpers import load_entity2id, load_embeddings, load_temporal_boundaries


P_ACT  = "http://example.org/hasevent_activity_concept_name"
P_FOLLOWS = "http://example.org/eventDirectlyFollows"  
UNK = "<UNK>"  # class of a missing or train-unseen categorical value

# The only case attributes used as features: both are recorded when the case
# starts, so every prefix may see them. The BPIC20 ones change during a case
# and for the BPIC13 ones it is unknown when they are recorded.
P_AMOUNT_REQ = "http://example.org/hascase_otherN_case_AMOUNT_REQ"
P_REG_DATE   = "http://example.org/hascase_otherN_case_REG_DATE"

NAN = float("nan")
TIME_COLUMNS = slice(0, 4)  # filled for event nodes
CASE_COLUMNS = slice(4, 6)  # filled for case nodes
N_NODE_FEATURES = 6


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


def build_case_map(id2ent: Dict[int, str], usable: set,
                   event_ts: Dict[str, Optional[float]]) -> Dict[str, List[int]]:
    """case_id → usable event node ids in temporal order (event# breaks ties)."""
    events_by_case: Dict[str, Dict[int, Tuple[int, Optional[float]]]] = {}
    for nid, uri in id2ent.items():
        m = EVENT_RE.search(uri)
        if m and uri in usable:
            ev_num, cid = int(m.group(1)), m.group(2)
            events_by_case.setdefault(cid, {})[ev_num] = (nid, event_ts.get(uri))
    return {cid: order_case_events(evs) for cid, evs in events_by_case.items()}


def split_pairs_by_time(case_map: Dict[str, List[int]], id2ent: Dict[int, str],
                        event_ts: Dict[str, Optional[float]], t1: float, t2: float):
    """Consecutive (src, dst) event pairs, assigned to train/val/test by dst time."""
    train_pairs: List[Tuple[int, int]] = []
    val_pairs:   List[Tuple[int, int]] = []
    test_pairs:  List[Tuple[int, int]] = []
    for cid, events in case_map.items():
        for src_id, dst_id in zip(events, events[1:]):
            ts = event_ts[id2ent[dst_id]]  # each pair is assigned by its target's time
            if ts < t1:
                train_pairs.append((src_id, dst_id))
            elif ts < t2:
                val_pairs.append((src_id, dst_id))
            else:
                test_pairs.append((src_id, dst_id))
    return train_pairs, val_pairs, test_pairs


def add_inverse_edges(ei: torch.Tensor, et: torch.Tensor, num_rel: int):
    s, o = ei
    return (torch.cat([ei, torch.stack([o, s])], dim=1),
            torch.cat([et, et + num_rel]))


def collect_case_triples(g, ent2id: Dict[str, int], rel2id: Dict[str, int],
                         case_map: Dict[str, List[int]]):
    """Group triples by case, keeping only those that touch an event node.

    Value-to-value triples (e.g. directlyFollows between shared attribute nodes)
    are dropped: they would connect events across cases and across time.
    """
    case_of_node = {nid: cid for cid, events in case_map.items() for nid in events}

    triples_by_case: Dict[str, List[Tuple[int, int, int]]] = {}
    for s, p, o in g:
        ps = str(p)
        if "rdf-schema" in ps or ps.endswith("#type"):
            continue
        if ps == P_TIME:  # time is a feature of the event node, not a node
            continue
        s_str, o_str = str(s), str(o)
        if s_str not in ent2id or o_str not in ent2id:
            continue
        s_id, o_id = ent2id[s_str], ent2id[o_str]
        cid = case_of_node.get(s_id)
        if cid is None:
            cid = case_of_node.get(o_id)
        if cid is None:
            continue
        if ps not in rel2id:
            rel2id[ps] = len(rel2id)
        triples_by_case.setdefault(cid, []).append((s_id, o_id, rel2id[ps]))
    return triples_by_case


def build_prefix_batch(pairs: List[Tuple[int, int]],
                       case_map: Dict[str, List[int]],
                       triples_by_case: Dict[str, List[Tuple[int, int, int]]],
                       follows_rel: int,
                       device):
    """Disjoint union of one prefix subgraph per (src, dst) pair.

    Each subgraph holds only the case's events up to and including src plus
    their attribute nodes; shared value nodes are duplicated per subgraph, so no
    path reaches a future event or another prefix. Consecutive prefix events are
    chained with follows_rel so the encoder can learn the event order.

    Returns (edge_index, edge_type, node_orig, src_pos), where node_orig maps
    each batched node to its original entity id and src_pos gives the batched
    node of each pair's src event (aligned with pairs).
    """
    event_rank: Dict[int, Tuple[str, int]] = {}  # event node → (case, position)
    for cid, events in case_map.items():
        for rank, nid in enumerate(events):
            event_rank[nid] = (cid, rank)

    src_l, dst_l, rel_l = [], [], []
    node_orig: List[int] = []
    src_pos:   List[int] = []

    for src_id, _ in pairs:
        cid, src_rank = event_rank[src_id]
        local: Dict[int, int] = {}

        def local_id(n: int) -> int:
            if n not in local:
                local[n] = len(node_orig)
                node_orig.append(n)
            return local[n]

        src_pos.append(local_id(src_id))
        for s, o, r in triples_by_case.get(cid, []):
            if any(n in event_rank and event_rank[n][1] > src_rank for n in (s, o)):
                continue
            src_l.append(local_id(s))
            dst_l.append(local_id(o))
            rel_l.append(r)

        prefix_events = case_map[cid][:src_rank + 1]
        for prev, nxt in zip(prefix_events, prefix_events[1:]):
            src_l.append(local_id(prev))
            dst_l.append(local_id(nxt))
            rel_l.append(follows_rel)

    ei = torch.tensor([src_l, dst_l], dtype=torch.long)
    et = torch.tensor(rel_l, dtype=torch.long)
    return (ei, et,
            torch.tensor(node_orig, dtype=torch.long),
            torch.tensor(src_pos, dtype=torch.long, device=device))


# node features

def log_seconds(later: Optional[float], earlier: Optional[float]) -> float:
    """log1p of the seconds from `earlier` to `later`; NaN if either is unknown."""
    if later is None or earlier is None:
        return NAN
    if later < earlier:
        raise ValueError("A feature may only look back in time.")
    return math.log1p(later - earlier)


def event_time_features(ts: Optional[float], previous_ts: Optional[float],
                        case_start: Optional[float]) -> List[float]:
    """[log seconds since the previous event, log seconds since the first event,
    hour of day, weekday] of an event."""
    if ts is None:
        return [NAN] * 4
    moment = datetime.fromtimestamp(ts, tz=timezone.utc)
    return [log_seconds(ts, previous_ts), log_seconds(ts, case_start),
            moment.hour, moment.weekday()]


def case_start_features(g, case_uri: str, case_start: Optional[float]) -> List[float]:
    """[log requested amount, log seconds from registration to the first event]."""
    case = URIRef(case_uri)
    # any=False raises on several values: such an attribute changes during the
    # case, so an early prefix would see a value that is only recorded later
    amount = g.value(case, URIRef(P_AMOUNT_REQ), any=False)
    registered = g.value(case, URIRef(P_REG_DATE), any=False)
    return [NAN if amount is None else math.log1p(float(amount)),
            log_seconds(case_start, parse_ts(registered))]


def build_node_features(g, ent2id: Dict[str, int], case_map: Dict[str, List[int]],
                        event_ts: Dict[str, Optional[float]]) -> torch.Tensor:
    """Numeric features per node, NaN where a feature does not apply.

    A row only uses what is known when its node first enters a prefix graph
    (the event itself, its predecessors and the case start), so every prefix
    graph containing the node shares the row.
    """
    node_ts = {ent2id[uri]: ts for uri, ts in event_ts.items()}
    features = [[NAN] * N_NODE_FEATURES for _ in ent2id]
    for cid, events in case_map.items():
        timestamps = [node_ts.get(event) for event in events]
        previous = timestamps[:1] + timestamps  # the first event has no gap
        for event, ts, previous_ts in zip(events, timestamps, previous):
            features[event][TIME_COLUMNS] = event_time_features(ts, previous_ts, timestamps[0])
        case_uri = CASE_URI.format(cid)
        features[ent2id[case_uri]][CASE_COLUMNS] = case_start_features(g, case_uri, timestamps[0])
    return torch.tensor(features)


def feature_stats(features: torch.Tensor, fit_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-column mean/std over `fit_nodes`, ignoring NaN."""
    fit = features[fit_nodes]
    mean = fit.nanmean(dim=0)
    std = (fit - mean).pow(2).nanmean(dim=0).sqrt().clamp(min=1e-6)
    return mean, std


def apply_standardize(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Scale each column with the given mean/std; NaN becomes 0."""
    return torch.nan_to_num((features - mean) / std)


def standardize(features: torch.Tensor, fit_nodes: torch.Tensor) -> torch.Tensor:
    """Scale each column with its mean/std over `fit_nodes`; NaN becomes 0."""
    mean, std = feature_stats(features, fit_nodes)
    return apply_standardize(features, mean, std)


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


def minmaxnorm(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return (x - lo) / (hi - lo).clamp_min(1e-8)


# optuna
time_w = 0.2
def objective(trial, data, args):
    hid_dim    = trial.suggest_categorical("hid_dim", [64, 128, 256])
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    num_bases  = trial.suggest_int("num_bases", 10, 50)
    dropout    = trial.suggest_float("dropout", 0.2, 0.5)
    head_hidden = trial.suggest_categorical("head_hidden", [64, 128, 256])
    otherC_weight_factor = trial.suggest_float("otherC_weight", 0.3, 1.0)
    otherN_weight_factor = trial.suggest_float("otherN_weight", 0.1, 0.5)

    (x_tr, ei_tr, et_tr, src_tr,
     x_val, ei_val, et_val, src_val,
     num_rel, train_idx, val_idx,
     y_act, y_time, y_c, y_n,
     device, act_weight, otherC_weights) = data

    encoder = RGCNEncoder(x_tr.size(1), hid_dim, hid_dim, num_rel * 2,
                          num_bases=num_bases, dropout=dropout).to(device)
    head = _make_head(hid_dim, y_act, y_c, y_n, device, head_hidden=head_hidden)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.optuna_epochs)

    best_v = float("inf")

    # Time range from training targets only, to avoid leaking validation info
    y_time_tr = y_time[train_idx]
    y_time_tr = y_time_tr[torch.isfinite(y_time_tr)]
    t_lo, t_hi = y_time_tr.min(), y_time_tr.max()

    for epoch in range(args.optuna_epochs):
        encoder.train(); head.train()
        optimizer.zero_grad()
        z   = encoder(x_tr, ei_tr, et_tr)
        out = head(z[src_tr])
        loss = compute_loss(out, y_act, y_time, y_c, y_n, train_idx, time_w, act_weight, otherC_weights, otherC_weight_factor, otherN_weight_factor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), max_norm=1.0)
        optimizer.step()

        encoder.eval(); head.eval()
        with torch.no_grad():
            z_v  = encoder(x_val, ei_val, et_val)
            v_out = head(z_v[src_val])
            m_act = y_act[val_idx] >= 0
            if m_act.any():
                v_loss = F.cross_entropy(v_out["act"][m_act], y_act[val_idx][m_act], weight=act_weight)
            else:
                v_loss = torch.tensor(float("inf"))
            m_t = torch.isfinite(y_time[val_idx])
            if m_t.any():
                v_loss = v_loss + 0.2 * F.mse_loss(
                    minmaxnorm(v_out["time"][m_t], t_lo, t_hi),
                    minmaxnorm(y_time[val_idx][m_t], t_lo, t_hi))

        scheduler.step()
        trial.report(v_loss.item(), epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_v = min(best_v, v_loss.item())

    return best_v


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
    ap.add_argument("--temporal_split", required=True, help="temporal_split.json from build_rdf2vec.py")
    ap.add_argument("--save_best_val",required=True)
    ap.add_argument("--save_model",   default=None,
                    help="Path to save the encoder/head checkpoint used by the explainers "
                         "(default: <save_best_val stem>_model.pt).")
    ap.add_argument("--vocabs_out",   default=None,  help="Path to save vocabs.json (for evaluate).")
    ap.add_argument("--epochs",        type=int, default=200,
                    help="Max epochs for final training. Early stopping may end sooner.")
    ap.add_argument("--optuna_epochs", type=int, default=50,
                    help="Epochs per Optuna trial (shorter than final training).")
    ap.add_argument("--trials",        type=int, default=30)
    ap.add_argument("--patience",      type=int, default=50,
                    help="Stop final training if val loss does not improve for this many epochs.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ent2id = load_entity2id(args.entity2id)
    id2ent = {i: e for e, i in ent2id.items()}
    x      = load_embeddings(args.emb).to(device)

    t1, t2 = load_temporal_boundaries(args.temporal_split)
    print(f"Temporal boundaries: t1={t1}  t2={t2}")

    print("Parsing TTL …")
    g = Graph()
    g.parse(args.ttl, format="turtle")

    usable = usable_event_uris(g)

    # event order inside a case comes from timestamps; event# only breaks ties
    event_ts = event_timestamps(g)
    events_by_case: Dict[str, Dict[int, Tuple[int, Optional[float]]]] = {}
    for nid, uri in id2ent.items():
        m = EVENT_RE.search(uri)
        if m and uri in usable:
            ev_num, cid = int(m.group(1)), m.group(2)
            events_by_case.setdefault(cid, {})[ev_num] = (nid, event_ts.get(uri))
    case_map: Dict[str, List[int]] = {   # case_id → event node ids in temporal order
        cid: order_case_events(evs) for cid, evs in events_by_case.items()}

    train_pairs: List[Tuple[int,int]] = []
    val_pairs:   List[Tuple[int,int]] = []
    test_pairs:  List[Tuple[int,int]] = []

    for cid, events in case_map.items():
        for src_id, dst_id in zip(events, events[1:]):
            ts = event_ts[id2ent[dst_id]]  # each pair is assigned by its target's time
            if ts < t1:
                train_pairs.append((src_id, dst_id))
            elif ts < t2:
                val_pairs.append((src_id, dst_id))
            else:
                src_ts = event_ts[id2ent[src_id]]
                #Test on the last timestamp on the src and dst and if its the same -> drop (of the test)
                if src_ts != ts:
                    test_pairs.append((src_id, dst_id))

    print(f"Pairs — train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}")

    rel2id: Dict[str, int] = {P_FOLLOWS: 0}
    triples_by_case = collect_case_triples(g, ent2id, rel2id, case_map)
    num_rel = len(rel2id)

    print("Building prefix graphs …")
    follows_rel = rel2id[P_FOLLOWS]
    ei_tr,  et_tr,  nodes_tr,  src_tr  = build_prefix_batch(train_pairs, case_map, triples_by_case, follows_rel, device)
    ei_val, et_val, nodes_val, src_val = build_prefix_batch(val_pairs,   case_map, triples_by_case, follows_rel, device)
    ei_te,  et_te,  nodes_te,  src_te  = build_prefix_batch(test_pairs,  case_map, triples_by_case, follows_rel, device)

    ei_tr,  et_tr  = add_inverse_edges(ei_tr,  et_tr,  num_rel)
    ei_val, et_val = add_inverse_edges(ei_val, et_val, num_rel)
    ei_te,  et_te  = add_inverse_edges(ei_te,  et_te,  num_rel)
    ei_tr,  et_tr  = ei_tr.to(device),  et_tr.to(device)
    ei_val, et_val = ei_val.to(device), et_val.to(device)
    ei_te,  et_te  = ei_te.to(device),  et_te.to(device)

    # scaled with the nodes of the train prefixes only: no val/test statistics
    raw_features = build_node_features(g, ent2id, case_map, event_ts)
    feat_mean, feat_std = feature_stats(raw_features, fit_nodes=nodes_tr.unique())
    node_features = apply_standardize(raw_features, feat_mean, feat_std)
    emb_dim = x.size(1)
    x = torch.cat([x, node_features.to(device)], dim=1)

    x_tr  = x[nodes_tr]   # node features duplicated per prefix subgraph
    x_val = x[nodes_val]
    x_te  = x[nodes_te]
    print(f"Prefix batches — train: {nodes_tr.size(0)} nodes  |  "
          f"val: {nodes_val.size(0)} nodes  |  test: {nodes_te.size(0)} nodes")

    test_pairs_t = torch.tensor(test_pairs, dtype=torch.long, device=device)  # kept for the checkpoint

    act_vocab: Dict[str, int] = {UNK: 0}
    c_vocabs:  Dict[str, Dict[str, int]] = {}
    n_keys:    set = set()

    for _, dst_id in train_pairs:
        a, _, oc, on = get_event_attrs(g, id2ent[dst_id])
        if a and a not in act_vocab:
            act_vocab[a] = len(act_vocab)
        for k, v in oc.items():
            if k not in c_vocabs:
                c_vocabs[k] = {UNK: 0}
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
    N_val = len(val_pairs)
    N_te = len(test_pairs)
    N = N_tr + N_val + N_te

    y_act  = torch.full((N,), -1, dtype=torch.long,  device=device)
    y_time = torch.full((N,), float("nan"),           device=device)
    y_c    = {_sanitize(k): torch.full((N,), -1, dtype=torch.long, device=device)
              for k in c_vocabs}
    y_n    = {_sanitize(k): torch.full((N,), float("nan"), device=device)
              for k in n_keys}

    _ts_cache: Dict[int, Optional[float]] = {}
    n_negative_dt = 0

    def _get_ts(eid: int) -> Optional[float]:
        if eid not in _ts_cache:
            _, t_str, _, _ = get_event_attrs(g, id2ent[eid])
            _ts_cache[eid] = parse_ts(t_str)
        return _ts_cache[eid]

    for offset, pairs in ((0, train_pairs), (N_tr, val_pairs), (N_tr + N_val, test_pairs)):
        for j, (src_id, dst_id) in enumerate(pairs):
            i = offset + j
            a, t_str, oc, on = get_event_attrs(g, id2ent[dst_id])
            y_act[i] = act_vocab.get(a, act_vocab[UNK])
            ts_src = _get_ts(src_id)
            ts_dst = parse_ts(t_str)
            if ts_src is not None and ts_dst is not None:
                n_negative_dt += ts_dst < ts_src
                y_time[i] = max(ts_dst - ts_src, 0.0)
            for k, vocab in c_vocabs.items():
                y_c[_sanitize(k)][i] = vocab.get(oc.get(k), vocab[UNK])
            for k, v in on.items():
                sk = _sanitize(k)
                if sk in y_n:
                    try:
                        y_n[sk][i] = float(v)
                    except (ValueError, TypeError):
                        pass

    if n_negative_dt:
        print(f"WARNING: {n_negative_dt} pairs have dst before src in time; "
              f"event order is not temporal")

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
    vocabs_path = args.vocabs_out or os.path.splitext(args.save_best_val)[0] + "_vocabs.json"
    with open(vocabs_path, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, indent=2)
    print(f"Saved vocabs/norm → {vocabs_path}")


    def output_pairs_info(pairs, N_offset, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (src_id, dst_id) in enumerate(pairs):
                idx = N_offset + i
                src_id, dst_id = int(src_id), int(dst_id)
                src_uri = id2ent[src_id]
                dst_uri = id2ent[dst_id]
                a, t_str, oc, on = get_event_attrs(g, dst_uri)

                entry = {
                    "pair_idx": i,
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "src_uri": src_uri,
                    "dst_uri": dst_uri,
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

    train_info_path = os.path.splitext(args.save_best_val)[0] + "_train_info.jsonl"
    val_info_path   = os.path.splitext(args.save_best_val)[0] + "_val_info.jsonl"
    test_info_path  = os.path.splitext(args.save_best_val)[0] + "_test_info.jsonl"
    output_pairs_info(train_pairs, 0, train_info_path)
    output_pairs_info(val_pairs, N_tr, val_info_path)
    output_pairs_info(test_pairs, N_tr + N_val, test_info_path)
    print(f"Wrote all training info to {train_info_path}")
    print(f"Wrote all validation info to {val_info_path}")
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

    train_idx = torch.arange(N_tr,                device=device)
    val_idx   = torch.arange(N_tr, N_tr + N_val,  device=device)
    test_idx  = torch.arange(N_tr + N_val, N,     device=device)

    # optuna validates on the real val set; test is never touched here
    data_bundle = (x_tr, ei_tr, et_tr, src_tr,
                   x_val, ei_val, et_val, src_val,
                   num_rel, train_idx, val_idx,
                   y_act, y_time, y_c, y_n,
                   device, act_weight, otherC_weights)

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.PercentilePruner(
                                    25.0, n_startup_trials=3, n_warmup_steps=10))
    study.optimize(
        lambda trial: objective(trial, data_bundle, args),
        n_trials=args.trials,
    )
    print(f"\nBest trial val-loss: {study.best_trial.value:.4f}")
    print(f"Best params:         {study.best_params}")

    best   = study.best_params
    hid    = best["hid_dim"]
    encoder = RGCNEncoder(x_tr.size(1), hid, hid, num_rel * 2,
                          num_bases=best["num_bases"],
                          dropout=best["dropout"]).to(device)
    head   = _make_head(hid, y_act, y_c, y_n, device, head_hidden=best["head_hidden"])
    opt    = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=best["lr"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    time_weight = time_w
    otherC_weight_factor = best["otherC_weight"]
    otherN_weight_factor = best["otherN_weight"]

    best_v = float("inf")
    no_improve = 0
    model_path = args.save_model or (os.path.splitext(args.save_best_val)[0] + "_model.pt")
    # everything the explainers need to rebuild the exact model input
    model_meta = {
        "best_params":       best,
        "in_dim":            int(x_tr.size(1)),
        "emb_dim":           int(emb_dim),
        "num_rel":           int(num_rel),       # base relations; encoder uses 2x (inverse edges)
        "encoder_num_rel":   int(num_rel * 2),
        "rel2id":            dict(rel2id),
        "follows_rel":       int(follows_rel),
        "n_act":             int(y_act.max().item()) + 1,
        "otherC_num_classes": {k: int(y_c[k].max().item()) + 1 for k in y_c},
        "otherN_keys":       list(y_n.keys()),
        "time_weight":       float(time_weight),
        "feat_mean":         feat_mean.detach().cpu(),
        "feat_std":          feat_std.detach().cpu(),
        "t1":                float(t1),
        "t2":                float(t2),
    }
    for epoch in range(args.epochs):
        encoder.train(); head.train()
        opt.zero_grad()
        z   = encoder(x_tr, ei_tr, et_tr)
        out = head(z[src_tr])
        loss = compute_loss(out, y_act, y_time, y_c, y_n, train_idx, time_weight, act_weight, otherC_weights, otherC_weight_factor, otherN_weight_factor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), max_norm=1.0)
        opt.step()

        encoder.eval(); head.eval()
        with torch.no_grad():
            z_v   = encoder(x_val, ei_val, et_val)
            v_out = head(z_v[src_val])
            m_act = y_act[val_idx] >= 0
            v_loss = (F.cross_entropy(v_out["act"][m_act], y_act[val_idx][m_act], weight=act_weight)
                      if m_act.any() else torch.tensor(float("inf")))
            m_t = torch.isfinite(y_time[val_idx])
            if m_t.any():
                v_loss = v_loss + time_weight * F.mse_loss(v_out["time"][m_t], y_time[val_idx][m_t])

        scheduler.step()
        if v_loss.item() < best_v:
            best_v = v_loss.item()
            no_improve = 0
            # test predictions stored only for offline evaluation; never used for selection
            with torch.no_grad():
                z_te  = encoder(x_te, ei_te, et_te)
                t_out = head(z_te[src_te])
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
                "pairs_val": test_pairs_t,
            }, args.save_best_val)
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "head_state_dict":    head.state_dict(),
                **model_meta,
            }, model_path)
            print(f"Epoch {epoch:4d}  val_loss={v_loss.item():.4f}  ← saved")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"Saved test predictions → {args.save_best_val}")
    print(f"Saved model checkpoint → {model_path}")


if __name__ == "__main__":
    main()
