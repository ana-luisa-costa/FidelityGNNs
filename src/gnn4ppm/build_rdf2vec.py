import os
import argparse
import time
import random
import json

import numpy as np
from rdflib import Graph
from gensim.models import Word2Vec

from src.utils.uri_helpers import EVENT_RE, P_TIME, parse_ts, order_case_events, usable_event_uris, event_timestamps
from src.utils.io_helpers import load_temporal_boundaries_from_csv

SKIPPED_PREDICATE_PARTS = (
    "rdf-schema", "#type", "/type", "directlyFollows", "contains",
    "hascaseID", "timestamp", "OtherC_id", "EventIDOther"
)

def read_graph(ttl_path):
    g = Graph()
    g.parse(ttl_path, format="turtle")
    return g


def is_token_predicate(predicate):
    return not any(part in predicate for part in SKIPPED_PREDICATE_PARTS)


def collect_case_events(g, usable):
    """Map case id → usable events in temporal order, each as (uri, attribute tokens)."""
    by_case = {}
    timestamps = {}
    for s, p, o in g:
        match = EVENT_RE.search(str(s))
        if not match or str(s) not in usable:
            continue
        p_str = str(p)
        if p_str == P_TIME:
            timestamps[str(s)] = parse_ts(str(o))
        if not is_token_predicate(p_str):
            continue
        position, case_id = int(match.group(1)), match.group(2)
        event = by_case.setdefault(case_id, {}).setdefault(position, (str(s), []))
        event[1].extend([p_str, str(o)])
    return {cid: order_case_events({k: (ev, timestamps.get(ev[0]))
                                    for k, ev in positions.items()})
            for cid, positions in by_case.items()}


def build_full_entity_set(g):
    entities = set()
    for s, p, o in g:
        p_str = str(p)
        if "rdf-schema" in p_str or p_str.endswith("#type") or p_str.endswith("/type"):
            continue
        entities.add(str(s))
        entities.add(str(o))
    return sorted(entities)


def event_walk(tokens, token_drop):
    """Shuffled attribute tokens of one event; other events never co-occur."""
    walk = [t for t in tokens if random.random() >= token_drop]
    random.shuffle(walk)
    return walk


def generate_walks(case_events, num_walks, token_drop):
    walks = []
    for events in case_events.values():
        for _, tokens in events:
            for _ in range(num_walks):
                walk = event_walk(tokens, token_drop)
                if len(walk) > 1:
                    walks.append(walk)
    return walks


def train_word2vec(walks, dim, window, epochs, seed):
    return Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=1,
        workers=os.cpu_count() or 4,
        epochs=epochs,
        sg=1,
        negative=10,
        seed=seed,
    )


def pool_event_vector(wv, past_events, decay):
    """Decayed mean of token vectors; `past_events` is ordered newest first."""
    pooled = np.zeros(wv.vector_size, dtype=np.float32)
    weight, total_weight = 1.0, 0.0
    for _, tokens in past_events:
        vectors = [wv[t] for t in tokens if t in wv]
        if vectors:
            pooled += weight * np.mean(vectors, axis=0)
            total_weight += weight
        weight *= decay
    return pooled / total_weight if total_weight else None


def build_embeddings(g, wv, case_events, max_events, decay):
    """One vector per entity: pooled past for events, the token vector for
    attribute values, and the vocabulary mean for the rest (cases, unseen)."""
    all_entities = build_full_entity_set(g)
    entity2id = {ent: i for i, ent in enumerate(all_entities)}

    unk = wv.vectors.mean(axis=0).astype(np.float32)
    embeddings = np.tile(unk, (len(all_entities), 1))

    for entity, idx in entity2id.items():
        if entity in wv:
            embeddings[idx] = wv[entity]

    for events in case_events.values():
        for position, (uri, _) in enumerate(events):
            idx = entity2id.get(uri)
            if idx is None:
                continue
            past = list(reversed(events[max(0, position + 1 - max_events):position + 1]))
            vector = pool_event_vector(wv, past, decay)
            if vector is not None:
                embeddings[idx] = vector

    return embeddings, entity2id


# main

def main():
    ap = argparse.ArgumentParser(
        description="Leakage-free embeddings from within-event walks")
    ap.add_argument("--ttl",         required=True,  help="Path to the TTL file.")
    ap.add_argument("--csv",         required=True,  help="Path to the event-log CSV (must have caseID_case_concept_name column).")
    ap.add_argument("--out",         default="data/processed/rdf2vec", help="Output directory.")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio",   type=float, default=0.1)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--dim",         type=int,   default=128)
    ap.add_argument("--num_walks",   type=int,   default=10, help="Walks per event.")
    ap.add_argument("--walk_length", type=int,   default=8,  help="Events pooled per event vector (the event plus its predecessors).")
    ap.add_argument("--window",      type=int,   default=20, help="Should cover all tokens of one event.")
    ap.add_argument("--epochs",      type=int,   default=30)
    ap.add_argument("--token_drop",  type=float, default=0.2, help="Per-token dropout for walk diversity.")
    ap.add_argument("--decay",       type=float, default=0.5, help="Weight decay per step into the past when pooling event vectors.")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t_total = time.time()

    t1, t2 = load_temporal_boundaries_from_csv(args.csv, args.train_ratio, args.val_ratio)

    split_path = os.path.join(args.out, "temporal_split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"t1": t1, "t2": t2}, f, indent=2)
    print(f"Saved temporal split → {split_path}")

    print(f"Loading {args.ttl} ...")
    t0 = time.time()
    g = read_graph(args.ttl)
    print(f"  Graph loaded  [{time.time() - t0:.1f}s]")

    usable = usable_event_uris(g)
    case_events = collect_case_events(g, usable)

    # walks only use events before t1, so RDF2Vec never sees val or test events
    timestamps = event_timestamps(g)
    train_case_events = collect_case_events(g, {uri for uri in usable if timestamps[uri] < t1})

    t0 = time.time()
    walks = generate_walks(train_case_events, args.num_walks, args.token_drop)
    print(f"Generated {len(walks)} walks from training-window events  "
          f"[{time.time() - t0:.1f}s]")

    t0 = time.time()
    model = train_word2vec(walks, args.dim, args.window, args.epochs, args.seed)
    print(f"Word2Vec trained on {len(model.wv)} tokens  [{time.time() - t0:.1f}s]")

    embeddings, entity2id = build_embeddings(g, model.wv, case_events,
                                             args.walk_length, args.decay)

    np.save(os.path.join(args.out, "entity_embeddings.npy"), embeddings)
    with open(os.path.join(args.out, "entity2id.json"), "w", encoding="utf-8") as f:
        json.dump(entity2id, f, indent=2)

    print(f"Saved embeddings shape={embeddings.shape} → {args.out}")
    print(f"\nTotal time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
