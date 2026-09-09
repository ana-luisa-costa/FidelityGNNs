import os
import argparse
import time
import random
import json

import numpy as np
from rdflib import Graph
from gensim.models import Word2Vec
import optuna
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans

from src.utils.uri_helpers import get_case_of_entity
from src.utils.io_helpers import load_case_split_from_csv



def read_graph(ttl_path):
    g = Graph()
    g.parse(ttl_path, format="turtle")
    return g


def build_adjacency(g, train_cases):
    adj = {}          # entity -> [(neighbor, predicate), ...]
    neighbor_sets = {}  # entity -> set of neighbor entities (for Node2Vec bias)
    entity_set = set()

    for s, p, o in g:
        s_str, p_str, o_str = str(s), str(p), str(o)

        if "rdf-schema" in p_str or p_str.endswith("#type") or p_str.endswith("/type"):
            continue
        if "directlyFollows" in p_str:
            continue  # aggregates the full log (train+test); walks stay train-case only

        s_case = get_case_of_entity(s_str)
        o_case = get_case_of_entity(o_str)

        if (s_case is not None and s_case not in train_cases) or \
           (o_case is not None and o_case not in train_cases):
            continue

        entity_set.add(s_str)
        entity_set.add(o_str)

        if s_str not in adj:
            adj[s_str] = []
            neighbor_sets[s_str] = set()
        if o_str not in adj:
            adj[o_str] = []
            neighbor_sets[o_str] = set()

        adj[s_str].append((o_str, p_str))
        adj[o_str].append((s_str, p_str))
        neighbor_sets[s_str].add(o_str)
        neighbor_sets[o_str].add(s_str)

    return adj, sorted(entity_set), neighbor_sets


def build_full_entity_set(g):
    entities = set()
    for s, p, o in g:
        p_str = str(p)
        if "rdf-schema" in p_str or p_str.endswith("#type") or p_str.endswith("/type"):
            continue
        entities.add(str(s))
        entities.add(str(o))
    return sorted(entities)


def biased_walk(adj, neighbor_sets, start_entity, walk_length, p=1.0, q=1.0):
    walk = [start_entity]
    prev = None
    current = start_entity

    for _ in range(walk_length - 1):
        neighbors = adj.get(current, [])
        if not neighbors:
            break

        if prev is None or (p == 1.0 and q == 1.0):
            next_entity, predicate = random.choice(neighbors)
        else:
            prev_neighbors = neighbor_sets.get(prev, set())
            weights = []
            for nb, _ in neighbors:
                if nb == prev:
                    weights.append(1.0 / p)  # return to previous
                elif nb in prev_neighbors:
                    weights.append(1.0)      # BFS (shared neighbor)
                else:
                    weights.append(1.0 / q)  # DFS (exploration)

            chosen = random.choices(neighbors, weights=weights, k=1)[0]
            next_entity, predicate = chosen

        walk.append(predicate)
        walk.append(next_entity)
        prev = current
        current = next_entity

    return walk


def generate_walks(adj, neighbor_sets, entities, num_walks, walk_length,
                   p=1.0, q=1.0):
    all_walks = []
    for entity in entities:
        for _ in range(num_walks):
            walk = biased_walk(adj, neighbor_sets, entity, walk_length, p, q)
            if len(walk) > 2:  # at least one entity-predicate-entity triple
                all_walks.append(walk)
    return all_walks


def train_word2vec(walks, dim, window, epochs):
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=1,
        workers=os.cpu_count() or 4,
        epochs=epochs,
        sg=0,
        negative=10,
        seed=42,
    )
    return model


def extract_embeddings(model, all_entities, train_entities):
    dim = model.wv.vector_size
    entity2id = {ent: i for i, ent in enumerate(all_entities)}

    # mean over shared (non case/event instance) train entities only
    train_vectors = []
    for entity in train_entities:
        if entity in model.wv and get_case_of_entity(entity) is None:
            train_vectors.append(model.wv[entity])

    if train_vectors:
        mean_embedding = np.mean(train_vectors, axis=0).astype(np.float32)
    else:
        mean_embedding = np.zeros(dim, dtype=np.float32)

    embeddings = np.full((len(all_entities), dim), mean_embedding, dtype=np.float32)
    for entity, idx in entity2id.items():
        # case/event instance nodes keep the neutral mean embedding: their
        # learned vectors encode the complete case, including future events
        if get_case_of_entity(entity) is not None:
            continue
        if entity in model.wv:
            embeddings[idx] = model.wv[entity]

    return embeddings, entity2id


def evaluate_embeddings(embeddings):
    if len(embeddings) > 10000:
        idx = np.random.choice(len(embeddings), 10000, replace=False)
        embeddings = embeddings[idx]
    kmeans = MiniBatchKMeans(n_clusters=10, n_init=3, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return silhouette_score(embeddings, labels)


def objective(trial, adj, neighbor_sets, train_entities):
    dim         = trial.suggest_categorical("dim", [64, 128, 256])
    num_walks   = trial.suggest_int("num_walks", 10, 50)
    walk_length = trial.suggest_int("walk_length", 6, 12)
    window      = trial.suggest_int("window", 3, 8)
    p           = trial.suggest_float("p", 0.25, 4.0, log=True)
    q           = trial.suggest_float("q", 0.25, 4.0, log=True)

    walks = generate_walks(adj, neighbor_sets, train_entities, num_walks,
                           walk_length, p, q)
    model = train_word2vec(walks, dim, window, epochs=10)
    embeddings, _ = extract_embeddings(model, train_entities, train_entities)
    return evaluate_embeddings(embeddings)


# main

def main():
    ap = argparse.ArgumentParser(description="Generate RDF2Vec embeddings")
    ap.add_argument("--ttl",         required=True,  help="Path to the TTL file.")
    ap.add_argument("--csv",         required=True,  help="Path to the event-log CSV (must have caseID_case_concept_name column).")
    ap.add_argument("--out",         default="data/processed/rdf2vec", help="Output directory.")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--dim",         type=int,   default=128)
    ap.add_argument("--num_walks",   type=int,   default=20)
    ap.add_argument("--walk_length", type=int,   default=10)
    ap.add_argument("--window",      type=int,   default=5)
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--p",           type=float, default=1.0, help="Node2Vec return parameter (high=less backtrack).")
    ap.add_argument("--q",           type=float, default=1.0, help="Node2Vec in-out parameter (low=BFS, high=DFS).")
    ap.add_argument("--tune",        action="store_true", help="Run Optuna hyperparameter search.")
    ap.add_argument("--trials",      type=int,   default=30)
    ap.add_argument("--no-temporal",  dest="temporal", action="store_false", help="Use random split instead of the default chronological split.")
    ap.set_defaults(temporal=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    t_total = time.time()

    t0 = time.time()
    train_cases, test_cases, _ = load_case_split_from_csv(args.csv, args.train_ratio, args.seed, temporal=args.temporal)

    split_path = os.path.join(args.out, "case_split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train": sorted(train_cases), "test": sorted(test_cases)}, f, indent=2)
    print(f"Saved case split → {split_path}  [{time.time() - t0:.1f}s]")

    print(f"Loading {args.ttl} ...")
    t0 = time.time()
    g = read_graph(args.ttl)
    print(f"  Graph loaded  [{time.time() - t0:.1f}s]")

    print("Building train-only adjacency ...")
    t0 = time.time()
    adj, train_entities, neighbor_sets = build_adjacency(g, train_cases)
    print(f"  Train-graph entities: {len(train_entities)}  [{time.time() - t0:.1f}s]")

    t0 = time.time()
    all_entities = build_full_entity_set(g)
    print(f"  Total entities (all): {len(all_entities)}  [{time.time() - t0:.1f}s]")

    # hyoerparameter
    if args.tune:
        print(f"Starting Optuna optimisation ({args.trials} trials) ...")
        t0 = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, adj, neighbor_sets, train_entities),
            n_trials=args.trials,
        )
        best = study.best_params
        print(f"Best params: {best}  [{time.time() - t0:.1f}s]")
        dim, n_walks, w_len, win = best["dim"], best["num_walks"], best["walk_length"], best["window"]
        p_val, q_val = best["p"], best["q"]
    else:
        dim, n_walks, w_len, win = args.dim, args.num_walks, args.walk_length, args.window
        p_val, q_val = args.p, args.q


    print(f"\nFinal run: dim={dim}, walks={n_walks}, length={w_len}, window={win}, p={p_val:.2f}, q={q_val:.2f}")
    t0 = time.time()
    walks = generate_walks(adj, neighbor_sets, train_entities, n_walks, w_len,
                           p_val, q_val)
    print(f"  Walks generated  [{time.time() - t0:.1f}s]")
    t0 = time.time()
    model = train_word2vec(walks, dim, win, args.epochs)
    print(f"  Word2Vec trained  [{time.time() - t0:.1f}s]")

    embeddings, entity2id = extract_embeddings(model, all_entities, train_entities)

    np.save(os.path.join(args.out, "entity_embeddings.npy"), embeddings)
    with open(os.path.join(args.out, "entity2id.json"), "w", encoding="utf-8") as f:
        json.dump(entity2id, f, indent=2)

    print(f"Saved embeddings shape={embeddings.shape} → {args.out}")
    print(f"Case/event instance nodes (train and test) initialized with mean train embedding")
    print(f"\nTotal time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()

# Best parameters:
# BPIC12_A: {'dim': 256, 'num_walks': 12, 'walk_length': 7, 'window': 3}: Best is trial 6 with value: 0.0766458585858345
# BPIC12_W: Best Parameters Found: {'dim': 128, 'num_walks': 11, 'walk_length': 6, 'window': 5}: Best is trial 9 with value: 0.04825189337134361
# BPIC13_O: Best Parameters Found: {'dim': 256, 'num_walks': 10, 'walk_length': 6, 'window': 4}: Best is trial 33 with value: 0.17513097822666168
# BPIC13_I: Best Parameters Found: {'dim': 256, 'num_walks': 10, 'walk_length': 6, 'window': 4}
# BPIC17_O: Best Parameters Found: {'dim': 128, 'num_walks': 7, 'walk_length': 7, 'window': 3}: Best is trial 1 with value: 0.06424785405397415
# BPIC20_P: Best Parameters Found: {'dim': 128, 'num_walks': 14, 'walk_length': 8, 'window': 3}: Best is trial 3 with value: 0.06942684203386307
# BPIC20_R: Best Parameters Found: {'dim': 256, 'num_walks': 14, 'walk_length': 7, 'window': 5}: Best is trial 5 with value: 0.07953798770904541
