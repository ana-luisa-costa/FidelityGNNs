
import argparse, sys, os, subprocess, glob

def find_file(dataset, ext, dir="data/raw"):
    path = os.path.join(dir, dataset)
    if not os.path.exists(path): raise FileNotFoundError(f"Dataset not found: {path}")
    files = glob.glob(os.path.join(path, f"*.{ext}"))
    if not files: raise FileNotFoundError(f"No {ext.upper()} file in {path}")
    if len(files) > 1: print(f"Multiple files found, using: {files[0]}")
    return files[0]

def run_cmd(cmd, name):
    print(f"\n{'='*70}\nStep: {name}\nCommand: {' '.join(cmd)}\n{'='*70}\n")
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        subprocess.run(cmd, check=True, env=env)
        print(f"✓ {name} completed\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} failed (exit {e.returncode})\n")
        return False

def build_cmd(step, dataset, ttl, csv=None, seed=None):
    if step == "build_rdf2vec": return ["python3", "-u", "src/gnn4ppm/build_rdf2vec.py", "--ttl", ttl, "--csv", csv, "--out", f"data/raw/{dataset}", "--seed", str(seed)]
    if step == "train": return ["python3", "-u", "src/gnn4ppm/train.py", "--ttl", ttl, "--emb", f"data/raw/{dataset}/entity_embeddings.npy", "--entity2id", f"data/raw/{dataset}/entity2id.json", "--case_split", f"data/raw/{dataset}/case_split.json", "--save_best_test", f"data/processed/{dataset}/best_val.pt"]
    if step == "evaluate": return ["python3", "-u", "src/gnn4ppm/evaluate.py", "--best", f"data/processed/{dataset}/best_val.pt", "--vocabs", f"data/processed/{dataset}/best_val_vocabs.json", "--entity2id", f"data/raw/{dataset}/entity2id.json", "--output", f"data/processed/{dataset}/evaluation.txt"]

def main():
    p = argparse.ArgumentParser(description="Run RGCN4PPM pipeline")
    p.add_argument("dataset")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    os.makedirs(f"data/processed/{a.dataset}", exist_ok=True)
    os.makedirs(f"data/raw/{a.dataset}", exist_ok=True)
    try:
        ttl = find_file(a.dataset, "ttl")
        csv = find_file(a.dataset, "csv")
        print(f"Dataset: {a.dataset}\nTTL: {ttl}\nCSV: {csv}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    steps = [("build_rdf2vec", a.skip_build, build_cmd("build_rdf2vec", a.dataset, ttl, csv, a.seed)), ("train", a.skip_train, build_cmd("train", a.dataset, ttl)), ("evaluate", a.skip_eval, build_cmd("evaluate", a.dataset, ttl))]
    for name, skip, cmd in steps:
        if skip: print(f"⊘ Skipping {name}\n")
        elif not run_cmd(cmd, name): print(f"Pipeline failed at {name}", file=sys.stderr); return 1
    print(f"{'='*70}\nPipeline completed for {a.dataset}\n{'='*70}\n")
    return 0

if __name__ == "__main__": sys.exit(main())
