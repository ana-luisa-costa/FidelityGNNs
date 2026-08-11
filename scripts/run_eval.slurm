#!/bin/bash
#SBATCH --job-name=rgcn_eval
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd ~/RGCN4PPM
source venv/bin/activate

echo "START: $(date)"

time python3 -u src/evaluate_2.py \
  --best data/processed/BPIC20_P/best_val.pt \
  --output data/processed/BPIC20_P/evaluation.txt \
  --vocabs data/processed/BPIC20_P/best_val_vocabs.json \
  --entity2id data/raw/BPIC20_P/entity2id.json \
  --show_examples 5

echo "END: $(date)"
