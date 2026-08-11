#!/bin/bash
#SBATCH --job-name=rgcn_train
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd ~/RGCN4PPM
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START: $(date)"

time python3 -u src/train_2.py \
  --ttl data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --save_best_val data/processed/BPIC13_O/best_val.pt

echo "END: $(date)"
