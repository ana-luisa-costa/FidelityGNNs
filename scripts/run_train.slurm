#!/bin/bash
#SBATCH --job-name=rgcn_train
#SBATCH --partition=compute
#SBATCH --nodelist=gpu-nvidia-h100
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
  --ttl data/raw/BPIC20_P/BPIC20_P.ttl \
  --emb data/raw/BPIC20_P/entity_embeddings.npy \
  --entity2id data/raw/BPIC20_P/entity2id.json \
  --case_split data/raw/BPIC20_P/case_split.json \
  --save_best_val data/processed/BPIC20_P/best_val.pt

echo "END: $(date)"

