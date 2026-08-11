#!/bin/bash
#SBATCH --job-name=rdf2vec_cpu
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd ~/RGCN4PPM
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "HOST: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "START: $(date)"

time python3 -u src/build_rdf2vec_2.py \
  --ttl data/raw/BPIC20_P/BPIC20_P.ttl \
  --csv data/raw/BPIC20_P/BPIC20_P.csv \
  --out data/raw/BPIC20_P

echo "END: $(date)"
