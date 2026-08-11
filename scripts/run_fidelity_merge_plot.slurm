#!/bin/bash
#SBATCH --job-name=gnn4ppm_fidmerge
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs

if [[ -d "${PROJECT_DIR}/.venv" ]]; then
  source "${PROJECT_DIR}/.venv/bin/activate"
elif [[ -d "${PROJECT_DIR}/venv" ]]; then
  source "${PROJECT_DIR}/venv/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn4ppm_mpl_cache"

BASE_DIR="${BASE_DIR:-plots_parallel}"
MERGED_DIR="${MERGED_DIR:-${BASE_DIR}/merged}"
EXPLAINERS="${EXPLAINERS:-gradient,ig,lime,shap,gnnexplainer,pgmexplainer}"
FORMATS="${FORMATS:-png}"
SEEDS="${SEEDS:-42,43,44}"

echo "host=$(hostname)"
echo "start=$(date)"
echo "base=${BASE_DIR}"
echo "merged=${MERGED_DIR}"
echo "seeds=${SEEDS}"

time python3 -u scripts/merge_fidelity_outputs.py \
  --base "${BASE_DIR}" \
  --out "${MERGED_DIR}" \
  --explainers "${EXPLAINERS}" \
  --seeds "${SEEDS}"

time python3 -u -m src.fidelity.plot_fidelity_curves \
  --summary "${MERGED_DIR}/fidelity_curves_summary.csv" \
  --out "${MERGED_DIR}" \
  --formats "${FORMATS}"

echo "end=$(date)"
