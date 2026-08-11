#!/bin/bash
#SBATCH --job-name=gnn4ppm_fidplot
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

OUT_DIR="${OUT_DIR:-plots_fidelity}"
SUMMARY="${SUMMARY:-${OUT_DIR}/fidelity_curves_summary.csv}"
FORMATS="${FORMATS:-png}"

echo "host=$(hostname)"
echo "start=$(date)"
echo "summary=${SUMMARY}"

time python3 -u -m src.fidelity.plot_fidelity_curves \
  --summary "${SUMMARY}" \
  --out "${OUT_DIR}" \
  --formats "${FORMATS}"

echo "end=$(date)"
