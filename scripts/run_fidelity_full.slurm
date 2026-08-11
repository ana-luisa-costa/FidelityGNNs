#!/bin/bash
#SBATCH --job-name=gnn4ppm_fidfull
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn4ppm_mpl_cache"

CONFIG="${CONFIG:-src/fidelity/default_config.json}"
OUT_DIR="${OUT_DIR:-plots_fidelity}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
EXPLAINERS="${EXPLAINERS:-gradient,ig,lime,shap,gnnexplainer,pgmexplainer}"
FIDELITY_METRICS="${FIDELITY_METRICS:-prob}"
TOP_K_VALUES="${TOP_K_VALUES:-5,10,15,20,25}"
TASKS="${TASKS:-activity,resource,role,lifecycle}"
HETEROGENEITY_MODE="${HETEROGENEITY_MODE:-full}"
FORMATS="${FORMATS:-png}"
SEED="${SEED:-42}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

echo "host=$(hostname)"
echo "start=$(date)"
echo "project=${PROJECT_DIR}"
echo "out=${OUT_DIR}"
echo "seed=${SEED}"
echo "sample_seed=${SAMPLE_SEED}"
echo "heterogeneity_mode=${HETEROGENEITY_MODE}"

time python3 -u -m src.fidelity.compute_fidelity_curves \
  --config "${CONFIG}" \
  --out "${OUT_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --explainers "${EXPLAINERS}" \
  --fidelity-metrics "${FIDELITY_METRICS}" \
  --top-k-values "${TOP_K_VALUES}" \
  --tasks "${TASKS}" \
  --heterogeneity-mode "${HETEROGENEITY_MODE}" \
  --seed "${SEED}" \
  --sample-seed "${SAMPLE_SEED}"

time python3 -u -m src.fidelity.plot_fidelity_curves \
  --summary "${OUT_DIR}/fidelity_curves_summary.csv" \
  --out "${OUT_DIR}" \
  --formats "${FORMATS}"

echo "end=$(date)"
