#!/bin/bash
# Submit one fidelity job per explainer, then merge and plot after all succeed.

set -euo pipefail

EXPLAINERS="${EXPLAINERS:-gradient,ig,lime,shap,gnnexplainer,pgmexplainer}"
FIDELITY_METRICS="${FIDELITY_METRICS:-prob}"
BASE_DIR="${BASE_DIR:-plots_parallel}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
TOP_K_VALUES="${TOP_K_VALUES:-5,10,15,20,25}"
TASKS="${TASKS:-activity,resource,role,lifecycle}"
HETEROGENEITY_MODE="${HETEROGENEITY_MODE:-full}"
CONFIG="${CONFIG:-src/fidelity/default_config.json}"
FORMATS="${FORMATS:-png}"
SEEDS="${SEEDS:-42,43,44}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

mkdir -p logs "${BASE_DIR}"
echo "heterogeneity_mode=${HETEROGENEITY_MODE}"

job_ids=()
IFS=',' read -ra names <<< "${EXPLAINERS}"
IFS=',' read -ra seeds <<< "${SEEDS}"
for raw_seed in "${seeds[@]}"; do
  seed="$(echo "${raw_seed}" | xargs)"
  [[ -z "${seed}" ]] && continue
  for raw_name in "${names[@]}"; do
    name="$(echo "${raw_name}" | xargs)"
    [[ -z "${name}" ]] && continue
    out_dir="${BASE_DIR}/seed_${seed}/${name}"
    job_id="$(
      CONFIG="${CONFIG}" \
      OUT_DIR="${out_dir}" \
      NUM_SAMPLES="${NUM_SAMPLES}" \
      EXPLAINERS="${name}" \
      FIDELITY_METRICS="${FIDELITY_METRICS}" \
      TOP_K_VALUES="${TOP_K_VALUES}" \
      TASKS="${TASKS}" \
      HETEROGENEITY_MODE="${HETEROGENEITY_MODE}" \
      SEED="${seed}" \
      SAMPLE_SEED="${SAMPLE_SEED}" \
      sbatch --parsable scripts/run_fidelity_curves.slurm
    )"
    job_ids+=("${job_id}")
    echo "submitted seed=${seed} ${name}: ${job_id} -> ${out_dir}"
  done
done

if [[ "${#job_ids[@]}" -eq 0 ]]; then
  echo "no fidelity jobs submitted; check SEEDS and EXPLAINERS" >&2
  exit 1
fi

dependency="$(IFS=:; echo "${job_ids[*]}")"
merge_job="$(
  BASE_DIR="${BASE_DIR}" \
  MERGED_DIR="${BASE_DIR}/merged" \
  EXPLAINERS="${EXPLAINERS}" \
  FORMATS="${FORMATS}" \
  SEEDS="${SEEDS}" \
  sbatch --parsable --dependency="afterok:${dependency}" scripts/run_fidelity_merge_plot.slurm
)"

echo "submitted merge+plot: ${merge_job} afterok:${dependency}"
echo "watch: squeue -u \$USER"
echo "logs:  ls -lt logs/"
echo "final: ${BASE_DIR}/merged"
