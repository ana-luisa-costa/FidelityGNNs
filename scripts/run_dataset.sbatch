#!/bin/bash
#SBATCH --job-name=fid
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Usage: sbatch --job-name=<DS>_<RUN> --export=ALL,DS=BPIC13_O,RUN=100_30_20 scripts/run_dataset.sbatch
: "${DS:?set DS}"; : "${RUN:?set RUN}"
KS="${KS:-1,2,3,4,5,6,7,8,9,10}"
IFS=_ read -r EP OPT TR <<< "$RUN"

declare -A TTL=(
  [BPIC12_A]=BPIC12_Subprocess_Application [BPIC12_W]=BPIC12_Subprocess_Work
  [BPIC12_WC]=BPIC12_WC [BPIC13_O]=BPIC13_OpenProblems
  [BPIC20_P]=BPIC20_P [BPIC20_R]=BPIC20_Request )

RAW=data/raw/$DS
OUT=data/processed_$RUN/$DS

# Skip missing datasets
if [ ! -d "$RAW" ]; then
  echo "Dataset $DS not found in $RAW; skipping."
  exit 0
fi

# Create output directory
mkdir -p "$OUT"

# Train if model doesn't exist
if [ ! -f "$OUT/best_val_model.pt" ]; then
  echo "Training $DS ..."
  python src/gnn4ppm/train.py \
    --ttl "$RAW/${TTL[$DS]}.ttl" --emb "$RAW/entity_embeddings.npy" \
    --entity2id "$RAW/entity2id.json" --temporal_split "$RAW/temporal_split.json" \
    --save_best_val "$OUT/best_val.pt" --save_model "$OUT/best_val_model.pt" \
    --vocabs_out "$OUT/best_val_vocabs.json" \
    --epochs "$EP" --optuna_epochs "$OPT" --trials "$TR"
fi

# Evaluate
echo "Evaluating $DS ..."
python src/gnn4ppm/evaluate.py --best "$OUT/best_val.pt" \
  --output "$OUT/evaluation.txt" --vocabs "$OUT/best_val_vocabs.json" \
  --entity2id "$RAW/entity2id.json" || true

# Compute fidelity curves for all heterogeneity modes
for MODE in full middle homogeneous; do
  echo "Computing fidelity curves ($MODE) for $DS with k=$KS ..."
  mkdir -p "$OUT/fidelity_curves_${MODE}_n20"
  python src/fidelity/compute_fidelity_curves.py \
    --config src/fidelity/configs/$DS.json \
    --model "$OUT/best_val_model.pt" --vocabs "$OUT/best_val_vocabs.json" \
    --best-test "$OUT/best_val.pt" \
    --out "$OUT/fidelity_curves_${MODE}_n20" \
    --heterogeneity-mode "$MODE" --top-k-values "$KS" --num-samples 20
done

echo "Done with $DS ($RUN)"
