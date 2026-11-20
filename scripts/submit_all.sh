#!/bin/bash
# Submits one job per (run, dataset) simultaneously.
# Skips BPIC13_I and BPIC17_O (no raw data).
# Logs are saved to logs/ directory.

mkdir -p logs

for RUN in 30_10_30 100_30_20 200_30_50; do
  for DS in BPIC12_A BPIC12_W BPIC12_WC BPIC13_O BPIC20_P BPIC20_R; do
    sbatch --job-name="${DS}_${RUN}" --export=ALL,DS=$DS,RUN=$RUN scripts/run_dataset.sbatch
  done
done

echo "18 jobs submitted (3 runs × 6 datasets). Check logs/ for output."
