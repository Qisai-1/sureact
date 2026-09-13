#!/usr/bin/env bash
#SBATCH --job-name=run-baselines
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results
source slurm/vllm.sh

require_cuda
python -u scripts/run_baselines.py \
  data/labels/labels_native_v6.json data/labels/labels_native_q14b.json \
  --confidence data/confidence_headline.json \
  --test-frac 0.20 --seeds 120 \
  --value-model-pattern 'results/value_s{seed}' --value-base Qwen/Qwen3-1.7B \
  --out results/headtohead.json
