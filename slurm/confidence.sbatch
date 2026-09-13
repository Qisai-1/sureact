#!/usr/bin/env bash
#SBATCH --job-name=confidence
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results
source slurm/vllm.sh

MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}

require_cuda
start_vllm "$MODEL" "$PORT" "logs/vllm_${SLURM_JOB_ID:-local}.log"

python -u scripts/compute_confidence.py \
  data/labels/labels_native_v6.json data/labels/labels_native_q14b.json \
  --out results/confidence.json
