#!/usr/bin/env bash
#SBATCH --job-name=label-toolsandbox
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results
source slurm/vllm.sh

MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}
TAG=${TAG:-${SLURM_JOB_ID:-local}}

require_cuda
start_vllm "$MODEL" "$PORT" "logs/vllm_${TAG}.log"

python -u scripts/label_toolsandbox.py \
  --scenarios "${SCENARIOS:-80}" \
  --continue-for "${CONTINUE_FOR:-30}" \
  --workers "${WORKERS:-12}" \
  --task-timeout 600 \
  --out "results/labels_toolsandbox_${TAG}.json" \
  ${EXTRA_ARGS:-}
