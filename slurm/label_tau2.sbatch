#!/usr/bin/env bash
#SBATCH --job-name=label-tau2
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
DOMAIN=${DOMAIN:-retail}
PORT=${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}
TAG=${TAG:-${SLURM_JOB_ID:-local}}

require_cuda
start_vllm "$MODEL" "$PORT" "logs/vllm_${TAG}.log"

python -u scripts/label_tau2.py \
  --domain "$DOMAIN" \
  --tasks "${TASKS:-114}" \
  --fork-at "${FORK_AT:-4}" \
  --continue-for "${CONTINUE_FOR:-45}" \
  --workers "${WORKERS:-6}" \
  --out "results/labels_tau2_${DOMAIN}_${TAG}.json"
