#!/usr/bin/env bash
#SBATCH --job-name=train-value
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-119%8
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results
source slurm/vllm.sh

SEED=${SLURM_ARRAY_TASK_ID:-0}
OUT=results/value_s${SEED}
[ -f "$OUT/head.pt" ] && { echo "seed $SEED done"; exit 0; }

require_cuda
python -u scripts/train_value.py \
  --labels data/labels/labels_native_v6.json data/labels/labels_native_q14b.json \
  --model Qwen/Qwen3-1.7B \
  --epochs 4 --batch 8 --max-len 1024 --lr 1e-4 \
  --lam 1.0 --eta 1.0 \
  --val-frac 0.20 --test-frac 0.20 --split-seed "$SEED" \
  --grad-checkpointing \
  --centre-targets --rank-weight 0 \
  --out "$OUT"
