#!/usr/bin/env bash

start_vllm() {
  local model=$1 port=$2 log=$3
  vllm serve "$model" --port "$port" --served-model-name "$model" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
    > "$log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

  for _ in $(seq 1 180); do
    if curl -sf -m 5 "http://localhost:${port}/v1/models" > /dev/null; then
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      tail -20 "$log" >&2
      return 1
    fi
    sleep 10
  done
  curl -sf -m 5 "http://localhost:${port}/v1/models" > /dev/null || return 1

  export SUREACT_LLM_BASE_URL="http://localhost:${port}/v1"
  export SUREACT_LLM_MODEL="$model"
  export HOSTED_VLLM_API_BASE="$SUREACT_LLM_BASE_URL"
  export HOSTED_VLLM_API_KEY=EMPTY
  export SUREACT_TEMPERATURE=0
  export SUREACT_SEED=0
}

require_cuda() {
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "CUDA unavailable on $(hostname); resubmit with --exclude=$(hostname)" >&2
    exit 3
  }
}
