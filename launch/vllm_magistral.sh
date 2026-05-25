#!/bin/bash
# Launch vLLM server for mistralai/Magistral-Small-2509.
#
# Hardware: 2× L40S 48GB (TP=2).
# Default port: 8004. Override with PORT env var.
#
# Key settings:
#   --tokenizer-mode mistral   \
#   --config-format mistral    \
#   --load-format mistral      — Mistral models require their own tokenizer/loader
#   --tool-call-parser mistral — uses Mistral tool-call format
#   --reasoning-parser mistral — strips Magistral thinking tokens
#   tool_choice=auto           — Magistral returns empty tool_calls with 'required';
#                                'auto' is the correct setting for this model

set -euo pipefail

HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_cache/hub}"
export HF_HUB_CACHE

VENV="${SAFEMASS_VENV:-$(dirname "$0")/../.venv}"
PORT="${PORT:-8004}"
LOGS="${LOGS_DIR:-/tmp/vllm_logs}"
mkdir -p "$LOGS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}" \
    "${VENV}/bin/python3" -m vllm.entrypoints.openai.api_server \
    --model mistralai/Magistral-Small-2509 \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --reasoning-parser mistral \
    --no-scheduler-reserve-full-isl \
    >> "${LOGS}/vllm_magistral_${PORT}.log" 2>&1
