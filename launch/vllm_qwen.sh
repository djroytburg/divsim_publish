#!/bin/bash
# Launch vLLM server for Qwen/Qwen3-32B.
#
# Hardware: 2× L40S 48GB (TP=2), GPUs selectable via CUDA_VISIBLE_DEVICES.
# Default port: 8002. Override with PORT env var.
#
# Key settings:
#   --tool-call-parser hermes        — Qwen3-32B uses the Hermes tool schema
#   chat_template_kwargs enable_thinking=False — suppress reasoning prefix tokens;
#     Qwen3 is a hybrid reasoning model but we want action output only
#   tool_choice=required             — Qwen reliably complies; forced to prevent
#     plain-text fallback

set -euo pipefail

HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_cache/hub}"
export HF_HUB_CACHE

VENV="${SAFEMASS_VENV:-$(dirname "$0")/../.venv}"
PORT="${PORT:-8002}"
LOGS="${LOGS_DIR:-/tmp/vllm_logs}"
mkdir -p "$LOGS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    "${VENV}/bin/python3" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --no-scheduler-reserve-full-isl \
    >> "${LOGS}/vllm_qwen_${PORT}.log" 2>&1
