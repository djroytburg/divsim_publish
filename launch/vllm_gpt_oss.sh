#!/bin/bash
# Launch vLLM server for openai/gpt-oss-20b.
#
# Hardware: 1× L40S 48GB (TP=1). The mxfp4 quantized weights fit in ~40 GB.
# Default port: 8003. Override with PORT env var.
#
# Key settings:
#   --tool-call-parser openai        — gpt-oss uses the OpenAI tool schema
#   --reasoning-parser openai_gptoss — strips thinking tokens from output
#   tool_choice=required             — gpt-oss reliably complies with required

set -euo pipefail

HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_cache/hub}"
export HF_HUB_CACHE

VENV="${SAFEMASS_VENV:-$(dirname "$0")/../.venv}"
PORT="${PORT:-8003}"
LOGS="${LOGS_DIR:-/tmp/vllm_logs}"
mkdir -p "$LOGS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" \
    "${VENV}/bin/python3" -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-20b \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --reasoning-parser openai_gptoss \
    --no-scheduler-reserve-full-isl \
    >> "${LOGS}/vllm_gptoss_${PORT}.log" 2>&1
