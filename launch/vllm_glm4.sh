#!/bin/bash
# Launch vLLM server for zai-org/GLM-4-32B-0414.
#
# Hardware: 2× L40S 48GB (TP=2).
# Default port: 8007. Override with PORT env var.
#
# Key settings:
#   --tool-call-parser hermes   — GLM-4-32B-0414 uses the Hermes tool schema
#   tool_choice=required        — GLM-4 returns plain text with 'auto';
#                                 'required' is necessary for reliable tool calls
#
# Warm-up check: the simulation runner verifies ≥50% of GLM-4 agents post in
# warm-up. If this check fails, the most likely cause is the server not being
# ready or returning malformed tool calls.

set -euo pipefail

HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_cache/hub}"
export HF_HUB_CACHE

VENV="${SAFEMASS_VENV:-$(dirname "$0")/../.venv}"
PORT="${PORT:-8007}"
LOGS="${LOGS_DIR:-/tmp/vllm_logs}"
mkdir -p "$LOGS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}" \
    "${VENV}/bin/python3" -m vllm.entrypoints.openai.api_server \
    --model zai-org/GLM-4-32B-0414 \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --no-scheduler-reserve-full-isl \
    >> "${LOGS}/vllm_glm4_${PORT}.log" 2>&1
