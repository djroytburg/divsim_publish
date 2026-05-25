#!/bin/bash
# Launch vLLM server for google/gemma-4-31B-it.
#
# Hardware: 2× L40S 48GB (TP=2). Note: Gemma shares the same GPU pair (0-1 by
# default here) as Magistral in the paper's triadic-sweep node layout. They
# cannot run simultaneously; stop one before starting the other.
# Default port: 8009. Override with PORT env var.
#
# Key settings:
#   --tool-call-parser gemma4   — requires the custom jinja chat template
#   --chat-template             — path to tool_chat_template_gemma4.jinja
#   tool_choice=auto            — Gemma 4 returns empty tool_calls with 'required'
#   --max-num-seqs 64           — tuned to avoid KV-cache saturation under
#                                 concurrent multi-agent load
#   --no-scheduler-reserve-full-isl — avoids a vLLM 0.19.0 scheduler deadlock
#
# Note on max_tokens: Gemma 4 output is capped at 512 tokens in the runner
# (compared to 1536 for other models) to limit KV-cache pressure.

set -euo pipefail

HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_cache/hub}"
export HF_HUB_CACHE

VENV="${SAFEMASS_VENV:-$(dirname "$0")/../.venv}"
PORT="${PORT:-8009}"
LOGS="${LOGS_DIR:-/tmp/vllm_logs}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${GEMMA_CHAT_TEMPLATE:-${SCRIPT_DIR}/../oasis/tool_chat_template_gemma4.jinja}"
mkdir -p "$LOGS"

if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: Gemma chat template not found at: $TEMPLATE" >&2
    echo "Set GEMMA_CHAT_TEMPLATE env var or ensure oasis/tool_chat_template_gemma4.jinja exists." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    "${VENV}/bin/python3" -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31B-it \
    --served-model-name google/gemma-4-31B-it \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --chat-template "$TEMPLATE" \
    --max-num-seqs 64 \
    --no-scheduler-reserve-full-isl \
    >> "${LOGS}/vllm_gemma_${PORT}.log" 2>&1
