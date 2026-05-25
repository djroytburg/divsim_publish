#!/bin/bash
# Smoke test: run one short dyadic simulation and verify basic outputs.
#
# Spec: 1 dyad (Qwen + Magistral), 20 steps, 4 agents total (2 per family).
# Expected runtime: < 5 minutes on 2× L40S GPUs.
#
# Required env vars:
#   VLLM_URL_A   — Qwen3-32B server URL (default: http://localhost:8002/v1)
#   VLLM_URL_B   — Magistral server URL (default: http://localhost:8004/v1)
#   SAFEMASS_VENV — path to Python venv
#   SEED_DB       — path to hf-zero-50.sqlite (default: ../oasis/seed_db/hf-zero-50.sqlite)
#
# After the run, assert_h_matrix.py is called to verify the output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV="${SAFEMASS_VENV:-${ROOT_DIR}/.venv}"
RUNNER="${OASIS_RUNNER:-${ROOT_DIR}/oasis/run_v9.py}"
SEED_DB="${SEED_DB:-${ROOT_DIR}/oasis/seed_db/hf-zero-50.sqlite}"
RUNS_DIR="${RUNS_DIR:-/tmp/divsim_smoke_runs}"
LOGS_DIR="${LOGS_DIR:-/tmp/divsim_smoke_logs}"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}"

export SEED_DB
export MODEL_NAME_A="${MODEL_NAME_A:-Qwen/Qwen3-32B}"
export MODEL_NAME_B="${MODEL_NAME_B:-mistralai/Magistral-Small-2509}"
export VLLM_URL_A="${VLLM_URL_A:-http://localhost:8002/v1}"
export VLLM_URL_B="${VLLM_URL_B:-http://localhost:8004/v1}"

DB="${RUNS_DIR}/smoke_qwen_mag_s42.db"
export OASIS_DB_PATH="${DB}"

echo "[smoke] Starting dyadic smoke run"
echo "  A: ${MODEL_NAME_A} @ ${VLLM_URL_A}"
echo "  B: ${MODEL_NAME_B} @ ${VLLM_URL_B}"
echo "  db: ${DB}"
echo "  steps=20  agents=4  seed=42"

"${VENV}/bin/python3" "${RUNNER}" \
    --mode hetero \
    --steps 20 \
    --agents 4 \
    --seed-posts 4 \
    --activate-prob 0.5 \
    --rng-seed 42 \
    --db "${DB}" \
    2>&1 | tee "${LOGS_DIR}/smoke_run.log"

echo ""
echo "[smoke] Run complete. Running assertions..."
"${VENV}/bin/python3" "${SCRIPT_DIR}/assert_h_matrix.py" --db "${DB}"
echo "[smoke] All assertions passed."
