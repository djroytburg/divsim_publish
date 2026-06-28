#!/bin/bash
# Four-way (tetradic) rotation sweep: 2 seeds x 12 distinct model->block permutations.
# Pool {gpt-oss, Qwen, Magistral, GLM-4}, n=40 agents (10/model), 100 steps,
# post-share balancer on, activation prob 0.3, run_v9 --mode hetero4.
# Serves all four models concurrently on one node (7 GPUs), runs two rotations in parallel.
# Resume-safe: completed DBs are skipped, partial ones (with a *_checkpoint.json) resume.
#
# Required env:
#   SAFEMASS_VENV   venv with vLLM serving stack (CUDA torch, transformers)
#   OASIS_VENV      venv with the OASIS client deps (run_v9.py)
#   HF_HUB_CACHE    HuggingFace model cache
#   RUNS_DIR        output dir for run DBs
#   SEEDS           (optional) space-separated seeds; default "42 271"
set -uo pipefail
SEEDS="${SEEDS:-42 271}"
: "${SAFEMASS_VENV:?set SAFEMASS_VENV}"; : "${OASIS_VENV:?set OASIS_VENV}"
: "${HF_HUB_CACHE:?set HF_HUB_CACHE}"; export HF_HUB_CACHE
RUNS="${RUNS_DIR:?set RUNS_DIR}"
EXP="${RUNS}/tetradic"; LOGS="${EXP}/logs"; mkdir -p "$EXP" "$LOGS"
exec >> "${LOGS}/rotations.log" 2>&1
echo "==== tetradic rotations start $(date -Is)  seeds=[$SEEDS] x12 ===="

wait_health(){ for i in $(seq 1 360); do curl -fsS "http://127.0.0.1:$1/health">/dev/null 2>&1&&return 0; sleep 5; done; return 1; }

# --- serve the four models (see launch/vllm_*.sh for per-model flag rationale) ---
CUDA_VISIBLE_DEVICES=0 "${SAFEMASS_VENV}/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b --served-model-name openai/gpt-oss-20b --tensor-parallel-size 1 --max-model-len 16384 \
  --port 8003 --gpu-memory-utilization 0.90 --host 127.0.0.1 --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss >>"${LOGS}/vllm_oss.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1,2 "${SAFEMASS_VENV}/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --served-model-name Qwen/Qwen3-32B --tensor-parallel-size 2 --max-model-len 16384 \
  --port 8002 --gpu-memory-utilization 0.90 --host 127.0.0.1 --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser hermes >>"${LOGS}/vllm_qwen.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3,4 "${SAFEMASS_VENV}/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model mistralai/Magistral-Small-2509 --tensor-parallel-size 2 --max-model-len 16384 --port 8004 \
  --gpu-memory-utilization 0.90 --host 127.0.0.1 --trust-remote-code --tokenizer-mode mistral \
  --load-format mistral --config-format mistral --enable-auto-tool-choice --tool-call-parser mistral \
  --reasoning-parser mistral >>"${LOGS}/vllm_mag.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5,6 "${SAFEMASS_VENV}/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model zai-org/GLM-4-32B-0414 --tensor-parallel-size 2 --max-model-len 16384 --port 8007 \
  --gpu-memory-utilization 0.90 --host 127.0.0.1 --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser hermes >>"${LOGS}/vllm_glm.log" 2>&1 &
for p in 8003 8002 8004 8007; do wait_health $p||{ echo "FATAL vllm $p"; pkill -9 -f vllm.entrypoints; exit 1; }; done
echo "[fleet] up $(date -Is)"

PY="${OASIS_VENV}/bin/python3"; cd "$RUNS"
url(){ case $1 in O)echo http://localhost:8003/v1;; Q)echo http://localhost:8002/v1;; M)echo http://localhost:8004/v1;; G)echo http://localhost:8007/v1;; esac; }
mdl(){ case $1 in O)echo openai/gpt-oss-20b;; Q)echo Qwen/Qwen3-32B;; M)echo mistralai/Magistral-Small-2509;; G)echo zai-org/GLM-4-32B-0414;; esac; }

# 12 distinct permutations of [gpt-oss, Qwen, Magistral, GLM-4] across blocks A B C D.
PERMS=( "O Q M G" "Q M G O" "M G O Q" "G O Q M" \
        "O G M Q" "G M Q O" "M Q O G" "Q O G M" \
        "O Q G M" "Q O M G" "M G Q O" "G M O Q" )

run_one(){ # <db> <seed> <permstring>
  local db="$1" sd="$2"; read -r a b c d <<< "$3"
  local name=$(basename "$db" .db); local ckpt="${db%.db}_checkpoint.json"
  if [ -f "$db" ] && [ ! -f "$ckpt" ]; then echo "  == $name complete, skip"; return 0; fi
  local res=""; [ -f "$ckpt" ] && res="--resume"
  VLLM_URL_A="$(url $a)" MODEL_NAME_A="$(mdl $a)" VLLM_URL_B="$(url $b)" MODEL_NAME_B="$(mdl $b)" \
  VLLM_URL_C="$(url $c)" MODEL_NAME_C="$(mdl $c)" VLLM_URL_D="$(url $d)" MODEL_NAME_D="$(mdl $d)" POST_SHARE_TOL=0.05 \
  $PY run_v9.py --mode hetero4 --steps 100 --agents 40 --seed-posts 0 --activate-prob 0.3 \
     --rng-seed "$sd" --db "$db" $res >>"${LOGS}/${name}.log" 2>&1
  echo "  <<< $name rc=$? $(date -Is)"
}

for sd in $SEEDS; do
  echo "==== seed $sd $(date -Is) ===="
  i=0
  while [ $i -lt ${#PERMS[@]} ]; do
    run_one "$EXP/tetra_s${sd}_r$((i+1)).db" $sd "${PERMS[$i]}" & p1=$!
    p2=""; if [ $((i+1)) -lt ${#PERMS[@]} ]; then run_one "$EXP/tetra_s${sd}_r$((i+2)).db" $sd "${PERMS[$((i+1))]}" & p2=$!; fi
    wait $p1 ${p2:-}
    i=$((i+2))
  done
done
echo "==== tetradic rotations done $(date -Is) ===="
