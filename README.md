# divsim: Reproducibility Bundle

**Paper:** *Disentangling Models from Personas in LLM Social Simulations*
Dani Roytburg & Daphne Ippolito, Carnegie Mellon University.

This bundle contains everything needed to reproduce the 5-model dyadic and triadic
sweep experiments reported in the paper. The simulations run on top of OASIS
(Open Agent Social Interaction Simulations), adapted with heterogeneous model support.

---

## Hardware assumptions

| Component | Spec |
|-----------|------|
| GPUs | 8 × NVIDIA L40S 48 GB |
| System RAM | ≥ 128 GB |
| CPU cores | ≥ 32 |
| Storage | ≥ 200 GB free (model weights + DBs) |

All five models run concurrently on a single 8-GPU node. Dyadic runs need any
two models active; triadic runs need any three. The sbatch templates in
`launch/` target this configuration.

---

## Models

| Short name | HuggingFace ID | GPU assignment | Notes |
|-----------|----------------|----------------|-------|
| Qwen | `Qwen/Qwen3-32B` | GPUs 0-1 (TP=2) | thinking disabled via chat-template |
| gpt-oss | `openai/gpt-oss-20b` | GPU 2 (TP=1, mxfp4) | reasoning parser enabled |
| Mag | `mistralai/Magistral-Small-2509` | GPUs 3-4 (TP=2) | mistral tokenizer/loader mode |
| GLM-4 | `zai-org/GLM-4-32B-0414` | GPUs 5-6 (TP=2) | hermes tool parser |
| Gemma | `google/gemma-4-31B-it` | GPUs 3-4 (TP=2) | custom jinja template required |

Note: Gemma shares GPUs 3-4 with Magistral. They cannot run simultaneously on
a single 8-GPU node. For triadic runs involving both, use two separate nodes or
run the Gemma-containing triplets in a separate job.

---

## Quick start

### 1. Environment setup

```bash
git clone https://github.com/camel-ai/oasis.git  # or use bundled oasis/
cd divsim_publish
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
```

Set these environment variables before running:

```bash
export HF_HUB_CACHE=/path/to/hf_cache/hub          # HuggingFace model cache
export SAFEMASS_VENV=$(pwd)/.venv                   # or path to your venv
export RUNS_DIR=/path/to/output/runs                # where DBs will be written
export SEED_DB=$(pwd)/oasis/seed_db/hf-zero-50.sqlite
export LOGS_DIR=/path/to/logs
```

### 2. Download models

```bash
python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_HUB_CACHE'] = os.environ['HF_HUB_CACHE']
for m in [
    'Qwen/Qwen3-32B',
    'openai/gpt-oss-20b',
    'mistralai/Magistral-Small-2509',
    'zai-org/GLM-4-32B-0414',
    'google/gemma-4-31B-it',
]:
    print(f'Downloading {m}...')
    snapshot_download(m)
    print(f'  done.')
"
```

### 3. Start vLLM servers

Launch each model server in a separate terminal (or as background processes):

```bash
# For a 4-model (non-Gemma) run:
bash launch/vllm_qwen.sh &
bash launch/vllm_gpt_oss.sh &
bash launch/vllm_magistral.sh &
bash launch/vllm_glm4.sh &

# For Gemma runs (requires GPUs 3-4 to be free — stop Magistral first):
bash launch/vllm_gemma.sh &
```

Wait for all target servers to respond at their health endpoints before
proceeding. The sweep scripts perform a pre-flight health check.

### 4. Run the dyadic sweep

```bash
source .venv/bin/activate
python sweep/sweep_dyadic.py --mode 2model --max-parallel 2
```

Optional: `--dry-run` to print all configs without launching.

### 5. Run the triadic sweep

```bash
python sweep/sweep_triadic.py --max-parallel 2
```

### 6. Generate figures

```bash
python analyses/plot_matrices.py      # H-matrix, swap stability, self-pref, Elo
python analyses/temporal_dynamics.py  # temporal decay quartile analysis
python analyses/persona_variance.py   # variance decomposition (OLS + permtest + GBT)
```

All figures are written to the same directory as the script by default.
Override with `--outdir <path>`.

### 7. Smoke test (no GPUs required for the assertion; short GPU run for the full test)

```bash
# Assert-only (uses an existing DB):
python smoke_test/assert_h_matrix.py --db <path/to/sweep_*.db>

# Full end-to-end smoke (requires 2 live vLLM servers, ~5 min on 2 GPUs):
bash smoke_test/run_smoke.sh
```

---

## Expected runtime

| Experiment | Steps | Agents | Time estimate (8× L40S) |
|-----------|-------|--------|------------------------|
| Single dyadic run | 100 | 20 | ~25–45 min |
| Full dyadic sweep (10 pairs × 2 orderings × 5 seeds) | 100 | 20 | ~50–150 GPU-hours |
| Single triadic run | 100 | 30 | ~35–60 min |
| Full triadic sweep (10 triplets × 3 orderings × 3 seeds) | 100 | 30 | ~35–90 GPU-hours |

---

## Data

Reference data (SQLite databases, one per run) are listed in `data/MANIFEST.md`
with SHA-256 checksums and sizes. The canonical data lives at:

```
/data/user_data/droytbur/oasis/runs/sweep_*.db
```

The manifest records all valid runs used in the paper (dyadic: nu ≥ 18, nc ≥ 200;
triadic: nu ≥ 26).

---

## Repository layout

```
divsim_publish/
├── README.md                    # this file
├── LICENSE                      # MIT
├── env/
│   ├── requirements.txt         # pinned Python dependencies
│   └── vllm_version.md          # vLLM 0.19.0 per-model launch notes
├── oasis/
│   ├── run_v9.py                # simulation runner
│   ├── seed_db/hf-zero-50.sqlite
│   └── tool_chat_template_gemma4.jinja
├── launch/
│   ├── vllm_qwen.sh
│   ├── vllm_gpt_oss.sh
│   ├── vllm_magistral.sh
│   ├── vllm_glm4.sh
│   └── vllm_gemma.sh
├── sweep/
│   ├── sweep_dyadic.py          # combinatorial 2-model sweep (5-model subset)
│   └── sweep_triadic.py         # preemption-safe 3-model sweep
├── analyses/
│   ├── plot_matrices.py         # H-matrix, swap stability, self-pref, Elo figures
│   ├── persona_variance.py      # OLS + permutation + GBT variance decomposition
│   └── temporal_dynamics.py     # temporal H decay analysis
├── data/
│   └── MANIFEST.md              # DB paths, SHA-256, size, (pair/triplet, seed) labels
├── smoke_test/
│   ├── run_smoke.sh             # short dyad smoke run (20 steps, 2 agents/family)
│   └── assert_h_matrix.py       # assert H matrix computable + basic sanity checks
└── docs/
    └── EXPERIMENT_PROTOCOL.md   # canonical reproducibility narrative
```
