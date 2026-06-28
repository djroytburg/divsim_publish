# divsim: Reproducibility Bundle

**Paper:** *Disentangling Models from Personas in LLM Social Simulations*
Dani Roytburg & Daphne Ippolito, Carnegie Mellon University.

This bundle reproduces the experiments in the paper: the 5-model dyadic and triadic
sweeps, the four-way (tetradic) sweep, and the lexical/content analyses. Engagement is
analyzed as **comments per post** (CPP = in-degree / posts), which removes the
posting-volume confound. The simulations run on top of OASIS (Open Agent Social
Interaction Simulations), adapted with heterogeneous model support.

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

### 5b. Run the four-way (tetradic) sweep

Serves all four of {gpt-oss, Qwen, Magistral, GLM-4} concurrently and runs the
2-seed x 12-rotation four-way sweep (`run_v9 --mode hetero4`, 40 agents):

```bash
SAFEMASS_VENV=... OASIS_VENV=... HF_HUB_CACHE=... RUNS_DIR=... bash launch/run_tetradic.sh
```

Resume-safe: completed DBs are skipped, partial ones resume. Outputs land in `$RUNS_DIR/tetradic/`.

### 6. Analyses and figures

Scripts read run DBs from `./runs` (symlink or point at `$RUNS_DIR`), the text corpus
from `./data/texts.jsonl`, and write outputs to `./out/`. Build the corpus once:

```bash
python analyses/extract_texts.py                 # -> data/texts.jsonl (posts+comments by model,persona)
```

Engagement / network structure (Findings 1-2):
```bash
python analyses/plot_fig_variance_overview.py    # Fig: CPP variance (model vs persona) + per-model H by mixture size
python analyses/variance_cpp.py                  # CPP variance decomposition + leave-one-model-out
python analyses/swap_cpp.py                       # gpt-oss->Magistral matched-swap CPP effect
python analyses/engagement_ceiling.py            # how well text predicts per-post engagement (ceiling)
python analyses/cv_and_length.py                 # held-out CV R^2 + length-controlled attractor (Appendix D)
python analyses/absolute_variance.py             # variance components, all regimes
python analyses/temporal_dynamics.py             # temporal H decay by quartile
python analyses/tetradic_recompute_H.py          # four-way per-model H over the 24 runs
python analyses/tetradic_results.py              # four-way H matrix + heatmap figure
python analyses/plot_matrices.py                 # H-matrix heatmaps (dyadic/triadic)
```

Content (Finding 3):
```bash
python analyses/lexical_table.py                 # model/persona classifier from TF-IDF (Table 1)
python analyses/persona_faithfulness.py          # persona recoverability from text (faithfulness)
python analyses/distinctive_words.py             # characteristic phrases per model (log-odds)
python analyses/svd_axes.py                      # SVD style axes + top terms
python analyses/content_engagement.py            # content -> CPP decomposition (length/style/model/persona)
python analyses/style_geometry.py                # model style directions, stability, engagement-direction cosines
python analyses/cluster_model_persona.py         # does style cluster by model vs persona (permutation-adjusted)
python analyses/length_regimes.py                # length/volume confound checks by regime
```

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
| Single four-way run | 100 | 40 | ~60–90 min |
| Full four-way sweep (2 seeds × 12 rotations) | 100 | 40 | ~50–70 GPU-hours |

---

## Data

Reference data (SQLite databases, one per run) are listed in `data/MANIFEST.md`
with SHA-256 checksums and sizes. The canonical data lives at:

```
/data/user_data/droytbur/oasis/runs/sweep_*.db
```

The manifest records all valid runs used in the paper (dyadic, triadic, and four-way),
including the four-way DBs under `$RUNS_DIR/tetradic/`. The content analyses additionally
use a text corpus (`data/texts.jsonl`) regenerated by `analyses/extract_texts.py`.

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
│   ├── vllm_qwen.sh / vllm_gpt_oss.sh / vllm_magistral.sh / vllm_glm4.sh / vllm_gemma.sh
│   └── run_tetradic.sh          # four-way sweep driver (serves 4 models, 2 seeds x 12 rotations)
├── sweep/
│   ├── sweep_dyadic.py          # combinatorial 2-model sweep (5-model subset)
│   └── sweep_triadic.py         # preemption-safe 3-model sweep
├── analyses/
│   ├── variance_cpp.py              # CPP variance decomposition + leave-one-model-out
│   ├── absolute_variance.py         # variance components across regimes
│   ├── plot_fig_variance_overview.py# Fig: CPP variance + per-model H by mixture size
│   ├── temporal_dynamics.py         # temporal H decay analysis
│   ├── tetradic_recompute_H.py / tetradic_results.py  # four-way H + heatmap
│   ├── plot_matrices.py / plot_fig1_network.py / housestyle.py  # figures
│   ├── extract_texts.py             # build data/texts.jsonl corpus
│   ├── lexical_table.py             # model/persona classifier (Table 1)
│   ├── svd_axes.py                  # SVD style axes + top terms
│   ├── content_engagement.py        # content -> CPP decomposition
│   ├── style_geometry.py            # style directions + engagement-direction cosines
│   └── cluster_model_persona.py / length_regimes.py  # supporting checks
├── data/
│   └── MANIFEST.md              # DB paths, SHA-256, size, (pair/triplet, seed) labels
├── smoke_test/
│   ├── run_smoke.sh             # short dyad smoke run (20 steps, 2 agents/family)
│   └── assert_h_matrix.py       # assert H matrix computable + basic sanity checks
└── docs/
    └── EXPERIMENT_PROTOCOL.md   # canonical reproducibility narrative
```
