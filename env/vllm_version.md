# vLLM Version and Per-Model Launch Notes

**Version used in paper:** vLLM 0.19.0

All five models are served via `vllm.entrypoints.openai.api_server`. Common
defaults across all models:

| Parameter | Value |
|-----------|-------|
| `--max-model-len` | 16384 |
| `--gpu-memory-utilization` | 0.90 |
| `--enable-auto-tool-choice` | (flag) |
| `--no-scheduler-reserve-full-isl` | (flag, required — avoids a vLLM 0.19.0 scheduler deadlock under sustained multi-agent load) |
| Input context cap | 12,000 tokens (set in runner via `_input_token_limit`) |
| Max output tokens | 1536 (512 for Gemma) |
| Temperature | 0.85 |
| BF16 weights | default |

---

## Per-model deviations

### Qwen/Qwen3-32B
- TP=2 (GPUs 0-1)
- `--tool-call-parser hermes`
- `tool_choice=required`
- Thinking suppressed via `chat_template_kwargs = {"enable_thinking": False}`

### openai/gpt-oss-20b
- TP=1 (GPU 2); mxfp4 quantized weights fit in ~40 GB
- `--tool-call-parser openai`
- `--reasoning-parser openai_gptoss`
- `tool_choice=required`

### mistralai/Magistral-Small-2509
- TP=2 (GPUs 3-4)
- `--tokenizer-mode mistral --config-format mistral --load-format mistral`
  (Mistral models require their own loader; standard HuggingFace format fails)
- `--tool-call-parser mistral`
- `--reasoning-parser mistral`
- `tool_choice=auto` — **Magistral returns empty `tool_calls` with `required`**;
  `auto` is the correct setting

### zai-org/GLM-4-32B-0414
- TP=2 (GPUs 5-6)
- `--tool-call-parser hermes`
- `tool_choice=required` — **GLM-4 silently returns plain text with `auto`**;
  this was discovered when warm-up showed only 1/10 agents posting per run

### google/gemma-4-31B-it
- TP=2 (GPUs 0-1 or 3-4 depending on node configuration)
- `--tool-call-parser gemma4`
- `--chat-template oasis/tool_chat_template_gemma4.jinja` (custom template, required)
- `tool_choice=auto` — same symptom as Magistral with `required`
- `--max-num-seqs 64` — tuned to avoid KV-cache saturation under concurrent load
- `max_tokens=512` (runner-side cap, lower than other models)
- **Note:** There is a discrepancy between the paper appendix (max_num_seqs=64)
  and an earlier production launch script (max_num_seqs=8). The appendix value
  (64) is the final setting used in paper runs.

---

## Known issues in vLLM 0.19.0

- **Scheduler deadlock**: Under sustained multi-agent load, the scheduler can
  deadlock. Mitigated by `--no-scheduler-reserve-full-isl` on all models.
- **Gemma KV-cache saturation**: Default scheduling settings cause OOM-like
  behavior under concurrent requests. `--max-num-seqs 64` and ensuring
  `--gpu-memory-utilization 0.90` resolves this.
- **Magistral/Gemma tool_choice=required**: Both models return empty
  `tool_calls` lists when `tool_choice=required`; the runner uses `auto` and
  handles non-tool responses with prose-to-action scaffolding.
