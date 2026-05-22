# ARLE Serve Long-Prompt Corruption Fix

## Goal

Fix the ARLE CUDA serve path that corrupted Qwen3.5 completions once prompts crossed the ~33-token prefill threshold, blocking capability eval and API-teacher OPD.

## Hypothesis

The direct `forward_token_logits` path was healthy because it advances one token at a time. The HTTP serve path failed because batched prefill took a different Qwen3.5 linear-attention path.

## Commands

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo test -p infer --lib --release --features cuda \
  test_gdr_prefill_matches_repeated_decode_at_long_prompt_threshold -- --nocapture
```

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/infer \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --port 8123 --num-slots 1 --max-seq-len 4096 \
  --chunked-prefill-size 4096 --max-num-batched-tokens 4096
```

```bash
.venv/bin/python scripts/arle_capability_eval.py \
  --base-url http://localhost:8123 \
  --model-id Qwen3___5-0___8B-Base \
  --tasks mmlu,gsm8k --n-samples 200 \
  --output bench-output/2026-05-22-capability-baseline-08b-retry-after-longprompt-fix
```

```bash
.venv/bin/python scripts/arle_capability_eval.py \
  --base-url http://localhost:8123 \
  --model-id Qwen3___5-4B \
  --tasks mmlu,gsm8k --n-samples 200 \
  --output bench-output/2026-05-22-capability-baseline-4b-retry-after-longprompt-fix
```

## Environment

- Backend: CUDA
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB
- Models: `Qwen3___5-0___8B-Base`, `Qwen3___5-4B`
- Feature set: `--release --features cuda`
- Key env: `NVCC_CCBIN=/usr/bin/g++-14`, `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`, `CUDARC_CUDA_VERSION=13010`, `TORCH_CUDA_ARCH_LIST=8.9`

## Results

### Root Cause

The first corrupt stage was Qwen3.5 gated-delta-rule batched prefill, not the sampler, tokenizer, HTTP extraction, chunking, or CUDA Graph decode.

Evidence:

- `forward_token_logits`/single-token decode stayed correct.
- Contiguous and paged prefill both diverged at the 33-token boundary before the fix.
- The targeted GDR test reproduced the exact boundary: `seq_len=32` matched repeated decode within BF16 tolerance, while `seq_len=33+` diverged before the fix.
- The bad path was the chunkwise GDR prefill path used by Qwen3.5 hybrid linear attention.

Fix:

- Added a native CUDA recurrent prefill kernel that replays the decode-equivalent recurrence over the prefill sequence.
- Routed Qwen3.5 GDR prefill `seq_len > 32` through that recurrent fallback for both single-request and packed/paged batch prefill.
- Fixed a second serve-only OOM: when the scheduler warmup allocated a 4096-token paged-prefill buffer, the next smaller request tried to allocate the new buffer before freeing the old one. Shape changes now drop and sync the old prefill buffer before allocation.

### Targeted Regression Test

| seq_len | max output diff vs repeated decode | max state diff vs repeated decode |
|---:|---:|---:|
| 32 | 0.000122 | 0.000244 |
| 33 | 0.000000 | 0.000000 |
| 34 | 0.000000 | 0.000000 |
| 35 | 0.000000 | 0.000000 |

### HTTP Serve Probe

`Qwen3___5-0___8B-Base`, `max_seq_len=4096`, greedy completion for:

`"Hello world. " * 10 + "The capital of France is"`

Before: 35-token prompt collapsed into garbage/Unicode.
After: ` Paris. The capital of France is Paris. The capital of France is Paris.`

First-token probe:

| prompt reps | prompt tokens | first generated token |
|---:|---:|---|
| 5 | 20 | ` Paris` |
| 9 | 32 | ` Paris` |
| 10 | 35 | ` Paris` |

### Capability Eval Retry

| model | task | scored | invalid | accuracy |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B-Base | MMLU | 142/171 | 29 | 0.514 |
| Qwen3.5-0.8B-Base | GSM8K | 194/200 | 6 | 0.015 |
| Qwen3.5-4B | MMLU | 150/171 | 21 | 0.773 |
| Qwen3.5-4B | GSM8K | 198/200 | 2 | 0.025 |

This passes the P0 retry gate: all invalid rates are below 30%, and 4B beats 0.8B on both tasks.

## Problems

- The recurrent fallback is correctness-first. It is not the final optimized GDR prefill path.
- GSM8K accuracy remains low for both Base models. The debug output is coherent but weak at arithmetic, so this is a model/prompt/eval capability floor, not the prior invalid-output collapse.
- `cargo clippy --workspace --all-targets -- -D warnings` still hits pre-existing DeepSeek unused warnings outside this fix; those were not touched.

## Learnings

- For Qwen3.5 hybrid models, direct single-token logits is not enough to validate serve correctness. Batched prefill must be compared against repeated decode at the exact threshold where the scheduler switches from short to long prompt behavior.
- Shape-reallocation code must drop the old large CUDA buffers before allocating smaller replacements; otherwise warmup can leave enough resident state to make ordinary requests OOM.

## Artefacts

- Probe output: `bench-output/2026-05-22-serve-long-prompt-probes/prefill-path-reps10-fp8-paged.txt`
- HTTP first-token probe: `bench-output/2026-05-22-serve-long-prompt-probes/serve-08b-after-sync-drop-maxseq4096-first-token.jsonl`
- HTTP multi-token probe: `bench-output/2026-05-22-serve-long-prompt-probes/serve-08b-after-sync-drop-maxseq4096-reps10.json`
- Targeted test: `bench-output/2026-05-22-serve-long-prompt-probes/test-gdr-prefill-long-threshold.txt`
- 0.8B eval: `bench-output/2026-05-22-capability-baseline-08b-retry-after-longprompt-fix/summary.json`
- 4B eval: `bench-output/2026-05-22-capability-baseline-4b-retry-after-longprompt-fix/summary.json`
