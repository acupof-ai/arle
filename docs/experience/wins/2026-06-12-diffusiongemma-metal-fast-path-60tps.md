# DiffusionGemma Metal Fast Path 60 TPS Gate

## Goal

Move `mlx-community/diffusiongemma-26B-A4B-it-4bit` on Metal toward the user
gate: one 64-token chat completion should reach 60 generated tokens/s before
spending time on other output sizes.

## Hypothesis

The first bridge was slow because the denoise loop crossed Rust/C++ and forced
host-visible sampled/argmax/entropy arrays every step. Keeping the canvas,
entropy-bound mask, renoise, and self-conditioning inside the C++/MLX bridge
should make step count the main speed knob. A lower `max_denoising_steps` can
hit the 60 tokens/s speed profile, with an explicit quality tradeoff.

## Params

- Backend: Metal
- Model: `mlx-community/diffusiongemma-26B-A4B-it-4bit`
- Prompt: `Write a short paragraph about ARLE runtime.`
- Request: `/v1/chat/completions`, `max_tokens=64`, `temperature=0`
- Serve flags:
  `--num-slots 1 --total-pages 16 --page-size 256 --max-prompt-tokens 512 --max-total-tokens 1024 --chunked-prefill-size 512`
- Speed-profile env:
  `ARLE_DIFFUSION_MAX_DENOISING_STEPS=4`
- Trace env:
  `ARLE_DIFFUSION_TRACE=1 ARLE_DIFFUSION_CPP_PROFILE=1`

## Env

- Host: local Apple Silicon Metal path
- Date: 2026-06-12
- Target binary: `target/release/arle`
- Reference runner: `/tmp/arle-mlx-vlm-bench/bin/python -m mlx_vlm.generate`

## Results

| Runner | Steps | Warm state | Wall time | Notes |
| --- | ---: | --- | ---: | --- |
| ARLE before per-step sampling fix | 48 max, adaptive 16 | warm | 8.87 s | Prompt KV cache + sorted MoE, still per-step host loop. |
| ARLE C++ fast path | 8 | warm | 2.03 s | `prefill_ms=109.4`, `denoise_ms=1673.5`, final canvas eval/copy `246.9`. |
| ARLE C++ fast path | 4 | warm | 1.026 s | Hits speed gate; output quality visibly degrades with repetitions. |
| `mlx-vlm` reference | 8 | generation timing | n/a | Same prompt and model; reported `Prompt: 163.0 tok/s`, `Generation: 43.3 tok/s`. |

## What Worked

- Added an optional `DiffusionBlockModel::generate` fast path so Metal
  DiffusionGemma can own the whole canvas denoise loop in C++/MLX.
- Moved entropy-bound acceptance, renoise, self-conditioning, and final commit
  handling into `diffusion_gemma_generate`.
- Added `ARLE_DIFFUSION_MAX_DENOISING_STEPS` to run explicit speed profiles
  without changing the model default.
- Kept final block out of the causal KV commit path when no next block is
  needed, avoiding unnecessary end-of-request cache mutation.
- Aligned the MoE router with the reference path: top-k on logits, then
  softmax over only top-k experts.
- Kept dense/dequantized embedding for token lookup and soft embeddings, but
  used the tied quantized embedding as the lm-head projection.

## Problems

The 60 tokens/s gate is not a default-quality result. At 4 denoise steps, the
sample output repeated words and punctuation. The 8-step profile is much more
usable; the `mlx-vlm` reference reports 43.3 generation tokens/s.

## Learnings

For DiffusionGemma, reported tokens/s is step-budget-sensitive. A 60 tokens/s
Metal result must be labeled as a speed profile unless a separate quality gate
licenses the lower denoise budget.

## Rule

Do not compare DiffusionGemma speed claims without recording `max_denoising_steps`,
warm/cold state, and whether the number is full request wall-clock or generation
only.
