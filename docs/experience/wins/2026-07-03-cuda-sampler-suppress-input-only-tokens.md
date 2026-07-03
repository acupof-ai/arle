# CUDA sampler masks non-generatable input-only special tokens (DSv4 BOS/image spam)

> Status: **pending-remote** — H20 pod re-test queued: agentic tool-calling via
> the OpenAI `tools` param must complete with structured `tool_calls` and NO
> BOS/`<image>` spam. This stub becomes the full entry once the pod run lands.

## Context

DeepSeek-V4-Flash decode degenerated into repeated **input-only** special
tokens — BOS (id 0) under greedy, `<｜image｜>` (id 129279) at temp 0.6 —
whenever generation got hard/long (agentic or tool prompts). The OpenAI
tools-API re-test (2026-07-02) failed on exactly this: any request degenerated,
never emitting a valid tool call.

## Root cause (SOLID, isolated)

Not the DSML tool render (byte-verified against the official
`encoding_dsv4.py` template). The sampler (`sample_cuda_token` /
`sample_cuda_token_scratched`) applied **no logit mask for input-only special
tokens**. Discriminators: (1) the spam reproduced with NO tools array (agentic
system prompt alone → coherent 400 chars THEN BOS spam), so the tool block
isn't the trigger; (2) the spammed token varied with temperature (BOS greedy,
`<image>` at 0.6) — a distribution-level flatten, not a temp-invariant
structural corruption. Once the distribution flattens, an unmasked input-only
token (no learned continuation) wins argmax and the model spirals.

## What Worked

Model-parameterized suppressed-token mask in the shared CUDA sampler:
- `deepseek-spec` const `DSV4_SUPPRESSED_TOKEN_IDS = [0, 128803, 128804,
  129279]` (BOS / `<｜User｜>` / `<｜Assistant｜>` / `<｜image｜>`) — all
  input-only; EOS/`<think>`/`</think>`/`｜DSML｜` deliberately kept generatable.
- `sample_cuda_token`/`_scratched` write `bf16::-inf` into those logit slots
  before argmax (greedy) and before `to_host` (non-greedy), so every path
  skips them. Empty set (Qwen, every non-DSv4 model) → zero device writes →
  **byte-identical**.

Cost: ≤4 single-element device writes per token (no new kernel). Verified via
the Mac CUDA typecheck lane + metal/cli lanes; clippy clean.

## Rule

A serve must mask input-only structural tokens (BOS / turn-delimiter / vision
placeholders) from the output logits — the model can put mass there under a
flattened distribution, and an input-only token has no learned continuation, so
sampling one starts an unrecoverable repeat-spiral. This is a decode-robustness
invariant, not a per-workload fix; the tool workload just exposed it.
