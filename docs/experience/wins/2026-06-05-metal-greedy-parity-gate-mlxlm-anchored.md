# Metal greedy-parity gate — token-exact vs upstream mlx_lm (dense + MoE)

**Date:** 2026-06-05. **Backend:** Metal (MLX), Apple Silicon (M-series).
**Status:** landed, both gates green. Closes the "Metal has no numeric oracle"
gap left when `infer/src/backend/metal` was deleted in the rewrite.

## Context

CUDA has `cuda_qwen3_greedy_parity` / `cuda_qwen35_greedy_parity`; **Metal had
none** — every `infer-metal` / `mlx-sys` numeric change shipped with no
regression gate. There is no live legacy Metal path to diff against, so the
oracle must be a **pinned snapshot**: a committed greedy continuation +
FNV-1a fingerprint that the deterministic MLX forward must reproduce bit-for-bit.

## What Worked

`crates/agent-bench/src/lib.rs` — `run_metal_greedy_parity(model)` drives one
raw-id greedy request through the real Metal engine (`drive_concurrent`, FNV
fingerprint) and asserts token-exact + fingerprint match vs
`test_data/metal_greedy_parity_gold.json`. Two `#[ignore]` (opt-in) gates:
`metal_qwen35_greedy_parity` (dense 0.8B, cheap CI) and
`metal_qwen36_greedy_parity` (canonical MoE 35B-A3B). `METAL_PARITY_BLESS=1`
prints fresh gold + skips the assertion.

**Anchored to ground truth, not self-reference.** The pinned gold was validated
against **upstream `mlx_lm` 0.31.2** greedy (temp=0) on the *identical* prompt
ids — an independent MLX forward implementation, same model + quant:

| model | ARLE Metal fingerprint | vs mlx_lm oracle | decoded |
|-------|------------------------|------------------|---------|
| Qwen3.5-0.8B-MLX-4bit | `0xf005cfaa7dc1793e` | **token-exact (32/32)** | " Paris, and the capital of the United States is Washington, D.C. ..." |
| Qwen3.6-35B-A3B-4bit | `0xb692668fc8e69d02` | **token-exact (32/32)** | " Paris, a city renowned for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. ..." |

Both gates pass non-bless. ARLE's mlx-sys bridge reproduces the reference MLX
forward exactly for both the dense and the MoE (router/expert/gated-delta) paths.

## The catch that mattered

The draft gold's `prompt_ids` `[9707, 11, 1879, 358, 1079, 264, 6722, 13]` were
**gibberish** in the actual tokenizer — they decode to `'.Q, inputow << a £.'`
in *both* the Qwen3.5 and Qwen3.6 tokenizers (the author hand-wrote ids assuming
"Hello, world I am a robot."). The first bless produced a degenerate
`"\n10000000..."` continuation. Pinning that fingerprint would have anchored the
gate to a meaningless prompt + a possible-bug-shaped output. Caught it by
decoding the actual ids and the continuation, replaced the prompt with
`[760, 6511, 314, 9338, 369]` = `"The capital of France is"` (verified identical
text in both tokenizers — shared Qwen vocab), and re-blessed against the mlx_lm
oracle.

## Rule

A pinned-snapshot regression gate must be **anchored to an independent upstream
oracle at bless time** (here: `mlx_lm` greedy on identical ids), never pinned to
its own first output — otherwise it can lock a latent forward bug in as "gold."
And **decode the prompt ids + the continuation before pinning**: a degenerate
continuation (`"\n10000..."`) was the tell that the hand-written prompt ids were
gibberish, not real text. Cross-tokenizer-check shared prompts when one gold
serves multiple models.
