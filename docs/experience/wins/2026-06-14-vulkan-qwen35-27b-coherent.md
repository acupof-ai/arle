# ARLE Vulkan 27B forward is coherent on the AMD Radeon 8060S

## Context

Bringing up `infer-vulkan` on the Strix Halo 8060S (gfx1151). The 27B
(`Qwen3.6-27B-Q8_0`, arch `qwen35`, dense) forward ran end-to-end with finite
logits but produced **garbage → then degenerate single-token repeats** ("France"
repeated for "The capital of France is"). Tokenizer was verified byte-identical
to llama.cpp, so the bug was purely in the host-f32 forward math.

## What Worked

**The decisive method:** a numpy reference (`tools/qwen35_ref.py`) was found to
predict the *same* wrong token as ARLE — proving the bug was a **shared
misunderstanding of the GGUF math, not a transcription error**. So we stopped
diffing numpy-vs-ARLE and instead diffed against **llama.cpp's actual GGUF model
code** (`src/models/qwen35.cpp`, `delta-net-base.cpp`, `ggml-cpu/ops.cpp`), which
decodes this exact file correctly. That immediately exposed three bugs, all the
**same class** — HF→GGUF conventions where the GGUF converter pre-applies a
transform and our forward wrongly re-applied it:

1. **RMSNorm `(1+w)` offset.** HF Qwen3.5 stores the norm scale zero-centered and
   applies `x·inv_rms·(1+w)`; the **GGUF converter folds the `+1` into the stored
   weight** (verified: `attn_norm`≈0.98, `output_norm`≈1.96). Fix: apply **plain**
   `x·inv_rms·w` (`rms_norm_weight`), matching llama.cpp `LLM_NORM_RMS`.
2. **Gated-delta decay.** GGUF `ssm_a` already stores `A = −exp(A_log)` (confirmed
   negative, −0.34..−0.004). The log-decay is `ssm_a·softplus(α+dt)`, so the gate
   is `exp(ssm_a·softplus)` — **not** `exp(−exp(ssm_a)·softplus)`. (llama.cpp
   `qwen35.cpp:232` + `delta-net-base.cpp:341`.)
3. **Partial rotary — the dominant one.** GGUF `rope.dimension_count = 64`: only
   the first **64 of 256** head dims are rotated. The config mapper hard-coded
   `rotary_dim = head_dim = 256`, so q/k were **over-rotated with wrong
   frequencies** → attention couldn't encode position → the model couldn't
   retrieve from context → it copied a recent prompt token. Fix:
   `rotary_dim = rope.dimension_count` (fall back to `head_dim` if absent).

**Ruled out from source (not a bug):** M-RoPE. `ggml_mrope_cache_init` is called
with `indep_sects = is_vision = false` for text, and `llama-batch.cpp:711-714`
broadcasts the same position across all RoPE sections for text → M-RoPE ≡ plain
NeoX RoPE for a text model.

**Result (on the 8060S):** numpy oracle now tops `" Paris"` (+15.68), `" London"`
(+12.78) for "The capital of France is"; the on-device test
`qwen35_27b_generates_coherent_text` **passes** with continuation **" Paris."**
(`GEN IDS` start `[11751=" Paris", 13="."]`). Decode is currently ~5 s/token
(correctness-first: heavy matmuls on-GPU via the proven Q8_0 GEMV, elementwise on
host f32). Perf parity (llama.cpp 7.2 tok/s) is the next phase, plan in
`docs/plans/amd-vulkan-perf-parity.md`.

## Rule

When a from-GGUF forward is numerically wrong but the tokenizer is correct,
**diff against the engine that loads the same GGUF correctly (llama.cpp's
`src/models/<arch>.cpp`), not against the HF/safetensors reference** — the GGUF
converter pre-applies conventions (norm `+1` fold, `A=−exp(A_log)`, partial-rotary
`rope.dimension_count`) that the HF math does not. A CPU reference that merely
mirrors your own code is not an oracle; the independent correct implementation is.
