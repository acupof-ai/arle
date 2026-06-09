# DSv4 256K+ host-bound was an admission spin — generic page pool capped at 128K

**Date:** 2026-06-09. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Commit:** (this, band-aid). **Scope:** `infer-api/loaded.rs`.

## Context — the "256K host-bound" mystery, solved

256K+ prefill pinned the engine thread at 100% CPU with **all 8 GPUs at 0%** for
minutes (never completing), while 64K/128K ran fine. Earlier probes established it
was a **pure userspace loop** (16/16 `/proc/syscall` = "running"), **pre-kernel-issue**
(GPU never lit, no forward logged), in the **scheduler** (not the model forward, not
`csa_select` @0.3%, not the compressor).

## Root cause — infinite admission spin (not O(N²) work)

End-to-end:
1. Request enters `waiting`. The engine runs `while !is_idle()` (`lib.rs:444`); each
   tick calls `admit_waiting`.
2. `admit_waiting` → `request_pages_needed_after_prefix` (`prefix.rs:25`):
   `pages_needed = (prompt_len + max_tokens) / page_size`. For 256K:
   `(200000+8)/16 = 12,501` pages.
3. The host `CudaKvPool` has `total_pages = 8192` — the `EngineLoadConfig` short-
   context default. **It is a DUMMY pool for DSv4** (the real KV is recurrent —
   SW ring + compressed — owned by the executor; comment at `loaded.rs:507`).
4. `12,501 > 8192` → `admit_waiting` rejects (`lib.rs:686-690`, `break`) → request
   stays in `waiting`.
5. `is_idle() = waiting.is_empty() && …` → **false** → the loop re-ticks → reject →
   **spin forever**: 100% CPU, no forward, GPU 0%.

**Threshold = `total_pages × page_size = 8192 × 16 = 131,072 = exactly 128K`.** That
is the measured 128K-completes / 256K-spins boundary, to the token.

This is the *same class* as the prefix-cache recurrent-KV bug: the generic scheduler's
**full-attention page model** vs DSv4's **recurrent KV**. The earlier fix
(`6e2e572e`) lifted `max_prompt_tokens` / `chunked_prefill_size` for DSv4 but **missed
`total_pages`** — so admission still rejected >128K.

## Fix (band-aid)

`loaded.rs`: for DSv4, size the dummy `CudaKvPool` to the model's max context —
`total_pages = (dsv4_max_seq_len() + 4096) / page_size`. `CudaKvPool::new` allocates
**no HBM** (just a `Vec<u32>` of page ids), so this is free.

**Verified:** 256K (204,338 tok) admits → forwards (GPU 100%) → completes in **55.4s**,
retrieves `738291` exact (was: infinite spin, never completed).

## Rule / follow-up

- A request stuck in `waiting` keeps `is_idle()` false → the engine **busy-spins**
  (100% CPU, GPU 0%). "Host-bound, GPU idle, pre-kernel-issue" on a long prompt is
  an **admission rejection spin** first — check `pages_needed` vs pool `total_pages`
  before profiling the forward.
- **Uniform rewrite — LANDED (follow-up commit).** The DSv4-only `if matches!` bump
  was replaced by `cuda_admission_total_pages(kind, config, page_size)`: one call
  site for every model, each backend declaring its KV token-capacity (full-attention
  / Qwen3.5-3.6-MoE keep the configured budget; DSv4 = `max_seq_len`, dummy pool).
  Behavior-identical to the band-aid for DSv4. Deeper still (not yet done): the
  engine should **reject a permanently-unfittable request** (`pages_needed > total
  capacity`) instead of spinning `while !is_idle()` — a model-agnostic anti-spin
  guard in `admit_waiting`, so any over-large prompt errors gracefully rather than
  hanging the engine.
