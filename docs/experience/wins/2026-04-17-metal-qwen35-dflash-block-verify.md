# Qwen3.5 DFlash block-verify: 13 → 47 tok/s (partial win — still underwater vs plain decode)

## Context

Starting state: Qwen3.5-4B-4bit DFlash on Metal ran single-session at
13.3 tok/s (Apple M4 Max, ~18% of the 75 tok/s Rust-path baseline).
Track-1 auto-downgrade kept 4×/8× concurrent at ~144 tok/s by disabling
DFlash whenever `open.len() >= 2`, but the single-session regression
was brutal and the prior error note
([2026-04-17-qwen35-dflash-acceptance-ceiling.md](../errors/2026-04-17-qwen35-dflash-acceptance-ceiling.md))
concluded — incorrectly — that custom Metal kernels were the only path
to parity.

## What worked

**Architectural fix**: the C++ verify loop ran 16 sequential
`cpp_model.step(S=1)` calls per block. The reference `dflash-mlx` runs
verify as one `S=16` forward. We already had all the primitives.

1. New FFI `qwen35_compiled_block_verify` (C++): runs the compiled
   model with `current_seq_len = block_size`,
   `current_last_logits_only = false`. Single forward, single
   `eval`, per-layer tape/KV update amortized.
2. New Rust wrapper `Qwen35StepDriver::step_block` threads the block
   through the existing flat-KV contract.
3. `qwen35_dflash_speculative_block` swapped its per-token loop for
   one `step_block` call, then reused existing tape/snapshot rollback
   on partial reject.
4. **kv_flat capacity fix** (`request_state.rs::decode_token`): the
   live-prefix import path downsizes `kv_capacity` to the replay
   driver's 256-entry allocation, and the DFlash path bypassed
   `run_step`'s `ensure_capacity`. Added a capacity-grow call before
   the speculative block to match the pattern used in prefill and
   single-token decode. Without this, the second repeat-prompt
   request died with `Shapes (1,4,16,256) and (1,4,N,256) cannot be
   broadcast` once `cache_len + block_size > 256`.

Files touched:
- `crates/mlx-sys/src/mlx_qwen35_model.cpp` — `qwen35_compiled_block_verify`
- `crates/mlx-sys/src/lib.rs` — FFI decl
- `infer/src/backend/metal/qwen35.rs` — `step_block` wrapper
- `infer/src/backend/metal/dflash.rs` — swap step loop for block_verify
- `infer/src/backend/metal/request_state.rs` — capacity growth on DFlash decode

## Bench (M4 Max, Metal, Qwen3.5-4B-4bit, 256 completion tokens)

**Against the broken DFlash baseline:**

| Workload        | Before | After | Δ       |
|-----------------|--------|-------|---------|
| single (repeat) | 13.3   | 46.8  | +3.5×   |
| single (repeat) | 13.3   | 46.9  | +3.5×   |
| single (repeat) | 13.3   | 47.2  | +3.5×   |
| 4× concurrent   | 148.1  | 153.1 | +3%     |
| 8× concurrent   | 143.9  | 154.7 | +8%     |

Same-prompt repeat runs now all complete without error — the live
prefix cache import path is stable.

**Against plain decode (same server binary, `--dflash-draft-model`
flag omitted):**

| Workload        | Plain decode | DFlash ON | DFlash delta |
|-----------------|--------------|-----------|--------------|
| single (run 1)  | 72.5         | 47.3      | **−35%**     |
| single (run 2)  | 73.0         | 47.0      | **−36%**     |
| single (run 3)  | 73.0         | 47.7      | **−35%**     |
| 4× concurrent   | 159.0        | 154.5     | −3%          |
| 8× concurrent   | 156.1        | 150.1     | −4%          |

Concurrent parity is real (Track-1 auto-downgrade disables DFlash at
`open.len() >= 2`, so those numbers are effectively the same
packed-decode path). **Single-session is still a regression.** The
block_verify fix moved the gap from −82% to −35%, but plain decode
is the faster single-session path on 4-bit Qwen3.5-4B today.

## Why single-session is still underwater

At ~28% acceptance, a 16-token block nets ~4.5 tokens. For block_verify
to beat plain decode we need `T_S16 < 4.5 × T_S1`. Measured ratio is
closer to `T_S16 ≈ 7 × T_S1` — the S=16 forward is only ~2.3× faster
per-token than 16 S=1 forwards, not the 4.5× the speculative math
requires. The remaining overhead is probably GDR layers still doing
per-step work inside the S=16 path (the linear-attention recurrence is
inherently sequential over the time axis, so the `S=16` batching
savings come entirely from the 8 full-attention layers).

## Rule

**When per-step overhead dominates, amortize the existing kernels
before writing new ones.** Custom Metal kernels were scoped at ~200 LOC
of C++; the actual fix was one new FFI plus a Rust wrapper using
primitives already in the compiled model. Measure per-step work
against equivalent batched work first; a "ceiling" that collapses
under a single change was never a ceiling.

**Corollary for live prefix cache interactions**: any decode path that
bypasses `run_step` must re-check `kv_capacity` against
`cache_len + expected_step_tokens`. Snapshot imports can legitimately
shrink capacity (smaller replay driver), and downstream paths need to
grow it back before stepping.

**Operational note:** Do not enable `--dflash-draft-model` for
single-session Qwen3.5-4B-4bit until either (a) the S=16 forward
drops below ~4.5× S=1 per-token (likely requires attention to the
GDR linear-attention time-axis batching inside the compiled model),
or (b) acceptance climbs above ~50%. Track-1 auto-downgrade already
protects concurrent traffic; the remaining question is whether to
also gate DFlash off for `open.len() == 1` on this quant.

## Follow-ups

- Profile `qwen35_compiled_block_verify` with MLX instruments to
  locate the GDR sequential bottleneck inside S=16. If every GDR
  layer still does 16 `gated_delta_step` calls, the S=16 savings
  come only from the 8 full-attn layers (24 GDR layers unchanged).
- Consider a `--dflash-single-session-threshold` knob that disables
  DFlash when `open.len() < N` for targets where speculative math
  doesn't pencil out.
- Revisit if/when acceptance rate improves (better draft model on
  4-bit target distribution, or bf16 target path).
