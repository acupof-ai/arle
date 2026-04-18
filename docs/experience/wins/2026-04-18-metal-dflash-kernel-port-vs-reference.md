# Metal DFlash kernel port — verified against bstnxbt/dflash-mlx reference

## Context

Speculative-decode "DFlash" on Apple Silicon. Reference implementation:
[`bstnxbt/dflash-mlx`](https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/kernels.py).
The Qwen3.5 hybrid model (8 full-attn + 24 GDR linear-attn layers) needs
all three reference kernels to make the verify-block path tractable:

| Reference (`dflash_mlx/kernels.py`) | Our port |
|---|---|
| `_make_gated_delta_kernel_with_tape` (line 12) | `gated_delta_tape_kernel()` — `crates/mlx-sys/src/mlx_qwen35_model.cpp:165` |
| `tape_replay_kernel` | `tape_replay_kernel()` — `crates/mlx-sys/src/mlx_bridge.cpp:97` |
| `batched_sdpa_2pass_exact` | `batched_sdpa_2pass_partials_kernel + _reduce_kernel` — `crates/mlx-sys/src/mlx_bridge.cpp:215, 321` |

All three are wired into the Qwen3.5 DFlash path:
`qwen35_dflash_speculative_block` (`infer/src/backend/metal/dflash.rs:1655`)
→ `qwen35_set_tape_mode(true)` → C++ `gdr_step` records tapes →
`drain_current_qwen35_gdr_tapes` → `mlx_tape_replay` per GDR layer
on partial accept.

## What Worked

The mechanical port is **functionally complete**. Concurrency-fallback
and block-verify wins ([2026-04-17-metal-qwen35-dflash-block-verify.md],
[2026-04-17-metal-qwen35-dflash-concurrency-fallback.md]) confirm the
kernels execute correctly:

- Single-request Qwen3.5 DFlash: 13.3 → 47 tok/s (block-verify path)
- 4× concurrent: 137.4 tok/s (auto-downgrade to packed batched)
- 8× concurrent: 128.7 tok/s

## Three deviations from the reference (intentional, documented here)

### 1. Tape stored as `bf16`, reference stores `float32`

Reference: `output_dtypes=["y", "state_out", "innovation_tape (f32)"]`.
Ours: `tape_out_dtypes = {bfloat16, float32, bfloat16}` at
`mlx_qwen35_model.cpp:791`, with explicit cast at line 808.

**Why:** `tape_replay_kernel` requires bf16 inputs for `g/k/tape` (it
runs the same `state * g`/`state += k * delta` arithmetic that needs
bf16 for SIMD-friendly throughput). Our `compute_g_impl` produces f32
because `neg_exp_a = -exp(A_log.f32)` is precomputed in f32. Casting to
bf16 in the tape store keeps the dtype contract.

**Cost:** ~8 mantissa bits of delta precision lost vs f32 tape.
Tolerable for ≤16-step replay (typical block) but a documented diff.
If acceptance climbs and longer blocks become viable, revisit.

### 2. No InT round-trip on per-step state

Reference inserts a `state[i] = static_cast<float>(static_cast<InT>(state[i]))`
round-trip at the end of every timestep, modeling the precision loss
of a non-fused implementation that stores state to memory between steps.

Our kernel keeps state in f32 throughout the inner loop — strictly
*more* precise than the reference for chained steps.

**Implication:** Our tape-replay output will diverge slightly from
"naive replay = re-run gated_delta_step T times in a chain." Both are
self-consistent (record + replay through our kernels match), so
correctness for partial-accept rollback is preserved. The divergence
matters only if comparing tape-replay against a reference Python
implementation that uses chained calls.

### 3. No `has_mask` / `vectorized` template variants

Reference generates four kernel variants from
`(has_mask: bool, vectorized: bool)` — vectorized=4D `g [B,T,Hv,Dk]`,
non-vectorized=3D `g [B,T,Hv]`; mask gates per-step compute.

Ours: 3D-`g`-only, no mask. Qwen3.5 uses 3D `g` (per-head scalar
decay), and DFlash blocks are always full (`block_size = 16`, no padding),
so neither variant is needed today. Adding `has_mask` would unblock
varlen / padded-batch DFlash (not on the roadmap).

## Beyond reference: `tape_replay_varlen` (mlx_bridge.cpp:154)

Added to support continuous-batching scheduler runs where each row may
have accepted a different prefix length `T_b`. Takes `steps: [B]` int32
array; each thread group reads its own `T_b = steps[b_idx]` and stops
the replay loop early. Reference has no equivalent — it assumes a
single-request loop.

## Why DFlash on Qwen3.5 still loses single-request despite the port

Mechanical ceiling is fine. The blockers are higher up the stack:

- **Acceptance ~28%** vs reference's 58%
  ([2026-04-17-qwen35-dflash-acceptance-ceiling.md]).
  At 28%, an `S=16` block accepts ~4.5 tokens — verify cost
  (4.6× verify_1 due to GPU pipeline underutilisation on 4-bit quant)
  exceeds the savings.
- **Better draft model** is the only lever that moves acceptance.
  Block-verify shrinks the constant cost per block, but can't change
  the acceptance ratio.

Auto-downgrade preserves c≥4 throughput by routing to packed
batched decode under load, so the regression is bounded to the
single-request path.

## Rule

**Port-vs-reference numerical diffs go in the win/error doc, not just
in code comments.** If we ever debug a tape-replay correctness bug
across implementations, future-us needs to know the bf16/round-trip
diffs are intentional, not regressions.
