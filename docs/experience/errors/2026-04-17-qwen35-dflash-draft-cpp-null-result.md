# DFlash draft C++ compile is a wash on Qwen3.5 single-session

## Context

After target-forward `mx::compile` turned out non-viable without a
cache-as-input redesign
([2026-04-17-qwen35-target-mx-compile-not-viable.md](2026-04-17-qwen35-target-mx-compile-not-viable.md)),
the parallel bet was the draft-side port: rewrite the DFlash draft
forward (5 layers, stateless w.r.t. persistent KV) in C++ as one
`mx::compile(shapeless=true)` graph and gate it behind
`DFLASH_DRAFT_CPP=1`. Hypothesis: fewer Rust→MLX hops and a fused
5-layer graph would cut draft latency enough to close part of the
`-35% vs plain decode` gap on 4-bit Qwen3.5-4B.

## What happened

Port landed (`crates/mlx-sys/src/mlx_dflash_draft_model.cpp`, 368 LOC).
Two tracing bugs surfaced during bench:

**Bug 1 — GQA reshape target.** `reshape(attn, {seq, hidden_size})`
was wrong for this draft model: `num_heads * head_dim = 4096 != hidden_size = 2560`
because `o_proj` does the 4096→2560 collapse. Fixed to `{-1, num_heads * head_dim}`.

**Bug 2 — shape baked into trace.** `const int seq = hidden_states.shape(0)`
at the top of the compiled lambda, then `reshape({1, seq, ...})`. That
`seq` became a trace-time constant. First call traced with seq=16 and
total_len=24; the second call with different lengths tripped:
`[reshape] Cannot reshape array of size 17408 into shape (1, 24, 8, 128)`.
Fixed by using `-1` for the variable dim in every reshape and deleting
the `seq`/`context_len`/`total_len` locals entirely.

## Numbers (M4 Max, 4-bit Qwen3.5-4B, 3×256 tokens, concurrency=1)

| Config | Throughput | Wall |
|---|---|---|
| DFlash baseline (no DRAFT_CPP) | 43.3 tok/s | 17.74s |
| `DFLASH_DRAFT_CPP=1` | 43.5 tok/s | 17.64s |

Delta: **+0.5% — within run-to-run noise.**

## Root cause (why the win didn't materialize)

Draft forward is not the bottleneck. DFlash single-session on 4-bit
Qwen3.5 is dominated by the target `block_verify` forward (32 layers,
8 full-attn + 24 GDR, quantized matmul), not the 5-layer draft. Fusing
the draft graph saved a handful of FFI crossings but the target
verify still runs the same number of Metal dispatches per block.

## Rule

**Before porting a subgraph to C++/compile, measure its share of the
step time, not just the structural reason it should be faster.**
"Stateless forward → compilable" is a correctness argument, not a
speedup argument. Single-session DFlash on 4-bit Qwen3.5 is
verify-bound; draft-side optimizations move the needle <1%.

**Corollary to the shape-baking rule:** `array::shape(N)` inside a
compiled lambda behaves exactly like `array::item<T>()` — the returned
integer is a trace-time value, not a dynamic graph input. Use `-1` in
reshape targets and never call `shape()` to derive intermediate
constants inside the traced function. The compile-as-input contract
must carry shape information via the input arrays themselves or
explicit rank-0 `array` inputs.

## What we keep

- `mlx_dflash_draft_model.cpp` and the `DFLASH_DRAFT_CPP=1` gate stay
  landed but default-off. It is correct; it just doesn't help today.
- The compile-as-input pattern (take active KV prefix as graph inputs,
  `concatenate` for state updates, no `slice_update` with runtime
  positions) is the only way MLX compile cooperates with stateful
  forwards. Captured for future target-side compile work.

## Follow-ups

- Block-verify profile window (`QWEN35_DFLASH_PROFILE=1`) is still the
  right next instrument — measure where the verify block spends its
  time before reaching for compile or custom kernels.
- Reference `dflash-mlx` `~56.7ms` verify vs our `~54ms` suggests we
  are already within a few percent on verify cost; the gap to plain
  decode is more likely acceptance-rate / block-amortization shaped
  than raw-kernel shaped.
- Target-forward cache-as-input refactor remains open but
  not-critical-path while single-session DFlash is still
  underwater vs plain decode.
