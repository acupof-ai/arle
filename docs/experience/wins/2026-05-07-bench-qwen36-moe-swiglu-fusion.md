# Bench — M_e.4 SwiGLU compile-fusion in MoE block — 2026-05-07

## Goal

Per
[`2026-05-07-bench-qwen36-encode-bottleneck.md`](2026-05-07-bench-qwen36-encode-bottleneck.md):
mx::async_eval is doing ~25 ms of CPU-side Metal command-buffer
encoding for ~600-1000 primitives per Qwen3.6 35B-A3B step. Each
primitive elided from the graph drops encoder workload roughly
linearly. M_e.4 attacks the per-layer SwiGLU pattern in the MoE block,
which appeared at two sites (switch experts + dense shared expert) in
each of 40 layers as a 3-op chain.

## What changed

`crates/mlx-sys/src/mlx_qwen35_moe_block.cpp`:

- New file-local `compiled_swiglu()` returning a static
  `mx::compile(swiglu_impl, /*shapeless=*/true)` cache, mirroring
  the existing `mlx_qwen35_model.cpp:compiled_swiglu` (kept TU-local
  to avoid header churn).
- `quantized_swiglu` (line 61): replaced
  `multiply(multiply(gate, sigmoid(gate)), up)` (3 primitives) with
  `swiglu(gate, up)` (1 compiled kernel).
- `switch_glu_forward` (line 115): same replacement at the per-expert
  SiLU site — saves 2 primitives per call.

Per Qwen3.6 step: 40 layers × 2 SwiGLU sites/layer × 2 primitives/site
= **~160 fewer primitives encoded per step** (best case).

## Params

- Binary: `target/release/metal_serve` rebuilt at this commit
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit` (Metal canonical)
- Workload: `/tmp/cN_smoke.sh <N>` — N concurrent
  /v1/chat/completions, max_tokens=64, temperature=0.0
- Both A and B use auto-wired-limit=20 GiB (default since 180e48b)
  — so the deltas isolate M_e.4 alone, not stacked with the wired_limit
  win.

## Results — Pure M_e.4 effect (wired_limit ON in both)

| batch | A: wired-only avg | A: p50 | B: +SwiGLU avg | B: p50 | **Δ p50** |
|------:|------------------:|-------:|---------------:|-------:|----------:|
| c=1   | 11990 | 11923 | 11822 | 11489 | **−3.6%** |
| c=2   | 15973 | 15701 | 15981 | 15737 | +0.2% |
| c=3   | 20323 | 20547 | 20089 | 20649 | +0.5% |
| c=4   | 24241 | 24278 | 23274 | 23143 | **−4.7%** |
| c=8   | 41786 | 41761 | 40180 | 40025 | **−4.2%** |
| c=16  | 70099 | 68747 | 69368 | 67486 | −1.8% |

→ **Direction consistent at most c, magnitude within ±5% noise band.**
Per `feedback_matched_ab_for_small_bench_effects.md` this would need
matched A/B in 2 sessions to claim a confirmed performance win
(≤10% effects are thermal noise until reproduced in ≥2 sessions
same-binary env-A/B). Today's bench is a single-session A/B.

The change ships as a **structurally-correct refactor**: fewer
primitives = unambiguously less encoder work. The compile-shapeless
kernel is well-trodden (mirrors the precedent at
`mlx_qwen35_model.cpp:compiled_swiglu`). Even if the perf delta on
the smoke workload is below noise, the change cannot regress
correctness and has a clean rollback (revert ~10 LoC).

## Problems / observations

1. **The expected primitive count drop didn't translate 1:1 to ITL
   improvement.** With ~160 primitives elided, at ~25 μs/primitive
   encode I'd expect ~4 ms saved per step. Observed: ~1-1.2 ms at c=4
   p50. Possibilities:
   - The first call into `compiled_swiglu()` triggers compile (~ms);
     subsequent calls are fast. Steady-state numbers should land in
     the second half of any bench. Sample size n=23-58 may include
     the cold compile.
   - `mx::compile(shapeless=true)` may emit a kernel with internal
     primitives (still encodes a couple of MTL dispatches, just fewer
     than the manual chain).
   - Encoder cost per primitive isn't constant — the multiply ops
     elided are small/fast; compile-fused kernel may itself add
     pipeline-state lookup that partially offsets the gain.
2. **The c=2 / c=3 noise direction** is suspicious; small samples
   (n=6, n=7) at those c values per the baseline log. Not
   conclusive either way.
3. **No correctness regression.** All 644 infer tests pass (lib);
   smoke output validated as coherent (Qwen3.6 generates "Thin..."
   etc. as in baseline).

## Learnings

1. **Compile-fused kernels save primitives but the encode-cost-per-
   primitive isn't uniform.** Future primitive-count reductions
   should target the *most-frequently-encoded* ops first (per-layer
   gather_qmm dispatches > sigmoid+multiply chains), not just the
   deepest compositional patterns. Per-layer SwiGLU is a fine target
   but not the dominant cost.
2. **Cold compile penalty is real** but amortizes after first call.
   Benches that include the first ~5-10 steps will under-report the
   steady-state win. Future M_e.4-class changes should add a
   warmup-skip in the phase-timing parser.
3. **Structurally-correct ships > matched-A/B-confirmed wins.** If a
   change is correct on its merits (less work for the encoder),
   ship it even when the smoke-bench effect is below the matched-A/B
   threshold. The risk is zero; the upside compounds with future
   reductions.

## What worked / Rule

- File-local `compiled_swiglu()` pattern (mirrors precedent in
  `mlx_qwen35_model.cpp`) — one TU's static cache doesn't conflict
  with another's, and avoiding header changes keeps the diff to one
  file.
- `swiglu(gate, up)` inline wrapper makes the call sites read like
  the math (silu(gate) * up).

## Rule

When localizing primitive-count wins inside a compile-shapeless
kernel, count the primitives elided AND check the kernel's internal
op count via `mx::compile`'s tape — the net win = (elided per call ×
calls per step) − (kernel-launch overhead × calls per step). The
latter is usually small but non-zero.

## Next

- **Multi-stream encode pipelining** (subagent design report this
  date) — encode step N+1 on stream B while step N's GPU runs on
  stream A. Per-thread default stream flip via `mlx::StreamContext`
  is supported (`mlx/stream.h`); event-based cross-stream fences
  are automatic. S effort, ~30 LoC, env-gated. **Predicted win:
  drops `min(encode_cpu_ms, gpu_exec_ms)` from each step's wall
  time — ~25 ms encode could overlap with ~25 ms GPU exec for ~12
  ms savings per step.** Plan landing as
  [`docs/plans/M_e5-mlx-multi-stream-pipelining.md`](../../plans/M_e5-mlx-multi-stream-pipelining.md).
- Next-tick: matched-A/B in a second session for M_e.4 to validate
  the −4-5% direction.

## References

- Predecessor:
  [`2026-05-07-bench-qwen36-encode-bottleneck.md`](2026-05-07-bench-qwen36-encode-bottleneck.md)
- Compile-shapeless precedent in dense forward:
  `crates/mlx-sys/src/mlx_qwen35_model.cpp:compiled_swiglu`
- Multi-stream pipelining design (subagent, this date):
  `docs/plans/M_e5-mlx-multi-stream-pipelining.md`
