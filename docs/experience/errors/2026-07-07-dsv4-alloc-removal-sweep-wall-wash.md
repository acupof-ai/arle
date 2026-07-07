# DSv4 decode alloc-removal sweep — wall-clock WASH (async alloc/memset overlap the GPU)

> Status: KILL (wall-clock) — 2026-07-07. Code kept (correctness-clean cleanup); sweep halted.

## Context
Launch-bound plan Step 1 (`docs/plans/2026-07-07-dsv4-decode-launch-bound-plan.md`):
DSv4 B=1/MTP decode nsys showed `cudaLaunchKernel` 39.8% + `cuStreamSynchronize`
26.6% = 66% wall, and `cuMemAllocAsync`+`Free` (7.7%) + `cuMemsetD8Async` (9.1%)
= 16.8% wall of per-step device allocation, zero `cuGraphLaunch`. Hypothesis:
pooling the ~45 per-layer allocs recovers ~10-12% TPOT. Landed commit 1
(shared-expert scratch, `de6fc4fd`) + commit 2 (MoE-tail 8-buffer scratch,
`4f589cfb`), both correctness-verified (coherent greedy decode).

## Measurement (the KILL)
Matched A/B, same binary-vs-parent, same prompt/256-tok/greedy/TP=4/EP=4/MTP-on,
DSv4-Flash-FP8, GPU 4-7, 3 runs each, `time_total` + `usage.completion_tokens`:

| | runs (s) | mean | tok/s |
|---|---|---|---|
| baseline (`c59aab9c`) | 5.573 / 5.602 / 5.631 | 5.602 | 45.70 |
| c1+c2 pooled | 5.557 / 5.632 / 5.573 | 5.587 | 45.82 |

**Δ = −0.27% wall — inside the ±0.7% run-to-run noise. WASH.**

## Root cause of the mis-inference
The 16.8% "alloc+memset wall" is an **API-time** share, not recoverable wall. On
an async CUDA pipeline, `cuMemAllocAsync`/`cuMemsetD8Async` are stream-queued and
**overlap GPU compute** — they do not block the wall. The wall is gated by the
per-step `ctx.sync()` (`ops.rs:467`, the 26.6% `cuStreamSynchronize`) + the serial
GPU chain + cross-process TP lockstep. Removing allocs cuts host-side launch/API
time that was **already hidden behind the GPU**, so wall doesn't move.

This is a verbatim repeat of `errors/2026-06-20-host-launch-bound-misinference-decode-is-foundation-bound.md`:
a profiler `API% ≈ wall` does NOT prove the API work is the wall — in an async
pipeline it overlaps and is hidden. The deciding evidence is the A/B, not the
profiler share. I re-inferred past my own recorded lesson (again).

## Disposition
- **Code kept**: commit 1+2 are correctness-clean, reduce alloc churn by
  construction, and match the eager-path scratch discipline — a legitimate
  cleanup. NOT a perf win; the wins entries are re-labeled accordingly.
- **Commit 3 (attn/ffn stream double-buffer + N-ring) KILLED before implementation**:
  same async-overlap fate; the remaining allocs are also hidden behind the GPU.
- **The real wall lever is foundation, not allocs/kernels**: per-step `ctx.sync`
  → device-side sampling (let host run ahead), and 4-process TP → single-process
  TP (kill the per-tick cross-process barrier). Both are architecture changes,
  out of the kernel/alloc scope. Do NOT attack decode wall via kernel/alloc
  micro-opts again without first moving the ctx.sync/lockstep foundation.

## Rule
Before pooling/fusing to cut a profiler API% (alloc, memset, launch), confirm the
work is NOT already overlapped by the GPU — async ops queued on the compute stream
are hidden behind kernels and their API% is unrecoverable wall. The isolating test
is a wall-clock A/B, and on a per-step-synced decode pipeline the answer is almost
always WASH. The lever is the sync/lockstep that stops host run-ahead, not the
overlapped API work. (3rd time this exact trap: see 2026-06-20, 2026-06-08.)
