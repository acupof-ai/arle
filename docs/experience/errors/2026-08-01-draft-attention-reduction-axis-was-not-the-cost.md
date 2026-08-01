# DSpark draft attention: the reduction axis was not the cost

## Context

`[dspark-draft-b]` phase splits put 33 ms of the 63 ms DSpark draft forward in
the attention kernel — 15× the neighbouring MLP GEMM that moves 10× more data.
Reading `nonpaged_prefill_attention_kernel` showed the QK loop gives every key
its own block-wide reduction: thread `dim` holds one head dimension, so a
2048-key window serializes 2048 dependent `warp_reduce_sum`es plus 128
`__syncthreads`. That looked like the whole story, so `2f7cdc145` swapped the
axis — one warp owns one key, lane strides `head_dim` in registers, `num_warps`
keys in flight, one reduction per key — and hoisted `expf` out of the per-thread
AV loop.

## Root Cause

The kernel is not reduction-latency-bound at the row counts that carry the load.
Matched A/B, same tree, same prompts, same seed, H20 GPU 0,
ThinkingCap-Qwen3.6-27B-FP8 + `dspark-fr-native` block 6, 128 reqs at c=16;
baseline binary built from the same HEAD with only the `.cu` reverted:

| draft rows | baseline attn ms | key-major attn ms | Δ | n (base/new) |
|---|---|---|---|---|
| 48 | 14.62 | 11.83 | −19% | 6 / 13 |
| 72 | 17.22 | 18.11 | +5% | 29 / 33 |
| 78 | 17.52 | 19.49 | +11% | 38 / 42 |
| 84 | 18.71 | 21.49 | +15% | 59 / 42 |
| 90 | 24.99 | 25.55 | +2% | 65 / 69 |
| 96 | 28.93 | 28.55 | −1% | 101 / 91 |

End-to-end (`probe.sh`, block 16, 64 reqs): c=1 TPOT 9.19 → 9.04 ms, c=16
103.86 → 105.37 ms, accept 0.411 → 0.412 and 0.395 → 0.395. Accept identical
confirms the reassociated softmax is numerically safe — it just does not pay.

At 96 rows the grid is 32 heads × 96 = 3072 blocks; there is far more
parallelism than needed to hide a per-key reduction. The key-major variant also
issues four 64-byte loads per warp where the dim-major one issued one 256-byte
load for the same bytes, which cancels the reduction saving at high occupancy
and loses at the middle row counts.

The row scaling — 2.4 ms at 12 rows → 28.9 ms at 96 — says only that the kernel
is *saturated*: past that point every bottleneck scales linearly. It does not
name the unit, and the second hypothesis it suggested (GQA re-read of the K/V
window) was also wrong. ncu on a pinned-shape standalone harness settled it:
L2 hit rate 99.58% with L2 only 7.29% busy, so the re-read is free; the cost is
**Compute (SM) 80.15% with the ALU pipeline at 61.9%** — a runtime-modulus IDIV
per key per thread. Removing that IDIV won 33% in a pinned-shape microbench and
still lost 2.7% in the serve — see
[the second revert](2026-08-01-draft-attention-idiv-win-is-microbench-only.md).

## Fix

Reverted (`aa4d2a6ec`). The change is a wash on the workload and a regression on
part of it, so it does not earn its diff.


## Rule

**ncu the shape that costs the time, before rewriting the kernel.** Naming an
inefficiency from source inspection ("this reduction is obviously serial") is a
hypothesis, and so is naming one from a scaling curve — a curve says the kernel
is saturated, never which unit saturated. When the serve-side profile is
confounded (ncu serializes launches, the batch collapses, the grid shrinks out
of the costly regime), extract the kernel verbatim into a standalone `.cu` with
the shape pinned from the model config. The same evidence bar that applies to a
bug applies to a cost.

**A matched A/B for a kernel needs a baseline binary from the same HEAD.** The
pre-existing pod binary predated the phase instrumentation and had no comparable
counters at all; rebuilding current HEAD with only the `.cu` reverted was the
only way to get an apples-to-apples number.
