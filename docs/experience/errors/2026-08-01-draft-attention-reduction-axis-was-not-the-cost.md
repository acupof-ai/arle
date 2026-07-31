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
parallelism than needed to hide a per-key reduction. What is left is traffic,
and the kernel forgoes all of it: **one block per (q_head, token) means each of
the 8 q heads sharing a kv head re-reads that head's whole K and V window.** The
key-major variant issues four 64-byte loads per warp where the dim-major one
issued one 256-byte load for the same bytes, which cancels the reduction saving
at high occupancy and loses at the middle row counts.

The row scaling is the tell and it was in the data before the rewrite: 2.4 ms at
12 rows → 28.9 ms at 96 rows is linear in rows, which is a bandwidth signature,
not a latency one. Latency-bound work flattens as parallelism grows.

## Fix

Reverted (`aa4d2a6ec`). The change is a wash on the workload and a regression on
part of it, so it does not earn its diff.

The founded next attempt is GQA reuse: batch the `gqa_ratio` q heads sharing a
kv head into one block so K/V is read once instead of eight times. That attacks
the measured driver rather than the one that merely looked wrong on inspection.

## Rule

**Read the scaling curve before rewriting the kernel.** A per-lap phase probe
already emits time against row count; linear-in-work says bandwidth and flat-
then-rising says latency, and that one plot picks the right rewrite. Naming an
inefficiency from source inspection ("this reduction is obviously serial") is a
hypothesis about the bottleneck, not a measurement of it — the same evidence bar
that applies to a bug applies to a cost.

**A matched A/B for a kernel needs a baseline binary from the same HEAD.** The
pre-existing pod binary predated the phase instrumentation and had no comparable
counters at all; rebuilding current HEAD with only the `.cu` reverted was the
only way to get an apples-to-apples number.
