# FA3 replaces the scalar ring-attention kernels: 2.17x per training step — and the gate that was missing all along

**Date:** 2026-08-04 · **Commits:** 2fe12a2fe (FA3 bwd substrate) + df75a1da2 (pair dispatch) + a15d3ec75 (test callsite) · **Pod:** 8xH20, real 27B · **Verdict: perf ACCEPT, default OFF pending the correctness gate**

## Context

nsys put the CP ring attention at 31% of the training step and growing with
sequence length. The kernels were flash-semantics but scalar: one warp per
(tile, q_row) walking KV columns with a 4-FMA + 5-shuffle dot product, zero
tensor-core use, and two global `atomicAdd`s per (row, col, dim) in backward.

## What worked

Vendored FA3 (hopper, hd256, bf16, sm90) already ships everything needed — the
forward shim existed; T1 added the backward instantiation plus a torch-free
`arle_fa3_bwd_hd256_bf16_cuda` mirroring `mha_bwd`'s param fill with every
scratch buffer caller-provided. The zigzag causal mask decomposes into
(q_run, k_run) pairs — diagonal pair causal, past pair full, future skipped,
partial overlap a loud error — the Megatron / ring-flash-attention shape. FA3's
normalized `(o, lse)` is exactly a flash-2 block-stats triple (m=lse, l=1), so
the existing (M, L, O) accumulators, finalize kernel and tape context are
untouched; one small merge kernel folds each pair in.

**G2, cp=2 seq=32768, same binary, `ARLE_CP_RING_FA3` A/B:**

| phase | FA3 | scalar | speedup |
|---|---|---|---|
| forward | 32.96 / 31.61 s | 89.49 / 89.27 s | **2.71x** |
| backward | 178.20 s | 369.70 s | **2.07x** |
| step elapsed | 212.1 / 210.7 s | 460.1 / 459.9 s | **2.17x** |

The scalar arm reproduces the standing reference (fwd 89.5 vs 91.7 s, bwd 369.7
vs 375 s), so the A/B is matched. Losses agree at the bf16 floor across all
three configurations — cp=1 anchor 10.870796, FA3 cp=2 10.871086, scalar cp=2
10.870268 (spread 8.2e-4, 7.5e-5 relative), and the FA3 arm sits *closer* to the
single-card anchor than the scalar ring does. Uniform loss reporting
(`fe6cc2346`) also passed its first gate: every rank prints the identical
world-sum in both arms.

## Why the default stays OFF

**The correctness A/B was hollow.** `cph_parity` runs at head_dim 128;
`ring_fa3_route` requires 256, so FA3 never engaged and the two arms came back
bit-identical — a gate on the scalar path wearing an FA3 label. Both CP parity
examples now run at the production head_dim 256.

**A grad divergence surfaced that predates FA3.** Post-backward global grad
norms: cp=1 3.744990, scalar cp=2 1.984009, FA3 cp=2 2.264733. cp=1 vs cp=2 is
1.89x apart and the two cp=2 arms are 14% apart, orders of magnitude outside the
loss spread. Under CP the math forces agreement — each rank's loss carries
inv_n = 1/global_count, so its grad is that rank's partial of the global mean and
`all_reduce_cp_grads` sums them to exactly the single-card grad.

No gate covered this: `nd_parallel_parity` runs the full step (backward,
all-reduce, optimizer) on cp=1 and cp=2 but only ever compared the **loss**. The
CP backward has never been checked against single card. Single card and CP also
run different attention backwards (fused SDPA recompute vs the ring), so neither
is ground truth — the gate must be three-way against the CPU f32 arm.

## Rule

- A parity gate whose config can't reach the code path under test is not a gate.
  Before trusting an A/B, prove the treatment arm actually engaged — bit-identical
  arms are the tell, not the reassurance.
- Gate the quantity you changed. A loss-only gate over a step that also computes
  gradients certifies the forward and silently exempts the backward.
