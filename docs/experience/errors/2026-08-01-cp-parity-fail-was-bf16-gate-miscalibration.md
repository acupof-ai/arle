# CP "parity FAIL" was a bf16-vs-f32 gate miscalibration, not a bug — 2026-08-01

## Context

The 256K context-parallel (CP) ring landed and `nd_parallel_parity` "FAILed":
seq=16 `rel_err=3.9e-3` at head_dim=2, then `5.2e-2` at head_dim=128, both
against a `REL_TOL=1e-3`. The 131072 gate showed `2.2e-2`. The loss was
asymmetric across zigzag ranks, which looked like a per-shard position bug. Two
sessions were spent hunting a CP correctness bug that did not exist.

## Root Cause

The gate compared **two independent bf16 attention implementations against each
other** at a near-identity tolerance: the single-card reference runs the
bf16 chunked-prefill kernel, the CP path runs the bf16 ring kernel. bf16
attention at a tiny random-weight scale is ~8% off f32 — so two correct bf16
paths differ from each other by several %. `1e-3` was measuring rounding, not
correctness. head_dim=2 only "passed closer" (0.39%) because the ref fell off
the prefill envelope onto f32 composed SDPA — a *different* precision pairing,
not a smaller bug. The "grows with head_dim" and "zigzag-rank-asymmetric"
signals were both bf16-noise artifacts, not a position/gather fault.

Proven by a 3-stage device bisection (each a single pod run):
1. **kernel** — hand-fed 2-block GQA zigzag `ring_block_fwd_merge` vs host
   `ring_forward_tile`: PASS (fp32 eps).
2. **transport** — 2-rank NCCL ring, zigzag shards, vs full-seq causal SDPA:
   PASS (3e-8, non-contiguous shard as clean as contiguous).
3. **forward + f32 anchor** — CP hidden vs single-card AND vs CPU-f32:
   `cp_vs_cpu_f32=6.6e-2 < single_vs_cpu_f32=8.0e-2`. CP tracks f32 **better**
   than the single card. A real bug drifts CP *away* from f32; this is the
   opposite.

## Fix

Anchor the gate on the f32 ground truth, never on bf16 byte-identity
(`4e1076a6b`): run a CPU-f32 forward and require `cp_vs_f32 <= single_vs_f32 +
TRACK_MARGIN` (2e-2). A wrong gather/shard/position/inv_n drifts CP from f32 far
past the margin; symmetric bf16 rounding passes. seq>4096 (f32 ref is O(seq²)
host RAM, the wall the ring avoids) falls back to a liveness + loose
bf16-vs-bf16 gross-error bound.

## Rule

When a "parity FAIL" pits two independent low-precision kernels (bf16/fp8)
against each other, the FIRST question is "what is the precision floor between
these two paths?" — compute one bf16-vs-f32 number before hunting a bug. Two
correct bf16 attention paths differ several % at small/random-weight scale; a
byte-identity tolerance there is a miscalibrated gate. Anchor the gate on the
f32 reference the low-precision paths approximate, and make the invariant "the
new path tracks f32 at least as well as the baseline does," not "the two match."
This is the same `correct-inference ≠ baseline-identity` trap, one layer down.
