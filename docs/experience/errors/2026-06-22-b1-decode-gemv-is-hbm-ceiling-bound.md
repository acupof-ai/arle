# B=1 FP8 decode GEMV is HBM-ceiling-bound (~50%), not a fixable kernel — and 180 t/s is above the single-stream roofline

## Context

Goal: speed up Qwen3.6-27B-FP8 (DENSE, 27 GB) CUDA decode and make NextN-MTP
spec-decode a net win on H20. The no-spec B=1 decode measured 45.7 tok/s; ncu of
the default scalar block-scaled GEMV (`fp8_f32_block_gemv_batch_kernel`) showed
**35% DRAM, 78% L1/TEX, 85% occupancy, weights read exactly N·K once**. The
working hypothesis (held across several sessions) was "the GEMV is
compute/issue-bound on the fp8 dequant+FMA; a SOTA memory-bound kernel
(Marlin/cp.async) gets it to ~90% HBM → 3× decode + amortized spec verify."

## Root Cause

The hypothesis was wrong on two independent counts, both now measured:

1. **The dominant stall is cold-weight DRAM latency, not dequant/activation.**
   ncu `WarpStateStats` (B=1, qkv): Long Scoreboard **6.55** cyc/issue (global
   load latency = cold weight) ≫ Short Scoreboard **0.50** (L1-hit / activation)
   ≈ Math throttle **2.00**. The whole "activation re-read / dequant is the
   bottleneck" framing was false — activation L1 hits are negligible.

2. **~50% HBM is the practical ceiling for this shape on H20, not 90%.** A
   trivial coalesced weight-read kernel that does NOTHING but XOR (no decode, no
   activation, no FMA) caps at **avg 50%** roofline (qkv/o_proj ~42% on the
   31 MB shapes, gate/down ~58% on 89 MB). The kernel runs 18–39 µs — too short
   for a cold 27–89 MB read to reach HBM steady state. Industry corroborates:
   SOTA H100 decode "rarely exceeds 50% HBM" / "~32% during distributed decode"
   (arxiv 2602.18568). The 35% GEMV is normal decode efficiency.

Four single-variable levers tested in `gemv_bench` (B=1, all cosine 1.0 vs
oracle), all WORSE than the 34.9% baseline because they attacked non-bottlenecks
and/or cost occupancy: N register-blocking R=2/4/8/16 → 33/27/23/17%; smem
activation staging → 22% (the `__syncthreads` fill serialized the cross-warp
overlap); depth-2 weight prefetch → 33% (nvcc -O3 already software-pipelines the
loop). Marlin's "3×" is the INT4-vs-FP16 quantization byte reduction — ARLE is
already FP8 (1 byte/weight = minimum read), so there is no Marlin headroom left.

**Decisive arithmetic:** 180 tok/s × 27 GB = 4.86 TB/s > H20's 4 TB/s HBM peak.
The 100%-HBM single-stream roofline is **148 tok/s**; ~50% ceiling ≈ 74 tok/s.
No dense B=1 kernel can hit 180 — it would require reading the weights faster
than HBM physically allows.

## Fix

Stop optimizing the B=1 GEMV — it is ceiling-bound; there is no code to copy and
no physical headroom (35% → at most ~50%, a 1.4× that still can't reach 148).
The only lever that breaks the memory wall is **speculative decode (MTP):
multiple accepted tokens per single weight-read pass**, whose effective tok/s
can exceed the roofline. Re-anchor on MTP economics — the verify must use the
weight-amortizing batched kernel (read weight once, MAC across the B=depth
columns), and acceptance rate is the dial. Measured in `gemv_bench`: the
weight-amortizing kernel makes a B=8 verify cost 2.4× a single decode (vs 5.9×
for the default `grid.y=B` kernel that re-reads the weight B times) — useful,
but the per-column activation+FMA still grows linearly, so amortization is
partial, and acceptance is what determines the net win.

## Rule

For a memory-bound decode GEMV: **measure the warp-stall reason and the
pure-streaming HBM ceiling BEFORE rewriting the kernel.** "35% roofline" is not
evidence of a bad kernel — short single-stream decode kernels are ~50%-ceiling
bound on H20/H100 (industry-confirmed), and the dominant stall is cold-weight
DRAM latency, not dequant. A throughput target above the
weight-bytes-per-second roofline (here 148 tok/s for 27 GB @ 4 TB/s) is
**only** reachable via spec-decode/quantization-byte-reduction, never via a
faster same-precision kernel. Don't chase B=1 kernel %roofline; chase
tokens-per-weight-pass (acceptance × amortized verify).
