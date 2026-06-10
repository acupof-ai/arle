# mHC params warp-parallel Sinkhorn tail: 39.51 → 41.86 tok/s (+5.9%) — Rung 1 of the fusion ladder

**Date:** 2026-06-10 (night). 8×H20 TP=8/EP=8 DSv4-Flash FP8, default serve
(masked eager, FlashMLA, allreduce), `dsv4_ab_bench.py` B=1.

## Root cause (census → kernel autopsy)

`dsv4_mhc_params_kernel` was the largest non-GEMM kernel in the decode
census: 61-86 inst/token × **35.5 µs** = 2.2 ms/token — on a kernel whose
output is 24 floats. Autopsy: at B=1 the launch is `<<<1, 256>>>` (one block
on a 78-SM H20), and after the block-parallel sumsq the code hit
`if (threadIdx.x != 0) return;` — **8 sigmoids + hc_sinkhorn_iters=20
Sinkhorn iterations (~32 dependent divisions each) ran on ONE THREAD**.
Pure serial-chain wall time at B=1 (every kernel is on the critical path).

## Fix

hc_mult==4 (production shape) tail goes warp-parallel: lanes 0-7 run the
pre/post sigmoids; lanes 0-15 each own one cell of the 4×4 Sinkhorn matrix
(lane = row*4+col), row sums via `__shfl_xor_sync` bits 0-1, column sums via
bits 2-3. Generic hc_mult path unchanged (single-thread fallback). FP sums
associate pairwise instead of left-to-right → last-ULP drift vs the old
tail; gated by correct-inference, not byte-identity.

## A/B

| arm | B=1 p50 | note |
|---|---|---|
| before (571ccbd1 lineage) | 39.51 | same day, same serve config |
| **warp tail** | **41.67 p50 / 41.86 mean** | output correct, 8/8 seq |

Predicted +1.7 ms/token (35.5→~8 µs × 86) ⇒ 42.3 tok/s; measured 41.9.
Cross-binary A/B (kernel replacement, no env flip) — interim binaries only
added graph-gated/inert changes to this path.

## Rules

- **A 1-block, 1-thread tail on the B=1 serial chain is wall time, not
  "small kernel noise"** — census by (count × duration), then autopsy the
  outliers; 35.5 µs for 24 output floats was the tell (same class as the
  mask-oblivious silu grid).
- Sinkhorn/iterative normalizers over k×k tiles are warp-shuffle shaped:
  k²≤32 cells = 1 lane each, xor-tree row/col sums.

## Next (Rung 1 remainder, census-priced)

pack_quantize 184/tok ×3.2 µs = 0.59 ms (fuse into rms_norm/producer
epilogue); splitKreduce 196/tok = 0.34 ms (cuBLAS splitK shape tuning or
in-house gemv); rms_norm 123/tok = 0.31 ms (fuse with hc_pre);
compressor_update 44/tok = 0.26 ms. Then Rung 2 (segment mini-megakernel)
absorbs the rest + inter-kernel gaps.

## Refs
- Census + ladder: [`errors/2026-06-10-dsv4-wholestep-graph-production-path-wash-rekill.md`](../errors/2026-06-10-dsv4-wholestep-graph-production-path-wash-rekill.md)
- Megakernel framing: Hazy "No Bubbles" (2025-05-27)
