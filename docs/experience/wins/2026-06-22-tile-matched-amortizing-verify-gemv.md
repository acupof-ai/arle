# TILE==B-matched amortizing verify GEMV — the spec-decode net-win lever (L4-validated)

## Context

NextN-MTP spec-decode on Qwen3.6-27B-FP8 (CUDA) net-LOST at every depth
(depth 1/2/4/8 ≈ 0.90/0.73/0.50/0.32× the no-spec baseline). Root cause, now
measured: B=1 decode is ~50%-HBM-ceiling-bound (see
[errors/2026-06-22-b1-decode-gemv-is-hbm-ceiling-bound](../errors/2026-06-22-b1-decode-gemv-is-hbm-ceiling-bound.md)),
so the only path past the roofline is spec-decode amortizing the weight read
across the verify batch. But the verify GEMV did NOT amortize: the default decode
kernel uses `grid.y = B`, re-reading the entire weight row once PER batch column
(gemv_bench: B=2/4/8 cost 2.15/3.59/6.50× a single decode — linear in B, zero
amortization), and the opt-in tiled kernel used a fixed `QWEN_GEMV_BATCH_TILE=8`
`sums[8]` array that craters occupancy at the small depths spec-decode actually
uses (H20 measured B=2 → 1.95×).

## What Worked

Make the tiled kernel's TILE a **compile-time template param** and instantiate
`TILE == B` (the exact spec depth) in the launcher, so `sums[TILE]` uses exactly
B registers and `grid.y = 1` (weight read ONCE). The extra verify columns then
hide in the B=1 cold-weight-load latency bubbles instead of spilling registers.

L4 sm_89 (gemv_bench `GEMV_AMORT`, cosine 1.0 vs oracle, roofline-relative
ratios transfer across GPUs):

| B | default (`grid.y=B`, re-read weight B×) | TILE==B (weight once) |
|---|---|---|
| 2 | 2.15× | **1.04×** |
| 4 | 3.59× | **1.07×** |
| 8 | 6.50× | **1.14×** |

A depth-4 verify producing up to 5 tokens now costs ~1.07 decodes (was 3.59),
so even modest acceptance flips MTP to a net win. H20's B=1 idle (65%, vs L4's
~38%) is larger → the amortization is at least as good there; L4 is the
conservative bound.

Ported to production `fp8_f32_block_gemv_batch_tiled_kernel` (now
`template<int TILE>`; launcher dispatches `TILE==B` for B≤8, `grid.y=1`; B>8
falls back to the fixed-8 tile). Behind the existing opt-in `ARLE_QWEN_GEMV_TILED`
(default decode path byte-unchanged). nvcc sm_89 `-c` compile-clean; kernel math
byte-identical to the scalar GEMV (`dot16_with_decoded`).

## pending-remote

The 27B-FP8 MTP **end-to-end net-win re-bench** (depth-sweep tok/s with the
amortizing verify, `ARLE_QWEN_GEMV_TILED=1`) needs an H20-class GPU — the H20
pod was decommissioned mid-session (tunnel to 180.184.176.218 went dark). Colab
L4 validated the kernel/amortization but can't hold the 27GB model. Re-run the
`mtp_spec_decode_gate_and_bench` depth-sweep when H20-class hardware is back;
expect the depth-2/3 rows to cross above 1.0× given verify is now ~1.05×.

## Rule

A batched decode/verify GEMV must instantiate the accumulator tile at the
ACTUAL batch (template on B), not a fixed max — a fixed-8 `sums[]` register array
craters occupancy at the small batches spec-decode uses, masking the
weight-read amortization that is the entire point of batching. Validate the
kernel + amortization curve in the standalone `gemv_bench` proxy (any sm_80+
GPU, cosine gate) before the model-level re-bench; the ratio (verify-cost /
single-decode) is the spec-decode go/no-go, and it transfers across GPUs.
