# P3.5 DSv4 FP4 Grouped Pair Scale Reuse KILL

## Context

Kernel: `dsv4_fp4_grouped_gemv_pair_batch_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

Scope: P3.5 A4 tested reusing the FP8 E8M0 scale for the low/high nibbles in
one FP4 packed byte after the P3.5 A2 pair-load win (`44d4f9a`). The local
shape uses `K=1024`, `scale_cols=8`, and `block_w=128`, so `k0` and `k1`
within each packed pair share a scale column.

## Formula Prediction

Hypothesis before edit:

- Baseline pair-load uses four `dsv4_block_scale` calls per packed pair:
  `a/k0`, `a/k1`, `b/k0`, `b/k1`.
- Treatment reused `k0` scale for `k1` when both K positions map to the same
  scale column, with a fallback for scale-column boundaries.
- On the local shape, all pairs are same-column pairs, so dynamic scale helper
  calls should drop from four to two per packed pair.

Predicted point delta was >3% if scale decode/addressing remained a meaningful
post-pair-load cost.

## Root Cause

The hypothesis was falsified. After packed-byte pair-load, scale reuse is not a
local bottleneck. The branch and `block_w` work cancel out the reduced helper
calls on t4, and t64 is effectively unchanged.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_grouped_gemv_pair --save-baseline p3_5_a4_before
```

Treatment reused the `k0` scale for `k1` when `k0 / block_w == k1 / block_w`.

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| t4/e4/512x1024 | 18.130 us | 18.208 us | +0.4223% | 0.00 | KILL |
| t64/e4/512x1024 | 178.55 us | 178.42 us | -0.0615% | 0.00 | KILL |

## Fix

Treatment reverted. Keep independent `dsv4_block_scale` calls for `k0` and
`k1` in the grouped FP4 pair kernel.

## Rule

Do not land low/high FP4 scale reuse for DSv4 grouped FP4 pair GEMV on the
local SM89 pair-load path. The measurable effect is near zero and not worth
the extra branch and generic boundary handling.
