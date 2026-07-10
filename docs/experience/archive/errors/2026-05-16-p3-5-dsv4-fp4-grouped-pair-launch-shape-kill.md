# P3.5 DSv4 FP4 Grouped Pair Launch Shape KILL

## Context

Kernel family: `dsv4_fp4_grouped_gemv_pair_batch_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

Scope: P3.5 A5 swept `GEMV_THREADS` / `GEMV_ROWS` after the P3.5 A2 pair-load
win (`44d4f9a`). The sweep used global macro edits as temporary probes; no
launch-shape runtime change was shipped.

## Formula Prediction

Baseline `256x4` gives `threads_per_row=64`, two warps per row, and eight FP4
packed pairs per thread for `K=1024`.

Variants:

- `256x8`: one warp per row, 16 pairs per thread, fewer row CTAs.
- `512x4`: 128 threads per row, four warps per row, four pairs per thread.
- `512x8`: 64 threads per row, eight rows per CTA, same per-row thread count
  as baseline but larger CTAs.

Predicted deltas were uncertain: fewer CTAs or less per-thread work could help,
but larger CTAs and extra reduction/occupancy pressure could dominate.

## Root Cause

The hypothesis was falsified. The current `256x4` shape remains the best local
choice for the grouped FP4 pair-load path. Increasing rows per CTA is noise or
slightly worse, while 512-thread CTAs regress, especially at t64.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_grouped_gemv_pair --save-baseline p3_5_a5_before
```

Baseline points:

| Shape | Baseline |
| --- | ---: |
| t4/e4/512x1024 | 18.111 us |
| t64/e4/512x1024 | 178.55 us |

Treatment results:

| Variant | t4 change | t64 change | p | Decision |
| --- | ---: | ---: | ---: | --- |
| 256 threads x 8 rows | +0.2152% | -0.6209% | 0.01 / 0.00 | KILL |
| 512 threads x 4 rows | +3.0750% | +6.2199% | 0.00 / 0.00 | KILL |
| 512 threads x 8 rows | +0.3673% | +0.9495% | 0.00 / 0.00 | KILL |

## Fix

Treatment macros reverted to `GEMV_THREADS=256` and `GEMV_ROWS=4`.

## Rule

Do not change the shared GEMV launch-shape macros for the DSv4 grouped FP4
pair-load path based on local SM89 evidence. If a future variant looks
promising, implement kernel-specific constants first, then rebench the target
and all other affected GEMV filters before shipping.
