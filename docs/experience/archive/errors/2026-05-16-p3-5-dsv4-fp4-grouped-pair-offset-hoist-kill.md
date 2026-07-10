# P3.5 DSv4 FP4 grouped pair offset hoist kill

## Context

Phase 3 P3.5 A7 tested a narrow redundant-addressing tweak for
`dsv4_fp4_grouped_gemv_pair_batch_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

The only treatment was to hoist:

- `row * bytes_per_row` into `weight_row_offset`
- `route * N + row` into `output_offset`

No scale math, FP4 decode, launch shape, accumulation order, or memory layout
was changed.

## Formula Prediction

Hypothesis: removing repeated integer address expressions from the hot loop and
tail store may reduce per-thread integer work. Risk: nvcc already performs the
same strength reduction, or extra live values increase register pressure.

## Root Cause

The hypothesis was falsified. The t4 shape improved by only 0.3014%, and t64
had no statistically significant movement. This is below the Phase 3 review
threshold and far below the 3% license threshold.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp4_grouped_gemv_pair --save-baseline p3_5_a7_before
```

Baseline results:

| Shape | Time |
|---|---:|
| `dsv4_mini_t4_e4_512x1024` | `18.098 us` |
| `dsv4_mini_t64_e4_512x1024` | `178.57 us` |

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp4_grouped_gemv_pair --baseline p3_5_a7_before
```

Treatment results:

| Shape | Time | Change | p-value | Decision |
|---|---:|---:|---:|---|
| `dsv4_mini_t4_e4_512x1024` | `18.059 us` | `-0.3014%` | `0.00` | KILL: below review threshold |
| `dsv4_mini_t64_e4_512x1024` | `178.55 us` | `-0.0151%` | `0.16` | KILL: no significant change |

## Fix

The treatment was reverted. No runtime patch was shipped.

## Rule

Do not land isolated row/output offset hoisting for the grouped FP4 pair GEMV
kernel. The compiler or existing addressing already removes nearly all of the
available cost on local SM89.
