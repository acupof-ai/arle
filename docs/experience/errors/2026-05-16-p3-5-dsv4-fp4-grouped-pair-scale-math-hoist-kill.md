# P3.5 DSv4 FP4 Grouped Pair Scale Math Hoist KILL

## Context

Kernel: `dsv4_fp4_grouped_gemv_pair_batch_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

Scope: P3.5 A1 tested hoisting the row-side scale geometry out of the inner
K loop and sharing the scale-column index between `scales_a` and `scales_b`.
The bench harness added in `c85ad3f` covers `N=512`, `K=1024`, four experts,
and total routes 4 / 64.

## Formula Prediction

Hypothesis before edit:

- SM89 constants: 64K registers/SM, 100KB shared memory/SM, 1536 threads/SM,
  672 GB/s HBM.
- Workload constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`,
  `threads_per_row=64`, `K=1024`, `scale_rows=4`, `scale_cols=8`,
  `block_w=128`.
- Baseline calls `dsv4_block_scale` twice per logical K element: once for
  `scales_a` and once for `scales_b`.
- Treatment hoisted `block_h`, `block_w`, `sr`, and `scale_row_offset`, then
  computed `sc = k / block_w` once per K element and reused it for both scale
  arrays.
- Weight loads, input loads, launch shape, FP4 decode, and accumulation order
  were unchanged.

Predicted point delta was >3% if the compiler did not fully common-subexpression
the inlined `dsv4_block_scale` calls.

## Root Cause

The hypothesis was falsified. Hoisting row-side scale geometry and sharing the
scale-column index is a real but sub-2% improvement on both local grouped-pair
shapes, which means the current kernel is not dominated by this scale-address
math. The remaining cost is more likely packed FP4 weight/input traffic,
duplicate packed-byte loads, or launch/grid work, but that remains hypothesis
until the later axes are tested.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_grouped_gemv_pair --save-baseline p3_5_a1_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_grouped_gemv_pair --baseline p3_5_a1_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| t4/e4/512x1024 | 25.840 us | 25.606 us | -1.0404% | 0.00 | KILL |
| t64/e4/512x1024 | 299.21 us | 296.00 us | -1.0402% | 0.00 | KILL |

## Fix

Treatment reverted. Keep the existing `dsv4_block_scale` calls in
`dsv4_fp4_grouped_gemv_pair_batch_kernel` until a larger axis licenses a
coherent rewrite.

## Rule

Do not land isolated row-side scale math hoisting for the DSv4 FP4 grouped pair
kernel on the local SM89 shapes. The effect is stable but below the Phase 3
license and review thresholds.
