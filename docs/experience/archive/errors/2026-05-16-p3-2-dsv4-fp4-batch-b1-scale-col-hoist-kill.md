# P3.2 DSv4 FP4 batch B1 scale-col hoist kill

## Context

Phase 3 P3.2 A1 tested scale-column hoisting in
`dsv4_fp4_gemv_batch_kernel`, the B=1 raw path behind
`dsv4_fp4_gemv_batch_cuda`.

## Formula Prediction

Hypothesis before edit:

- SM89 constants: 64K registers/SM, 100KB shared memory/SM, 1536 threads/SM,
  672 GB/s HBM.
- Workload constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`,
  `threads_per_row=64`, `K=1024`, `scale_cols=8`, `block_w=128`, `B=1`.
- Baseline does one `k / block_w` and one E8M0 scale decode per inner-loop
  element. Each thread handles about 16 K elements.
- Treatment loops over `scale_cols`, decodes the scale once per scale-column
  range, and runs an inner K loop inside the range. This reduces each thread
  from about 16 scale decodes to about 8 scale decodes and removes per-k
  integer division.
- Weight and input memory traffic are unchanged.

Predicted point delta was -4% to -9% for `1024x1024` and -3% to -8% for
`512x1024`.

## Root Cause

The hypothesis was falsified. The scale-column outer loop adds enough control
flow, bounds checks, and segmented-loop overhead to outweigh the removed
per-k scale-index work. This mirrors the P3.1 FP8 B=1 finding: source-level
scale-column hoisting is not a standalone win on these local SM89 B=1 raw
GEMV shapes.

## Evidence

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --save-baseline p3_2_a1_before
```

Treatment command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --baseline p3_2_a1_before
```

| Shape | Baseline point | Treatment point | Criterion change | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| `dsv4_mini_hidden_1024x1024` | `11.337 us` | `11.580 us` | `+2.1624%` | `0.00 < 0.05` | KILL |
| `dsv4_mini_moe_512x1024` | `8.9283 us` | `9.9734 us` | `+11.722%` | `0.00 < 0.05` | KILL |

## Fix

Reverted the A1 runtime change. Keep per-k scale-column indexing in
`dsv4_fp4_gemv_batch_kernel`.

## Tradeoff

- LOC complexity: treatment added an outer `scale_cols` loop plus range bounds.
- SM89 specificity: measured locally on RTX 4070 Ti SUPER / SM89.
- Shared memory budget: unchanged.
- Register budget: slightly worse due scale/range temporaries.
- CUDA Graph compatibility: unchanged for fixed shapes.
- Generality across batch sizes: B=1 only; B>1 tiled path was not touched.
- Generality across shape: both hidden and MoE regressed.
- Numerical correctness margin: changed accumulation order; correctness would
  have needed a tolerance gate if licensed, but performance failed.

## Rule

Do not ship scale-column hoisting for `dsv4_fp4_gemv_batch_kernel` B=1 on the
current local shapes. The segmented loop cost dominates the removed per-k scale
work.
