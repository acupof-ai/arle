# P3.1 DSv4 FP8 batch B1 row pointer hoist kill

## Context

Phase 3 P3.1 A7 tested a small inner-loop redundancy removal in
`dsv4_fp8_gemv_batch_kernel`, the B=1 raw path behind
`dsv4_fp8_gemv_batch_cuda`.

## Formula Prediction

Hypothesis before edit:

- SM89 constants: 64K registers/SM, 100KB shared memory/SM, 1536 threads/SM,
  672 GB/s HBM.
- Workload constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`,
  `threads_per_row=64`, `K=1024`, `B=1`.
- Baseline source indexes the row weight as `weight[row * K + k]` inside the
  K loop. The treatment hoists `weight + row * K` once and indexes
  `row_weight[k]`.
- This removes an apparent loop-invariant multiply/add from source. The
  compiler may already hoist or strength-reduce it, so expected benefit was
  small.

Predicted point delta was -0.5% to -2%. This was expected to be a likely KILL
unless the compiler missed the source-level invariant.

## Root Cause

The hypothesis did not produce a shippable improvement. The measured point
changes were below 1% and Criterion reported both shapes as within the noise
threshold. This means the source-level row pointer hoist is not a material
standalone optimization for the current SM89 B=1 path.

## Evidence

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp8_gemv_batch_b1 --save-baseline p3_1_a7_before
```

Treatment command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp8_gemv_batch_b1 --baseline p3_1_a7_before
```

| Shape | Baseline point | Treatment point | Criterion change | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| `dsv4_mini_hidden_1024x1024` | `9.7906 us` | `9.7634 us` | `-0.2550%` | `0.00 < 0.05` | KILL |
| `dsv4_mini_moe_512x1024` | `8.2294 us` | `8.2072 us` | `-0.4803%` | `0.00 < 0.05` | KILL |

Criterion labeled both changes "within noise threshold". The point estimates
do not meet the `>=3%` license gate or the `2-3%` review bucket.

## Fix

Reverted the A7 runtime change. Keep the existing source expression:

```cuda
weight[row * K + k]
```

## Tradeoff

- LOC complexity: negligible, but not worth carrying without evidence.
- SM89 specificity: measured locally on RTX 4070 Ti SUPER / SM89.
- Shared memory budget: unchanged.
- Register budget: treatment may add a live pointer register.
- CUDA Graph compatibility: unchanged.
- Generality across batch sizes: B=1 only; B>1 tiled path was not touched.
- Numerical correctness margin: pointer-equivalent change, but no correctness
  gate was required because the performance gate failed.

## Rule

Do not ship standalone row pointer hoisting for
`dsv4_fp8_gemv_batch_kernel` B=1. Any benefit is below measurement noise on
the current local shapes.
