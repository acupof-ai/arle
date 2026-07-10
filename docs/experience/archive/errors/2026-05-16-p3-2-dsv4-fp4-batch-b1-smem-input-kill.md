# P3.2 DSv4 FP4 batch B1 smem input broadcast kill

## Context

Phase 3 P3.2 A3 tested shared-memory activation broadcast in
`dsv4_fp4_gemv_batch_kernel`, the B=1 raw path behind
`dsv4_fp4_gemv_batch_cuda`, after the A2 pair-load optimization landed.

## Formula Prediction

Hypothesis before edit:

- SM89 constants: 64K registers/SM, 100KB shared memory/SM, 1536 threads/SM,
  672 GB/s HBM.
- Workload constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`,
  `threads_per_row=64`, `K=1024`, `B=1`.
- After A2 pair-load, packed FP4 weight traffic is about
  `4 rows * 512B = 2KB` per CTA, while the same input vector is still reread
  by four row groups, about `4 rows * 1024 * 2B = 8KB` per CTA.
- Treatment stages the input vector once into dynamic shared memory, about
  `1024 * 2B = 2KB` per CTA, then all row groups read from shared memory.

Predicted point delta was -6% to -12% for `1024x1024` and -5% to -10% for
`512x1024`.

## Root Cause

The hypothesis was falsified. Even after reducing FP4 weight loads with A2,
the cooperative input copy plus `__syncthreads()` is more expensive than the
avoided B=1 input rereads. The input vector appears cache-friendly enough in
this component path that shared-memory broadcast is not the binding axis.

## Evidence

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --save-baseline p3_2_a3_before
```

Treatment command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --baseline p3_2_a3_before
```

| Shape | Baseline point | Treatment point | Criterion change | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| `dsv4_mini_hidden_1024x1024` | `9.5130 us` | `10.067 us` | `+5.7967%` | `0.00 < 0.05` | KILL |
| `dsv4_mini_moe_512x1024` | `8.0154 us` | `8.5757 us` | `+7.0367%` | `0.00 < 0.05` | KILL |

## Fix

Reverted the A3 runtime change. Keep direct global input reads in
`dsv4_fp4_gemv_batch_kernel`.

## Tradeoff

- LOC complexity: treatment added dynamic shared memory, a K-size threshold,
  cooperative copy, and a branch.
- SM89 specificity: measured locally on RTX 4070 Ti SUPER / SM89.
- Shared memory budget: +2KB per CTA for K=1024; fallback needed for larger K.
- Register budget: slightly worse due source pointer and branch state.
- CUDA Graph compatibility: dynamic shared memory depends on K but remains
  stable for fixed shapes.
- Generality across batch sizes: B=1 only; B>1 tiled path was not touched.
- Generality across shape: both hidden and MoE shapes regressed.
- Numerical correctness margin: intended source-equivalent input reuse, but no
  correctness gate was needed because performance failed.

## Rule

Do not stage the B=1 FP4 batch input vector into shared memory for
`dsv4_fp4_gemv_batch_kernel`. The copy and synchronization cost dominates the
avoided global input rereads on the current local shapes.
