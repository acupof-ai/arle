# P3.1 DSv4 FP8 batch B1 smem input broadcast kill

## Context

Phase 3 P3.1 A3 tested shared-memory activation broadcast in
`dsv4_fp8_gemv_batch_kernel`, the B=1 raw path behind
`dsv4_fp8_gemv_batch_cuda`.

## Formula Prediction

Hypothesis before edit:

- SM89 constants: 64K registers/SM, 100KB shared memory/SM, 1536 threads/SM,
  672 GB/s HBM.
- Workload constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`,
  `threads_per_row=64`, `K=1024`, `B=1`.
- Baseline has four row groups per CTA. Each row group reads the same B=1
  input vector, so approximate input HBM traffic is `4 * K * 2B = 8KB` per
  CTA. Weight traffic remains about `4 * K * 1B = 4KB` per CTA.
- Treatment cooperatively stages the input vector once into dynamic shared
  memory, reducing input HBM traffic to about `K * 2B = 2KB` per CTA. For
  K=1024, shared-memory footprint is 2KB per CTA, so occupancy should remain
  thread-limited rather than smem-limited.

Predicted point delta was -8% to -15% for `1024x1024` and -6% to -12% for
`512x1024`.

## Root Cause

The hypothesis was falsified. The input vector is cache-friendly enough on the
local SM89 B=1 path that cooperative copy plus `__syncthreads()` costs more
than the avoided row-group rereads. This means the standalone input broadcast
axis is not the current binding memory-access issue for this kernel.

## Evidence

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp8_gemv_batch_b1 --save-baseline p3_1_a3_before
```

Treatment command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp8_gemv_batch_b1 --baseline p3_1_a3_before
```

| Shape | Baseline point | Treatment point | Criterion change | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| `dsv4_mini_hidden_1024x1024` | `9.7797 us` | `10.160 us` | `+3.8190%` | `0.00 < 0.05` | KILL |
| `dsv4_mini_moe_512x1024` | `8.2308 us` | `8.5186 us` | `+3.4012%` | `0.00 < 0.05` | KILL |

## Fix

Reverted the A3 runtime change. Keep direct global input reads in
`dsv4_fp8_gemv_batch_kernel`.

## Tradeoff

- LOC complexity: treatment added dynamic shared memory, a launch-time smem
  threshold, cooperative copy, and an extra synchronization.
- SM89 specificity: no explicit SM-specific path, but the measured decision is
  SM89-local.
- Shared memory budget: +2KB per CTA for K=1024; larger K would reduce
  occupancy or need fallback.
- Register budget: small increase for input source selection.
- CUDA Graph compatibility: dynamic shared memory size depends on K but remains
  graph-capturable for fixed shapes.
- Generality across batch sizes: B=1 only; B>1 tiled path was not touched.
- Numerical correctness margin: intended exact source reuse, but correctness
  was not relevant because performance regressed.

## Rule

Do not stage the B=1 input vector into shared memory for
`dsv4_fp8_gemv_batch_kernel` on current SM89 shapes. The input rereads are not
expensive enough to pay for the copy and synchronization.
