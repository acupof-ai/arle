# P3.5 DSv4 FP4 Grouped Pair Smem Input KILL

## Context

Kernel: `dsv4_fp4_grouped_gemv_pair_batch_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

Scope: P3.5 A3 tested staging the activation vector into dynamic shared memory
once per CTA after the P3.5 A2 pair-load win (`44d4f9a`). Each CTA computes
four output rows for one route/expert, so the four row groups read the same
`x[0..K)`.

## Formula Prediction

Hypothesis before edit:

- Workload constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`, `K=1024`, FP4 pair
  loop, four rows per CTA.
- Baseline input traffic per CTA is roughly `4 rows * 1024 BF16 = 8 KiB`.
- Treatment input traffic per CTA is roughly `1024 BF16 = 2 KiB`, plus shared
  memory reads and one CTA-wide `__syncthreads()`.
- Dynamic shared memory use is `K * sizeof(bf16) = 2 KiB`, small versus SM89's
  100 KiB/SM budget.

Predicted point delta was 0-6% because L1 cache reuse could already cover most
of the repeated input loads.

## Root Cause

The hypothesis was falsified. Shared staging regressed both local shapes. The
activation vector appears cache-friendly enough after pair-load, and the extra
copy plus synchronization costs dominate the saved global input reads.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_grouped_gemv_pair --save-baseline p3_5_a3_before
```

Treatment staged `x[0..K)` into `extern __shared__` BF16 memory and launched
the grouped pair kernel with `K * sizeof(__nv_bfloat16)` dynamic shared bytes.

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| t4/e4/512x1024 | 18.064 us | 18.716 us | +3.5098% | 0.00 | KILL |
| t64/e4/512x1024 | 178.57 us | 180.87 us | +1.2837% | 0.00 | KILL |

## Fix

Treatment reverted. Keep direct global activation reads in the grouped FP4 pair
kernel.

## Rule

Do not stage DSv4 grouped FP4 pair GEMV activations into shared memory on the
local SM89 pair-load path. The CTA-wide copy/sync is more expensive than the
input-cache benefit on both t4 and t64 shapes.
