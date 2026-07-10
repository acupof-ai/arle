# P3.2 DSv4 FP4 Batch B1 Row Pointer Hoist KILL

## Context

Kernel: `dsv4_fp4_gemv_batch_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

Scope: P3.2 A7 tested whether hoisting the per-row weight base pointer from
`weight[row * bytes_per_row + pair]` to `row_weight[pair]` improves the local
RTX 4070 Ti SUPER DSv4 FP4 B=1 batch GEMV path after the A2 pair-load win.

## Root Cause

Hypothesis: the inner-loop row offset multiply may still cost enough to matter
after pair-load removed duplicate packed-byte loads.

Evidence did not support the hypothesis. Matched Criterion baseline/treatment
using:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --baseline p3_2_a7_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| hidden 1024x1024 | 9.5435 us | 9.5567 us | +0.1854% | 0.06 | KILL |
| MoE 512x1024 | 7.9763 us | 7.9812 us | +0.1238% | 0.32 | KILL |

Criterion reported "No change in performance detected" for both shapes.

## Fix

Treatment reverted. Keep the existing indexed load:
`weight[row * bytes_per_row + pair]`.

## Rule

For this kernel, row pointer hoisting is below the >=3% license gate and lacks
statistical significance. Do not reintroduce it unless a different shape family
shows fresh matched evidence.
