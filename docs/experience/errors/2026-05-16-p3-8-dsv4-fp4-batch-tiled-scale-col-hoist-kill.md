# P3.8 DSv4 FP4 batch tiled scale-column hoist KILL

## Context

Phase 3 P3.8 A1 tested a scale-column hoist in
`dsv4_fp4_gemv_batch_tiled_kernel` at
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

The candidate mirrored the FP8 tiled A1 idea: compute scale row geometry once,
iterate by scale column, and reuse the decoded E8M0 scale for the k range.

## Root Cause

KILL. The hoist regressed both local FP4 tiled batch shapes. Unlike the FP8
tiled kernel, FP4 still processes one nibble per `k`; changing the loop nest
from monotonic `k += threads_per_row` to per-scale-column segments appears to
hurt scheduling/coalescing enough to dominate the removed `dsv4_block_scale`
integer math.

The root-cause mechanism above is a hypothesis. The decision does not depend
on it: matched Criterion A/B shows a clear regression.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp4_gemv_batch/ --save-baseline p3_8_a1_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp4_gemv_batch/ --baseline p3_8_a1_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv4_mini_hidden_1024x1024` | `19.202 us` | `21.523 us` | `+11.876%` | `0.00` | KILL |
| `dsv4_mini_moe_512x1024` | `13.787 us` | `14.867 us` | `+7.7440%` | `0.00` | KILL |

## Fix

The candidate patch was fully reverted before commit. No runtime code changed.

## Rule

Do not port the FP8 tiled scale-column hoist mechanically to FP4 tiled batch.
FP4 needs a pair/nibble-aware treatment with its own A/B evidence.
