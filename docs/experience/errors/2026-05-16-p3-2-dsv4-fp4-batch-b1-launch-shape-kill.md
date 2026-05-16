# P3.2 DSv4 FP4 batch B1 launch shape kill

## Context

Phase 3 P3.2 A5 tested launch-shape changes for
`dsv4_fp4_gemv_batch_kernel`, the B=1 raw path behind
`dsv4_fp4_gemv_batch_cuda`, after A2 pair-load landed.

## Formula Prediction

Hypothesis before edit:

- SM89 constants: 64K registers/SM, 100KB shared memory/SM, 1536 threads/SM,
  672 GB/s HBM.
- Baseline constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`,
  `threads_per_row=64`.
- Candidate grid: `256x8`, `512x4`, and `512x8`. Shapes with
  `threads_per_row < 32` were excluded because the existing full-warp
  reduction would mix row groups.
- `256x8` reduces row-grid CTAs while keeping eight warps per CTA.
- `512x4` doubles per-row lanes to 128 and uses 16 warps per CTA.
- `512x8` combines fewer CTAs with 16-warps CTAs.

Predicted result was uncertain after A2 because pair-load changes the balance
between per-row work and scheduling pressure. License still required no
regressing shape and `>=3%` point improvement.

## Root Cause

The hypothesis was falsified. The current `256x4` launch shape remains the
best measured compromise for the local B=1 FP4 pair-load path. Larger row
grouping or larger CTAs either regress one shape or remain below the review
bucket.

## Evidence

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --save-baseline p3_2_a5_before
```

Treatment command for each launch shape:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --baseline p3_2_a5_before
```

| Variant | Shape | Baseline point | Treatment point | Criterion change | p-value | Verdict |
|---|---|---:|---:|---:|---:|---|
| `256x8` | `dsv4_mini_hidden_1024x1024` | `9.5250 us` | `9.3492 us` | `-1.7069%` | `0.00 < 0.05` | KILL |
| `256x8` | `dsv4_mini_moe_512x1024` | `7.9774 us` | `8.4306 us` | `+5.5820%` | `0.00 < 0.05` | KILL |
| `512x4` | `dsv4_mini_hidden_1024x1024` | `9.5250 us` | `9.9075 us` | `+4.0821%` | `0.00 < 0.05` | KILL |
| `512x4` | `dsv4_mini_moe_512x1024` | `7.9774 us` | `8.2470 us` | `+3.3961%` | `0.00 < 0.05` | KILL |
| `512x8` | `dsv4_mini_hidden_1024x1024` | `9.5250 us` | `9.6634 us` | `+1.7313%` | `0.00 < 0.05` | KILL |
| `512x8` | `dsv4_mini_moe_512x1024` | `7.9774 us` | `8.0790 us` | `+1.2116%` | `0.00 < 0.05` | KILL |

`256x8` improved the hidden shape but regressed MoE. Both 512-thread variants
regressed both shapes.

## Fix

Reverted all launch-shape runtime changes. Keep the existing constants:

```cuda
#define GEMV_THREADS 256
#define GEMV_ROWS 4
```

## Tradeoff

- LOC complexity: a licensed version would need scoped P3.2 constants rather
  than global GEMV macro changes shared by sibling kernels.
- SM89 specificity: measured locally on RTX 4070 Ti SUPER / SM89.
- Shared memory budget: unchanged.
- Register budget: unchanged per thread, but CTA shape changes occupancy and
  scheduling pressure.
- CUDA Graph compatibility: unchanged for fixed shapes.
- Generality across batch sizes: B=1 only; B>1 tiled path was not touched.
- Generality across shape: every non-baseline variant regressed at least one
  shape.
- Numerical correctness margin: shape-only changes should preserve the same
  mathematical result within reduction-order tolerance, but no correctness gate
  was needed because the performance gate failed.

## Rule

Do not change `dsv4_fp4_gemv_batch_kernel` B=1 launch shape away from
`256x4` on current SM89 pair-load shapes.
