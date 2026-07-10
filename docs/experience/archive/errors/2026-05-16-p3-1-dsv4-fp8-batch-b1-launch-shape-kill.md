# P3.1 DSv4 FP8 batch B1 launch shape kill

## Context

Phase 3 P3.1 A5 tested launch-shape changes for
`dsv4_fp8_gemv_batch_kernel`, the B=1 raw path behind
`dsv4_fp8_gemv_batch_cuda`.

## Formula Prediction

Hypothesis before edit:

- SM89 constants: 64K registers/SM, 100KB shared memory/SM, 1536 threads/SM,
  672 GB/s HBM.
- Baseline constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`,
  `threads_per_row=64`. This is 8 warps per CTA, four output rows per CTA,
  and about 48 resident warps/SM if thread-limited.
- Candidate grid: `256x8`, `512x4`, and `512x8`. Shapes with
  `threads_per_row < 32` were excluded because the existing full-warp
  reduction would mix row groups.
- `256x8` halves row-grid CTAs but keeps eight warps per CTA. Expected win:
  lower launch/grid overhead and better row grouping. Risk: less per-row CTA
  scheduling flexibility.
- `512x4` doubles per-row lanes to 128 and 16 warps per CTA. Expected win:
  fewer loop iterations per thread. Risk: lower CTA residency and extra idle
  lanes in warp-level reductions.
- `512x8` combines fewer CTAs with 16-warps CTAs. Expected mixed result:
  hidden shape may benefit from larger work per CTA; MoE shape may be
  occupancy- or scheduling-limited.

Predicted point delta was -2% to -6% for at least one shape, but the license
gate required no regressing shape and `>=3%` point improvement.

## Root Cause

The hypothesis was falsified. Larger row grouping and larger CTAs do not
produce a shippable local component win for this SM89 B=1 raw path. The
512-thread variants sometimes help one shape, but every tested non-baseline
shape either regressed a shape or stayed below the license threshold. The
current 256-thread, four-row launch shape remains the best measured compromise
for the local hidden and MoE shapes.

## Evidence

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp8_gemv_batch_b1 --save-baseline p3_1_a5_before
```

Treatment command for each launch shape:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp8_gemv_batch_b1 --baseline p3_1_a5_before
```

| Variant | Shape | Baseline point | Treatment point | Criterion change | p-value | Verdict |
|---|---|---:|---:|---:|---:|---|
| `256x8` | `dsv4_mini_hidden_1024x1024` | `9.7707 us` | `9.9040 us` | `+1.3511%` | `0.00 < 0.05` | KILL |
| `256x8` | `dsv4_mini_moe_512x1024` | `8.2506 us` | `9.5318 us` | `+15.290%` | `0.00 < 0.05` | KILL |
| `512x4` | `dsv4_mini_hidden_1024x1024` | `9.7707 us` | `10.036 us` | `+2.7624%` | `0.00 < 0.05` | KILL |
| `512x4` | `dsv4_mini_moe_512x1024` | `8.2506 us` | `8.1314 us` | `-1.5124%` | `0.00 < 0.05` | KILL |
| `512x8` | `dsv4_mini_hidden_1024x1024` | `9.7707 us` | `9.6418 us` | `-1.1839%` | `0.00 < 0.05` | KILL |
| `512x8` | `dsv4_mini_moe_512x1024` | `8.2506 us` | `8.4930 us` | `+2.7878%` | `0.00 < 0.05` | KILL |

`512x8` improved the hidden shape, but the point estimate was below the
2% review bucket and the MoE shape regressed. `512x4` improved only the MoE
shape and regressed hidden. `256x8` regressed both shapes.

## Fix

Reverted all launch-shape runtime changes. Keep the existing constants:

```cuda
#define GEMV_THREADS 256
#define GEMV_ROWS 4
```

## Tradeoff

- LOC complexity: a licensed version would need scoped P3.1 constants rather
  than changing global GEMV macros shared by sibling kernels.
- SM89 specificity: decision is local to RTX 4070 Ti SUPER / SM89.
- Shared memory budget: unchanged.
- Register budget: unchanged per thread, but 512-thread CTAs change occupancy
  and scheduling pressure.
- CUDA Graph compatibility: unchanged for fixed shapes.
- Generality across batch sizes: B=1 only; B>1 tiled path was not touched.
- Numerical correctness margin: shape-only change should preserve arithmetic
  order per row group enough to require a correctness gate if licensed, but no
  correctness validation was needed because all variants were killed.

## Rule

Do not change `dsv4_fp8_gemv_batch_kernel` B=1 launch shape away from
`256x4` on current SM89 shapes. Mixed wins are below threshold and fail the
worst-shape gate.
