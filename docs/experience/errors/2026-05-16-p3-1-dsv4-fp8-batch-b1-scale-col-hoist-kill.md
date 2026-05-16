# P3.1 DSv4 FP8 batch B1 scale-col hoist kill

## Context

Phase 3 P3.1 A1 tested `dsv4_fp8_gemv_batch_kernel`, the B=1 raw path behind
`dsv4_fp8_gemv_batch_cuda`. The active B>1 path dispatches to
`dsv4_fp8_gemv_batch_tiled_kernel`, so `2e849bd` first added the dedicated
`ops_cuda/dsv4_fp8_gemv_batch_b1` Criterion case.

## Formula Prediction

Hypothesis before edit:

- SM89 constants: 64K registers/SM, 100KB shared memory/SM, 1536 threads/SM,
  672 GB/s HBM.
- Workload constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`,
  `threads_per_row=64`, `K=1024`, `scale_cols=8`, `block_w=128`, `B=1`.
- Baseline inner loop performs one `k / block_w` and one E8M0 scale decode per
  visited `k`. Per row, that is 1024 scale-column divisions and 1024 scale
  decodes.
- Treatment loops over scale columns and decodes one scale per participating
  thread per scale column. For this shape, that is up to `64 * 8 = 512` scale
  decodes per row and no per-`k` scale-column division.

Predicted point delta was -5% to -10% for `1024x1024` and -4% to -9% for
`512x1024`, assuming integer division and scale decode were material in the
inner loop.

## Root Cause

The hypothesis was falsified. The nested scale-column loop adds control-flow
and loop-bound overhead that outweighs the removed division/scale-decode work
on the local SM89 B=1 raw path. The scale bytes are tiny and likely already
cache-friendly; that mechanism is a hypothesis, but the regression is measured.

## Evidence

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp8_gemv_batch_b1 --save-baseline p3_1_a1_before
```

Treatment command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp8_gemv_batch_b1 --baseline p3_1_a1_before
```

| Shape | Baseline point | Treatment point | Criterion change | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| `dsv4_mini_hidden_1024x1024` | `9.7787 us` | `9.8825 us` | `+0.9738%` | `0.00 < 0.05` | KILL |
| `dsv4_mini_moe_512x1024` | `8.3019 us` | `9.2298 us` | `+11.275%` | `0.00 < 0.05` | KILL |

## Fix

Reverted the A1 runtime change. Keep the original per-`k` scale-column lookup
in `dsv4_fp8_gemv_batch_kernel`.

## Tradeoff

- LOC complexity: treatment added a nested scale-column loop and branch.
- SM89 specificity: no explicit SM-specific code, but the measured decision is
  SM89-local.
- Shared memory budget: unchanged.
- Register budget: likely slightly worse due extra loop bounds and `scale`
  lifetime.
- CUDA Graph compatibility: unchanged.
- Generality across batch sizes: B=1 only; B>1 tiled path was not touched.
- Numerical correctness margin: unchanged in intent, but correctness was not
  relevant because performance regressed.

## Rule

Do not apply scale-column hoist to `dsv4_fp8_gemv_batch_kernel` B=1 on the
current SM89 shapes. A small, cache-friendly scale tensor plus extra nested-loop
control is a measured regression, not a cleanup.
