# P3.6 DSv4 FP8 batch tiled pair-load kill

## Context

Phase 3 P3.6 A2 tested `uint16_t` pair-loads for
`dsv4_fp8_gemv_batch_tiled_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

A1 scale-column hoist from `a47f723` was the baseline. The only A2 treatment
changed the inner K loop so each thread processed two adjacent FP8 weight bytes
per iteration. Scale-column loop structure, batch tile, launch shape, and
shared-memory reduction were unchanged.

## Formula Prediction

Hypothesis: each scale-column block has 128 K elements on the measured shapes.
Pair-loading FP8 weights lets 64 row threads cover the block in one iteration
instead of two, which may reduce loop overhead and global weight load
instructions.

Risk: the baseline already has coalesced byte loads across adjacent lanes, and
pair-loading increases per-thread work and register pressure while reducing the
amount of lane-level parallelism within each scale column.

## Root Cause

The hypothesis was falsified. Hidden improved by less than 1%, and MoE had no
statistically significant movement. The small local gain is below the Phase 3
review threshold and far below the 3% license threshold.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp8_gemv_batch/ --save-baseline p3_6_a2_before
```

Baseline results:

| Shape | Time |
|---|---:|
| `dsv4_mini_hidden_1024x1024` | `21.684 us` |
| `dsv4_mini_moe_512x1024` | `15.592 us` |

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp8_gemv_batch/ --baseline p3_6_a2_before
```

Treatment results:

| Shape | Time | Change | p-value | Decision |
|---|---:|---:|---:|---|
| `dsv4_mini_hidden_1024x1024` | `21.578 us` | `-0.5938%` | `0.00` | KILL: below review threshold |
| `dsv4_mini_moe_512x1024` | `15.563 us` | `-0.1434%` | `0.22` | KILL: no significant change |

## Fix

The treatment was reverted. Keep the A1 scalar FP8 weight load inside the
scale-column loop.

## Rule

Do not pair-load FP8 weights in `dsv4_fp8_gemv_batch_tiled_kernel` on local
SM89. The measured B=4 shapes do not license the extra per-thread work or
register pressure.
