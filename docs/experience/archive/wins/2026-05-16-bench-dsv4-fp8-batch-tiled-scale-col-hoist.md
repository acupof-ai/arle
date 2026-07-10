# DSv4 FP8 Batch Tiled Scale-Column Hoist Win

## Context

Phase 3 P3.6 A1 optimized `dsv4_fp8_gemv_batch_tiled_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

The existing local Criterion bench covers the B=4 tiled path behind
`dsv4_fp8_gemv_batch_cuda`:

- `dsv4_mini_hidden_1024x1024`
- `dsv4_mini_moe_512x1024`

## What Worked

The baseline computed `k / block_w` and decoded the E8M0 scale inside the
per-`k` loop. The tiled FP8 kernel already reuses each decoded weight across the
batch tile, so this treatment only moved scale-column selection outside the
inner K loop:

- precompute `weight_row` and `scale_row`
- loop over scale columns
- decode one E8M0 scale per scale column
- keep FP8 weight decode and batch-tile accumulation unchanged

This isolates scale-column math from launch shape, shared memory, FP8 decode,
and accumulation-order changes.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp8_gemv_batch/ --save-baseline p3_6_a1_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp8_gemv_batch/ --baseline p3_6_a1_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv4_mini_hidden_1024x1024` | `22.612 us` | `21.651 us` | `-4.3105%` | `0.00` | LICENSE |
| `dsv4_mini_moe_512x1024` | `20.291 us` | `15.593 us` | `-23.144%` | `0.00` | LICENSE |

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --lib --features cuda \
  test_dsv4_fp8_batched_gemv -- --nocapture
```

Result: `test ops::tests::test_dsv4_fp8_batched_gemv ... ok`.

## Tradeoffs

- LOC complexity: moderate; the hot loop now has an outer scale-column loop.
- SM89 specificity: measured locally on RTX 4070 Ti SUPER / SM89.
- Shared memory budget: unchanged.
- Register budget: slightly higher due `weight_row`, `scale_row`, loop bounds,
  and a longer-lived decoded scale.
- CUDA Graph compatibility: unchanged; ABI and launch shape are stable.
- Generality across batch sizes: only the existing B=4 tiled path was measured;
  B=1 raw path had a separate P3.1 A1 KILL and remains untouched.
- Numerical correctness margin: accumulation order across K is grouped by scale
  column, but K order within each scale column is unchanged and the direct
  batched FP8 correctness test passed.

## Rule

For `dsv4_fp8_gemv_batch_tiled_kernel` on local SM89 B=4 shapes, scale-column
loop hoisting is licensed when both hidden and MoE shapes pass the 3% Criterion
gate and `test_dsv4_fp8_batched_gemv` passes. Do not apply this conclusion to
the B=1 raw FP8 path; P3.1 measured that as a regression.
