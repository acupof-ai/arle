# DSv4 FP4 Batch Tiled Pair-Load Win

## Context

Phase 3 P3.8 A2 optimized `dsv4_fp4_gemv_batch_tiled_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

The previous A1 attempt to hoist scale columns was killed because it regressed
both local FP4 tiled batch shapes. This treatment keeps the original row/block
schedule and only changes the inner FP4 load granularity.

## What Worked

The tiled FP4 batch path now iterates over packed weight bytes instead of one
nibble at a time. Each thread loads one packed byte, decodes both FP4 values,
loads both corresponding BF16 activations, and accumulates the pair for each
active batch slot.

Launch shape, ABI, output layout, per-row reduction, and batch tiling stay
unchanged.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp4_gemv_batch/ --save-baseline p3_8_a2_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp4_gemv_batch/ --baseline p3_8_a2_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv4_mini_hidden_1024x1024` | `19.212 us` | `18.536 us` | `-3.5192%` | `0.00` | LICENSE |
| `dsv4_mini_moe_512x1024` | `13.838 us` | `12.588 us` | `-9.0276%` | `0.00` | LICENSE |

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --lib --features cuda \
  test_dsv4_fp4_batched_gemv -- --nocapture
```

Result:
`test_dsv4_fp4_batched_gemv_b1_raw ... ok` and
`test_dsv4_fp4_batched_gemv ... ok`.

Compile gate:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo check -p infer --features cuda
```

Result: PASS with pre-existing DSv4 warnings.

## Tradeoffs

- License strength: both local SM89 B=4 FP4 tiled shapes cross the 3% gate.
- Scope: this changes only the B>1 tiled FP4 kernel; the B=1 raw FP4 kernel
  and grouped expert pair kernel are unchanged.
- Numerical behavior: accumulation order now pairs adjacent FP4 nibbles in a
  single loop iteration. The existing batched FP4 correctness test still
  passes against the CPU reference.
- CUDA Graph compatibility: unchanged; ABI and launch shape are stable.

## Rule

For `dsv4_fp4_gemv_batch_tiled_kernel`, process packed FP4 bytes as pairs.
The pair-load treatment is licensed locally for both DSv4 mini hidden and MoE
B=4 tiled batch GEMV shapes.
