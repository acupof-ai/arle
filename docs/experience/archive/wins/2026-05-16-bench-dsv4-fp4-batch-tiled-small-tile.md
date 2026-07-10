# DSv4 FP4 Batch Tiled Small-Tile Win

## Context

Phase 3 P3.8 A3 optimized `dsv4_fp4_gemv_batch_tiled_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

A2 pair-load from `5392c4f` was the baseline. The local benchmark uses B=4,
but the FP4 tiled kernel still executed the fixed `DSV4_BATCH_TILE=16`
accumulator, reduction, and shared-memory write path.

## What Worked

The treatment adds a `tile_batches <= 4` fast path mirroring the already
licensed FP8 tiled shape:

- use `float sums4[4]` instead of `float sums[16]`
- keep the FP4 packed-byte pair load from A2
- reduce/write four active slots through a smaller shared-memory tile
- return before the original 16-slot fallback

The B>4 path keeps the existing 16-slot implementation.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp4_gemv_batch/ --save-baseline p3_8_a3_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp4_gemv_batch/ --baseline p3_8_a3_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv4_mini_hidden_1024x1024` | `18.515 us` | `11.528 us` | `-37.751%` | `0.00` | LICENSE |
| `dsv4_mini_moe_512x1024` | `12.635 us` | `8.7732 us` | `-30.630%` | `0.00` | LICENSE |

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

The batched FP4 test now covers B=2 for the small-tile fast path and B=5 for
the original 16-slot fallback.

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

- License strength: both local SM89 B=4 FP4 tiled shapes cross the 3% gate by
  a wide margin.
- Scope: this changes only the B>1 tiled FP4 kernel; B=1 raw and grouped
  expert kernels are unchanged.
- Code shape: the kernel now has a small-tile fast path plus the original
  16-slot fallback, matching the FP8 tiled precedent.
- Shared memory: static shared-memory declarations now include both the small
  tile and fallback tile, as in the FP8 implementation.
- CUDA Graph compatibility: unchanged; ABI and launch shape are stable.

## Rule

For `dsv4_fp4_gemv_batch_tiled_kernel`, do not run B<=4 through the full
16-slot tile. The four-slot fast path is licensed by local SM89 Criterion and
correctness evidence.
