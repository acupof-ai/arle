# DSv4 FP8 Batch Tiled Small-Tile Win

## Context

Phase 3 P3.6 A7 optimized `dsv4_fp8_gemv_batch_tiled_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

A1 scale-column hoist from `a47f723` was the baseline. The local bench uses
B=4, but the tiled kernel used a fixed `DSV4_BATCH_TILE=16` path. That meant
the B=4 case still executed 16-slot accumulation, warp reduction, and shared
memory write loops, with 12 inactive slots guarded by branches.

## What Worked

The treatment adds a `tile_batches <= 4` fast path:

- use `float sums4[4]` instead of `float sums[16]`
- accumulate only four batch slots
- reduce/write four slots into `smem4`
- return before the original 16-slot fallback

The B>4 path keeps the existing 16-slot implementation. The unit test now runs
both B=2 and B=5 cases so the small fast path and fallback path are both
covered.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp8_gemv_batch/ --save-baseline p3_6_a7_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp8_gemv_batch/ --baseline p3_6_a7_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv4_mini_hidden_1024x1024` | `21.631 us` | `11.847 us` | `-45.274%` | `0.00` | LICENSE |
| `dsv4_mini_moe_512x1024` | `15.592 us` | `11.326 us` | `-27.325%` | `0.00` | LICENSE |

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

- LOC complexity: moderate; the kernel now has a small-tile fast path plus the
  original 16-slot fallback.
- SM89 specificity: measured locally on RTX 4070 Ti SUPER / SM89.
- Shared memory budget: static shared memory increases because both `smem4` and
  the fallback `smem` are present in the same compiled kernel.
- Register budget: lower in the fast path due four accumulators instead of
  sixteen; fallback is intended to remain equivalent.
- CUDA Graph compatibility: unchanged; ABI and launch shape are stable.
- Generality across batch sizes: B=4 benchmark licenses the fast path; unit
  coverage includes B=2 fast path and B=5 fallback. Larger B performance was
  not benchmarked in this tranche.
- Numerical correctness margin: accumulation order for B<=4 is unchanged per
  slot; fewer inactive zero slots are reduced.

## Rule

For `dsv4_fp8_gemv_batch_tiled_kernel`, do not force small batches through the
full 16-slot tile. A dedicated `tile_batches <= 4` path is licensed by local
SM89 Criterion and correctness evidence.
