# DSv4 FP4 Grouped Pair GEMV Pair-Load Win

## Context

Phase 3 P3.5 A2 optimized `dsv4_fp4_grouped_gemv_pair_batch_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

The microbench added in `c85ad3f` covers the local DSv4 grouped expert gate/up
shape with `N=512`, `K=1024`, four experts, and total routes 4 / 64:

- `dsv4_mini_t4_e4_512x1024`
- `dsv4_mini_t64_e4_512x1024`

## What Worked

The baseline mapped one logical K element to one thread-loop iteration. For
FP4, two adjacent K elements share one packed byte, so adjacent lanes loaded
the same byte from `weight_a` and `weight_b`.

The treatment maps one loop iteration to one packed byte and handles both
low/high nibbles. It keeps the same scale helper calls for `k0` and `k1`, so
this axis isolates packed weight load duplication from P3.5 A1 scale math.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_grouped_gemv_pair --save-baseline p3_5_a2_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_grouped_gemv_pair --baseline p3_5_a2_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| t4/e4/512x1024 | 25.857 us | 18.146 us | -29.823% | 0.00 | LICENSE |
| t64/e4/512x1024 | 299.05 us | 178.56 us | -40.286% | 0.00 | LICENSE |

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --lib --features cuda \
  test_dsv4_fp4_grouped_gemv_pair -- --nocapture
```

Result: `test ops::tests::test_dsv4_fp4_grouped_gemv_pair ... ok`.

## Tradeoffs

- LOC complexity: moderate; the loop now handles two K elements per iteration.
- SM89 specificity: measured locally on RTX 4070 Ti SUPER / SM89; the duplicate
  packed-byte pattern is not SM-specific.
- Shared memory budget: unchanged.
- Register budget: higher due two inputs, four scale values, and four decoded
  nibbles per loop iteration.
- CUDA Graph compatibility: unchanged; ABI and launch shape are stable.
- Generality across batch sizes: both route totals 4 and 64 pass the license
  gate.
- Generality across shape: measured on the DSv4 mini grouped expert shape
  `512x1024`; hidden-size rows are not a grouped gate/up target in this bench.
- Numerical correctness margin: accumulation order changed; direct FFI
  correctness against CPU reference passed with `0.01` tolerance.

## Rule

For DSv4 FP4 grouped pair GEMV, packed-byte pair-load is licensed on local SM89
when scale math is kept constant and both t4/t64 grouped shapes pass Criterion
plus direct FFI correctness. Treat this separately from the non-grouped B=1
FP4 pair-load; batch and grouped route geometry need their own evidence.
