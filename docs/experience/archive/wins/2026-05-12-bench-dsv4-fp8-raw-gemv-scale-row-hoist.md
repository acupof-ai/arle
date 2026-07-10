# DeepSeek V4 raw FP8 GEMV scale-row hoist - 2026-05-12

## Goal

- Test whether the scale-row hoist that helped FP4 also improves the raw
  DeepSeek V4 FP8 GEMV component kernel when applied as a separate variable.

## Hypothesis

- `dsv4_fp8_gemv_kernel` still called `dsv4_block_scale` for every `k`, which
  recomputes scale block geometry and the fixed scale row for the output row.
  Hoisting `block_h`, `block_w`, and `scale_row_offset` outside the inner loop
  should reduce integer work without changing native FP8 E4M3 decode,
  E8M0 scale decode, memory layout, reduction, or launch shape.

## Command

Component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  dsv4_fp8_gemv
```

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda test_dsv4_fp8_gemv -- --nocapture
```

## Environment

- Backend: CUDA
- Operator: `dsv4_fp8_gemv_cuda`
- Hardware: NVIDIA GeForce RTX 4070 Ti SUPER, SM89, 16376 MiB VRAM
- Driver / CUDA: 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- Feature set: `cargo bench -p infer --features cuda --bench ops_bench`
- Non-default flags / env vars: `CUDARC_CUDA_VERSION=13010`,
  `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `TORCH_CUDA_ARCH_LIST=8.9`

## Params

| Param | hidden shape | MoE shape |
|---|---:|---:|
| rows | 1024 | 512 |
| cols | 1024 | 1024 |
| scale_rows | 8 | 4 |
| scale_cols | 8 | 8 |
| scale block | 128x128 | 128x128 |
| input | BF16 row vector | BF16 row vector |
| weights | FP8 E4M3 bytes | FP8 E4M3 bytes |
| scales | FP8 E8M0 bytes, all `127` (=1.0) | FP8 E8M0 bytes, all `127` (=1.0) |

## Results - Component A/B

Only `dsv4_fp8_gemv_kernel` changed:

```cpp
// before: dsv4_block_scale(scales, row, k, N, K, scale_rows, scale_cols)
// after: hoist block_h/block_w/scale_row_offset outside the k loop
```

| Shape | Native-FP8 baseline | Scale-row hoist | Delta |
|---|---:|---:|---:|
| 1024x1024 | point `10.741 us` | `9.7688-9.7837 us`, point `9.7742 us` | `-9.00%` |
| 512x1024 | point `8.7933 us` | `8.2291-8.2435 us`, point `8.2376 us` | `-6.32%` |

Criterion's saved-baseline comparison also reported statistically significant
improvement:

| Shape | Criterion time change | p-value |
|---|---:|---:|
| 1024x1024 | `-9.0576% .. -8.8916%`, point `-8.9750%` | `0.00 < 0.05` |
| 512x1024 | `-6.4945% .. -6.2267%`, point `-6.3578%` | `0.00 < 0.05` |

Throughput:

| Shape | Native-FP8 baseline | Scale-row hoist | Delta |
|---|---:|---:|---:|
| 1024x1024 | `97.629 Gelem/s` | `107.28 Gelem/s` | `+9.89%` |
| 512x1024 | `59.625 Gelem/s` | `63.645 Gelem/s` | `+6.74%` |

## Results - Correctness

```text
test ops::tests::test_dsv4_fp8_gemv_preserves_finite_nan_pattern ... ok
test ops::tests::test_dsv4_fp8_gemv ... ok
```

The finite-pattern test preserves the project-specific DeepSeek FP8 E4M3
semantics for `0x7f/0xff` as `+/-448`, rather than CUDA's NaN treatment for
those byte patterns.

## Problems

- This is a component bench only. DeepSeek V4 CUDA serving is not yet a
  request-level performance target, so no Guidellm A/B is available.
- This patch intentionally touches only the non-batch FP8 GEMV kernel. The
  FP4 result is covered by the separate FP4 scale-row-hoist entry, and batch
  kernels still need their own A/B.
- The root-cause mechanism is still hypothesis-grade without an instruction
  profile. The evidence is the controlled component A/B, not proof of which
  integer operation dominated.

## Learnings

- Scale-row/block geometry hoisting is a stronger raw FP8 win than raw FP4 on
  the tested SM89 DSV4-mini shapes.
- Keep the conclusion scoped to raw component latency. End-to-end impact
  remains deferred until a runnable DeepSeek V4 CUDA serving path can provide
  wall-clock data.

## Delta vs Baseline

Baseline:
[`2026-05-12-bench-dsv4-fp8-raw-gemv-native-decode.md`](2026-05-12-bench-dsv4-fp8-raw-gemv-native-decode.md)

| metric | baseline | now | delta |
|---|---:|---:|---:|
| FP8 1024x1024 latency median | 10.741 us | 9.7742 us | -9.00% |
| FP8 512x1024 latency median | 8.7933 us | 8.2376 us | -6.32% |
