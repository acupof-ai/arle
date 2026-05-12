# DeepSeek V4 batch FP8 GEMV scale-row hoist - 2026-05-12

## Goal

- Test whether the single-token FP8 scale-row hoist also improves the raw
  DeepSeek V4 FP8 batch GEMV component kernel as a separate control variable.

## Hypothesis

- `dsv4_fp8_gemv_batch_kernel` still called `dsv4_block_scale` for every
  `(batch,row,k)` element. For a fixed output row, `block_h`, `block_w`, and
  `scale_row_offset` are invariant across the inner `k` loop and all batch
  rows. Hoisting those values should reduce integer work without changing FP8
  decode, E8M0 scale decode, batch layout, reduction, or launch shape.

## Command

Component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  dsv4_fp8_gemv_batch
```

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda \
  test_dsv4_fp8_batched_gemv -- --nocapture
```

## Environment

- Backend: CUDA
- Operator: `dsv4_fp8_gemv_batch_cuda`
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
| batch | 4 | 4 |
| rows | 1024 | 512 |
| cols | 1024 | 1024 |
| scale_rows | 8 | 4 |
| scale_cols | 8 | 8 |
| scale block | 128x128 | 128x128 |
| input | BF16 `[batch, cols]` | BF16 `[batch, cols]` |
| weights | FP8 E4M3 bytes | FP8 E4M3 bytes |
| scales | FP8 E8M0 bytes, all `127` (=1.0) | FP8 E8M0 bytes, all `127` (=1.0) |

## Results - Component A/B

Only `dsv4_fp8_gemv_batch_kernel` changed:

```cpp
// before: dsv4_block_scale(scales, row, k, N, K, scale_rows, scale_cols)
// after: hoist block_h/block_w/scale_row_offset outside the k loop
```

| Shape | Batch bench baseline | Scale-row hoist | Delta |
|---|---:|---:|---:|
| b4 1024x1024 | point `23.394 us` | `19.507-19.513 us`, point `19.510 us` | `-16.60%` |
| b4 512x1024 | point `14.902 us` | `13.034-13.059 us`, point `13.042 us` | `-12.48%` |

Criterion's saved-baseline comparison also reported statistically significant
improvement:

| Shape | Criterion time change | p-value |
|---|---:|---:|
| b4 1024x1024 | `-16.631% .. -16.574%`, point `-16.603%` | `0.00 < 0.05` |
| b4 512x1024 | `-12.521% .. -12.308%`, point `-12.422%` | `0.00 < 0.05` |

Throughput:

| Shape | Batch bench baseline | Scale-row hoist | Delta |
|---|---:|---:|---:|
| b4 1024x1024 | `179.29 Gelem/s` | `214.98 Gelem/s` | `+19.91%` |
| b4 512x1024 | `140.73 Gelem/s` | `160.80 Gelem/s` | `+14.26%` |

## Results - Correctness

```text
test ops::tests::test_dsv4_fp8_batched_gemv ... ok
```

## Problems

- This is a component bench only. DeepSeek V4 CUDA serving is not yet a
  request-level performance target, so no Guidellm A/B is available.
- This patch intentionally touches only the FP8 batch kernel. FP4 batch uses a
  separate A/B and should not be inferred from this result.
- The root-cause mechanism is still hypothesis-grade without an instruction
  profile. The decision evidence is the controlled component A/B.

## Learnings

- Scale-row/block hoisting is a larger win in the FP8 batch kernel than in the
  single-token FP8 kernel on the tested SM89 DSV4-mini shapes.
- Batch DSV4 kernels are worth keeping in the raw operator bench harness;
  single-token data alone understated the scale-lookup opportunity.

## Delta vs Baseline

Baseline:
`65e0c3d bench(cuda): add dsv4 raw batch gemv cases`

| metric | baseline | now | delta |
|---|---:|---:|---:|
| FP8 batch b4 1024x1024 latency median | 23.394 us | 19.510 us | -16.60% |
| FP8 batch b4 512x1024 latency median | 14.902 us | 13.042 us | -12.48% |
