# DeepSeek V4 raw FP4 GEMV scale-row hoist - 2026-05-12

## Goal

- Test whether explicitly hoisting FP4 GEMV block-scale row math out of the
  inner loop improves the raw DeepSeek V4 FP4 component kernel.

## Hypothesis

- `dsv4_block_scale` computes scale block geometry and scale row selection for
  every `k` in the GEMV loop. For a fixed output row, `block_h`, `block_w`, and
  the selected scale row are loop-invariant. Hoisting those values in
  `dsv4_fp4_gemv_kernel` should reduce integer work without changing FP4
  packing, FP4 decode, E8M0 scale decode, reduction, or launch shape.

## Command

Component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  dsv4_fp4_gemv
```

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda test_dsv4_fp4_gemv -- --nocapture
```

## Environment

- Backend: CUDA
- Operator: `dsv4_fp4_gemv_cuda`
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
| weights | packed FP4 E2M1 bytes | packed FP4 E2M1 bytes |
| scales | FP8 E8M0 bytes, all `127` (=1.0) | FP8 E8M0 bytes, all `127` (=1.0) |

## Results - Component A/B

Only `dsv4_fp4_gemv_kernel` changed:

```cpp
// before: dsv4_block_scale(scales, row, k, N, K, scale_rows, scale_cols)
// after: hoist block_h/block_w/scale_row_offset outside the k loop
```

| Shape | FP4 LUT baseline | Scale-row hoist | Delta |
|---|---:|---:|---:|
| 1024x1024 | point `11.610 us` | `11.346-11.365 us`, point `11.354 us` | `-2.20%` |
| 512x1024 | point `9.1251 us` | `8.9925-9.0097 us`, point `8.9996 us` | `-1.38%` |

Criterion's saved-baseline comparison also reported statistically significant
improvement:

| Shape | Criterion time change | p-value |
|---|---:|---:|
| 1024x1024 | `-3.2206% .. -2.0242%`, point `-2.4786%` | `0.00 < 0.05` |
| 512x1024 | `-1.5632% .. -1.2673%`, point `-1.4182%` | `0.00 < 0.05` |

Throughput:

| Shape | FP4 LUT baseline | Scale-row hoist | Delta |
|---|---:|---:|---:|
| 1024x1024 | `90.314 Gelem/s` | `92.352 Gelem/s` | `+2.26%` |
| 512x1024 | `57.455 Gelem/s` | `58.257 Gelem/s` | `+1.40%` |

## Results - Correctness

```text
test ops::tests::test_dsv4_fp4_gemv ... ok
```

## Problems

- This is a component bench only. DeepSeek V4 CUDA serving is not yet a
  request-level performance target, so no Guidellm A/B is available.
- This patch intentionally touches only the non-batch FP4 GEMV kernel. The
  FP8 and batch kernels still use `dsv4_block_scale`; applying the same idea
  there requires separate A/B runs.
- The root-cause mechanism is still hypothesis-grade without an instruction
  profile. The evidence is the controlled component A/B, not proof of which
  integer operation dominated.

## Learnings

- After the FP4 LUT decode win, explicitly hoisting scale-row/block geometry
  still gives a smaller but stable raw FP4 GEMV improvement on SM89.
- Keep this as a local component win only. End-to-end impact remains deferred
  until a runnable DeepSeek V4 CUDA serving path can provide wall-clock data.

## Delta vs Baseline

Baseline:
[`2026-05-12-bench-dsv4-fp4-raw-gemv-lut-decode.md`](2026-05-12-bench-dsv4-fp4-raw-gemv-lut-decode.md)

| metric | baseline | now | delta |
|---|---:|---:|---:|
| FP4 1024x1024 latency median | 11.610 us | 11.354 us | -2.20% |
| FP4 512x1024 latency median | 9.1251 us | 8.9996 us | -1.38% |
