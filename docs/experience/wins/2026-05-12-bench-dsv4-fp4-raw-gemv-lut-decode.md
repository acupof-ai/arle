# DeepSeek V4 raw FP4 GEMV LUT decode - 2026-05-12

## Goal

- Continue the DeepSeek V4 raw FP8/FP4 GEMV pass after the FP8 native-decode
  win by optimizing the FP4 E2M1 weight decode helper without changing packed
  layout, block-scale lookup, reduction, or launch shape.

## Hypothesis

- `dsv4_decode_fp4_e2m1` previously decoded each 4-bit value with sign/exp/mant
  extraction plus `exp2f`. FP4 E2M1 has only 16 values, so a device constant
  LUT should remove per-weight transcendental work and preserve exact decode
  semantics.

## Command

Component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  dsv4_fp4_gemv --quiet
```

FP8 no-regression check:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  dsv4_fp8_gemv --quiet
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
- Operator: `dsv4_fp4_gemv_cuda` / `dsv4_fp4_gemv_batch_cuda`
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

Only `dsv4_decode_fp4_e2m1` changed:

```cpp
// before: sign/exp/mant decode with exp2f
// after: DSV4_FP4_E2M1_LUT[bits & 0x0f]
```

| Shape | Post-FP8 baseline | LUT candidate | Delta |
|---|---:|---:|---:|
| 1024x1024 | `13.371-13.392 us`, point `13.380 us` | `11.594-11.632 us`, point `11.610 us` | `-13.23%` |
| 512x1024 | `10.397-10.419 us`, point `10.407 us` | `9.1189-9.1350 us`, point `9.1251 us` | `-12.32%` |

Throughput:

| Shape | Post-FP8 baseline | LUT candidate | Delta |
|---|---:|---:|---:|
| 1024x1024 | `78.370 Gelem/s` | `90.314 Gelem/s` | `+15.24%` |
| 512x1024 | `50.378 Gelem/s` | `57.455 Gelem/s` | `+14.05%` |

## Results - FP8 No-Regression Check

| Shape | Native-FP8 before LUT | After FP4 LUT | Delta |
|---|---:|---:|---:|
| 1024x1024 | `10.778 us` | `10.741 us` | `-0.34%` |
| 512x1024 | `8.8517 us` | `8.7933 us` | `-0.66%` |

## Results - Correctness

```text
test ops::tests::test_dsv4_fp4_gemv ... ok
```

The existing GPU test exercises packed FP4 bytes `0x21` and `0xb3` with the
project convention `even col -> low nibble`, giving `[5.0, -3.0]` for the
two-row dot product.

## Problems

- This is a component bench only. DeepSeek V4 CUDA serving is not yet a
  request-level performance target, so no Guidellm A/B is available.
- The block-scale division/clamp path is unchanged. Any future claim about
  scale-hoisting requires a separate control experiment.
- The LUT is in device constant memory. It wins on SM89 for these synthetic
  DSV4-mini shapes; larger-checkpoint shapes should still be re-benched once
  the full DSV4 CUDA path is runnable.

## Learnings

- FP4 E2M1 decode is small enough that a 16-entry LUT is faster than
  reconstructing the value with `exp2f` in the GEMV inner loop.
- After the FP8 native-decode and FP4 LUT wins, the remaining visible DSV4 raw
  GEMV decode overhead is more likely in block-scale lookup/reuse or memory
  layout, but that remains hypothesis-grade until measured.

## Delta vs Baseline

| metric | baseline | now | delta |
|---|---:|---:|---:|
| FP4 1024x1024 latency median | 13.380 us | 11.610 us | -13.23% |
| FP4 512x1024 latency median | 10.407 us | 9.1251 us | -12.32% |
| FP8 1024x1024 latency median | 10.778 us | 10.741 us | -0.34% |
| FP8 512x1024 latency median | 8.8517 us | 8.7933 us | -0.66% |

