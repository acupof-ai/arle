# DeepSeek V4 raw FP8 GEMV native decode - 2026-05-12

## Goal

- Continue the FP8 / FP4 quantized-operator pass on the newly landed
  DeepSeek V4 raw FP8/FP4 GEMV kernels, starting with the FP8 weight decode
  path while keeping FP4 and block-scale logic unchanged.

## Hypothesis

- `dsv4_fp8_gemv_kernel` decoded every FP8 E4M3 byte with integer field
  extraction plus `exp2f`. CUDA 13.2 exposes `__nv_fp8_e4m3` widening
  conversion in `/opt/cuda/targets/x86_64-linux/include/cuda_fp8.hpp`.
  Replacing the normal-code path with the CUDA conversion should remove the
  per-weight `exp2f` cost.
- Guard `0x7f/0xff` explicitly. CUDA documents these E4M3 values as NaN,
  while the existing DeepSeek V4 kernel treated them as signed finite max
  `±448`. The optimization must preserve that existing behavior.

## Command

Component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  dsv4_fp8_gemv --quiet
```

FP4 no-regression check:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  dsv4_fp4_gemv --quiet
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
- Operator: `dsv4_fp8_gemv_cuda` / `dsv4_fp8_gemv_batch_cuda`
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
| weights | raw FP8 E4M3 bytes | raw FP8 E4M3 bytes |
| scales | FP8 E8M0 bytes, all `127` (=1.0) | FP8 E8M0 bytes, all `127` (=1.0) |

## Results - Component A/B

Only `dsv4_decode_fp8_e4m3` changed:

```cpp
// before: manual sign/exp/mant decode with exp2f
// after: special-case 0x7f/0xff, otherwise __nv_fp8_e4m3 -> float
```

| Shape | Baseline | Native-decode candidate | Delta |
|---|---:|---:|---:|
| 1024x1024 | `14.162-14.501 us`, point `14.280 us` | `10.767-10.791 us`, point `10.778 us` | `-24.52%` |
| 512x1024 | `11.883-11.936 us`, point `11.913 us` | `8.8359-8.8682 us`, point `8.8517 us` | `-25.70%` |

Throughput:

| Shape | Baseline | Native-decode candidate | Delta |
|---|---:|---:|---:|
| 1024x1024 | `73.430 Gelem/s` | `97.285 Gelem/s` | `+32.49%` |
| 512x1024 | `44.010 Gelem/s` | `59.230 Gelem/s` | `+34.58%` |

## Results - FP4 No-Regression Check

FP4 code was not changed; this rerun checks the same translation unit did not
regress the FP4 path.

| Shape | Baseline | After FP8 change | Delta |
|---|---:|---:|---:|
| 1024x1024 | `13.409 us` | `13.380 us` | `-0.22%` |
| 512x1024 | `10.520 us` | `10.407 us` | `-1.07%` |

## Results - Correctness

```text
test ops::tests::test_dsv4_fp8_gemv ... ok
test ops::tests::test_dsv4_fp8_gemv_preserves_finite_nan_pattern ... ok
```

The new test covers `0x7f -> +448` and `0xff -> -448`, preserving the
pre-existing DeepSeek V4 finite-max behavior instead of inheriting CUDA's NaN
interpretation for those two raw FP8 byte patterns.

## Problems

- This is a component bench only. DeepSeek V4 CUDA serving is still in scaffold
  phase, so there is no Guidellm request-level A/B for this operator yet.
- The benchmark uses synthetic DSV4-mini-sized shapes and 128x128 scale blocks.
  It validates the operator hot loop but not full large-checkpoint distribution
  effects.
- The block-scale division/clamp path remains unchanged and is still a separate
  hypothesis, not evidence.

## Learnings

- On SM89, native CUDA FP8 widening conversion is materially faster than the
  hand-written E4M3 decode with `exp2f` for raw DeepSeek V4 FP8 GEMV.
- Preserving `0x7f/0xff` semantics requires an explicit guard because CUDA
  E4M3 treats those byte patterns as NaN.
- FP4 raw GEMV is now the next better target; this change does not explain or
  improve FP4 decode cost beyond the no-regression rerun.

## Delta vs Baseline

| metric | baseline | now | delta |
|---|---:|---:|---:|
| FP8 1024x1024 latency median | 14.280 us | 10.778 us | -24.52% |
| FP8 512x1024 latency median | 11.913 us | 8.8517 us | -25.70% |
| FP4 1024x1024 latency median | 13.409 us | 13.380 us | -0.22% |
| FP4 512x1024 latency median | 10.520 us | 10.407 us | -1.07% |

