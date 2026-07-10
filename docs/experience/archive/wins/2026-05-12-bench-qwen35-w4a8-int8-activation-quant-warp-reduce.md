# Qwen3.5 W4A8 INT8 activation quant warp reduction - 2026-05-12

## Goal

- Optimize the W4A8 Marlin activation quantization operator used before
  `gemm_w4a8_marlin_cuda`: BF16 row-major activations -> INT8 rows + FP32
  per-row scales.

## Hypothesis

- `quantize_bf16_rows_to_int8_cuda` used the same 256-thread shared-memory
  tree reduction that the W4+FP8 activation quantizer previously used. For
  Qwen3.5-shaped rows, replacing it with warp reductions plus one per-warp
  shared reduce should reduce synchronization cost without changing the INT8
  scale formula, rounding, clamp, or FFI surface.

## Command

Baseline and candidate component A/B:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/quantize_bf16_rows_to_int8 --quiet
```

Correctness:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda \
  ffi::gemm::tests::int8_row_quantization_scales_match_absmax -- --nocapture
```

## Environment

- **Backend:** CUDA
- **Operator:** `quantize_bf16_rows_to_int8_cuda`
- **Runtime path:** W4A8 Marlin linear path
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, SM89
- **Driver / CUDA:** 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- **Feature set:** `cargo bench -p infer --features cuda --bench ops_bench`
- **Non-default flags / env vars:** `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `TORCH_CUDA_ARCH_LIST=8.9`

## Params

| Param | hidden shape | intermediate shape |
|---|---:|---:|
| rows | 2048 | 2048 |
| cols | 2560 | 9216 |
| input | BF16 row-major | BF16 row-major |
| output | INT8 row-major | INT8 row-major |
| scales | FP32 per row | FP32 per row |
| scale formula | `absmax / 127.0` | `absmax / 127.0` |

## Results

| Shape | Baseline | Candidate first run | Candidate final rerun | Delta vs baseline |
|---|---:|---:|---:|---:|
| 2048x2560 | `26.635-26.693 us`, point `26.671 us` | `24.691-24.710 us`, point `24.700 us` | `24.699-24.710 us`, point `24.706 us` | `-7.37%` |
| 2048x9216 | `120.34-120.55 us`, point `120.44 us` | `118.93-119.06 us`, point `118.99 us` | `119.00-119.14 us`, point `119.07 us` | `-1.14%` |

Throughput final rerun:

| Shape | Throughput |
|---|---:|
| 2048x2560 | `212.17-212.27 Gelem/s`, point `212.21 Gelem/s` |
| 2048x9216 | `158.42-158.61 Gelem/s`, point `158.52 Gelem/s` |

Correctness:

```text
test ffi::gemm::tests::int8_row_quantization_scales_match_absmax ... ok
```

The test covers an all-zero row, a nonzero row with `cols=513`, the
`absmax / 127.0` scale, and the expected `-127` quantized value for the row's
largest-magnitude element.

## Problems

- This is a component bench only. The local workspace does not have a Qwen3.5
  W4A8 checkpoint matching this shape, so an apples-to-apples Qwen3.5 W4A8
  Guidellm serving A/B is pending-model.
- The larger 2048x9216 shape improves by only `1.14%`; the change is kept
  because it also improves the 2048x2560 shape by `7.37%` and keeps the same
  interface and math.

## Learnings

- INT8 activation quantization follows the same reduction behavior as the
  W4+FP8 activation quantizer: warp reductions reduce sync overhead for
  hidden-size rows, while intermediate-size rows are mostly scan/conversion
  bound.
- Do not project the component delta to whole-request throughput without a
  matching W4A8 model-level run. This operator is only one part of the Marlin
  W4A8 linear path.

