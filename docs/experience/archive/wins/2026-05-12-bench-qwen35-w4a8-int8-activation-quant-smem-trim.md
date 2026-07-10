# Qwen3.5 W4A8 INT8 activation quant launch smem trim - 2026-05-12

## Goal

- Continue the FP8/FP4/W4 quantization operator pass by testing the same
  dynamic shared-memory launch trim on the W4A8 INT8 activation quantizer:
  BF16 row-major activations -> INT8 rows + FP32 per-row scales.

## Hypothesis

- `quantize_bf16_rows_to_int8_kernel` uses warp reductions and writes only one
  float per warp to dynamic shared memory. The launch still reserved one float
  per thread (`256 * sizeof(float)`). Matching allocation to the eight warp
  slots should reduce unnecessary dynamic shared memory without changing math,
  block size, or layout.

## Command

Baseline and candidate component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/quantize_bf16_rows_to_int8 --quiet
```

Correctness and feature checks:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda \
  ffi::gemm::tests::int8_row_quantization_scales_match_absmax -- --nocapture

CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda

CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo check -p infer --features cuda

CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo clippy -p infer --features cuda -- -D warnings

CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo check -p infer --bench ops_bench --no-default-features --features cuda,no-cuda
```

## Environment

- **Backend:** CUDA
- **Operator:** `quantize_bf16_rows_to_int8_cuda`
- **Runtime path:** W4A8 Marlin activation quantization
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, SM89
- **Driver / CUDA:** 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- **Commit before change:** `d9ffb72`
- **Feature set:** `cargo bench -p infer --features cuda --bench ops_bench`
- **Non-default flags / env vars:** `CUDARC_CUDA_VERSION=13010`,
  `NVCC_CCBIN=/usr/bin/g++-14`,
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
| block threads | 256 | 256 |
| dynamic smem before | 1024 bytes | 1024 bytes |
| dynamic smem after | 32 bytes | 32 bytes |

## Results

Only the dynamic shared-memory byte count changed. Quantization formula, clamp,
block size, grid shape, and output layout are unchanged.

| Shape | Baseline | Candidate first run | Candidate final rerun | Delta vs baseline |
|---|---:|---:|---:|---:|
| 2048x2560 | `24.638-24.668 us`, point `24.654 us` | `24.658-24.678 us`, point `24.667 us` | `24.674-24.695 us`, point `24.686 us` | `+0.13%` |
| 2048x9216 | `118.99-119.11 us`, point `119.05 us` | `117.18-117.31 us`, point `117.26 us` | `117.22-117.37 us`, point `117.31 us` | `-1.46%` |

Throughput final rerun:

| Shape | Throughput |
|---|---:|
| 2048x2560 | `212.30-212.48 Gelem/s`, point `212.39 Gelem/s` |
| 2048x9216 | `160.81-161.01 Gelem/s`, point `160.90 Gelem/s` |

Correctness:

```text
test ffi::gemm::tests::int8_row_quantization_scales_match_absmax ... ok
```

Full crate test gate:

```text
test result: ok. 39 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Feature checks:

```text
cargo check -p infer --features cuda ... ok
cargo clippy -p infer --features cuda -- -D warnings ... ok
cargo check -p infer --bench ops_bench --no-default-features --features cuda,no-cuda ... ok
```

## Problems

- The hidden-size 2048x2560 shape regressed by `+0.13%` by point estimate. This
  is small, but it is not a win and should not be reported as one.
- The 2048x9216 MLP intermediate shape improves by `-1.46%` and is stable
  across two treatment runs. This is the only licensed benefit.
- This is a component bench. It does not by itself prove model-level TTFT/ITL
  movement; W4A8 serving wins remain governed by the prior matched Guidellm
  entries.
- Local CUDA is 13.2, while `cudarc 0.18.2` supports up to CUDA 13.1 bindings.
  The bench and checks use `CUDARC_CUDA_VERSION=13010` explicitly as a local
  measurement workaround.

## Learnings

- The launch-smem trim is consistently useful for wide 9216-column row
  quantization, but not for 2560-column rows.
- Shape-specific evidence matters here: treating the two Qwen3.5 activation
  shapes as one bucket would hide the hidden-size non-win.
- Keep this as a narrow W4A8 intermediate-layer activation quantization cleanup,
  not a general W4A8 serving-performance claim.
