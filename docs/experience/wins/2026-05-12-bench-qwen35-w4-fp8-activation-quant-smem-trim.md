# Qwen3.5 W4+FP8 activation quant launch smem trim - 2026-05-12

## Goal

- Continue the FP8/FP4 quantization operator pass by reducing fixed launch
  overhead in the opt-in W4+FP8 prefill activation quantizer:
  BF16 row-major activations -> FP8 E4M3 rows + FP32 per-row scales.

## Hypothesis

- `quantize_bf16_rows_to_fp8_e4m3_kernel` only stores one reduction value per
  warp in dynamic shared memory. The launch still reserved one float per
  thread (`256 * sizeof(float)`). Reserving only `num_warps` floats should keep
  the math and occupancy contract identical while trimming unnecessary dynamic
  shared-memory allocation.

## Command

Baseline and candidate component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/quantize_bf16_rows_to_fp8_e4m3 --quiet
```

Correctness and crate test gate:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda \
  ffi::gemm::tests::fp8_row_quantization_scales_match_absmax -- --nocapture

CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda
```

Feature checks:

```bash
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
- **Operator:** `quantize_bf16_rows_to_fp8_e4m3_cuda`
- **Runtime path:** opt-in W4+FP8 prefill, `INFER_MARLIN_W4_FP8_PREFILL=1`
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, SM89
- **Driver / CUDA:** 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- **Commit before change:** `55cf8b3`
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
| output | FP8 E4M3 row-major | FP8 E4M3 row-major |
| scales | FP32 per row | FP32 per row |
| scale formula | `absmax / 448.0` | `absmax / 448.0` |
| block threads | 256 | 256 |
| dynamic smem before | 1024 bytes | 1024 bytes |
| dynamic smem after | 32 bytes | 32 bytes |

## Results

Only the dynamic shared-memory byte count changed. FP8 format, scale formula,
block size, grid shape, and output layout are unchanged.

| Shape | Baseline | Candidate first run | Candidate final rerun | Delta vs baseline |
|---|---:|---:|---:|---:|
| 2048x2560 | `24.486-24.509 us`, point `24.497 us` | `24.398-24.412 us`, point `24.406 us` | `24.451-24.488 us`, point `24.466 us` | `-0.13%` |
| 2048x9216 | `117.85-117.92 us`, point `117.89 us` | `116.17-116.26 us`, point `116.23 us` | `116.22-116.32 us`, point `116.27 us` | `-1.37%` |

Throughput final rerun:

| Shape | Throughput |
|---|---:|
| 2048x2560 | `214.10-214.43 Gelem/s`, point `214.30 Gelem/s` |
| 2048x9216 | `162.27-162.40 Gelem/s`, point `162.33 Gelem/s` |

Correctness:

```text
test ffi::gemm::tests::fp8_row_quantization_scales_match_absmax ... ok
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

- The 2048x2560 shape is noise-level: the final candidate interval slightly
  overlaps the baseline interval. It is not evidence for a material hidden-row
  speedup.
- The 2048x9216 shape is a stable small win across two treatment runs and is
  the more expensive activation-quant shape in the Qwen3.5 MLP path.
- This is still a component bench only. The affected runtime path is
  experimental and opt-in; the local workspace does not currently have a
  W4-hybrid checkpoint with the W4+FP8 side buffer needed for a real Guidellm
  serving A/B. Serving validation remains **pending-model**.
- Local CUDA is 13.2, while `cudarc 0.18.2` supports up to CUDA 13.1 bindings.
  The bench and tests therefore use `CUDARC_CUDA_VERSION=13010` explicitly.
  This is a local measurement workaround, not a Cargo feature change.

## Learnings

- After the prior warp-reduction win, the remaining launch-smem trim is small
  but real for the wide 9216-column FP8 activation quantization shape.
- The change is low risk because it only matches the launch allocation to the
  kernel's actual `smem[warp_id]` use: eight floats for a 256-thread block.
- Do not generalize this as a serving win until the W4+FP8 opt-in path has a
  matched model-level Guidellm A/B.
