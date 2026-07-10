# Qwen3.5 W4+FP8 activation quant warp reduction - 2026-05-12

## Goal

- Optimize the opt-in W4+FP8 prefill activation quantization operator used by
  `run_marlin_w4_fp8_prefill`: BF16 row-major activations -> FP8 E4M3 rows +
  FP32 per-row scales.

## Hypothesis

- The previous per-row absmax used a full shared-memory tree reduction with
  one `__syncthreads()` per stride. For Qwen3.5 prefill shapes, replacing that
  with warp reductions plus one per-warp shared reduce should cut reduction
  synchronization cost without changing the FP8 format, scale formula, or FFI
  surface.

## Command

Baseline and candidate component A/B:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/quantize_bf16_rows_to_fp8_e4m3 --quiet
```

Correctness:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda \
  ffi::gemm::tests::fp8_row_quantization_scales_match_absmax -- --nocapture
```

## Environment

- **Backend:** CUDA
- **Operator:** `quantize_bf16_rows_to_fp8_e4m3_cuda`
- **Runtime path:** opt-in W4+FP8 prefill, `INFER_MARLIN_W4_FP8_PREFILL=1`
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
| output | FP8 E4M3 row-major | FP8 E4M3 row-major |
| scales | FP32 per row | FP32 per row |
| scale formula | `absmax / 448.0` | `absmax / 448.0` |

## Results

| Shape | Baseline | Candidate first run | Candidate final rerun | Delta vs baseline |
|---|---:|---:|---:|---:|
| 2048x2560 | `26.472-26.501 us`, point `26.484 us` | `24.458-24.492 us`, point `24.475 us` | `24.462-24.493 us`, point `24.477 us` | `-7.58%` |
| 2048x9216 | `119.19-119.40 us`, point `119.29 us` | `117.90-118.05 us`, point `117.99 us` | `117.80-117.93 us`, point `117.85 us` | `-1.21%` |

Throughput final rerun:

| Shape | Throughput |
|---|---:|
| 2048x2560 | `214.05-214.33 Gelem/s`, point `214.19 Gelem/s` |
| 2048x9216 | `160.05-160.23 Gelem/s`, point `160.15 Gelem/s` |

Correctness:

```text
test ffi::gemm::tests::fp8_row_quantization_scales_match_absmax ... ok
```

The test covers the per-row scale contract for an all-zero row and a nonzero
row with `cols=513`, so it exercises multi-iteration row scans and non-warp
tail columns.

## Problems

- This is a component bench only. The affected runtime path is experimental and
  opt-in; the local workspace does not currently have a W4-hybrid checkpoint
  with the W4+FP8 side buffer needed for a real Guidellm serving A/B. Serving
  validation is therefore **pending-model**, not silently claimed.
- The larger 2048x9216 shape improves by only `1.21%`; this is a kept change
  because it also improves the 2048x2560 shape by `7.58%` and keeps the same
  interface and math.
- Full `cargo test --release -p cuda-kernels --features cuda` is currently
  blocked by an unrelated stable failure in
  `paged_kv::tests::retain_release_without_free_slot_does_not_move_pages`.
  The targeted FP8 row-quantization test above passes, and the failing test is
  outside this diff.

## Learnings

- For row-wise FP8 activation quantization, replacing a 256-thread shared tree
  reduction with warp reductions is enough to reduce synchronization cost on
  Qwen3.5-shaped prefill activations.
- The benefit is shape-dependent: hidden-size rows show a larger win than
  intermediate-size rows because the latter are more dominated by row scan and
  conversion work.
- Do not use this result as evidence for default Qwen3.5 serving speed. The
  path is behind `INFER_MARLIN_W4_FP8_PREFILL=1` and requires W4-hybrid model
  side tensors.
