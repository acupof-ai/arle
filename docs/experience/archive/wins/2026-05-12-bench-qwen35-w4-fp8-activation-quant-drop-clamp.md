# Qwen3.5 W4+FP8 activation quant drop redundant clamp - 2026-05-12

## Goal

- Continue the FP8 / FP4 quantization operator pass by optimizing the opt-in
  W4+FP8 prefill activation quantizer:
  BF16 row-major activations -> FP8 E4M3 rows + FP32 per-row scales.

## Hypothesis

- `quantize_bf16_rows_to_fp8_e4m3_kernel` computes per-row
  `scale = absmax / 448.0`, so normal row values are already in the FP8 E4M3
  finite range before conversion. The kernel still applied explicit
  `fminf/fmaxf` clamp before constructing `__nv_fp8_e4m3`. CUDA's local
  `/opt/cuda/targets/x86_64-linux/include/cuda_fp8.hpp` documents that the
  `__nv_fp8_e4m3(float)` constructor uses `__NV_SATFINITE` for out-of-range
  values. Removing the hand clamp should reduce per-element float instructions
  without changing the scale formula, layout, or conversion saturation policy.

## Command

Component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/quantize_bf16_rows_to_fp8_e4m3 --quiet
```

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda \
  ffi::gemm::tests::fp8_row_quantization_scales_match_absmax -- --nocapture
```

## Environment

- Backend: CUDA
- Operator: `quantize_bf16_rows_to_fp8_e4m3_cuda`
- Runtime path: opt-in W4+FP8 prefill, `INFER_MARLIN_W4_FP8_PREFILL=1`
- Hardware: NVIDIA GeForce RTX 4070 Ti SUPER, SM89
- Driver / CUDA: 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- Feature set: `cargo bench -p infer --features cuda --bench ops_bench`
- Non-default flags / env vars: `CUDARC_CUDA_VERSION=13010`,
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

## Results - Component A/B

Only the explicit clamp before FP8 conversion changed:

```cpp
// before
float qf = x / scale;
qf = fminf(448.0f, fmaxf(-448.0f, qf));
out = __nv_fp8_e4m3(qf);

// after
float qf = x / scale;
out = __nv_fp8_e4m3(qf);
```

| Shape | Baseline | Candidate first run | Candidate final rerun | Delta vs baseline |
|---|---:|---:|---:|---:|
| 2048x2560 | `24.399-24.450 us`, point `24.423 us` | `23.685-23.743 us`, point `23.711 us` | `23.664-23.694 us`, point `23.678 us` | `-3.05%` |
| 2048x9216 | `116.22-116.31 us`, point `116.27 us` | `115.44-115.51 us`, point `115.48 us` | `115.44-115.50 us`, point `115.48 us` | `-0.68%` |

Final rerun throughput:

| Shape | Throughput |
|---|---:|
| 2048x2560 | `221.27-221.55 Gelem/s`, point `221.43 Gelem/s` |
| 2048x9216 | `163.42-163.49 Gelem/s`, point `163.45 Gelem/s` |

## Results - Correctness

```text
test ffi::gemm::tests::fp8_row_quantization_scales_match_absmax ... ok
```

The existing test covers an all-zero row, a nonzero row with `cols=513`, and
the `absmax / 448.0` scale contract.

## Problems

- This is a component bench only. The affected runtime path is opt-in and the
  local workspace still does not have a W4-hybrid checkpoint with the W4+FP8
  side buffer needed for a Qwen3.5 Guidellm serving A/B. Serving validation is
  `pending-model`.
- The correctness test validates scale semantics and zero-row output, but does
  not exhaustively compare every nonzero FP8 byte. The CUDA header check above
  is therefore part of the evidence for preserving saturation behavior.
- The larger 2048x9216 shape improves by less than 1%; keep the change because
  both shapes have separated positive intervals and the hidden shape improves
  by `3.05%`.

## Learnings

- The hand clamp was redundant with CUDA FP8 constructor saturation on this
  path and cost measurable time, especially on hidden-size rows.
- This result is specific to the FP8 activation quantizer. It does not apply to
  W4A8 INT8 activation quantization, where the integer round/clamp path remains
  required.
- The obvious FP8 activation quant low-level instruction axis is now mostly
  exhausted after warp reduction, launch-smem trim, and clamp removal.

## Delta vs Baseline

| metric | baseline | now | delta |
|---|---:|---:|---:|
| hidden latency median | 24.423 us | 23.678 us | -3.05% |
| hidden throughput median | 214.67 Gelem/s | 221.43 Gelem/s | +3.15% |
| intermediate latency median | 116.27 us | 115.48 us | -0.68% |
| intermediate throughput median | 162.33 Gelem/s | 163.45 Gelem/s | +0.69% |

## Artefacts

- Serving gate: `pending-model` for Qwen3.5 W4-hybrid checkpoint with W4+FP8
  side buffers.
