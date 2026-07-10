# Qwen3.5 W4A8 INT8 activation quant integer round - 2026-05-12

## Goal

- Continue the FP8/FP4/W4 quantization operator pass by optimizing the W4A8
  INT8 activation quantizer used before Marlin W4A8 GEMM:
  BF16 row-major activations -> INT8 rows + FP32 per-row scales.

## Hypothesis

- The current quantization pass rounds with `nearbyintf`, clamps in float, and
  then casts to `int8_t`. Other INT8 KV quantization kernels already use
  `__float2int_rn` followed by integer clamp. For the same round-to-nearest
  semantics, replacing the float round/clamp sequence with integer round/clamp
  should reduce conversion work without changing the scale formula, row
  reduction, output layout, or FFI surface.

## Command

Component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/quantize_bf16_rows_to_int8 --quiet
```

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda \
  ffi::gemm::tests::int8_row_quantization_scales_match_absmax -- --nocapture
```

## Environment

- Backend: CUDA
- Operator: `quantize_bf16_rows_to_int8_cuda`
- Runtime path: W4A8 Marlin activation quantization
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
| output | INT8 row-major | INT8 row-major |
| scales | FP32 per row | FP32 per row |
| scale formula | `absmax / 127.0` | `absmax / 127.0` |

## Results - Component A/B

Only the final quantization round/clamp changed:

```cpp
// before
float qf = nearbyintf(x / scale);
qf = fminf(127.0f, fmaxf(-128.0f, qf));
out = static_cast<int8_t>(qf);

// after
int q = __float2int_rn(x / scale);
q = max(-128, min(127, q));
out = static_cast<int8_t>(q);
```

| Shape | Baseline | Candidate first run | Candidate final rerun | Delta vs baseline |
|---|---:|---:|---:|---:|
| 2048x2560 | `24.653-24.686 us`, point `24.669 us` | `24.218-24.254 us`, point `24.232 us` | `24.211-24.234 us`, point `24.224 us` | `-1.80%` |
| 2048x9216 | `117.25-117.34 us`, point `117.30 us` | `116.02-116.16 us`, point `116.10 us` | `116.12-116.21 us`, point `116.17 us` | `-0.96%` |

Final rerun throughput:

| Shape | Throughput |
|---|---:|
| 2048x2560 | `216.34-216.55 Gelem/s`, point `216.43 Gelem/s` |
| 2048x9216 | `162.41-162.55 Gelem/s`, point `162.48 Gelem/s` |

## Results - Correctness

```text
test ffi::gemm::tests::int8_row_quantization_scales_match_absmax ... ok
```

The existing correctness test covers an all-zero row, a nonzero row with
`cols=513`, the `absmax / 127.0` scale contract, and the expected largest
magnitude quantized value.

## Problems

- This is a component bench only. The local workspace has Qwen3 W4A8
  checkpoints, but not a Qwen3.5 W4A8 checkpoint matching the bench shape, so
  Qwen3.5 serving A/B remains pending-model.
- The larger 2048x9216 shape improves by less than 1% on the final rerun. The
  change is kept because both shapes have separated positive intervals, the
  hidden shape improves by `1.80%`, and the code path is simpler than the float
  round/clamp sequence.
- This result is specific to INT8 W4A8 activation quantization. It does not
  imply a FP8 activation quant win because the FP8 path uses NVIDIA FP8
  conversion constructors, not `nearbyintf`.

## Learnings

- For W4A8 activation quantization, integer round/clamp is the measured faster
  path on sm_89 after the earlier warp-reduction and launch-smem trims.
- The improvement is modest but stable across the two Qwen3.5-shaped rows.
- The INT8 activation quantizer is now aligned with the INT8 KV quantization
  style for the final round/clamp step.

## Delta vs Baseline

| metric | baseline | now | delta |
|---|---:|---:|---:|
| hidden latency median | 24.669 us | 24.224 us | -1.80% |
| hidden throughput median | 212.53 Gelem/s | 216.43 Gelem/s | +1.84% |
| intermediate latency median | 117.30 us | 116.17 us | -0.96% |
| intermediate throughput median | 160.90 Gelem/s | 162.48 Gelem/s | +0.98% |

## Artefacts

- Serving gate: `pending-model` for Qwen3.5 W4A8 checkpoint.
