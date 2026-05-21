# ARLE TurboQuant 9B Layer-Local GEMV Parity

## Goal

After `b8864cf` fixed tensor-local TurboQuant dequant parity, full-model
Qwen3.5-9B-TQ4 logits still failed against BF16 PyTorch
([`docs/experience/errors/2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill.md`](../experience/errors/2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill.md)).

This tranche tests the next attribution level: does the fused decode GEMV path
match the reference path that bulk-dequants the same TQ4 weight and then calls
cuBLAS GEMM?

## Hypothesis

If fused TurboQuant GEMV diverges from bulk-dequant+cuBLAS on the same weight
and same input vector, the remaining full-model drift is still inside
`crates/cuda-kernels/csrc/gemm/turboquant_weight_gemv.cu`.

If they agree, the full-model drift is likely elsewhere: projection-family
coverage, dense fallback weights, attention/norm integration, or accumulated
4-bit quantization error.

## Params

- Model: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4`
- Projection: `model.language_model.layers.0.mlp.gate_proj`
- Shape: `[rows=12288, cols=4096]`
- Quantization: TurboQuant 4-bit, group size 128
- Input: deterministic BF16 vector `[1, 4096]`, seed `1592594996`
- GPU: RTX 4070 Ti SUPER
- CUDA env:
  `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `CUDARC_CUDA_VERSION=13010`, `TORCH_CUDA_ARCH_LIST=8.9`,
  `CARGO_BUILD_JOBS=1`

Command:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p infer --example turboquant_weight_gemv_parity --release --features cuda -- \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --tensor-base model.language_model.layers.0.mlp.gate_proj \
  --seed 1592594996 \
  --output bench-output/2026-05-21-qwen35-9b-tq4-gemv-layerlocal/layer0-gate-proj-gemv-parity.json
```

Raw artifacts:

- `bench-output/2026-05-21-qwen35-9b-tq4-gemv-layerlocal/layer0-gate-proj-gemv-parity-run.txt`
- `bench-output/2026-05-21-qwen35-9b-tq4-gemv-layerlocal/layer0-gate-proj-gemv-parity.json`

## Results

| Comparison | Max abs | Mean abs | RMSE | RMSE/reference-RMS | Max rel | Mean rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fused TQ GEMV vs bulk-dequant+cuBLAS | `0.0078125` | `0.0005269` | `0.0010201` | `0.0024978` | `1526.1993` | `0.1386` |

`max_rel` is a near-zero denominator artifact: the reference output at the
max-rel index is close to zero. The scale-stable metric is
`RMSE/reference-RMS = 0.2498%`, and max absolute error is one BF16-scale step
for this output range.

First 8 output entries:

| Index | Fused GEMV | Bulk dequant + cuBLAS | Abs err | Rel err |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `-0.4140625` | `-0.4140625` | `0.0` | `0.0` |
| 1 | `0.20703125` | `0.20703125` | `0.0` | `0.0` |
| 2 | `0.71875` | `0.71875` | `0.0` | `0.0` |
| 3 | `0.18066406` | `0.1796875` | `0.00097656` | `0.00543` |
| 4 | `0.07177734` | `0.072265625` | `0.00048828` | `0.00676` |
| 5 | `0.56640625` | `0.56640625` | `0.0` | `0.0` |
| 6 | `0.166015625` | `0.166015625` | `0.0` | `0.0` |
| 7 | `0.64453125` | `0.64453125` | `0.0` | `0.0` |

## Decision

Layer 0 `mlp.gate_proj` fused TurboQuant GEMV reproduces the bulk-dequant
cuBLAS reference. This does not support "fused GEMV compute path is the
remaining root cause" for this projection.

Do not run the 9B-TQ4 OPD bench yet. The next attribution step is a projection
family scan on layer 0 (`q/k/v/o/gate/up/down`) with the same harness, followed
by layer-0 forward parity if all projection-local GEMV checks pass.

## Problems

This is one projection, not a model-wide license. It does not test:

- `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, or `down_proj`;
- dense fallback modules, embeddings, final norm, or LM head;
- attention/RoPE/norm integration;
- accumulated 4-bit quantization drift across the full network.

## Learnings

After tensor-local dequant passes, layer-local GEMV can quickly separate
"fused kernel compute bug" from broader model-path drift. Use scale-stable
RMSE/reference-RMS and max_abs first; raw max_rel is not SOLID when the
reference value is near zero.
