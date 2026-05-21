# ARLE TurboQuant 9B Layer-Local GEMV Parity

## Goal

After `b8864cf` fixed tensor-local TurboQuant dequant parity, full-model
Qwen3.5-9B-TQ4 logits still failed against BF16 PyTorch
([`docs/experience/errors/2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill.md`](../experience/errors/2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill.md)).

This tranche tests the next attribution level: does the fused decode GEMV path
match the reference path that bulk-dequants the same TQ4 weight and then calls
cuBLAS GEMM? The first commit tested one projection. This update extends the
same parametric harness across sampled projection families and layers.

## Hypothesis

If fused TurboQuant GEMV diverges from bulk-dequant+cuBLAS on the same weight
and same input vector, the remaining full-model drift is still inside
`crates/cuda-kernels/csrc/gemm/turboquant_weight_gemv.cu`.

If they agree, the full-model drift is likely elsewhere: projection-family
coverage, dense fallback weights, attention/norm integration, or accumulated
4-bit quantization error.

## Params

- Model: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4`
- Projections:
  - layer 0 MLP: `gate_proj`, `up_proj`, `down_proj`
  - layer 3 full attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - layer 3 MLP: `gate_proj`, `up_proj`, `down_proj`
  - layer 10 MLP: `gate_proj`
- Note: Qwen3.5-9B `config.json` has
  `layer_types[0..12] = [linear_attention, linear_attention,
  linear_attention, full_attention, ...]`, with `full_attention_interval=4`.
  Layer 0 therefore has no `self_attn.q/k/v/o` tensors. The first full-attn
  projection scan uses layer 3 instead.
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
  --tensor-base model.language_model.layers.3.self_attn.q_proj \
  --seed 1592594996 \
  --output bench-output/2026-05-21-qwen35-9b-tq4-projection-gemv-scan/layer3-self-attn-q.json
```

Raw artifacts:

- `bench-output/2026-05-21-qwen35-9b-tq4-gemv-layerlocal/layer0-gate-proj-gemv-parity-run.txt`
- `bench-output/2026-05-21-qwen35-9b-tq4-gemv-layerlocal/layer0-gate-proj-gemv-parity.json`
- `bench-output/2026-05-21-qwen35-9b-tq4-projection-gemv-scan/summary.json`
- `bench-output/2026-05-21-qwen35-9b-tq4-projection-gemv-scan/*.json`
- `bench-output/2026-05-21-qwen35-9b-tq4-projection-gemv-scan/*.run.txt`

## Results

Gate: `RMSE/reference-RMS <= 1%`.

| Label | Tensor | Shape | Max abs | RMSE/reference-RMS | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| layer0-mlp-gate | `model.language_model.layers.0.mlp.gate_proj` | `12288x4096` | `0.0078125` | `0.2498%` | PASS |
| layer0-mlp-up | `model.language_model.layers.0.mlp.up_proj` | `12288x4096` | `0.0078125` | `0.2519%` | PASS |
| layer0-mlp-down | `model.language_model.layers.0.mlp.down_proj` | `4096x12288` | `0.015625` | `0.2558%` | PASS |
| layer3-self-attn-q | `model.language_model.layers.3.self_attn.q_proj` | `8192x4096` | `0.015625` | `0.2627%` | PASS |
| layer3-self-attn-k | `model.language_model.layers.3.self_attn.k_proj` | `1024x4096` | `0.015625` | `0.2771%` | PASS |
| layer3-self-attn-v | `model.language_model.layers.3.self_attn.v_proj` | `1024x4096` | `0.0078125` | `0.2523%` | PASS |
| layer3-self-attn-o | `model.language_model.layers.3.self_attn.o_proj` | `4096x4096` | `0.0078125` | `0.2436%` | PASS |
| layer3-mlp-gate | `model.language_model.layers.3.mlp.gate_proj` | `12288x4096` | `0.0078125` | `0.2549%` | PASS |
| layer3-mlp-up | `model.language_model.layers.3.mlp.up_proj` | `12288x4096` | `0.0078125` | `0.2567%` | PASS |
| layer3-mlp-down | `model.language_model.layers.3.mlp.down_proj` | `4096x12288` | `0.015625` | `0.2555%` | PASS |
| layer10-mlp-gate | `model.language_model.layers.10.mlp.gate_proj` | `12288x4096` | `0.0078125` | `0.2546%` | PASS |

`max_rel` is a near-zero denominator artifact: the reference output at the
max-rel index is close to zero. The scale-stable metric is
`RMSE/reference-RMS`. All sampled projections landed in the `0.24%-0.28%`
band, and max absolute error was one to two BF16-scale steps for these output
ranges.

First 8 output entries for the original layer 0 `mlp.gate_proj` check:

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

Layer-local fused TurboQuant GEMV reproduces the bulk-dequant cuBLAS reference
for every sampled quantized projection:

- layer 0 linear-attn-block MLP;
- the first full-attn block's q/k/v/o projections and MLP;
- a mid-layer MLP gate projection.

This does not support "TurboQuant fused GEMV compute path is the remaining
root cause" for the Qwen3.5-9B-TQ4 full-model logits failure.

Do not run the 9B-TQ4 OPD bench yet. The next attribution step should move to
the dense path. Since layer 0 is `linear_attention`, the first dense suspect is
the layer-0 linear-attention forward path (`in_proj_qkv`, `in_proj_z`,
`in_proj_a`, `in_proj_b`, `conv1d`, `A_log`, `dt_bias`, `norm`, and
`out_proj`) against a PyTorch BF16 reference on the same hidden input.
Embedding, final norm, and LM head remain secondary dense suspects.

## Problems

This is projection-local evidence, not a model-wide license. It does not test:

- every layer and every projection, though it samples layer 0, layer 3, and
  layer 10;
- dense fallback modules, especially layer-0 `linear_attn`, embeddings, final
  norm, or LM head;
- attention/RoPE/norm integration;
- accumulated 4-bit quantization drift across the full network.

## Learnings

After tensor-local dequant passes, layer-local GEMV can quickly separate
"fused kernel compute bug" from broader model-path drift. Use scale-stable
RMSE/reference-RMS and max_abs first; raw max_rel is not SOLID when the
reference value is near zero.
