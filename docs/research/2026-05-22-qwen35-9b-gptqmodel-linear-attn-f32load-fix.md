# Qwen3.5-9B GPTQModel Linear-Attention F32-Load Fix

## Context

The layer-local GPTQModel W4 GEMV scan passed, and the dense-module scan showed
embedding, final RMSNorm, and untied `lm_head` were clean. The next failing gate
was layer-0 `linear_attention`: before this fix ARLE produced all-NaN output for
the 9B GPTQModel checkpoint while the PyTorch BF16 source reference was finite.

## Root Cause

`load_tensor_1d_f32` assumed every tensor it loads is physically `F32` and
reinterpreted raw bytes as `f32`. That is correct for the BF16 source
checkpoint's `linear_attn.A_log` and `linear_attn.norm.weight`, but not for the
DavidWen GPTQModel checkpoint, where those tensors are stored as `BF16`.

Observed before the fix:

| Stage | Expected | Before fix |
| --- | ---: | ---: |
| `A_log` length | `32` | `16` |
| `norm.weight` length | `128` | `64` |
| first non-finite stage | none | `gdr_g_cumsum` |
| `gdr_g_cumsum` non-finite | `0` | `6 -inf` |
| final layer-0 output | finite | `4096 NaN` |

The loaded lengths were exactly half the expected values because pairs of BF16
values were being interpreted as one `f32`.

## Fix

`load_tensor_1d_f32` and `load_tensor_1d_f32_sharded` now convert 1D `F32`,
`BF16`, and `F16` tensors into `f32` by dtype instead of doing a blind f32 byte
reinterpret.

## Evidence

Commands:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo check -p infer --example qwen35_linear_attn_substage_dump --release --features cuda

INFER_EXPERIMENTAL_GPTQMODEL_W4=1 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p infer --example qwen35_linear_attn_substage_dump --release --features cuda -- \
  --model-path /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --token-id 9419 \
  --output bench-output/2026-05-22-qwen35-9b-gptqmodel-linear-attn-substages-after-f32load-fix/qwen35-9b-gptqmodel.json

INFER_EXPERIMENTAL_GPTQMODEL_W4=1 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p infer --example qwen35_linear_attn_parity --release --features cuda -- \
  --model-path /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --reference-model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --python /home/ckl/projects/arle/.venv/bin/python \
  --python-device cpu \
  --token-id 9419 \
  --output bench-output/2026-05-22-qwen35-9b-gptqmodel-linear-attn-parity-after-f32load-fix/layer0-linear-attn.json
```

After the fix:

| Check | Result |
| --- | ---: |
| `A_log` length | `32` |
| `norm.weight` length | `128` |
| first non-finite stage | `<none>` |
| `gdr_g_cumsum` non-finite | `0` |
| `gdr_output` non-finite | `0` |
| `out_proj` non-finite | `0` |
| layer-0 linear-attn RMSE/reference-RMS | `4.0655%` |
| layer-0 linear-attn gate | PASS (`<5%`) |

Artifacts:

```text
bench-output/2026-05-22-qwen35-9b-gptqmodel-linear-attn-substages/
bench-output/2026-05-22-qwen35-9b-gptqmodel-linear-attn-substages-after-f32load-fix/
bench-output/2026-05-22-qwen35-9b-gptqmodel-linear-attn-parity-after-f32load-fix/
```

## Decision

This licenses the 1D dtype-load fix and removes layer-0 `linear_attention` NaNs
as the immediate GPTQModel blocker. It does not license the 9B GPTQModel teacher
for OPD: the follow-up multi-token generation smoke still collapses to repeated
`!` tokens. That failure is recorded separately as a generation-quality kill.

