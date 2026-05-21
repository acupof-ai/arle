# Qwen3.5-9B GPTQModel Dense Tensor Gate KILL

## Context

After the DavidWen2025 GPTQModel W4 projection scan passed ARLE CUDA W4A16
GEMV vs faithful GPTQ reference at <=0.25% RMSE/reference-RMS, the next gate
was dense fallback parity. This checks whether the checkpoint preserves the
non-quantized tensors that Qwen3.5 hybrid layers depend on.

Checkpoint paths:

- GPTQModel: `/home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit`
- BF16 source: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B`

Command:

```bash
INFER_EXPERIMENTAL_GPTQMODEL_W4=1 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
.venv/bin/python scripts/qwen35_tq4_dense_parity.py \
  --source-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --tq-model /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --token-id 9419 \
  --output-dir bench-output/2026-05-22-qwen35-9b-gptqmodel-dense-parity
```

## Root Cause

Dense tensor bit-compare fails before module parity:

| Metric | Value |
| --- | ---: |
| Dense tensors scanned | `576` |
| Non-identical dense tensors | `48` |
| Affected tensor families | `24 x linear_attn.A_log`, `24 x linear_attn.norm.weight` |
| Gate | KILL |

The GPTQModel checkpoint casts these linear-attention dense fallback tensors
from the BF16 source checkpoint's `float32` to `bfloat16`. `A_log` values are
representable in BF16 on this scan (`max_abs=0`), but every sampled
`linear_attn.norm.weight` differs after BF16 rounding, with max absolute drift
up to `0.00390625`.

The checkpoint also contains both `lm_head.weight` and
`model.language_model.lm_head.weight`; those two tensors are bit-identical.
The parity script now maps the nested duplicate to the top-level BF16 source
key so this is not counted as a failure.

## Evidence

Raw artifacts:

```text
bench-output/2026-05-22-qwen35-9b-gptqmodel-dense-parity/
```

Relevant prior gates:

- W4 projection GEMV path passed:
  `docs/research/2026-05-22-arle-qwen35-9b-gptqmodel-w4-gemv-parity.md`
- 0.8B BF16 layer-0 linear-attention implementation parity passed:
  `bench-output/2026-05-22-qwen35-linear-attn-parity/qwen35-08b-base-layer0-linear-attn.json`

## Fix

No runtime fix was applied. This is a checkpoint-quality/layout gate failure,
not a proven ARLE kernel bug.

The harness was made more general by skipping GPTQ side tensors
(`.qweight`, `.qzeros`, `.scales`, `.g_idx`) in the dense-copy scan and by
mapping the GPTQModel nested duplicate `model.language_model.lm_head.weight`
to the BF16 source's top-level `lm_head.weight`.

## Rule

For quantized hybrid Qwen3.5 checkpoints, dense fallback tensors are part of
the quality contract. A standard GPTQ W4 projection path can be kernel-clean
and still fail generation if the checkpoint mutates recurrent or
linear-attention dense tensors. Do not proceed to OPD or headline docs until
dense tensor parity or a tighter module-level forward parity gate licenses the
actual checkpoint.
