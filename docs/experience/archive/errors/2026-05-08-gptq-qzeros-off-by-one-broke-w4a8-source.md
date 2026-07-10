# GPTQ qzeros off-by-one broke the W4A8 source checkpoint

## Context

W4A8 GPTQ end-to-end validation failed with token-id-0 garbage:

```text
BF16: " Paris. The capital of Germany is Berlin..."
W4A8: "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
W4A8 vs BF16: matched first 0/32 tokens, diff 100.0%
```

The first hypotheses focused on `marlin_w4a8_kernel.cu`, `s_group`
range, and GPTQ-aware W4A8 scale layout. Those were reasonable but
wrong. Direct single-layer PR #31 kernel checks matched an independent
PyTorch reference at roughly 0.4-0.55% relative error across q/k/v/o and
MLP projections, so the W4A8 kernel could read GPTQ-aware packed tensors.

## Root Cause

`scripts/convert_gptq.py` decoded GPTQ `qzeros` without the AutoGPTQ
offset:

```python
zeros_unpacked = z_expanded.reshape(num_groups, N)
```

For this checkpoint, `model.layers.0.self_attn.q_proj.qzeros` unpacked
to stored value 7. GPTQ/AutoGPTQ-family checkpoints store zero-points as
`zero - 1`, so stored 7 means the real zero-point is 8. The converter
used 7 directly, shifting every converted W4A16 weight by one group scale.

That bad W4A16 source then fed the W4A8 repack. W4A8 inherited garbage
source weights; it was not the primary bug.

## Fix

Decode real zero-points by adding one during GPTQ conversion:

```python
zeros_unpacked = z_expanded.reshape(num_groups, N) + 1
```

Validation used new gitignored checkpoints:

```bash
.venv/bin/python scripts/convert_gptq.py \
  infer/models/Qwen3-4B-GPTQ-Int4 \
  --output infer/models/Qwen3-4B-GPTQ-Int4-converted-zpfix

.venv/bin/python scripts/convert_gptq_w4a16_to_w4a8_marlin.py \
  --src infer/models/Qwen3-4B-GPTQ-Int4-converted-zpfix \
  --dst infer/models/Qwen3-4B-GPTQ-W4A8-zpfix
```

Results:

| Gate | Result |
|---|---|
| Corrected W4A16 source, solo vs concurrent | PASS, coherent English, identical 30-token continuation |
| Corrected W4A8 vs BF16 | PASS, 32/32 tokens matched, diff 0.0% |

Commands:

```bash
INFER_TEST_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-GPTQ-Int4-converted-zpfix \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda --test greedy_consistency \
  test_greedy_solo_vs_concurrent -- --nocapture

INFER_TEST_W4A8_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-GPTQ-W4A8-zpfix \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda --test greedy_consistency \
  test_w4a8_vs_bf16_token_diff -- --nocapture
```

## Rule

Before debugging a downstream quantized kernel, validate the immediate
source checkpoint through an end-to-end quality smoke. For GPTQ/AutoGPTQ
converters specifically: packed `qzeros` represent `(zero - 1)`, not the
real zero-point.

`INFER_TEST_W4A8_MODEL_PATH` is the correct override for the W4A8 side of
`test_w4a8_vs_bf16_token_diff`; `INFER_TEST_MODEL_PATH` only changes the
BF16 baseline path in that test.
