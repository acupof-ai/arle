# Qwen3.5-9B GPTQ-INT4 Loader Gate KILL

## Context

After the 9B-TQ4 generation-quality kill, we tried a standard community GPTQ
candidate as the cleaner 9B teacher path. The intended candidate in the brief
was `mssfj/Qwen3.5-9B-GPTQ-INT4`, but the active downloader on this host used
ModelScope repo `DavidWen2025/Qwen3.5-9B-GPTQ-4bit` and materialized:

```text
/home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit
```

Config summary:

- `quant_method=gptq`
- `bits=4`
- `group_size=128`
- `sym=true`
- `checkpoint_format=gptq`
- `tie_word_embeddings=false`

Raw artifacts:

```text
bench-output/2026-05-21-qwen35-9b-gptq-int4-generation/
```

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/arle serve \
  --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --port 8123 \
  -- \
  --num-slots 1 \
  --max-seq-len 256 \
  --chunked-prefill-size 128 \
  --max-num-batched-tokens 128
```

## Result

ARLE failed before HTTP readiness. No generation smoke or OPD bench was run.

Error:

```text
Failed to create scheduler: worker=0 cuda_ordinal=0 gpu=0:
model.language_model.layers.0.linear_attn.in_proj_qkv.weight:
quantized shape mismatch for bits=4: qweight cols=8192 implies K=16384,
but scales groups=8192 and group_size=128
```

Layer-0 tensor shapes:

```text
model.language_model.layers.0.linear_attn.in_proj_qkv.qweight (512, 8192) torch.int32
model.language_model.layers.0.linear_attn.in_proj_qkv.qzeros (32, 1024) torch.int32
model.language_model.layers.0.linear_attn.in_proj_qkv.scales (32, 8192) torch.float16
model.language_model.layers.0.mlp.gate_proj.qweight (512, 12288) torch.int32
model.language_model.layers.0.mlp.gate_proj.qzeros (32, 1536) torch.int32
model.language_model.layers.0.mlp.gate_proj.scales (32, 12288) torch.float16
model.language_model.lm_head.weight (248320, 4096) torch.bfloat16
```

## Root Cause

The checkpoint uses GPTQModel-style GPTQ tensors:

- `qweight` is `int32` with shape `[K / 8, N]`
- `scales` is `[K / group_size, N]`
- `qzeros` is present even though the top-level config says `sym=true`
- `g_idx` is present

ARLE's current generic quantized loader path is not reading that layout as a
GPTQModel tensor. It treats `qweight.shape[1]` as packed K columns for its
uniform row-major quant path. For layer 0 `in_proj_qkv`, that makes
`qw_cols=8192` imply `K=16384`, conflicting with
`scales_groups * group_size = 32 * 128 = 4096`.

So the earlier assumption "symmetric GPTQ is accepted directly" was too broad.
ARLE parses the metadata, but this specific GPTQModel checkpoint layout is not
loadable through the current Qwen3.5 weight path.

## Fix

Do not run the generation smoke, OPD bench, or headline switch on this
candidate.

The next viable axis is a loader tranche, not a quality tranche:

1. Add an explicit GPTQModel layout branch for `qweight [K/8, N] int32`,
   `scales [K/group_size, N]`, optional `qzeros`, and `g_idx`.
2. Decide whether the existing Marlin repack path can consume these tensors
   directly or whether ARLE should convert them to its internal row-major
   layout first.
3. Add a tensor-local parity gate against PyTorch/Transformers GPTQ output for
   one projection before full-model serve.

Only after the loader gate passes should the 3-prompt generation smoke be
rerun.

## Rule

Quant metadata acceptance is not loader acceptance. For GPTQ checkpoints, the
first gate must inspect actual qweight/scales/qzeros tensor shapes and confirm
the loader supports that physical layout before starting serve-quality tests.
