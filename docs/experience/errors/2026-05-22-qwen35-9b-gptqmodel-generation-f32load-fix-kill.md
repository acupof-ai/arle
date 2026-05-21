# Qwen3.5-9B GPTQModel Generation Still Fails After F32-Load Fix

## Context

The DavidWen GPTQModel 4-bit checkpoint is locally complete and ARLE can serve
it behind `INFER_EXPERIMENTAL_GPTQMODEL_W4=1`. A loader bug was fixed in this
tranche: BF16 `linear_attn.A_log` / `linear_attn.norm.weight` tensors are now
converted to f32 instead of being reinterpreted as f32 bytes. That fixed the
layer-0 `linear_attention` NaN failure and brought the layer-local parity gate
to `4.0655%` RMSE/reference-RMS.

## Command

```bash
INFER_EXPERIMENTAL_GPTQMODEL_W4=1 \
RUST_LOG=info \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/arle serve --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --port 8123 -- \
  --num-slots 1 \
  --max-seq-len 256 \
  --chunked-prefill-size 128 \
  --max-num-batched-tokens 128
```

Then three non-streaming greedy `/v1/completions` requests:

```text
Hello, world! Tell me a short story about a small robot.
Explain on-policy distillation in two sentences.
Write a Python function that returns the Fibonacci sequence up to n.
```

Artifacts:

```text
bench-output/2026-05-22-qwen35-9b-gptqmodel-generation-after-f32load-fix/
```

## Result

All three completions were exactly 64 exclamation marks:

```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

The server loaded and stayed within memory:

```text
post_model_load free=3.19 GB / total=16.72 GB
```

## Root Cause

The previous layer-0 `linear_attention` NaN root cause is fixed, but it was not
the only quality blocker. The remaining failure is above a single layer-0 module
gate: either accumulated quantization sensitivity, a deeper hybrid
linear-attention divergence, full-attention drift, or a later-layer dense/quant
interaction.

## Fix

Do not switch the 9B GPTQModel checkpoint into headline docs and do not run OPD
with it yet. The next SOLID axis is a full-depth stage scan after the f32-load
fix:

1. Run layer-by-layer finite checks for every linear-attention block.
2. If all layers are finite, compare full-model dominant logits against the BF16
   source reference.
3. If full logits still drift, isolate first divergent layer output rather than
   trying another OPD bench.

## Rule

Passing a first-layer module parity gate is not enough to license a quantized
teacher. For 9B GPTQModel, the gate sequence is now:

1. Tensor/local loader dtype sanity.
2. Projection GEMV parity.
3. Every hybrid linear-attention layer finite.
4. Full-model dominant-logit parity.
5. Multi-token generation quality.
6. OPD KL trajectory.

