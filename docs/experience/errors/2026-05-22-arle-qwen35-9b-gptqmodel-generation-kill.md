# Qwen3.5-9B GPTQModel Loader Opens, Generation Quality KILL

## Context

The local checkpoint
`/home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit`
is complete and uses GPTQModel-style W4 tensors:

- `qweight`: `int32`, shape `[K / 8, N]`
- `scales`: `float16`, shape `[K / group_size, N]`
- `qzeros`: present, all symmetric `7` nibbles
- `g_idx`: present and identity-by-group for layer-0 probes

The previous loader gate failed before HTTP readiness because ARLE's generic
W4 loader interpreted `qweight.shape[1]` as packed row-major K.

## Change Under Test

I added an explicit GPTQModel W4 physical-layout import branch that converts
`qweight [K/8,N]` plus `scales [K/group,N]` into ARLE's internal W4A16
row-major layout, then repacks the Marlin prefill side buffer.

Because the generation-quality gate below failed, this branch is gated behind:

```text
INFER_EXPERIMENTAL_GPTQMODEL_W4=1
```

The default loader now fails closed with an actionable message instead of
silently serving a model that fails quality smoke.

## Command

```bash
RUST_LOG=info \
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

Raw artifacts:

```text
bench-output/2026-05-22-qwen35-9b-gptq-int4-loader/
```

## Result

The experimental loader branch reached HTTP readiness:

```text
Model loaded: elapsed_ms=80814, model_id=Qwen3___5-9B-GPTQ-4bit
GPU memory @ post_model_load: free=3.22 GB / total=16.72 GB
Server listening on 0.0.0.0:8123
```

But the 3-prompt greedy generation smoke failed quality. Each completion was
64 exclamation marks:

```text
Hello, world! Tell me a short story about a small robot.
=> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Explain on-policy distillation in two sentences.
=> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Write a Python function that returns the Fibonacci sequence up to n.
=> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

A one-token prompt also failed:

```text
Hello
=> !!!!!!!!!!!!!!!!
```

## Attribution So Far

Tensor-local Python dequant of layer-0 `mlp.gate_proj` against the BF16 source
shows the best zero-point convention is still `q - 8`:

| Formula | RMSE / source RMS |
| --- | ---: |
| `q - 7` | 41.45% |
| `q - 7.5` | 23.93% |
| `q - 8` | 13.88% |
| reversed nibble order, `q - 8` | 141.94% |

This confirms the original qzero convention and nibble order are plausible, but
it is not enough to license full-model inference. The remaining possibilities
are:

1. The DavidWen GPTQModel quantization is intrinsically too lossy for this
   Qwen3.5 hybrid architecture.
2. ARLE's W4A16/Marlin path still diverges from the GPTQModel runtime path at
   layer-local matmul or at a dense fallback module.
3. There is a model-path issue outside W4 tensors, for example linear-attention
   dense fallback, embeddings, or output projection under this specific
   checkpoint.

## Fix

Do not run OPD, headline-switch, or user-facing README updates on this
checkpoint yet.

The loader branch remains available only behind
`INFER_EXPERIMENTAL_GPTQMODEL_W4=1` for reproducible investigation.

Next TODO:

1. Install or repair a PyTorch/GPTQModel reference environment for this exact
   checkpoint.
2. Compare one layer-0 projection output:
   ARLE W4A16 GEMV vs GPTQModel/PyTorch reference on the same hidden vector.
3. If projection parity passes, scan dense fallback modules:
   embedding, linear-attention dense tensors, final norm, and untied lm_head.
4. Only after generation is coherent, rerun the 9B-GPTQ -> 0.8B OPD bench.

## Rule

For quantized teacher checkpoints, `loads + HTTP ready` is still only a loader
smoke. The first user-facing license gate must include multi-token generation
coherence or full-model logits parity against the native quant runtime.
