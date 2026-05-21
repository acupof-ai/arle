# Qwen3.5-9B GPTQModel Live Recheck

## Context

The local ModelScope checkpoint is complete:

```text
/home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit
```

This recheck verifies the current `main` state after the GPTQModel W4 loader,
layer-local GEMV parity, dense tensor scan, API teacher, and multi-teacher
tranches. The loader remains gated behind `INFER_EXPERIMENTAL_GPTQMODEL_W4=1`.

## Command

```bash
RUST_LOG=info \
INFER_EXPERIMENTAL_GPTQMODEL_W4=1 \
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

Artifacts:

```text
bench-output/2026-05-22-qwen35-9b-gptqmodel-live-recheck/
```

## Result

Loader / serve smoke still passes:

```text
Model loaded: elapsed_ms=81784, model_id=Qwen3___5-9B-GPTQ-4bit
GPU memory @ post_model_load: free=3.19 GB / total=16.72 GB
Server listening on 0.0.0.0:8123
```

But multi-token greedy generation still fails quality. All three prompts return
64 exclamation marks:

| Prompt | Output |
| --- | --- |
| `Hello, world! Tell me a short story about a small robot.` | `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!` |
| `Explain on-policy distillation in two sentences.` | `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!` |
| `Write a Python function that returns the Fibonacci sequence up to n.` | `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!` |

## Decision

This checkpoint is **not licensed** as an OPD teacher or user-facing headline.
It is useful only as a reproducible loader/parity investigation target.

The next useful axis is not another serve smoke. It is module-local attribution
for the remaining full-model quality failure: dense fallback modules and
linear-attention state/norm behavior under this exact GPTQModel checkpoint.

## 2026-05-22 Serve Sanity After OPD Memory Kill

After the BF16-frozen-student OPD memory retry, the same local checkpoint was
rechecked as an inference-only server to keep the claims separated:

```bash
INFER_EXPERIMENTAL_GPTQMODEL_W4=1 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
RUST_LOG=info \
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

Artifacts:

```text
bench-output/2026-05-22-qwen35-9b-gptqmodel-serve-recheck-after-opd-kill-2/
```

Serve/load result:

```text
Model loaded: elapsed_ms=78813, model_id=Qwen3___5-9B-GPTQ-4bit
Server listening on 0.0.0.0:8123
```

One-token completion with the exact loaded model id succeeded:

```json
{
  "model": "Qwen3___5-9B-GPTQ-4bit",
  "choices": [{"text": "!", "finish_reason": "length"}],
  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
}
```

Operational note: `/v1/completions` rejects aliases not matching the loaded
model id. A request using `model=qwen35-9b-gptq` returned `model_not_found`;
using `model=Qwen3___5-9B-GPTQ-4bit` succeeded.

Decision unchanged: the checkpoint is inference-loadable and can decode, but
multi-token generation quality remains unlicensed from the earlier recheck.
Do not use this as a user-facing headline until the quality/parity issue is
separately fixed.
