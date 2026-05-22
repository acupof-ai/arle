# Qwen3.5 Serve Loads Saved LoRA Adapters

## Context

P1-A saved OPD student checkpoints as PEFT-style LoRA adapter directories, but
`arle serve` could only load `INFER_LORA_PATH` for the older Qwen3 path. Qwen3.5
serve rejected the train-produced adapter because there was no Qwen3.5 loader
or merge hook.

## What Worked

Added a Qwen3.5 PEFT adapter loader and `INFER_LORA_PATH` hook in CUDA bootstrap.
For P1-B the adapter is merged at load time into dense BF16 q/v projection
weights on Qwen3.5 full-attention layers. Linear-attention layers are left
untouched, matching the LoRA checkpoint emitted by the OPD save path.

The loader now:

- reads `adapter_config.json` and respects `target_modules`;
- parses PEFT safetensors keys with the Qwen3.5 `language_model.layers.*`
  prefix;
- supports F32/BF16 adapter tensors;
- merges `scale * B @ A` into dense BF16 q/v weights before serving.

Smoke command:

```bash
INFER_LORA_PATH=runs/2026-05-22-p1-save-smoke-08b-self-teach-bench/final \
./target/release/arle serve --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --port 8123 -- --num-slots 1 --max-seq-len 4096 \
  --chunked-prefill-size 4096 --max-num-batched-tokens 4096
```

Evidence:

- `infer::backend::cuda::bootstrap` logged the Qwen3.5 LoRA attach path.
- `infer::model::qwen35::lora` loaded 12 q/v adapters, 24 tensors, r=16,
  alpha=32.
- `infer::model::qwen35::weights` merged the adapter successfully.
- `/healthz` returned `{"status":"ok","service":"arle"}`.
- `/v1/completions` for `Hello, world!` returned normal text, not the previous
  long-prompt garbage/`!` failure shape.

Verification:

```bash
CARGO_BUILD_JOBS=1 cargo check -p infer --no-default-features --features cuda,no-cuda
NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
  CUDARC_CUDA_VERSION=13010 TORCH_CUDA_ARCH_LIST=8.9 CARGO_BUILD_JOBS=1 \
  cargo test -p infer --lib qwen35::lora --release --features cuda
```

Result: 4 Qwen3.5 LoRA unit tests passed. The infer build still emits
pre-existing DeepSeek/main warnings; those are unrelated to this tranche.

## Rule

Adapter checkpoint support needs a serve-side load gate before capability eval.
For Qwen3.5, the safe first implementation is load-time merge into dense BF16
full-attention q/v weights. Runtime adapter injection and quantized-base LoRA
are separate axes.
