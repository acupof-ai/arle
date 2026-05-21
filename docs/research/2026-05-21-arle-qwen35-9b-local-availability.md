# Qwen3.5-9B Local Availability Check

## Goal

Check whether any already-downloaded Qwen3.5-9B checkpoint on the 4070 Ti SUPER
box is currently usable as an ARLE CUDA teacher.

Raw artifacts:

```text
bench-output/2026-05-21-qwen35-9b-local-availability/
```

## Inventory

| Path | Size | Status |
| --- | ---: | --- |
| `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B` | 19 GB | Complete BF16 Base checkpoint; not serve-usable on this 16 GB GPU. |
| `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-Instruct` | 4 KB | Incomplete directory; no usable checkpoint files. |
| `/home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit` | 11 GB | Complete GPTQModel-format checkpoint; blocked by loader physical-layout mismatch. |
| `/home/ckl/.cache/modelscope/hub/RedHatAI/Qwen3___5-9B-FP8-dynamic` | 20 MB | Metadata/tokenizer only; full weights not downloaded because compressed-tensors layout is not supported by the current FP8 loader. |

## BF16 Base Serve Probe

Command:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
timeout 180s ./target/release/arle serve \
  --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --port 8123 \
  -- \
  --num-slots 1 \
  --max-seq-len 128 \
  --chunked-prefill-size 128 \
  --max-num-batched-tokens 128
```

Result:

```text
GPU memory @ pre_model_load: free=15.74 GB / total=16.72 GB
Memory-mapped 4 shard(s) (19306.3 MB) in 0ms
Failed to create scheduler: worker=0 cuda_ordinal=0 gpu=0:
H2D copy failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

No HTTP server was started, so no generation or OPD bench was run.

## Prior Loader Gates

The two compressed candidates are already blocked by loader gates:

- GPTQ-INT4:
  [`docs/experience/errors/2026-05-21-arle-qwen35-9b-gptq-int4-loader-kill.md`](../experience/errors/2026-05-21-arle-qwen35-9b-gptq-int4-loader-kill.md)
- FP8 compressed-tensors:
  [`docs/experience/errors/2026-05-21-arle-qwen35-9b-fp8-compressed-tensors-layout-kill.md`](../experience/errors/2026-05-21-arle-qwen35-9b-fp8-compressed-tensors-layout-kill.md)

## Conclusion

No currently downloaded 9B checkpoint is usable as an ARLE CUDA serve/OPD
teacher on this 16 GB GPU.

The complete BF16 Base checkpoint remains useful as a source/reference model
for offline quantization and parity checks, but it does not fit as a runtime
teacher. The usable runtime path today remains the validated Qwen3.5-4B BF16
teacher; a 9B runtime teacher needs one of these loader tranches first:

1. GPTQModel physical-layout loader for `[K/8, N]` `qweight` +
   `[K/group_size, N]` `scales`.
2. Compressed-tensors FP8 loader for `.weight_scale`.
3. A new ARLE-native quantized 9B artifact that passes tensor-local parity
   before full-model serve.
