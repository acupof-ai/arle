# Qwen3.5-9B GPTQModel -> 0.8B OPD Memory Kill

## Context

After the 1D BF16 dense-load fix, the DavidWen2025
`Qwen3.5-9B-GPTQ-4bit` checkpoint passed the single-token full-logits gate
against the BF16 source:

- top-64 dominant relerr: `0.1242236`
- top-64 RMSE/reference-RMS: `0.0428670`
- argmax: ARLE `11`, PyTorch BF16 `11`

That licensed a functional OPD bench attempt through `InferTeacher`:

```bash
INFER_EXPERIMENTAL_GPTQMODEL_W4=1 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --steps 100 \
  --rollout-len 4 \
  --lr 1e-5 \
  --eval-steps 0,25,50,100 \
  --prompt-max-tokens 16 \
  --no-cuda-graph
```

## Failure

The 100-step bench failed before `eval_summary step=0`:

```text
model_summary student_hidden=1024 student_layers=24 student_vocab=248320 \
student_model_elements=769809216 student_trainable_elements=638976 \
student_load_seconds=10.228212 infer_load_seconds=88.458470
Error: Autograd(TapeInvariant("cuda htod copy failed"))
```

A smaller control also failed in the same place:

```bash
cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --steps 1 \
  --rollout-len 1 \
  --lr 1e-5 \
  --eval-steps 0,1 \
  --prompt-max-tokens 1 \
  --no-cuda-graph
```

Control failure:

```text
model_summary student_hidden=1024 student_layers=24 student_vocab=248320 \
student_model_elements=769809216 student_trainable_elements=638976 \
student_load_seconds=9.047388 infer_load_seconds=77.055216
Error: Autograd(TapeInvariant("cuda htod copy failed"))
```

Observed live while the minimal control was still running:

```text
GPU memory used: 14399 MiB / 16376 MiB
GPU memory free: 1545 MiB
```

Follow-up upload diagnostics changed the generic autograd error to include the
failed H2D tensor shape, bytes, and CUDA driver error. Re-running the same
single-token control failed on a small projection-weight upload:

```text
Error: Autograd(TapeInvariant("cuda htod copy failed: shape=[1024, 3584] \
len=3670016 bytes=14680064 err=DriverError(CUDA_ERROR_OUT_OF_MEMORY, \
\"out of memory\")"))
```

That is only `14.68 MB`, so the failure is not one unusually large tensor. The
GPU is already effectively out of contiguous allocation headroom before the
student's first full forward can finish uploading its f32 base weights.

## Root Cause

This is not a prompt-length or rollout-length problem. The single-token,
rollout-1 control fails before the first eval result, after both models load.

Root cause: the current train-side LoRA student keeps the 0.8B base weights as
f32 autograd tensors and uploads them through the generic f32 CUDA path during
the first student forward. With the 9B GPTQModel teacher resident in `infer`,
the remaining 16 GB GPU headroom is exhausted before even a `14.68 MB`
student-projection upload can complete.

## Decision

KILL the 9B GPTQModel -> 0.8B LoRA OPD bench on the current train-side f32
student base. The teacher checkpoint is usable enough for raw inference and
first-token full-logits parity, but the full OPD path does not fit on this 16
GB card until the student base stops expanding to f32 in the train runtime.

## Rule

For cross-runtime OPD on 16 GB cards, quantizing/compressing only the teacher
is insufficient if the train-side LoRA student materializes its frozen base in
f32. The next license gate must be one of:

- BF16/frozen base weights for `load_qwen35_lora_from_hf_dir` and train-side
  matmul against frozen BF16 tensors;
- a train-side memory profile that prints the failed H2D tensor shape/bytes and
  proves another allocation is the true blocker;
- a smaller student or a lower-memory student-loader mode for the headline
  demo.

Do not run more 9B OPD benches with the current f32 LoRA base and expect a
different result; the single-token control already ruled that out.
