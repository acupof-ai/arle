# ARLE Infer Raw Token Logits API

## Goal

Open Path B for OPD by letting `LoadedInferenceEngine` return device-resident
token logits directly from the infer runtime, without sampler or text decode.

## Hypothesis

A scheduler-side raw-logits control channel can reuse the existing Qwen forward
path and return `[seq_len, vocab_size]` BF16 logits as a CUDA `DeviceVec`.

## Params

- Model: ModelScope `Qwen/Qwen3-0.6B`
- Prompt token IDs: `[1, 3, 8]`
- Positions: `[0, 1, 2]`
- Backend: CUDA, RTX 4070 Ti SUPER
- Env: `NVCC_CCBIN=/usr/bin/g++-14`, `CUDARC_CUDA_VERSION=13010`,
  `TORCH_CUDA_ARCH_LIST=8.9`

## Results

- `cargo check -p infer --release --features cuda`: pass
- `cargo test -p infer --test forward_token_logits --release --features cuda`: pass
- Smoke assertion: returned shape is `[3, vocab_size]`, buffer length matches,
  and all downloaded logits are finite.

## Problems

- V1 supports contiguous positions starting at zero. Non-contiguous position
  control remains deferred until an OPD caller needs it.
- This is not a throughput claim; perf licensing starts when `InferTeacher`
  is wired into an OPD step.

## Learnings

The scheduler can return owned device logits through a side channel without
touching the normal completion request path. This gives Commit 2 a concrete
D2D bridge source.
