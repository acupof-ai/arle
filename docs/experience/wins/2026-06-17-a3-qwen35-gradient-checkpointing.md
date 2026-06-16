# A3 Qwen35 Gradient Checkpointing

## Context

OPD rollout generation is already on the inference engine; the remaining
training-side memory lever is activation checkpointing for the autograd
student forward/backward. This tranche adds a real replay checkpoint op and
wires Qwen35 transformer layers behind an opt-in flag. Defaults stay unchanged.

## What Worked

- Added `autograd::ops::checkpoint`: forward runs the segment with tape disabled,
  frees segment-local temporaries, records one checkpoint entry, and backward
  replays the segment on an inner tape with the upstream gradient as the seed.
- Inner checkpoint backward is collect-only, so replay does not directly mutate
  parameter `.grad`; outer backward remains the single accumulation owner.
- Qwen35 full training forward can checkpoint each transformer layer with
  `Qwen35Model::set_gradient_checkpointing(true)`.
- HF loader opt-in: `ARLE_OPD_GRADIENT_CHECKPOINTING=1` enables checkpointing for
  trainable/LoRA students only. Frozen teachers are not checkpointed.

## Results

Local no-cuda gates:

```text
cargo test -p autograd --release --no-default-features --features no-cuda checkpoint_ -- --nocapture
2 passed

cargo test -p train --release --no-default-features --features no-cuda --lib qwen35::tests:: -- --nocapture
3 passed
qwen35_checkpoint_fd name=model.language_model.layers.1.self_attn.q_proj.weight.lora_b index=10 eps=1.0e-3 analytic=-1.475234479e-1 numeric=-1.476407051e-1 rel_err=7.942e-4
```

Remote CUDA gate on `.62` (`iv-ye8is8fbi8s6iplibbg7`), GPU4, GPU3 avoided:

```text
CUDA_VISIBLE_DEVICES=4 ARLE_CUDA_TEST_DEVICE=0 ARLE_CUDA_DISABLE_FLASHMLA=1 \
CUDARC_CUDA_VERSION=12090 CUDA_HOME=/usr/local/cuda \
INFER_TILELANG_PYTHON=/root/tl-venv/bin/python \
CARGO_TARGET_DIR=/data01/arle-target-a3-checkpoint-cuda-noflash \
cargo test -p train --release --no-default-features --features cuda --lib \
  qwen35_gradient_checkpointing_lora_cuda_finite_diff_gate -- --nocapture

qwen35_checkpoint_fd name=model.language_model.layers.1.self_attn.q_proj.weight.lora_b index=10 eps=1.0e-3 analytic=-1.475234330e-1 numeric=-1.476407051e-1 rel_err=7.943e-4
test qwen35::tests::qwen35_gradient_checkpointing_lora_cuda_finite_diff_gate ... ok
```

Additional local gates:

```text
cargo fmt --check
cargo clippy -p autograd --release --no-default-features --features no-cuda --lib -- -D warnings
cargo clippy -p train --release --no-default-features --features no-cuda --lib -- -D warnings
CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,nccl,no-cuda --lib
cargo test -p train --release --no-default-features --features no-cuda --lib opd::tests:: -- --nocapture
```

## Rule

Checkpointing is only licensed when the replay path is gradient-equivalent under
relative finite difference. A source-only "tape entries dropped" claim is not
enough; the CPU and CUDA LoRA element gates above are the correctness evidence.
