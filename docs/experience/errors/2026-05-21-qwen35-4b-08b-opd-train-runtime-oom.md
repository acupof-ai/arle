# Qwen3.5-4B -> 0.8B OPD train-runtime OOM

## Context

The requested headline bench was ARLE OPD with real ModelScope weights:

- Teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B`
- Student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Mode: LoRA r=16, `attention-qv`, `lr=1e-5`, `rollout_len=8`, 200 steps
- Prompts: `examples/opd/sample-prompts.jsonl`
- Artefacts: `bench-output/2026-05-21-qwen35-4b-08b-opd/`

The first attempt hit `InvalidConfig("rollout KV cache requires full-attention layers")`
because the 0.8B-Base checkpoint uses hybrid/linear-attention layers. Commit
`b0e9904` added a full-forward rollout/decode fallback for hybrid Qwen3.5 models.

## Evidence

After the fallback, the real run loaded both checkpoints and failed during step-0
eval:

```text
model_summary ... teacher_param_elements=4222528512 student_model_elements=769809216 student_trainable_elements=638976 teacher_load_seconds=85.107514 student_load_seconds=31.114663
Error: Autograd(TapeInvariant("cuda htod copy failed"))
```

`nvidia-smi` monitor from the same run:

```text
peak_used_mib=15902
```

The train/autograd `Qwen35Model` path stores weights as f32 tensors. The element
counts imply:

```text
teacher_f32=15.730 GiB
student_f32=2.868 GiB
teacher_plus_student_f32=18.598 GiB
lora_trainable_f32=0.002 GiB
```

On a 16 GiB RTX 4070 Ti SUPER, the frozen teacher alone nearly fills the card
when uploaded through the f32 autograd backend. The failure happens at a CUDA
host-to-device copy boundary, matching the observed monotonic memory climb to
the card ceiling.

## Root Cause

The current train-side OPD path is not the intended architecture for a 4B
teacher. It loads the frozen teacher into `TensorStore` as f32 autograd tensors,
then lazily uploads those f32 buffers during teacher forward. That is valid for
Qwen3-0.6B-class self-teach benches but not for a 4B frozen teacher on a 16 GiB
card.

This is not a LoRA adapter issue: LoRA trainable state is only about 0.002 GiB.
The binding memory is the frozen teacher representation.

## Fix

Do not use this train/autograd f32 teacher path for the headline 4B -> 0.8B
bench. The next viable axis is the planned infer-teacher bridge:

- run the 4B teacher through `infer` in BF16 or supported quantized format,
- return teacher logits to OPD without materializing f32 teacher weights in
  autograd,
- keep the 0.8B LoRA student in train/autograd.

An alternate but lower-leverage path is adding frozen BF16 weight support to the
train backend, but that duplicates runtime authority already owned by `infer`.

## Rule

For cross-size OPD, do a memory representation check before launching the bench:
`teacher_param_elements * dtype_bytes + student_base + optimizer_state` must fit
with activation headroom. A frozen large teacher must not use the f32 autograd
weight path on 16 GiB GPUs.
