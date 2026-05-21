# Qwen3.5-4B BF16 Teacher Track 1 KILL

## Context

Track 1 attempted to unblock the 4B -> 0.8B OPD README bench by loading the
Qwen3.5-4B teacher as frozen BF16 in the train/autograd runtime instead of
expanding it to f32. The memory hypothesis was correct: Qwen3.5-4B teacher +
Qwen3.5-0.8B LoRA student fits on the RTX 4070 Ti SUPER 16 GB card.

Safety run:

```text
cargo run -p train --example opd_step_cuda_realckpt_lora_bench --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --example-prompts-file examples/opd/sample-prompts.jsonl \
  --lr 1e-5 --steps 2 --rollout-len 8 --eval-steps 0,2 \
  --safety-first-step-max-seconds 0.5
```

## Evidence

Compile and unit gates passed before the safety run:

```text
cargo check -p train --example opd_step_cuda_realckpt_train --release --features cuda
cargo test -p autograd --test test_cuda_bf16_frozen --release --features cuda
```

Memory fit:

```text
nvidia-smi during run: 12258 MiB used, 3686 MiB free
model_summary teacher_dtype=bf16-frozen
teacher_param_elements=4222528512
student_model_elements=769809216
student_trainable_elements=638976
teacher_load_seconds=75.682262
student_load_seconds=10.116536
```

The wall-clock gate failed:

```text
eval_summary step=0 ... eval_seconds=626.695893
safety_stop first_step_seconds=8.515559 max_allowed_seconds=0.500000
Error: first OPD step took 8.515559s, exceeding the 0.500000s safety ceiling
```

## Root Cause

The BF16 loader fixes memory residency, not the binding runtime architecture.
The 4B teacher still runs through the train/autograd Qwen3.5 forward path, so
each OPD step pays a full 4B teacher forward in the training substrate rather
than using `infer`'s serving-optimized decode/prefill runtime.

This means Track 1 does not produce the README-quality 4B -> 0.8B bench. It
also makes the 200-step run impractical as a near-term headline number because
the eval loop alone took more than 10 minutes at step 0.

## Fix

Killed before committing the BF16 runtime code. The next viable route is Path B:

1. Add a raw-logits API on the loaded `infer` engine.
2. Bridge `cuda_kernels::DeviceVec` / raw logits into autograd without D2H.
3. Keep the 4B teacher in `infer`, where paged KV, CUDA graph decode, and model
   loading are already optimized.
4. Re-run the same 4B -> 0.8B LoRA bench only after the teacher path is
   infer-backed.

## Rule

For large-teacher OPD, memory fit is necessary but not sufficient. License the
teacher execution architecture with first-step wall-clock before running long
convergence or updating README numbers.
