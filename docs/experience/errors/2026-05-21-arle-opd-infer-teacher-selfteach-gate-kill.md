# InferTeacher OPD Self-Teach Gate Kill

## Context

Path B Commit 3 landed the scheduler-backed `InferTeacher` bridge:
`LoadedInferenceEngine::forward_token_logits` returns BF16 logits, and train
imports them through the autograd BF16 D2D bridge. The numerical gate passed on
Qwen3.5-0.8B-Base self-teach:

- Test: `cargo test -p train --test test_infer_teacher --release --features cuda -- --nocapture`
- Result: top-64 dominant logit relerr `1.0285222e-2`, below the relaxed
  `5e-2` BF16 gate.

The next gate was a first runnable OPD step with `InferTeacher` on
Qwen3.5-0.8B-Base self-teach, target `<= 0.4s`.

## Result

The wall-clock gate failed.

| Variant | Prompt / Rollout | CUDA graph | Step seconds | Gate |
| --- | --- | --- | ---: | --- |
| Full-finetune student + InferTeacher | `[9419]`, rollout 8 | off | `10.812417` | FAIL |
| LoRA r16 attention-qv student + InferTeacher | `[9419]`, rollout 8 | off | `3.304193` | FAIL |
| LoRA r16 attention-qv student + InferTeacher | `[9419]`, rollout 8 | on | `3.267453` | FAIL |

Ignored raw artefacts:

- `bench-output/2026-05-21-qwen35-08b-infer-teacher-selfteach/run.txt`
- `bench-output/2026-05-21-qwen35-08b-infer-teacher-selfteach/run-cudagraph.txt`

A non-matched in-process LoRA control using the existing real-checkpoint bench
also landed in multi-second territory:

- Command: `cargo run -p train --example opd_step_cuda_realckpt_lora_bench --release --features cuda -- --teacher-model <0.8B> --student-model <0.8B> --steps 1 --rollout-len 8 --lr 1e-5 --eval-steps 1 --safety-first-step-max-seconds 5`
- Result: `train_step step=1 ... step_seconds=4.143257`
- Caveat: this control uses the built-in prompt set and runs heavy eval, so it
  is not a matched attribution run. It only shows that Qwen3.5-0.8B train-side
  LoRA OPD is not currently near the earlier Qwen3-0.6B-class timing.

## Root Cause Status

Not licensed as root cause yet.

The strongest source-level hypothesis is that `forward_token_logits` v1 scores
the rollout by looping over tokens and calling `forward_with_logits(&[token])`
once per token inside `infer/src/scheduler/cuda/runtime/fetch.rs`. That makes
teacher scoring a serial decode loop for the full rollout. The CUDA graph A/B
did not materially change wall-clock (`3.304s -> 3.267s`), so graph launch
overhead is not the primary fix.

However, the current measurements do not yet isolate:

- rollout student forward vs infer teacher scoring,
- D2D BF16 import vs infer forward,
- student final forward/backward vs optimizer,
- Qwen3.5-0.8B shape effects: vocab `248320`, head_dim `256`, 24 layers.

Therefore the SOLID conclusion is limited to: Path B Commit 4 is blocked by a
multi-second 0.8B self-teach OPD step, and the exact dominant phase still needs
matched phase attribution.

## Fix

Do not advance to 4B-to-0.8B distillation yet.

Next tranche should be a matched phase profile for Qwen3.5-0.8B self-teach with
the same prompt and rollout length, reporting:

1. rollout student forward,
2. infer teacher raw-logits forward,
3. BF16 D2D import,
4. student KL forward,
5. backward,
6. grad clip,
7. optimizer.

License criterion for resuming Path B: after the dominant phase is fixed,
Qwen3.5-0.8B LoRA self-teach via `InferTeacher` must be `<= 0.4s` for
rollout_len 8, or the architecture remains blocked for today's README claim.

## Rule

Do not extrapolate Qwen3-0.6B OPD CUDA timings to Qwen3.5-0.8B. The 0.8B
checkpoint has a much larger vocab and different head geometry, and the
infer-teacher bridge adds a scheduler-backed raw-logits path that needs its own
phase attribution before it can be called viable.
