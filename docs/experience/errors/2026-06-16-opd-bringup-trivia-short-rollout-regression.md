# OPD bring-up regression was a setup bug, not an OPD verdict

## Context

The first H20 teacher-student OPD capability bring-up used a real 4B teacher and
0.8B student, but trained on `examples/opd/opd-diverse-1k.jsonl` with
`--rollout-len 8` and `--lora-target-set attention-qv --lora-rank 16`. The
mechanics probes were healthy: teacher and student logits differed, raw KL was
substantial, and a single OPD step moved student logits toward the teacher.

The capability curve still regressed:

| Arm | MMLU delta | GSM8K delta |
|---|---:|---:|
| forward step500 | -1.19 pp | -5.00 pp |
| reverse step500 | -0.13 pp | -5.00 pp |
| KL-temp=2 step500 | -2.25 pp | -9.00 pp |

Those numbers are a setup failure, not evidence that OPD is marginal.

## Root Cause

The bring-up recipe trained the wrong behavior for the eval target:

- The prompt corpus was short trivia QA. Completions were usually 2-10 token
  direct answers, for example "Birds spread seeds of what?" -> "oaks".
- `--rollout-len 8` let the student generate only a short answer prefix.
- Q/V-only LoRA rank 16 was too little capacity for transferring reasoning or
  new knowledge.

On-policy distillation pulls the student toward the teacher on the student's own
rollout distribution. Here that distribution was short direct-answer trivia, so
training encouraged concise answer behavior and suppressed the long chain of
thought needed by GSM8K. The observed pattern matched that failure mode:
step250 had a noisy MMLU spike, step500 regressed, and GSM8K worsened as
training continued.

The ceiling check confirms there was signal to transfer: on the same
MMLU/GSM8K `n=100 seed=0` eval, Qwen3.5-4B scored MMLU 0.782 and GSM8K 0.790
versus the 0.8B base at MMLU 0.532 and GSM8K 0.390.

## Fix

The corrected bring-up arm is:

- GSM8K train questions only as prompts (`Q: ...\nA:`), letting the student roll
  out its own reasoning chain.
- `--rollout-len 256`, long enough for the teacher to score full reasoning.
- `--lora-target-set all-linear --lora-rank 32 --lora-alpha 64`, including MLP
  capacity.
- `--kl-mask completion`, temperature sampling, cosine LR, and servable
  checkpoint saves.

The first all-linear/rank32 one-step probe passed, including checkpoint save.
The long corrected run is queued behind an unrelated DSv4 serve occupying all
8 H20 GPUs, so the final direction remains pending.

## Rule

Capability distillation bring-up must match the target capability distribution.
For GSM8K, short trivia prompts plus 8-token rollouts train short-answer style,
not multi-step reasoning. Before judging an OPD recipe, first validate teacher
ceiling, rollout length, corpus distribution, and adapter capacity against the
campaign metric.
