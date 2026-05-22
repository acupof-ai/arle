# P2 lr=1e-5 Sweep Did Not Fix the OPD Capability Valley

## Context

P1-B established the first closed OPD loop for Qwen3.5-4B teacher ->
Qwen3.5-0.8B-Base LoRA student. At `lr=2e-5`, the 2k-step run showed
the expected OPD U-curve:

- base 0.8B MMLU: 51.4%
- step 1000: 47.9% (-3.48pp valley)
- step 2000: 50.0% (-1.41pp, partial recovery)
- teacher 4B MMLU: 77.3%

This tranche tested the cheapest root-cause hypothesis: if the valley was
mostly learning-rate driven, halving LR to `1e-5` should make the valley
shallower and catch up by 2k steps.

Run shape matched P1-B except for LR:

- teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B`
- student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- prompts: `examples/opd/sample-prompts.jsonl`
- steps: 2000
- rollout_len: 8
- prompt_max_tokens: 16
- eval_steps: 0,500,1000,2000
- lr: `1e-5`
- checkpoints: `runs/2026-05-22-p2-distill-lr1e5/`

Artifacts:

- train log: `bench-output/2026-05-22-p2-distill-lr1e5/run.txt`
- step1000 eval: `bench-output/2026-05-22-capability-after-distill-lr1e5-step1000/`
- step2000 eval: `bench-output/2026-05-22-capability-after-distill-lr1e5-step2000/`
- comparison table: `bench-output/2026-05-22-capability-after-distill-lr1e5-compare.md`

## Evidence

Training completed without OOM, NaN, or step-time guard failures.

| LR | Step | Train KL | Held-out KL | MMLU |
| --- | ---: | ---: | ---: | ---: |
| 2e-5 | 0 | 1.510384544190e-5 | 1.739055323924e-5 | 51.4% |
| 2e-5 | 500 | 1.406700874895e-5 | 1.606478099347e-5 | not measured |
| 2e-5 | 1000 | 1.357229820087e-5 | 1.597982964086e-5 | 47.9% |
| 2e-5 | 2000 | 1.317703839732e-5 | 1.598908033884e-5 | 50.0% |
| 1e-5 | 0 | 1.510384544190e-5 | 1.739055323924e-5 | 51.4% |
| 1e-5 | 500 | 1.445267150757e-5 | 1.635090598029e-5 | not measured |
| 1e-5 | 1000 | 1.404491410995e-5 | 1.606972091395e-5 | 50.6% |
| 1e-5 | 2000 | 1.356088523607e-5 | 1.598121707502e-5 | 48.5% |

Capability comparison:

| Label | MMLU |
| --- | ---: |
| base 0.8B | 51.4% (73/142, invalid 29) |
| lr=2e-5 step1000 | 47.9% (81/169, invalid 2) |
| lr=2e-5 step2000 | 50.0% (83/166, invalid 5) |
| lr=1e-5 step1000 | 50.6% (86/170, invalid 1) |
| lr=1e-5 step2000 | 48.5% (82/169, invalid 2) |
| teacher 4B | 77.3% (116/150, invalid 21) |

The lower LR did make the step1000 valley much shallower:

- `lr=2e-5 step1000`: -3.48pp vs base
- `lr=1e-5 step1000`: -0.82pp vs base

But it did not recover by step2000:

- `lr=2e-5 step2000`: 50.0%
- `lr=1e-5 step2000`: 48.5%

That fails the predeclared license gate: `lr=1e-5 step2000` is worse than
the `lr=2e-5 step2000` baseline.

## Root Cause

The OPD valley is not fixed by simply halving LR.

The evidence supports a narrower conclusion:

- LR changes valley shape: lower LR reduces the early step1000 drop.
- LR alone does not determine 2k recovery: lower LR gave worse step2000 MMLU
  even though held-out KL continued to improve.
- KL is still an incomplete proxy for MMLU during the recovery phase. At
  `lr=1e-5`, held-out KL at step2000 was slightly better than `lr=2e-5`,
  while MMLU was worse.

This means the valley is at least partly a capability-dynamics issue, not a
pure KL minimization issue.

## Rule

Do not treat lower LR as the OPD valley fix.

Next tranche should isolate one of these two variables:

1. **Longer horizon at lr=1e-5**: continue the same checkpoint to 4k or 6k
   steps and eval at 3k/4k/6k. This tests whether the lower LR needs a longer
   recovery horizon.
2. **GKD lambda mixing**: add a controlled lambda mix between teacher KL and
   the student's original next-token distribution. This directly targets the
   capability valley where KL improves while MMLU lags.

The cheaper follow-up is horizon extension from the existing
`runs/2026-05-22-p2-distill-lr1e5/final` checkpoint. The more diagnostic
algorithmic follow-up is GKD lambda mixing.
