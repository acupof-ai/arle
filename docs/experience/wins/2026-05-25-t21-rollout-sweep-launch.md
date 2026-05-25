# T21 Rollout Sweep Launch

Status: variant A running. Verdict pending.

Related:
`docs/research/2026-05-25-opd-methodology-audit.md`,
`docs/experience/errors/2026-05-25-t20-corpus-diversity-aborted.md`, and
`docs/experience/wins/2026-05-25-t18-recipe-variant-result.md`.

## Context

The OPD methodology audit ranks `rollout_len=8` as the highest-impact method
gap versus TRL/GKD-style runs. T21 tests that axis directly: keep the P5/T18
20-prompt corpus and change only `--rollout-len`.

T20 corpus diversity was stopped before capability sweep so this run isolates
rollout horizon instead of mixing corpus and rollout variables.

## Variant A

Single variable changed from P5:

| field | P5/T18 | T21-A |
| --- | ---: | ---: |
| prompts | `examples/opd/sample-prompts.jsonl` | same |
| prompt max tokens | 16 | same |
| learning rate | 2e-5 | same |
| teacher | Qwen3.5-4B | same |
| student | Qwen3.5-0.8B-Base LoRA | same |
| rollout len | 8 | 32 |
| steps | 5000 / 3000 | 1500 |
| save every | 250 | 250 |

## Launch Command

The HEAD rebuild is currently blocked by local TileLang AOT environment drift,
so this launch uses the existing release example binary that passed the T20
sanity run. This is acceptable for T21 because the task is no-code and changes
only CLI arguments.

```bash
mkdir -p runs/2026-05-25-t21-rollout-32
RUST_BACKTRACE=1 target/release/examples/opd_step_cuda_infer_teacher_train \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --steps 1500 \
  --rollout-len 32 \
  --lr 2e-5 \
  --eval-steps 0,250,500,750,1000,1250,1500 \
  --prompt-max-tokens 16 \
  --max-step-seconds 240 \
  --save-student-checkpoint runs/2026-05-25-t21-rollout-32 \
  --save-every 250 \
  2>&1 | tee runs/2026-05-25-t21-rollout-32/run.txt
```

Expected checkpoints:

```text
runs/2026-05-25-t21-rollout-32/step_000250
runs/2026-05-25-t21-rollout-32/step_000500
runs/2026-05-25-t21-rollout-32/step_000750
runs/2026-05-25-t21-rollout-32/step_001000
runs/2026-05-25-t21-rollout-32/step_001250
runs/2026-05-25-t21-rollout-32/step_001500
```

## License-Or-Kill

Variant A PASS:

- Any checkpoint MMLU > 51.41% base.

Variant A PARTIAL:

- Any checkpoint MMLU > 50.59% T18 winner but still below 51.41% base.

Variant A KILL:

- No checkpoint beats 50.59% T18 winner.

Run variant B (`--rollout-len 64`) only if variant A is PASS or PARTIAL.

## Rule

The rollout sweep must keep the 20-prompt corpus to stay comparable with P5/T18.
If rollout improves capability, corpus diversity is not a necessary precondition
for recovery; if it does not, the next method gaps remain prompt masking,
temperature/sampling, and sequence-windowed anchored training.
