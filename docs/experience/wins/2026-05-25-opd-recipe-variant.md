# OPD Recipe Variant, T18

Status: running. Verdict pending.

Related: `docs/experience/wins/2026-05-25-p5-pure-opd-5k-capability-sweep.md`
and `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T18.

## Context

P5 pure OPD found a capability winner at step 2000, not at the heldout-KL
winner:

| checkpoint | train_kl | heldout_kl | MMLU | GSM8K |
| --- | ---: | ---: | ---: | ---: |
| base | n/a | 1.739e-5 | 51.41% | 1.55% |
| P5 step_001000 | 1.357e-5 | 1.598e-5 | 47.93% | 2.22% |
| P5 step_002000 | 1.318e-5 | 1.599e-5 | 50.00% | 1.60% |
| P5 step_003000 | 1.299e-5 | 1.603e-5 | 49.40% | 3.76% |
| P5 step_004000 | 1.288e-5 | 1.611e-5 | 45.78% | 2.73% |
| P5 step_005000 | 1.281e-5 | 1.618e-5 | 42.26% | 1.09% |

T18 tests whether a lower learning rate can reduce the late MMLU collapse
without changing the P5 workload shape.

## Variant Choice

Selected variant: **A, `--lr 1e-5`**.

Reason: the current training example exposes `--lr` but no
`--lr-decay-after-step` flag, so Variant A is the only no-code, single-knob
recipe change available for this run.

## Launch Command

```bash
mkdir -p bench-output/2026-05-25-t18-recipe-variant-lr1e-5
nvidia-smi > bench-output/2026-05-25-t18-recipe-variant-lr1e-5/nvidia-smi-before.txt
NVCC_CCBIN=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo run -p train --example opd_step_cuda_infer_teacher_train \
  --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --steps 5000 \
  --rollout-len 8 \
  --prompt-max-tokens 16 \
  --lr 1e-5 \
  --save-student-checkpoint runs/2026-05-25-t18-recipe-variant/lr1e-5 \
  --save-every 250 \
  --eval-steps 0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000 \
  2>&1 | tee bench-output/2026-05-25-t18-recipe-variant-lr1e-5/run.txt
nvidia-smi > bench-output/2026-05-25-t18-recipe-variant-lr1e-5/nvidia-smi-after.txt
```

Expected checkpoint roots:

```text
runs/2026-05-25-t18-recipe-variant/lr1e-5/step_000250
runs/2026-05-25-t18-recipe-variant/lr1e-5/step_000500
...
runs/2026-05-25-t18-recipe-variant/lr1e-5/step_005000
```

Capability eval will use the same direct-`infer` sequential sweep discipline
as T14, with `scripts/arle_capability_eval.py --tasks mmlu,gsm8k --n-samples
200`.

## License-Or-Kill

PASS:

- Any checkpoint reaches MMLU >= 51.41% base, or
- Any checkpoint is strictly above the P5 winner, MMLU > 50.00%.

KILL:

- No checkpoint beats 50.00% MMLU through step 5000.

## Results

Pending.

## Rule

OPD recipe changes need one isolated knob per run. LR-only is attributable;
adding schedule code or SFT/GKD anchors would make this T18 result unusable as
evidence for the low-LR axis.
