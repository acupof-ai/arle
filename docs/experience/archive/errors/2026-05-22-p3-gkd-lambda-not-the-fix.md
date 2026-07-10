# P3 GKD Lambda Mixing Did Not Remove The OPD Valley

## Context

P2 showed the 2k-step capability valley was not simply lr-driven:
`lr=2e-5` dipped hard at step 1000 and partially recovered by step
2000, while `lr=1e-5` made the valley shallower but regressed by step
2000. The next literature-backed probe was GKD lambda mixing:

```text
loss = lambda * SFT_proxy_loss + (1 - lambda) * OPD_KL_loss
lambda = 0.3
```

This tranche added `--gkd-lambda` to
`crates/train/examples/opd_step_cuda_infer_teacher_train.rs` and mixed a
hard-token next-token SFT proxy into OPD. The SFT proxy uses the
student's on-policy rollout labels and is scaled by `1 / vocab_size` to
match `kl_distill_loss`'s internal mean-over-positions-and-vocab
normalization, avoiding a CE term that overwhelms KL by roughly the vocab
size.

## Run

Command shape matched P1-B except for `--gkd-lambda 0.3`:

```bash
cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --steps 2000 --rollout-len 8 --lr 2e-5 \
  --eval-steps 0,500,1000,2000 \
  --prompt-max-tokens 16 --max-step-seconds 30 \
  --save-student-checkpoint runs/2026-05-22-p3-distill-gkd-lambda03 \
  --save-every 500 \
  --gkd-lambda 0.3
```

Artifacts:

- Train log: `bench-output/2026-05-22-p3-distill-gkd-lambda03/run.txt`
- Capability eval:
  - `bench-output/2026-05-22-capability-gkd-lambda03-step001000/`
  - `bench-output/2026-05-22-capability-gkd-lambda03-step002000/`
- Comparison table:
  `bench-output/2026-05-22-capability-gkd-lambda03-compare.md`

The train run completed without OOM or NaN. Mean step time was 5.399860s,
median was 5.528686s.

## Results

### KL

| Step | train_kl | heldout_kl |
| ---: | ---: | ---: |
| 0 | 1.510384544190e-5 | 1.739055323924e-5 |
| 500 | 1.418320249513e-5 | 1.618700571271e-5 |
| 1000 | 1.400358473802e-5 | 1.641301241762e-5 |
| 2000 | 1.418923363872e-5 | 1.714250402074e-5 |

KL improved early, but held-out KL rebounded by step 2000 and nearly
returned to the starting point.

### Capability

| Label | GSM8K | MMLU |
| --- | ---: | ---: |
| base 0.8B | 1.5% (3/194, inv 6) | 51.4% (73/142, inv 29) |
| pure OPD lr=2e-5 step1000 | missing | 47.9% (81/169, inv 2) |
| pure OPD lr=2e-5 step2000 | 1.6% (3/188, inv 12) | 50.0% (83/166, inv 5) |
| GKD lambda=0.3 step1000 | 2.6% (5/189, inv 11) | 48.2% (82/170, inv 1) |
| GKD lambda=0.3 step2000 | 1.6% (3/191, inv 9) | 47.0% (77/164, inv 7) |
| teacher 4B | 2.5% (5/198, inv 2) | 77.3% (116/150, inv 21) |

Gate result:

- Step 1000 MMLU barely improved over pure OPD: 48.2% vs 47.9%
  (+0.31pp), which is too small to call a meaningful valley fix.
- Step 2000 regressed below both pure OPD and base: 47.0% vs pure OPD
  50.0% and base 51.4%.
- GSM8K showed a short step-1000 bump, then returned to pure OPD level
  at step 2000.

## Root Cause

The lambda=0.3 hard-token SFT proxy is not the stabilizer needed for this
2k-step OPD setup. It changes the trajectory slightly, so the implementation
is active, but it does not materially shallow the MMLU valley and it worsens
the step-2000 endpoint.

The most likely reason is not an implementation crash or numerical issue:
the training loop completed, adapter save/load worked, invalid rates stayed
low, and losses stayed finite. The failure is algorithmic for this exact
mix: the hard-token SFT anchor on student rollouts does not preserve enough
base-model capability while OPD optimizes toward the 4B teacher distribution.

## Rule

Do not claim GKD lambda mixing as the OPD U-curve fix from lambda=0.3.
The next SOLID branch is either:

1. run a single-variable lambda sweep at `lambda=0.5` using the same harness,
   or
2. switch the SFT anchor from student-rollout hard labels to prompt/corpus
   ground-truth tokens, because the current proxy may be too weak or too
   on-policy to preserve base capability.

Any follow-up must keep the same 0/1000/2000 capability eval gate; KL-only
movement is not enough.
