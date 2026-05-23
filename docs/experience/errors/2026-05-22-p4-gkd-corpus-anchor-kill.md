# P4 Corpus-Truth GKD Anchor Made The 2k Capability Valley Worse

## Context

P3 tested GKD lambda mixing with a student-rollout SFT anchor and failed the
2k capability gate. The next hypothesis was that the anchor was wrong:
student rollouts are noisy labels, so SFT on those labels may reinforce the
behavior OPD is trying to repair.

This tranche changed the GKD SFT anchor to corpus-truth completion tokens from
`examples/opd/sample-prompts.jsonl`:

```text
loss = lambda * corpus_sft_loss + (1 - lambda) * opd_kl_loss
lambda = 0.3
lr = 2e-5
```

The implementation added completion/target parsing to the prompt loader,
`--sft-anchor student-rollout|corpus-truth`, and a corpus-truth CE path that
forwards the student over `prompt + completion` and uses completion positions
as the hard-label SFT target. CE is scaled by `1 / vocab_size` to keep the
term on the same normalization scale as `kl_distill_loss`.

## Run

Command shape matched P3 except for `--sft-anchor corpus-truth` and the
prompt file now carrying curated `completion` fields:

```bash
cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --steps 2000 --rollout-len 8 --lr 2e-5 \
  --eval-steps 0,500,1000,2000 \
  --prompt-max-tokens 16 --max-step-seconds 30 \
  --save-student-checkpoint runs/2026-05-22-p4-distill-gkd-corpus-anchor \
  --save-every 500 \
  --gkd-lambda 0.3 \
  --sft-anchor corpus-truth
```

Artifacts:

- Train log: `bench-output/2026-05-22-p4-distill-gkd-corpus-anchor/run.txt`
- Capability eval:
  - `bench-output/2026-05-22-capability-gkd-corpus-anchor-step001000/`
  - `bench-output/2026-05-22-capability-gkd-corpus-anchor-step002000/`
- Comparison table:
  `bench-output/2026-05-22-capability-gkd-corpus-anchor-compare.md`

The run completed without OOM, NaN, or step-time guard failures. It wrote
valid LoRA adapters at `step_001000`, `step_002000`, and `final`.

## Evidence

### KL

| Step | train_kl | heldout_kl |
| ---: | ---: | ---: |
| 0 | 1.510384544190e-5 | 1.739055323924e-5 |
| 500 | 1.437269963844e-5 | 1.646543455536e-5 |
| 1000 | 1.410800621215e-5 | 1.694407205832e-5 |
| 2000 | 1.373122518089e-5 | 1.727154153741e-5 |

Train KL kept improving, but held-out KL followed the same warning pattern as
P3: early improvement followed by rebound toward the initial value.

### Capability

| Label | GSM8K | MMLU |
| --- | ---: | ---: |
| base 0.8B | 1.5% (3/194, inv 6) | 51.4% (73/142, inv 29) |
| pure OPD lr=2e-5 step2000 | 1.6% (3/188, inv 12) | 50.0% (83/166, inv 5) |
| GKD student-rollout step2000 | 1.6% (3/191, inv 9) | 47.0% (77/164, inv 7) |
| GKD corpus-truth step1000 | 2.0% (4/199, inv 1) | 45.0% (76/169, inv 2) |
| GKD corpus-truth step2000 | 1.0% (2/191, inv 9) | 41.0% (64/156, inv 15) |
| teacher 4B | 2.5% (5/198, inv 2) | 77.3% (116/150, inv 21) |

Gate result:

- `GKD corpus step2000` is below base by 10.38pp MMLU.
- It is also below pure OPD step2000 by 9.0pp MMLU.
- It is below the P3 student-rollout GKD endpoint by 6.0pp MMLU.

That fails both predeclared PASS conditions. This is a KILL.

## Root Cause

The corpus-truth anchor is active, but it is not a stabilizer for this 2k
setup.

What is evidenced:

- The code path runs end-to-end and writes loadable LoRA checkpoints.
- Invalid rates stayed low enough for the capability numbers to be usable.
- The training proxy improved on train KL, but MMLU collapsed harder than both
  pure OPD and student-rollout GKD.

What is still a hypothesis:

- The curated completions in `examples/opd/sample-prompts.jsonl` are likely
  too small and too synthetic to preserve MMLU capability. They provide a
  hard-token anchor, but not one aligned with the capability eval
  distribution.
- Lambda=0.3 may still be too weak or too strong for a real corpus, but this
  run rules out "just switch the P3 anchor to the current corpus-truth file"
  as a 2k fix.

## Rule

Do not claim GKD lambda mixing is fixed by replacing the student-rollout anchor
with the current prompt-file corpus completions.

The next SOLID branches are:

1. graduate to a longer-horizon run with the best pure-OPD or lr-sweep
   checkpoint, because every 2k GKD variant tested so far failed MMLU; or
2. build a real SFT anchor corpus that matches the eval distribution before
   retesting GKD lambda mixing.

Small hand-curated completions are not enough evidence for a capability
stabilizer claim.
