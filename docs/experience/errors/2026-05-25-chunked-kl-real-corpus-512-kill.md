# Chunked KL Did Not Unblock Real-Corpus 512-Token GKD

Related:
`docs/projects/2026-05-24-opd-mainline-task-backlog.md` T5b,
`docs/research/2026-05-24-bf16-frozen-base-impl-path.md`,
`docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md`, and
`docs/experience/wins/2026-05-25-chunked-logits-kl-code-patch.md`.

## Context

T5a added `kl_distill_loss_chunked(...)` and CPU parity tests, but deliberately
did not switch production OPD/GKD callsites. T5b was the first GPU acceptance
run after P5 freed the 16 GB RTX 4070 Ti SUPER:

- real MMLU+GSM8K SFT-anchor corpus:
  `examples/opd/sft-anchor-mmlu-gsm8k.jsonl`
- `--prompt-max-tokens 512`
- `--rollout-len 8`
- `--gkd-lambda 0.3`
- `--sft-anchor corpus-truth`
- teacher: Qwen3.5-4B
- student: Qwen3.5-0.8B-Base LoRA

The code wiring commit (`291ec53`) added a default-off `--kl-chunk-size N`
knob to the OPD example and `GkdLossConfig`, then validated:

```bash
cargo test -p train --lib
NVCC_CCBIN=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo check -p train --features cuda --example opd_step_cuda_infer_teacher_train
```

Both checks passed before the GPU acceptance attempts.

## Evidence

Three sequential GPU controls changed only `--kl-chunk-size`:

| run | chunk size | artifact | verdict |
| --- | ---: | --- | --- |
| c64 | 64 | `bench-output/2026-05-25-t5b-gkd-real-corpus-512-chunked-kl/run.txt` | KILL |
| c8 | 8 | `bench-output/2026-05-25-t5b-gkd-real-corpus-512-chunked-kl-c8/run.txt` | KILL |
| c1 | 1 | `bench-output/2026-05-25-t5b-gkd-real-corpus-512-chunked-kl-c1/run.txt` | KILL |

All three runs loaded teacher + student and then failed before
`eval_summary step=0` and before any `train_step` line:

```text
model_summary teacher_source=infer student_hidden=1024 student_layers=24
student_vocab=248320 student_model_elements=769809216
student_trainable_elements=638976 ...
Error: TapeInvariant("cuda alloc_zeros failed (slice)")
```

GPU memory returned to browser-only residual after each failed run
(`nvidia-smi`: 1093 MiB used, 0% utilization), so the failures were clean
process-local allocation failures, not leaked GPU state from P5 or T14.

## Root Cause

Chunking the KL expression alone is not sufficient for this end-to-end shape.
The current callsites still full-materialize teacher and student logits before
the chunked loss receives them:

- `crates/train/examples/opd_step_cuda_infer_teacher_train.rs:994-995`
  computes full per-prompt teacher logits and student logits in eval.
- `crates/train/src/opd.rs:1097-1124` computes full rollout teacher logits
  and full rollout student logits before calling `kl_distill_loss_for_config`.
- `crates/train/src/loss.rs:109-110` then slices those existing full logits.

The c1 control is the important kill signal: even a one-token slice still fails
at `cuda alloc_zeros failed (slice)`. That means reducing KL chunk size does
not remove the dominant live allocation pressure at this stage. It only shrinks
post-forward KL intermediates after the large `[B, S, V]` logits already exist.

This also confirms the caveat in the T5a wins entry: the synthetic 8x memory
number covered KL intermediates only, not full end-to-end teacher/student
logit residency.

## Fix

No further code was attempted in T5b. Extending this into true streaming logits
would cross the task boundary: the teacher and student forward APIs need a
sequence-windowed logits path, or the acceptance has to move to mitigation 1
(`--prompt-max-tokens 256`) to cheaply test GKD behavior on this 16 GB GPU.

T5b verdict: **KILL** for the 512-token real-corpus acceptance gate.

## Rule

Do not claim "chunked KL fixes logits OOM" unless the callsite avoids full
`[B, S, V]` teacher/student logits before the loss. A chunked loss over already
materialized logits is only a partial mitigation.

Next licensed branches:

- Cheap experiment: rerun real-corpus GKD at `--prompt-max-tokens 256` to test
  whether GKD+corpus-truth has a useful signal on consumer 16 GB.
- Structural fix: add true sequence-windowed teacher/student logits and score
  KL per window without retaining full `[B, S, V]` logits.
