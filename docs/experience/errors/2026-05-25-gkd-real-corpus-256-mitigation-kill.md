# GKD Real Corpus 256-Token Mitigation KILL

## Context

T15 tested the pre-licensed mitigation from
[`2026-05-25-chunked-kl-real-corpus-512-kill.md`](2026-05-25-chunked-kl-real-corpus-512-kill.md):
retry real-corpus GKD with a shorter prompt cap before changing the forward API.

Command shape:

```bash
RUST_BACKTRACE=1 target/release/examples/opd_step_cuda_infer_teacher_train \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sft-anchor-mmlu-gsm8k.jsonl \
  --steps 500 \
  --rollout-len 8 \
  --lr 2e-5 \
  --eval-steps 0,100,200,300,400,500 \
  --prompt-max-tokens 256 \
  --max-step-seconds 240 \
  --save-student-checkpoint runs/2026-05-25-t15-gkd-real-corpus-256 \
  --save-every 100 \
  --gkd-lambda 0.3 \
  --sft-anchor corpus-truth \
  --kl-chunk-size 8
```

Input corpus and split:

| Item | Value |
|---|---:|
| rows | 56 |
| train prompts | 52 |
| heldout prompts | 4 |
| truncated rows | 0 |
| completion rows | 56 |
| prompt cap | 256 tokens |
| rollout len | 8 |

GPU gate before launch:

| GPU | Used MiB | Total MiB | Util % |
|---|---:|---:|---:|
| RTX 4070 Ti SUPER | 1093 | 16376 | 0 |

## Root Cause

Verdict: **KILL (hardware)**.

The run failed before `eval_summary step=0` and before any `train_step` lines:

```text
model_summary teacher_source=infer student_hidden=1024 student_layers=24 student_vocab=248320 student_model_elements=769809216 student_trainable_elements=638976 student_load_seconds=8.424202 teacher_load_seconds=2.309157
Error: TapeInvariant("cuda alloc_zeros failed")
```

Evidence table:

| Gate | Result |
|---|---|
| reached `eval_summary step=0` | no |
| `train_step` count | 0 |
| checkpoint files written | none |
| failure | `TapeInvariant("cuda alloc_zeros failed")` |

This reproduces the 512-token failure mode at a lower prompt cap. The validated
fact is that prompt shortening to 256 is not enough to make real-corpus
GKD+corpus-truth fit on this 16 GB card with the current full-logit path.

The root-cause hypothesis remains the same as the 512-token KILL: the eval/train
KL path still materializes prompt logits at `[B, S, V]` before chunked KL can
reduce loss memory. With `V=248320`, `S=256` is still large enough that the
student/teacher/tape allocation stack crosses the available headroom after
loading the Qwen3.5-4B teacher and 0.8B student LoRA path.

## Fix

Do not keep reducing `--kl-chunk-size`; that only chunks the loss after logits
already exist. Do not treat `--prompt-max-tokens 256` as a usable mitigation.

Next licensed branch: implement a true sequence-windowed forward path so the
teacher/student logits are produced and consumed as `[B, window, V]` windows
rather than materializing the full prompt dimension first.

This T15 run did not measure heldout signal, train KL, or capability. It never
passed the hardware gate required to make those metrics meaningful.

## Rule

For real-corpus GKD on ARLE's current 16 GB CUDA target, chunked KL is not a
memory fix unless the forward API also becomes sequence-windowed. A mitigation
PASS must prove the run reaches `eval_summary step=0` and at least 10
`train_step` lines before any heldout-KL signal claim is allowed.
