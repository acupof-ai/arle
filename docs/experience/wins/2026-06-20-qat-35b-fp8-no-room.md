# QAT-OPD 35B room-check — FP8 is already lossless vs bf16 (no QAT room)

## Context

Mainline goal: QAT/quant-aware distillation of the deployed
Qwen3.6-35B-A3B-**FP8** student from the **bf16** Qwen3.6-35B-A3B teacher
(same arch, same vocab → token-KL valid), to recover any FP8 quant loss.
Run on the H20 pod, GPUs 5/7 (4/6 were the running QAT eval gate).

The gate question (Step 2 "room check"): gap = bf16_MATH − FP8_MATH. FP8 was
already measured at 0.870 on MATH-100. Proceed to a full QAT-OPD run only if
the bf16 teacher clears FP8 by ≥ ~3pp (= the recoverable room). Otherwise the
35B *lift* needs a stronger teacher (DSv4-Flash), not same-family QAT.

## What Worked

**Read the running gate instead of re-running it.** The QAT gate
(`/data01/arle-opd-runs/qat_gate.log`) ran two independent in-house
MATH-100 evals (n=100, concurrency 4, max_tokens 4096, seed 0) — one against
the FP8 student (gpu4:8851), one against the bf16 teacher (gpu6:8852):

| Model | MATH-100 acc_valid | correct | elapsed |
|-------|--------------------|---------|---------|
| FP8 35B (deployed/quantized) | **0.870** | 87/100 | 1223 s |
| bf16 35B (teacher)           | **0.870** | 87/100 | 1481 s |

**Gap = 0.870 − 0.870 = 0.0 pp.**

The two evals are genuinely independent (distinct md5 of the per-question
dumps; distinct wall times 1223 vs 1481 s — not a copy artifact). Per-question
cross-check: 4 disagreements total, **symmetric** — bf16 wins idx {210, 306},
FP8 wins idx {103, 326}. The flips are decode jitter, not a systematic FP8
quant deficit. Net accuracy identical.

## Verdict — no meaningful QAT room

0.0 pp ≪ the ~3 pp threshold. FP8 Qwen3.6-35B-A3B is **already lossless**
against its own bf16 parent on MATH-100. There is nothing for same-family
QAT-OPD to recover. The full 50-step run was **NOT** burned (correctly, per
the room-check gate).

**Implication for the mainline:** the 35B *lift* (raising 35B above its own
ceiling, not just recovering quant loss) needs a **stronger teacher than the
same 35B in higher precision** — i.e. DSv4-Flash (8×H20) via InferTeacher,
not a bf16-vs-FP8 self-distill. Same-family precision-only distillation has no
signal here.

## Smoke (Step 1) — VRAM wall, single-process OPD cannot host the bf16 teacher

The bf16-teacher OPD smoke OOM'd; recorded for the next attempt's planning:

- **Single GPU (CUDA_VISIBLE_DEVICES=5):** OOM at `load infer teacher` —
  `MoE expert group alloc failed: CUDA_ERROR_OUT_OF_MEMORY`. The OPD process
  holds the **training student** (FP8 35B materialized, ~35 GB, single-GPU-locked
  for LoRA re-merge) resident; `ARLE_OPD_ENGINE_OFFLOAD=teacher` offloads only
  the *rollout* student before the teacher load. Peak = training-student (~35 GB)
  + bf16 teacher (~70 GB) ≈ 105 GB > 97 GB H20 — over by ~8 GB.
- **Teacher TP=2 (CUDA_VISIBLE_DEVICES=5,7, INFER_TP_SIZE=2):** not viable in the
  single-process `train opd` path — `multi-rank TP requires
  INFER_NCCL_UNIQUE_ID (128 hex bytes from the launcher's ncclGetUniqueId
  broadcast)`. The OPD process has no multiproc NCCL coordinator, and
  `student LoRA re-merge is currently single-GPU only`. So neither the teacher
  nor the student infer engine can TP-shard inside OPD.

Both walls are moot given the no-room verdict (full run not needed). If a
future bf16-teacher OPD *is* wanted, the fit requires either: (a) a multiproc
teacher coordinator so the teacher can TP-shard off-process, or (b) a
teacher-device placement flag (teacher on a dedicated GPU, student on another),
neither of which exists today.

## Rule

Same-family precision-only QAT distillation (FP8 student ← bf16 same-model
teacher) only has signal if the higher-precision parent measurably beats the
quantized child on the eval. Read the precision A/B FIRST (it's a cheap gate);
a 0-pp gap means the quant is lossless and the *lift* needs a stronger teacher
(cross-model, e.g. DSv4-Flash), not more precision of the same model.

Cross-link: [`2026-06-20-suffix-detach-lever.md`](2026-06-20-suffix-detach-lever.md)
(the backward-speed lever that would have powered the full run, had there been room).
