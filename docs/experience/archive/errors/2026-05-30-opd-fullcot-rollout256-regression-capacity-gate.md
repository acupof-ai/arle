# OPD full-CoT (rollout-256) ALSO regresses GSM8K −11pp — it's the capacity gate, NOT truncation

## Context

The definitive, unconfounded test of OPD-reasoning at ARLE's scale. The prior
rollout-64 run regressed GSM8K −10.3pp, hypothesized to be **truncation** (the
student generated only ~64 of the ~100–300 CoT tokens GSM8K needs). To remove
that confound we built infer-engine weight offload/time-share (commits
`1cfef624`, `88d3a172`) to fit **rollout-256 full CoT** on the 16GB card, and
ran the matched experiment: Qwen3.5-4B-W4A8 teacher → Qwen3.5-0.8B student LoRA
(r16, attention-qv), gsm8k-train.jsonl, 200 steps, lr 2e-5, kl_chunk 16,
completion-only, `ARLE_OPD_ENGINE_OFFLOAD=1`. Training: 200 steps clean, rollout
lengths **300–358 tokens (full reasoning traces, no truncation)**, KL −39%
(1.017e-4 → 6.19e-5).

## Result — full CoT regresses MORE, not less. Truncation hypothesis REFUTED.

Multi-seed GSM8K (5 seeds, n=200, 8-shot), trained-final vs matched base
(`runs/2026-05-28-base-multiseed-eval`, base 34.3%):

| config | GSM8K mean | paired Δ | McNemar |
|---|---:|---:|---|
| base | 34.3% | — | — |
| OPD rollout-64 (partial CoT) | 24.0% | −10.28pp [−11.95,−8.60] | χ²≈44 |
| **OPD rollout-256 (FULL CoT)** | **22.8%** | **−11.27pp** [−13.38,−9.17], t=−10.50 | **χ²=53.4** (T-only 61, C-only 174), p≪0.001 |

Full CoT is **−11.3pp**, slightly worse than the truncated −10.3pp. Every seed
regresses (−8.4 to −15.0pp). The regression is **not** caused by short rollouts.

## Root cause — capacity gate + reverse-KL collapse (not setup, not truncation)

KL dropped 39% (the student moved strongly toward the 4B teacher's token
distribution) while GSM8K accuracy fell 11pp. The student became *more like the
teacher's distribution and worse at solving problems*. Removing the truncation
confound leaves the **capacity gate** (Apple 2026 "Unmasking OPD": a student
that cannot comprehend/represent a much-stronger teacher's reasoning is
*degraded* by being pulled toward it) plus **reverse-KL entropy collapse** (the
deep-research-flagged risk of mode-collapse onto a narrower, lower-accuracy
distribution). The 0.8B base already solves GSM8K at a respectable 34.3%;
forcing its distribution toward a 4B it cannot match corrupts that ability.

This matches the literature exactly: the demonstrated big on-policy-distillation
gains are at **7B–32B students** (R1-distill +5–22 AIME); no source shows gains
into a sub-1B student. **0.8B ← 4B is below the capacity floor for reasoning
distillation.**

## Ultimate-metric verdict (conclusive)

**OPD does NOT improve the effect at ARLE's scale — it regresses reasoning
~11pp, at both CoT lengths, strongly significant.** The MMLU side is at best a
borderline +2pp (not significant). The perf/VRAM/infra wins this session (8×
step, −62% teacher VRAM via W4, the offload engine) are real *enablers* but the
method+scale do not deliver a capability gain; they conclusively deliver a
capability *loss* for reasoning. The honest path to a positive effect, if
pursued, is a **larger student** (≥1.5–7B, where distillation is known to
work), not more rollout length or more OPD tuning at 0.8B.

## Rule

- **KL↓ is not evidence of capability↑ — it can be capability↓.** Across two
  runs KL fell 34–39% while GSM8K accuracy fell 10–11pp. Distilling a small
  student toward a much-stronger teacher's distribution can *destroy* the
  student's own working behavior (capacity gate). Always measure the capability
  metric; never accept the training loss as a proxy.
- **Spend the experiment that removes the confound before concluding a
  mechanism.** The truncation hypothesis was plausible and testable; building
  offload to run full-CoT refuted it cleanly and pointed to the real cause
  (capacity). The infra cost bought a definitive answer, not a guess.
- **Match the teacher-student gap to the student's capacity floor.** Reasoning
  distillation gains are documented at 7–32B students; sub-1B students regress.
  Don't run OPD-reasoning below the capacity floor and expect a gain.
