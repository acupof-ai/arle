# OPD on GSM8K at rollout-64 REGRESSED reasoning −10pp (truncated CoT actively hurts)

## Context

First OPD capability run on a **reasoning** task (the research-licensed
direction: big on-policy-distillation gains live on long-CoT reasoning, not
MMLU knowledge-MC). Config: Qwen3.5-4B teacher (W4-resident) → Qwen3.5-0.8B
student LoRA (rank 16, α 32, attention-qv), `examples/opd/gsm8k-train.jsonl`
(1000 GSM8K train prompts, CoT completions), 200 steps, lr 2e-5,
kl_chunk_size 16, completion-only mask, **rollout-len 64** (the only stable
length on 16GB — rollout≥128 OOMs, see
[`2026-05-29-opd-rollout128-train-crash.md`](2026-05-29-opd-rollout128-train-crash.md)).
Training: 200 steps, ~18.5s/step, KL −34% (9.4e-5 → 6.1e-5, normalized).

## Result — significant REGRESSION on GSM8K, borderline +2pp on MMLU

Multi-seed eval (5 seeds, n=200, 8-shot), trained-final vs the matched base
(`runs/2026-05-28-base-multiseed-eval`, same `Qwen3___5-0___8B-Base`):

| task | base mean | OPD mean | paired Δ | stat |
|---|---:|---:|---:|---|
| **GSM8K** | 34.3% | **24.0%** | **−10.28pp** [−11.95,−8.60] | t=−12.03; McNemar χ²≈44 (T-only 66 vs C-only 169), p≪0.001 |
| MMLU | ~50.5% | 51.8% | +1.7–2.1pp | per-seed t=+2.45 (sig); McNemar χ²=3.698 (just **below** 3.841, NOT sig) |

Every GSM8K seed worse (−8.9 to −13.5pp). This is not noise — it is a strong,
consistent regression on the reasoning task.

## Root cause — rollout-64 truncates the reasoning trace

GSM8K solutions need ~100–300 tokens of chain-of-thought to reach the `#### N`
answer. At **rollout-64 the student generates only ~64 tokens** — so its
on-policy rollouts are **truncated, incomplete reasoning**. The reverse-KL
trains the student to match the 4B teacher's token distribution **on chains
that never finish**, which teaches teacher-like *prefixes* while degrading the
student's ability to *complete* the reasoning to a correct answer → −10pp.

This **confirms the deep-research finding** (`2026-05-29` research): the
on-policy-distillation gain grows with rollout length (advantage +1.80→+3.55
Avg@k as rollouts go 128→4096 tokens); **short/truncated CoT is below the
threshold where it helps and into the regime where it actively hurts.** MMLU
(single-token answer, no chain to truncate) was untouched by the truncation and
got a tiny generic +2pp (borderline, reproduces the prior marginal MMLU result,
McNemar still <3.841).

## Implication / next step (NOT yet licensed)

This does **not** prove OPD can't work at this scale — it proves **rollout-64 is
the wrong regime for reasoning** (truncation). The genuine test is **rollout-256+
full-CoT**, which is VRAM-blocked on the 16GB card (3 co-resident models). The
licensed unblock is the **time-share offload** (offload the idle infer-student
1.6GB / teacher 3.1GB to CPU during the backward — verl/vLLM-sleep pattern), a
scoped infer-engine offload API, cheaper than autograd gradient checkpointing.

Honest risk before investing in that: even at full CoT, the **capacity gate**
(Apple 2026 "Unmasking OPD": a 0.8B may not absorb a 4B teacher's reasoning
style) could keep it null/negative. And this run shows the setup *can* hurt.

## Rule

- **For reasoning distillation, rollout length must cover the full reasoning
  trace, or it backfires.** A rollout shorter than the task's CoT trains the
  student on incomplete reasoning and *degrades* the reasoning capability
  (here −10pp GSM8K). Don't run reasoning OPD at a rollout that truncates the
  answer — a "fits-in-memory" short rollout is worse than not training.
- **KL↓ ≠ accuracy↑.** Training KL dropped 34% while GSM8K accuracy dropped
  10pp. Always verify the capability metric, never infer it from the loss.
- The ultimate-metric verdict for OPD at this scale (0.8B student, short
  rollout) is **null-to-negative**; perf/VRAM wins are enablers, not effect.
