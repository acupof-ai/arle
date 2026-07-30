# Data-free MoE expert merge collapses Qwen3.6-35B-A3B to a repetition generator

Date: 2026-07-30 · Model: Qwen3.6-35B-A3B-FP8 (256 experts, top-8, 40 layers)

## Context

Goal: shrink the MoE by merging 256 routed experts → N via ISO-Merger (SVD
decomposition + Procrustes alignment + group averaging), data-free. Three merge
recipes tried at 4:1 (N=64): uniform (iso64), frequency-weighted (cal), routing
co-activation clustering (coact), plus a router-preserving variant (preserve256:
merge to 64 groups, write each group weight back to all 256 slots, keep the
original 256-way router).

## Root Cause

**4:1 expert averaging destroys the per-expert weight geometry.** Averaging 4
experts' `[gate,up,down]` into one produces a weight that no longer computes a
coherent transform. All variants generate repetition garbage, not text:

| model | prompt "月食是" | MMLU (n=100) |
|-------|------------------|--------------|
| teacher (256) | "当月球运行至地球阴影…月食可分为月偏食" ✅ | ~68% |
| iso64 | "月食？月食是月食？月食是月食？…" | 30% |
| cal | "食食，食食是食食，食食是食食…" | — |
| coact | " 11 21 21 21 21…" | — |
| preserve256 | "：212 212 212…" | 29% |

`preserve256` keeps the original router yet still collapses → **router repack is
not the cause; expert weight averaging is.** Recipe (ISO/freq/coact) only
changes the *shape* of the garbage.

**The MMLU ~30% was an eval artifact.** Multiple-choice scores read the first
A/B/C/D token's logits; a model that can no longer generate coherently still
lands 25-30% by chance. The score masked total collapse for several analysis
rounds — see feedback: inspect generation before trusting any benchmark.

## Recovery attempts (both failed)

- **Low-rank distillation** (iso64, `train opd`, all-linear LoRA rank 16, lr
  1e-5, forward-KL from the 256 teacher over HTTP): MMLU at step 10/40/50 =
  30/28/30% — flat within noise. Low-rank increments cannot undo the high-rank
  information loss of 4:1 averaging (cf. DSv4 layers are high-rank/orthogonal).
- **Higher rank (256) probe**: engine thread died under VRAM contention before a
  checkpoint; moot given the generation-collapse finding.

## Rule

Data-free MoE expert merging is dead for this architecture. **2:1 (iso128) also
collapses** — grammar partly returns but semantics are wrong and it still loops
("月食是一種…食物… it is a special food, it is a special food…"; "中国的首都是
()。`<think><think>`…"). So the failure is not a 4:1 cliff; even halving the
ratio produces an unusable model. No light fine-tune recovers 4:1. Storage-only
compression (256-logical / 64-physical remap) is output-bit-identical to
preserve256 → also collapsed, so it cannot rescue quality either. Before
declaring any compressed/quantized/merged model "weak", inspect open-ended
generation — an MCQ score cannot distinguish broken from weak.

## Status

CLOSED, negative. 4:1 (iso64/cal/coact/preserve256) + low-rank distill + 2:1
(iso128) all collapse to incoherent/looping generation. Data-free ISO expert
merge does not preserve capability at any ratio that saves meaningful memory.
Only remaining live path is training-in-the-loop compression (co-train the
merged experts from scratch of the distill signal), out of scope here.
