# Qwen3.5/3.6 batched decode stage 1 — the SGLang-comparison structural fix

**Date:** 2026-06-11. **Backend:** CUDA, Qwen3.6-35B-A3B, H20.
**Status: pending-remote** — pod c-sweep 1/2/4 (aggregate tok/s + per-stream
ITL + needle PASS at c=2). Today c≥2 KILLS the engine (single-row ensure),
so the baseline for the c>1 cells is "engine death", not a number.

## Context

The SGLang qwen3-next P/D comparison
(`docs/reviews/2026-06-11-sglang-qwen3next-pd-comparison.md`) pinned the gap
axis: SGLang's decode unit is the whole running batch; ARLE qwen35 was
single-row-per-tick. This tranche re-ports the deleted monolith's proven
packed-batch design (`e81b98fb~1` infer/src/model/qwen35/batch_decode.rs)
using DSv4's fresh executor pattern (`Dsv4DecodeBatch`).

## What Worked (implementation; numbers pending)

- B rows pack as `seq_len = B`: embedding/all GEMMs/norms/MoE (R=8B routes,
  still hand-kernel below the DeepGEMM 1024 floor)/residuals/lm_head/argmax
  run batched exactly like a B-token prefill chunk. Per-row only where state
  is per-slot: full attention (offset-pointer per-row launches over the
  slot's contiguous cache — better than DSv4's copy-in/out, verified the
  HD256 kernels index `token*dim + …` so column offsets work), conv1d + GDR
  via the revived in-tree batch kernels with per-layer `[B]` device pointer
  tables (restaged only when the row→slot mapping changes).
- `gdr_decode_batch_cuda` carried the SAME warp_norms race the single kernel
  was fixed for (fix never propagated) — barrier + dim guard added before
  reviving it.
- all-reduce message-length proof: every reduced buffer is an exact-shape
  `[hidden, B]` slot — no capacity-sized buffer enters a collective (the
  workspace len()-semantics hazard). TP=2 allowed: identical plan per rank,
  fixed loop order, no collectives inside per-row loops.
- B=1 single-row path byte-identical; whole-step decode graph gated to
  rows==1 plans (bucketed batch graphs are stage 2). Mixed plans: per-prefill
  single-row sub-steps + one decode sub-batch (DSv4 pattern).
- `ARLE_QWEN35_BATCHED_DECODE=0` runs rows>1 as sequential single-row
  forwards — the honest same-binary A/B arm (the old behavior was death).

## Formula

c=4: token-parallel ops amortize ~4×/token, per-row ops (10 attn layers,
sampling) stay linear → predicted aggregate **~2.3–3.2× single-stream**
tok/s at B=4. License: c-sweep + needle at c=2 + per-stream ITL sanity.

## Coexistence note

Commit `9c979dd1` (MTP, lead-authored) swept this tranche's in-progress
executor.rs hunks mid-session, leaving HEAD referencing
`Qwen35BatchDecodeState` without its definition — THIS commit repairs HEAD.
(Same commit-race class as anti-pattern #30; two sessions one tree.)

## Rule

- A revived dead kernel inherits none of the fixes its live sibling
  accumulated — diff the pair (here: the warp_norms barrier) before wiring.
- When the decode unit changes (row → batch), re-verify every collective's
  message length and every buffer-shape assumption; exact-shape slots make
  the proof structural.
