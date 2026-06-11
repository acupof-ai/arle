# Decode-band MoE kernel LICENSED — 2.26× single-stream, c=4 plateau lifted

**Date:** 2026-06-11. **Commit:** `9e37bc77`. **A/B:** same binary,
`ARLE_QWEN35_MOE_DECODE_KERNEL` env flip, H20 idle box.

## Result

| c | old hand kernels | new decode kernels | Δ |
|---|---|---|---|
| 1 | 40.64 | **91.97** | **+126%** |
| 2 | 66.25 | **141.59** | +114% |
| 4 | 98.12 | **185.38** | +89% |
| needle@c2 | 2.75 s PASS | **1.65 s PASS** | −40% |

Mechanism check: nsys had pinned the two hand grouped kernels at 76.6% of
decode GPU time (487 µs/layer at R=8, 3% HBM efficiency); the formula
predicted 27.3 → 9–11 ms/token — measured 10.9 ms (91.97 tok/s). First
formula of the campaign to land inside its own band. Campaign cumulative:
single-stream 36.0 → 92.0 tok/s (2.56×), aggregate c=4 engine-death → 185.

Per the "直接抄业界最好的" directive, the adoption survey
(docs/reviews/2026-06-11-decode-moe-kernel-adoption-survey.md) ran first:
the industry-best liftable kernel (DeepGEMM masked, DeepSeek production
class) was already in-tree and measured neutral-to-worse at decode; the only
GEMV-class industry precedent (llama.cpp mmvf/mmf) is the same algorithm
family as this kernel. A/B over adoption was the evidence-backed call.

Note: the old arm's c=4 here (98.1) is above the earlier b2077c65 sweep
(65.5) — different build plus box variance; both arms measured back-to-back
same-session, so the licensed Δ is the within-session pair.

## Rule

- A weight-read-bound op at small M wants warp-per-row exactly-once streaming,
  not tensor-core tiles — on both our hand kernels AND the lifted
  industry tensor-core path, the 16–128-row tile floor burned 8–30× the
  ideal bytes at R=8.
