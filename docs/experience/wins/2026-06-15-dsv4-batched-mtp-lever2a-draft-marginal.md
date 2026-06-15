# DSv4 batched MTP lever 2a (batched draft) — correct, draft phase −11%, throughput at noise floor (1-layer MoE limit)

## Context
Batched MTP fold WIN (+81% @c=12) left the per-slot DRAFT serial (~14% of the wave).
Lever 2a batches the MTP-head draft over N slots (`mtp_forward_level_batched`, gated
`ARLE_DSV4_BATCHED_MTP_DRAFT`), to amortize the MTP layer's `dsv4_moe_forward` — the
hypothesis being a verify-class win ([plan](../../plans/dsv4-batched-mtp-lever2-draft-commit.md)).

## Result (pod 8×H20, same binary, draft ON vs OFF, both BATCHED_MTP=1, c=12)
| | agg decode tok/s | draft phase (n=4) |
|---|---|---|
| draft ON (2a) | 76.45 | **13.7 ms** |
| draft OFF (lever-1) | 74.66 | 15.4 ms |

- **Correctness PASS**: word-needle both arms 6/8 own, **cross-contam=0** (identical —
  2a correct; the 2 misses are the shared uncommon-word recall limit, not a batched bug).
- **Draft phase −11%** (15.4→13.7 ms, clean same-profiler signal) — REAL but small
  (1.7 ms of the ~110 ms wave ≈ 1.5%).
- **Throughput +2.4%** (76.45 vs 74.66) — **at/below the noise floor** (baseline varies
  ±2% run-to-run; per [[feedback_matched_ab_for_small_bench_effects]] a ≤10% single-sweep
  effect needs ×2 repeats). The draft-phase −11% is the trustworthy signal; the
  throughput gain is marginal.

## Root cause (the predicted limit, confirmed)
The MTP-head is **1 transformer layer × 2 depth levels** — its `dsv4_moe_forward` is
too small to amortize like the verify's **60 layers**. Batching the draft shrinks its
phase ~11% but that's ~1.5% of the wave. The draft's per-slot attention (looped) doesn't
amortize either. So lever 2a is a correct micro-optimization, not a step change.

## Implication — decode is verify-compute-bound; residual levers diminish
- Lever 2a (draft, HAS a MoE): ~+2% (noise floor).
- Lever 2b (commit, `commit_layer_fold` is attention/KV-only, NO MoE): will be SMALLER.
- Lever 3 (CUDA-graph verify): the batched verify is GPU-compute-bound (big 60-layer MoE
  kernels) → less launch-gap than the per-row path → likely also modest.

**batched MTP +81% is near the decode ceiling. The verify (DSv4 60-layer MoE, ~70% of
the wave) is the inherent floor.** Bigger throughput needs reducing verify COMPUTE
(lower MTP depth, trades acceptance) or the DP-attn architectural track (prefill/scaling,
3-4 weeks, [scope](../../plans/dsv4-dp-attention.md)).

## Disposition
Lever 2a is CORRECT + a small clean draft-phase reduction (−11%), gated OFF
`ARLE_DSV4_BATCHED_MTP_DRAFT` (zero risk). Keep it (stacks, free), but don't claim a
throughput win without repeats. Lever 2b deferred (smaller, no MoE). The campaign's
honest verdict: the +81% fold win is the decode story; residuals are single-digit %.

## Rule
- **Amortization scales with the batched op's SIZE.** The verify won big (60-layer MoE
  over M rows); the draft's 1-layer MoE wins marginally for the same batching pattern.
  Estimate the amortizable-compute SIZE before assuming a batching lever is a step change
  ([[feedback_measure_batching_before_ceiling]]).
- **A measured phase reduction ≠ a throughput win at the noise floor.** Draft −11% is
  clean (same profiler); +2.4% throughput is within run-to-run ±2% — separate the
  trustworthy phase signal from the noisy wall-clock.
