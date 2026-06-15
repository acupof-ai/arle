# DSv4 lever 2b (batched commit fold) — WASH (attention-only commit doesn't amortize), KILL

## Context
Completing the lever-2 residual investigation. Lever 2b batches the per-slot
`commit_accepted_fold` into ONE 60-layer pass over all slots
(`commit_accepted_fold_batched`, gated `ARLE_DSV4_BATCHED_MTP_COMMIT`). Predicted <1%
(`commit_layer_fold` is attention/KV-only, no MoE).

## Result (pod 8×H20, same binary, COMMIT on vs off, both BATCHED_MTP=1, c=12)
| | agg decode tok/s | commit phase (n=8) |
|---|---|---|
| commit ON (2b) | 68.43 (avg_active 9) | ~36 ms |
| commit OFF (baseline) | 74.82 (avg_active 12) | ~32 ms (n=7) |

- **Commit phase did NOT shrink** (commiton ~36 ms ≈ commitoff extrapolated ~36 ms at
  n=8). The batched commit shares the 60-layer HOST loop (one pass vs N) but the per-slot
  `commit_layer_fold` GPU compute (per-slot attention/compressor re-derive) dominates and
  does NOT amortize — there's no MoE to group. As predicted.
- Perf wash/slight-loss (68.43 vs 74.82, confounded by avg_active 9 vs 12 run-to-run).
- Correct: cross-contam=0 both arms; 5/8 vs 6/8 own is run-to-run word-recall noise.

## Verdict: KILL (stays gated OFF)
Lever 2b gives no win. The commit residual is attention/KV compute that is genuinely
per-slot (no amortizable MoE). Combined with lever 2a (draft, marginal +2% — 1-layer
MoE) and lever 3 (deferred, verify compute-bound), this CLOSES the lever-2/3 residual
investigation: **the batched MTP +77% deploy is the decode story; the residual phases
(draft, commit) don't amortize and the verify is GPU-compute-bound (DSv4 60-layer MoE)**.

## Rule
- **Batched-commit amortizes only if the committed op has amortizable (weight-read)
  compute.** The verify won (60-layer MoE over M rows); the commit fold is attention-only
  → batching shares the host loop but not the GPU compute → wash. Check the op's compute
  KIND (MoE vs attention-only) before batching it
  ([[feedback_measure_batching_before_ceiling]]).
- The lever-2 arc (2a draft marginal, 2b commit wash) empirically confirms the
  verify-compute-bound principle: the residual ~30% (draft+commit) is per-slot/small,
  not the verify's amortizable MoE.
