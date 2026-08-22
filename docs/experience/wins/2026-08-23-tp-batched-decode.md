# TP batched decode: rows>1 goes through the batched paged forward — CUDA, 2026-08-23

> Status: Verified (ce190fc20) — c≥2 ITL 1.55×–17.89×, aggregate 87 → 1,300 tok/s at c=32

## Context

After the TP decode graph landed (2026-08-22 entry), TP2 decode was still flat at
~88 tok/s aggregate from c=2 up: under TP, every rows>1 decode sub-batch fell
back to B per-row eager forwards (`executor/qwen35.rs` `is_single` gate). The
batched paged forward was already TP-complete — attention all-reduces inside the
attention kernels, FFN all-reduce per layer — and was only gated by a
never-validated single-GPU assumption.

## Change

`ce190fc20` — deleted the `is_single` branch and the per-row fallback loop in
`submit_decode_batch`; rows>1 always routes to `submit_decode_batch_paged`.
c=1 is untouched (single-row graph path).

## Verification

ThinkingCap-Qwen3.6-27B-NVFP4, 8×H20, TP2, FP8 KV, spec off, matched GPU pairs
(treatment 0,1 / baseline 6,7 — a cross-NUMA pair (2,4) cost +60% ITL on the
baseline and was discarded). Baseline = Phase 1 binary rebuilt with `cuda,nccl`
(per-row at c≥2, graph at c=1). Simultaneous A/B, 64 synthetic prompts, 128
tokens, c=1..32:

| c | batched ITL ms | per-row ITL ms | speedup | batched wall s | per-row wall s |
|---|---|---|---|---|---|
| 1 | 14.75 | 13.91 | wash | 122.9 | 115.7 |
| 2 | 14.59 | 22.56 | 1.55× | 62.8 | 96.0 |
| 4 | 13.42 | 44.94 | 3.35× | 29.7 | 94.4 |
| 8 | 16.29 | 90.32 | 5.55× | 18.1 | 94.1 |
| 16 | 14.68 | 181.09 | 12.34× | 8.9 | 93.9 |
| 32 | 20.25 | 362.40 | 17.89× | 6.3 | 93.9 |

Aggregate at c=32: 87 → 1,300 tok/s (14.9×). TTFT improves with it (936 → 580
ms at c=32, less CPU contention). Per-row ITL grows linearly with batch size
(B forwards per step); batched ITL stays flat — one forward per step.

Correctness: two needle ladders ×3 runs concurrently against the batched serve
(mixed batches, NEEDLE_MAX_TOKENS=512): 54/54 exact across all 9 lengths. The
Phase 1 len=300 counting-loop degeneracy did not reproduce (batch composition
changes MoE routing order). The bench repetition gate's per-arm mismatch
(batched 0 vs per-row 8 failures at c≥2) is the same routing-order effect, not
corruption — the needle content gate is clean.

## Rule

A per-row fallback loop under TP is a throughput ceiling, not a safety net:
once the batched forward has its collectives in place, the single-GPU gate is
unvalidated caution, and validating it is one A/B away. Match A/B GPU pairs for
NUMA as well as model — a cross-socket pair cost +60% ITL and masqueraded as a
treatment effect.
