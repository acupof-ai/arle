# ISO-Merger SVD can't be kernel-accelerated without breaking bf16 parity

Date: 2026-08-01 · Tool: [`scripts/iso_merger.py`](../../../scripts/iso_merger.py)

## Context

ISO-Merger spends ~15.8s per big MLP tensor (5120×17408), all in SVD: 3 full
`torch.linalg.svd` for base/expert frames (9.45s) + 2 for polar retraction
(6.3s). Read is free (page cache). "Speed it up with a kernel" — the obvious
levers are Gram+eigh (avoid the rectangular SVD) and eigh-polar (avoid the
tall-skinny SVD). Both are real math identities, not approximations in theory.

## Root Cause

**ISO merges in the Stiefel tangent space, which needs high-precision singular
frames — every Gram/low-rank shortcut squares or truncates the exact thing the
merge is sensitive to.** Each candidate is fast *in isolation* but fails
end-to-end bf16 parity against the validated SVD path (MMLU 0.808 / GSM8K 0.949
/ de-censoring graft). The isolated micro-benchmark lies; only the full
`merge_one_tensor` output, quantized to the bf16 save dtype, is the honest test.

| candidate | isolated | end-to-end merge | bf16 within-1-ulp | verdict |
|---|---|---|---|---|
| eigh polar `P = X(XᵀX)^{-1/2}` | 7.6× | 1.3× | 0.73 | reject |
| Gram+eigh SVD `W Wᵀ → eigh` | 9.1× | 2.5× | 0.63–0.93 | reject |
| randomized `svd_lowrank(q=0.9r)` | 1.2× | — | — | reject (q≈rank, no win) |

Gram squares the condition number → small-mode frame vectors lose precision →
tangent projection + polar amplify it. eigh-polar is only 1.3× end-to-end
because polar is a minority of the time; not worth breaking parity for.
`svd_lowrank` needs q≪rank to win, but ISO keeps 90% of modes (q≈0.9·rank).

## Fix

**No kernel change.** cuSOLVER's general SVD is already the right kernel for
this precision requirement. The only parity-safe speedup is data-parallel
sharding across GPUs (`--shard-mod N --shard-idx k`), already in the tool:
140min → 29min on 4 cards, ~15min on 8. Zero numerics risk.

## Rule

**A numerics shortcut is validated by the end-to-end output in the save dtype,
never by an isolated micro-benchmark.** For any SVD-based merge/decomposition,
A/B the full pipeline and gate on bf16 within-1-ulp fraction ≥ 0.999 — the
isolated reconstruction error (1.8e-6 for Gram here) does not predict it.
Offline one-shot tools: don't trade validated correctness for wall-clock;
parallelize instead.
