# DSpark train: loss decreases but acceptance rate does not improve

## Context

The DSpark train sidecar (`--dspark-train`) runs acceptance-weighted policy
gradient + probability matching on the Markov head while serving. E2E verified
on H20 (2026-07-20): trainer runs, loss decreases, weights hot-swap, zero errors.
But the *goal* of training — higher draft acceptance rate — was never measured
until 2026-07-22.

## Benchmark (2026-07-22, H20, Qwen3.6-27B-FP8 + dspark-aeon)

Single serve process, `--dspark-train`, before/after 5 min training:

| Phase | accept_rate | accepted/drafted |
|-------|-------------|------------------|
| Before training | 0.4363 | 2666/6110 |
| After 5 min training | 0.4320 | 2661/6160 |
| **Delta** | **−0.98%** | ↓ |

Trainer was active: loss 0.0318 → −0.0243, baseline EMA 0.5014 → 0.4874,
n=64 batches. Loss decreased ~5× but acceptance rate *slightly decreased*.

## Root Cause — verified (#169, 2026-07-24)

The trainer optimized a mis-parameterized, mis-aligned objective — loss could
fall while serve acceptance stayed flat. Four defects, all static-verified:

- **D1 — w2 frame transposed.** Serve is `[vocab, rank]` row-major
  (`infer-cuda/src/qwen35/dspark.rs:564` load ensure; `ops.rs:146` gemm
  contract). Trainer wrapped the same flat buffer as `[rank, vocab]` and did
  `emb @ w2` (`train/src/dspark_train.rs:192,252`) — a different index
  permutation. Fixed: `[vocab, rank]` + `matmul_bt`.
- **D2 — bias leaked the row's own label.** Serve conditions on the PREVIOUS
  chain token (`dspark.rs:966-975`); trainer embedded the token at position j
  itself (`dspark_train.rs:240-245`).
- **D3 — row alignment ignored head mode.** Serve `first_row =
  !next_token_heads` (`dspark.rs:964`): same-position heads never draft row 0
  and pair draft row j ↔ target row j−1; trainer used j↔j, rows 0.., token =
  cond = chain[j]. Fixed per-mode; decay now indexes the draft position within
  trained rows, not the raw row.
- **D4 — the mode flag wasn't captured.** `DsparkExperience` lacked
  `next_token_heads`; added at both capture fns + all call sites (qwen35 +
  DSv4).

## Prior hypothesis — alpha weighting

The first-pass hypothesis blamed `prob_match_alpha = 0.9` (L2 drowns PG).
The 0.9 → 0.5 flip (`b719eb252`) is predicted no-op pending re-benchmark —
with D1-D3, both loss terms back-propagated through the wrong frame and rows,
so no weighting could have fixed acceptance.

## Accept gate — resolved 2026-07-24

Post-fix re-benchmark (721a553bc, same protocol): **still flat** — windowed
accept 0.4694 → 0.4583, slope −0.0118/5 min; trainer healthy (loss 0.0020 →
−0.0867, 0 failed hot-swaps). Decoded case dump (N=2048, `ARLE_DSPARK_DUMP`):
P(drafted == target_argmax) 0.580 vs base-head argmax agreement 0.479 — the
markov bias earns ~10pp at serve, so the alignment fixes work; the residual
gap is CAPACITY (rank-r bigram bias over 248320 vocab, 640 experiences per
7.5 min; low-reward tercile has base-trunk agreement 0.28 — flat-target
contexts a bigram bias cannot fix). #169 closed fixed-and-attributed; the
acceptance lever is full draft-head training (#127).

## Rule

- A training loss decrease is **not** evidence of effectiveness — measure the
  actual target metric (acceptance rate) with a before/after A/B.
- Loss weights that match an upstream reference (DeepSpec's 0.9) may not be
  optimal for a different objective (we optimize acceptance, they optimize
  downstream task quality).
