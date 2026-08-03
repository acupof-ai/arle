# The DSpark online sidecar degrades the head whatever the loss is

Date: 2026-08-03 · `/host/arle-runs/warm-ce-tv/` · build `dsparkloss2`

## Context

The trainer had a real credit-assignment bug — a chain-level reward broadcast
to every drafted token, so above-baseline chains raised the log-prob of their
own rejected tail
([root cause](2026-08-03-dspark-pg-chain-mean-credit.md)). It was fixed, then
the whole objective was replaced with the paper's
(`fc481a181`, `2c55fce10`): CE on the trunk's token + L1 total variation,
0.1 / 0.9, no policy gradient at all.

This is the GPU rerun of the 2026-07-28 warm-start that produced the degrading
head — same shape (cold start, w2 = 0, ISO off, single H20, same 20-prompt
corpus, 256 tokens/request, `--dspark-train`).

## Result — the fix did not change the outcome

| objective | steps | accept_ema first → last | relative |
|---|---|---|---|
| PG + squared diff (2026-07-28) | 44 | 0.4141 → 0.3242 | **−22%** |
| CE + L1 TV (this run) | 47 | 0.7730 → 0.6217 | **−20%** |

Monotone decline, step by step, no plateau. The two `accept_ema` values are
different quantities — the old one was the chain mean `accepted/block_size`,
the new one is the position-weighted per-token accept rate — so only the
*trend* is comparable. It is the same trend at the same magnitude.

Serve-side `/v1/stats` accept_rate on the untrained head, before any step:
0.1557 (block 16). Per-step `loss` is not a convergence signal here — every
step is a different online batch, so it tracks batch difficulty, not progress.

## Root cause: parameters ≫ samples

The Markov head is `vocab × rank × 2` = 248320 × 256 × 2 ≈ **127 M
parameters**. This run trained it on 47 steps × 8 experiences ≈ **380 chains**.
The paper's recipe is 1.3 M target-regenerated samples × 10 epochs against a
whole draft backbone. Fitting 1e8 parameters to 1e2 samples memorises the batch
and generalises negatively — which is exactly a monotone acceptance decline on
held-out traffic.

The objective was genuinely wrong and is now right. It was not the binding
constraint.

## Rule

**The in-serve sidecar is refinement, not training.** Cold-starting a head
through it cannot work at any loss function, because the data rate of one
serve's traffic is three orders of magnitude below what the head's parameter
count needs. A head has to come from DeepSpec's offline trainer
(`config/dspark/dspark_qwen3_*.py`, 8 GPUs, ships data prep) and be loaded via
`scripts/convert_dspark_speculators.py`; the sidecar may then nudge it.

Corollary for reading old logs: `spectrum_drift` from before `fc481a181` is not
evidence of divergence — the probe's cold-parameter guard was an absolute floor
on a sum of σ⁴, which a near-zero head clears. Post-fix the same head reports
`[1.4e-5, 0.00e0]` where it used to report `2.82e14`.
