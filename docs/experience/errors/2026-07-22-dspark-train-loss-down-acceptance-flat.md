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

## Root Cause (hypothesis)

`prob_match_alpha = 0.9` (DeepSpec default) means 90% of the gradient comes
from L2 probability matching (`Σ(softmax(draft) − softmax(target))²`) and only
10% from policy gradient. The L2 loss pulls the draft *distribution* toward
the target distribution, but acceptance depends on whether the draft's
**argmax** matches the target's argmax — a distribution can be closer in L2
yet have the same or worse top-1 agreement. The PG signal (reward =
accepted/block_size) that directly optimizes acceptance is drowned out.

## Fix being tested

Lower `prob_match_alpha` 0.9 → 0.5 (PG:prob_match = 1:1) so the PG gradient
gets equal weight. Commit `b719eb252`. Benchmark pending (agent killed by user
mid-run; needs re-run).

## Rule

- A training loss decrease is **not** evidence of effectiveness — measure the
  actual target metric (acceptance rate) with a before/after A/B.
- Loss weights that match an upstream reference (DeepSpec's 0.9) may not be
  optimal for a different objective (we optimize acceptance, they optimize
  downstream task quality).
