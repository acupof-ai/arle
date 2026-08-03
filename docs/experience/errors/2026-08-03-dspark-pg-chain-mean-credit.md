# DSpark trainer rewarded the tokens that caused the rejection

Date: 2026-08-03 · `crates/train/src/dspark_train.rs`

## Context

The one DSpark head on the pod (`/host/arle-runs/warm-head/`) got *worse* as it
trained: over 44 steps its `accept_ema` fell 0.414 → 0.324. That is a
single-run internal metric, so it is not exposed to the co-tenancy confound
that invalidated the same day's serve-side sweep
([retraction](2026-08-03-dspark-block-curve-was-cotenant-noise.md)) — and the
cause below is a property of the update rule, derivable from the source without
any benchmark.

The head's serve-side effect is **unmeasured**: the arms I ran to compare it
against the un-headed drafter shared that sweep's co-tenancy defect. Re-measure
on an empty box before quoting a number for it.

## Root Cause

**Chain-level reward broadcast to every drafted token.** The policy-gradient
term computed one scalar per chain, `reward = accepted / block_size`, and
pushed it into `row_rewards` for every row of that chain, so every token in the
block shared one advantage `(reward − baseline)`.

Verify stops at the first rejection, so `accepted` is the accepted *prefix
length*: row `t` drafted `chain[t+1]` and is accepted iff `t < accepted`. That
per-position label is exact, free, and was discarded.

Consequence: on any chain whose reward beat the baseline, the update **raised**
the log-prob of the whole block — including the rejected tail, i.e. exactly the
tokens that ended the chain. The baseline is the running mean, so roughly half
of all chains do this. Acceptance drifts down monotonically, which is what both
the serve A/B and the training log measured.

The `loss_decay_gamma` position weighting (`exp(−t/4)`) softens the tail but
does not change its sign.

## Fix

Per-token credit `r_t = 1[t < accepted]`, and centre the baseline on the same
weighted quantity the advantage subtracts it from (previously the baseline
tracked `accepted/block_size` while the advantage subtracted it from a 0/1
credit — two different scales). Accepted tokens now get `+(1 − b)`, rejected
`−b`: standard REINFORCE-with-baseline at token granularity.

Second, unrelated defect found while reading the log: the ISO spectrum probe's
cold-parameter guard (`den <= 1e-30`) is an absolute floor on a sum of σ⁴. A
near-zero head clears it while still being ~1e14× smaller than the current
spectrum, so the probe reported `spectrum_drift = 2.82e14` — a division
artifact, **not** a training blow-up. Guard is now relative to the current
spectrum. Do not read that number as evidence of divergence.

## Rule

**A chain-level reward is not a token-level credit.** When the verifier stops
at the first rejection, per-position accept/reject is already known exactly —
using the chain mean instead does not merely add variance, it flips the
gradient sign on the rejected tail. Check the sign of the credit each trained
row receives before trusting any acceptance-reward loop.
