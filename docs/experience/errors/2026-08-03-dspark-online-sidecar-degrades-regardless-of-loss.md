# RETRACTED — the DSpark accept_ema "decline" is an EMA that never settled

Date: 2026-08-03 · `/host/arle-runs/warm-ce-tv/` · build `dsparkloss2`

> This entry originally concluded "the objective fix did not change the outcome;
> the sidecar degrades the head at any loss". **Both halves are withdrawn.** The
> run measured nothing about whether training helps or hurts. Nothing shipped on
> the retracted claim.

## What was claimed

A GPU rerun of the 2026-07-28 warm-start with the paper's CE + L1-TV objective
in place of the buggy policy gradient. `accept_ema` fell 0.7730 → 0.5838 over 61
steps (−24%), against the old objective's 0.4141 → 0.3242 over 44 (−22%). Read
as: same monotone degradation, so the loss was never the binding constraint.

## Why it is wrong

**1. The decline is the EMA converging from its seed, with no trend underneath.**
`baseline_ema_alpha` is 0.01 and the EMA is seeded from the *first batch*. At
α=0.01 the time constant is ~100 steps, so in 61 steps it never reaches steady
state: from any unlucky seed above the true mean it must fall monotonically,
trained or not. Fitting `e_k = m + (e_1 − m)·0.99^(k−1)` — a constant mean, no
trend term — to the whole trajectory:

| step | observed | pure-EMA model | resid |
|---|---|---|---|
| 1 | 0.7730 | 0.7730 | 0.0000 |
| 20 | 0.7077 | 0.7004 | +0.0073 |
| 36 | 0.6553 | 0.6491 | +0.0062 |
| 49 | 0.6133 | 0.6131 | +0.0002 |
| 61 | 0.5838 | 0.5838 | 0.0000 |

Residuals ≤ 0.7 pp against an implied **constant** batch mean m = 0.355. A
constant-mean model explains the entire "degradation". The 2026-07-28 run used
the same α and the same seeding, so its 0.414 → 0.324 is the same artifact and
is withdrawn as evidence too.

**2. The serve side did not move at all — and could not have.** Every pass
produced byte-identical spec counters (+1541 chains / +23115 drafted / +3600
accepted each), so `accept_rate` was bit-identical at every checkpoint:

```
ACCEPT_BEFORE           accept_rate=0.15574302401038287   (untrained, w2=0)
ACCEPT_AFTER_CUMULATIVE accept_rate=0.15574302401038287   (after 49 steps)
PRE_CLEAN               accept_rate=0.15574302401038287
POST_CLEAN              accept_rate=0.15574302401038287   (after 61 steps)
```

Identical to 17 significant figures across 61 optimizer steps. A live head whose
`w2` grew from zero changes the draft logits, hence the drafted tokens, hence
the accepted counts. Bit-identical counters mean the trained head **never
reached the drafter**. The head file is written every 8 steps
(`saved markov head at step 8/16/24…`), but no hot-swap into the engine is
observable in the log.

So the experiment has one real finding, and it is not about the loss: **the
train→serve publish path is not taking effect.** Open — it needs a direct probe
(`update_dspark_markov_weights` reached? bias non-zero on the device?), not
another training run.

## What still stands

The policy-gradient credit bug ([entry](2026-08-03-dspark-pg-chain-mean-credit.md))
is real **by derivation** — a chain-level reward broadcast to every drafted token
raises the log-prob of the rejected tail on any above-baseline chain. That is a
property of the update rule, readable in the source. Its *empirical* support was
the 07-28 accept_ema fall, and that support is withdrawn here.

The objective now matching the paper (`2c55fce10`) also stands on its own terms.
Neither is measured.

## Rule

**An EMA seeded from the first sample is not a trend line.** Before reading
direction off a smoothed metric, check the number of steps against the
smoothing time constant (`α=0.01` ⇒ ~100 steps) and fit the constant-mean model
first — if that fits, there is no trend to report. Prefer the raw per-step
value, or a metric with an independent scale, for any claim about direction.

**Bit-identical counters across an intervention mean the intervention did not
land.** Read them as a wiring check before reading them as a result.
