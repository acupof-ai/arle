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

Identical to 17 significant figures across 61 optimizer steps. Each eval pass
contributed byte-identical counts (`+1541` chains / `+23115` drafted / `+3600`
accepted; `23115 = 1541 × 15`, the fixed DFlash draft width at block 16), so the
passes are deterministic replays and the head changed nothing.

**It is not the publish channel** (2026-08-04). The run's log shows 7 successful
publishes at steps 8…56 under `world_size=1`; the single `weight update failed`
is the step-62 off-cadence publish after engine shutdown. The path
`publish → update_dspark_markov_weights → run_on_engine → update_markov_weights`
also invalidates the decode graph and the prefix cache at
`executor/qwen35.rs:3647`.

The head was delivered and was too small to matter. `w2` starts at 0 and AdamW
moves each element ~lr per step, so after 61 steps at `lr = 1e-4`
`|w2| ≤ 6.1e-3` and `bias = Σ_{r<256} w1[c][r]·w2[v][r]` lands near 1e-3 — while
the serve adds it into a **bf16** buffer (`dspark.rs:1517`) whose half-ulp at
|logit| 8–16 is 0.016–0.031. Two orders under the rounding floor: `base + bias`
returns `base` bit-for-bit. Compounding it, the cold-start `w1` was
`0.02·sin(0.1·(i mod 1000))`, which aliases with period `gcd(1000, rank)` — 125
distinct rows for a 248320 vocab, all inside a ~4-dim subspace, and since
`∂bias/∂w2 = w1[c]` that is the whole head's ceiling, not just the init's.

Fixed in the 2026-08-04 tranche (hashed `w1` init, `lr` default 1e-3, a permanent
`rms|w1| rms|w2| est|bias|` line at publish). Whether a rank-256 additive bias on
frozen draft logits can move greedy acceptance **at all** is still unmeasured.

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

**Bit-identical output does not mean the intervention never arrived — it can
mean the intervention was smaller than the arithmetic that carries it.** Before
tracing plumbing, price the effect against the precision it has to survive: a
correction under half an ulp of the value it is added to is discarded exactly,
and the symptom is indistinguishable from a dead wire. Here the ~1e-3 bias met a
0.03 bf16 floor, and seven confirmed publishes were mistaken for zero.
