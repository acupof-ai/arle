# The DSpark online head does move drafting — my threshold for it was wrong twice

Date: 2026-08-04 · `/host/arle-runs/warm-ce-tv-v2/` · build `bf16floor` (`ea4798cc5`)

## Context

The 2026-08-03 run froze serve `accept_rate` at `0.15574302401038287` across 61
training steps ([entry](2026-08-03-dspark-online-sidecar-degrades-regardless-of-loss.md)).
Diagnosis: the trained bias (~1e-3) was under half a bf16 ulp of the base draft
logit (~0.03), so `base + bias` returned `base` bit-for-bit. Fixes shipped in
`ea4798cc5`: hashed `w1` cold start, `lr` default 1e-4 → 1e-3, and a permanent
`rms|w1| rms|w2| est|bias|` line at every publish.

This run re-ran the same recipe at a 260-step budget, 281 steps and 35 publishes.

## What the run settled

**The loop is closed.** `accept_rate` is no longer frozen — it moves across
passes (0.0978 → 0.1023 → 0.1030 → 0.1050). The head reaches the drafter and
changes drafted tokens. Consistency check: an untrained head still reproduces
`chains=1541 drafted=23115 accepted=3600` bit-for-bit, so the no-op property of
`w2 = 0` holds exactly.

**The bf16 floor is not a necessary condition, and my model of it was wrong
twice.** `est|bias|` peaked at 6.114e-3 and never approached 0.03, yet drafting
changed anyway. `est|bias| = sqrt(rank)·rms|w1|·rms|w2|` is an RMS over all
`(c, v)` pairs; argmax flips are a **tail** event, and the tail runs well above
the RMS. I gated a tail event on a mean.

The growth rate was wrong too, in both directions:

| estimate | model | predicted steps to 0.03 |
|---|---|---|
| first | AdamW moves each element ~lr/step | 162 |
| second | linear extrapolation from publish #1 | ~5000 |
| measured | `rms\|w2\|` 3.867e-4 (step 8) → 3.437e-2 (step 280) | superlinear: 88.9× over 35× the steps |

Growth accelerates because `∂bias/∂w1 ∝ w2`, so `w1` unfreezes as `w2` grows.
Both estimates assumed linearity. The measured AdamW step is ~0.035·lr, not ~lr:
most vocab entries get gradients that flip sign between steps and cancel.

## Direction is NOT established — do not read one from this run

- `accept_ema` fell 0.6228 → 0.2834, but a **constant-mean** EMA still fits:
  at α=0.01, `0.99^280 = 0.060`, so `0.28 + 0.343·0.060 = 0.30` vs 0.283
  observed. Same trap as 2026-08-03; no trend is readable.
- The baseline is confounded. A driver script from an earlier, killed launch of
  the same recipe survived and kept hitting the same serve while this run
  measured `ACCEPT_BEFORE`, which came back at 122 chains against the old
  baseline's 1541. The two are not comparable.

## The premise remains unsupported, and the scale gap is the reason

The reference never trains the Markov head alone — `freeze=True` in
`deepspec/trainer/base_trainer.py:279` applies to the target-tied
embedding/lm_head, and the draft backbone, Markov head and confidence head train
jointly. Rows per optimizer step:

| | rows/step |
|---|---|
| DeepSpec (`global_batch_size 512 × num_anchors 512 × block_size 7`) | 1,835,008 |
| this sidecar (measured, `rows=` in the step log) | 120 |

**15,292×**, before counting the reference's 10 epochs. This sidecar spends 120
rows of gradient on 63.6M parameters (`248320 × 256 × 2`) — ~530k parameters per
training row. That is the mechanism behind the 0.035·lr effective step, and it is
not a learning-rate problem.

## Fix

Deleted `cold_start_default_lr_reaches_the_bf16_floor`. It encoded a falsified
model *and* a falsified threshold, and it passed — the worst combination.
`markov_w1_init_rows_are_distinct` stays: that one is exact.

Operational: launching a long pod run through `~/bin/pod` with `setsid nohup …&`
does not survive the tool-level timeout — the tunnel session dies
(`tn: wait: remote command exited without exit status or exit signal`) and the
serve takes a shutdown signal mid-run. The first attempt died at step 14 that
way; training kept stepping off the buffered backlog, which looks exactly like a
live engine. Use `setsid --fork`, and check `grep -c 'shutdown signal'` plus
whether drive passes actually complete before reading any counter.

## Rule

**Gate the event you care about, not a summary statistic of it.** An RMS, a mean,
or an EMA can sit orders of magnitude away from the tail that actually decides
the outcome — a bias whose RMS is 5× under the rounding floor still flipped
argmaxes. Before trusting a numeric gate, ask which moment of the distribution
the real event lives in.

**A threshold derived from an unvalidated model is not a gate, it is a guess that
returns green.** Both of mine were wrong by ~30× in opposite directions and one
of them shipped as a passing test. Extrapolate only after the first two measured
points disagree with the model, and prefer the instrument that reports the raw
quantity over any test that predicts it.
