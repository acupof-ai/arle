# DSpark's −50% at c=16 was a default-off training sidecar's stream sync (#183)

> Status: gate landed, **re-measure pending-remote** (#183). The DSpark
> accept-or-kill verdict (#128) is NOT decidable from the 2026-07-25 run.

## Context

First run in which DSv4 DSpark actually took the spec path, counter-proven:
`/v1/stats.spec_decode` went `available:false, chains:0` → `chains 64, drafted
320, accepted 176, accept_rate 0.55` after one greedy request; the no-spec arm
stayed `available:false` throughout. The old `--dspark-max-prompt-tokens 64`
routing gate no longer exists, so the `docs/baselines.md` "not triggered" row is
finally superseded.

Matched A/B, DSv4-Flash-FP8, 4×H20 TP=4/EP=4, 128-in/128-out, 60 s/point, seed
20260416, 0 errors at all 8 points:

| c | no-spec out tok/s | DSpark out tok/s | Δ | committed tok/step |
|---|---:|---:|---:|---:|
| 1 | 42.7 | 43.3 | +1.4% | 2.48 |
| 4 | 81.4 | 58.5 | −28.1% | 2.53 |
| 8 | 139.8 | 73.7 | −47.3% | 2.54 |
| 16 | 174.4 | 87.3 | −49.9% | 2.45 |

Acceptance is healthy and flat across concurrency (accept_rate 0.29–0.31, 2.45–
2.54 committed tokens per verify step, block 5, `conf_threshold 0` so every
chain drafts full width). The draft head works. The step does not.

## Root Cause

The step carried a cost that belongs to a default-off feature. Both DSpark
verify lanes (`executor/dsv4.rs:1843`, `:2140`) called
`capture_dspark_experience{,_hidden}` unconditionally, and those do **two
vocab-wide bf16 D2H copies plus two full `ctx.sync()` per verify step** before
any shape guard runs. The only consumer is the `--dspark-train` RL sidecar,
which was off.

A full stream sync per step serializes a TP=4 NCCL pipeline — which is exactly
the shape of the observed loss: ~0 at c=1, worsening monotonically to −50% at
c=16. So the concurrency collapse is not attributable to verify overhead until
the sync is gone.

## Fix

`capturing()` = `BUFFER.get().is_some()`. The sidecar is the only thing that
initializes the global buffer (`spawn_dspark_train_sidecar` →
`engine.dspark_experience_buffer()` → `buffer()`), so its presence *is* the
`--dspark-train` signal — no second flag to drift out of sync. Both capture
entry points bail on `!capturing()` before touching device memory.

## Rule

- **A hot-path producer for a default-off consumer must be gated on the consumer,
  not on its own arguments.** The #176 guard checked shapes and dropped the push
  — but the D2H and the syncs ran *before* the guard, so the cost was paid on
  every step to build data that was then thrown away. Cheap-check-first is not
  optional when the check is "is anyone listening".
- **Derive the gate from existing state instead of adding a flag.** A `OnceLock`
  that only one caller initializes is already the enable bit.
- **A spec-decode verdict needs the arms to differ only in speculation.** Here
  the DSpark arm also differed in "pays a stream sync per step" and in slot count
  (22 vs 32, #184). Acceptance was measured correctly; wall-clock was not
  measuring what its label claimed.
