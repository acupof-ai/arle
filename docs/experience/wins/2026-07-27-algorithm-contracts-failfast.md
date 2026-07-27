# Public algorithm contracts — fail-fast + Dr.GRPO reduction, 2026-07-27

> Status: correctness complete; no runtime perf claim (config-gate + reduction).

## Goal

Close three public algorithm contracts so an advertised knob either does what it
says or refuses before any model loads — no silent no-op mid-training.

## What changed

- `train_cli.rs`: `reject_unimplemented_gkd_objectives` fails before model/store
  init when `--gkd-entropy-weight != 0` (AEPO objective does not exist) or
  `--teacher-topk` is set (no engine-side top-k producer wired). Called at all
  five OPD entrypoints (opd from-dirs/smoke, self-opd from-dir/smoke, agent-opd).
  Previously these ran a loud-TODO eprintln and silently fell back to dense KL at
  step time — after the model was already loaded.
- `update_strategy.rs`: extracted the PG token-mean denominator to a pure
  `token_mean_norm`. Confirms Dr.GRPO's fixed-`norm_const` length-debiasing is
  separated from group averaging (which lives in `centered_advantages`); the code
  was already correct, this pins it.

## Results

Local:

```text
cli:  unimplemented_gkd_objectives_fail_before_load          passed
train update_strategy::tests: 12 passed (incl. dr_grpo_norm_is_fixed_constant)
CUDA/no-CUDA typecheck (cli): passed
fmt + diff --check: passed
```

Dr.GRPO oracle pins: fixed budget (4096) divides every token regardless of
trajectory length (10 vs 4000 → same 4096); GRPO falls back to batch tokens;
PerSeqTokenMean divides by own length (the bias); empty guard → 1.

## Problems

Reference-policy KL (independent reference sidecar, distinct from behavior
logprobs) is deferred to a separate tranche — it is a new evidence producer, not
a fail-fast. Today `kl_coef` is hardcoded 0.0 so the k3-against-behavior term is
inert; turning it on without a reference sidecar would be wrong, so the flag
stays absent until the producer lands.

## Learnings

**A no-op flag must fail at parse, not at step.** A flag that loads a 27B model,
runs rollouts, then logs "NOT YET IMPLEMENTED, running unweighted KL" burns a
GPU-hour to tell you the objective you asked for does not exist. Reject before
init; defense-in-depth checks stay in the library.
