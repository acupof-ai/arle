# Finite-step transaction — train optimizer mutation, 2026-07-27

> Status: correctness complete; H20 negative-case gate pending-remote.

## Goal

Make every train optimizer mutation one all-or-nothing transaction: finite loss
→ complete backward → finite global grad norm (f64) → optional clip → one
`optimizer.step` → clear grads. Any failure clears pending grads and advances no
parameter, moment, schedule, baseline, or artifact.

## What changed

- `grad_clip.rs`: `FiniteStepError` + `ensure_finite_loss` + `finite_optimizer_step`
  (the single pre-mutation gate). Idempotently zeroes grads on every reject path.
- `opd.rs`: all CE / PG / GKD / critic / windowed / chunked-KL steps converge on
  `finite_optimizer_step`; loss readback validated before backward; deleted
  `sanitize_non_finite_grads` (production sanitize-and-continue).
- `update_strategy.rs`: `GlobalTokenMean` accumulation is one fallible unit — any
  mid-loop error clears pending grads so a failed batch cannot leak into the next.
- `dspark_train.rs`: loss readback before backward; `finite_optimizer_step`;
  `baseline_ema`/`steps` advance only after a successful step; ISO retraction
  deferred so tensor cleanup + tape reset run unconditionally.

## Results

Local:

```text
test_grad_clip:    5 passed  (incl. 2 new fault-injection cases)
test_dspark_train: 6 passed
test_opd_step:     20 passed
update_strategy:   11 passed
release + CUDA/no-CUDA typecheck + fmt + diff --check: passed
```

Fault-injection proves NaN/Inf loss and non-finite grad norm both reject before
any parameter/grad mutation.

## Problems

H20 CE/PG/GKD/DSpark negative-case acceptance is not yet run — pending-remote.

## Learnings

**One gate, every path.** A per-call-site `step`/`zero_grad` pair is where
sanitize-and-continue and half-committed baselines hide. Routing all mutation
through `finite_optimizer_step` makes "no mutation on non-finite" a single
provable invariant instead of N inconsistent ones.
