# DAPO dynamic sampling — rollout refill, not post-hoc filtering, 2026-07-27

> Status: correctness complete; default off; H20 end-to-end gate pending-remote.

## Goal

DAPO's dynamic sampling is a rollout-orchestration policy, not a batch filter:
a group that carries no learning signal (zero-variance reward, or all-truncated)
must not consume an effective slot — draw a replacement task and roll again until
the round trains as many effective groups as it scheduled, bounded so an
impossible corpus terminates deterministically.

## What changed

- `cc_harness.rs`: `RefillBudget` — pure accounting. `complete(committed,
  trained, tokens)` records the group and returns whether to boot a replacement,
  gated by three independent stop conditions: target reached, launch cap,
  cumulative-token budget. No rollout ownership; the caller launches.
- `train_cli.rs` (agent-opd round loop): `for round_tasks` → `while pos <
  work.len()` over a growable work vec. A dead group (`planned_training_count ==
  0`, the update's own filter) appends a replacement drawn from the round's
  unscheduled tasks; the launch is deferred to end-of-body so a refill thread
  never races the pre-writeback VRAM release. Metrics gain
  `groups_launched/effective/discarded`.
- `args.rs`: `--dynamic-sampling` (default off), `--dynamic-sampling-max-factor`
  (launch cap ×scheduled, default 3), `--dynamic-sampling-token-budget`.

Off ⇒ `max_launches == target` ⇒ `complete` always returns false ⇒ `work` never
grows ⇒ one code path, byte-identical to the pre-change loop.

## Results

Local:

```text
cc_harness::tests: 6 passed (4 new: refill-to-target, all-dead-terminates,
                             token-budget-halts, off-mode-inert)
train lib:         176 passed
CUDA/no-CUDA typecheck (cli): passed
fmt + diff --check: passed
```

`all_dead_corpus_terminates_at_launch_cap` pins the load-bearing invariant: an
impossible (every-group-zero-variance) corpus stops at the launch cap with zero
effective groups and no optimizer step — no infinite refill loop.

## Problems

End-to-end DAPO acceptance is `#[cfg(cuda)]` (the whole round loop) → H20
pending-remote: (1) a mixed corpus reaches the effective batch by replacement;
(2) an all-zero-variance corpus terminates at the budget with no step; (3) a
normal-path DAPO update is nonempty with finite ratio/KL telemetry.

## Learnings

**Dynamic sampling lives in the launcher, not the loss.** Filtering a dead group
out of the batch (post-hoc) leaves the round short a slot; DAPO's point is to
refill that slot. Keeping `UpdatePreset` responsible only for the math and moving
refill to the orchestration loop is the separation that lets the same launcher
serve every preset — and makes deterministic termination a property of one pure
`RefillBudget`, not the OOM-sensitive CUDA loop.
