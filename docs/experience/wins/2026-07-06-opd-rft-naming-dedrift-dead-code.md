# OPD/RFT naming de-drift + dead rubric_writeback_ce_step deleted

`bench-exempt` — dead-code deletion + docs, no hot-path change.

## Context
Two issues: `rubric_writeback_ce_step` (singular) had zero callers
(`run_rubric_rounds` uses `_batched`); and "OPD-only" reads as "all distillation",
but `agent-opd`/`rubric-opd` do reward-selected trajectories + masked CE — no
teacher, no KL. They are on-policy RFT sharing the OPD substrate.

## What Worked
- Deleted `rubric_writeback_ce_step` (grep confirmed zero callers).
- Naming de-drifted in `lib.rs` header, `architecture.md` train row, and the
  pivot doc: OPD (`opd`/`self-opd`) vs RFT (`agent-opd`/`rubric-opd`).
- Kept byte-identical baselines intact: `masked_writeback_ce_step` +
  `_frozen_prompt_kv` (A/B, guarded by `test_frozen_prompt_kv_writeback.rs`) are
  NOT merged — the review's "unify the 4 paths" instinct was wrong.

## Rule
Before unifying near-duplicate functions, read the docs and count callers: one
may be a deliberate byte-identical baseline (keep), another dead (delete) —
opposite actions.
