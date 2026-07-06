# OPD/RFT naming de-drift + dead rubric_writeback_ce_step deleted

`bench-exempt` — dead-code deletion of a zero-caller function plus doc/comment
edits. No hot-path change (the function was never called; `_batched` is the live
path). No latency delta.

## Context

The review of the OPD stack found two naming/dead-code issues:
1. `crates/train/src/opd.rs::rubric_writeback_ce_step` (singular) had **zero
   callers** — `run_rubric_rounds` uses the `_batched` sibling exclusively. Its
   doc even claimed it "serves both Mode A and Mode B", but nothing wired it.
2. **Name drift**: "OPD-only" reads as "everything is On-Policy Distillation",
   but `agent-opd` / `rubric-opd` do execution/rubric-reward trajectory
   selection + completion/response-masked CE — **no teacher forward, no KL**.
   They are on-policy RFT/rejection-sampling sharing the OPD substrate, not
   distillation.

## What Worked

- **Deleted** `rubric_writeback_ce_step` (singular, ~62 lines incl. doc). Grep
  confirmed no callers in `crates/{train,cli}`, tests, or examples before
  removal; `_batched` remains the live rubric writeback.
- **De-drifted the naming** in three places without touching behavior:
  - `crates/train/src/lib.rs` header — spells out the two objective families
    (OPD: `opd`/`self-opd` = teacher/EMA + KL; RFT: `agent-opd`/`rubric-opd` =
    reward-selected + masked CE) and that the `opd` in RFT subcommand names is
    the shared substrate, not the objective.
  - `docs/architecture.md` `train` row — same two-family split.
  - `docs/projects/2026-05-18-opd-only-pivot.md` — a dated naming-clarification
    note under the status header.
- **Kept the byte-identical baselines intact**: `masked_writeback_ce_step`,
  `_frozen_prompt_kv`, and `_dispatch` are deliberate A/B baseline + opt-in
  variant (guarded by `test_frozen_prompt_kv_writeback.rs`) — NOT merged. The
  review's first instinct ("unify the 4 CE writeback paths") was wrong; merging
  would destroy the byte-identical baseline the A/B gate depends on.

Verification: `cargo test -p train --features no-cuda` 163 lib + all integration
green; `cargo clippy -p train` clean on opd.rs.

## Rule

Before "unifying near-duplicate functions", read the doc comments and count
callers: a function that looks redundant may be a deliberate byte-identical A/B
baseline (keep it) or genuinely dead (delete it) — those are opposite actions.
Naming drift ("OPD-only" covering non-distillation RFT) is fixed in docs +
module header, not by renaming subcommands that users already depend on.
