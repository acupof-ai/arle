# TP two-phase commit converged from three inline copies into one seam operation

Date: 2026-09-10 · Lane: lane/d2 · Step 2 of the architecture refactor

## Context

The KV tier demote/promote paths in `executor/qwen35.rs` run a cross-rank
two-phase commit: each rank attempts locally, a scalar min-reduce is the
verdict, and disagreement either aborts the round or rolls the local attempt
back. The shape was inlined four times (two rounds each in `demote_slot` and
`promote_slot`), and the rollback decision — the only part with transaction
semantics — sat inside the backend, so none of the four paths
(agree / peer-fails / local-fails / rollback-fails) could be tested without a
GPU and a multi-rank setup.

## What worked

- **One named operation on the seam.** `infer_seam::agree_abort` and
  `infer_seam::agree_rollback` take the local result and an injected `agree`
  closure. Production passes the backend's min-reduce; tests pass a fake that
  constructs partial-rank failure. The four demote/promote rounds are now
  one-liners; the protocol has one implementation.
- **The collective runs on every path, structurally.** `agree` is called
  before the verdict branch with the local ok-flag as input, so a rank whose
  attempt failed still enters the reduce — the lockstep the old code
  maintained by hand. A test asserts the collective saw `0` on local failure.
- **Borrow narrowing fell out of the convergence.** `tp_min_usize` took
  `&self`, which made the agree closure borrow all of `self` and conflict with
  the rollback closure's `&mut self.slot_tier`. It now takes `&Qwen35Model`,
  so the closures capture disjoint fields.
- **Mutation negative control.** Flipping the verdict check (`== 0` → `== 1`)
  and the rollback guard (`if local` → `if !local`) each turned the suite red;
  reverted via reverse sed, 8/8 green after.

## Rule

A cross-rank commit protocol is a named seam operation, not inline backend
code: `attempt`/`rollback` are values, `agree` is the injected collective, and
the host half is testable on Mac with a fake agree. The value of extracting a
boundary is the duplicated structure collapsing from N copies to one — the
`cargo tree` gate proves the boundary is right, the dedup proves it paid off.

## Note

`restore_recurrent_sidecar` was named in the original triplication report but
carries no TP round on inspection — the repeated shape lived in
demote/promote only.
