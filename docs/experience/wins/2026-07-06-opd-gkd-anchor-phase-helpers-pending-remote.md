# OPD step: 490-line gkd_anchor split into phase helpers

`bench-exempt` — mechanical extraction, zero behavior change. cuda
`pending-remote` (no nvcc); non-cuda build + train test suite green.

## Context
`opd_step_with_teacher_forward_profiled_gkd_anchor` was ~490 lines: 6 phases + 2
early-return routes inline in one closure. The 5-layer overload chain above it
has live external callers (CLI, examples, tests), so collapsing it is a separate
pod-gated task (P6b plan); this split is API-preserving.

## What Worked
- Extracted `run_opd_rollout_phase`, `run_windowed_gkd_route`,
  `run_chunked_kl_route` (+ borrowed `GkdRouteCtx`). Main closure now reads
  rollout → windowed? → chunked-KL? → teacher/student/KL/backward.
- The `let result = (||{…})();` wrapper + post-step `cleanup_after_backward` stay
  in the outer function (ordering is load-bearing,
  `errors/2026-05-21-arle-cuda-opd-post-step-cleanup-kill.md`).
- Parent 490 → 251 lines. Signature + overload chain + callers unchanged.
- Verified: `cargo test -p train --features no-cuda` green; side-by-side re-read
  confirms cuda-gated blocks are byte-parity (statements/args/ordering).

## Rule
Extraction is licensed by byte-identical ordering, not cleverness. Don't collapse
an overload chain with live callers without a path to behavior verification.

## Renamed 2026-08-13

The helpers this entry names were renamed in `18096ec7f`..HEAD; `route` was
dropped because the word already means MoE expert routing in this crate, and
both functions run a complete step body (zero_grad → backward →
finite_optimizer_step → `OpdStepOutcome`) rather than a segment of one.

| Then | Now |
|------|-----|
| `run_windowed_gkd_route` | `windowed_gkd_step` |
| `run_chunked_kl_route` | `chunked_kl_step` |
| `GkdRouteCtx` | `GkdStepCtx` |
| `run_opd_rollout_phase` | `rollout_phase` |

User-facing error text still says "Route B" for the windowed path.
