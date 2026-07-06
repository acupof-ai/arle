# OPD step: 490-line gkd_anchor split into phase helpers (API-preserving)

`bench-exempt` — pure mechanical extraction of one function's body into private
helpers; zero behavior change (every statement, argument, cfg gate, and ordering
preserved). No hot-path/latency delta. CUDA behavior is `pending-remote` (no
nvcc on this Mac); non-CUDA build + full train test suite green locally.

## Context

`opd_step_with_teacher_forward_profiled_gkd_anchor` (`crates/train/src/opd.rs`)
was a ~490-line function with 6 phases + 2 early-return routes (windowed,
chunked-KL) inline in one closure — the exact "can't parse the intent in one
pass" shape AGENTS.md's code-as-poetry rule flags. The 5-layer overload chain
above it (`_teacher_forward` → `_profiled` → `_profiled_gkd` → `_gkd_anchor`) was
left intact: every layer has direct external callers (CLI ×4, examples, 6 test
sites), so collapsing it is a public-API break with no local behavior
verification — deferred to a plan doc (P6b), not done here.

## What Worked

- Extracted the two early-return routes into private
  `run_windowed_gkd_route<O,T>` and `run_chunked_kl_route<O,T>` returning
  `Result<OpdStepOutcome>`, plus the rollout phase into `run_opd_rollout_phase`.
- A private borrowed-context struct `GkdRouteCtx<'a, T>` carries the shared
  `&`/Copy fields (student, teacher, prompt_ids, cfg, gkd_config, param slices,
  engine_offload, rollout, positions, cfg-gated `infer_rollout`), keeping the
  helper arg counts sane; `&mut` refs (store, tape, optimizer, profile) stay
  explicit params.
- The main closure now reads as a linear sequence: rollout → `if
  logits_window_size { return run_windowed_gkd_route }` → `if kl_chunk_size &&
  lambda==0 { return run_chunked_kl_route }` → teacher/student/KL/backward. The
  `let result = (||{…})();` wrapper and post-step `cleanup_after_backward` stay in
  the outer function unchanged (its ordering is load-bearing —
  `errors/2026-05-21-arle-cuda-opd-post-step-cleanup-kill.md`).
- **Parent function 490 → 251 lines (−48%).** Public signature + all 4 overload
  wrappers + every caller untouched.

Verification: `cargo test -p train --features no-cuda` 163 lib + all integration
green; `cargo clippy -p train` no new opd.rs warnings; final side-by-side re-read
of both cuda-gated helper blocks confirms exact statement/arg/ordering parity.
CUDA build pending-remote (cudarc needs nvcc). A blocker surfaced — the repo's
concurrent WIP in `infer-server` (a `stream_options` field added to the struct
but not the `anthropic.rs` initializer) breaks the `train` dep chain; verified
against a temporary local fix that was reverted, leaving that WIP untouched.

## Rule

Mechanical extraction of a giant function is licensed by "zero behavior change +
byte-identical ordering", not by cleverness — pull the early-return routes into
named helpers with a borrowed-context struct, keep the load-bearing
cleanup/`let result` scaffold in the parent, and prove it with the existing test
suite. Don't collapse an overload chain whose layers have live external callers
without a path to behavior verification.
