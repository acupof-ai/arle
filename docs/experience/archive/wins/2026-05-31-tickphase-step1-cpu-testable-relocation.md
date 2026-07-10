# TickPhase Step 1 — relocate pure classifier so the CPU-testability proof actually runs

## Context

Step 1 of [`docs/plans/control-plane-scheduler-phase.md`](../../plans/control-plane-scheduler-phase.md)
landed in commit `1ca20ddd` as a pure addition to
`infer/src/scheduler/cuda/execution.rs`: `enum TickPhase`, a pure free fn
`classify_tick_phase(bool, bool, bool) -> TickPhase`, the `&self` wrapper
`Scheduler::tick_phase`, and a CPU sweep test. The design's headline property
(doc:66-71) is the `oplib::linear::plan` parallel — a GPU-free
`assert_eq!(tick_phase(state), …)` unit test that **runs** on the dev box
without CUDA.

That property was not actually achieved: the entire `scheduler/cuda` subtree is
`#[cfg(feature = "cuda")]` (`scheduler.rs:25`), so the "CPU-testable" tests
were compiled out under `--features no-cuda` (filtered, **0 passed**) and the
test binary link-failed under `--features cuda,no-cuda` on a Mac with no nvcc.
The tests ran on **no** configuration available locally — see the paired errors
entry [`2026-05-31-tickphase-test-behind-feature-gate-never-ran`](../errors/2026-05-31-tickphase-test-behind-feature-gate-never-ran.md).

## What Worked

Relocated the **pure** parts — `enum TickPhase` + `classify_tick_phase` + the
two sweep tests — into a new **non-feature-gated** module
`infer/src/scheduler/tick_phase.rs`, declared `mod tick_phase;` in
`scheduler.rs` right after `mod types;` (i.e. a sibling of the cuda-gated `mod
cuda`, mirroring exactly where `oplib` sits relative to its gating). The
cuda-only `&self` wrapper `Scheduler::tick_phase` stays in `execution.rs` and
now `use`s the relocated `TickPhase` + `classify_tick_phase`. `step()` is left
byte-identical. The enum + free fn carry `#[cfg_attr(not(test),
allow(dead_code))]` because under a non-test no-cuda build neither caller is
compiled.

Verification (all run locally on M-series Mac, no nvcc):

```
# The headline fix — tests now genuinely RUN under no-cuda (was 0 passed / filtered):
$ cargo test -p infer --no-default-features --features no-cuda classify_tick_phase
running 2 tests
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 634 filtered out

# step() byte-identical (extract step body from HEAD vs worktree, diff):
HEAD step body lines: 210 ; worktree: 210 → BYTE-IDENTICAL

# Canonical Mac typecheck + clippy gate, clean:
$ cargo check  -p infer --no-default-features --features cuda,no-cuda      # Finished, ok
$ cargo clippy -p infer --no-default-features --features cuda,no-cuda -- -D warnings   # exit 0, 0 warnings
# My files emit zero clippy diagnostics under both no-cuda and cuda,no-cuda.
```

Diff: `scheduler.rs` +1 (`mod tick_phase;`); `execution.rs` +1 import / −141
(enum+classify+oracle+2 classify tests moved out; the 15 budget tests in its
`mod tests` are preserved untouched); new `scheduler/tick_phase.rs`.

## Bench exemption

In-scope path (`infer/src/`) but **bench-exempt**: `tick_phase` still has zero
non-test call sites (`step()` is not rewired — that is Step 2), so it is
provably unreachable on the hot path and the relocation is a pure code-move +
module declaration. A guidellm run would measure a bit-identical binary
(`step()` byte-identical, confirmed). This commit also closes the
missing-wins-entry gap from `1ca20ddd`, which shipped no wins entry.

## Rule

A CPU-testability proof must live in a module that compiles **and runs** on the
dev platform's `no-cuda` feature set. A "GPU-free" test placed behind a
platform feature-gate (`#[cfg(feature = "cuda")]`) runs **nowhere** on a
non-CUDA host — it is filtered under `no-cuda` and link-fails under `cuda` with
no nvcc. Put pure, host-neutral logic + its tests in a non-gated sibling module
(the `oplib` pattern); keep only the device-touching wrapper behind the gate.
See [[feedback_docs_are_not_truth]] — the commit message claimed "2 passed
under no-cuda"; the command actually filtered. Verify a test RUNS by reading the
`N passed` count, never the exit code.
