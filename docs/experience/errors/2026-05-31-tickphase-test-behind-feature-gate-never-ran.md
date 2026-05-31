# TickPhase "CPU-testable" tests were behind a platform feature-gate → ran nowhere

## Context

Commit `1ca20ddd` (Step 1 of
[`control-plane-scheduler-phase.md`](../../plans/control-plane-scheduler-phase.md))
added a pure `classify_tick_phase` + a CPU sweep test inside
`infer/src/scheduler/cuda/execution.rs`. The whole point (design doc:66-71,
the `oplib::linear::plan` parallel) was a **GPU-free** `assert_eq!` test that
**runs** on the dev box without CUDA. The commit message asserted:
`...--features no-cuda ... => 2 passed`.

That claim was non-reproducible. `scheduler/cuda` is `#[cfg(feature = "cuda")]`
(`scheduler.rs:25`), so:

- `cargo test -p infer --no-default-features --features no-cuda tick_phase`
  → `0 passed; N filtered out` (the test is cfg-compiled-out, never run) — **yet
  `cargo` exits 0**, which looks green.
- `cargo test -p infer --no-default-features --features cuda,no-cuda …`
  → test binary **link-fails** on a Mac with no nvcc (missing CUDA kernel
  symbols).

So the "CPU-testability proof" executed on **no** locally-available config. The
compute-plane comparator `oplib::linear` — which lives in a non-gated module —
*does* run under no-cuda (`5 passed; 624 filtered out`), which is exactly the
property Step 1 was supposed to copy and didn't.

## Root Cause

Two compounding mistakes:

1. **Placed a purity proof behind a platform feature-gate.** Pure logic over
   plain booleans was put inside the cuda-gated module next to the `&self`
   wrapper that genuinely needs the gate. The pure part has no CUDA dependency
   and should never have inherited the gate.
2. **Conflated "`cargo` exited 0" with "the test ran."** `0 passed; M filtered
   out` exits 0. The verification read the exit code, not the `N passed` count,
   so a test that never executed was reported as "2 passed."

## Fix

Relocated the pure `enum TickPhase` + `classify_tick_phase` + the two sweep
tests into a new **non-gated** module `infer/src/scheduler/tick_phase.rs`
(declared `mod tick_phase;` in `scheduler.rs`, sibling of the gated `mod cuda`,
mirroring `oplib`). The device-touching `Scheduler::tick_phase` wrapper stays in
`execution.rs` and imports the relocated items. `step()` byte-identical. Now
`cargo test -p infer --no-default-features --features no-cuda classify_tick_phase`
→ `2 passed; 634 filtered out` (genuinely runs). See the paired win
[`2026-05-31-tickphase-step1-cpu-testable-relocation`](../wins/2026-05-31-tickphase-step1-cpu-testable-relocation.md).

## Rule

Verify a test **runs** by checking the `N passed` count matches the number you
expect — `0 passed; M filtered out` also exits 0, so a green exit code proves
nothing for a feature-gated test. Never put GPU-free / host-neutral logic (or
its CPU tests) behind `#[cfg(feature = "cuda")]`; keep it in a non-gated sibling
module and gate only the device-touching wrapper. When a design names a
specific testability property (here: "runs under no-cuda like `oplib`"), the
verification must reproduce that **exact** command and read its pass count.
