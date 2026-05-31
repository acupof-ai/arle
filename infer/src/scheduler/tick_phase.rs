//! Pure, GPU-free classification of which phase one scheduler tick (`step()`)
//! is in.
//!
//! `step()` (`scheduler/cuda/execution.rs`) is already a state machine — three
//! `Option` fields (`pending_prefill`, `pending_decode`, `deferred_decode_emit`)
//! plus early-returns encode three branches. This module names those branches
//! ([`TickPhase`]) and isolates the classification into a pure function
//! ([`classify_tick_phase`]) so it can be exhaustively unit-tested on CPU with
//! no GPU and no `cuda` feature — the same CPU-testability discipline the
//! compute plane has via `oplib::linear::plan`.
//!
//! This module is intentionally **not** behind `#[cfg(feature = "cuda")]`: the
//! classification is over plain booleans, so it must compile and its sweep test
//! must *run* under the dev platform's `no-cuda` feature set. The `&self`
//! wrapper that reads the booleans off a live `Scheduler` lives in the
//! cuda-gated `scheduler/cuda/execution.rs` (`Scheduler::tick_phase`), since it
//! depends on cuda-only scheduler fields. See
//! `docs/plans/control-plane-scheduler-phase.md`.

/// The three states of one scheduler tick (`step()`), named.
///
/// `step()` (`scheduler/cuda/execution.rs`) branches on three `Option` fields
/// (`pending_prefill`, `pending_decode`, `deferred_decode_emit`) plus
/// early-returns. This enum names those branches so the readback→launch cycle
/// is legible and the "cleared before launch" invariant becomes a match-arm
/// fact rather than a runtime `assert!`. See
/// `docs/plans/control-plane-scheduler-phase.md`.
///
/// Built and consumed by `Scheduler::tick_phase` (cuda-gated wrapper) and the
/// pure [`classify_tick_phase`]; under a non-`cuda`, non-test build neither
/// caller is compiled, hence `allow(dead_code)`.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TickPhase {
    /// A prior prefill launch is in flight; this tick reads it back.
    /// Entered when `pending_prefill.is_some()`.
    ReadbackPrefill,
    /// A prior decode launch (or a deferred decode emit that requires a
    /// readback before the next launch) is in flight; this tick reads it back.
    /// Entered when `pending_decode.is_some()
    /// || deferred_decode_requires_readback_before_launch()` and no prefill
    /// readback is pending.
    ReadbackDecode,
    /// No readback is outstanding: this tick plans and launches GPU work
    /// (snapshot → build_candidate_plan → launch_gpu_command →
    /// dispatch_decode_emits).
    PlanAndLaunch,
}

/// PURE. Classify a tick from the three pending-state predicates `step()`
/// reads, in `step()`'s exact branch order.
///
/// Inputs are the already-evaluated booleans (no `self`, no device, no I/O),
/// so this is GPU-free and CPU-testable — the killer property the unit-test
/// sweep below exercises. `Scheduler::tick_phase` (cuda-gated) is the thin
/// `&self` wrapper that computes these three booleans off the scheduler's own
/// fields.
///
/// - `pending_prefill_some` = `self.pending_prefill.is_some()`
///   (the `step()` prefill-readback guard, checked first).
/// - `pending_decode_some` = `self.pending_decode.is_some()`.
/// - `deferred_requires_readback` =
///   `self.deferred_decode_requires_readback_before_launch()`.
///
/// [`TickPhase::ReadbackPrefill`] takes precedence (prefill is checked first in
/// `step()` and early-returns), then [`TickPhase::ReadbackDecode`], then
/// [`TickPhase::PlanAndLaunch`].
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(super) fn classify_tick_phase(
    pending_prefill_some: bool,
    pending_decode_some: bool,
    deferred_requires_readback: bool,
) -> TickPhase {
    if pending_prefill_some {
        TickPhase::ReadbackPrefill
    } else if pending_decode_some || deferred_requires_readback {
        TickPhase::ReadbackDecode
    } else {
        TickPhase::PlanAndLaunch
    }
}

#[cfg(test)]
mod tests {
    use super::{TickPhase, classify_tick_phase};

    /// Reference oracle for [`classify_tick_phase`], written directly against
    /// `step()`'s control flow (`scheduler/cuda/execution.rs`): prefill
    /// readback is the first guard (`pending_prefill.is_some()`) and
    /// early-returns, so it dominates; then the decode-readback launch guard
    /// (`pending_decode.is_some() || deferred_decode_requires_readback_before_launch()`);
    /// otherwise the tick falls through to plan-and-launch. If
    /// `classify_tick_phase` ever diverges from `step()`'s branch order this
    /// oracle catches it.
    fn step_branch_oracle(
        pending_prefill_some: bool,
        pending_decode_some: bool,
        deferred_requires_readback: bool,
    ) -> TickPhase {
        // step():935 — prefill readback guard, checked first, early-returns.
        if pending_prefill_some {
            return TickPhase::ReadbackPrefill;
        }
        // step():968 — decode readback launch guard.
        if pending_decode_some || deferred_requires_readback {
            return TickPhase::ReadbackDecode;
        }
        // step():990+ — plan → launch fall-through.
        TickPhase::PlanAndLaunch
    }

    /// The headline CPU property: over every combination of the three
    /// pending-state predicates `step()` reads, `classify_tick_phase` returns
    /// exactly what `step()`'s branch order would select. Pure, no GPU, runs
    /// under the default / `no-cuda` feature set.
    #[test]
    fn classify_tick_phase_matches_step_branch_order_over_full_sweep() {
        for &pending_prefill_some in &[false, true] {
            for &pending_decode_some in &[false, true] {
                for &deferred_requires_readback in &[false, true] {
                    assert_eq!(
                        classify_tick_phase(
                            pending_prefill_some,
                            pending_decode_some,
                            deferred_requires_readback,
                        ),
                        step_branch_oracle(
                            pending_prefill_some,
                            pending_decode_some,
                            deferred_requires_readback,
                        ),
                        "tick phase diverged for (prefill={pending_prefill_some}, \
                         decode={pending_decode_some}, deferred={deferred_requires_readback})"
                    );
                }
            }
        }
    }

    /// Spot-check the documented-expected phase for the canonical states so a
    /// regression in the oracle itself can't silently mask a classifier
    /// regression.
    #[test]
    fn classify_tick_phase_documented_expected_for_canonical_states() {
        // No pending work → plan-and-launch.
        assert_eq!(
            classify_tick_phase(false, false, false),
            TickPhase::PlanAndLaunch
        );
        // Prefill readback dominates even if decode/deferred also pending.
        assert_eq!(
            classify_tick_phase(true, false, false),
            TickPhase::ReadbackPrefill
        );
        assert_eq!(
            classify_tick_phase(true, true, true),
            TickPhase::ReadbackPrefill
        );
        // Decode readback when no prefill but decode pending.
        assert_eq!(
            classify_tick_phase(false, true, false),
            TickPhase::ReadbackDecode
        );
        // Deferred-emit-requires-readback also routes to decode readback.
        assert_eq!(
            classify_tick_phase(false, false, true),
            TickPhase::ReadbackDecode
        );
        // Both decode predicates set → still decode readback.
        assert_eq!(
            classify_tick_phase(false, true, true),
            TickPhase::ReadbackDecode
        );
    }
}
