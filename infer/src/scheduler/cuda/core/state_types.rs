//! Pending decode/prefill bookkeeping types + tier prefetch + store-dedup keys.
//!
//! Split out of `core.rs` (pure structural refactor — no behavior change).
//! These structs travel together as the scheduler's per-step "in-flight"
//! state carried across loop turns.

use crate::types::BlockFingerprint;
use fastrace::Span;
use std::time::Instant;

pub(in crate::scheduler::cuda) struct PendingDecode {
    pub decode_indices: Vec<usize>,
    pub slot_indices: Vec<usize>,
    /// True only when `sample_batch_greedy_launch` actually fired the argmax kernel.
    pub greedy_launched: bool,
    pub decode_spans: Vec<(usize, Span)>,
    pub mixed_prefill: Option<PendingMixedPrefill>,
}

pub(in crate::scheduler::cuda) struct PendingPrefillRow {
    pub slot_idx: usize,
    pub total_tokens: usize,
    pub next_progress: usize,
}

pub(in crate::scheduler::cuda) struct PendingPrefill {
    pub rows: Vec<PendingPrefillRow>,
    pub uses_paged: bool,
    pub prefill_spans: Vec<(usize, Span)>,
}

pub(in crate::scheduler::cuda) struct PendingMixedPrefill {
    pub rows: Vec<PendingPrefillRow>,
    pub uses_paged: bool,
    pub prefill_spans: Vec<(usize, Span)>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::scheduler::cuda) struct PrefetchTicketState {
    pub host_blocks: usize,
    pub disk_blocks: usize,
    pub remote_blocks: usize,
}

/// Runtime counters and local profiling state owned by the scheduler thread.
pub(in crate::scheduler::cuda) struct SchedulerRuntimeStats {
    /// Lifetime completed request count.
    pub total_completed: u64,
    /// Lifetime generated token count.
    pub total_generated_tokens: u64,
    /// EMA step timing (microseconds) for `/v1/stats` profiling.
    pub step_timing_admission_us: f64,
    pub step_timing_decode_us: f64,
    pub step_timing_emit_us: f64,
    pub step_timing_prefill_us: f64,
    pub step_timing_total_us: f64,
    pub step_timing_cleanup_us: f64,
    pub step_timing_loop_total_us: f64,
    /// Throttled GPU memory query state and peak high-water mark.
    pub last_mem_query: std::time::Instant,
    pub peak_mem_bytes: u64,
    /// Set when a prefill batch fails with an out-of-memory error.
    /// While this is in the future, `assign_slots` serializes new
    /// prefill admits (one at a time, only when no GPU work is in
    /// flight) so a transient workspace shortage doesn't cascade into
    /// every subsequent request OOMing too.
    pub prefill_oom_cooldown_until: Option<std::time::Instant>,
    /// Startup burst mode: once a fresh queue begins with no runnable decode
    /// rows, drain prefills contiguously before resuming mixed scheduling.
    pub first_batch_prefill: FirstBatchPrefillStats,
}

impl SchedulerRuntimeStats {
    pub(in crate::scheduler::cuda) fn new() -> Self {
        Self {
            total_completed: 0,
            total_generated_tokens: 0,
            step_timing_admission_us: 0.0,
            step_timing_decode_us: 0.0,
            step_timing_emit_us: 0.0,
            step_timing_prefill_us: 0.0,
            step_timing_total_us: 0.0,
            step_timing_cleanup_us: 0.0,
            step_timing_loop_total_us: 0.0,
            last_mem_query: std::time::Instant::now(),
            peak_mem_bytes: 0,
            prefill_oom_cooldown_until: None,
            first_batch_prefill: FirstBatchPrefillStats::default(),
        }
    }

    pub(in crate::scheduler::cuda) fn record_loop_phase_timing(
        &mut self,
        cleanup_us: u128,
        loop_total_us: u128,
    ) {
        fn update_ema(ema: &mut f64, val: u128) {
            const ALPHA: f64 = 0.1;
            let v = val as f64;
            if *ema == 0.0 {
                *ema = v;
            } else {
                *ema = ALPHA * v + (1.0 - ALPHA) * *ema;
            }
        }

        update_ema(&mut self.step_timing_cleanup_us, cleanup_us);
        update_ema(&mut self.step_timing_loop_total_us, loop_total_us);
    }
}

#[derive(Debug, Default)]
pub(in crate::scheduler::cuda) struct FirstBatchPrefillStats {
    pub active: bool,
    pub started_at: Option<Instant>,
    pub prefill_rows: u64,
    pub prefill_tokens: u64,
    cohort: Vec<(usize, u64)>,
    sealed: bool,
}

impl FirstBatchPrefillStats {
    pub(in crate::scheduler::cuda) fn maybe_enter(
        &mut self,
        no_decode_running: bool,
        has_prefill_candidates: bool,
    ) -> bool {
        if self.active || !no_decode_running || !has_prefill_candidates {
            return false;
        }
        self.active = true;
        self.started_at = Some(Instant::now());
        self.prefill_rows = 0;
        self.prefill_tokens = 0;
        self.cohort.clear();
        self.sealed = false;
        true
    }

    pub(in crate::scheduler::cuda) fn include_candidate(
        &mut self,
        slot_idx: usize,
        request_id: u64,
    ) {
        if !self.active || self.sealed {
            return;
        }
        let key = (slot_idx, request_id);
        if !self.cohort.contains(&key) {
            self.cohort.push(key);
        }
    }

    pub(in crate::scheduler::cuda) fn seal(&mut self) {
        if self.active {
            self.sealed = true;
        }
    }

    pub(in crate::scheduler::cuda) fn contains_candidate(
        &self,
        slot_idx: usize,
        request_id: u64,
    ) -> bool {
        self.cohort.contains(&(slot_idx, request_id))
    }

    pub(in crate::scheduler::cuda) fn record_prefill_step(&mut self, rows: u64, tokens: u64) {
        if self.active {
            self.prefill_rows += rows;
            self.prefill_tokens += tokens;
        }
    }

    pub(in crate::scheduler::cuda) fn maybe_exit(
        &mut self,
        has_prefill_candidates: bool,
        runnable_decode_rows: usize,
    ) -> Option<(u128, u64, u64, usize)> {
        if !self.active || has_prefill_candidates {
            return None;
        }
        self.active = false;
        self.cohort.clear();
        self.sealed = false;
        let elapsed_ms = self
            .started_at
            .take()
            .map(|started| started.elapsed().as_millis())
            .unwrap_or(0);
        Some((
            elapsed_ms,
            self.prefill_rows,
            self.prefill_tokens,
            runnable_decode_rows,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{FirstBatchPrefillStats, SchedulerRuntimeStats};

    #[test]
    fn loop_total_timing_includes_work_after_step_phase() {
        let mut stats = SchedulerRuntimeStats::new();
        stats.step_timing_total_us = 100.0;

        stats.record_loop_phase_timing(10, 125);

        assert!(stats.step_timing_loop_total_us > stats.step_timing_total_us);
    }

    #[test]
    fn first_batch_prefill_state_drains_until_no_prefill_candidates() {
        let mut stats = FirstBatchPrefillStats::default();

        assert!(stats.maybe_enter(true, true));
        assert!(stats.active);
        stats.include_candidate(0, 10);
        stats.seal();
        stats.include_candidate(1, 11);
        assert!(stats.contains_candidate(0, 10));
        assert!(!stats.contains_candidate(1, 11));
        stats.record_prefill_step(2, 4096);
        assert_eq!(stats.prefill_rows, 2);
        assert_eq!(stats.prefill_tokens, 4096);

        assert_eq!(stats.maybe_exit(true, 1), None);
        assert!(stats.active);

        let Some((_elapsed_ms, rows, tokens, decode_rows)) = stats.maybe_exit(false, 16) else {
            panic!("first-batch mode should exit after prefill candidates drain");
        };
        assert_eq!(rows, 2);
        assert_eq!(tokens, 4096);
        assert_eq!(decode_rows, 16);
        assert!(!stats.active);
        assert!(!stats.contains_candidate(0, 10));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(in crate::scheduler::cuda) struct StoreDedupKey {
    pub fingerprint: BlockFingerprint,
    pub target: crate::kv_tier::StoreTarget,
}
