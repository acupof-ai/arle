//! Pending decode/prefill bookkeeping types + tier prefetch + store-dedup keys.
//!
//! Split out of `core.rs` (pure structural refactor — no behavior change).
//! These structs travel together as the scheduler's per-step "in-flight"
//! state carried across loop turns.

use crate::types::BlockFingerprint;
use fastrace::Span;

pub(in crate::scheduler::cuda) struct FirstBatchMode {
    pub active: bool,
    pub started_at: Option<std::time::Instant>,
    pub prefill_rows: u64,
    pub prefill_tokens: u64,
}

impl FirstBatchMode {
    pub(in crate::scheduler::cuda) fn new() -> Self {
        Self {
            active: false,
            started_at: None,
            prefill_rows: 0,
            prefill_tokens: 0,
        }
    }

    pub(in crate::scheduler::cuda) fn start(&mut self) {
        self.active = true;
        self.started_at = Some(std::time::Instant::now());
        self.prefill_rows = 0;
        self.prefill_tokens = 0;
    }

    pub(in crate::scheduler::cuda) fn record_prefill(&mut self, rows: u64, tokens: u64) {
        if self.active {
            self.prefill_rows = self.prefill_rows.saturating_add(rows);
            self.prefill_tokens = self.prefill_tokens.saturating_add(tokens);
        }
    }

    pub(in crate::scheduler::cuda) fn finish(&mut self) -> (u128, u64, u64) {
        let duration_us = self
            .started_at
            .take()
            .map_or(0, |started_at| started_at.elapsed().as_micros());
        let prefill_rows = self.prefill_rows;
        let prefill_tokens = self.prefill_tokens;
        self.active = false;
        self.prefill_rows = 0;
        self.prefill_tokens = 0;
        (duration_us, prefill_rows, prefill_tokens)
    }
}

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

#[cfg(test)]
mod tests {
    use super::{FirstBatchMode, SchedulerRuntimeStats};

    #[test]
    fn first_batch_mode_records_prefill_work_until_finish() {
        let mut mode = FirstBatchMode::new();

        mode.record_prefill(1, 128);
        assert_eq!(mode.prefill_rows, 0);

        mode.start();
        mode.record_prefill(2, 4096);
        mode.record_prefill(1, 1024);

        let (_duration_us, rows, tokens) = mode.finish();
        assert_eq!((rows, tokens), (3, 5120));
        assert!(!mode.active);
        assert_eq!((mode.prefill_rows, mode.prefill_tokens), (0, 0));
    }

    #[test]
    fn loop_total_timing_includes_work_after_step_phase() {
        let mut stats = SchedulerRuntimeStats::new();
        stats.step_timing_total_us = 100.0;

        stats.record_loop_phase_timing(10, 125);

        assert!(stats.step_timing_loop_total_us > stats.step_timing_total_us);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(in crate::scheduler::cuda) struct StoreDedupKey {
    pub fingerprint: BlockFingerprint,
    pub target: crate::kv_tier::StoreTarget,
}
