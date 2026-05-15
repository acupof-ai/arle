//! Pending decode/prefill bookkeeping types + tier prefetch + store-dedup keys.
//!
//! Split out of `core.rs` (pure structural refactor — no behavior change).
//! These structs travel together as the scheduler's per-step "in-flight"
//! state carried across loop turns.

use crate::types::BlockFingerprint;
use fastrace::Span;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::scheduler::cuda) enum GpuStageKind {
    Prefill,
    Readback,
}

impl GpuStageKind {
    pub(in crate::scheduler::cuda) fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Readback => "readback",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::scheduler::cuda) enum GpuStageState {
    Queued,
    InFlight,
    Ready,
    Completed,
}

#[derive(Clone, Debug)]
pub(in crate::scheduler::cuda) struct GpuStageHandle {
    pub id: u64,
    pub kind: GpuStageKind,
    pub state: GpuStageState,
    pub slots: Vec<usize>,
    pub rows: usize,
    pub queued_at: std::time::Instant,
    pub launched_at: Option<std::time::Instant>,
    pub ready_at: Option<std::time::Instant>,
}

impl GpuStageHandle {
    pub(in crate::scheduler::cuda) fn new(id: u64, kind: GpuStageKind, slots: Vec<usize>) -> Self {
        Self {
            id,
            kind,
            state: GpuStageState::Queued,
            rows: slots.len(),
            slots,
            queued_at: std::time::Instant::now(),
            launched_at: None,
            ready_at: None,
        }
    }

    pub(in crate::scheduler::cuda) fn mark_inflight(&mut self) {
        log::trace!(
            "gpu stage {} {} inflight rows={} queued_us={}",
            self.id,
            self.kind.as_str(),
            self.rows,
            self.queued_at.elapsed().as_micros(),
        );
        self.state = GpuStageState::InFlight;
        self.launched_at = Some(std::time::Instant::now());
    }

    pub(in crate::scheduler::cuda) fn mark_ready(&mut self) {
        let launch_us = self
            .launched_at
            .map_or(0, |launched_at| launched_at.elapsed().as_micros());
        log::trace!(
            "gpu stage {} {} ready rows={} launch_us={}",
            self.id,
            self.kind.as_str(),
            self.rows,
            launch_us,
        );
        self.state = GpuStageState::Ready;
        self.ready_at = Some(std::time::Instant::now());
    }

    pub(in crate::scheduler::cuda) fn mark_completed(&mut self) {
        let ready_us = self
            .ready_at
            .map_or(0, |ready_at| ready_at.elapsed().as_micros());
        log::trace!(
            "gpu stage {} {} completed rows={} ready_us={}",
            self.id,
            self.kind.as_str(),
            self.rows,
            ready_us,
        );
        self.state = GpuStageState::Completed;
    }

    pub(in crate::scheduler::cuda) fn contains_slot(&self, slot_idx: usize) -> bool {
        self.slots.contains(&slot_idx)
    }
}

pub(in crate::scheduler::cuda) struct PendingDecode {
    pub stage: GpuStageHandle,
    pub decode_indices: Vec<usize>,
    pub slot_indices: Vec<usize>,
    /// True only when `sample_batch_greedy_launch` actually fired the argmax kernel.
    pub greedy_launched: bool,
    /// Model-owned async readback ring slot for the in-flight greedy sample.
    pub async_slot_idx: Option<usize>,
    /// Timestamp taken after the model enqueued the async D2H readback.
    pub readback_started_at: Option<std::time::Instant>,
    /// True when the launch was routed through the Phase 2 speculative decode
    /// verifier path. P2.3 is restricted to a single-token greedy canary; K-token
    /// speculation must use separate pending verifier metadata.
    pub speculative: bool,
    pub decode_spans: Vec<(usize, Span)>,
    pub mixed_prefill: Option<PendingMixedPrefill>,
}

pub(in crate::scheduler::cuda) struct PendingPrefillRow {
    pub slot_idx: usize,
    pub total_tokens: usize,
    pub next_progress: usize,
}

pub(in crate::scheduler::cuda) struct PendingPrefill {
    pub stage: GpuStageHandle,
    pub rows: Vec<PendingPrefillRow>,
    pub uses_paged: bool,
    pub prefill_spans: Vec<(usize, Span)>,
}

pub(in crate::scheduler::cuda) struct PendingMixedPrefill {
    pub stage: GpuStageHandle,
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
    pub step_timing_snapshot_us: f64,
    pub step_timing_cpu_plan_us: f64,
    pub step_timing_gpu_completion_wait_us: f64,
    /// Monotonic scheduler-state epoch used to validate snapshot-derived CPU plans.
    pub scheduler_epoch: u64,
    /// Throttled GPU memory query state and peak high-water mark.
    pub last_mem_query: std::time::Instant,
    pub peak_mem_bytes: u64,
    /// Set when a prefill batch fails with an out-of-memory error.
    /// While this is in the future, `assign_slots` serializes new
    /// prefill admits (one at a time, only when no GPU work is in
    /// flight) so a transient workspace shortage doesn't cascade into
    /// every subsequent request OOMing too.
    pub prefill_oom_cooldown_until: Option<std::time::Instant>,
    pub next_gpu_stage_id: u64,
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
            step_timing_snapshot_us: 0.0,
            step_timing_cpu_plan_us: 0.0,
            step_timing_gpu_completion_wait_us: 0.0,
            scheduler_epoch: 0,
            last_mem_query: std::time::Instant::now(),
            peak_mem_bytes: 0,
            prefill_oom_cooldown_until: None,
            next_gpu_stage_id: 1,
        }
    }

    pub(in crate::scheduler::cuda) fn next_gpu_stage_id(&mut self) -> u64 {
        let id = self.next_gpu_stage_id;
        self.next_gpu_stage_id = self.next_gpu_stage_id.wrapping_add(1).max(1);
        id
    }

    pub(in crate::scheduler::cuda) fn advance_epoch(&mut self) -> u64 {
        self.scheduler_epoch = self.scheduler_epoch.wrapping_add(1).max(1);
        self.scheduler_epoch
    }

    pub(in crate::scheduler::cuda) fn record_pipeline_timing(
        &mut self,
        snapshot_us: u128,
        cpu_plan_us: u128,
        gpu_completion_wait_us: u128,
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

        update_ema(&mut self.step_timing_snapshot_us, snapshot_us);
        update_ema(&mut self.step_timing_cpu_plan_us, cpu_plan_us);
        update_ema(
            &mut self.step_timing_gpu_completion_wait_us,
            gpu_completion_wait_us,
        );
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
    use super::SchedulerRuntimeStats;

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
