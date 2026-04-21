use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use tokio::sync::mpsc;

use crate::sampler::SamplingParams;
use crate::scheduler::policy::{AdmissionPolicy, QueueBoundAdmission, SchedulerSignals};
use crate::server_engine::CompletionStreamDelta;
use crate::types::SessionId;

/// Preemption strategy when GPU memory is exhausted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum PreemptionMode {
    /// Evict request KV cache and recompute from scratch when resumed.
    /// Cheaper in GPU memory, more expensive when rescheduled.
    #[default]
    Recompute,
    /// Swap KV cache to CPU memory and swap back in when resumed.
    /// Preserves decoded state at the cost of CPU memory.
    Swap,
}

/// Scheduler configuration.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Maximum number of concurrently active request slots.
    pub max_slots: usize,
    /// Maximum number of tokens advanced for one prefilling request in a
    /// single scheduler tick.
    pub chunked_prefill_size: usize,
    /// Maximum total prefill tokens admitted across the whole scheduler tick.
    /// Multiple requests share this budget.
    pub max_prefill_tokens: usize,
    /// Maximum number of prefilling requests to advance in one scheduler step.
    /// `None` means no explicit request-count cap.
    pub prefill_max_requests: Option<usize>,
    /// Allow mixing a prefill batch into a decode batch when the model
    /// supports it. Mirrors SGLang's `--enable-mixed-chunk`.
    pub enable_mixed_chunk: bool,
    /// Maximum requests allowed in the waiting queue.
    /// `submit()` returns `Err(SchedulerFull)` when the queue is at capacity.
    pub max_waiting_requests: usize,
    /// Strategy to use when a running request must be preempted.
    pub preemption_mode: PreemptionMode,
    /// Fraction of total GPU memory for weights + KV cache (SGLang-compatible).
    /// The remaining (1 - fraction) is headroom. Default 0.88.
    pub mem_fraction_static: f64,
    /// Minimum sequence length per slot when auto-sizing KV cache.
    pub min_seq_len: usize,
    /// Fallback KV pool budget (bytes) when GPU memory query fails.
    pub kv_pool_fallback_bytes: usize,
    /// Prefix-cache eviction high-water mark as a fraction of
    /// `max_total_tokens`. Above this the scheduler evicts LRU radix
    /// blocks back to `prefix_cache_low_water`. Default 0.75.
    pub prefix_cache_high_water: f64,
    /// Prefix-cache eviction low-water mark (default 0.50). The high/low
    /// gap prevents evict-then-insert thrash.
    pub prefix_cache_low_water: f64,
    /// Hard cap for radix-retained pages as a fraction of
    /// `max_total_tokens`. Above this fresh publishes are dropped to
    /// keep free-list headroom. Default 0.90.
    pub prefix_cache_retain_hard_cap: f64,
    /// Soft-pin extension, in **radix logical clock ticks**, applied to
    /// session-owned blocks on publish and refreshed on lookup hit. One
    /// tick = one successful `lookup`, `lookup_or_stage`, or `insert`
    /// call (see `prefix_cache.rs::tick`). Default 64. Re-tune against
    /// real session-trace benches by assigning this field explicitly.
    pub prefix_cache_keepalive_ticks: u64,
    /// T1 host-pinned pool eviction high-water mark as a fraction of
    /// the pool's `capacity_bytes`. Above this the coordinator spills
    /// LRU blocks out to T2 disk. Default 0.85. Tuning for T1→T2
    /// spill threshold lives on the config, not on an env var —
    /// same policy as the T0 watermarks in Tier C.
    pub t1_host_pinned_high_water: f64,
    /// T1 host-pinned pool eviction low-water mark. Spill runs down to
    /// this fraction of the pool's capacity before stopping. Default
    /// 0.70. Must be strictly less than `t1_host_pinned_high_water`.
    pub t1_host_pinned_low_water: f64,
    /// Soft-pin extension applied to blocks freshly-demoted into T1.
    /// Prevents a just-demoted host block from being spilled back out by
    /// the same cleanup tick. Default 128 radix logical clock ticks.
    pub t1_host_pinned_keepalive_ticks: u64,
    /// Root directory used by the session snapshot disk store.
    pub disk_store_root: PathBuf,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_slots: 4,
            chunked_prefill_size: 512,
            max_prefill_tokens: 16384,
            prefill_max_requests: None,
            enable_mixed_chunk: false,
            max_waiting_requests: 256,
            preemption_mode: PreemptionMode::Recompute,
            mem_fraction_static: 0.88,
            min_seq_len: 256,
            kv_pool_fallback_bytes: 4 * 1024 * 1024 * 1024,
            // Defaults match the M3b shipped constants in
            // `scheduler/cuda/core.rs`. Tune via explicit field
            // assignment on a `SchedulerConfig`; env overrides are
            // reserved for genuinely debug-only knobs.
            prefix_cache_high_water: 0.75,
            prefix_cache_low_water: 0.50,
            prefix_cache_retain_hard_cap: 0.90,
            prefix_cache_keepalive_ticks: 64,
            // T1 host-pinned watermarks — mirror T0 policy at a
            // slightly higher retention target because host pinned
            // pool churn is cheaper than GPU pool churn.
            t1_host_pinned_high_water: 0.85,
            t1_host_pinned_low_water: 0.70,
            t1_host_pinned_keepalive_ticks: 128,
            disk_store_root: std::env::temp_dir().join("infer-kv"),
        }
    }
}

impl SchedulerConfig {
    /// Runtime-oriented defaults for the CUDA-backed serving scheduler.
    ///
    /// This keeps the existing `Default` implementation stable for the
    /// CPU-only scheduling/accounting layer while making the serving defaults
    /// explicit at the call site. Callers that want to tune the prefix-cache
    /// watermarks or keepalive ticks should assign directly to the relevant
    /// field after calling this — no env-var escape hatches.
    pub fn runtime_defaults(max_slots: usize) -> Self {
        Self {
            max_slots,
            chunked_prefill_size: 4096,
            enable_mixed_chunk: true,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_slots == 0 {
            anyhow::bail!("max_slots must be ≥ 1");
        }
        if self.chunked_prefill_size == 0 {
            anyhow::bail!("chunked_prefill_size must be ≥ 1");
        }
        if self.max_prefill_tokens == 0 {
            anyhow::bail!("max_prefill_tokens must be ≥ 1");
        }
        if matches!(self.prefill_max_requests, Some(0)) {
            anyhow::bail!("prefill_max_requests must be ≥ 1 when provided");
        }
        if self.min_seq_len == 0 {
            anyhow::bail!("min_seq_len must be ≥ 1");
        }
        if self.min_seq_len > 32768 {
            anyhow::bail!("min_seq_len must be ≤ 32768");
        }
        if !(0.0 < self.mem_fraction_static && self.mem_fraction_static <= 1.0) {
            anyhow::bail!("mem_fraction_static must be in (0, 1]");
        }
        if !(0.0 < self.prefix_cache_high_water && self.prefix_cache_high_water < 1.0) {
            anyhow::bail!("prefix_cache_high_water must be in (0, 1)");
        }
        if !(0.0 < self.prefix_cache_low_water
            && self.prefix_cache_low_water < self.prefix_cache_high_water)
        {
            anyhow::bail!("prefix_cache_low_water must be in (0, prefix_cache_high_water)");
        }
        if !(self.prefix_cache_high_water <= self.prefix_cache_retain_hard_cap
            && self.prefix_cache_retain_hard_cap <= 1.0)
        {
            anyhow::bail!(
                "prefix_cache_retain_hard_cap must satisfy prefix_cache_high_water ≤ cap ≤ 1"
            );
        }
        if self.prefix_cache_keepalive_ticks == 0 {
            anyhow::bail!("prefix_cache_keepalive_ticks must be ≥ 1");
        }
        if !(0.0 < self.t1_host_pinned_high_water && self.t1_host_pinned_high_water < 1.0) {
            anyhow::bail!("t1_host_pinned_high_water must be in (0, 1)");
        }
        if !(0.0 < self.t1_host_pinned_low_water
            && self.t1_host_pinned_low_water < self.t1_host_pinned_high_water)
        {
            anyhow::bail!("t1_host_pinned_low_water must be in (0, t1_host_pinned_high_water)");
        }
        if self.t1_host_pinned_keepalive_ticks == 0 {
            anyhow::bail!("t1_host_pinned_keepalive_ticks must be ≥ 1");
        }
        Ok(())
    }
}

#[cfg(any(feature = "cuda", test))]
const REQUEST_INPUT_SLACK_TOKENS: usize = 5;

/// Backend-agnostic request length limits derived from the active scheduler
/// envelope. Mirrors SGLang's `max_req_len` / `max_req_input_len` contract.
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RequestLengthContract {
    max_request_len: usize,
    max_request_input_len: usize,
}

#[cfg(any(feature = "cuda", test))]
impl RequestLengthContract {
    pub(crate) fn derive(
        available_pool_tokens: usize,
        effective_max_seq_len: Option<usize>,
    ) -> Self {
        let context_len = effective_max_seq_len.unwrap_or(available_pool_tokens);
        let max_request_len = context_len
            .saturating_sub(1)
            .min(available_pool_tokens.saturating_sub(1));
        let max_request_input_len = max_request_len.saturating_sub(REQUEST_INPUT_SLACK_TOKENS);
        Self {
            max_request_len,
            max_request_input_len,
        }
    }

    pub(crate) fn max_request_len(self) -> usize {
        self.max_request_len
    }

    pub(crate) fn max_request_input_len(self) -> usize {
        self.max_request_input_len
    }

    pub(crate) fn admits_prompt_len(self, prompt_tokens: usize) -> bool {
        prompt_tokens < self.max_request_input_len
    }

    pub(crate) fn clamp_max_tokens(
        self,
        prompt_tokens: usize,
        requested_max_tokens: usize,
    ) -> usize {
        requested_max_tokens.min(
            self.max_request_len
                .saturating_sub(prompt_tokens)
                .saturating_sub(1),
        )
    }
}

/// Request priority level. Higher-priority requests are scheduled first
/// when multiple requests are waiting.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
pub enum RequestPriority {
    /// Below-normal priority (background batch jobs).
    Low = 0,
    /// Standard priority (default for API requests).
    #[default]
    Normal = 1,
    /// Above-normal priority (interactive / SLA-sensitive requests).
    High = 2,
}

/// Request sent from HTTP handler to scheduler.
pub struct IncomingRequest {
    pub prompt: String,
    /// Optional cached tokenization of `prompt`.
    ///
    /// Requests can remain queued across many ticks; caching tokens here avoids
    /// repeated tokenizer work for the same prompt.
    pub prompt_tokens: Option<Vec<u32>>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop: Option<Vec<String>>,
    /// Scheduling priority. Higher-priority requests are served first.
    pub priority: RequestPriority,
    /// Optional client-supplied session identifier used for sticky routing.
    ///
    /// When present, the scheduler will (once A1's RadixCache integration
    /// lands) prefer to route successive turns of the same session to the
    /// slot or radix subtree that already holds their KV prefix. `None`
    /// preserves the legacy slot-affinity behaviour. See
    /// `docs/projects/agent-first-architecture.md::A2`.
    pub session_id: Option<SessionId>,
    /// Channel to send streaming deltas back to the HTTP handler.
    pub delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
}

/// Error returned when the scheduler's waiting queue is full.
#[derive(Debug)]
pub struct SchedulerFull;

impl std::fmt::Display for SchedulerFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "scheduler waiting queue is full")
    }
}

impl std::error::Error for SchedulerFull {}

/// Handle for submitting requests to the scheduler. Cloneable and Send.
#[derive(Clone)]
pub struct SchedulerHandle {
    tx: mpsc::UnboundedSender<IncomingRequest>,
    model_id: Arc<str>,
    /// Shared count of items currently in the waiting channel.
    waiting_count: Arc<AtomicUsize>,
    /// Maximum allowed waiting requests (0 = unlimited).
    max_waiting: usize,
}

impl SchedulerHandle {
    fn admission_allows(&self, queued_requests: usize) -> bool {
        if self.max_waiting == 0 {
            return true;
        }

        QueueBoundAdmission {
            max_queued_requests: self.max_waiting,
        }
        .allow(SchedulerSignals::queue_state(queued_requests, 0))
    }

    /// Create a handle from raw parts (useful for testing).
    pub fn from_parts(tx: mpsc::UnboundedSender<IncomingRequest>, model_id: &str) -> Self {
        Self {
            tx,
            model_id: Arc::from(model_id),
            waiting_count: Arc::new(AtomicUsize::new(0)),
            max_waiting: 0,
        }
    }

    /// Create a handle with a maximum waiting queue size.
    pub fn with_max_waiting(
        tx: mpsc::UnboundedSender<IncomingRequest>,
        model_id: &str,
        max_waiting: usize,
    ) -> Self {
        Self {
            tx,
            model_id: Arc::from(model_id),
            waiting_count: Arc::new(AtomicUsize::new(0)),
            max_waiting,
        }
    }

    /// Create a handle that shares its waiting count with the scheduler.
    pub fn with_shared_waiting_count(
        tx: mpsc::UnboundedSender<IncomingRequest>,
        model_id: &str,
        max_waiting: usize,
        waiting_count: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            tx,
            model_id: Arc::from(model_id),
            waiting_count,
            max_waiting,
        }
    }

    /// Submit a request to the scheduler.
    ///
    /// Returns `Ok(())` on success.
    /// Returns `Err(SchedulerFull)` if the waiting queue is at capacity.
    /// Returns `Err(SchedulerFull)` if the scheduler has shut down.
    pub fn submit(&self, req: IncomingRequest) -> std::result::Result<(), SchedulerFull> {
        loop {
            let current = self.waiting_count.load(Ordering::Relaxed);
            if !self.admission_allows(current) {
                return Err(SchedulerFull);
            }
            if self
                .waiting_count
                .compare_exchange(current, current + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        self.tx.send(req).map_err(|_| {
            self.waiting_count.fetch_sub(1, Ordering::Relaxed);
            SchedulerFull
        })
    }

    /// Decrement the waiting count (called by the scheduler when it consumes a request).
    pub fn consume_one(&self) {
        self.waiting_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Current number of requests in the waiting channel.
    pub fn waiting_count(&self) -> usize {
        self.waiting_count.load(Ordering::Relaxed)
    }

    /// Returns the model identifier string for this scheduler.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether the queue is currently full.
    pub fn is_full(&self) -> bool {
        !self.admission_allows(self.waiting_count())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact-equality asserts against literal defaults (0.50, 0.75, 0.90, ...)
mod tests {
    use super::*;

    #[test]
    fn runtime_defaults_match_documented_defaults() {
        let cfg = SchedulerConfig::runtime_defaults(8);
        assert_eq!(cfg.max_slots, 8);
        assert_eq!(cfg.chunked_prefill_size, 4096);
        assert_eq!(cfg.max_prefill_tokens, 16384);
        assert_eq!(cfg.prefill_max_requests, None);
        assert!(cfg.enable_mixed_chunk);
        assert_eq!(cfg.prefix_cache_high_water, 0.75);
        assert_eq!(cfg.prefix_cache_low_water, 0.50);
        assert_eq!(cfg.prefix_cache_retain_hard_cap, 0.90);
        assert_eq!(cfg.prefix_cache_keepalive_ticks, 64);
        assert_eq!(cfg.t1_host_pinned_high_water, 0.85);
        assert_eq!(cfg.t1_host_pinned_low_water, 0.70);
        assert_eq!(cfg.t1_host_pinned_keepalive_ticks, 128);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn scheduler_config_rejects_inverted_t1_watermarks() {
        let mut cfg = SchedulerConfig::runtime_defaults(4);
        cfg.t1_host_pinned_low_water = 0.90;
        cfg.t1_host_pinned_high_water = 0.85;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheduler_config_rejects_t1_high_water_out_of_range() {
        let mut cfg = SchedulerConfig::runtime_defaults(4);
        cfg.t1_host_pinned_high_water = 1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheduler_config_rejects_zero_t1_keepalive() {
        let mut cfg = SchedulerConfig::runtime_defaults(4);
        cfg.t1_host_pinned_keepalive_ticks = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheduler_config_accepts_prefill_budget_smaller_than_chunk() {
        let mut cfg = SchedulerConfig::runtime_defaults(4);
        cfg.max_prefill_tokens = cfg.chunked_prefill_size - 1;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn scheduler_config_rejects_zero_prefill_max_requests() {
        let mut cfg = SchedulerConfig::runtime_defaults(4);
        cfg.prefill_max_requests = Some(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheduler_config_rejects_inverted_watermarks() {
        let mut cfg = SchedulerConfig::runtime_defaults(4);
        cfg.prefix_cache_low_water = 0.80;
        cfg.prefix_cache_high_water = 0.75;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheduler_config_rejects_retain_cap_below_high_water() {
        let mut cfg = SchedulerConfig::runtime_defaults(4);
        cfg.prefix_cache_retain_hard_cap = 0.60;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheduler_config_accepts_retain_cap_at_unit_boundary() {
        let mut cfg = SchedulerConfig::runtime_defaults(4);
        cfg.prefix_cache_retain_hard_cap = 1.0;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn scheduler_config_rejects_retain_cap_above_unit_boundary() {
        let mut cfg = SchedulerConfig::runtime_defaults(4);
        cfg.prefix_cache_retain_hard_cap = 1.01;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheduler_config_tunable_via_direct_field_assignment() {
        let mut cfg = SchedulerConfig::runtime_defaults(4);
        cfg.prefix_cache_high_water = 0.80;
        cfg.prefix_cache_low_water = 0.60;
        cfg.prefix_cache_retain_hard_cap = 0.95;
        cfg.prefix_cache_keepalive_ticks = 128;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn request_length_contract_respects_context_and_pool_limits() {
        let contract = RequestLengthContract::derive(60_064, Some(4_608));
        assert_eq!(contract.max_request_len(), 4_607);
        assert_eq!(contract.max_request_input_len(), 4_602);
        assert!(contract.admits_prompt_len(4_601));
        assert!(!contract.admits_prompt_len(4_602));

        let pool_bound = RequestLengthContract::derive(2_048, Some(4_608));
        assert_eq!(pool_bound.max_request_len(), 2_047);
        assert_eq!(pool_bound.max_request_input_len(), 2_042);
    }

    #[test]
    fn request_length_contract_clamps_completion_budget_like_sglang() {
        let contract = RequestLengthContract::derive(60_064, Some(4_608));
        assert_eq!(contract.clamp_max_tokens(4_097, 1_024), 509);
        assert_eq!(contract.clamp_max_tokens(4_097, 128), 128);
    }

    #[test]
    fn request_length_contract_saturates_small_envelopes() {
        let contract = RequestLengthContract::derive(3, Some(2));
        assert_eq!(contract.max_request_len(), 1);
        assert_eq!(contract.max_request_input_len(), 0);
        assert!(!contract.admits_prompt_len(0));
        assert_eq!(contract.clamp_max_tokens(0, 16), 0);
    }
}
