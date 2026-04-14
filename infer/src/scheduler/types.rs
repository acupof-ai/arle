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
    /// Chunked prefill chunk size (tokens per prefill step).
    pub prefill_chunk_size: usize,
    /// Maximum requests allowed in the waiting queue.
    /// `submit()` returns `Err(SchedulerFull)` when the queue is at capacity.
    pub max_waiting_requests: usize,
    /// Strategy to use when a running request must be preempted.
    pub preemption_mode: PreemptionMode,
    /// Prefill chunk size cap when decode requests are active.
    /// Smaller values reduce decode latency at the cost of prefill throughput.
    pub decode_active_prefill_cap: usize,
    /// GPU memory (bytes) reserved as headroom when auto-sizing KV cache.
    pub gpu_reserved_bytes: usize,
    /// Minimum sequence length per slot when auto-sizing KV cache.
    pub min_seq_len: usize,
    /// GPU memory (bytes) reserved for KV pool headroom.
    pub kv_pool_headroom_bytes: usize,
    /// Fallback KV pool budget (bytes) when GPU memory query fails.
    pub kv_pool_fallback_bytes: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_slots: 4,
            prefill_chunk_size: 512,
            max_waiting_requests: 256,
            preemption_mode: PreemptionMode::Recompute,
            decode_active_prefill_cap: 512,
            gpu_reserved_bytes: 512 * 1024 * 1024,
            min_seq_len: 256,
            kv_pool_headroom_bytes: 4 * 1024 * 1024 * 1024,
            kv_pool_fallback_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

impl SchedulerConfig {
    /// Runtime-oriented defaults for the CUDA-backed serving scheduler.
    ///
    /// This keeps the existing `Default` implementation stable for the
    /// CPU-only scheduling/accounting layer while making the serving defaults
    /// explicit at the call site.
    pub fn runtime_defaults(max_slots: usize) -> Self {
        Self {
            max_slots,
            prefill_chunk_size: 4096,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_slots == 0 {
            anyhow::bail!("max_slots must be ≥ 1");
        }
        if self.prefill_chunk_size == 0 {
            anyhow::bail!("prefill_chunk_size must be ≥ 1");
        }
        if self.decode_active_prefill_cap == 0 {
            anyhow::bail!("decode_active_prefill_cap must be ≥ 1");
        }
        if self.min_seq_len == 0 {
            anyhow::bail!("min_seq_len must be ≥ 1");
        }
        if self.min_seq_len > 32768 {
            anyhow::bail!("min_seq_len must be ≤ 32768");
        }
        Ok(())
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
