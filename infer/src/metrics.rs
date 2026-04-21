//! Prometheus-compatible metrics for the inference server.
//!
//! Metrics are collected in a single `Metrics` struct that the scheduler
//! updates and the HTTP server reads. The `/metrics` endpoint renders them
//! in the Prometheus text exposition format.
//!
//! # Exposed metrics
//!
//! | Name | Type | Description |
//! |------|------|-------------|
//! | `infer_requests_total` | counter | Total completed requests |
//! | `infer_requests_active` | gauge | Currently-running requests |
//! | `infer_requests_waiting` | gauge | Requests waiting in queue |
//! | `infer_scheduler_running_batch` | gauge | Requests currently in the decode-running batch |
//! | `infer_scheduler_prefill_queue` | gauge | Requests currently queued for prefill continuation |
//! | `infer_tokens_generated_total` | counter | Total output tokens generated |
//! | `infer_tokens_prompt_total` | counter | Total prompt tokens processed |
//! | `infer_queue_wait_seconds` | histogram | Submit-to-admit queueing latency |
//! | `infer_active_ttft_seconds` | histogram | Admit-to-first-token service latency |
//! | `infer_service_seconds` | histogram | First-token-to-finish service latency |
//! | `infer_ttft_seconds` | histogram | Time-to-first-token latency |
//! | `infer_tpot_seconds` | histogram | Time-per-output-token latency |
//! | `infer_e2e_seconds` | histogram | End-to-end request latency |
//! | `infer_kv_gpu_utilization` | gauge | GPU KV cache utilization (0–1) |
//! | `infer_kv_gpu_blocks_free` | gauge | Free GPU KV blocks |
//! | `infer_kv_gpu_blocks_total` | gauge | Total GPU KV blocks |
//! | `infer_prefix_hits_total` | counter | Prefix-cache lookup hits |
//! | `infer_prefix_lookups_total` | counter | Prefix-cache lookups |
//! | `infer_memory_active_bytes` | gauge | Active MLX allocator memory |
//! | `infer_memory_peak_bytes` | gauge | Peak MLX allocator memory |
//! | `infer_memory_cache_bytes` | gauge | Cached MLX allocator memory |

use std::fmt::Write as FmtWrite;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ============================================================================
// Histogram
// ============================================================================

/// Fixed-bucket histogram for latency metrics.
///
/// Bucket boundaries follow a log-linear scale covering [1ms, 60s].
pub struct Histogram {
    /// Upper inclusive bounds for each bucket (seconds).
    buckets: Vec<f64>,
    /// Count of observations falling into each bucket (cumulative, like Prometheus).
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

/// Default latency buckets in seconds, covering 1ms … 60s.
/// Finer granularity in the 10–100ms range where ITL/TPOT typically falls.
pub const LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.075, 0.100,
    0.150, 0.200, 0.300, 0.500, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0,
];

impl Histogram {
    /// Create a new histogram with the given bucket boundaries (in seconds).
    /// Buckets are sorted ascending. Duplicate boundaries are de-duplicated by sort.
    pub fn new(buckets: &[f64]) -> Self {
        let mut sorted = buckets.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let counts = vec![0u64; sorted.len()];
        Self {
            buckets: sorted,
            counts,
            sum: 0.0,
            count: 0,
        }
    }

    /// Record one observation.
    pub fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        for (i, &bound) in self.buckets.iter().enumerate() {
            if value <= bound {
                self.counts[i] += 1;
                break;
            }
        }
    }

    /// Render as Prometheus histogram lines.
    pub fn render(&self, name: &str, labels: &str) -> String {
        let mut out = String::new();
        let mut cumulative = 0u64;
        for (i, &bound) in self.buckets.iter().enumerate() {
            cumulative += self.counts[i];
            writeln!(
                out,
                "{name}_bucket{{{labels}le=\"{bound:.3}\"}} {cumulative}"
            )
            .unwrap();
        }
        writeln!(out, "{name}_bucket{{{labels}le=\"+Inf\"}} {}", self.count).unwrap();
        writeln!(out, "{name}_sum{{{labels}}} {:.6}", self.sum).unwrap();
        writeln!(out, "{name}_count{{{labels}}} {}", self.count).unwrap();
        out
    }

    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// Estimate percentile using linear interpolation on bucket counts.
    /// Returns `None` if no observations.
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let target = (p * self.count as f64).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, &bound) in self.buckets.iter().enumerate() {
            cumulative += self.counts[i];
            if cumulative >= target {
                return Some(bound);
            }
        }
        // All observations are in the +Inf bucket.
        self.buckets.last().copied()
    }
}

// ============================================================================
// HistogramSet — holds TTFT, TPOT, E2E histograms behind a Mutex
// ============================================================================

pub struct HistogramSet {
    pub queue_wait: Histogram,
    pub active_ttft: Histogram,
    pub service: Histogram,
    pub ttft: Histogram,
    pub tpot: Histogram,
    pub e2e: Histogram,
}

impl HistogramSet {
    /// Create a new set of TTFT, TPOT, and E2E histograms using the default latency buckets.
    pub fn new() -> Self {
        Self {
            queue_wait: Histogram::new(LATENCY_BUCKETS),
            active_ttft: Histogram::new(LATENCY_BUCKETS),
            service: Histogram::new(LATENCY_BUCKETS),
            ttft: Histogram::new(LATENCY_BUCKETS),
            tpot: Histogram::new(LATENCY_BUCKETS),
            e2e: Histogram::new(LATENCY_BUCKETS),
        }
    }
}

impl Default for HistogramSet {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ServerMetrics
// ============================================================================

/// Shared server metrics — cheap to clone (Arc internals).
#[derive(Clone)]
pub struct ServerMetrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    // Counters (atomic for lock-free updates from scheduler thread).
    pub requests_total: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    pub tokens_prompt_total: AtomicU64,
    pub requests_failed_total: AtomicU64,
    pub prefix_hits_total: AtomicU64,
    pub prefix_lookups_total: AtomicU64,

    // DFlash speculative decode counters.
    pub dflash_blocks_total: AtomicU64,
    pub dflash_accepted_tokens_total: AtomicU64,
    pub dflash_draft_tokens_total: AtomicU64,

    // Gauges (atomic).
    pub requests_active: AtomicU64,
    pub requests_waiting: AtomicU64,
    pub scheduler_running_batch: AtomicU64,
    pub scheduler_prefill_queue: AtomicU64,
    pub kv_gpu_blocks_free: AtomicU64,
    pub kv_gpu_blocks_total: AtomicU64,
    pub memory_active_bytes: AtomicU64,
    pub memory_peak_bytes: AtomicU64,
    pub memory_cache_bytes: AtomicU64,

    // Histograms (mutex-protected — infrequent writes per request).
    pub histograms: Mutex<HistogramSet>,

    // Model metadata.
    pub model_id: String,
}

impl ServerMetrics {
    pub fn new(model_id: &str) -> Self {
        Self {
            inner: Arc::new(MetricsInner {
                requests_total: AtomicU64::new(0),
                tokens_generated_total: AtomicU64::new(0),
                tokens_prompt_total: AtomicU64::new(0),
                requests_failed_total: AtomicU64::new(0),
                prefix_hits_total: AtomicU64::new(0),
                prefix_lookups_total: AtomicU64::new(0),
                dflash_blocks_total: AtomicU64::new(0),
                dflash_accepted_tokens_total: AtomicU64::new(0),
                dflash_draft_tokens_total: AtomicU64::new(0),
                requests_active: AtomicU64::new(0),
                requests_waiting: AtomicU64::new(0),
                scheduler_running_batch: AtomicU64::new(0),
                scheduler_prefill_queue: AtomicU64::new(0),
                kv_gpu_blocks_free: AtomicU64::new(0),
                kv_gpu_blocks_total: AtomicU64::new(0),
                memory_active_bytes: AtomicU64::new(0),
                memory_peak_bytes: AtomicU64::new(0),
                memory_cache_bytes: AtomicU64::new(0),
                histograms: Mutex::new(HistogramSet::new()),
                model_id: model_id.to_string(),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Update helpers (called by scheduler)
    // -----------------------------------------------------------------------

    /// Record a completed request: update counters and observe latency histograms.
    pub fn record_request_completed(
        &self,
        prompt_tokens: u64,
        generated_tokens: u64,
        ttft_s: f64,
        tpot_s: f64,
        e2e_s: f64,
    ) {
        self.record_request_completed_detailed(
            prompt_tokens,
            generated_tokens,
            0.0,
            ttft_s,
            ttft_s,
            tpot_s,
            e2e_s,
        );
    }

    /// Record a completed request with queueing and service phases broken out.
    pub fn record_request_completed_detailed(
        &self,
        prompt_tokens: u64,
        generated_tokens: u64,
        queue_wait_s: f64,
        active_ttft_s: f64,
        ttft_s: f64,
        tpot_s: f64,
        e2e_s: f64,
    ) {
        self.inner.requests_total.fetch_add(1, Ordering::Relaxed);
        self.inner
            .tokens_prompt_total
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.inner
            .tokens_generated_total
            .fetch_add(generated_tokens, Ordering::Relaxed);

        if let Ok(mut h) = self.inner.histograms.lock() {
            h.queue_wait.observe(queue_wait_s);
            h.active_ttft.observe(active_ttft_s);
            h.ttft.observe(ttft_s);
            if generated_tokens > 1 {
                h.tpot.observe(tpot_s);
            }
            h.service.observe((e2e_s - ttft_s).max(0.0));
            h.e2e.observe(e2e_s);
        }
    }

    /// Increment the failed-request counter.
    pub fn record_request_failed(&self) {
        self.inner
            .requests_failed_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record one prefix-cache lookup and whether it hit a reusable prefix.
    pub fn record_prefix_lookup(&self, hit: bool) {
        self.inner
            .prefix_lookups_total
            .fetch_add(1, Ordering::Relaxed);
        if hit {
            self.inner.prefix_hits_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Set the number of currently-active requests.
    pub fn set_active(&self, n: u64) {
        self.inner.requests_active.store(n, Ordering::Relaxed);
    }

    /// Set the number of requests currently waiting in the queue.
    pub fn set_waiting(&self, n: u64) {
        self.inner.requests_waiting.store(n, Ordering::Relaxed);
    }

    /// Set scheduler-owned queue occupancy counters.
    pub fn set_scheduler_occupancy(&self, running_batch: u64, prefill_queue: u64) {
        self.inner
            .scheduler_running_batch
            .store(running_batch, Ordering::Relaxed);
        self.inner
            .scheduler_prefill_queue
            .store(prefill_queue, Ordering::Relaxed);
    }

    /// Update the GPU KV block gauges.
    pub fn set_kv_gpu_blocks(&self, free: u64, total: u64) {
        self.inner.kv_gpu_blocks_free.store(free, Ordering::Relaxed);
        self.inner
            .kv_gpu_blocks_total
            .store(total, Ordering::Relaxed);
    }

    /// Update MLX allocator memory gauges in bytes.
    pub fn set_memory_bytes(&self, active: u64, peak: u64, cache: u64) {
        self.inner
            .memory_active_bytes
            .store(active, Ordering::Relaxed);
        self.inner.memory_peak_bytes.store(peak, Ordering::Relaxed);
        self.inner
            .memory_cache_bytes
            .store(cache, Ordering::Relaxed);
    }

    // -----------------------------------------------------------------------
    // Read helpers
    // -----------------------------------------------------------------------

    pub fn requests_total(&self) -> u64 {
        self.inner.requests_total.load(Ordering::Relaxed)
    }

    pub fn tokens_generated_total(&self) -> u64 {
        self.inner.tokens_generated_total.load(Ordering::Relaxed)
    }

    pub fn tokens_prompt_total(&self) -> u64 {
        self.inner.tokens_prompt_total.load(Ordering::Relaxed)
    }

    pub fn requests_active(&self) -> u64 {
        self.inner.requests_active.load(Ordering::Relaxed)
    }

    pub fn requests_waiting(&self) -> u64 {
        self.inner.requests_waiting.load(Ordering::Relaxed)
    }

    pub fn scheduler_running_batch(&self) -> u64 {
        self.inner.scheduler_running_batch.load(Ordering::Relaxed)
    }

    pub fn scheduler_prefill_queue(&self) -> u64 {
        self.inner.scheduler_prefill_queue.load(Ordering::Relaxed)
    }

    pub fn kv_gpu_utilization(&self) -> f64 {
        let total = self.inner.kv_gpu_blocks_total.load(Ordering::Relaxed);
        let free = self.inner.kv_gpu_blocks_free.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        (total - free) as f64 / total as f64
    }

    pub fn prefix_hit_rate(&self) -> f64 {
        let lookups = self.inner.prefix_lookups_total.load(Ordering::Relaxed);
        if lookups == 0 {
            return 0.0;
        }
        self.inner.prefix_hits_total.load(Ordering::Relaxed) as f64 / lookups as f64
    }

    /// Record one DFlash speculative block execution.
    pub fn record_dflash_block(&self, accepted_inputs: usize, block_size: usize) {
        self.inner
            .dflash_blocks_total
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .dflash_accepted_tokens_total
            .fetch_add(accepted_inputs as u64, Ordering::Relaxed);
        self.inner
            .dflash_draft_tokens_total
            .fetch_add(block_size as u64, Ordering::Relaxed);
    }

    /// DFlash acceptance rate: fraction of generated tokens that came from draft
    /// predictions (industry-standard speculative decode metric).
    /// Formula: (accepted_inputs - blocks) / accepted_inputs
    ///        = accepted_from_draft / total_generated
    pub fn dflash_acceptance_rate(&self) -> f64 {
        let accepted = self
            .inner
            .dflash_accepted_tokens_total
            .load(Ordering::Relaxed);
        if accepted == 0 {
            return 0.0;
        }
        let blocks = self.inner.dflash_blocks_total.load(Ordering::Relaxed);
        // accepted = sum(matched + 1), blocks = N
        // accepted_from_draft = accepted - blocks = sum(matched)
        // rate = sum(matched) / sum(matched + 1)
        let from_draft = accepted.saturating_sub(blocks);
        from_draft as f64 / accepted as f64
    }

    /// Like [`dflash_acceptance_rate`](Self::dflash_acceptance_rate) but
    /// returns `None` before any speculative block has executed, so HTTP
    /// callers can surface "unknown" (JSON `null`) instead of a misleading
    /// `0.0`. Used by `/v1/models` — the Prometheus gauge stays a flat `f64`.
    pub fn dflash_acceptance_rate_opt(&self) -> Option<f64> {
        let blocks = self.inner.dflash_blocks_total.load(Ordering::Relaxed);
        if blocks == 0 {
            return None;
        }
        Some(self.dflash_acceptance_rate())
    }

    /// DFlash utilization: fraction of total speculative capacity used.
    /// Formula: sum(accepted_inputs) / sum(block_size)
    pub fn dflash_utilization(&self) -> f64 {
        let drafted = self.inner.dflash_draft_tokens_total.load(Ordering::Relaxed);
        if drafted == 0 {
            return 0.0;
        }
        self.inner
            .dflash_accepted_tokens_total
            .load(Ordering::Relaxed) as f64
            / drafted as f64
    }

    // -----------------------------------------------------------------------
    // Prometheus text format rendering
    // -----------------------------------------------------------------------

    /// Render all metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let model = &self.inner.model_id;
        let labels = if model.is_empty() {
            String::new()
        } else {
            format!("model=\"{model}\",")
        };

        let mut out = String::new();

        // Counters
        out.push_str("# HELP infer_requests_total Total completed inference requests.\n");
        out.push_str("# TYPE infer_requests_total counter\n");
        writeln!(
            out,
            "infer_requests_total{{{labels}}} {}",
            self.inner.requests_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_tokens_generated_total Total output tokens generated.\n");
        out.push_str("# TYPE infer_tokens_generated_total counter\n");
        writeln!(
            out,
            "infer_tokens_generated_total{{{labels}}} {}",
            self.inner.tokens_generated_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_tokens_prompt_total Total prompt tokens processed.\n");
        out.push_str("# TYPE infer_tokens_prompt_total counter\n");
        writeln!(
            out,
            "infer_tokens_prompt_total{{{labels}}} {}",
            self.inner.tokens_prompt_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_requests_failed_total Total failed inference requests.\n");
        out.push_str("# TYPE infer_requests_failed_total counter\n");
        writeln!(
            out,
            "infer_requests_failed_total{{{labels}}} {}",
            self.inner.requests_failed_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_prefix_lookups_total Total prefix-cache lookups.\n");
        out.push_str("# TYPE infer_prefix_lookups_total counter\n");
        writeln!(
            out,
            "infer_prefix_lookups_total{{{labels}}} {}",
            self.inner.prefix_lookups_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_prefix_hits_total Total reusable prefix-cache hits.\n");
        out.push_str("# TYPE infer_prefix_hits_total counter\n");
        writeln!(
            out,
            "infer_prefix_hits_total{{{labels}}} {}",
            self.inner.prefix_hits_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_prefix_hit_rate Reusable prefix-cache hit rate [0,1].\n");
        out.push_str("# TYPE infer_prefix_hit_rate gauge\n");
        writeln!(
            out,
            "infer_prefix_hit_rate{{{labels}}} {:.4}",
            self.prefix_hit_rate()
        )
        .unwrap();

        // DFlash speculative decode counters
        out.push_str(
            "# HELP infer_dflash_blocks_total Total DFlash speculative blocks executed.\n",
        );
        out.push_str("# TYPE infer_dflash_blocks_total counter\n");
        writeln!(
            out,
            "infer_dflash_blocks_total{{{labels}}} {}",
            self.inner.dflash_blocks_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_dflash_accepted_tokens_total Total tokens accepted from DFlash speculative blocks.\n");
        out.push_str("# TYPE infer_dflash_accepted_tokens_total counter\n");
        writeln!(
            out,
            "infer_dflash_accepted_tokens_total{{{labels}}} {}",
            self.inner
                .dflash_accepted_tokens_total
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_dflash_acceptance_rate DFlash acceptance rate: fraction of generated tokens from draft [0,1].\n",
        );
        out.push_str("# TYPE infer_dflash_acceptance_rate gauge\n");
        writeln!(
            out,
            "infer_dflash_acceptance_rate{{{labels}}} {:.4}",
            self.dflash_acceptance_rate()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_dflash_utilization DFlash speculative capacity utilization [0,1].\n",
        );
        out.push_str("# TYPE infer_dflash_utilization gauge\n");
        writeln!(
            out,
            "infer_dflash_utilization{{{labels}}} {:.4}",
            self.dflash_utilization()
        )
        .unwrap();

        // Gauges
        out.push_str("# HELP infer_requests_active Currently running requests.\n");
        out.push_str("# TYPE infer_requests_active gauge\n");
        writeln!(
            out,
            "infer_requests_active{{{labels}}} {}",
            self.inner.requests_active.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_requests_waiting Requests waiting in queue.\n");
        out.push_str("# TYPE infer_requests_waiting gauge\n");
        writeln!(
            out,
            "infer_requests_waiting{{{labels}}} {}",
            self.inner.requests_waiting.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_running_batch Requests currently held in the running decode batch.\n",
        );
        out.push_str("# TYPE infer_scheduler_running_batch gauge\n");
        writeln!(
            out,
            "infer_scheduler_running_batch{{{labels}}} {}",
            self.inner.scheduler_running_batch.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_prefill_queue Requests currently queued for prefill continuation.\n",
        );
        out.push_str("# TYPE infer_scheduler_prefill_queue gauge\n");
        writeln!(
            out,
            "infer_scheduler_prefill_queue{{{labels}}} {}",
            self.inner.scheduler_prefill_queue.load(Ordering::Relaxed)
        )
        .unwrap();

        let total = self.inner.kv_gpu_blocks_total.load(Ordering::Relaxed);
        let free = self.inner.kv_gpu_blocks_free.load(Ordering::Relaxed);
        let utilization = if total == 0 {
            0.0
        } else {
            (total - free) as f64 / total as f64
        };

        out.push_str("# HELP infer_kv_gpu_utilization GPU KV cache utilization [0,1].\n");
        out.push_str("# TYPE infer_kv_gpu_utilization gauge\n");
        writeln!(out, "infer_kv_gpu_utilization{{{labels}}} {utilization:.4}").unwrap();

        out.push_str("# HELP infer_kv_gpu_blocks_free Free GPU KV cache blocks.\n");
        out.push_str("# TYPE infer_kv_gpu_blocks_free gauge\n");
        writeln!(out, "infer_kv_gpu_blocks_free{{{labels}}} {free}").unwrap();

        out.push_str("# HELP infer_kv_gpu_blocks_total Total GPU KV cache blocks.\n");
        out.push_str("# TYPE infer_kv_gpu_blocks_total gauge\n");
        writeln!(out, "infer_kv_gpu_blocks_total{{{labels}}} {total}").unwrap();

        out.push_str("# HELP infer_memory_active_bytes Active MLX allocator memory in bytes.\n");
        out.push_str("# TYPE infer_memory_active_bytes gauge\n");
        writeln!(
            out,
            "infer_memory_active_bytes{{{labels}}} {}",
            self.inner.memory_active_bytes.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_memory_peak_bytes Peak MLX allocator memory in bytes.\n");
        out.push_str("# TYPE infer_memory_peak_bytes gauge\n");
        writeln!(
            out,
            "infer_memory_peak_bytes{{{labels}}} {}",
            self.inner.memory_peak_bytes.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_memory_cache_bytes Cached MLX allocator memory in bytes.\n");
        out.push_str("# TYPE infer_memory_cache_bytes gauge\n");
        writeln!(
            out,
            "infer_memory_cache_bytes{{{labels}}} {}",
            self.inner.memory_cache_bytes.load(Ordering::Relaxed)
        )
        .unwrap();

        // Histograms
        if let Ok(h) = self.inner.histograms.lock() {
            out.push_str("# HELP infer_queue_wait_seconds Submit-to-admit queue latency.\n");
            out.push_str("# TYPE infer_queue_wait_seconds histogram\n");
            out.push_str(&h.queue_wait.render("infer_queue_wait_seconds", &labels));

            out.push_str("# HELP infer_active_ttft_seconds Admit-to-first-token latency.\n");
            out.push_str("# TYPE infer_active_ttft_seconds histogram\n");
            out.push_str(&h.active_ttft.render("infer_active_ttft_seconds", &labels));

            out.push_str("# HELP infer_ttft_seconds Time to first token latency.\n");
            out.push_str("# TYPE infer_ttft_seconds histogram\n");
            out.push_str(&h.ttft.render("infer_ttft_seconds", &labels));

            out.push_str("# HELP infer_tpot_seconds Time per output token latency.\n");
            out.push_str("# TYPE infer_tpot_seconds histogram\n");
            out.push_str(&h.tpot.render("infer_tpot_seconds", &labels));

            out.push_str("# HELP infer_service_seconds First-token-to-finish service latency.\n");
            out.push_str("# TYPE infer_service_seconds histogram\n");
            out.push_str(&h.service.render("infer_service_seconds", &labels));

            out.push_str("# HELP infer_e2e_seconds End-to-end request latency.\n");
            out.push_str("# TYPE infer_e2e_seconds histogram\n");
            out.push_str(&h.e2e.render("infer_e2e_seconds", &labels));
        }

        out
    }

    /// Render a simple human-readable summary (for `/v1/stats` or logging).
    pub fn render_summary(&self) -> String {
        let histograms = self.inner.histograms.lock().ok();
        let ttft_p50 = histograms
            .as_ref()
            .and_then(|h| h.ttft.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let queue_p50 = histograms
            .as_ref()
            .and_then(|h| h.queue_wait.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let active_ttft_p50 = histograms
            .as_ref()
            .and_then(|h| h.active_ttft.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let ttft_p99 = histograms
            .as_ref()
            .and_then(|h| h.ttft.percentile(0.99))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let tpot_p50 = histograms
            .as_ref()
            .and_then(|h| h.tpot.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let service_p50 = histograms
            .as_ref()
            .and_then(|h| h.service.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let active_mb =
            self.inner.memory_active_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        let peak_mb =
            self.inner.memory_peak_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        let cache_mb =
            self.inner.memory_cache_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);

        let dflash_blocks = self.inner.dflash_blocks_total.load(Ordering::Relaxed);
        let dflash_suffix = if dflash_blocks > 0 {
            format!(
                " dflash_blocks={} dflash_accept={:.1}% util={:.1}%",
                dflash_blocks,
                self.dflash_acceptance_rate() * 100.0,
                self.dflash_utilization() * 100.0,
            )
        } else {
            String::new()
        };

        format!(
            "requests={} active={} waiting={} running_batch={} prefill_queue={} tokens_out={} kv_util={:.1}% prefix_hit_rate={:.1}% active_mem={:.1}MB peak_mem={:.1}MB cache_mem={:.1}MB queue_p50={} active_ttft_p50={} ttft_p50={} ttft_p99={} service_p50={} tpot_p50={}{}",
            self.requests_total(),
            self.requests_active(),
            self.requests_waiting(),
            self.scheduler_running_batch(),
            self.scheduler_prefill_queue(),
            self.tokens_generated_total(),
            self.kv_gpu_utilization() * 100.0,
            self.prefix_hit_rate() * 100.0,
            active_mb,
            peak_mb,
            cache_mb,
            queue_p50,
            active_ttft_p50,
            ttft_p50,
            ttft_p99,
            service_p50,
            tpot_p50,
            dflash_suffix,
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_observe_and_percentile() {
        let mut h = Histogram::new(LATENCY_BUCKETS);
        // Observe 100 values all = 0.05s (should fall into the 0.05 bucket).
        for _ in 0..100 {
            h.observe(0.05);
        }
        assert_eq!(h.count(), 100);
        assert!((h.sum() - 5.0).abs() < 1e-6);
        // p50 should be in the 0.05 bucket.
        assert_eq!(h.percentile(0.50), Some(0.05));
    }

    #[test]
    fn histogram_render_has_inf_bucket() {
        let mut h = Histogram::new(LATENCY_BUCKETS);
        h.observe(0.1);
        let rendered = h.render("test_latency", "");
        assert!(rendered.contains("le=\"+Inf\""));
        assert!(rendered.contains("test_latency_count"));
    }

    #[test]
    fn server_metrics_prometheus_render() {
        let m = ServerMetrics::new("Qwen3-4B");
        m.record_request_completed(128, 256, 0.05, 0.02, 1.5);
        m.set_active(2);
        m.set_waiting(5);
        m.set_scheduler_occupancy(3, 4);
        m.set_kv_gpu_blocks(100, 200);
        m.record_prefix_lookup(true);
        m.set_memory_bytes(1234, 5678, 42);

        let rendered = m.render_prometheus();
        assert!(rendered.contains("infer_requests_total"));
        assert!(rendered.contains("infer_requests_total{model=\"Qwen3-4B\",} 1"));
        assert!(rendered.contains("infer_requests_active{model=\"Qwen3-4B\",} 2"));
        assert!(rendered.contains("infer_requests_waiting{model=\"Qwen3-4B\",} 5"));
        assert!(rendered.contains("infer_scheduler_running_batch{model=\"Qwen3-4B\",} 3"));
        assert!(rendered.contains("infer_scheduler_prefill_queue{model=\"Qwen3-4B\",} 4"));
        assert!(rendered.contains("infer_prefix_hit_rate{model=\"Qwen3-4B\",} 1.0000"));
        assert!(rendered.contains("infer_memory_active_bytes{model=\"Qwen3-4B\",} 1234"));
        assert!(rendered.contains("infer_queue_wait_seconds_count"));
        assert!(rendered.contains("infer_active_ttft_seconds_count"));
        assert!(rendered.contains("infer_ttft_seconds_count"));
        assert!(rendered.contains("infer_tpot_seconds_count"));
        assert!(rendered.contains("infer_service_seconds_count"));
        assert!(rendered.contains("infer_e2e_seconds_count"));
        assert!(rendered.contains("infer_kv_gpu_blocks_free{model=\"Qwen3-4B\",} 100"));
        assert!(rendered.contains("infer_kv_gpu_blocks_total{model=\"Qwen3-4B\",} 200"));
    }

    #[test]
    fn server_metrics_render_summary() {
        let m = ServerMetrics::new("Qwen3-8B");
        let s = m.render_summary();
        assert!(s.contains("requests=0"));
        assert!(s.contains("active=0"));
        assert!(s.contains("queue_p50="));
        assert!(s.contains("prefix_hit_rate=0.0%"));
    }

    #[test]
    fn server_metrics_clone_shares_state() {
        let m1 = ServerMetrics::new("test");
        let m2 = m1.clone();
        m1.set_active(7);
        assert_eq!(m2.requests_active(), 7);
    }

    #[test]
    fn histogram_empty_percentile() {
        let h = Histogram::new(LATENCY_BUCKETS);
        assert_eq!(h.percentile(0.99), None);
    }
}
