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
//! | `infer_scheduler_scheduled_rows` | gauge | Rows scheduled in the most recent scheduler tick |
//! | `infer_scheduler_scheduled_decode_rows` | gauge | Decode rows scheduled in the most recent scheduler tick |
//! | `infer_scheduler_scheduled_prefill_rows` | gauge | Prefill rows scheduled in the most recent scheduler tick |
//! | `infer_scheduler_decode_tokens` | gauge | Decode tokens advanced in the most recent scheduler tick |
//! | `infer_scheduler_prefill_tokens` | gauge | Prefill tokens advanced in the most recent scheduler tick |
//! | `infer_scheduler_batch_width` | gauge | Total GPU batch width in the most recent scheduler tick |
//! | `infer_scheduler_step_phase_*_microseconds` | gauge | EMA scheduler tick phase duration |
//! | `infer_scheduler_step_cleanup_microseconds` | gauge | EMA scheduler cleanup duration |
//! | `infer_scheduler_loop_total_microseconds` | gauge | EMA full scheduler loop duration |
//! | `infer_preprocess_*` | gauge | HTTP preprocess queue and tokenization timing |
//! | `infer_scheduler_pipeline_*` | gauge/counter | Scheduler pipeline snapshot/plan/GPU-command telemetry |
//! | `infer_runtime_topology_*` | gauge | NUMA/GPU/NIC topology and worker placement |
//! | `infer_runtime_h2d_latency_*` | gauge/counter | Host-to-device copy latency telemetry |
//! | `infer_scheduler_plan_total` | counter | Scheduler ticks by selected plan label |
//! | `infer_prefill_path_mixed_batch_total` | counter | Mixed decode+prefill path outcomes |
//! | `infer_prefill_path_mixed_batch_fallback_total` | counter | Mixed decode+prefill fallback reasons |
//! | `infer_spec_draft_tokens_total` | counter | Draft tokens proposed by Phase 2 speculative decode |
//! | `infer_spec_verified_tokens_total` | counter | Draft tokens checked by the target verifier |
//! | `infer_spec_accepted_tokens_total` | counter | Draft tokens accepted by the verifier |
//! | `infer_spec_acceptance_rate` | gauge | Aggregate accepted / verified token ratio |
//! | `infer_spec_sparse_view_empty_total` | counter | Sparse self-spec decode rows that could not build a sparse KV view |
//! | `infer_spec_step_latency_us` | histogram | Speculative decode step latency |
//! | `infer_metal_decode_batches_total` | counter | Metal decode batches executed on a batched GPU path |
//! | `infer_metal_decode_batched_rows_total` | counter | Metal decode rows executed on a batched GPU path |
//! | `infer_metal_decode_scalar_rows_total` | counter | Metal decode rows executed by the scalar per-request path |
//! | `infer_metal_decode_batch_fallback_rows_total` | counter | Metal decode rows scheduled together but forced to scalar fallback |
//! | `infer_metal_qwen35_packed_decode_batches_total` | counter | Qwen3.5 packed decode batches executed |
//! | `infer_metal_qwen35_packed_decode_rows_total` | counter | Qwen3.5 packed decode rows executed |
//! | `infer_kv_coordinator_queue_capacity` | gauge | Coordinator queue capacity |
//! | `infer_kv_fetch_queue_depth` | gauge | In-flight staged KV fetch tickets |
//! | `infer_kv_fetch_waiters` | gauge | Requests waiting on staged KV fetches |
//! | `infer_kv_store_queue_depth` | gauge | In-flight staged KV spill/store tickets |
//! | `infer_kv_fetch_backpressure` | gauge | Staged KV fetch queue backpressure flag (0/1) |
//! | `infer_kv_store_backpressure` | gauge | Staged KV store queue backpressure flag (0/1) |
//! | `infer_kv_store_submitted_total` | counter | Submitted staged KV spill/store tickets |
//! | `infer_kv_store_completed_total` | counter | Completed staged KV spill/store tickets |
//! | `infer_kv_store_failed_total` | counter | Failed staged KV spill/store tickets |
//! | `infer_kv_store_rejected_total` | counter | Rejected staged KV spill/store tickets |
//! | `infer_tier_fetch_wait_seconds` | gauge | Oldest outstanding staged fetch wait |
//! | `infer_tier_store_wait_seconds` | gauge | Oldest outstanding staged store wait |
//! | `infer_tokens_generated_total` | counter | Total output tokens generated |
//! | `infer_tokens_prompt_total` | counter | Total prompt tokens processed |
//! | `infer_queue_wait_seconds` | histogram | Submit-to-admit queueing latency |
//! | `infer_active_ttft_seconds` | histogram | Admit-to-first-token service latency |
//! | `infer_service_seconds` | histogram | First-token-to-finish service latency |
//! | `infer_ttft_seconds` | histogram | Time-to-first-token latency |
//! | `infer_tpot_seconds` | histogram | Time-per-output-token latency |
//! | `infer_e2e_seconds` | histogram | End-to-end request latency |
//! | `infer_scheduler_step_seconds` | histogram | End-to-end scheduler tick latency |
//! | `infer_kv_gpu_utilization` | gauge | GPU KV cache utilization (0–1) |
//! | `infer_kv_gpu_blocks_free` | gauge | Free GPU KV blocks |
//! | `infer_kv_gpu_blocks_total` | gauge | Total GPU KV blocks |
//! | `infer_prefix_hits_total` | counter | Prefix-cache lookup hits |
//! | `infer_prefix_lookups_total` | counter | Prefix-cache lookups |
//! | `infer_prefix_hit_rate` | gauge | Fraction of prefix lookups with any reused tokens |
//! | `infer_prefix_reused_tokens_total` | counter | Prefix tokens skipped by reuse |
//! | `infer_prefix_lookup_prompt_tokens_total` | counter | Prompt tokens seen by prefix lookup |
//! | `infer_prefix_skip_rate` | gauge | Fraction of prompt tokens skipped by prefix reuse |
//! | `infer_prefix_request_hit_rate` | gauge | Prefix hit rate for the most recent lookup |
//! | `infer_prefix_request_skip_rate` | gauge | Prompt-token skip rate for the most recent lookup |
//! | `infer_prefix_lookup_latency_microseconds` | gauge | Latency of the most recent scheduler prefix lookup |
//! | `infer_session_affinity_hit_total` | counter | Session-tagged requests that reused a prefix |
//! | `infer_session_affinity_miss_total` | counter | Session-tagged requests without prefix reuse |
//! | `infer_session_slot_pressure_evictions_hard_total` | counter | Inactive session slots evicted under hard pressure |
//! | `infer_prefix_aware_admit_deferrals_total` | counter | Cold admission candidates deferred by PrefixAware soft-cap |
//! | `infer_matched_prefix_tokens` | gauge | Matched prefix tokens for the most recent prefix lookup |
//! | `infer_resume_prefill_tokens` | gauge | Effective prefill tokens for the most recent prefix lookup |
//! | `infer_prefix_lookup_reusable_tokens` | gauge | Reusable tokens selected by the most recent scheduler prefix lookup |
//! | `infer_tier_fetch_staged_host_blocks_total` | counter | Request-weighted staged blocks found in T1 |
//! | `infer_tier_fetch_staged_disk_blocks_total` | counter | Request-weighted staged blocks found in T2 |
//! | `infer_tier_fetch_staged_remote_blocks_total` | counter | Request-weighted staged blocks found in T3 |
//! | `infer_tier_fetch_promoted_blocks_total` | counter | Staged blocks promoted back into T0 |
//! | `infer_tier_fetch_fallback_total` | counter | Staged-prefix fallbacks back to cold prefill |
//! | `infer_tier_fetch_recall_rate` | gauge | Promoted staged blocks / staged blocks |
//! | `infer_memory_active_bytes` | gauge | Active MLX allocator memory |
//! | `infer_memory_peak_bytes` | gauge | Peak MLX allocator memory |
//! | `infer_memory_cache_bytes` | gauge | Cached MLX allocator memory |

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[path = "metrics/histogram.rs"]
mod histogram;
#[path = "metrics/render.rs"]
mod render;

pub use histogram::{Histogram, HistogramSet, LATENCY_BUCKETS};
use histogram::{micros_to_secs, secs_to_micros};

use crate::model_arch::ModelArchSummary;
use crate::runtime_topology::{
    AffinityApplyResult, NumaMemoryStats, RuntimeTopology, WorkerPlacement,
};
use crate::server_engine::PrefillPathStats;

// ============================================================================
// ServerMetrics
// ============================================================================

/// Scheduler plan selected for one runtime tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerPlanLabel {
    Idle,
    Decode,
    Prefill,
    Split,
    Mixed,
}

const MIXED_BATCH_FALLBACK_REASONS: [&str; 13] = [
    "unsupported_model",
    "inactive_paged_pool",
    "lora_enabled",
    "unsupported_kv_format",
    "empty_decode_batch",
    "decode_slot_count_mismatch",
    "empty_prefill_batch",
    "prefill_start_position_count_mismatch",
    "empty_prefill_tokens",
    "prefill_slot_in_decode_batch",
    "duplicate_prefill_slot",
    "prefill_seq_len_mismatch",
    "scheduler_pre_dispatch_fallback",
];

/// Shared server metrics — cheap to clone (Arc internals).
#[derive(Clone)]
pub struct ServerMetrics {
    inner: Arc<MetricsInner>,
}

fn ratio_ppm(numer: u64, denom: u64) -> u64 {
    if denom == 0 {
        return 0;
    }
    numer.saturating_mul(1_000_000) / denom
}

#[derive(Clone, Debug, Default)]
struct RequestCacheStats {
    session_id: Option<String>,
    prefix_hit: bool,
    prompt_tokens: u64,
    matched_prefix_tokens: u64,
    resume_prefill_tokens: u64,
}

impl RequestCacheStats {
    fn prefix_hit_rate(&self) -> f64 {
        if self.prefix_hit { 1.0 } else { 0.0 }
    }

    fn prefix_skip_rate(&self) -> f64 {
        if self.prompt_tokens == 0 {
            return 0.0;
        }
        self.matched_prefix_tokens as f64 / self.prompt_tokens as f64
    }
}

#[derive(Clone, Debug, Default)]
struct SessionCacheStats {
    prefix_lookups_total: u64,
    prefix_hits_total: u64,
    prefix_lookup_prompt_tokens_total: u64,
    prefix_reused_tokens_total: u64,
    session_affinity_hit: u64,
    session_affinity_miss: u64,
    matched_prefix_tokens_total: u64,
    resume_prefill_tokens_total: u64,
    last_matched_prefix_tokens: u64,
    last_resume_prefill_tokens: u64,
}

impl SessionCacheStats {
    fn observe(
        &mut self,
        matched_prefix_tokens: u64,
        prompt_tokens: u64,
        resume_prefill_tokens: u64,
    ) {
        self.prefix_lookups_total = self.prefix_lookups_total.saturating_add(1);
        self.prefix_lookup_prompt_tokens_total = self
            .prefix_lookup_prompt_tokens_total
            .saturating_add(prompt_tokens);
        self.prefix_reused_tokens_total = self
            .prefix_reused_tokens_total
            .saturating_add(matched_prefix_tokens);
        self.matched_prefix_tokens_total = self
            .matched_prefix_tokens_total
            .saturating_add(matched_prefix_tokens);
        self.resume_prefill_tokens_total = self
            .resume_prefill_tokens_total
            .saturating_add(resume_prefill_tokens);
        self.last_matched_prefix_tokens = matched_prefix_tokens;
        self.last_resume_prefill_tokens = resume_prefill_tokens;
        if matched_prefix_tokens > 0 {
            self.prefix_hits_total = self.prefix_hits_total.saturating_add(1);
            self.session_affinity_hit = self.session_affinity_hit.saturating_add(1);
        } else {
            self.session_affinity_miss = self.session_affinity_miss.saturating_add(1);
        }
    }

    fn prefix_hit_rate(&self) -> f64 {
        if self.prefix_lookups_total == 0 {
            return 0.0;
        }
        self.prefix_hits_total as f64 / self.prefix_lookups_total as f64
    }

    fn prefix_skip_rate(&self) -> f64 {
        if self.prefix_lookup_prompt_tokens_total == 0 {
            return 0.0;
        }
        self.prefix_reused_tokens_total as f64 / self.prefix_lookup_prompt_tokens_total as f64
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeTopologyMetrics {
    pub numa_nodes: u64,
    pub gpus: u64,
    pub nics: u64,
    pub worker_id: u64,
    pub worker_gpu_ordinal: u64,
    pub worker_numa_node: i64,
    pub worker_cpu_count: u64,
    pub worker_nic_count: u64,
    pub affinity_applied: bool,
    pub affinity_threads: u64,
    pub affinity_failed_threads: u64,
    pub affinity_reason: String,
    pub preprocess_groups: u64,
    pub preprocess_workers: u64,
    pub detokenizer_groups: u64,
    pub detokenizer_workers: u64,
    pub numastat_local_pages: u64,
    pub numastat_remote_pages: u64,
    pub numastat_total_pages: u64,
    pub numastat_nodes: Vec<(i32, u64)>,
    pub h2d_latency_last_us: u64,
    pub h2d_latency_max_us: u64,
    pub h2d_latency_count: u64,
    pub numa_route_local_total: u64,
    pub numa_route_cross_total: u64,
    pub numa_route_unknown_total: u64,
    pub numa_route_cost_last: u64,
    pub numa_migration_total: u64,
    pub numa_rebalance_total: u64,
}

struct MetricsInner {
    // Counters (atomic for lock-free updates from scheduler thread).
    pub requests_total: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    pub tokens_prompt_total: AtomicU64,
    pub requests_failed_total: AtomicU64,
    pub prefix_hits_total: AtomicU64,
    pub prefix_lookups_total: AtomicU64,
    pub prefix_reused_tokens_total: AtomicU64,
    pub prefix_lookup_prompt_tokens_total: AtomicU64,
    pub session_affinity_hit_total: AtomicU64,
    pub session_affinity_miss_total: AtomicU64,
    pub session_slot_pressure_evictions_hard_total: AtomicU64,
    pub prefix_aware_admit_deferrals_total: AtomicU64,
    pub tier_fetch_staged_host_blocks_total: AtomicU64,
    pub tier_fetch_staged_disk_blocks_total: AtomicU64,
    pub tier_fetch_staged_remote_blocks_total: AtomicU64,
    pub tier_fetch_promoted_blocks_total: AtomicU64,
    pub tier_fetch_fallback_total: AtomicU64,
    pub spec_draft_tokens_total: AtomicU64,
    pub spec_verified_tokens_total: AtomicU64,
    pub spec_accepted_tokens_total: AtomicU64,
    pub spec_sparse_view_empty_total: AtomicU64,

    // DFlash speculative decode counters.
    pub dflash_blocks_total: AtomicU64,
    pub dflash_accepted_tokens_total: AtomicU64,
    pub dflash_draft_tokens_total: AtomicU64,
    pub metal_decode_batches_total: AtomicU64,
    pub metal_decode_batched_rows_total: AtomicU64,
    pub metal_decode_scalar_rows_total: AtomicU64,
    pub metal_decode_batch_fallback_rows_total: AtomicU64,
    pub metal_qwen35_packed_decode_batches_total: AtomicU64,
    pub metal_qwen35_packed_decode_rows_total: AtomicU64,

    // Gauges (atomic).
    pub requests_active: AtomicU64,
    pub requests_waiting: AtomicU64,
    pub scheduler_running_batch: AtomicU64,
    pub scheduler_prefill_queue: AtomicU64,
    pub scheduler_scheduled_rows: AtomicU64,
    pub scheduler_scheduled_decode_rows: AtomicU64,
    pub scheduler_scheduled_prefill_rows: AtomicU64,
    pub scheduler_decode_tokens: AtomicU64,
    pub scheduler_prefill_tokens: AtomicU64,
    pub scheduler_batch_width: AtomicU64,
    pub scheduler_step_last_us: AtomicU64,
    pub scheduler_step_admission_us: AtomicU64,
    pub scheduler_step_prefill_us: AtomicU64,
    pub scheduler_step_decode_us: AtomicU64,
    pub scheduler_step_emit_us: AtomicU64,
    pub scheduler_step_total_us: AtomicU64,
    pub scheduler_step_cleanup_us: AtomicU64,
    pub scheduler_loop_total_us: AtomicU64,
    pub scheduler_step_phase_samples: AtomicU64,
    pub preprocess_queue_depth: AtomicU64,
    pub preprocess_wait_us: AtomicU64,
    pub preprocess_tokenize_us: AtomicU64,
    pub scheduler_pipeline_snapshot_us: AtomicU64,
    pub scheduler_pipeline_cpu_plan_us: AtomicU64,
    pub scheduler_pipeline_gpu_completion_wait_us: AtomicU64,
    pub scheduler_pipeline_gpu_command_queue_depth: AtomicU64,
    pub scheduler_pipeline_cpu_plan_accept_total: AtomicU64,
    pub scheduler_pipeline_cpu_plan_stale_total: AtomicU64,
    pub scheduler_plan_idle_total: AtomicU64,
    pub scheduler_plan_decode_total: AtomicU64,
    pub scheduler_plan_prefill_total: AtomicU64,
    pub scheduler_plan_split_total: AtomicU64,
    pub scheduler_plan_mixed_total: AtomicU64,
    pub prefill_path_mixed_ok_true_total: AtomicU64,
    pub prefill_path_mixed_ok_false_total: AtomicU64,
    pub prefill_path_mixed_unsupported_model_total: AtomicU64,
    pub prefill_path_mixed_inactive_paged_pool_total: AtomicU64,
    pub prefill_path_mixed_lora_enabled_total: AtomicU64,
    pub prefill_path_mixed_unsupported_kv_format_total: AtomicU64,
    pub prefill_path_mixed_empty_decode_batch_total: AtomicU64,
    pub prefill_path_mixed_decode_slot_count_mismatch_total: AtomicU64,
    pub prefill_path_mixed_empty_prefill_batch_total: AtomicU64,
    pub prefill_path_mixed_prefill_start_position_count_mismatch_total: AtomicU64,
    pub prefill_path_mixed_empty_prefill_tokens_total: AtomicU64,
    pub prefill_path_mixed_prefill_slot_in_decode_batch_total: AtomicU64,
    pub prefill_path_mixed_duplicate_prefill_slot_total: AtomicU64,
    pub prefill_path_mixed_prefill_seq_len_mismatch_total: AtomicU64,
    pub prefill_path_mixed_scheduler_pre_dispatch_fallback_total: AtomicU64,
    pub spec_acceptance_rate_ppm: AtomicU64,
    pub kv_coordinator_queue_capacity: AtomicU64,
    pub kv_fetch_queue_depth: AtomicU64,
    pub kv_fetch_waiters: AtomicU64,
    pub kv_store_queue_depth: AtomicU64,
    pub kv_fetch_backpressure: AtomicU64,
    pub kv_store_backpressure: AtomicU64,
    pub kv_store_submitted_total: AtomicU64,
    pub kv_store_completed_total: AtomicU64,
    pub kv_store_failed_total: AtomicU64,
    pub kv_store_rejected_total: AtomicU64,
    pub tier_fetch_wait_us: AtomicU64,
    pub tier_store_wait_us: AtomicU64,
    pub kv_gpu_blocks_free: AtomicU64,
    pub kv_gpu_blocks_total: AtomicU64,
    pub matched_prefix_tokens: AtomicU64,
    pub resume_prefill_tokens: AtomicU64,
    pub prefix_request_hit_ppm: AtomicU64,
    pub prefix_request_skip_ppm: AtomicU64,
    pub prefix_lookup_latency_us: AtomicU64,
    pub prefix_lookup_reusable_tokens: AtomicU64,
    pub prefix_lookup_ready_on_gpu: AtomicU64,
    pub prefix_lookup_direct_gpu_attach: AtomicU64,
    pub prefix_lookup_staged: AtomicU64,
    pub prefix_lookup_prefetch: AtomicU64,
    pub prefix_lookup_recompute: AtomicU64,
    pub memory_active_bytes: AtomicU64,
    pub memory_peak_bytes: AtomicU64,
    pub memory_cache_bytes: AtomicU64,

    // Histograms (mutex-protected — infrequent writes per request).
    pub histograms: Mutex<HistogramSet>,
    pub latest_request_cache: Mutex<RequestCacheStats>,
    pub session_cache: Mutex<HashMap<String, SessionCacheStats>>,
    pub runtime_topology: Mutex<RuntimeTopologyMetrics>,

    // Model metadata.
    pub model_id: String,
    pub model_arch: Mutex<Option<ModelArchSummary>>,
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
                prefix_reused_tokens_total: AtomicU64::new(0),
                prefix_lookup_prompt_tokens_total: AtomicU64::new(0),
                session_affinity_hit_total: AtomicU64::new(0),
                session_affinity_miss_total: AtomicU64::new(0),
                session_slot_pressure_evictions_hard_total: AtomicU64::new(0),
                prefix_aware_admit_deferrals_total: AtomicU64::new(0),
                tier_fetch_staged_host_blocks_total: AtomicU64::new(0),
                tier_fetch_staged_disk_blocks_total: AtomicU64::new(0),
                tier_fetch_staged_remote_blocks_total: AtomicU64::new(0),
                tier_fetch_promoted_blocks_total: AtomicU64::new(0),
                tier_fetch_fallback_total: AtomicU64::new(0),
                spec_draft_tokens_total: AtomicU64::new(0),
                spec_verified_tokens_total: AtomicU64::new(0),
                spec_accepted_tokens_total: AtomicU64::new(0),
                spec_sparse_view_empty_total: AtomicU64::new(0),
                dflash_blocks_total: AtomicU64::new(0),
                dflash_accepted_tokens_total: AtomicU64::new(0),
                dflash_draft_tokens_total: AtomicU64::new(0),
                metal_decode_batches_total: AtomicU64::new(0),
                metal_decode_batched_rows_total: AtomicU64::new(0),
                metal_decode_scalar_rows_total: AtomicU64::new(0),
                metal_decode_batch_fallback_rows_total: AtomicU64::new(0),
                metal_qwen35_packed_decode_batches_total: AtomicU64::new(0),
                metal_qwen35_packed_decode_rows_total: AtomicU64::new(0),
                requests_active: AtomicU64::new(0),
                requests_waiting: AtomicU64::new(0),
                scheduler_running_batch: AtomicU64::new(0),
                scheduler_prefill_queue: AtomicU64::new(0),
                scheduler_scheduled_rows: AtomicU64::new(0),
                scheduler_scheduled_decode_rows: AtomicU64::new(0),
                scheduler_scheduled_prefill_rows: AtomicU64::new(0),
                scheduler_decode_tokens: AtomicU64::new(0),
                scheduler_prefill_tokens: AtomicU64::new(0),
                scheduler_batch_width: AtomicU64::new(0),
                scheduler_step_last_us: AtomicU64::new(0),
                scheduler_step_admission_us: AtomicU64::new(0),
                scheduler_step_prefill_us: AtomicU64::new(0),
                scheduler_step_decode_us: AtomicU64::new(0),
                scheduler_step_emit_us: AtomicU64::new(0),
                scheduler_step_total_us: AtomicU64::new(0),
                scheduler_step_cleanup_us: AtomicU64::new(0),
                scheduler_loop_total_us: AtomicU64::new(0),
                scheduler_step_phase_samples: AtomicU64::new(0),
                preprocess_queue_depth: AtomicU64::new(0),
                preprocess_wait_us: AtomicU64::new(0),
                preprocess_tokenize_us: AtomicU64::new(0),
                scheduler_pipeline_snapshot_us: AtomicU64::new(0),
                scheduler_pipeline_cpu_plan_us: AtomicU64::new(0),
                scheduler_pipeline_gpu_completion_wait_us: AtomicU64::new(0),
                scheduler_pipeline_gpu_command_queue_depth: AtomicU64::new(0),
                scheduler_pipeline_cpu_plan_accept_total: AtomicU64::new(0),
                scheduler_pipeline_cpu_plan_stale_total: AtomicU64::new(0),
                scheduler_plan_idle_total: AtomicU64::new(0),
                scheduler_plan_decode_total: AtomicU64::new(0),
                scheduler_plan_prefill_total: AtomicU64::new(0),
                scheduler_plan_split_total: AtomicU64::new(0),
                scheduler_plan_mixed_total: AtomicU64::new(0),
                prefill_path_mixed_ok_true_total: AtomicU64::new(0),
                prefill_path_mixed_ok_false_total: AtomicU64::new(0),
                prefill_path_mixed_unsupported_model_total: AtomicU64::new(0),
                prefill_path_mixed_inactive_paged_pool_total: AtomicU64::new(0),
                prefill_path_mixed_lora_enabled_total: AtomicU64::new(0),
                prefill_path_mixed_unsupported_kv_format_total: AtomicU64::new(0),
                prefill_path_mixed_empty_decode_batch_total: AtomicU64::new(0),
                prefill_path_mixed_decode_slot_count_mismatch_total: AtomicU64::new(0),
                prefill_path_mixed_empty_prefill_batch_total: AtomicU64::new(0),
                prefill_path_mixed_prefill_start_position_count_mismatch_total: AtomicU64::new(0),
                prefill_path_mixed_empty_prefill_tokens_total: AtomicU64::new(0),
                prefill_path_mixed_prefill_slot_in_decode_batch_total: AtomicU64::new(0),
                prefill_path_mixed_duplicate_prefill_slot_total: AtomicU64::new(0),
                prefill_path_mixed_prefill_seq_len_mismatch_total: AtomicU64::new(0),
                prefill_path_mixed_scheduler_pre_dispatch_fallback_total: AtomicU64::new(0),
                spec_acceptance_rate_ppm: AtomicU64::new(0),
                kv_coordinator_queue_capacity: AtomicU64::new(0),
                kv_fetch_queue_depth: AtomicU64::new(0),
                kv_fetch_waiters: AtomicU64::new(0),
                kv_store_queue_depth: AtomicU64::new(0),
                kv_fetch_backpressure: AtomicU64::new(0),
                kv_store_backpressure: AtomicU64::new(0),
                kv_store_submitted_total: AtomicU64::new(0),
                kv_store_completed_total: AtomicU64::new(0),
                kv_store_failed_total: AtomicU64::new(0),
                kv_store_rejected_total: AtomicU64::new(0),
                tier_fetch_wait_us: AtomicU64::new(0),
                tier_store_wait_us: AtomicU64::new(0),
                kv_gpu_blocks_free: AtomicU64::new(0),
                kv_gpu_blocks_total: AtomicU64::new(0),
                matched_prefix_tokens: AtomicU64::new(0),
                resume_prefill_tokens: AtomicU64::new(0),
                prefix_request_hit_ppm: AtomicU64::new(0),
                prefix_request_skip_ppm: AtomicU64::new(0),
                prefix_lookup_latency_us: AtomicU64::new(0),
                prefix_lookup_reusable_tokens: AtomicU64::new(0),
                prefix_lookup_ready_on_gpu: AtomicU64::new(0),
                prefix_lookup_direct_gpu_attach: AtomicU64::new(0),
                prefix_lookup_staged: AtomicU64::new(0),
                prefix_lookup_prefetch: AtomicU64::new(0),
                prefix_lookup_recompute: AtomicU64::new(0),
                memory_active_bytes: AtomicU64::new(0),
                memory_peak_bytes: AtomicU64::new(0),
                memory_cache_bytes: AtomicU64::new(0),
                histograms: Mutex::new(HistogramSet::new()),
                latest_request_cache: Mutex::new(RequestCacheStats::default()),
                session_cache: Mutex::new(HashMap::new()),
                runtime_topology: Mutex::new(RuntimeTopologyMetrics::default()),
                model_id: model_id.to_string(),
                model_arch: Mutex::new(None),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Update helpers (called by scheduler)
    // -----------------------------------------------------------------------

    /// Publish backend-neutral model architecture metadata after load.
    pub fn set_model_arch(&self, summary: ModelArchSummary) {
        if let Ok(mut model_arch) = self.inner.model_arch.lock() {
            *model_arch = Some(summary);
        }
    }

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

    /// Record one prefix-cache lookup plus how many prompt tokens it skipped.
    pub fn record_prefix_lookup(&self, reused_tokens: usize, prompt_tokens: usize) {
        self.record_request_cache(
            None,
            reused_tokens,
            prompt_tokens,
            prompt_tokens.saturating_sub(reused_tokens),
        );
    }

    /// Record request-level prefix reuse and effective prefill accounting.
    ///
    /// `session_affinity_*` here is observational only: a session-tagged
    /// request counts as a hit when the existing prefix path reused tokens.
    /// It does not change admission policy.
    pub fn record_request_cache(
        &self,
        session_id: Option<&crate::types::SessionId>,
        matched_prefix_tokens: usize,
        prompt_tokens: usize,
        resume_prefill_tokens: usize,
    ) {
        let matched_prefix_tokens = matched_prefix_tokens.min(prompt_tokens);
        let resume_prefill_tokens = resume_prefill_tokens.min(prompt_tokens);
        self.inner
            .prefix_lookups_total
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .prefix_reused_tokens_total
            .fetch_add(matched_prefix_tokens as u64, Ordering::Relaxed);
        self.inner
            .prefix_lookup_prompt_tokens_total
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        if matched_prefix_tokens > 0 {
            self.inner.prefix_hits_total.fetch_add(1, Ordering::Relaxed);
        }
        self.inner
            .matched_prefix_tokens
            .store(matched_prefix_tokens as u64, Ordering::Relaxed);
        self.inner
            .resume_prefill_tokens
            .store(resume_prefill_tokens as u64, Ordering::Relaxed);
        self.inner.prefix_request_hit_ppm.store(
            if matched_prefix_tokens > 0 {
                1_000_000
            } else {
                0
            },
            Ordering::Relaxed,
        );
        self.inner.prefix_request_skip_ppm.store(
            ratio_ppm(matched_prefix_tokens as u64, prompt_tokens as u64),
            Ordering::Relaxed,
        );

        let session_string = session_id.map(|id| id.as_str().to_string());
        if let Ok(mut latest) = self.inner.latest_request_cache.lock() {
            *latest = RequestCacheStats {
                session_id: session_string.clone(),
                prefix_hit: matched_prefix_tokens > 0,
                prompt_tokens: prompt_tokens as u64,
                matched_prefix_tokens: matched_prefix_tokens as u64,
                resume_prefill_tokens: resume_prefill_tokens as u64,
            };
        }
        if let Some(session_id) = session_string {
            if matched_prefix_tokens > 0 {
                self.inner
                    .session_affinity_hit_total
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                self.inner
                    .session_affinity_miss_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            if let Ok(mut sessions) = self.inner.session_cache.lock() {
                sessions.entry(session_id).or_default().observe(
                    matched_prefix_tokens as u64,
                    prompt_tokens as u64,
                    resume_prefill_tokens as u64,
                );
            }
        }
    }

    /// Record the scheduler admission-side prefix lookup decision without
    /// incrementing lookup counters that are already updated at request prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn record_prefix_lookup_detail(
        &self,
        prompt_tokens: usize,
        matched_prefix_tokens: usize,
        reusable_tokens: usize,
        lookup_latency_us: u64,
        ready_on_gpu: bool,
        direct_gpu_attach: bool,
        staged: bool,
        prefetch: bool,
        recompute: bool,
    ) {
        let matched_prefix_tokens = matched_prefix_tokens.min(prompt_tokens);
        let reusable_tokens = reusable_tokens.min(prompt_tokens);
        self.inner
            .matched_prefix_tokens
            .store(matched_prefix_tokens as u64, Ordering::Relaxed);
        self.inner
            .prefix_lookup_reusable_tokens
            .store(reusable_tokens as u64, Ordering::Relaxed);
        self.inner
            .prefix_lookup_latency_us
            .store(lookup_latency_us, Ordering::Relaxed);
        self.inner.prefix_request_hit_ppm.store(
            if matched_prefix_tokens > 0 {
                1_000_000
            } else {
                0
            },
            Ordering::Relaxed,
        );
        self.inner.prefix_request_skip_ppm.store(
            ratio_ppm(matched_prefix_tokens as u64, prompt_tokens as u64),
            Ordering::Relaxed,
        );
        self.inner
            .prefix_lookup_ready_on_gpu
            .store(ready_on_gpu as u64, Ordering::Relaxed);
        self.inner
            .prefix_lookup_direct_gpu_attach
            .store(direct_gpu_attach as u64, Ordering::Relaxed);
        self.inner
            .prefix_lookup_staged
            .store(staged as u64, Ordering::Relaxed);
        self.inner
            .prefix_lookup_prefetch
            .store(prefetch as u64, Ordering::Relaxed);
        self.inner
            .prefix_lookup_recompute
            .store(recompute as u64, Ordering::Relaxed);
    }

    /// Mark that the most recent staged prefix lookup was queued for prefetch.
    pub fn record_prefix_lookup_prefetch_queued(&self) {
        self.inner
            .prefix_lookup_prefetch
            .store(1, Ordering::Relaxed);
    }

    /// Record request-weighted staged fetch blocks by slower-tier source.
    pub fn record_tier_fetch_plan(
        &self,
        host_blocks: usize,
        disk_blocks: usize,
        remote_blocks: usize,
    ) {
        self.inner
            .tier_fetch_staged_host_blocks_total
            .fetch_add(host_blocks as u64, Ordering::Relaxed);
        self.inner
            .tier_fetch_staged_disk_blocks_total
            .fetch_add(disk_blocks as u64, Ordering::Relaxed);
        self.inner
            .tier_fetch_staged_remote_blocks_total
            .fetch_add(remote_blocks as u64, Ordering::Relaxed);
    }

    /// Record staged blocks successfully promoted back into T0.
    pub fn record_tier_fetch_promoted(&self, promoted_blocks: usize) {
        self.inner
            .tier_fetch_promoted_blocks_total
            .fetch_add(promoted_blocks as u64, Ordering::Relaxed);
    }

    /// Record one staged-prefix fallback back to cold prefill.
    pub fn record_tier_fetch_fallback(&self) {
        self.inner
            .tier_fetch_fallback_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record inactive session slots evicted by hard pressure reclamation.
    pub fn record_session_slot_pressure_evictions_hard(&self, slots: usize) {
        self.inner
            .session_slot_pressure_evictions_hard_total
            .fetch_add(slots as u64, Ordering::Relaxed);
    }

    /// Record a cold request deferred by PrefixAwareAdmission's soft cap.
    pub fn record_prefix_aware_admit_deferral(&self) {
        self.inner
            .prefix_aware_admit_deferrals_total
            .fetch_add(1, Ordering::Relaxed);
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

    /// Update per-tick scheduler gauges.
    pub fn set_scheduler_step(
        &self,
        scheduled_rows: u64,
        scheduled_decode_rows: u64,
        scheduled_prefill_rows: u64,
        decode_tokens: u64,
        prefill_tokens: u64,
        batch_width: u64,
    ) {
        self.inner
            .scheduler_scheduled_rows
            .store(scheduled_rows, Ordering::Relaxed);
        self.inner
            .scheduler_scheduled_decode_rows
            .store(scheduled_decode_rows, Ordering::Relaxed);
        self.inner
            .scheduler_scheduled_prefill_rows
            .store(scheduled_prefill_rows, Ordering::Relaxed);
        self.inner
            .scheduler_decode_tokens
            .store(decode_tokens, Ordering::Relaxed);
        self.inner
            .scheduler_prefill_tokens
            .store(prefill_tokens, Ordering::Relaxed);
        self.inner
            .scheduler_batch_width
            .store(batch_width, Ordering::Relaxed);
    }

    /// Record end-to-end scheduler tick latency.
    pub fn observe_scheduler_step(&self, step_s: f64) {
        self.inner
            .scheduler_step_last_us
            .store(secs_to_micros(step_s), Ordering::Relaxed);
        if let Ok(mut h) = self.inner.histograms.lock() {
            h.scheduler_step.observe(step_s.max(0.0));
        }
    }

    /// Update scheduler tick phase EMAs in microseconds.
    pub fn set_scheduler_step_phase_us(
        &self,
        admission_us: f64,
        prefill_us: f64,
        decode_us: f64,
        emit_us: f64,
        total_us: f64,
    ) {
        self.inner
            .scheduler_step_admission_us
            .store(admission_us.max(0.0).round() as u64, Ordering::Relaxed);
        self.inner
            .scheduler_step_prefill_us
            .store(prefill_us.max(0.0).round() as u64, Ordering::Relaxed);
        self.inner
            .scheduler_step_decode_us
            .store(decode_us.max(0.0).round() as u64, Ordering::Relaxed);
        self.inner
            .scheduler_step_emit_us
            .store(emit_us.max(0.0).round() as u64, Ordering::Relaxed);
        self.inner
            .scheduler_step_total_us
            .store(total_us.max(0.0).round() as u64, Ordering::Relaxed);
        self.inner
            .scheduler_step_phase_samples
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Update scheduler loop cleanup and full-loop EMAs in microseconds.
    pub fn set_scheduler_loop_phase_us(&self, cleanup_us: f64, loop_total_us: f64) {
        self.inner
            .scheduler_step_cleanup_us
            .store(cleanup_us.max(0.0).round() as u64, Ordering::Relaxed);
        self.inner
            .scheduler_loop_total_us
            .store(loop_total_us.max(0.0).round() as u64, Ordering::Relaxed);
    }

    /// Update HTTP preprocess queue and tokenization gauges.
    pub fn set_preprocess_stage(&self, queue_depth: u64, wait_us: u64, tokenize_us: u64) {
        self.inner
            .preprocess_queue_depth
            .store(queue_depth, Ordering::Relaxed);
        self.inner
            .preprocess_wait_us
            .store(wait_us, Ordering::Relaxed);
        self.inner
            .preprocess_tokenize_us
            .store(tokenize_us, Ordering::Relaxed);
    }

    /// Update scheduler pipeline split gauges for the most recent tick.
    pub fn set_scheduler_pipeline_us(
        &self,
        snapshot_us: u64,
        cpu_plan_us: u64,
        gpu_completion_wait_us: u64,
        gpu_command_queue_depth: u64,
    ) {
        self.inner
            .scheduler_pipeline_snapshot_us
            .store(snapshot_us, Ordering::Relaxed);
        self.inner
            .scheduler_pipeline_cpu_plan_us
            .store(cpu_plan_us, Ordering::Relaxed);
        self.inner
            .scheduler_pipeline_gpu_completion_wait_us
            .store(gpu_completion_wait_us, Ordering::Relaxed);
        self.inner
            .scheduler_pipeline_gpu_command_queue_depth
            .store(gpu_command_queue_depth, Ordering::Relaxed);
    }

    pub fn set_runtime_topology(
        &self,
        topology: &RuntimeTopology,
        placement: &WorkerPlacement,
        affinity: &AffinityApplyResult,
    ) {
        let mut metrics = self
            .inner
            .runtime_topology
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        metrics.numa_nodes = topology.numa_nodes.len() as u64;
        metrics.gpus = topology.gpus.len() as u64;
        metrics.nics = topology.nics.len() as u64;
        metrics.worker_id = placement.worker_id as u64;
        metrics.worker_gpu_ordinal = placement.gpu_ordinal as u64;
        metrics.worker_numa_node = placement.numa_node.map_or(-1, i64::from);
        metrics.worker_cpu_count = placement.cpus.len() as u64;
        metrics.worker_nic_count = placement.nics.len() as u64;
        metrics.affinity_applied = affinity.applied;
        metrics.affinity_threads = affinity.applied_threads as u64;
        metrics.affinity_failed_threads = affinity.failed_threads as u64;
        metrics.affinity_reason.clone_from(&affinity.reason);
    }

    pub fn set_preprocess_topology(&self, groups: usize, workers: usize) {
        let mut metrics = self
            .inner
            .runtime_topology
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        metrics.preprocess_groups = groups as u64;
        metrics.preprocess_workers = workers as u64;
    }

    pub fn set_detokenizer_topology(&self, groups: usize, workers: usize) {
        let mut metrics = self
            .inner
            .runtime_topology
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        metrics.detokenizer_groups = groups as u64;
        metrics.detokenizer_workers = workers as u64;
    }

    pub fn set_runtime_numastat(&self, stats: &NumaMemoryStats, local_node: Option<i32>) {
        self.set_runtime_numastat_for_nodes(stats, &[local_node]);
    }

    pub fn set_runtime_numastat_for_nodes(
        &self,
        stats: &NumaMemoryStats,
        local_nodes: &[Option<i32>],
    ) {
        let local_nodes = local_nodes
            .iter()
            .flatten()
            .copied()
            .collect::<HashSet<_>>();
        let local_pages = stats
            .per_node_pages
            .iter()
            .filter(|(node, _)| local_nodes.contains(node))
            .map(|(_, pages)| *pages)
            .sum();
        let mut metrics = self
            .inner
            .runtime_topology
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        metrics.numastat_local_pages = local_pages;
        metrics.numastat_total_pages = stats.total_pages;
        metrics.numastat_remote_pages = stats.total_pages.saturating_sub(local_pages);
        metrics.numastat_nodes.clone_from(&stats.per_node_pages);
    }

    pub fn observe_h2d_latency_us(&self, latency_us: u64) {
        let mut metrics = self
            .inner
            .runtime_topology
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        metrics.h2d_latency_last_us = latency_us;
        metrics.h2d_latency_max_us = metrics.h2d_latency_max_us.max(latency_us);
        metrics.h2d_latency_count = metrics.h2d_latency_count.saturating_add(1);
    }

    pub fn record_numa_route(&self, route_cost: u32, local: Option<bool>) {
        let mut metrics = self
            .inner
            .runtime_topology
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match local {
            Some(true) => {
                metrics.numa_route_local_total = metrics.numa_route_local_total.saturating_add(1);
            }
            Some(false) => {
                metrics.numa_route_cross_total = metrics.numa_route_cross_total.saturating_add(1);
            }
            None => {
                metrics.numa_route_unknown_total =
                    metrics.numa_route_unknown_total.saturating_add(1);
            }
        }
        metrics.numa_route_cost_last = u64::from(route_cost);
    }

    pub fn record_numa_migration(&self) {
        let mut metrics = self
            .inner
            .runtime_topology
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        metrics.numa_migration_total = metrics.numa_migration_total.saturating_add(1);
    }

    pub fn record_numa_rebalance(&self) {
        let mut metrics = self
            .inner
            .runtime_topology
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        metrics.numa_rebalance_total = metrics.numa_rebalance_total.saturating_add(1);
    }

    pub fn record_scheduler_cpu_plan_accept(&self) {
        self.inner
            .scheduler_pipeline_cpu_plan_accept_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_scheduler_cpu_plan_stale(&self) {
        self.inner
            .scheduler_pipeline_cpu_plan_stale_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record the scheduler plan selected for one runtime tick.
    pub fn record_scheduler_plan(&self, label: SchedulerPlanLabel) {
        let counter = match label {
            SchedulerPlanLabel::Idle => &self.inner.scheduler_plan_idle_total,
            SchedulerPlanLabel::Decode => &self.inner.scheduler_plan_decode_total,
            SchedulerPlanLabel::Prefill => &self.inner.scheduler_plan_prefill_total,
            SchedulerPlanLabel::Split => &self.inner.scheduler_plan_split_total,
            SchedulerPlanLabel::Mixed => &self.inner.scheduler_plan_mixed_total,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn mixed_batch_fallback_counter(&self, reason: &str) -> &AtomicU64 {
        match reason {
            "inactive_paged_pool" => &self.inner.prefill_path_mixed_inactive_paged_pool_total,
            "lora_enabled" => &self.inner.prefill_path_mixed_lora_enabled_total,
            "unsupported_kv_format" => &self.inner.prefill_path_mixed_unsupported_kv_format_total,
            "empty_decode_batch" => &self.inner.prefill_path_mixed_empty_decode_batch_total,
            "decode_slot_count_mismatch" => {
                &self
                    .inner
                    .prefill_path_mixed_decode_slot_count_mismatch_total
            }
            "empty_prefill_batch" => &self.inner.prefill_path_mixed_empty_prefill_batch_total,
            "prefill_start_position_count_mismatch" => {
                &self
                    .inner
                    .prefill_path_mixed_prefill_start_position_count_mismatch_total
            }
            "empty_prefill_tokens" => &self.inner.prefill_path_mixed_empty_prefill_tokens_total,
            "prefill_slot_in_decode_batch" => {
                &self
                    .inner
                    .prefill_path_mixed_prefill_slot_in_decode_batch_total
            }
            "duplicate_prefill_slot" => &self.inner.prefill_path_mixed_duplicate_prefill_slot_total,
            "prefill_seq_len_mismatch" => {
                &self.inner.prefill_path_mixed_prefill_seq_len_mismatch_total
            }
            "scheduler_pre_dispatch_fallback" => {
                &self
                    .inner
                    .prefill_path_mixed_scheduler_pre_dispatch_fallback_total
            }
            _ => &self.inner.prefill_path_mixed_unsupported_model_total,
        }
    }

    /// Record one mixed decode+prefill path execution.
    pub fn record_prefill_path_mixed_ok_true(&self) {
        self.inner
            .prefill_path_mixed_ok_true_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record one mixed decode+prefill fallback and its model-reported reason.
    pub fn record_prefill_path_mixed_ok_false(&self, reason: &str) {
        self.inner
            .prefill_path_mixed_ok_false_total
            .fetch_add(1, Ordering::Relaxed);
        self.mixed_batch_fallback_counter(reason)
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Update the staged KV coordinator queue gauges and cumulative store counters.
    pub fn set_kv_coordinator(
        &self,
        queue_capacity: u64,
        fetch_queue_depth: u64,
        fetch_waiters: u64,
        store_queue_depth: u64,
        fetch_backpressure: bool,
        store_backpressure: bool,
        store_submitted_total: u64,
        store_completed_total: u64,
        store_failed_total: u64,
        store_rejected_total: u64,
    ) {
        self.inner
            .kv_coordinator_queue_capacity
            .store(queue_capacity, Ordering::Relaxed);
        self.inner
            .kv_fetch_queue_depth
            .store(fetch_queue_depth, Ordering::Relaxed);
        self.inner
            .kv_fetch_waiters
            .store(fetch_waiters, Ordering::Relaxed);
        self.inner
            .kv_store_queue_depth
            .store(store_queue_depth, Ordering::Relaxed);
        self.inner
            .kv_fetch_backpressure
            .store(u64::from(fetch_backpressure), Ordering::Relaxed);
        self.inner
            .kv_store_backpressure
            .store(u64::from(store_backpressure), Ordering::Relaxed);
        self.inner
            .kv_store_submitted_total
            .store(store_submitted_total, Ordering::Relaxed);
        self.inner
            .kv_store_completed_total
            .store(store_completed_total, Ordering::Relaxed);
        self.inner
            .kv_store_failed_total
            .store(store_failed_total, Ordering::Relaxed);
        self.inner
            .kv_store_rejected_total
            .store(store_rejected_total, Ordering::Relaxed);
    }

    /// Update oldest outstanding tier wait gauges.
    pub fn set_tier_wait_seconds(&self, fetch_wait_s: f64, store_wait_s: f64) {
        self.inner
            .tier_fetch_wait_us
            .store(secs_to_micros(fetch_wait_s), Ordering::Relaxed);
        self.inner
            .tier_store_wait_us
            .store(secs_to_micros(store_wait_s), Ordering::Relaxed);
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

    pub fn scheduler_scheduled_rows(&self) -> u64 {
        self.inner.scheduler_scheduled_rows.load(Ordering::Relaxed)
    }

    pub fn scheduler_scheduled_decode_rows(&self) -> u64 {
        self.inner
            .scheduler_scheduled_decode_rows
            .load(Ordering::Relaxed)
    }

    pub fn scheduler_scheduled_prefill_rows(&self) -> u64 {
        self.inner
            .scheduler_scheduled_prefill_rows
            .load(Ordering::Relaxed)
    }

    pub fn scheduler_decode_tokens(&self) -> u64 {
        self.inner.scheduler_decode_tokens.load(Ordering::Relaxed)
    }

    pub fn scheduler_prefill_tokens(&self) -> u64 {
        self.inner.scheduler_prefill_tokens.load(Ordering::Relaxed)
    }

    pub fn scheduler_batch_width(&self) -> u64 {
        self.inner.scheduler_batch_width.load(Ordering::Relaxed)
    }

    pub fn scheduler_step_last_seconds(&self) -> f64 {
        micros_to_secs(self.inner.scheduler_step_last_us.load(Ordering::Relaxed))
    }

    pub fn scheduler_step_phase_us(&self) -> Option<(u64, u64, u64, u64, u64)> {
        if self
            .inner
            .scheduler_step_phase_samples
            .load(Ordering::Relaxed)
            == 0
        {
            return None;
        }
        Some((
            self.inner
                .scheduler_step_admission_us
                .load(Ordering::Relaxed),
            self.inner.scheduler_step_prefill_us.load(Ordering::Relaxed),
            self.inner.scheduler_step_decode_us.load(Ordering::Relaxed),
            self.inner.scheduler_step_emit_us.load(Ordering::Relaxed),
            self.inner.scheduler_step_total_us.load(Ordering::Relaxed),
        ))
    }

    pub fn scheduler_loop_phase_us(&self) -> Option<(u64, u64)> {
        if self
            .inner
            .scheduler_step_phase_samples
            .load(Ordering::Relaxed)
            == 0
        {
            return None;
        }
        Some((
            self.inner.scheduler_step_cleanup_us.load(Ordering::Relaxed),
            self.inner.scheduler_loop_total_us.load(Ordering::Relaxed),
        ))
    }

    pub fn preprocess_stage_us(&self) -> (u64, u64, u64) {
        (
            self.inner.preprocess_queue_depth.load(Ordering::Relaxed),
            self.inner.preprocess_wait_us.load(Ordering::Relaxed),
            self.inner.preprocess_tokenize_us.load(Ordering::Relaxed),
        )
    }

    pub fn scheduler_pipeline_us(&self) -> (u64, u64, u64, u64) {
        (
            self.inner
                .scheduler_pipeline_snapshot_us
                .load(Ordering::Relaxed),
            self.inner
                .scheduler_pipeline_cpu_plan_us
                .load(Ordering::Relaxed),
            self.inner
                .scheduler_pipeline_gpu_completion_wait_us
                .load(Ordering::Relaxed),
            self.inner
                .scheduler_pipeline_gpu_command_queue_depth
                .load(Ordering::Relaxed),
        )
    }

    pub fn scheduler_pipeline_plan_totals(&self) -> (u64, u64) {
        (
            self.inner
                .scheduler_pipeline_cpu_plan_accept_total
                .load(Ordering::Relaxed),
            self.inner
                .scheduler_pipeline_cpu_plan_stale_total
                .load(Ordering::Relaxed),
        )
    }

    pub fn runtime_topology_snapshot(&self) -> RuntimeTopologyMetrics {
        self.inner
            .runtime_topology
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn scheduler_plan_totals(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.inner.scheduler_plan_idle_total.load(Ordering::Relaxed),
            self.inner
                .scheduler_plan_decode_total
                .load(Ordering::Relaxed),
            self.inner
                .scheduler_plan_prefill_total
                .load(Ordering::Relaxed),
            self.inner
                .scheduler_plan_split_total
                .load(Ordering::Relaxed),
            self.inner
                .scheduler_plan_mixed_total
                .load(Ordering::Relaxed),
        )
    }

    pub fn prefill_path_stats(&self) -> PrefillPathStats {
        PrefillPathStats {
            ok_true_count: self
                .inner
                .prefill_path_mixed_ok_true_total
                .load(Ordering::Relaxed),
            ok_false_count: self
                .inner
                .prefill_path_mixed_ok_false_total
                .load(Ordering::Relaxed),
            ok_false_reasons: MIXED_BATCH_FALLBACK_REASONS
                .iter()
                .map(|reason| {
                    (
                        (*reason).to_string(),
                        self.mixed_batch_fallback_counter(reason)
                            .load(Ordering::Relaxed),
                    )
                })
                .collect(),
        }
    }

    pub fn kv_coordinator_queue_capacity(&self) -> u64 {
        self.inner
            .kv_coordinator_queue_capacity
            .load(Ordering::Relaxed)
    }

    pub fn kv_fetch_queue_depth(&self) -> u64 {
        self.inner.kv_fetch_queue_depth.load(Ordering::Relaxed)
    }

    pub fn kv_fetch_waiters(&self) -> u64 {
        self.inner.kv_fetch_waiters.load(Ordering::Relaxed)
    }

    pub fn kv_store_queue_depth(&self) -> u64 {
        self.inner.kv_store_queue_depth.load(Ordering::Relaxed)
    }

    pub fn kv_fetch_backpressure(&self) -> bool {
        self.inner.kv_fetch_backpressure.load(Ordering::Relaxed) != 0
    }

    pub fn kv_store_backpressure(&self) -> bool {
        self.inner.kv_store_backpressure.load(Ordering::Relaxed) != 0
    }

    pub fn kv_store_submitted_total(&self) -> u64 {
        self.inner.kv_store_submitted_total.load(Ordering::Relaxed)
    }

    pub fn kv_store_completed_total(&self) -> u64 {
        self.inner.kv_store_completed_total.load(Ordering::Relaxed)
    }

    pub fn kv_store_failed_total(&self) -> u64 {
        self.inner.kv_store_failed_total.load(Ordering::Relaxed)
    }

    pub fn kv_store_rejected_total(&self) -> u64 {
        self.inner.kv_store_rejected_total.load(Ordering::Relaxed)
    }

    pub fn tier_fetch_wait_seconds(&self) -> f64 {
        micros_to_secs(self.inner.tier_fetch_wait_us.load(Ordering::Relaxed))
    }

    pub fn tier_store_wait_seconds(&self) -> f64 {
        micros_to_secs(self.inner.tier_store_wait_us.load(Ordering::Relaxed))
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

    pub fn prefix_skip_rate(&self) -> f64 {
        let prompt_tokens = self
            .inner
            .prefix_lookup_prompt_tokens_total
            .load(Ordering::Relaxed);
        if prompt_tokens == 0 {
            return 0.0;
        }
        self.inner
            .prefix_reused_tokens_total
            .load(Ordering::Relaxed) as f64
            / prompt_tokens as f64
    }

    pub fn prefix_request_hit_rate(&self) -> f64 {
        self.inner.prefix_request_hit_ppm.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    pub fn prefix_request_skip_rate(&self) -> f64 {
        self.inner.prefix_request_skip_ppm.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    pub fn session_affinity_hit_total(&self) -> u64 {
        self.inner
            .session_affinity_hit_total
            .load(Ordering::Relaxed)
    }

    pub fn session_affinity_miss_total(&self) -> u64 {
        self.inner
            .session_affinity_miss_total
            .load(Ordering::Relaxed)
    }

    pub fn matched_prefix_tokens(&self) -> u64 {
        self.inner.matched_prefix_tokens.load(Ordering::Relaxed)
    }

    pub fn resume_prefill_tokens(&self) -> u64 {
        self.inner.resume_prefill_tokens.load(Ordering::Relaxed)
    }

    pub fn prefix_lookup_latency_us(&self) -> u64 {
        self.inner.prefix_lookup_latency_us.load(Ordering::Relaxed)
    }

    pub fn prefix_lookup_reusable_tokens(&self) -> u64 {
        self.inner
            .prefix_lookup_reusable_tokens
            .load(Ordering::Relaxed)
    }

    pub fn prefix_lookup_ready_on_gpu(&self) -> bool {
        self.inner
            .prefix_lookup_ready_on_gpu
            .load(Ordering::Relaxed)
            != 0
    }

    pub fn prefix_lookup_direct_gpu_attach(&self) -> bool {
        self.inner
            .prefix_lookup_direct_gpu_attach
            .load(Ordering::Relaxed)
            != 0
    }

    pub fn prefix_lookup_staged(&self) -> bool {
        self.inner.prefix_lookup_staged.load(Ordering::Relaxed) != 0
    }

    pub fn prefix_lookup_prefetch(&self) -> bool {
        self.inner.prefix_lookup_prefetch.load(Ordering::Relaxed) != 0
    }

    pub fn prefix_lookup_recompute(&self) -> bool {
        self.inner.prefix_lookup_recompute.load(Ordering::Relaxed) != 0
    }

    pub fn session_slot_pressure_evictions_hard(&self) -> u64 {
        self.inner
            .session_slot_pressure_evictions_hard_total
            .load(Ordering::Relaxed)
    }

    pub fn prefix_aware_admit_deferrals_total(&self) -> u64 {
        self.inner
            .prefix_aware_admit_deferrals_total
            .load(Ordering::Relaxed)
    }

    pub fn tier_fetch_staged_host_blocks_total(&self) -> u64 {
        self.inner
            .tier_fetch_staged_host_blocks_total
            .load(Ordering::Relaxed)
    }

    pub fn tier_fetch_staged_disk_blocks_total(&self) -> u64 {
        self.inner
            .tier_fetch_staged_disk_blocks_total
            .load(Ordering::Relaxed)
    }

    pub fn tier_fetch_staged_remote_blocks_total(&self) -> u64 {
        self.inner
            .tier_fetch_staged_remote_blocks_total
            .load(Ordering::Relaxed)
    }

    pub fn tier_fetch_promoted_blocks_total(&self) -> u64 {
        self.inner
            .tier_fetch_promoted_blocks_total
            .load(Ordering::Relaxed)
    }

    pub fn tier_fetch_fallback_total(&self) -> u64 {
        self.inner.tier_fetch_fallback_total.load(Ordering::Relaxed)
    }

    pub fn tier_fetch_staged_blocks_total(&self) -> u64 {
        self.tier_fetch_staged_host_blocks_total()
            + self.tier_fetch_staged_disk_blocks_total()
            + self.tier_fetch_staged_remote_blocks_total()
    }

    pub fn tier_fetch_recall_rate(&self) -> f64 {
        let staged_blocks = self.tier_fetch_staged_blocks_total();
        if staged_blocks == 0 {
            return 0.0;
        }
        self.tier_fetch_promoted_blocks_total() as f64 / staged_blocks as f64
    }

    pub fn spec_draft_tokens_total(&self) -> u64 {
        self.inner.spec_draft_tokens_total.load(Ordering::Relaxed)
    }

    pub fn spec_verified_tokens_total(&self) -> u64 {
        self.inner
            .spec_verified_tokens_total
            .load(Ordering::Relaxed)
    }

    pub fn spec_accepted_tokens_total(&self) -> u64 {
        self.inner
            .spec_accepted_tokens_total
            .load(Ordering::Relaxed)
    }

    pub fn spec_sparse_view_empty_total(&self) -> u64 {
        self.inner
            .spec_sparse_view_empty_total
            .load(Ordering::Relaxed)
    }

    pub fn spec_acceptance_rate(&self) -> f64 {
        self.inner.spec_acceptance_rate_ppm.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    pub fn spec_step_latency_count(&self) -> u64 {
        self.inner
            .histograms
            .lock()
            .map_or(0, |h| h.spec_step_latency_us.count())
    }

    pub fn record_spec_step(
        &self,
        draft_tokens: usize,
        verified_tokens: usize,
        accepted_tokens: usize,
        latency_us: u64,
    ) {
        self.inner
            .spec_draft_tokens_total
            .fetch_add(draft_tokens as u64, Ordering::Relaxed);
        let verified_total = self
            .inner
            .spec_verified_tokens_total
            .fetch_add(verified_tokens as u64, Ordering::Relaxed)
            .saturating_add(verified_tokens as u64);
        let accepted_total = self
            .inner
            .spec_accepted_tokens_total
            .fetch_add(accepted_tokens as u64, Ordering::Relaxed)
            .saturating_add(accepted_tokens as u64);
        let acceptance_ppm = if verified_total == 0 {
            0
        } else {
            ((accepted_total as f64 / verified_total as f64) * 1_000_000.0).round() as u64
        };
        self.inner
            .spec_acceptance_rate_ppm
            .store(acceptance_ppm, Ordering::Relaxed);
        if let Ok(mut histograms) = self.inner.histograms.lock() {
            histograms.spec_step_latency_us.observe(latency_us as f64);
        }
    }

    pub fn record_spec_sparse_view_empty(&self, rows: usize) {
        self.inner
            .spec_sparse_view_empty_total
            .fetch_add(rows as u64, Ordering::Relaxed);
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

    /// Record one Metal decode batch that stayed on a batched GPU path.
    pub fn record_metal_decode_batch(&self, rows: usize) {
        self.inner
            .metal_decode_batches_total
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .metal_decode_batched_rows_total
            .fetch_add(rows as u64, Ordering::Relaxed);
    }

    /// Record one Metal decode row that ran through the scalar per-request path.
    pub fn record_metal_decode_scalar_row(&self) {
        self.inner
            .metal_decode_scalar_rows_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record rows the scheduler grouped but the backend could not batch.
    pub fn record_metal_decode_batch_fallback(&self, rows: usize) {
        self.inner
            .metal_decode_batch_fallback_rows_total
            .fetch_add(rows as u64, Ordering::Relaxed);
    }

    /// Record one Qwen3.5 packed decode batch.
    pub fn record_metal_qwen35_packed_decode_batch(&self, rows: usize) {
        self.inner
            .metal_qwen35_packed_decode_batches_total
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .metal_qwen35_packed_decode_rows_total
            .fetch_add(rows as u64, Ordering::Relaxed);
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
        m.set_scheduler_step(4, 3, 1, 3, 128, 4);
        m.observe_scheduler_step(0.012);
        m.set_scheduler_step_phase_us(100.0, 200.0, 300.0, 400.0, 1000.0);
        m.set_scheduler_loop_phase_us(50.0, 1050.0);
        m.set_preprocess_stage(2, 11, 22);
        m.set_scheduler_pipeline_us(33, 44, 55, 1);
        m.set_runtime_topology(
            &crate::runtime_topology::RuntimeTopology {
                numa_nodes: vec![crate::runtime_topology::NumaNodeTopology {
                    node: 0,
                    cpus: vec![0, 1],
                }],
                gpus: vec![crate::runtime_topology::GpuTopology {
                    ordinal: 0,
                    pci_bus_id: Some("17:00.0".to_string()),
                    uuid: None,
                    numa_node: Some(0),
                    local_cpus: vec![0, 1],
                    nearest_nics: vec!["mlx5_0".to_string()],
                }],
                nics: vec![crate::runtime_topology::NicTopology {
                    name: "mlx5_0".to_string(),
                    pci_bus_id: Some("18:00.0".to_string()),
                    numa_node: Some(0),
                    local_cpus: vec![0, 1],
                }],
                fallback_cpus: vec![0, 1],
            },
            &crate::runtime_topology::WorkerPlacement {
                worker_id: 0,
                gpu_ordinal: 0,
                numa_node: Some(0),
                cpus: vec![0, 1],
                nics: vec!["mlx5_0".to_string()],
                route_cost: 0,
            },
            &crate::runtime_topology::AffinityApplyResult {
                label: "test".to_string(),
                applied: true,
                requested_cpus: vec![0, 1],
                applied_threads: 2,
                failed_threads: 0,
                reason: "applied".to_string(),
            },
        );
        m.set_preprocess_topology(1, 2);
        m.set_detokenizer_topology(1, 1);
        m.set_runtime_numastat(
            &crate::runtime_topology::NumaMemoryStats {
                total_pages: 9,
                per_node_pages: vec![(0, 7), (1, 2)],
            },
            Some(0),
        );
        m.observe_h2d_latency_us(77);
        m.observe_h2d_latency_us(99);
        m.record_numa_route(0, Some(true));
        m.record_numa_route(100, Some(false));
        m.record_numa_migration();
        m.record_numa_rebalance();
        m.record_scheduler_cpu_plan_accept();
        m.record_scheduler_cpu_plan_stale();
        m.record_scheduler_plan(SchedulerPlanLabel::Decode);
        m.record_scheduler_plan(SchedulerPlanLabel::Mixed);
        m.record_scheduler_plan(SchedulerPlanLabel::Mixed);
        m.record_prefill_path_mixed_ok_true();
        m.record_prefill_path_mixed_ok_false("prefill_seq_len_mismatch");
        m.record_prefill_path_mixed_ok_false("scheduler_pre_dispatch_fallback");
        m.set_kv_coordinator(16, 3, 5, 2, true, false, 7, 5, 1, 2);
        m.set_tier_wait_seconds(0.25, 0.5);
        m.set_kv_gpu_blocks(100, 200);
        m.record_request_cache(
            Some(&crate::types::SessionId::from("w3-warm-000")),
            64,
            128,
            64,
        );
        m.record_prefix_aware_admit_deferral();
        m.record_tier_fetch_plan(2, 3, 4);
        m.record_tier_fetch_promoted(6);
        m.record_tier_fetch_fallback();
        m.record_metal_decode_batch(3);
        m.record_metal_decode_scalar_row();
        m.record_metal_decode_batch_fallback(2);
        m.record_metal_qwen35_packed_decode_batch(3);
        m.set_memory_bytes(1234, 5678, 42);

        let rendered = m.render_prometheus();
        assert!(rendered.contains("infer_requests_total"));
        assert!(rendered.contains("infer_requests_total{model=\"Qwen3-4B\",} 1"));
        assert!(rendered.contains("infer_requests_active{model=\"Qwen3-4B\",} 2"));
        assert!(rendered.contains("infer_requests_waiting{model=\"Qwen3-4B\",} 5"));
        assert!(rendered.contains("infer_scheduler_running_batch{model=\"Qwen3-4B\",} 3"));
        assert!(rendered.contains("infer_scheduler_prefill_queue{model=\"Qwen3-4B\",} 4"));
        assert!(rendered.contains("infer_scheduler_scheduled_rows{model=\"Qwen3-4B\",} 4"));
        assert!(rendered.contains("infer_scheduler_scheduled_decode_rows{model=\"Qwen3-4B\",} 3"));
        assert!(rendered.contains("infer_scheduler_scheduled_prefill_rows{model=\"Qwen3-4B\",} 1"));
        assert!(rendered.contains("infer_scheduler_decode_tokens{model=\"Qwen3-4B\",} 3"));
        assert!(rendered.contains("infer_scheduler_prefill_tokens{model=\"Qwen3-4B\",} 128"));
        assert!(rendered.contains("infer_scheduler_batch_width{model=\"Qwen3-4B\",} 4"));
        assert!(rendered.contains(
            "infer_scheduler_step_phase_admission_microseconds{model=\"Qwen3-4B\",} 100"
        ));
        assert!(
            rendered.contains(
                "infer_scheduler_step_phase_prefill_microseconds{model=\"Qwen3-4B\",} 200"
            )
        );
        assert!(
            rendered.contains(
                "infer_scheduler_step_phase_decode_microseconds{model=\"Qwen3-4B\",} 300"
            )
        );
        assert!(
            rendered
                .contains("infer_scheduler_step_phase_emit_microseconds{model=\"Qwen3-4B\",} 400")
        );
        assert!(
            rendered.contains(
                "infer_scheduler_step_phase_total_microseconds{model=\"Qwen3-4B\",} 1000"
            )
        );
        assert!(
            rendered.contains("infer_scheduler_step_cleanup_microseconds{model=\"Qwen3-4B\",} 50")
        );
        assert!(
            rendered.contains("infer_scheduler_loop_total_microseconds{model=\"Qwen3-4B\",} 1050")
        );
        assert!(rendered.contains("infer_preprocess_queue_depth{model=\"Qwen3-4B\",} 2"));
        assert!(rendered.contains("infer_preprocess_wait_microseconds{model=\"Qwen3-4B\",} 11"));
        assert!(
            rendered.contains("infer_preprocess_tokenize_microseconds{model=\"Qwen3-4B\",} 22")
        );
        assert!(
            rendered
                .contains("infer_scheduler_pipeline_snapshot_microseconds{model=\"Qwen3-4B\",} 33")
        );
        assert!(
            rendered
                .contains("infer_scheduler_pipeline_cpu_plan_microseconds{model=\"Qwen3-4B\",} 44")
        );
        assert!(rendered.contains(
            "infer_scheduler_pipeline_gpu_completion_wait_microseconds{model=\"Qwen3-4B\",} 55"
        ));
        assert!(
            rendered.contains(
                "infer_scheduler_pipeline_gpu_command_queue_depth{model=\"Qwen3-4B\",} 1"
            )
        );
        assert!(rendered.contains(
            "infer_scheduler_pipeline_cpu_plan_total{model=\"Qwen3-4B\",outcome=\"accept\",} 1"
        ));
        assert!(rendered.contains(
            "infer_scheduler_pipeline_cpu_plan_total{model=\"Qwen3-4B\",outcome=\"stale\",} 1"
        ));
        assert!(rendered.contains("infer_runtime_topology_numa_nodes{model=\"Qwen3-4B\",} 1"));
        assert!(rendered.contains("infer_runtime_worker_affinity_applied{model=\"Qwen3-4B\",} 1"));
        assert!(
            rendered.contains(
                "infer_runtime_numastat_pages{model=\"Qwen3-4B\",placement=\"local\",} 7"
            )
        );
        assert!(rendered.contains(
            "infer_runtime_h2d_latency_microseconds{model=\"Qwen3-4B\",stat=\"max\",} 99"
        ));
        assert!(
            rendered.contains(
                "infer_scheduler_numa_route_total{model=\"Qwen3-4B\",outcome=\"cross\",} 1"
            )
        );
        assert!(rendered.contains("infer_scheduler_numa_migration_total{model=\"Qwen3-4B\",} 1"));
        assert!(
            rendered.contains("infer_scheduler_plan_total{model=\"Qwen3-4B\",plan=\"decode\",} 1")
        );
        assert!(
            rendered.contains("infer_scheduler_plan_total{model=\"Qwen3-4B\",plan=\"mixed\",} 2")
        );
        assert!(rendered.contains(
            "infer_prefill_path_mixed_batch_total{model=\"Qwen3-4B\",outcome=\"ok_true\",} 1"
        ));
        assert!(rendered.contains(
            "infer_prefill_path_mixed_batch_total{model=\"Qwen3-4B\",outcome=\"ok_false\",} 2"
        ));
        assert!(
            rendered.contains(
                "infer_prefill_path_mixed_batch_fallback_total{model=\"Qwen3-4B\",reason=\"prefill_seq_len_mismatch\",} 1"
            )
        );
        assert!(
            rendered.contains(
                "infer_prefill_path_mixed_batch_fallback_total{model=\"Qwen3-4B\",reason=\"scheduler_pre_dispatch_fallback\",} 1"
            )
        );
        assert!(rendered.contains("infer_spec_draft_tokens_total{model=\"Qwen3-4B\",} 0"));
        assert!(rendered.contains("infer_spec_verified_tokens_total{model=\"Qwen3-4B\",} 0"));
        assert!(rendered.contains("infer_spec_accepted_tokens_total{model=\"Qwen3-4B\",} 0"));
        assert!(rendered.contains("infer_spec_sparse_view_empty_total{model=\"Qwen3-4B\",} 0"));
        assert!(rendered.contains("infer_spec_acceptance_rate{model=\"Qwen3-4B\",} 0.000000"));
        assert!(rendered.contains("infer_kv_coordinator_queue_capacity{model=\"Qwen3-4B\",} 16"));
        assert!(rendered.contains("infer_kv_fetch_queue_depth{model=\"Qwen3-4B\",} 3"));
        assert!(rendered.contains("infer_kv_fetch_waiters{model=\"Qwen3-4B\",} 5"));
        assert!(rendered.contains("infer_kv_store_queue_depth{model=\"Qwen3-4B\",} 2"));
        assert!(rendered.contains("infer_kv_store_submitted_total{model=\"Qwen3-4B\",} 7"));
        assert!(rendered.contains("infer_kv_store_completed_total{model=\"Qwen3-4B\",} 5"));
        assert!(rendered.contains("infer_kv_store_failed_total{model=\"Qwen3-4B\",} 1"));
        assert!(rendered.contains("infer_kv_store_rejected_total{model=\"Qwen3-4B\",} 2"));
        assert!(rendered.contains("infer_kv_fetch_backpressure{model=\"Qwen3-4B\",} 1"));
        assert!(rendered.contains("infer_kv_store_backpressure{model=\"Qwen3-4B\",} 0"));
        assert!(rendered.contains("infer_tier_fetch_wait_seconds{model=\"Qwen3-4B\",} 0.250000"));
        assert!(rendered.contains("infer_tier_store_wait_seconds{model=\"Qwen3-4B\",} 0.500000"));
        assert!(rendered.contains("infer_prefix_hit_rate{model=\"Qwen3-4B\",} 1.0000"));
        assert!(rendered.contains("infer_prefix_reused_tokens_total{model=\"Qwen3-4B\",} 64"));
        assert!(
            rendered.contains("infer_prefix_lookup_prompt_tokens_total{model=\"Qwen3-4B\",} 128")
        );
        assert!(rendered.contains("infer_prefix_skip_rate{model=\"Qwen3-4B\",} 0.5000"));
        assert!(rendered.contains("infer_prefix_request_hit_rate{model=\"Qwen3-4B\",} 1.0000"));
        assert!(rendered.contains("infer_prefix_request_skip_rate{model=\"Qwen3-4B\",} 0.5000"));
        assert!(rendered.contains("infer_session_affinity_hit_total{model=\"Qwen3-4B\",} 1"));
        assert!(rendered.contains("infer_session_affinity_miss_total{model=\"Qwen3-4B\",} 0"));
        assert!(
            rendered.contains(
                "infer_session_slot_pressure_evictions_hard_total{model=\"Qwen3-4B\",} 0"
            )
        );
        assert!(
            rendered.contains("infer_prefix_aware_admit_deferrals_total{model=\"Qwen3-4B\",} 1")
        );
        assert!(rendered.contains("infer_matched_prefix_tokens{model=\"Qwen3-4B\",} 64"));
        assert!(rendered.contains("infer_resume_prefill_tokens{model=\"Qwen3-4B\",} 64"));
        assert!(
            rendered.contains("infer_tier_fetch_staged_host_blocks_total{model=\"Qwen3-4B\",} 2")
        );
        assert!(
            rendered.contains("infer_tier_fetch_staged_disk_blocks_total{model=\"Qwen3-4B\",} 3")
        );
        assert!(
            rendered.contains("infer_tier_fetch_staged_remote_blocks_total{model=\"Qwen3-4B\",} 4")
        );
        assert!(rendered.contains("infer_tier_fetch_promoted_blocks_total{model=\"Qwen3-4B\",} 6"));
        assert!(rendered.contains("infer_tier_fetch_fallback_total{model=\"Qwen3-4B\",} 1"));
        assert!(rendered.contains("infer_tier_fetch_recall_rate{model=\"Qwen3-4B\",} 0.6667"));
        assert!(rendered.contains("infer_metal_decode_batches_total{model=\"Qwen3-4B\",} 1"));
        assert!(rendered.contains("infer_metal_decode_batched_rows_total{model=\"Qwen3-4B\",} 3"));
        assert!(rendered.contains("infer_metal_decode_scalar_rows_total{model=\"Qwen3-4B\",} 1"));
        assert!(
            rendered
                .contains("infer_metal_decode_batch_fallback_rows_total{model=\"Qwen3-4B\",} 2")
        );
        assert!(
            rendered
                .contains("infer_metal_qwen35_packed_decode_batches_total{model=\"Qwen3-4B\",} 1")
        );
        assert!(
            rendered.contains("infer_metal_qwen35_packed_decode_rows_total{model=\"Qwen3-4B\",} 3")
        );
        assert!(rendered.contains("infer_memory_active_bytes{model=\"Qwen3-4B\",} 1234"));
        assert!(rendered.contains("infer_queue_wait_seconds_count"));
        assert!(rendered.contains("infer_active_ttft_seconds_count"));
        assert!(rendered.contains("infer_ttft_seconds_count"));
        assert!(rendered.contains("infer_tpot_seconds_count"));
        assert!(rendered.contains("infer_service_seconds_count"));
        assert!(rendered.contains("infer_e2e_seconds_count"));
        assert!(rendered.contains("infer_scheduler_step_seconds_count"));
        assert!(rendered.contains("infer_spec_step_latency_us_count{model=\"Qwen3-4B\",} 0"));
        assert!(rendered.contains("infer_kv_gpu_blocks_free{model=\"Qwen3-4B\",} 100"));
        assert!(rendered.contains("infer_kv_gpu_blocks_total{model=\"Qwen3-4B\",} 200"));
    }

    #[test]
    fn server_metrics_render_summary() {
        let m = ServerMetrics::new("Qwen3-8B");
        m.set_kv_coordinator(16, 0, 0, 0, false, false, 3, 2, 1, 4);
        m.record_request_cache(
            Some(&crate::types::SessionId::from("w3-warm-000")),
            32,
            128,
            96,
        );
        m.record_tier_fetch_plan(1, 2, 0);
        m.record_tier_fetch_promoted(2);
        m.record_tier_fetch_fallback();
        m.record_session_slot_pressure_evictions_hard(3);
        m.record_prefix_aware_admit_deferral();
        m.record_metal_decode_batch(4);
        m.record_metal_decode_scalar_row();
        m.record_metal_decode_batch_fallback(3);
        m.record_metal_qwen35_packed_decode_batch(4);
        m.set_scheduler_step_phase_us(11.0, 22.0, 33.0, 44.0, 110.0);
        m.set_scheduler_loop_phase_us(55.0, 165.0);
        m.set_preprocess_stage(1, 7, 8);
        m.set_scheduler_pipeline_us(9, 10, 11, 1);
        m.record_scheduler_cpu_plan_accept();
        m.record_scheduler_plan(SchedulerPlanLabel::Prefill);
        m.record_scheduler_plan(SchedulerPlanLabel::Split);
        m.record_scheduler_plan(SchedulerPlanLabel::Mixed);
        m.record_scheduler_plan(SchedulerPlanLabel::Mixed);
        m.record_prefill_path_mixed_ok_true();
        m.record_prefill_path_mixed_ok_false("prefill_seq_len_mismatch");
        m.record_prefill_path_mixed_ok_false("scheduler_pre_dispatch_fallback");
        let s = m.render_summary();
        assert!(s.contains("requests=0"));
        assert!(s.contains("active=0"));
        assert!(s.contains("scheduled=0"));
        assert!(s.contains(
            "step_phase_us=adm:11,prefill:22,decode:33,emit:44,total:110,cleanup:55,loop_total:165"
        ));
        assert!(
            s.contains("preprocess=depth:1,wait_us:7,tokenize_us:8 pipeline=snapshot_us:9,cpu_plan_us:10,gpu_wait_us:11,gpu_q:1,plan_accept:1,plan_stale:0")
        );
        assert!(s.contains("plan_label=idle:0,decode:0,prefill:1,split:1,mixed:2"));
        assert!(s.contains(
            "prefill_path=ok_true:1,ok_false:2,prefill_seq_len_mismatch:1,scheduler_pre_dispatch_fallback:1"
        ));
        assert!(
            s.contains(
                "spec=draft:0,verified:0,accepted:0,empty_sparse_views:0,accept_rate:0.0%,step_latency_count:0"
            )
        );
        assert!(s.contains("queue_p50="));
        assert!(s.contains("prefix_hit_rate=100.0%"));
        assert!(s.contains("prefix_skip_rate=25.0%"));
        assert!(s.contains("prefix_request_hit_rate=100.0%"));
        assert!(s.contains("prefix_request_skip_rate=25.0%"));
        assert!(s.contains("session_affinity_hit=1"));
        assert!(s.contains("session_affinity_miss=0"));
        assert!(s.contains("session_slot_pressure_evictions_hard=3"));
        assert!(s.contains("prefix_aware_admit_deferrals=1"));
        assert!(s.contains("matched_prefix_tokens=32"));
        assert!(s.contains("resume_prefill_tokens=96"));
        assert!(s.contains("tier_recall=66.7%"));
        assert!(s.contains("tier_src=h:1/d:2/r:0"));
        assert!(s.contains("tier_promoted=2"));
        assert!(s.contains("tier_fallback=1"));
        assert!(s.contains("metal_decode=batch:1/4,scalar:1,fallback:3,qwen35_packed:1/4"));
        assert!(s.contains("kv_store=sub:3,done:2,fail:1,rej:4"));
    }

    #[test]
    fn server_metrics_render_stats_json_agent_cache_fields() {
        let m = ServerMetrics::new("Qwen3-4B");
        m.set_model_arch(crate::model_arch::ModelArchSummary {
            arch: crate::model_registry::ModelArch::Qwen3,
            hidden_size: 2560,
            vocab_size: 151_936,
            num_hidden_layers: 36,
            num_kv_layers: 36,
            num_kv_heads: 8,
            num_q_heads: 32,
            head_dim: 128,
            kv_cache_bytes_per_token: 147_456,
        });
        m.record_request_cache(
            Some(&crate::types::SessionId::from("w3-warm-000")),
            64,
            128,
            64,
        );
        m.record_prefill_path_mixed_ok_true();
        m.record_prefill_path_mixed_ok_false("prefill_seq_len_mismatch");
        m.record_prefill_path_mixed_ok_false("scheduler_pre_dispatch_fallback");
        m.record_prefix_aware_admit_deferral();
        m.set_preprocess_stage(3, 21, 34);
        m.set_scheduler_pipeline_us(55, 89, 144, 1);
        m.set_runtime_topology(
            &crate::runtime_topology::RuntimeTopology {
                numa_nodes: vec![crate::runtime_topology::NumaNodeTopology {
                    node: 1,
                    cpus: vec![2, 3],
                }],
                gpus: Vec::new(),
                nics: Vec::new(),
                fallback_cpus: vec![2, 3],
            },
            &crate::runtime_topology::WorkerPlacement {
                worker_id: 0,
                gpu_ordinal: 0,
                numa_node: Some(1),
                cpus: vec![2, 3],
                nics: Vec::new(),
                route_cost: 0,
            },
            &crate::runtime_topology::AffinityApplyResult {
                label: "test".to_string(),
                applied: true,
                requested_cpus: vec![2, 3],
                applied_threads: 1,
                failed_threads: 0,
                reason: "applied".to_string(),
            },
        );
        m.set_preprocess_topology(1, 2);
        m.set_detokenizer_topology(1, 1);
        m.observe_h2d_latency_us(123);
        m.record_scheduler_cpu_plan_accept();
        m.record_scheduler_cpu_plan_stale();

        let payload = m.render_stats_json();
        assert_eq!(payload["prefix_hit_rate"], serde_json::json!(1.0));
        assert_eq!(payload["prefix_skip_rate"], serde_json::json!(0.5));
        assert_eq!(
            payload["engine_model_arch"]["arch"],
            serde_json::json!("Qwen3")
        );
        assert_eq!(
            payload["engine_model_arch"]["num_kv_layers"],
            serde_json::json!(36)
        );
        assert_eq!(
            m.snapshot_engine_telemetry()
                .model_arch
                .as_ref()
                .map(crate::model_arch::ModelArchSummary::arch_label),
            Some("Qwen3")
        );
        assert_eq!(
            payload["engine_prefill_path_stats"]["ok_true_count"],
            serde_json::json!(1)
        );
        assert_eq!(
            payload["engine_prefill_path_stats"]["ok_false_count"],
            serde_json::json!(2)
        );
        assert_eq!(
            payload["engine_prefill_path_stats"]["ok_false_reasons"]["prefill_seq_len_mismatch"],
            serde_json::json!(1)
        );
        assert_eq!(
            payload["engine_prefill_path_stats"]["ok_false_reasons"]["scheduler_pre_dispatch_fallback"],
            serde_json::json!(1)
        );
        assert_eq!(payload["preprocess"]["queue_depth"], serde_json::json!(3));
        assert_eq!(payload["preprocess"]["wait_us"], serde_json::json!(21));
        assert_eq!(payload["preprocess"]["tokenize_us"], serde_json::json!(34));
        assert_eq!(
            payload["scheduler_pipeline"]["snapshot_us"],
            serde_json::json!(55)
        );
        assert_eq!(
            payload["scheduler_pipeline"]["cpu_plan_us"],
            serde_json::json!(89)
        );
        assert_eq!(
            payload["scheduler_pipeline"]["gpu_completion_wait_us"],
            serde_json::json!(144)
        );
        assert_eq!(
            payload["scheduler_pipeline"]["gpu_command_queue_depth"],
            serde_json::json!(1)
        );
        assert_eq!(
            payload["scheduler_pipeline"]["cpu_plan_accept_total"],
            serde_json::json!(1)
        );
        assert_eq!(
            payload["scheduler_pipeline"]["cpu_plan_stale_total"],
            serde_json::json!(1)
        );
        assert_eq!(
            payload["runtime_topology"]["worker_numa_node"],
            serde_json::json!(1)
        );
        assert_eq!(
            payload["runtime_topology"]["preprocess_workers"],
            serde_json::json!(2)
        );
        assert_eq!(
            payload["runtime_topology"]["h2d_latency_last_us"],
            serde_json::json!(123)
        );
        assert_eq!(payload["session_affinity_hit"], serde_json::json!(1));
        assert_eq!(payload["session_affinity_miss"], serde_json::json!(0));
        assert_eq!(
            payload["session_slot_pressure_evictions_hard"],
            serde_json::json!(0)
        );
        assert_eq!(
            payload["prefix_aware_admit_deferrals"],
            serde_json::json!(1)
        );
        assert_eq!(payload["matched_prefix_tokens"], serde_json::json!(64));
        assert_eq!(payload["resume_prefill_tokens"], serde_json::json!(64));
        assert_eq!(
            payload["last_request"]["session_id"],
            serde_json::json!("w3-warm-000")
        );
        assert_eq!(
            payload["last_request"]["prefix_skip_rate"],
            serde_json::json!(0.5)
        );
        assert_eq!(
            payload["sessions"]["w3-warm-000"]["prefix_hit_rate"],
            serde_json::json!(1.0)
        );
        assert_eq!(
            payload["sessions"]["w3-warm-000"]["prefix_skip_rate"],
            serde_json::json!(0.5)
        );
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
