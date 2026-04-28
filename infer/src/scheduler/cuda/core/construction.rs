//! `Scheduler<M>` constructor methods (`new`, `with_max_seq_len`, `with_config`).
//!
//! Split out of `core.rs` (pure structural refactor — no behavior change).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use anyhow::Result;
use log::info;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use super::super::policy::TieredKvPolicy;
use super::super::{SchedulerConfig, SchedulerHandle, SeedableRng};
use super::{
    CONTIGUOUS_KV_TOKENS, PREFIX_CACHE_BLOCK_SIZE, Scheduler, SchedulerRuntimeStats,
    spawn_emit_worker,
};
use crate::backend::cuda::paged_kv::PagedKVPool;
use crate::kv_tier::transport::DiskStore;
use crate::model::{GenerationState, ModelForward};
use crate::prefix_cache::RadixCache;
use crate::tokenizer::Tokenizer;

impl<M: ModelForward> Scheduler<M> {
    /// Create a new scheduler and its handle.
    ///
    /// `num_slots` controls how many concurrent requests can be active (each gets
    /// its own KV cache). More slots = more GPU memory usage.
    pub fn new(
        model: M,
        tokenizer: Tokenizer,
        model_id: &str,
        num_slots: usize,
        seed: u64,
        metrics: crate::metrics::ServerMetrics,
    ) -> Result<(Self, SchedulerHandle)> {
        Self::with_config(
            model,
            tokenizer,
            model_id,
            seed,
            metrics,
            SchedulerConfig::runtime_defaults(num_slots),
            None,
            crate::model::kv_cache::KVCacheDtype::BF16,
            crate::model::kv_cache::KVFormat::BF16,
        )
    }

    /// Create a new scheduler with an explicit max sequence length override.
    pub fn with_max_seq_len(
        model: M,
        tokenizer: Tokenizer,
        model_id: &str,
        num_slots: usize,
        seed: u64,
        metrics: crate::metrics::ServerMetrics,
        max_seq_len_override: Option<usize>,
    ) -> Result<(Self, SchedulerHandle)> {
        Self::with_config(
            model,
            tokenizer,
            model_id,
            seed,
            metrics,
            SchedulerConfig::runtime_defaults(num_slots),
            max_seq_len_override,
            crate::model::kv_cache::KVCacheDtype::BF16,
            crate::model::kv_cache::KVFormat::BF16,
        )
    }

    /// Create a scheduler from an explicit runtime configuration.
    pub fn with_config(
        model: M,
        tokenizer: Tokenizer,
        model_id: &str,
        seed: u64,
        metrics: crate::metrics::ServerMetrics,
        config: SchedulerConfig,
        max_seq_len_override: Option<usize>,
        kv_cache_dtype: crate::model::kv_cache::KVCacheDtype,
        kv_pool_format: crate::model::kv_cache::KVFormat,
    ) -> Result<(Self, SchedulerHandle)> {
        config.validate()?;

        let (tx, rx) = mpsc::unbounded_channel();
        let (wakeup_tx, wakeup_rx) = crossbeam_channel::unbounded();
        let effective_max_seq_len =
            Self::compute_max_seq_len(&model, &config, max_seq_len_override);
        let effective_prefill_token_budget = config.max_prefill_tokens;
        let effective_mixed_prefill_token_budget = config.mixed_prefill_token_budget();

        // When the model writes prefill K/V directly to the paged pool, the
        // per-slot contiguous scratch buffer is unused by prefill. Shrink it
        // to the minimum that single-token decode / INT8 working buffers
        // still require, and reclaim the freed bytes into the pool budget.
        let model_uses_paged_prefill = model.prefill_uses_paged_pool();
        let contiguous_tokens = if model_uses_paged_prefill {
            // Single-token decode path still allocates per-slot contiguous
            // K/V of this size; 1 page's worth is enough.
            PREFIX_CACHE_BLOCK_SIZE
        } else {
            CONTIGUOUS_KV_TOKENS
        };

        let mut states = Vec::with_capacity(config.max_slots);
        let mut slot_materialized_prompt_lens = Vec::with_capacity(config.max_slots);
        let mut slot_owned_blocks = Vec::with_capacity(config.max_slots);
        for i in 0..config.max_slots {
            let mut state = model.create_state()?;
            state.set_max_seq_len(contiguous_tokens);
            state.set_kv_dtype(kv_cache_dtype);
            states.push(state);
            slot_materialized_prompt_lens.push(0);
            slot_owned_blocks.push(Vec::new());
            info!("Initialized state slot {}/{}", i + 1, config.max_slots);
        }

        let paged_kv_pool = {
            let bytes_per_token = model.kv_cache_bytes_per_token();
            let contiguous_cost = config.max_slots * contiguous_tokens * bytes_per_token;
            // Estimated runtime workspace — kept for the OOM safety check
            // below, NOT subtracted from the budget. SGLang's
            // `profile_max_num_token` (`sglang/srt/model_executor/model_runner_kv_cache_mixin.py:171-177`)
            // does not pre-deduct workspace; it lets workspace allocate
            // from the leftover headroom dynamically. Pre-deducting our
            // ~0.9 GB workspace at default settings ate KV-pool capacity
            // 1:1 — flipping to SGLang's policy grows the pool by the
            // same amount with negligible OOM risk as long as
            // `headroom >= runtime_workspace + safety`. The assertion
            // below makes that contract explicit.
            let runtime_workspace = model.scheduler_runtime_workspace_bytes(
                crate::model::SchedulerRuntimeWorkspaceBudget {
                    max_batch_size: config.max_slots,
                    prefill_tokens: effective_prefill_token_budget,
                    mixed_prefill_tokens: effective_mixed_prefill_token_budget,
                    max_seq_len: effective_max_seq_len,
                    kv_pool_format,
                },
            );
            let budget_bytes = match crate::backend::cuda::tensor::DeviceContext::gpu_memory_info()
            {
                Ok((free, total)) => {
                    // Headroom = `(1 - mem_fraction_static) × X`. SGLang's
                    // `profile_max_num_token` uses `pre_model_load_memory`
                    // (free at process start, before model load) for X;
                    // the bootstrap path captures that snapshot into
                    // `config.pre_model_free_bytes`. When unavailable we
                    // fall back to `total`, which over-counts the driver
                    // overhead (~500 MB-1 GB on L4) and shrinks the KV
                    // pool by that much.
                    let headroom_base = config.pre_model_free_bytes.unwrap_or(total);
                    let headroom =
                        ((headroom_base as f64) * (1.0 - config.mem_fraction_static)) as usize;
                    if headroom < runtime_workspace {
                        log::warn!(
                            "TokenKVPool: estimated workspace {:.1} GB exceeds headroom {:.1} GB \
                             (mem-fraction={:.0}%). Risk of OOM mid-request — consider lowering \
                             --mem-fraction-static.",
                            runtime_workspace as f64 / 1e9,
                            headroom as f64 / 1e9,
                            config.mem_fraction_static * 100.0,
                        );
                    }
                    free.saturating_sub(contiguous_cost.saturating_add(headroom))
                }
                Err(_) => config.kv_pool_fallback_bytes,
            };

            info!(
                "TokenKVPool budget: {:.1} GB (contiguous={:.1} GB, est_workspace={:.1} GB, fraction={:.0}%)",
                budget_bytes as f64 / 1e9,
                contiguous_cost as f64 / 1e9,
                runtime_workspace as f64 / 1e9,
                config.mem_fraction_static * 100.0,
            );

            let ctx = model.device_context();
            PagedKVPool::with_format(
                ctx,
                model.num_kv_layers(),
                model.num_kv_heads(),
                model.head_dim(),
                config.max_slots,
                budget_bytes,
                kv_pool_format,
            )?
        };
        let host_block_bytes = paged_kv_pool.storage_bytes_for_tokens(PREFIX_CACHE_BLOCK_SIZE);
        let host_pool_capacity = host_block_bytes
            .saturating_mul(config.max_slots.saturating_mul(16).max(1))
            .max(64 * 1024 * 1024);
        let host_pinned_pool = crate::kv_tier::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(host_pool_capacity)?,
        );

        info!(
            "Scheduler ready: model={}, slots={}, seed={}, max_seq_len={}, max_waiting={}, chunked_prefill_size={}, max_num_batched_tokens={}, max_prefill_tokens={}, prefill_max_requests={}, host_pool={:.1}MB",
            model_id,
            config.max_slots,
            seed,
            effective_max_seq_len.map_or_else(|| "32768 (default)".to_string(), |n| n.to_string()),
            config.max_waiting_requests,
            config.chunked_prefill_size,
            config.max_num_batched_tokens,
            config.max_prefill_tokens,
            config
                .prefill_max_requests
                .map_or_else(|| "none".to_string(), |v| v.to_string()),
            host_pool_capacity as f64 / 1e6,
        );

        let waiting_count = Arc::new(AtomicUsize::new(0));
        let disk_store = Arc::new(DiskStore::new(config.disk_store_root.clone()));
        let cluster_shared_backend = config
            .cluster_shared_backend
            .as_ref()
            .map(crate::kv_tier::ClusterSharedBackendConfig::build);
        let coordinator_queue_capacity = config.max_slots.max(16);
        let mut coord_builder = crate::kv_tier::CoordinatorBuilder::new(coordinator_queue_capacity)
            .disk_store(Arc::clone(&disk_store));
        if let Some(backend) = cluster_shared_backend.clone() {
            coord_builder = coord_builder.cluster_shared_backend(backend);
        }
        let (coordinator, coordinator_handle, coordinator_events) = coord_builder.build();
        let coordinator_thread = Some(coordinator.spawn("infer-tiered-kv-coord"));
        let (emit_tx, emit_events, emit_thread) = spawn_emit_worker(tokenizer.clone());
        let max_slots = config.max_slots;
        let max_waiting_requests = config.max_waiting_requests;
        let prefix_cache_keepalive_ticks = config.prefix_cache_keepalive_ticks;
        let scheduler = Self {
            config,
            metrics,
            model,
            tokenizer,
            model_fingerprint: blake3::hash(model_id.as_bytes()).as_bytes().to_vec(),
            states,
            slot_materialized_prompt_lens,
            prefix_cache: RadixCache::with_soft_pin_keepalive(
                PREFIX_CACHE_BLOCK_SIZE,
                prefix_cache_keepalive_ticks,
            ),
            disk_store,
            cluster_shared_backend,
            tier_policy: TieredKvPolicy::default(),
            host_pinned_pool,
            block_to_pages: HashMap::new(),
            block_owner_slots: HashMap::new(),
            slot_owned_blocks,
            coordinator_handle,
            coordinator_events,
            coordinator_thread,
            emit_tx,
            emit_events,
            emit_thread: Some(emit_thread),
            emit_gate_waiting: HashMap::new(),
            store_waiting: HashMap::new(),
            store_dedupe: HashMap::new(),
            store_ticket_keys: HashMap::new(),
            store_ticket_started_at: HashMap::new(),
            fetch_waiting: HashMap::new(),
            fetch_dedupe: HashMap::new(),
            fetch_ticket_keys: HashMap::new(),
            fetch_ticket_started_at: HashMap::new(),
            prefetch_fetching: HashMap::new(),
            request_rx: rx,
            wakeup_rx,
            wakeup_live: true,
            waiting_count: Arc::clone(&waiting_count),
            waiting: VecDeque::new(),
            active: (0..max_slots).map(|_| None).collect(),
            prefill_queue: VecDeque::new(),
            running_batch: VecDeque::new(),
            effective_max_seq_len,
            next_id: 0,
            rng: StdRng::seed_from_u64(seed),
            paged_kv_pool,
            decode_bufs: None,
            prefill_ctx: None,
            stats: SchedulerRuntimeStats::new(),
            pending_decode: None,
            pending_prefill: None,
        };

        let handle = SchedulerHandle::with_shared_waiting_count_and_wakeup(
            tx,
            wakeup_tx,
            model_id,
            max_waiting_requests,
            Arc::clone(&waiting_count),
        )
        .with_tokenizer(scheduler.tokenizer.clone());
        debug_assert_eq!(handle.waiting_count(), 0);

        Ok((scheduler, handle))
    }
}
