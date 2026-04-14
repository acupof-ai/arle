use super::*;
use crate::prefix_cache::{BlockId, RadixCache};
use crate::scheduler::policy::{ChunkingPolicy, DecodeAwareChunking, SchedulerSignals};
use crate::types::InferenceMode;

/// Block size (in tokens) for the global `RadixCache` prefix observer.
/// Chosen to match the M0.3 target paged-pool page size so that when
/// M2 dual residency wires the radix directly onto the pool, the block
/// boundaries already agree.
pub(super) const PREFIX_CACHE_BLOCK_SIZE: usize = 16;

/// CUDA-backed scheduler state and initialization.
pub struct Scheduler<M: ModelForward> {
    pub(super) config: SchedulerConfig,
    pub(super) model: M,
    pub(super) tokenizer: Tokenizer,
    /// Per-slot states (KV caches, decode buffers). Stored separately from
    /// slot metadata so we can pass `&mut [M::State]` to batched decode.
    pub(super) states: Vec<M::State>,
    /// Per-slot cached prompts for prefix reuse — authoritative for the
    /// per-slot KV state until M2 dual residency lands. Lookups still
    /// scan this linearly in `best_prefix_slot`; the radix cache below
    /// is a shadow observer that accumulates cross-slot prefix stats
    /// and will take over slot selection once pool pages gain refcounts.
    pub(super) cached_prompts: Vec<Vec<u32>>,
    /// Global cross-slot prefix observer. Owned by the single-writer
    /// scheduler thread, no lock needed. Populated on `cleanup()` with
    /// each completed request's prompt and queried on `assign_slots()`
    /// to surface the best achievable prefix hit length in the logs.
    ///
    /// M1 scope: infrastructure only, behavior unchanged. The
    /// `BlockId`s stored under each node are **synthetic** — one fresh
    /// id per `PREFIX_CACHE_BLOCK_SIZE`-token block, minted by
    /// `next_prefix_block_id`. They do not correspond to any
    /// paged-pool index yet; the mapping onto pool pages is M2.
    pub(super) prefix_cache: RadixCache,
    /// Monotonic counter for synthetic `BlockId`s inserted into
    /// `prefix_cache`. Wraps on u32 overflow, which at 16 tokens per
    /// block and ~10 M blocks per overflow cycle is not a concern for
    /// any realistic deployment lifetime.
    pub(super) next_prefix_block_id: u32,
    pub(super) request_rx: mpsc::UnboundedReceiver<IncomingRequest>,
    /// Shared waiting count with the handle (for backpressure decrement).
    pub(super) waiting_count: Arc<AtomicUsize>,
    pub(super) waiting: VecDeque<IncomingRequest>,
    pub(super) active: Vec<ActiveRequest>,
    pub(super) next_id: u64,
    pub(super) rng: StdRng,
    /// Paged KV cache pool shared across all slots (for batched decode).
    pub(super) paged_kv_pool: PagedKVPool,
    /// Pre-allocated buffers for batched decode (reused across steps).
    /// Typed via `M::DecodeContext` — no downcasting needed.
    pub(super) decode_bufs: Option<M::DecodeContext>,
    /// Round-robin index for fair decode scheduling.
    pub(super) last_served: usize,
    /// Lifetime stats.
    pub(super) total_completed: u64,
    pub(super) total_generated_tokens: u64,
    /// EMA step timing (microseconds) for /v1/stats profiling.
    pub(super) step_timing_decode_us: f64,
    pub(super) step_timing_emit_us: f64,
    pub(super) step_timing_prefill_us: f64,
    pub(super) step_timing_total_us: f64,
}

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
    ) -> Result<(Self, SchedulerHandle)> {
        Self::with_config(
            model,
            tokenizer,
            model_id,
            seed,
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
        max_seq_len_override: Option<usize>,
    ) -> Result<(Self, SchedulerHandle)> {
        Self::with_config(
            model,
            tokenizer,
            model_id,
            seed,
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
        config: SchedulerConfig,
        max_seq_len_override: Option<usize>,
        kv_cache_dtype: crate::model::kv_cache::KVCacheDtype,
        kv_pool_format: crate::model::kv_cache::KVFormat,
    ) -> Result<(Self, SchedulerHandle)> {
        config.validate()?;

        let (tx, rx) = mpsc::unbounded_channel();
        let effective_max_seq_len =
            Self::compute_max_seq_len(&model, &config, max_seq_len_override);

        let mut states = Vec::with_capacity(config.max_slots);
        let mut cached_prompts = Vec::with_capacity(config.max_slots);
        for i in 0..config.max_slots {
            let mut state = model.create_state()?;
            if let Some(max_seq) = effective_max_seq_len {
                state.set_max_seq_len(max_seq);
            }
            state.set_kv_dtype(kv_cache_dtype);
            states.push(state);
            cached_prompts.push(Vec::new());
            info!("Initialized state slot {}/{}", i + 1, config.max_slots);
        }

        let paged_kv_pool = {
            let contiguous_max = effective_max_seq_len.unwrap_or(1024);
            let bytes_per_token = model.kv_cache_bytes_per_token();
            let contiguous_cost = config.max_slots * contiguous_max * bytes_per_token;
            let headroom = config.kv_pool_headroom_bytes;
            let budget_bytes = match crate::backend::cuda::tensor::DeviceContext::gpu_memory_info()
            {
                Ok((free, _)) => free.saturating_sub(contiguous_cost.saturating_add(headroom)),
                Err(_) => config.kv_pool_fallback_bytes,
            };

            info!(
                "TokenKVPool budget: {:.1} GB (contiguous KV={:.1} GB, headroom={:.1} GB)",
                budget_bytes as f64 / 1e9,
                contiguous_cost as f64 / 1e9,
                headroom as f64 / 1e9,
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

        info!(
            "Scheduler ready: model={}, slots={}, seed={}, max_seq_len={}, max_waiting={}, prefill_chunk_size={}",
            model_id,
            config.max_slots,
            seed,
            effective_max_seq_len.map_or_else(|| "32768 (default)".to_string(), |n| n.to_string()),
            config.max_waiting_requests,
            config.prefill_chunk_size,
        );

        let waiting_count = Arc::new(AtomicUsize::new(0));
        let scheduler = Self {
            config: config.clone(),
            model,
            tokenizer,
            states,
            cached_prompts,
            prefix_cache: RadixCache::new(PREFIX_CACHE_BLOCK_SIZE),
            next_prefix_block_id: 0,
            request_rx: rx,
            waiting_count: Arc::clone(&waiting_count),
            waiting: VecDeque::new(),
            active: Vec::new(),
            next_id: 0,
            rng: StdRng::seed_from_u64(seed),
            paged_kv_pool,
            decode_bufs: None,
            last_served: 0,
            total_completed: 0,
            total_generated_tokens: 0,
            step_timing_decode_us: 0.0,
            step_timing_emit_us: 0.0,
            step_timing_prefill_us: 0.0,
            step_timing_total_us: 0.0,
        };

        let handle = SchedulerHandle::with_shared_waiting_count(
            tx,
            model_id,
            config.max_waiting_requests,
            Arc::clone(&waiting_count),
        );
        debug_assert_eq!(handle.waiting_count(), 0);

        Ok((scheduler, handle))
    }

    /// Compute the effective max_seq_len per slot based on available GPU memory.
    fn compute_max_seq_len(
        model: &M,
        config: &SchedulerConfig,
        override_val: Option<usize>,
    ) -> Option<usize> {
        use crate::backend::cuda::tensor::DeviceContext;

        const DEFAULT_MAX_SEQ: usize = 4096;

        if let Some(val) = override_val {
            info!("KV cache: using explicit --max-seq-len={}", val);
            return Some(val);
        }

        let (free_bytes, total_bytes) = match DeviceContext::gpu_memory_info() {
            Ok(info) => info,
            Err(e) => {
                info!(
                    "KV cache: could not query GPU memory ({}), using default max_seq_len={}",
                    e, DEFAULT_MAX_SEQ
                );
                return None;
            }
        };

        let reserved = config.gpu_reserved_bytes;
        let min_seq = config.min_seq_len;
        let available = free_bytes.saturating_sub(reserved);
        let bytes_per_token = model.kv_cache_bytes_per_token();
        let total_kv_budget = available;
        let per_slot_budget = total_kv_budget / config.max_slots.max(1);
        let affordable_seq_len = per_slot_budget / bytes_per_token.max(1);
        let effective = affordable_seq_len.clamp(min_seq, DEFAULT_MAX_SEQ);

        info!(
            "KV cache auto-sizing: gpu_free={:.1} GB, gpu_total={:.1} GB, \
             reserved={:.1} GB, bytes_per_token={}, num_slots={}, \
             affordable_seq_len={}, effective_max_seq_len={}",
            free_bytes as f64 / 1e9,
            total_bytes as f64 / 1e9,
            reserved as f64 / 1e9,
            bytes_per_token,
            config.max_slots,
            affordable_seq_len,
            effective,
        );

        if affordable_seq_len < min_seq {
            error!(
                "KV cache: only {} tokens affordable per slot (need at least {}). \
                 Reduce --num-slots or free GPU memory.",
                affordable_seq_len, min_seq,
            );
        }

        Some(effective)
    }

    /// Insert a completed request's prompt into the global
    /// [`RadixCache`] prefix observer.
    ///
    /// Each block-sized chunk of `prompt_tokens` gets a fresh synthetic
    /// [`BlockId`] minted from `next_prefix_block_id`. Trailing partial
    /// blocks are dropped per `RadixCache::insert` semantics. The ref
    /// counts for all newly-inserted blocks start at zero so the next
    /// lookup will be the only holder, and the scheduler's lookup path
    /// releases immediately — no permanent ref accumulates.
    ///
    /// Called from `cleanup()` whenever a finished request has a
    /// publishable cached prompt. M1 does not wire any eviction; the
    /// radix grows monotonically for the life of the process. M2 will
    /// attach eviction to paged-pool free events.
    pub(super) fn publish_to_prefix_cache(&mut self, prompt_tokens: &[u32]) {
        let block_size = self.prefix_cache.block_size();
        let num_blocks = prompt_tokens.len() / block_size;
        if num_blocks == 0 {
            return;
        }
        let mut blocks: Vec<BlockId> = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(BlockId(self.next_prefix_block_id));
            self.next_prefix_block_id = self.next_prefix_block_id.wrapping_add(1);
        }
        let inserted = self.prefix_cache.insert(prompt_tokens, &blocks);
        if inserted != num_blocks * block_size {
            // `insert` returns the number of tokens actually stored; a
            // mismatch here would only happen if `blocks.len()` is
            // smaller than `num_blocks`, which we just constructed, so
            // this is a hard invariant violation. Log at warn so the
            // scheduler loop does not panic.
            warn!(
                "prefix_cache.insert: expected {} tokens, got {} (num_blocks={}, tokens={})",
                num_blocks * block_size,
                inserted,
                num_blocks,
                prompt_tokens.len(),
            );
        }
    }

    pub(super) fn prefill_chunk_size(&self) -> usize {
        let signals = SchedulerSignals::queue_state(
            self.waiting.len(),
            self.active
                .iter()
                .filter(|req| matches!(req.phase, Phase::Decoding))
                .count(),
        );
        DecodeAwareChunking {
            decode_active_chunk: self.config.decode_active_prefill_cap,
            idle_chunk: self.config.prefill_chunk_size,
        }
        .next_chunk_size(InferenceMode::Prefill, signals)
        .max(1)
        .min(self.config.prefill_chunk_size)
    }

    /// Pre-capture CUDA Graphs for batched decode at common batch sizes.
    ///
    /// Uses SGLang-style batch size schedule: 1, 2, 4, 8, 12, 16, 24, 32, 40, ...
    /// up to min(num_slots, 256). This covers the most common concurrent request
    /// counts without capturing every single size.
    pub(super) fn warmup_cuda_graphs(&mut self) {
        let num_slots = self.states.len();
        if !self.paged_kv_pool.is_active() {
            return;
        }

        let max_bs = num_slots.min(256);
        let warmup_sizes = Self::cuda_graph_batch_sizes(max_bs);

        info!(
            "Warming up CUDA Graphs for {} batch sizes (max {})...",
            warmup_sizes.len(),
            max_bs,
        );
        let t0 = std::time::Instant::now();

        for slot in 0..max_bs {
            if let Err(e) = self.paged_kv_pool.alloc_tokens(slot, 1) {
                error!("Warmup: pool alloc for slot {} failed: {}", slot, e);
                return;
            }
        }

        // Lazy-init decode context before warmup.
        if self.decode_bufs.is_none() {
            match self
                .model
                .create_decode_context(self.states.len(), &self.paged_kv_pool)
            {
                Ok(ctx) => self.decode_bufs = Some(ctx),
                Err(e) => {
                    error!("Warmup: failed to create decode context: {}", e);
                    return;
                }
            }
        }

        let dummy_tokens: Vec<u32> = vec![0; max_bs];
        let slot_indices: Vec<usize> = (0..max_bs).collect();
        let mut captured = 0;
        for &bs in &warmup_sizes {
            let tokens = &dummy_tokens[..bs];
            let si = &slot_indices[..bs];
            let decode_ctx = self
                .decode_bufs
                .as_mut()
                .expect("invariant: decode_bufs initialized in warmup block above");

            // Pre-decode: scheduler-level work via DecodeContextOps.
            {
                use crate::model::DecodeContextOps;
                let ctx = self.model.device_context();
                decode_ctx.set_batch_size(bs);
                if let Err(e) = decode_ctx.upload_token_ids(ctx, tokens) {
                    info!(
                        "Warmup: upload_token_ids for B={} failed ({}), skipping",
                        bs, e
                    );
                    break;
                }
                match decode_ctx.update_metadata(ctx, &self.paged_kv_pool, si) {
                    Ok(reallocated) => {
                        if reallocated {
                            decode_ctx.invalidate_graph_cache(bs);
                        }
                    }
                    Err(e) => {
                        info!(
                            "Warmup: update_metadata for B={} failed ({}), skipping",
                            bs, e
                        );
                        break;
                    }
                }
                if let Err(e) = decode_ctx.plan_attention(
                    ctx,
                    bs,
                    self.model.num_q_heads(),
                    self.model.num_kv_heads(),
                    1,
                    self.model.head_dim(),
                    self.paged_kv_pool.format,
                ) {
                    info!(
                        "Warmup: plan_attention for B={} failed ({}), skipping",
                        bs, e
                    );
                    break;
                }
            }

            if let Err(e) = self.model.forward_decode_batch(
                tokens,
                &mut self.states,
                si,
                Some(&mut self.paged_kv_pool),
                decode_ctx,
                false,
            ) {
                info!(
                    "Warmup: graph capture for B={} failed ({}), skipping larger sizes",
                    bs, e
                );
                break;
            }
            let _ = self.model.device_context().sync();
            captured += 1;
        }

        for slot in 0..max_bs {
            self.paged_kv_pool.free_slot(slot);
            let _ = self.states[slot].reset();
        }

        info!(
            "CUDA Graph warmup done in {:.0}ms ({} batch sizes captured, max {})",
            t0.elapsed().as_secs_f64() * 1e3,
            captured,
            warmup_sizes.last().copied().unwrap_or(0),
        );
    }

    /// Generate batch size schedule for CUDA Graph warmup.
    /// Pattern: 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 96, 128, ..., max_bs
    fn cuda_graph_batch_sizes(max_bs: usize) -> Vec<usize> {
        let mut sizes = Vec::new();
        // Small sizes: 1, 2, 4, 8
        for &bs in &[1, 2, 4, 8] {
            if bs <= max_bs {
                sizes.push(bs);
            }
        }
        // From 12 to 32, step by 4 (covers common concurrency levels)
        let mut bs = 12;
        while bs <= 32.min(max_bs) {
            sizes.push(bs);
            bs += 4;
        }
        // From 48 onward, step by 16
        bs = 48;
        while bs <= max_bs {
            sizes.push(bs);
            bs += 16;
        }
        // Ensure max_bs itself is included
        if sizes.last() != Some(&max_bs) && max_bs > 8 {
            sizes.push(max_bs);
        }
        sizes
    }
}
