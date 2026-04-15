use std::collections::HashMap;

use super::*;
use crate::kv_tier::BlockLocation;
use crate::prefix_cache::{BlockId, RadixCache};
use crate::scheduler::policy::{
    ChunkingPolicy, DecodeAwareChunking, SchedulerSignals, SessionBiasedLru,
};
use crate::types::InferenceMode;

/// Block size (in tokens) for the global `RadixCache` prefix observer.
/// Chosen to match the M0.3 target paged-pool page size so that when
/// M2 dual residency wires the radix directly onto the pool, the block
/// boundaries already agree.
pub(super) const PREFIX_CACHE_BLOCK_SIZE: usize = 16;

/// High-water mark for pool pages held by the radix cache, as a
/// fraction of `max_total_tokens`. When `retained_count / total`
/// exceeds this, `cleanup()` triggers LRU eviction down to
/// [`PREFIX_CACHE_EVICT_LOW_WATER`].
///
/// Rationale: at 75% retained the next fresh admission is likely to
/// hit pool OOM on `alloc_tokens`. Keeping headroom above the free
/// list avoids having to retry alloc with eviction in the hot path.
pub(super) const PREFIX_CACHE_HIGH_WATER: f64 = 0.75;

/// Low-water mark for the same eviction loop: evict down to this
/// fraction of `max_total_tokens`. The gap between high and low
/// prevents thrash (evict-then-re-insert on every cleanup).
pub(super) const PREFIX_CACHE_LOW_WATER: f64 = 0.50;
/// Hard cap for radix-held retained pages. Above this threshold, new
/// completed prompts are not published into the retained T0 prefix cache so
/// fresh admissions cannot be starved by pinned-but-cold pages.
pub(super) const PREFIX_CACHE_RETAIN_HARD_CAP: f64 = 0.90;
/// Number of logical radix-clock ticks to soft-pin freshly published session
/// prefixes before they become evictable again.
pub(super) const PREFIX_CACHE_KEEPALIVE_TICKS: u64 = 64;

fn prefix_cache_retain_hard_cap_pages(total_pages: usize) -> usize {
    (total_pages as f64 * PREFIX_CACHE_RETAIN_HARD_CAP) as usize
}

fn can_publish_prefix_pages(retained_pages: usize, total_pages: usize, new_pages: usize) -> bool {
    retained_pages.saturating_add(new_pages) <= prefix_cache_retain_hard_cap_pages(total_pages)
}

/// CUDA-backed scheduler state and initialization.
pub struct Scheduler<M: ModelForward> {
    pub(super) config: SchedulerConfig,
    pub(super) model: M,
    pub(super) tokenizer: Tokenizer,
    /// Per-slot states (KV caches, decode buffers). Stored separately from
    /// slot metadata so we can pass `&mut [M::State]` to batched decode.
    pub(super) states: Vec<M::State>,
    /// Number of prompt tokens still materialized in each slot's contiguous
    /// state. This is the scheduler's only slot-local prefix-reuse metadata
    /// after M2b removes `cached_prompts: Vec<Vec<u32>>`.
    ///
    /// A non-zero value means: if the slot is free and the global radix says
    /// an incoming request matches a prefix owned by this slot, `step_new()`
    /// may reuse the first `matched_len` tokens already present in the slot's
    /// contiguous state instead of restarting cold.
    pub(super) slot_materialized_prompt_lens: Vec<usize>,
    /// Global cross-slot prefix observer and (as of M2a) the authority
    /// on which pool pages must survive a slot's `free_slot` call.
    /// Owned by the single-writer scheduler thread, no lock needed.
    ///
    /// The `BlockId`s stored under each node are **real physical pool
    /// page indices** pulled from `paged_kv_pool.token_indices(slot_idx)`
    /// at publish time. The pool's `page_ref_count` tracks how many radix
    /// nodes reference each page, and `free_slot` leaves pinned pages in
    /// limbo instead of returning them to the free list. CUDA admission now
    /// consults this radix-owned state to pick a reusable free slot, but
    /// reuse is still limited to slots whose contiguous state remains
    /// materialized; cross-slot page aliasing is intentionally unsupported.
    pub(super) prefix_cache: RadixCache,
    /// Side map from `BlockId` → full contiguous page span for that
    /// block. The radix stores just the first page of each block
    /// (block id = `slot_pages[i * block_size]`), but the actual
    /// `block_size` pages belonging to that block can be arbitrary
    /// pool indices because the LIFO `free_slots` allocator produces
    /// non-contiguous ranges after a few alloc/free cycles. This map
    /// keeps the full span so eviction can release the right pages.
    ///
    /// Invariant: every `BlockId` inserted into `prefix_cache` has
    /// an entry here with exactly `prefix_cache.block_size()` pages,
    /// and every page in the value appears in `page_ref_count > 0`.
    /// Entries are removed when eviction releases the block.
    pub(super) block_to_pages: HashMap<BlockId, Vec<u32>>,
    /// Best-effort mapping from a radix block to the free slot whose
    /// contiguous state still materializes that prefix. This is intentionally
    /// separate from `prefix_cache`: the radix owns reusable bytes / page pins,
    /// while this map only tracks which free slot can safely reuse those bytes
    /// without cross-slot page aliasing.
    pub(super) block_owner_slots: HashMap<BlockId, usize>,
    /// Reverse index for `block_owner_slots`, keyed by slot.
    pub(super) slot_owned_blocks: Vec<Vec<BlockId>>,
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
    fn eviction_signals(&self) -> SchedulerSignals {
        SchedulerSignals::queue_state(
            self.waiting.len(),
            self.active
                .iter()
                .filter(|req| matches!(req.phase, Phase::Decoding))
                .count(),
        )
    }

    fn prefix_cache_watermarks_pages(&self) -> (usize, usize) {
        let total = self.paged_kv_pool.max_total_pages;
        let high = (total as f64 * PREFIX_CACHE_HIGH_WATER) as usize;
        let low = (total as f64 * PREFIX_CACHE_LOW_WATER) as usize;
        (high, low)
    }

    pub(super) fn refresh_session_keepalive(
        &mut self,
        blocks: &[BlockId],
        session_id: Option<crate::types::SessionId>,
    ) {
        let keepalive_deadline = session_id.as_ref().map(|_| {
            self.prefix_cache
                .logical_clock()
                .saturating_add(PREFIX_CACHE_KEEPALIVE_TICKS)
        });
        for &bid in blocks {
            let _ = self
                .prefix_cache
                .set_block_session_id(bid, session_id.clone());
            let _ = self
                .prefix_cache
                .set_block_soft_pin_until(bid, keepalive_deadline);
        }
    }

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
        let mut slot_materialized_prompt_lens = Vec::with_capacity(config.max_slots);
        let mut slot_owned_blocks = Vec::with_capacity(config.max_slots);
        for i in 0..config.max_slots {
            let mut state = model.create_state()?;
            if let Some(max_seq) = effective_max_seq_len {
                state.set_max_seq_len(max_seq);
            }
            state.set_kv_dtype(kv_cache_dtype);
            states.push(state);
            slot_materialized_prompt_lens.push(0);
            slot_owned_blocks.push(Vec::new());
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
            slot_materialized_prompt_lens,
            prefix_cache: RadixCache::new(PREFIX_CACHE_BLOCK_SIZE),
            block_to_pages: HashMap::new(),
            block_owner_slots: HashMap::new(),
            slot_owned_blocks,
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

    /// Fold a completed request's prompt into the global
    /// [`RadixCache`] prefix observer and pin the corresponding pool
    /// pages so they survive the subsequent `free_slot` call.
    ///
    /// The [`BlockId`]s stored in the radix are **real physical pool page
    /// indices** pulled from `paged_kv_pool.token_indices(slot_idx)`. For a prompt of
    /// `L` tokens and the radix's `block_size = B`, the first
    /// `num_blocks = L / B` full blocks are inserted under the page
    /// ids covering positions `[0, num_blocks * B)` — i.e. exactly the
    /// contiguous pool pages that hold those tokens' KV state. The
    /// trailing `L % B` tokens are dropped per `RadixCache::insert`
    /// semantics (partial blocks are never cached).
    ///
    /// After inserting, `paged_kv_pool.retain_pages` bumps the
    /// refcount on each page. Because the scheduler's `cleanup()`
    /// calls this method **before** `free_slot`, the pool's
    /// refcount-aware `free_slot` will leave these pages in limbo
    /// (out of any slot, out of the free list, still physically
    /// alive in HBM) instead of recycling them. This is the T0-only
    /// dual-residency data model that the current safe same-slot
    /// resurrection path consumes.
    ///
    /// Caller contract: `slot_idx` must currently own the pages in
    /// `paged_kv_pool.token_indices(slot_idx)` and the slot must not
    /// have been `free_slot`ed yet. `prompt_tokens.len()` must equal
    /// the number of tokens currently allocated to the slot (i.e.
    /// `paged_kv_pool.seq_len(slot_idx)`). Both are true at the
    /// `cleanup()` call site where this is invoked.
    pub(super) fn publish_to_prefix_cache(
        &mut self,
        slot_idx: usize,
        prompt_tokens: &[u32],
        session_id: Option<crate::types::SessionId>,
    ) {
        let block_size = self.prefix_cache.block_size();
        let num_blocks = prompt_tokens.len() / block_size;
        if num_blocks == 0 {
            return;
        }
        let required_tokens = num_blocks * block_size;
        let block_pages: Vec<Vec<u32>> = (0..num_blocks)
            .map(|block_i| {
                self.paged_kv_pool
                    .page_indices_for_token_range(slot_idx, block_i * block_size, block_size)
                    .to_vec()
            })
            .collect();
        let required_pages = block_pages.iter().map(|pages| pages.len()).sum::<usize>();
        let retained_pages = self.paged_kv_pool.retained_count();
        let total_pages = self.paged_kv_pool.max_total_pages;
        if !can_publish_prefix_pages(retained_pages, total_pages, required_pages) {
            info!(
                "prefix cache publish skipped for slot {}: retain hard cap hit \
                 (retained={}, new_pages={}, cap={}, total={})",
                slot_idx,
                retained_pages,
                required_pages,
                prefix_cache_retain_hard_cap_pages(total_pages),
                total_pages,
            );
            return;
        }

        let blocks: Vec<BlockId> = block_pages
            .iter()
            .map(|pages| {
                BlockId(
                    *pages
                        .first()
                        .expect("full radix block must map to at least one physical page"),
                )
            })
            .collect();

        let inserted = self.prefix_cache.insert(prompt_tokens, &blocks);
        if inserted != required_tokens {
            warn!(
                "prefix_cache.insert: expected {} tokens, got {} (slot={}, num_blocks={}, prompt={})",
                required_tokens,
                inserted,
                slot_idx,
                num_blocks,
                prompt_tokens.len(),
            );
            return;
        }

        // Pin every physical page that backs the inserted blocks.
        // The radix refs a "block" as a unit, and the *entire*
        // `block_size`-wide span must survive `free_slot` so the
        // reuse path can read the full KV state back out.
        let slot_pages: Vec<u32> = block_pages
            .iter()
            .flat_map(|pages| pages.iter().copied())
            .collect();
        self.paged_kv_pool.retain_pages(&slot_pages);
        self.slot_owned_blocks[slot_idx].clear();
        let block_byte_len = self
            .model
            .kv_cache_bytes_per_token()
            .saturating_mul(block_size)
            .min(u32::MAX as usize) as u32;
        for (block_i, &bid) in blocks.iter().enumerate() {
            let pages_for_block = block_pages[block_i].clone();
            // If a prior publish already registered this BlockId
            // (same pool page as first-of-block), the existing entry
            // is authoritative — don't clobber it. The `retain_pages`
            // call above bumped the refcount a second time, so the
            // first `release_pages` will leave the entry with
            // refcount > 0 and the eviction path will correctly
            // short-circuit on the second release.
            self.block_to_pages
                .entry(bid)
                .or_insert_with(|| pages_for_block);
            self.block_owner_slots.insert(bid, slot_idx);
            self.slot_owned_blocks[slot_idx].push(bid);
            let _ = self.prefix_cache.set_block_location(
                bid,
                BlockLocation::Gpu {
                    slot: slot_idx as u32,
                },
            );
            let _ = self.prefix_cache.set_block_byte_len(bid, block_byte_len);
        }
        self.refresh_session_keepalive(&blocks, session_id);
    }

    /// Remove the transient "this free slot still owns a materialized prompt
    /// state" mapping for `slot_idx`.
    pub(super) fn clear_slot_prefix_ownership(&mut self, slot_idx: usize) {
        for bid in self.slot_owned_blocks[slot_idx].drain(..) {
            self.block_owner_slots.remove(&bid);
        }
    }

    /// Release radix-held pool pages back to the free list once the
    /// pool's retained fraction exceeds the high-water mark.
    ///
    /// Policy: when `retained / total > PREFIX_CACHE_HIGH_WATER`,
    /// evict LRU radix blocks until `retained / total ≤
    /// PREFIX_CACHE_LOW_WATER`. The gap between high and low marks
    /// prevents thrash where every completed request immediately
    /// evicts the block it just inserted.
    ///
    /// Each evicted `BlockId` is looked up in `block_to_pages` and
    /// the full per-block page span is released via
    /// `paged_kv_pool.release_pages`. If the refcount hits zero the
    /// pages rejoin the pool's primary free list immediately; if
    /// another radix block also references them the refcount just
    /// decrements and the pages stay in limbo.
    ///
    /// Returns the number of pool pages actually reclaimed (0 when
    /// not under pressure). Called at the end of `cleanup()` so the
    /// eviction cost is amortised over request completions, not the
    /// admission hot path.
    pub(super) fn evict_prefix_cache_if_pressured(&mut self) -> usize {
        let total = self.paged_kv_pool.max_total_pages;
        if total == 0 {
            return 0;
        }
        let retained = self.paged_kv_pool.retained_count();
        let (high, target) = self.prefix_cache_watermarks_pages();
        if retained <= high {
            return 0;
        }
        let want_free = retained.saturating_sub(target);
        let pages_per_block = self
            .prefix_cache
            .block_size()
            .div_ceil(self.paged_kv_pool.page_size);
        let blocks_to_evict = want_free.div_ceil(pages_per_block.max(1));
        if blocks_to_evict == 0 {
            return 0;
        }
        let evicted = self.prefix_cache.evict_with_policy(
            &SessionBiasedLru::default(),
            self.eviction_signals(),
            blocks_to_evict,
        );
        let mut reclaimed_pages: usize = 0;
        for bid in evicted {
            if let Some(pages) = self.block_to_pages.remove(&bid) {
                self.block_owner_slots.remove(&bid);
                let freed_now = self.paged_kv_pool.release_pages(&pages);
                reclaimed_pages += freed_now.len();
            }
        }
        if reclaimed_pages > 0 {
            info!(
                "prefix cache eviction: released {} pool pages back to free list \
                 ({} evicted blocks; retained now {})",
                reclaimed_pages,
                blocks_to_evict,
                self.paged_kv_pool.retained_count(),
            );
        }
        reclaimed_pages
    }

    /// Best-effort synchronous reclamation used on the hot path when pool
    /// allocation fails. Unlike the watermark-based cleanup path, this may run
    /// below the usual high-water mark: the immediate goal is to recover enough
    /// prefix-cache pages to satisfy one allocation.
    pub(super) fn evict_prefix_cache_for_allocation(&mut self, required_tokens: usize) -> usize {
        let shortage = required_tokens.saturating_sub(self.paged_kv_pool.free_count());
        if shortage == 0 {
            return 0;
        }

        let pages_per_block = self
            .prefix_cache
            .block_size()
            .div_ceil(self.paged_kv_pool.page_size);
        let mut reclaimed_pages = 0usize;
        let mut reclaimed_tokens = 0usize;
        let mut remaining = shortage;
        while remaining > 0 {
            let blocks_to_evict = remaining
                .div_ceil((pages_per_block * self.paged_kv_pool.page_size).max(1))
                .max(1);
            let evicted = self.prefix_cache.evict_with_policy(
                &SessionBiasedLru::default(),
                self.eviction_signals(),
                blocks_to_evict,
            );
            if evicted.is_empty() {
                break;
            }
            for bid in evicted {
                if let Some(pages) = self.block_to_pages.remove(&bid) {
                    self.block_owner_slots.remove(&bid);
                    let freed_now = self.paged_kv_pool.release_pages(&pages);
                    reclaimed_pages += freed_now.len();
                    reclaimed_tokens += freed_now.len() * self.paged_kv_pool.page_size;
                }
            }
            remaining = shortage.saturating_sub(reclaimed_tokens);
        }

        if reclaimed_pages > 0 {
            info!(
                "prefix cache emergency eviction: reclaimed {} pool pages for allocation \
                 (required={}, free_now={})",
                reclaimed_pages,
                required_tokens,
                self.paged_kv_pool.free_count(),
            );
        }
        reclaimed_pages
    }

    /// Allocate pool pages, forcing prefix-cache eviction and retrying once on
    /// OOM. This is the M2b safety net for bursty admissions between cleanup
    /// passes.
    pub(super) fn alloc_pool_tokens_with_retry(
        &mut self,
        slot: usize,
        count: usize,
    ) -> Result<Vec<u32>> {
        match self.paged_kv_pool.alloc_tokens(slot, count) {
            Ok(indices) => Ok(indices),
            Err(first_err) => {
                let reclaimed = self.evict_prefix_cache_for_allocation(count);
                if reclaimed == 0 {
                    Err(first_err)
                } else {
                    self.paged_kv_pool.alloc_tokens(slot, count).map_err(|retry_err| {
                        anyhow::anyhow!(
                            "TokenKVPool alloc retry failed after reclaiming {reclaimed} pages: \
                             first error: {first_err}; retry error: {retry_err}"
                        )
                    })
                }
            }
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

#[cfg(test)]
mod tests {
    use super::{can_publish_prefix_pages, prefix_cache_retain_hard_cap_pages};

    #[test]
    fn retain_hard_cap_is_ninety_percent_of_pool() {
        assert_eq!(prefix_cache_retain_hard_cap_pages(100), 90);
        assert_eq!(prefix_cache_retain_hard_cap_pages(16), 14);
    }

    #[test]
    fn publish_is_denied_once_new_pages_cross_hard_cap() {
        assert!(can_publish_prefix_pages(70, 100, 20));
        assert!(can_publish_prefix_pages(80, 100, 10));
        assert!(!can_publish_prefix_pages(81, 100, 10));
        assert!(!can_publish_prefix_pages(90, 100, 1));
    }
}
