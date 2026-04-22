use std::collections::HashMap;

use super::{
    ActiveRequest, Arc, AtomicUsize, GenerationState, IncomingRequest, ModelForward, PagedKVPool,
    Phase, Result, SchedulerConfig, SchedulerHandle, SeedableRng, StdRng, Tokenizer, VecDeque,
    error, info, mpsc, warn,
};
use crate::kv_tier::transport::DiskStore;
use crate::kv_tier::{
    BlockLocation, ClusterSharedBackend, CoordinatorQueueStats, ReadmissionBlock, ReadmissionKey,
    ReadmissionPlan, ReadmissionSource, TieredKvPolicy,
};
use crate::prefix_cache::{BlockId, BlockMetadata, BlockMetadataUpdate, RadixCache};
use crate::scheduler::policy::{SchedulerSignals, SessionBiasedLru};
use crate::types::{BlockFingerprint, KvContentContext};

/// Block size (in tokens) for the global `RadixCache` prefix observer.
/// Chosen to match the M0.3 target paged-pool page size so that when
/// M2 dual residency wires the radix directly onto the pool, the block
/// boundaries already agree.
pub(super) const PREFIX_CACHE_BLOCK_SIZE: usize = 16;

/// Contiguous KV working buffer per slot (tokens). Only prefill uses it;
/// decode writes directly to the paged pool via `decode_prep_paged`.
/// Prefill chunk size is capped to this value to prevent buffer overflow.
pub(super) const CONTIGUOUS_KV_TOKENS: usize = 512;

// Prefix-cache and T1 watermark / keepalive tunables live on
// `crate::scheduler::types::SchedulerConfig`. Per the project env-var
// policy (`docs/environment.md` §0), these are **not** env-driven —
// callers assign directly to `SchedulerConfig` fields.

fn prefix_cache_retain_hard_cap_pages(total_pages: usize, cap_fraction: f64) -> usize {
    (total_pages as f64 * cap_fraction) as usize
}

fn can_publish_prefix_pages(
    retained_pages: usize,
    total_pages: usize,
    new_pages: usize,
    cap_fraction: f64,
) -> bool {
    retained_pages.saturating_add(new_pages)
        <= prefix_cache_retain_hard_cap_pages(total_pages, cap_fraction)
}

pub(super) struct PendingDecode {
    pub decode_indices: Vec<usize>,
    pub slot_indices: Vec<usize>,
    /// True only when `sample_batch_greedy_launch` actually fired the argmax kernel.
    pub greedy_launched: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) struct StoreDedupKey {
    pub fingerprint: BlockFingerprint,
    pub target: crate::kv_tier::StoreTarget,
}

/// CUDA-backed scheduler state and initialization.
pub struct Scheduler<M: ModelForward> {
    pub(super) config: SchedulerConfig,
    pub(super) metrics: crate::metrics::ServerMetrics,
    pub(super) model: M,
    pub(super) tokenizer: Tokenizer,
    /// Stable within one engine instance; real weight checksum upgrade is M5-era work.
    pub(super) model_fingerprint: Vec<u8>,
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
    pub(super) disk_store: Arc<DiskStore>,
    pub(super) cluster_shared_backend: Option<ClusterSharedBackend>,
    pub(super) tier_policy: TieredKvPolicy,
    pub(super) host_pinned_pool: crate::kv_tier::SharedHostPinnedPool,
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
    pub(super) coordinator_handle: crate::kv_tier::CoordinatorHandle,
    pub(super) coordinator_events: crossbeam_channel::Receiver<crate::kv_tier::CoordinatorEvent>,
    pub(super) coordinator_thread: Option<std::thread::JoinHandle<anyhow::Result<()>>>,
    pub(super) store_waiting:
        HashMap<crate::kv_tier::StoreTicket, Vec<(BlockId, crate::kv_tier::HostPinnedRegion)>>,
    pub(super) store_dedupe: HashMap<StoreDedupKey, crate::kv_tier::StoreTicket>,
    pub(super) store_ticket_keys: HashMap<crate::kv_tier::StoreTicket, StoreDedupKey>,
    pub(super) fetch_waiting: HashMap<crate::kv_tier::FetchTicket, Vec<(usize, u64)>>,
    pub(super) fetch_dedupe: HashMap<ReadmissionKey, crate::kv_tier::FetchTicket>,
    pub(super) fetch_ticket_keys: HashMap<crate::kv_tier::FetchTicket, ReadmissionKey>,
    pub(super) request_rx: mpsc::UnboundedReceiver<IncomingRequest>,
    pub(super) wakeup_rx: crossbeam_channel::Receiver<()>,
    pub(super) wakeup_live: bool,
    /// Shared waiting count with the handle (for backpressure decrement).
    pub(super) waiting_count: Arc<AtomicUsize>,
    pub(super) waiting: VecDeque<IncomingRequest>,
    pub(super) active: Vec<Option<ActiveRequest>>,
    pub(super) prefill_queue: VecDeque<usize>,
    pub(super) running_batch: VecDeque<usize>,
    pub(super) effective_max_seq_len: Option<usize>,
    pub(super) next_id: u64,
    pub(super) rng: StdRng,
    /// Paged KV cache pool shared across all slots (for batched decode).
    pub(super) paged_kv_pool: PagedKVPool,
    /// Pre-allocated buffers for batched decode (reused across steps).
    /// Typed via `M::DecodeContext` — no downcasting needed.
    pub(super) decode_bufs: Option<M::DecodeContext>,
    /// Lifetime stats.
    pub(super) total_completed: u64,
    pub(super) total_generated_tokens: u64,
    /// EMA step timing (microseconds) for /v1/stats profiling.
    pub(super) step_timing_decode_us: f64,
    pub(super) step_timing_emit_us: f64,
    pub(super) step_timing_prefill_us: f64,
    pub(super) step_timing_total_us: f64,
    /// Throttled GPU memory query — last poll time and peak high-water mark.
    pub(super) last_mem_query: std::time::Instant,
    pub(super) peak_mem_bytes: u64,
    /// Pending decode state for GPU/CPU overlap.
    pub(super) pending_decode: Option<PendingDecode>,
}

impl<M: ModelForward> Scheduler<M> {
    fn eviction_signals(&self) -> SchedulerSignals {
        SchedulerSignals::queue_state(self.waiting.len(), self.running_batch.len())
    }

    fn prefix_cache_watermarks_pages(&self) -> (usize, usize) {
        let total = self.paged_kv_pool.max_total_pages;
        let high = (total as f64 * self.config.prefix_cache_high_water) as usize;
        let low = (total as f64 * self.config.prefix_cache_low_water) as usize;
        (high, low)
    }

    pub fn session_fingerprints(&self, session_id: &str) -> Vec<BlockFingerprint> {
        self.prefix_cache.fingerprints_for_session(session_id)
    }

    pub fn read_block_payload(&self, fingerprint: BlockFingerprint) -> Option<Vec<u8>> {
        let block_id = self.prefix_cache.block_id_for_fingerprint(fingerprint)?;
        let pages = self.block_to_pages.get(&block_id)?;
        self.paged_kv_pool
            .copy_pages_to_host(self.model.device_context(), pages)
            .ok()
    }

    pub fn install_restored_kv(
        &mut self,
        payloads: &HashMap<BlockFingerprint, Vec<u8>>,
    ) -> Box<dyn FnMut(BlockFingerprint) -> Option<BlockId> + Send> {
        let pages_per_block = self
            .prefix_cache
            .block_size()
            .div_ceil(self.paged_kv_pool.page_size)
            .max(1);
        let mut prepared = HashMap::with_capacity(payloads.len());

        for (&fingerprint, payload) in payloads {
            let Ok(pages) = self.paged_kv_pool.alloc_detached_pages(pages_per_block) else {
                break;
            };
            if self
                .paged_kv_pool
                .copy_pages_from_host(self.model.device_context(), &pages, payload)
                .is_err()
            {
                continue;
            }

            let block_id = BlockId(
                *pages
                    .first()
                    .expect("detached restored block must allocate at least one page"),
            );
            self.block_to_pages.insert(block_id, pages);
            prepared.insert(fingerprint, block_id);
        }

        Box::new(move |fingerprint| prepared.remove(&fingerprint))
    }

    pub fn kv_format_tag(&self) -> u8 {
        self.paged_kv_pool.format.stable_tag().unwrap_or(0)
    }

    pub fn session_disk_store(&self) -> &DiskStore {
        self.disk_store.as_ref()
    }

    pub fn session_radix_cache(&self) -> &RadixCache {
        &self.prefix_cache
    }

    fn block_metadata(&self, block_id: BlockId) -> Option<BlockMetadata> {
        self.prefix_cache.block_metadata(block_id)
    }

    fn host_region_from_metadata(
        metadata: &BlockMetadata,
    ) -> Option<crate::kv_tier::HostPinnedRegion> {
        match metadata.location.as_ref() {
            Some(BlockLocation::HostPinned { offset }) => Some(crate::kv_tier::HostPinnedRegion {
                offset: *offset,
                len: metadata.byte_len as usize,
            }),
            _ => None,
        }
    }

    pub(super) fn build_staged_prefix_plan(
        &self,
        lookup: &crate::kv_tier::LookupOutcome,
    ) -> Option<ReadmissionPlan> {
        let mut blocks = Vec::new();
        for block in &lookup.blocks {
            if matches!(block.hit_kind, crate::kv_tier::HitKind::Miss) {
                break;
            }
            let block_id = block.block_id?;
            let metadata = self.block_metadata(block_id)?;
            let fingerprint = metadata.fingerprint?;
            let source = match block.hit_kind {
                crate::kv_tier::HitKind::ReadyOnGpu => None,
                crate::kv_tier::HitKind::StagingFromHost => Some(ReadmissionSource::HostPinned {
                    region: Self::host_region_from_metadata(&metadata)?,
                }),
                crate::kv_tier::HitKind::StagingFromDisk => match metadata.location.as_ref()? {
                    BlockLocation::Disk {
                        fingerprint,
                        payload_len,
                    } => Some(ReadmissionSource::Disk {
                        fingerprint: *fingerprint,
                        payload_len: *payload_len,
                    }),
                    BlockLocation::Remote { desc } => Some(ReadmissionSource::Remote {
                        desc: desc.clone(),
                        payload_len: u64::from(metadata.byte_len),
                    }),
                    _ => return None,
                },
                crate::kv_tier::HitKind::Miss => break,
            };
            blocks.push(ReadmissionBlock {
                block_id,
                fingerprint,
                source,
            });
        }
        if blocks.is_empty() {
            return None;
        }
        Some(ReadmissionPlan::new(lookup.matched_len, blocks))
    }

    fn host_pool_usage_fraction(&self) -> f64 {
        let Ok(pool) = self.host_pinned_pool.lock() else {
            return 1.0;
        };
        if pool.capacity_bytes() == 0 {
            return 0.0;
        }
        pool.reserved_bytes() as f64 / pool.capacity_bytes() as f64
    }

    fn host_pool_remaining_bytes(&self) -> usize {
        let Ok(pool) = self.host_pinned_pool.lock() else {
            return 0;
        };
        pool.remaining_bytes()
    }

    fn demote_block_budget(&self, candidates: &[BlockId]) -> usize {
        let Some(first_block) = candidates.first().copied() else {
            return 0;
        };
        let Some(metadata) = self.block_metadata(first_block) else {
            return 0;
        };
        let block_bytes = metadata.byte_len as usize;
        if block_bytes == 0 {
            return 0;
        }
        self.host_pool_remaining_bytes() / block_bytes
    }

    pub(super) fn release_host_region(&self, region: crate::kv_tier::HostPinnedRegion) {
        if let Ok(mut pool) = self.host_pinned_pool.lock() {
            pool.release(region);
        }
    }

    pub(super) fn clear_fetch_waiting_for_slot(&mut self, slot_idx: usize, request_id: u64) {
        let mut emptied = Vec::new();
        for (ticket, waiters) in &mut self.fetch_waiting {
            waiters.retain(|&(queued_slot, queued_id)| {
                !(queued_slot == slot_idx && queued_id == request_id)
            });
            if waiters.is_empty() {
                emptied.push(*ticket);
            }
        }
        for ticket in emptied {
            self.fetch_waiting.remove(&ticket);
            if let Some(key) = self.fetch_ticket_keys.remove(&ticket) {
                self.fetch_dedupe.remove(&key);
            }
            let _ = self.coordinator_handle.cancel_fetch(ticket);
        }
    }

    pub(super) fn coordinator_queue_stats(&self) -> CoordinatorQueueStats {
        self.coordinator_handle
            .stats()
            .with_fetch_waiters(self.fetch_waiting.values().map(std::vec::Vec::len).sum())
    }

    pub(super) fn attach_gpu_prefix_blocks(
        &mut self,
        slot_idx: usize,
        blocks: &[BlockId],
        token_count: usize,
    ) -> Result<()> {
        if blocks.is_empty() || token_count == 0 {
            return Ok(());
        }

        let mut pages = Vec::new();
        for &block_id in blocks {
            let block_pages = self.block_to_pages.get(&block_id).ok_or_else(|| {
                anyhow::anyhow!("missing page span for radix block {:?}", block_id)
            })?;
            pages.extend_from_slice(block_pages);
        }

        self.paged_kv_pool
            .attach_pages(slot_idx, &pages, token_count)
    }

    pub(super) fn release_attached_prefix_blocks(&mut self, blocks: &[BlockId]) {
        if !blocks.is_empty() {
            self.prefix_cache.release(blocks);
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
            // SGLang-compatible: size the KV pool after subtracting runtime
            // workspaces that are allocated after model load. Otherwise a
            // large pool can leave no contiguous room for decode / prefill
            // attention workspaces and fail mid-request.
            let runtime_workspace = model.scheduler_runtime_workspace_bytes(
                config.max_slots,
                effective_prefill_token_budget,
            );
            let budget_bytes = match crate::backend::cuda::tensor::DeviceContext::gpu_memory_info()
            {
                Ok((free, total)) => {
                    let headroom = ((total as f64) * (1.0 - config.mem_fraction_static)) as usize;
                    free.saturating_sub(
                        contiguous_cost
                            .saturating_add(headroom)
                            .saturating_add(runtime_workspace),
                    )
                }
                Err(_) => config.kv_pool_fallback_bytes,
            };

            info!(
                "TokenKVPool budget: {:.1} GB (contiguous={:.1} GB, runtime_workspace={:.1} GB, fraction={:.0}%)",
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
        let (coordinator, coordinator_handle, coordinator_events) =
            crate::kv_tier::Coordinator::new_with_backends(
                coordinator_queue_capacity,
                Some(Arc::clone(&disk_store)),
                cluster_shared_backend.clone(),
            );
        let coordinator_thread = Some(coordinator.spawn("infer-tiered-kv-coord"));
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
            store_waiting: HashMap::new(),
            store_dedupe: HashMap::new(),
            store_ticket_keys: HashMap::new(),
            fetch_waiting: HashMap::new(),
            fetch_dedupe: HashMap::new(),
            fetch_ticket_keys: HashMap::new(),
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
            total_completed: 0,
            total_generated_tokens: 0,
            step_timing_decode_us: 0.0,
            step_timing_emit_us: 0.0,
            step_timing_prefill_us: 0.0,
            step_timing_total_us: 0.0,
            last_mem_query: std::time::Instant::now(),
            peak_mem_bytes: 0,
            pending_decode: None,
        };

        let handle = SchedulerHandle::with_shared_waiting_count_and_wakeup(
            tx,
            wakeup_tx,
            model_id,
            max_waiting_requests,
            Arc::clone(&waiting_count),
        );
        debug_assert_eq!(handle.waiting_count(), 0);

        Ok((scheduler, handle))
    }

    pub(super) fn active_len(&self) -> usize {
        self.active.iter().filter(|req| req.is_some()).count()
    }

    pub(super) fn has_decode_work(&self) -> bool {
        self.running_batch.iter().any(|&slot_idx| {
            self.request(slot_idx).is_some_and(|req| {
                matches!(req.phase, Phase::Decoding) && !req.delta_tx.is_closed()
            })
        })
    }

    pub(super) fn is_fetch_wait_bound(&self) -> bool {
        self.active_len() > 0
            && self.pending_decode.is_none()
            && self.prefill_queue.is_empty()
            && !self.has_decode_work()
            && self
                .active
                .iter()
                .flatten()
                .all(|req| matches!(req.phase, Phase::WaitingFetch))
    }

    pub(super) fn request(&self, slot_idx: usize) -> Option<&ActiveRequest> {
        self.active.get(slot_idx)?.as_ref()
    }

    pub(super) fn request_mut(&mut self, slot_idx: usize) -> Option<&mut ActiveRequest> {
        self.active.get_mut(slot_idx)?.as_mut()
    }

    pub(super) fn queue_prefill(&mut self, slot_idx: usize) {
        if !self.prefill_queue.contains(&slot_idx) {
            self.prefill_queue.push_back(slot_idx);
        }
    }

    pub(super) fn dequeue_prefill(&mut self, slot_idx: usize) {
        self.prefill_queue.retain(|&queued| queued != slot_idx);
    }

    pub(super) fn queue_running(&mut self, slot_idx: usize) {
        if !self.running_batch.contains(&slot_idx) {
            self.running_batch.push_back(slot_idx);
        }
    }

    pub(super) fn dequeue_running(&mut self, slot_idx: usize) {
        self.running_batch.retain(|&queued| queued != slot_idx);
    }

    pub(super) fn pool_partial_tail_capacity(&self) -> usize {
        let page_size = self.paged_kv_pool.page_size.max(1);
        (0..self.states.len())
            .map(|slot_idx| {
                let used = self.paged_kv_pool.seq_len(slot_idx) % page_size;
                if used == 0 { 0 } else { page_size - used }
            })
            .sum()
    }

    pub(super) fn pool_free_pages(&self) -> usize {
        let page_size = self.paged_kv_pool.page_size.max(1);
        let free_page_tokens = self
            .paged_kv_pool
            .free_count()
            .saturating_sub(self.pool_partial_tail_capacity());
        free_page_tokens / page_size
    }

    pub(super) fn additional_pages_needed_for_seq_len(
        &self,
        seq_len: usize,
        additional_tokens: usize,
    ) -> usize {
        if additional_tokens == 0 {
            return 0;
        }
        let page_size = self.paged_kv_pool.page_size.max(1);
        let current_pages = seq_len.div_ceil(page_size);
        let future_pages = seq_len
            .saturating_add(additional_tokens)
            .div_ceil(page_size);
        future_pages.saturating_sub(current_pages)
    }

    pub(super) fn additional_pages_needed_for_slot(
        &self,
        slot_idx: usize,
        additional_tokens: usize,
    ) -> usize {
        self.additional_pages_needed_for_seq_len(
            self.paged_kv_pool.seq_len(slot_idx),
            additional_tokens,
        )
    }

    pub(super) fn finish_slot(&mut self, slot_idx: usize) {
        let request_id = self.request(slot_idx).map(|req| req.id);
        if let Some(req) = self.request_mut(slot_idx) {
            req.phase = Phase::Finished;
        }
        if let Some(request_id) = request_id {
            self.clear_fetch_waiting_for_slot(slot_idx, request_id);
        }
        self.dequeue_prefill(slot_idx);
        self.dequeue_running(slot_idx);
    }

    pub(super) fn move_to_decode(&mut self, slot_idx: usize) {
        self.dequeue_prefill(slot_idx);
        if let Some(req) = self.request_mut(slot_idx) {
            req.phase = Phase::Decoding;
        }
        self.queue_running(slot_idx);
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

        let headroom = ((total_bytes as f64) * (1.0 - config.mem_fraction_static)) as usize;
        let min_seq = config.min_seq_len;
        let available = free_bytes.saturating_sub(headroom);
        let bytes_per_token = model.kv_cache_bytes_per_token();
        let total_kv_budget = available;
        let per_slot_budget = total_kv_budget / config.max_slots.max(1);
        let affordable_seq_len = per_slot_budget / bytes_per_token.max(1);
        let effective = affordable_seq_len.clamp(min_seq, DEFAULT_MAX_SEQ);

        info!(
            "KV cache auto-sizing: gpu_free={:.1} GB, gpu_total={:.1} GB, \
             headroom={:.1} GB, bytes_per_token={}, num_slots={}, \
             affordable_seq_len={}, effective_max_seq_len={}",
            free_bytes as f64 / 1e9,
            total_bytes as f64 / 1e9,
            headroom as f64 / 1e9,
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
        session_id: Option<&crate::types::SessionId>,
    ) {
        let block_size = self.prefix_cache.block_size();
        // The slot's `seq_len` is the actual ground truth for how many tokens
        // are currently allocated in the paged pool. `prompt_tokens.len()`
        // is the snapshot the caller saved at request submission time and
        // can be LARGER than the slot's current footprint after a
        // recompute-style preemption: in that case the slot was rolled back
        // to a shorter prefix (or zero) but cleanup() still calls us with
        // the original prompt. Capping `num_blocks` to the slot's actual
        // page coverage prevents the
        // `paged_kv.rs:595 page_indices_for_token_range` index-OOB panic
        // that fires when we ask for pages past the slot's allocation.
        // See docs/experience/wins/2026-04-15-bench-longseq-int8-splits32.md
        // and 2026-04-15-bench-hbm-peak-throughput.md for the trigger
        // sequences.
        let slot_tokens_now = self.paged_kv_pool.seq_len(slot_idx);
        let publishable_tokens = prompt_tokens.len().min(slot_tokens_now);
        let num_blocks = publishable_tokens / block_size;
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
        let required_pages = block_pages.iter().map(std::vec::Vec::len).sum::<usize>();
        let retained_pages = self.paged_kv_pool.retained_count();
        let total_pages = self.paged_kv_pool.max_total_pages;
        let retain_cap_fraction = self.config.prefix_cache_retain_hard_cap;
        if !can_publish_prefix_pages(
            retained_pages,
            total_pages,
            required_pages,
            retain_cap_fraction,
        ) {
            info!(
                "prefix cache publish skipped for slot {}: retain hard cap hit \
                 (retained={}, new_pages={}, cap={}, total={})",
                slot_idx,
                retained_pages,
                required_pages,
                prefix_cache_retain_hard_cap_pages(total_pages, retain_cap_fraction),
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
        // M4 review A4: `stable_tag()` is now `Option<u8>`. If the
        // live pool format has no assigned tag (a future
        // TurboQuant bit-pair combination that shipped to the pool
        // but not to the disk format), publish silently with no
        // fingerprints — persistence is not available for that
        // format yet. Warn once per publish so operators notice.
        let kv_format_tag = if let Some(tag) = self.paged_kv_pool.format.stable_tag() {
            tag
        } else {
            warn!(
                "prefix_cache publish: live KV format has no stable_tag assignment; \
                 fingerprints skipped for slot {} (format = {:?})",
                slot_idx, self.paged_kv_pool.format,
            );
            // Zero = "unset"; persistence code refuses format 0
            // at load time, so this can never drive a cross-format
            // reload. Still stamp fingerprints because Tier C's
            // O(1) block_index and M4c's reconcile both want a
            // non-zero fingerprint on each published node.
            0
        };
        let mut parent_fingerprint: Option<BlockFingerprint> = None;
        let mut block_fingerprints: Vec<BlockFingerprint> = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let fp = BlockFingerprint::compute(
                KvContentContext {
                    model_fingerprint: &self.model_fingerprint,
                    kv_format_tag,
                    parent: parent_fingerprint,
                },
                &prompt_tokens[i * block_size..(i + 1) * block_size],
            );
            block_fingerprints.push(fp);
            parent_fingerprint = Some(fp);
        }

        // Slice prompt_tokens to the actually-publishable prefix so the
        // radix insert walks only the blocks we have fingerprints for.
        // Without this, when `slot_tokens_now < prompt_tokens.len()` the
        // insert path would try to traverse blocks past `num_blocks` and
        // mismatch its own internal block count.
        let publishable_prompt = &prompt_tokens[..required_tokens];
        let inserted = self.prefix_cache.insert_with_fingerprints(
            publishable_prompt,
            &blocks,
            &block_fingerprints,
        );
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
            .paged_kv_pool
            .storage_bytes_for_tokens(block_size)
            .min(u32::MAX as usize) as u32;
        let keepalive_deadline = session_id.as_ref().map(|_| {
            self.prefix_cache
                .logical_clock()
                .saturating_add(self.config.prefix_cache_keepalive_ticks)
        });
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
            let _ = self.prefix_cache.update_block_metadata(
                bid,
                BlockMetadataUpdate {
                    location: Some(BlockLocation::Gpu {
                        slot: slot_idx as u32,
                    }),
                    byte_len: Some(block_byte_len),
                    session_id: Some(session_id.cloned()),
                    soft_pin_until: Some(keepalive_deadline),
                    entry_state: None,
                },
            );
        }
    }

    /// Remove the transient "this free slot still owns a materialized prompt
    /// state" mapping for `slot_idx`.
    pub(super) fn clear_slot_prefix_ownership(&mut self, slot_idx: usize) {
        for bid in self.slot_owned_blocks[slot_idx].drain(..) {
            self.block_owner_slots.remove(&bid);
        }
    }

    fn demote_block_to_host(&mut self, block_id: BlockId) -> Result<usize> {
        let Some(metadata) = self.block_metadata(block_id) else {
            return Ok(0);
        };
        if !matches!(metadata.location, Some(BlockLocation::Gpu { .. })) {
            return Ok(0);
        }
        let Some(pages) = self.block_to_pages.get(&block_id).cloned() else {
            return Ok(0);
        };

        let payload = self
            .paged_kv_pool
            .copy_pages_to_host(self.model.device_context(), &pages)?;
        let region = {
            let mut pool = self.host_pinned_pool.lock()?;
            pool.reserve(payload.len()).ok_or_else(|| {
                anyhow::anyhow!(
                    "host pinned pool exhausted while demoting block {:?} ({} bytes)",
                    block_id,
                    payload.len()
                )
            })?
        };
        if let Err(err) = self.host_pinned_pool.write_region(region, &payload) {
            self.release_host_region(region);
            return Err(err);
        }

        self.block_to_pages.remove(&block_id);
        self.block_owner_slots.remove(&block_id);
        let _ = self.prefix_cache.update_block_metadata(
            block_id,
            BlockMetadataUpdate {
                location: Some(BlockLocation::HostPinned {
                    offset: region.offset,
                }),
                soft_pin_until: metadata.session_id.as_ref().map(|_| {
                    Some(
                        self.prefix_cache
                            .logical_clock()
                            .saturating_add(self.config.t1_host_pinned_keepalive_ticks),
                    )
                }),
                ..BlockMetadataUpdate::default()
            },
        );

        Ok(self.paged_kv_pool.release_pages(&pages).len())
    }

    fn delete_disk_block_if_present(&self, metadata: &BlockMetadata) {
        let Some(BlockLocation::Disk {
            fingerprint,
            payload_len,
        }) = metadata.location.as_ref()
        else {
            return;
        };
        let location = crate::kv_tier::transport::disk::DiskBlockLocation {
            path: match self.disk_store.block_path_for(*fingerprint) {
                Ok(path) => path,
                Err(_) => return,
            },
            payload_len: *payload_len,
            fingerprint: *fingerprint,
        };
        let _ = self.disk_store.delete_block(&location);
    }

    fn drop_cached_blocks(&mut self, blocks: &[BlockId]) -> usize {
        let mut reclaimed_pages = 0usize;
        for &block_id in blocks {
            let metadata = self.block_metadata(block_id);
            if let Some(pages) = self.block_to_pages.remove(&block_id) {
                reclaimed_pages += self.paged_kv_pool.release_pages(&pages).len();
            }
            if let Some(region) = metadata.as_ref().and_then(Self::host_region_from_metadata) {
                self.release_host_region(region);
            }
            self.block_owner_slots.remove(&block_id);
            if let Some(meta) = metadata.as_ref() {
                self.delete_disk_block_if_present(meta);
            }
        }
        reclaimed_pages
    }

    fn spill_host_blocks_if_pressured(&mut self) -> usize {
        let usage = self.host_pool_usage_fraction();
        if usage <= self.config.t1_host_pinned_high_water {
            return 0;
        }

        let coordinator_stats = self.coordinator_queue_stats();
        if coordinator_stats.store_backpressured() {
            return 0;
        }

        let (reserved_bytes, target_reserved_bytes) = match self.host_pinned_pool.lock() {
            Ok(pool) => (
                pool.reserved_bytes(),
                (pool.capacity_bytes() as f64 * self.config.t1_host_pinned_low_water) as usize,
            ),
            Err(_) => return 0,
        };
        let bytes_to_spill = reserved_bytes.saturating_sub(target_reserved_bytes);
        if bytes_to_spill == 0 {
            return 0;
        }

        let mut spilled_bytes = 0usize;
        let candidates = self.prefix_cache.select_blocks_with_policy(
            &SessionBiasedLru::default(),
            self.eviction_signals(),
            self.prefix_cache.cached_block_count(),
            Some(crate::kv_tier::Tier::HostPinned),
        );
        for block_id in candidates {
            if self.store_waiting.values().any(|pending| {
                pending
                    .iter()
                    .any(|(waiting_block, _)| *waiting_block == block_id)
            }) {
                continue;
            }
            let Some(metadata) = self.block_metadata(block_id) else {
                continue;
            };
            let Some(fingerprint) = metadata.fingerprint else {
                continue;
            };
            let Some(region) = Self::host_region_from_metadata(&metadata) else {
                continue;
            };
            let target = self.tier_policy.choose_store_target(
                &metadata,
                self.coordinator_handle.stats(),
                self.cluster_shared_backend.is_some(),
            );
            let store_key = StoreDedupKey {
                fingerprint,
                target,
            };
            if let Some(ticket) = self.store_dedupe.get(&store_key).copied() {
                let _ = self.prefix_cache.mark_block_store_pending(block_id);
                self.store_waiting
                    .entry(ticket)
                    .or_default()
                    .push((block_id, region));
                spilled_bytes = spilled_bytes.saturating_add(metadata.byte_len as usize);
                if spilled_bytes >= bytes_to_spill {
                    break;
                }
                continue;
            }
            let Some(ticket) =
                self.coordinator_handle
                    .submit_store(vec![crate::kv_tier::StoreRequest {
                        block_id,
                        fingerprint,
                        kv_format_tag: self.kv_format_tag(),
                        host_pool: self.host_pinned_pool.clone(),
                        host_region: region,
                        target,
                    }])
            else {
                break;
            };
            let _ = self.prefix_cache.mark_block_store_pending(block_id);
            self.store_waiting.insert(ticket, vec![(block_id, region)]);
            self.store_dedupe.insert(store_key, ticket);
            self.store_ticket_keys.insert(ticket, store_key);
            spilled_bytes = spilled_bytes.saturating_add(metadata.byte_len as usize);
            if spilled_bytes >= bytes_to_spill {
                break;
            }
        }
        spilled_bytes
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
        let candidates = self.prefix_cache.select_blocks_with_policy(
            &SessionBiasedLru::default(),
            self.eviction_signals(),
            blocks_to_evict,
            Some(crate::kv_tier::Tier::Gpu),
        );
        let mut reclaimed_pages = 0usize;
        let demote_budget = self.demote_block_budget(&candidates);
        for bid in candidates.iter().take(demote_budget).copied() {
            match self.demote_block_to_host(bid) {
                Ok(freed_now) => {
                    reclaimed_pages += freed_now;
                    if reclaimed_pages >= want_free {
                        break;
                    }
                }
                Err(err) => {
                    warn!("failed to demote block {:?} to host tier: {}", bid, err);
                }
            }
        }
        if reclaimed_pages < want_free {
            let remaining_pages = want_free.saturating_sub(reclaimed_pages);
            let blocks_to_drop = remaining_pages.div_ceil(pages_per_block.max(1)).max(1);
            let evicted = self.prefix_cache.evict_with_policy(
                &SessionBiasedLru::default(),
                self.eviction_signals(),
                blocks_to_drop,
                Some(crate::kv_tier::Tier::Gpu),
            );
            if !evicted.is_empty() {
                let dropped_pages = self.drop_cached_blocks(&evicted);
                reclaimed_pages += dropped_pages;
                if demote_budget == 0 {
                    warn!(
                        "prefix cache pressure fallback: host tier full, dropped {} GPU blocks ({} pages) to reclaim immediate T0 headroom",
                        evicted.len(),
                        dropped_pages
                    );
                } else {
                    warn!(
                        "prefix cache pressure fallback: dropped {} GPU blocks ({} pages) after host demotion reclaimed only {}/{} pages",
                        evicted.len(),
                        dropped_pages,
                        reclaimed_pages.saturating_sub(dropped_pages),
                        want_free
                    );
                }
            }
        }
        let _ = self.spill_host_blocks_if_pressured();
        if reclaimed_pages > 0 {
            info!(
                "prefix cache demotion: released {} pool pages back to free list \
                 (retained now {}, host usage {:.0}%)",
                reclaimed_pages,
                self.paged_kv_pool.retained_count(),
                self.host_pool_usage_fraction() * 100.0,
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
            let blocks_to_move = remaining
                .div_ceil((pages_per_block * self.paged_kv_pool.page_size).max(1))
                .max(1);
            let candidates = self.prefix_cache.select_blocks_with_policy(
                &SessionBiasedLru::default(),
                self.eviction_signals(),
                blocks_to_move,
                Some(crate::kv_tier::Tier::Gpu),
            );
            if candidates.is_empty() {
                break;
            }
            let before = reclaimed_pages;
            let demote_budget = self.demote_block_budget(&candidates);
            for bid in candidates.iter().take(demote_budget).copied() {
                match self.demote_block_to_host(bid) {
                    Ok(freed_now) => {
                        reclaimed_pages += freed_now;
                        reclaimed_tokens += freed_now * self.paged_kv_pool.page_size;
                    }
                    Err(err) => {
                        warn!(
                            "failed to demote block {:?} during alloc reclaim: {}",
                            bid, err
                        );
                    }
                }
            }
            let _ = self.spill_host_blocks_if_pressured();
            if reclaimed_pages == before {
                break;
            }
            remaining = shortage.saturating_sub(reclaimed_tokens);
        }

        if reclaimed_tokens < shortage {
            let blocks_to_drop = remaining
                .div_ceil((pages_per_block * self.paged_kv_pool.page_size).max(1))
                .max(1);
            let evicted = self.prefix_cache.evict_with_policy(
                &SessionBiasedLru::default(),
                self.eviction_signals(),
                blocks_to_drop,
                Some(crate::kv_tier::Tier::Gpu),
            );
            reclaimed_pages += self.drop_cached_blocks(&evicted);
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
        if count > 0 {
            self.paged_kv_pool
                .cow_tail_page_for_append(self.model.device_context(), slot)?;
        }
        match self.paged_kv_pool.alloc_tokens(slot, count) {
            Ok(indices) => Ok(indices),
            Err(first_err) => {
                let reclaimed = self.evict_prefix_cache_for_allocation(count);
                if reclaimed == 0 {
                    Err(first_err)
                } else {
                    self.paged_kv_pool
                        .alloc_tokens(slot, count)
                        .map_err(|retry_err| {
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
        // When the model writes prefill K/V directly to the paged pool, there
        // is no per-slot contiguous scratch to size the chunk against, so the
        // `CONTIGUOUS_KV_TOKENS` cap does not apply and the configured
        // `chunked_prefill_size` is the only upper bound.
        let contiguous_cap = if self.model.prefill_uses_paged_pool() {
            usize::MAX
        } else {
            CONTIGUOUS_KV_TOKENS
        };
        let out = self.config.chunked_prefill_size.max(1).min(contiguous_cap);
        log::debug!(
            "prefill_chunk_size: chunked_prefill_size={} cap={contiguous_cap} paged={} => {out}",
            self.config.chunked_prefill_size,
            self.model.prefill_uses_paged_pool(),
        );
        out
    }

    /// Pre-capture CUDA Graphs for batched decode at common batch sizes.
    ///
    /// Uses SGLang-style batch size schedule: 1, 2, 4, 8, 12, 16, 24, 32, 40, ...
    /// up to min(num_slots, 256). This covers the most common concurrent request
    /// counts without capturing every single size.
    ///
    /// Two-pass warmup:
    /// 1. Pass 1 drives forward_decode_batch per batch size, which populates the
    ///    cublasLt heuristic algo cache for every shape. In graph-capture mode
    ///    it also records a graph per batch size.
    /// 2. `autotune_all_cached_gemms_cuda` benchmarks all heuristic candidates and
    ///    replaces each shape's algo with the measured-fastest one.
    /// 3. Pass 2 (graph-capture mode only) re-captures graphs with the autotuned
    ///    algorithms. Eager decode (e.g. LoRA) skips pass 2 since no graphs
    ///    were cached.
    pub(super) fn warmup_cuda_graphs(&mut self) {
        let num_slots = self.states.len();
        if !self.paged_kv_pool.is_active() {
            return;
        }

        let graph_capture_enabled = self.model.supports_cuda_graph_decode();
        let max_bs = num_slots.min(256);
        let warmup_sizes = Self::cuda_graph_batch_sizes(max_bs);

        if graph_capture_enabled {
            info!(
                "Warming up CUDA Graphs for {} batch sizes (max {})...",
                warmup_sizes.len(),
                max_bs,
            );
        } else {
            info!(
                "Graph capture disabled (eager decode, e.g. LoRA); running \
                 eager warmup + cublasLt autotune for {} batch sizes (max {})...",
                warmup_sizes.len(),
                max_bs,
            );
        }
        let t0 = std::time::Instant::now();

        // Track how many slots we actually allocated so any early exit below
        // still frees them in the cleanup loop. Previously, a failing
        // `alloc_tokens` or `create_decode_context` would `return` with slots
        // still holding warmup tokens — `free_slots()` would then consider
        // them free while the pool still had dirty state, and the first real
        // request could inherit stale paged-KV entries.
        let mut allocated: usize = 0;
        let mut warmed: usize = 0;
        debug_assert!(
            self.paged_kv_pool.page_size > 0,
            "paged KV pool page size must be non-zero"
        );

        'warmup: {
            for slot in 0..max_bs {
                if let Err(e) = self.paged_kv_pool.alloc_tokens(slot, 1) {
                    error!("Warmup: pool alloc for slot {} failed: {}", slot, e);
                    break 'warmup;
                }
                allocated = slot + 1;
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
                        break 'warmup;
                    }
                }
            }

            let dummy_tokens: Vec<u32> = vec![0; max_bs];
            let slot_indices: Vec<usize> = (0..max_bs).collect();

            // Pass 1: drive forward for each warmup batch size. Populates the
            // cublasLt heuristic algo cache for all GEMM shapes used by decode.
            // In graph-capture mode, also captures a graph per batch size.
            warmed = self.warmup_graphs_pass(&warmup_sizes, &dummy_tokens, &slot_indices);

            // Autotune: benchmark all heuristic candidates, replace with measured best.
            // Runs regardless of graph mode so eager LoRA decode lands on the same
            // tuned algorithms as graph-mode decode.
            if warmed > 0 {
                info!("Autotuning cublasLt GEMM algorithms ({} shapes)...", warmed);
                let t_at = std::time::Instant::now();
                unsafe {
                    cuda_kernels::ffi::autotune_all_cached_gemms_cuda(
                        self.model.device_context().stream.cu_stream(),
                    );
                }
                info!(
                    "cublasLt autotune done in {:.0}ms",
                    t_at.elapsed().as_secs_f64() * 1e3,
                );

                if graph_capture_enabled {
                    // Invalidate graphs captured with heuristic algos.
                    {
                        use crate::model::DecodeContextOps;
                        let decode_ctx = self
                            .decode_bufs
                            .as_mut()
                            .expect("invariant: decode_bufs initialized above");
                        for &bs in &warmup_sizes[..warmed] {
                            decode_ctx.invalidate_graph_cache(bs);
                        }
                    }

                    // Pass 2: re-capture with autotuned algorithms.
                    let recaptured =
                        self.warmup_graphs_pass(&warmup_sizes, &dummy_tokens, &slot_indices);
                    info!(
                        "Re-captured {} graphs with autotuned GEMM algorithms",
                        recaptured,
                    );
                }
            }
        }

        // Always reached: frees any slots the warmup body allocated, whether
        // the body ran to completion or bailed on an error above.
        for slot in 0..allocated {
            self.paged_kv_pool.free_slot(slot);
            let _ = self.states[slot].reset();
        }

        let mode = if graph_capture_enabled {
            "CUDA Graph warmup"
        } else {
            "Eager warmup + cublasLt autotune"
        };
        info!(
            "{} done in {:.0}ms ({} batch sizes, max {})",
            mode,
            t0.elapsed().as_secs_f64() * 1e3,
            warmed,
            warmup_sizes.last().copied().unwrap_or(0),
        );
    }

    /// Single pass of graph warmup: set up metadata and forward for each batch size.
    fn warmup_graphs_pass(
        &mut self,
        warmup_sizes: &[usize],
        dummy_tokens: &[u32],
        slot_indices: &[usize],
    ) -> usize {
        let mut captured = 0;
        for &bs in warmup_sizes {
            let tokens = &dummy_tokens[..bs];
            let si = &slot_indices[..bs];
            let page_size = self.paged_kv_pool.page_size;
            let decode_ctx = self
                .decode_bufs
                .as_mut()
                .expect("invariant: decode_bufs initialized in warmup block above");

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
                    page_size,
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
        captured
    }

    /// Generate batch size schedule for CUDA Graph warmup.
    ///
    /// Warm up EVERY batch size from 1..=min(max_bs, 64). This eliminates
    /// graph-miss eager fallbacks when the batch composition changes during
    /// request transitions, which was the primary source of p99 ITL spikes
    /// (100-150ms outliers at B=16).
    ///
    /// Beyond 64 we use a sparse schedule (step by 16) since the marginal
    /// difference between B=65 and B=64 graphs is negligible.
    fn cuda_graph_batch_sizes(max_bs: usize) -> Vec<usize> {
        let mut sizes = Vec::new();
        // Dense: every size from 1 to min(64, max_bs)
        let dense_limit = 64.min(max_bs);
        for bs in 1..=dense_limit {
            sizes.push(bs);
        }
        // Sparse: from 80 onward, step by 16
        let mut bs = 80;
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

impl<M: ModelForward> Drop for Scheduler<M> {
    fn drop(&mut self) {
        // Non-blocking shutdown hint. If the command channel is full we
        // still force-disconnect below; the coordinator's `rx.recv_timeout`
        // will then observe Disconnected and return from `run_once`.
        self.coordinator_handle.try_send_shutdown();
        // Force-disconnect both sides of the coordinator by swapping our
        // handle and events receiver for dummy channels we immediately
        // drop. Without this, `thread.join()` can deadlock: a blocking
        // `send(Shutdown)` on a full command channel, or a coordinator
        // that is itself stuck on `self.events.send(...)` because the
        // scheduler stopped draining events before reaching Drop.
        //
        // Dropping `_old_handle` here is the last `CoordinatorCommand`
        // sender (the scheduler was the only owner), so the command
        // channel disconnects. Dropping `_old_events` kills the
        // coordinator's event path on its next send. Either one is
        // sufficient to unwedge `run_once`; we do both for safety.
        let (dummy_coord, dummy_handle, dummy_events) = crate::kv_tier::Coordinator::new(1);
        drop(dummy_coord);
        let old_handle = std::mem::replace(&mut self.coordinator_handle, dummy_handle);
        let old_events = std::mem::replace(&mut self.coordinator_events, dummy_events);
        drop(old_handle);
        drop(old_events);
        if let Some(thread) = self.coordinator_thread.take() {
            match thread.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => warn!("Coordinator thread shutdown failed: {}", err),
                Err(_) => warn!("Coordinator thread panicked during shutdown"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{can_publish_prefix_pages, prefix_cache_retain_hard_cap_pages};
    use crate::kv_tier::{CoordinatorQueueStats, QueueControlStats};

    const HARD_CAP: f64 = 0.90;

    #[test]
    fn retain_hard_cap_is_ninety_percent_of_pool() {
        assert_eq!(prefix_cache_retain_hard_cap_pages(100, HARD_CAP), 90);
        assert_eq!(prefix_cache_retain_hard_cap_pages(16, HARD_CAP), 14);
    }

    #[test]
    fn publish_is_denied_once_new_pages_cross_hard_cap() {
        assert!(can_publish_prefix_pages(70, 100, 20, HARD_CAP));
        assert!(can_publish_prefix_pages(80, 100, 10, HARD_CAP));
        assert!(!can_publish_prefix_pages(81, 100, 10, HARD_CAP));
        assert!(!can_publish_prefix_pages(90, 100, 1, HARD_CAP));
    }

    #[test]
    fn coordinator_queue_stats_reserve_submit_headroom() {
        let stats = CoordinatorQueueStats {
            capacity: 16,
            total_queued: 14,
            total_in_flight: 0,
            plan: QueueControlStats {
                capacity: 16,
                queued: 0,
                in_flight: 0,
                submitted: 0,
                completed: 0,
                failed: 0,
                cancelled: 0,
                rejected: 0,
            },
            fetch: QueueControlStats {
                capacity: 16,
                queued: 14,
                in_flight: 0,
                submitted: 0,
                completed: 0,
                failed: 0,
                cancelled: 0,
                rejected: 0,
            },
            store: QueueControlStats {
                capacity: 16,
                queued: 14,
                in_flight: 0,
                submitted: 0,
                completed: 0,
                failed: 0,
                cancelled: 0,
                rejected: 0,
            },
            fetch_waiters: 19,
        };
        assert!(!stats.fetch_backpressured());
        assert!(!stats.store_backpressured());

        let saturated = CoordinatorQueueStats {
            fetch: QueueControlStats {
                queued: 15,
                ..stats.fetch
            },
            store: QueueControlStats {
                queued: 15,
                ..stats.store
            },
            fetch_waiters: 23,
            total_queued: 15,
            ..stats
        };
        assert!(saturated.fetch_backpressured());
        assert!(saturated.store_backpressured());
    }

    #[test]
    fn coordinator_queue_stats_ignore_zero_capacity() {
        let stats = CoordinatorQueueStats {
            capacity: 0,
            total_queued: 8,
            total_in_flight: 0,
            plan: QueueControlStats {
                capacity: 0,
                queued: 0,
                in_flight: 0,
                submitted: 0,
                completed: 0,
                failed: 0,
                cancelled: 0,
                rejected: 0,
            },
            fetch: QueueControlStats {
                capacity: 0,
                queued: 8,
                in_flight: 0,
                submitted: 0,
                completed: 0,
                failed: 0,
                cancelled: 0,
                rejected: 0,
            },
            store: QueueControlStats {
                capacity: 0,
                queued: 8,
                in_flight: 0,
                submitted: 0,
                completed: 0,
                failed: 0,
                cancelled: 0,
                rejected: 0,
            },
            fetch_waiters: 13,
        };
        assert!(!stats.fetch_backpressured());
        assert!(!stats.store_backpressured());
    }
}
