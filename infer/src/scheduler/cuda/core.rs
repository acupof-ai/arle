use std::collections::HashMap;

use super::budget::{
    coordinator_submit_headroom, partial_tail_capacity, prefix_cache_reclaim_goal_pages,
    waiting_admission_shortage_pages,
};
use super::policy::TieredKvPolicy;
use super::{
    ActiveRequest, Arc, AtomicUsize, IncomingRequest, ModelForward, PagedKVPool, Phase, Result,
    SchedulerConfig, StdRng, Tokenizer, VecDeque, error, info, mpsc, warn,
};
use crate::kv_tier::transport::DiskStore;
use crate::kv_tier::{
    BlockLocation, ClusterSharedBackend, CoordinatorQueueStats, ReadmissionBlock, ReadmissionKey,
    ReadmissionPlan, ReadmissionSource,
};
use crate::prefix_cache::{
    BlockId, BlockMetadata, BlockMetadataUpdate, BlockSelectionIntent, RadixCache,
};
use crate::scheduler::policy::{SchedulerSignals, SessionBiasedLru};
use crate::server_engine::FinishReason;
use crate::types::{BlockFingerprint, KvContentContext};

#[path = "core/helpers.rs"]
mod helpers;

#[path = "core/state_types.rs"]
mod state_types;

#[path = "core/emit_worker.rs"]
mod emit_worker;

#[path = "core/construction.rs"]
mod construction;

#[path = "core/warmup.rs"]
mod warmup;

// Re-export types and helpers that the rest of the `cuda` module references
// via `core::*`.
pub(in crate::scheduler::cuda) use emit_worker::{EmitCommand, EmitEvent, spawn_emit_worker};
pub(in crate::scheduler::cuda) use helpers::{
    CONTIGUOUS_KV_TOKENS, PREFIX_CACHE_BLOCK_SIZE, can_publish_prefix_pages,
    host_spill_target_bytes, is_full_sealed_prefix, prefix_cache_retain_hard_cap_pages,
    sealed_block_token_count,
};
pub(in crate::scheduler::cuda) use state_types::{
    PendingDecode, PendingMixedPrefill, PendingPrefill, PendingPrefillRow, PrefetchTicketState,
    SchedulerRuntimeStats, StoreDedupKey,
};
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
    emit_tx: crossbeam_channel::Sender<EmitCommand>,
    pub(super) emit_events: crossbeam_channel::Receiver<EmitEvent>,
    emit_thread: Option<std::thread::JoinHandle<()>>,
    pub(super) emit_gate_waiting: HashMap<u64, usize>,
    pub(super) store_waiting:
        HashMap<crate::kv_tier::StoreTicket, Vec<(BlockId, crate::kv_tier::HostPinnedRegion)>>,
    pub(super) store_dedupe: HashMap<StoreDedupKey, crate::kv_tier::StoreTicket>,
    pub(super) store_ticket_keys: HashMap<crate::kv_tier::StoreTicket, StoreDedupKey>,
    pub(super) store_ticket_started_at: HashMap<crate::kv_tier::StoreTicket, std::time::Instant>,
    pub(super) fetch_waiting: HashMap<crate::kv_tier::FetchTicket, Vec<(usize, u64)>>,
    pub(super) fetch_dedupe: HashMap<ReadmissionKey, crate::kv_tier::FetchTicket>,
    pub(super) fetch_ticket_keys: HashMap<crate::kv_tier::FetchTicket, ReadmissionKey>,
    pub(super) fetch_ticket_started_at: HashMap<crate::kv_tier::FetchTicket, std::time::Instant>,
    pub(super) prefetch_fetching: HashMap<crate::kv_tier::FetchTicket, PrefetchTicketState>,
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
    /// Pre-allocated buffers for batched prefill that may hold GPU resources
    /// across loop turns when async prefill overlap is enabled.
    pub(super) prefill_ctx: Option<M::PrefillContext>,
    /// Lifetime counters and local profiling state.
    pub(super) stats: SchedulerRuntimeStats,
    /// Pending decode state for GPU/CPU overlap.
    pub(super) pending_decode: Option<PendingDecode>,
    /// Pending prefill state for GPU/CPU overlap.
    pub(super) pending_prefill: Option<PendingPrefill>,
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

    fn waiting_admission_shortage_pages(&self) -> usize {
        waiting_admission_shortage_pages(
            self.pool_free_pages(),
            self.paged_kv_pool.page_size.max(1),
            self.active.iter().filter(|req| req.is_none()).count(),
            self.waiting.len(),
            self.waiting.iter().map(|incoming| {
                (
                    incoming.prompt_tokens.as_ref().map(std::vec::Vec::len),
                    incoming.max_tokens,
                )
            }),
        )
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

    fn block_id_for_pages(pages: &[u32]) -> BlockId {
        BlockId(
            *pages
                .first()
                .expect("full sealed block must map to at least one physical page"),
        )
    }

    fn sealed_block_byte_len(&self) -> u32 {
        self.paged_kv_pool
            .storage_bytes_for_tokens(self.prefix_cache.block_size())
            .min(u32::MAX as usize) as u32
    }

    fn block_keepalive_deadline(
        &self,
        session_id: Option<&crate::types::SessionId>,
        keepalive_ticks: u64,
    ) -> Option<u64> {
        session_id.map(|_| {
            self.prefix_cache
                .logical_clock()
                .saturating_add(keepalive_ticks)
        })
    }

    fn slot_sealed_block_pages(&self, slot_idx: usize, block_count: usize) -> Vec<Vec<u32>> {
        let block_size = self.prefix_cache.block_size();
        (0..block_count)
            .map(|block_i| {
                self.paged_kv_pool
                    .page_indices_for_token_range(slot_idx, block_i * block_size, block_size)
                    .to_vec()
            })
            .collect()
    }

    fn flattened_pages_for_blocks(&self, blocks: &[BlockId]) -> Result<Vec<u32>> {
        let mut pages = Vec::new();
        for &block_id in blocks {
            let block_pages = self.block_to_pages.get(&block_id).ok_or_else(|| {
                anyhow::anyhow!("missing page span for sealed radix block {:?}", block_id)
            })?;
            pages.extend_from_slice(block_pages);
        }
        Ok(pages)
    }

    pub(super) fn record_sealed_gpu_blocks<I>(
        &mut self,
        slot_idx: usize,
        blocks: I,
        session_id: Option<&crate::types::SessionId>,
        keepalive_ticks: u64,
        track_slot_owner: bool,
    ) where
        I: IntoIterator<Item = (BlockId, Vec<u32>)>,
    {
        let block_byte_len = self.sealed_block_byte_len();
        let keepalive_deadline = self.block_keepalive_deadline(session_id, keepalive_ticks);
        for (block_id, pages) in blocks {
            self.block_to_pages.entry(block_id).or_insert(pages);
            if track_slot_owner {
                self.block_owner_slots.insert(block_id, slot_idx);
                self.slot_owned_blocks[slot_idx].push(block_id);
            }
            let _ = self.prefix_cache.update_block_metadata(
                block_id,
                BlockMetadataUpdate {
                    location: Some(BlockLocation::Gpu {
                        slot: slot_idx as u32,
                    }),
                    byte_len: Some(block_byte_len),
                    session_id: Some(session_id.cloned()),
                    soft_pin_until: Some(keepalive_deadline),
                    entry_state: None,
                    ..BlockMetadataUpdate::default()
                },
            );
        }
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
        let block_size = self.prefix_cache.block_size();
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
        if !is_full_sealed_prefix(lookup.matched_len, block_size, blocks.len()) {
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
        let Ok(reserved_bytes) = pool.reserved_bytes() else {
            log::warn!("failed to query host pool reserved bytes for usage fraction");
            return 1.0;
        };
        reserved_bytes as f64 / pool.capacity_bytes() as f64
    }

    fn host_pool_demote_headroom_bytes(&self) -> usize {
        let Ok(pool) = self.host_pinned_pool.lock() else {
            return 0;
        };
        let capacity = pool.capacity_bytes();
        if capacity == 0 {
            return 0;
        }
        let demote_high_water = (capacity as f64 * self.config.t1_host_pinned_high_water) as usize;
        let Ok(reserved_bytes) = pool.reserved_bytes() else {
            log::warn!("failed to query host pool reserved bytes for demote headroom");
            return 0;
        };
        demote_high_water.saturating_sub(reserved_bytes)
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
        let store_submit_headroom = coordinator_submit_headroom(
            self.coordinator_queue_stats().capacity,
            self.coordinator_queue_stats().active(),
        );
        let host_demote_headroom = self.host_pool_demote_headroom_bytes() / block_bytes;
        store_submit_headroom.min(host_demote_headroom)
    }

    pub(super) fn release_host_region(&self, region: crate::kv_tier::HostPinnedRegion) {
        if let Err(err) = self.host_pinned_pool.release_region(region) {
            log::warn!(
                "failed to release host pinned region offset={} len={}: {}",
                region.offset,
                region.len,
                err
            );
        }
    }

    pub(super) fn materialize_prefetched_host_blocks(
        &mut self,
        fetched_blocks: &[crate::kv_tier::FetchedBlock],
    ) -> usize {
        let mut materialized = 0usize;
        for block in fetched_blocks {
            let Some(metadata) = self.block_metadata(block.block_id) else {
                if block.release_after_promote {
                    self.release_host_region(block.host_region);
                }
                continue;
            };
            let keepalive_deadline = metadata.session_id.as_ref().map(|_| {
                Some(
                    self.prefix_cache
                        .logical_clock()
                        .saturating_add(self.config.t1_host_pinned_keepalive_ticks),
                )
            });
            let updated = self.prefix_cache.update_block_metadata(
                block.block_id,
                BlockMetadataUpdate {
                    location: Some(BlockLocation::HostPinned {
                        offset: block.host_region.offset,
                    }),
                    byte_len: Some(block.byte_len as u32),
                    host_spill_pin_until: keepalive_deadline,
                    entry_state: Some(crate::kv_tier::IndexEntryState::Ready),
                    ..BlockMetadataUpdate::default()
                },
            );
            if updated {
                materialized += 1;
            } else if block.release_after_promote {
                self.release_host_region(block.host_region);
            }
        }
        materialized
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
            if self.prefetch_fetching.contains_key(&ticket) {
                continue;
            }
            if let Some(key) = self.fetch_ticket_keys.remove(&ticket) {
                self.fetch_dedupe.remove(&key);
            }
            self.fetch_ticket_started_at.remove(&ticket);
            let _ = self.coordinator_handle.cancel_fetch(ticket);
        }
    }

    pub(super) fn coordinator_queue_stats(&self) -> CoordinatorQueueStats {
        self.coordinator_handle
            .stats()
            .with_fetch_waiters(self.fetch_waiting.values().map(std::vec::Vec::len).sum())
    }

    pub(super) fn current_tier_wait_seconds(&self) -> (f64, f64) {
        let now = std::time::Instant::now();
        let fetch_wait_s = self
            .fetch_ticket_started_at
            .values()
            .map(|started_at| now.duration_since(*started_at).as_secs_f64())
            .fold(0.0, f64::max);
        let store_wait_s = self
            .store_ticket_started_at
            .values()
            .map(|started_at| now.duration_since(*started_at).as_secs_f64())
            .fold(0.0, f64::max);
        (fetch_wait_s, store_wait_s)
    }

    pub(super) fn has_pending_store_work(&self) -> bool {
        !self.store_waiting.is_empty()
    }

    fn host_pool_spill_target_bytes(&self, intent: BlockSelectionIntent) -> usize {
        let Ok(pool) = self.host_pinned_pool.lock() else {
            return 0;
        };
        let capacity_bytes = pool.capacity_bytes();
        if capacity_bytes == 0 {
            return 0;
        }
        let reserved_bytes = match pool.reserved_bytes() {
            Ok(bytes) => bytes,
            Err(err) => {
                log::warn!("failed to query host pool reserved bytes for spill target: {err}");
                return 0;
            }
        };
        host_spill_target_bytes(
            reserved_bytes,
            capacity_bytes,
            self.config.t1_host_pinned_high_water,
            self.config.t1_host_pinned_low_water,
            intent,
        )
    }

    pub(super) fn trigger_background_store_drain(&mut self) -> bool {
        if !self.has_pending_store_work()
            && self.host_pool_spill_target_bytes(BlockSelectionIntent::Drain) == 0
        {
            return false;
        }
        let _ = self.spill_host_blocks_if_pressured(BlockSelectionIntent::Drain);
        self.has_pending_store_work()
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
        let sealed_prefix_tokens =
            sealed_block_token_count(self.prefix_cache.block_size(), blocks.len());
        if token_count > sealed_prefix_tokens {
            return Err(anyhow::anyhow!(
                "attached prefix overran sealed blocks: tokens={} sealed_tokens={} blocks={}",
                token_count,
                sealed_prefix_tokens,
                blocks.len()
            ));
        }
        // `blocks` are always full sealed radix blocks. `token_count` may stop
        // inside the last attached block so the slot keeps a private hot tail
        // and reaches the only COW boundary at append time.
        let pages = self.flattened_pages_for_blocks(blocks)?;
        self.paged_kv_pool
            .attach_pages(slot_idx, &pages, token_count)
    }

    pub(super) fn release_attached_prefix_blocks(&mut self, blocks: &[BlockId]) {
        if !blocks.is_empty() {
            self.prefix_cache.release(blocks);
        }
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

    pub(super) fn slot_is_emit_gated(&self, slot_idx: usize) -> bool {
        self.request(slot_idx).is_some_and(|req| {
            self.emit_gate_waiting
                .get(&req.id)
                .is_some_and(|&waiting_slot| waiting_slot == slot_idx)
        })
    }

    pub(super) fn slot_is_runnable_decode(&self, slot_idx: usize) -> bool {
        self.request(slot_idx)
            .is_some_and(|req| matches!(req.phase, Phase::Decoding) && !req.delta_tx.is_closed())
            && !self.slot_is_emit_gated(slot_idx)
    }

    pub(super) fn has_runnable_decode_work(&self) -> bool {
        self.running_batch
            .iter()
            .any(|&slot_idx| self.slot_is_runnable_decode(slot_idx))
    }

    pub(super) fn is_fetch_wait_bound(&self) -> bool {
        self.active_len() > 0
            && self.pending_decode.is_none()
            && self.pending_prefill.is_none()
            && self.prefill_queue.is_empty()
            && !self.has_decode_work()
            && self
                .active
                .iter()
                .flatten()
                .all(|req| matches!(req.phase, Phase::WaitingFetch))
    }

    pub(super) fn has_pending_gpu_work(&self) -> bool {
        self.pending_decode.is_some() || self.pending_prefill.is_some()
    }

    pub(super) fn slot_has_pending_gpu_work(&self, slot_idx: usize) -> bool {
        self.pending_decode.as_ref().is_some_and(|pending| {
            pending.decode_indices.contains(&slot_idx)
                || pending
                    .mixed_prefill
                    .as_ref()
                    .is_some_and(|mixed| mixed.rows.iter().any(|row| row.slot_idx == slot_idx))
        }) || self
            .pending_prefill
            .as_ref()
            .is_some_and(|pending| pending.rows.iter().any(|row| row.slot_idx == slot_idx))
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
        partial_tail_capacity(
            (0..self.states.len()).map(|slot_idx| self.paged_kv_pool.seq_len(slot_idx)),
            self.paged_kv_pool.page_size,
        )
    }

    pub(super) fn pool_free_pages(&self) -> usize {
        let page_size = self.paged_kv_pool.page_size.max(1);
        let free_page_tokens = self
            .paged_kv_pool
            .free_count()
            .saturating_sub(self.pool_partial_tail_capacity());
        free_page_tokens / page_size
    }

    pub(super) fn additional_pages_needed_for_slot(
        &self,
        slot_idx: usize,
        additional_tokens: usize,
    ) -> usize {
        self.paged_kv_pool
            .append_pages_needed(slot_idx, additional_tokens)
    }

    pub(super) fn dispatch_emit(&mut self, slot_idx: usize) {
        let Some((
            request_id,
            prompt_tokens,
            tokens,
            latest_logprob,
            delta_tx,
            stops,
            gated,
            trace_context,
        )) = self.request(slot_idx).map(|req| {
            (
                req.id,
                req.prompt_tokens.len(),
                req.pending_emit_tokens(),
                req.latest_logprob,
                req.delta_tx.clone(),
                req.stops_for_emit_dispatch(),
                req.has_stop_sequences(),
                req.trace_context,
            )
        })
        else {
            return;
        };
        if tokens.is_empty() {
            return;
        }
        self.emit_tx
            .send(EmitCommand::Append {
                request_id,
                prompt_tokens,
                tokens,
                latest_logprob,
                delta_tx,
                stops,
                gated,
                trace_context,
            })
            .expect("emit worker channel disconnected during append dispatch");
        if let Some(req) = self.request_mut(slot_idx) {
            req.mark_emit_dispatched();
        }
        if gated {
            self.emit_gate_waiting.insert(request_id, slot_idx);
        }
    }

    pub(super) fn queue_emit_finish(&mut self, slot_idx: usize, reason: FinishReason) {
        let Some((request_id, prompt_tokens, generated_tokens, delta_tx, stops, trace_context)) =
            self.request(slot_idx).map(|req| {
                (
                    req.id,
                    req.prompt_tokens.len(),
                    req.generated_tokens.clone(),
                    req.delta_tx.clone(),
                    req.stops_for_emit_dispatch(),
                    req.trace_context,
                )
            })
        else {
            return;
        };
        let completion_tokens = generated_tokens.len();
        if completion_tokens == 0 {
            log::warn!(
                "Request {request_id}: queueing finish with 0 generated tokens reason={:?}",
                reason
            );
        }
        self.emit_tx
            .send(EmitCommand::Finish {
                request_id,
                prompt_tokens,
                completion_tokens,
                generated_tokens,
                reason,
                delta_tx,
                stops,
                trace_context,
            })
            .expect("emit worker channel disconnected during finish dispatch");
    }

    pub(super) fn defer_finish_until_emit_gate(
        &mut self,
        slot_idx: usize,
        reason: FinishReason,
    ) -> bool {
        let Some(req) = self.request_mut(slot_idx) else {
            return false;
        };
        if !req.has_stop_sequences() {
            return false;
        }
        req.pending_finish_reason = Some(reason);
        true
    }

    pub(super) fn finish_slot(&mut self, slot_idx: usize) {
        let request_id = self.request(slot_idx).map(|req| req.id);
        if let Some(req) = self.request_mut(slot_idx) {
            req.pending_finish_reason = None;
            req.phase = Phase::Finished;
        }
        if let Some(request_id) = request_id {
            self.clear_fetch_waiting_for_slot(slot_idx, request_id);
            self.emit_gate_waiting.remove(&request_id);
            let _ = self.emit_tx.send(EmitCommand::Abort { request_id });
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
        let sealed_block_count = publishable_tokens / block_size;
        if sealed_block_count == 0 {
            return;
        }
        let sealed_token_count = sealed_block_token_count(block_size, sealed_block_count);
        let sealed_block_pages = self.slot_sealed_block_pages(slot_idx, sealed_block_count);
        let required_pages = sealed_block_pages
            .iter()
            .map(std::vec::Vec::len)
            .sum::<usize>();
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

        let blocks: Vec<BlockId> = sealed_block_pages
            .iter()
            .map(|pages| Self::block_id_for_pages(pages))
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
        let mut block_fingerprints: Vec<BlockFingerprint> = Vec::with_capacity(sealed_block_count);
        for i in 0..sealed_block_count {
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

        // Publish only sealed full blocks. Any decode-time hot tail stays
        // request-private until it fills and later becomes its own sealed
        // block on a subsequent publish.
        let publishable_prompt = &prompt_tokens[..sealed_token_count];
        let inserted = self.prefix_cache.insert_with_fingerprints(
            publishable_prompt,
            &blocks,
            &block_fingerprints,
        );
        if inserted != sealed_token_count {
            warn!(
                "prefix_cache.insert: expected {} tokens, got {} (slot={}, num_blocks={}, prompt={})",
                sealed_token_count,
                inserted,
                slot_idx,
                sealed_block_count,
                prompt_tokens.len(),
            );
            return;
        }

        // Pin every physical page that backs the inserted blocks.
        // The radix refs a "block" as a unit, and the *entire*
        // `block_size`-wide span must survive `free_slot` so the
        // reuse path can read the full KV state back out.
        let slot_pages: Vec<u32> = sealed_block_pages
            .iter()
            .flat_map(|pages| pages.iter().copied())
            .collect();
        self.paged_kv_pool.retain_pages(&slot_pages);
        debug_assert!(
            self.slot_owned_blocks[slot_idx].is_empty(),
            "publish_to_prefix_cache must start from an unowned slot frontier"
        );
        self.record_sealed_gpu_blocks(
            slot_idx,
            blocks.into_iter().zip(sealed_block_pages),
            session_id,
            self.config.prefix_cache_keepalive_ticks,
            true,
        );
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
            pool.reserve(payload.len())?.ok_or_else(|| {
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
                host_spill_pin_until: metadata.session_id.as_ref().map(|_| {
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

    fn spill_host_blocks_if_pressured(&mut self, intent: BlockSelectionIntent) -> usize {
        let bytes_to_spill = self.host_pool_spill_target_bytes(intent);
        if bytes_to_spill == 0 {
            return 0;
        }

        let coordinator_stats = self.coordinator_queue_stats();
        let mut store_submit_headroom =
            coordinator_submit_headroom(coordinator_stats.capacity, coordinator_stats.active());
        if store_submit_headroom == 0 {
            return 0;
        }

        let mut spilled_bytes = 0usize;
        let candidates = self.prefix_cache.select_blocks_with_policy(
            &SessionBiasedLru::default(),
            self.eviction_signals(),
            self.prefix_cache.cached_block_count(),
            Some(crate::kv_tier::Tier::HostPinned),
            intent,
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
            if store_submit_headroom == 0 {
                break;
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
            store_submit_headroom = store_submit_headroom.saturating_sub(1);
            let _ = self.prefix_cache.mark_block_store_pending(block_id);
            self.store_waiting.insert(ticket, vec![(block_id, region)]);
            self.store_dedupe.insert(store_key, ticket);
            self.store_ticket_keys.insert(ticket, store_key);
            self.store_ticket_started_at
                .insert(ticket, std::time::Instant::now());
            spilled_bytes = spilled_bytes.saturating_add(metadata.byte_len as usize);
            if spilled_bytes >= bytes_to_spill {
                break;
            }
        }
        spilled_bytes
    }

    /// Release radix-held pool pages back to the free list once the
    /// pool crosses the retention watermark or queued admissions need
    /// more GPU headroom than the free list currently provides.
    ///
    /// Policy: reclaim enough pages to satisfy the larger of:
    ///
    /// - the usual watermark gap (`retained > high_water`, reclaim down to
    ///   `low_water`)
    /// - the next wave of queued admissions that can fill the currently free
    ///   slots, based on their cached prompt token lengths + `max_tokens`
    ///
    /// This keeps a single eviction policy/ranking path while allowing
    /// `cleanup()` to restore steady-state active-set occupancy under long
    /// prompts instead of parking half the pool in radix-retained pages.
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
        let waiting_shortage = self.waiting_admission_shortage_pages();
        let want_free = prefix_cache_reclaim_goal_pages(retained, high, target, waiting_shortage);
        if want_free == 0 {
            return 0;
        }
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
            BlockSelectionIntent::Evict,
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
        let _ = self.spill_host_blocks_if_pressured(BlockSelectionIntent::Spill);
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
    pub(super) fn evict_prefix_cache_for_allocation(&mut self, required_pages: usize) -> usize {
        let shortage_pages = required_pages.saturating_sub(self.pool_free_pages());
        if shortage_pages == 0 {
            return 0;
        }

        let pages_per_block = self
            .prefix_cache
            .block_size()
            .div_ceil(self.paged_kv_pool.page_size);
        let mut reclaimed_pages = 0usize;
        let mut remaining_pages = shortage_pages;
        while remaining_pages > 0 {
            let blocks_to_move = remaining_pages.div_ceil(pages_per_block.max(1)).max(1);
            let candidates = self.prefix_cache.select_blocks_with_policy(
                &SessionBiasedLru::default(),
                self.eviction_signals(),
                blocks_to_move,
                Some(crate::kv_tier::Tier::Gpu),
                BlockSelectionIntent::Evict,
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
                    }
                    Err(err) => {
                        warn!(
                            "failed to demote block {:?} during alloc reclaim: {}",
                            bid, err
                        );
                    }
                }
            }
            let _ = self.spill_host_blocks_if_pressured(BlockSelectionIntent::Spill);
            if reclaimed_pages == before {
                break;
            }
            remaining_pages = shortage_pages.saturating_sub(reclaimed_pages);
        }

        if reclaimed_pages < shortage_pages {
            let blocks_to_drop = remaining_pages.div_ceil(pages_per_block.max(1)).max(1);
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
                 (required_pages={}, free_pages_now={})",
                reclaimed_pages,
                required_pages,
                self.pool_free_pages(),
            );
        }
        reclaimed_pages
    }

    fn try_alloc_pool_tokens_once(&mut self, slot: usize, count: usize) -> Result<Vec<u32>> {
        if count > 0 {
            self.paged_kv_pool
                .cow_tail_page_for_append(self.model.device_context(), slot)?;
        }
        self.paged_kv_pool.alloc_tokens(slot, count)
    }

    /// Allocate pool pages, forcing prefix-cache eviction and retrying once on
    /// OOM. This is the M2b safety net for bursty admissions between cleanup
    /// passes.
    pub(super) fn alloc_pool_tokens_with_retry(
        &mut self,
        slot: usize,
        count: usize,
    ) -> Result<Vec<u32>> {
        let required_pages = self.additional_pages_needed_for_slot(slot, count);
        if required_pages > self.pool_free_pages() {
            self.evict_prefix_cache_for_allocation(required_pages);
        }
        match self.try_alloc_pool_tokens_once(slot, count) {
            Ok(indices) => Ok(indices),
            Err(first_err) => {
                let required_pages = self.additional_pages_needed_for_slot(slot, count);
                let reclaimed = self.evict_prefix_cache_for_allocation(required_pages);
                if reclaimed == 0 {
                    Err(first_err)
                } else {
                    self.try_alloc_pool_tokens_once(slot, count)
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
        let (dummy_coord, dummy_handle, dummy_events) =
            crate::kv_tier::CoordinatorBuilder::new(1).build();
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
        let (dummy_emit_tx, dummy_emit_rx) = crossbeam_channel::unbounded();
        drop(dummy_emit_rx);
        let old_emit_tx = std::mem::replace(&mut self.emit_tx, dummy_emit_tx);
        drop(old_emit_tx);
        if let Some(thread) = self.emit_thread.take() {
            if thread.join().is_err() {
                warn!("Emit worker thread panicked during shutdown");
            }
        }
    }
}

#[cfg(test)]
#[path = "core/tests.rs"]
mod tests;
