use super::*;
use crate::qwen35::alloc_recurrent_block;
use std::cmp::Ordering;

/// Set the host slot's accounted length to `target`. The engine pre-budgets
/// the full spec chain (#197), so this normally no-ops; a warm row or a chain
/// shorter than the budget truncates the over-allocation instead of leaving
/// the host pool ahead of the device truth.
fn set_host_slot_to(host_kv: &mut dyn KvPool, slot: usize, target: usize) -> Result<()> {
    match target.cmp(&host_kv.seq_len(slot)) {
        Ordering::Less => host_kv.truncate_slot(slot, target),
        Ordering::Equal => Ok(()),
        Ordering::Greater => host_kv.alloc(slot, target - host_kv.seq_len(slot)),
    }
}

fn speculative_chain_fits(start: usize, depth: usize, max_seq_len: usize) -> bool {
    start
        .checked_add(depth)
        .is_some_and(|last_position| last_position < max_seq_len)
}

/// One chain in a batched DSpark verify; `row0` indexes the shared logits and tap
/// features.
struct DsparkChain {
    /// Index of the originating row in the tick's `decode_rows`.
    out: usize,
    slot: usize,
    start: usize,
    row0: usize,
    chain: Vec<u32>,
    partial_ctx: bool,
}

fn merge_tier_io_stats(
    slot: &kv_native_sys::TierIoStats,
    recall: &kv_native_sys::TierIoStats,
) -> infer_seam::KvTierIoStats {
    let mode = if [slot.mode, recall.mode].contains(&kv_native_sys::DiskIoMode::Direct) {
        infer_seam::KvTierIoMode::Direct
    } else if [slot.mode, recall.mode].contains(&kv_native_sys::DiskIoMode::Mmap) {
        infer_seam::KvTierIoMode::Mmap
    } else {
        infer_seam::KvTierIoMode::Disabled
    };
    infer_seam::KvTierIoStats {
        mode,
        useful_read_bytes: slot
            .useful_read_bytes
            .saturating_add(recall.useful_read_bytes),
        useful_write_bytes: slot
            .useful_write_bytes
            .saturating_add(recall.useful_write_bytes),
        submitted_read_bytes: slot
            .submitted_read_bytes
            .saturating_add(recall.submitted_read_bytes),
        submitted_write_bytes: slot
            .submitted_write_bytes
            .saturating_add(recall.submitted_write_bytes),
        metadata_write_bytes: slot
            .metadata_write_bytes
            .saturating_add(recall.metadata_write_bytes),
        failures: slot.failures.saturating_add(recall.failures),
        completion_wait_ns: slot
            .completion_wait_ns
            .saturating_add(recall.completion_wait_ns),
    }
}

static QWEN35_GRAPH_CAPTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static QWEN35_GRAPH_REPLAYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Device addresses a slot's captured decode graph was baked against: the graph
/// replays against FIXED pointers, so replaying a stale bake reads freed memory.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Qwen35GraphBake {
    token_ids_ptr: u64,
    start_pos_ptr: u64,
    logits_ptr: u64,
    ws_epoch: u64,
}

/// Per-slot decode-graph state on a DEDICATED `seq_len == 1` workspace — the main
/// workspace re-shapes on every prefill chunk and would invalidate captures.
struct Qwen35DecodeGraph {
    ws: crate::qwen35::Qwen35Workspace,
    graphs: Vec<crate::graph::CudaGraphState>,
    baked: Vec<Option<Qwen35GraphBake>>,
}

impl Qwen35DecodeGraph {
    fn new(num_slots: usize, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Self {
        Self {
            ws: crate::qwen35::Qwen35Workspace::new(),
            graphs: (0..num_slots)
                .map(|_| crate::graph::CudaGraphState::new(stream.clone()))
                .collect(),
            baked: vec![None; num_slots],
        }
    }
}

/// Qwen3.5 / Qwen3.6 hybrid executor. Owns per-slot KV + recurrent state inside the
/// model, so the host [`KvPool`] is consulted only for the slot's logical `seq_len`.
/// Prefill stays single-row; mixed plans run per-prefill sub-steps, then one decode
/// sub-batch.
pub(crate) struct Qwen35CudaExecutor {
    pub(crate) model: crate::qwen35::Qwen35Model,
    /// A slot's recurrent state is EMPTY until its first request activates it.
    slots: Vec<crate::qwen35::Qwen35SlotState>,
    /// Whole-slot capacity spill: a parked request's snapshot, restored byte-exact on
    /// resume. Keyed by the engine session key — a namespace disjoint from
    /// `recall_tier`'s `tier_block_u64(slot, page)` keys.
    slot_tier: KvTierStore,
    /// Free-list of detached recurrent blocks (~147 MiB each), so only ACTIVE slots
    /// hold a block rather than all `num_slots`.
    recurrent_pool: Vec<crate::qwen35::RecurrentBlock>,
    /// Forwards are strictly serial on this executor, so ONE workspace serves every
    /// slot.
    workspace: crate::qwen35::Qwen35Workspace,
    pub(crate) num_slots: usize,
    /// ANY capture failure clears this — eager is the permanent fallback, never fatal.
    decode_graph_armed: bool,
    /// Re-`None`d whenever baked addresses go stale: weight offload/reload, LoRA
    /// re-merge.
    decode_graph: Option<Qwen35DecodeGraph>,
    /// Per-slot fixed-capacity page table for the paged decode-graph lane: device
    /// addresses are capture-stable, contents refresh each step outside the graph.
    paged_decode_meta: Vec<Option<crate::loader::PageMeta>>,
    batch_decode: Option<crate::qwen35::Qwen35BatchDecodeState>,
    /// Recall cycle opt-in (`--kv-recall`): layers a working-set restriction on the
    /// SAME paged `full_attn_kv` pool; off attends the full resident page set.
    kv_recall: bool,
    recall_cfg: infer_core::RecallConfig,
    recall: Vec<crate::recall::CudaRecallState>,
    /// Shared paged full-attn KV pool, profile-sized from measured free VRAM. Both the
    /// default forward and the recall cycle use it; `Option` only so OPD offload can
    /// drop it.
    full_attn_kv: Option<PagedKVPool>,
    /// Stored so `ensure_kv_pool` rebuilds the pool with the same format after release.
    kv_format: KVFormat,
    /// L3 write-through tier: source of truth for evict-dropped middle blocks, sized to
    /// ONE pool page image. Keyed by `tier_block_u64(slot, page)`, so slot A never
    /// prefetches slot B's KV.
    recall_tier: Option<KvTierStore>,
    /// One-step eviction keepalive: a page dropped at step N is parked here until the
    /// start of step N+1, so `alloc_tokens` can never hand the in-flight attention's
    /// page to the new token. Holds (logical, physical).
    recall_keepalive: Vec<Vec<(usize, u32)>>,
    /// Per-rank L2 byte budget (`--kv-dram` ÷ world size).
    recall_budget_bytes: usize,

    /// Stamped into the durable recall manifest so a restart drops stale KV after an
    /// OPD weight update.
    weights_epoch: String,
    /// NVMe root for durable recall spill (`--kv-disk`).
    disk_root: Option<std::path::PathBuf>,
    /// Budget bytes for durable NVMe recall spill (`--kv-disk-limit`).
    disk_budget: Option<usize>,
    /// The constructed pool's own `max_total_pages`; `ensure_kv_pool` rebuilds at this
    /// size.
    kv_pool_sized_pages: usize,
    /// Eviction coordination only: the tail host-pool page of each published prefix →
    /// its sidecar key in `slot_tier`, so a sidecar's lifetime rides the radix blocks.
    sidecar_page_key: std::collections::HashMap<u32, u64>,
    /// Per-slot recurrent snapshot captured at `L* = align_down16(prompt_len - 1)`, the
    /// exact-resend restore target. The device recurrent state cannot rewind, so
    /// snapshot-position must equal key-position or restore double-advances the
    /// residue.
    prefill_boundary_snapshot: Vec<Option<(usize, crate::qwen35::Qwen35RecurrentSnapshot)>>,
    /// Per-slot recurrent snapshots at every stride boundary `S` crossed during
    /// prefill,
    /// each carrying the state at EXACTLY `S` for a future cross-conversation restore.
    periodic_boundary_snapshots: Vec<Vec<(usize, crate::qwen35::Qwen35RecurrentSnapshot)>>,
    /// DSpark block-draft runtime (`--spec-type dspark`): draft head, per-slot ctx
    /// caches, tap/scratch buffers.
    pub(crate) dspark: Option<crate::qwen35::dspark::Qwen35DsparkExec>,
    /// MTP spec-decode state (`--spec-type mtp`): the spec state plus the seed
    /// (pending token + hidden) for the next spec step.
    pub(crate) mtp: Option<MtpExec>,
}

/// Per-slot MTP spec-decode state; created lazily by the first warm decode step.
pub(crate) struct MtpExec {
    slots: Vec<Option<MtpSlotState>>,
    /// Cumulative counters (host-side, no device sync) — the /v1/stats spec source.
    pub(crate) accepts: usize,
    pub(crate) rejects: usize,
    pub(crate) chains: usize,
}

struct MtpSlotState {
    spec: crate::qwen35::Qwen35SpecSlotState,
    pending: u32,
    hidden: DeviceVec,
}

impl std::fmt::Debug for Qwen35CudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35CudaExecutor")
            .field("model", &self.model)
            .field("num_slots", &self.num_slots)
            .field("decode_graph_armed", &self.decode_graph_armed)
            .field(
                "captured_decode_slots",
                &self
                    .decode_graph
                    .as_ref()
                    .map_or(0, |dg| dg.graphs.iter().filter(|g| g.is_captured()).count()),
            )
            .finish()
    }
}

impl Qwen35CudaExecutor {
    /// `|_| false`: demote/promote is a no-op for Qwen35, so demoted pages are never
    /// restorable.
    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        pages_only_reusable_prefix_blocks(blocks, |_| false)
    }

    /// Host page ids index device storage rows 1:1, so a mirrored prefix reads its KV
    /// straight out of HBM.
    fn mirror_host_slot(
        &mut self,
        host_kv: &dyn KvPool,
        slot: usize,
        seq_len: usize,
    ) -> Result<()> {
        let host_pages = host_kv.page_indices(slot);
        let pool = self
            .full_attn_kv
            .as_mut()
            .expect("full_attn_kv present (full_attn_paged)");
        let global_pages = seq_len.div_ceil(pool.page_size);
        let need = host_kv.shard_local_page_count(global_pages);
        ensure!(
            host_pages.len() >= need,
            "host pool holds {} pages for slot {slot}, {need} needed to cover {seq_len} tokens",
            host_pages.len()
        );
        pool.mirror_slot(slot, &host_pages[..need], seq_len)
    }

    /// Store the slot's recurrent state into `slot_tier`, keyed by the token hash of
    /// the
    /// published radix prefix at `L* = align_down16(tokens.len() - 1)` — the boundary a
    /// future exact resend restores to (`prefix.rs:72`). Falls back to the live
    /// snapshot
    /// keyed at the sealed page grain when no prefill snapshot was captured at `L*`.
    pub(crate) fn save_recurrent_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> anyhow::Result<()> {
        if !self.slots[slot].has_recurrent() {
            return Ok(());
        }
        if tokens.len() < 256 {
            return Ok(());
        }
        // Drain up front so every exit path leaves the slot clean for its next
        // occupant;
        // a leaked entry would double-save under a later publish.
        let periodic = std::mem::take(&mut self.periodic_boundary_snapshots[slot]);
        let boundary = tokens.len().saturating_sub(1) / SUPPORTED_PAGE_SIZE * SUPPORTED_PAGE_SIZE;
        let pending = self.prefill_boundary_snapshot[slot].take();
        let mat_len = matched_len
            .min(self.slots[slot].seq_len())
            .min(tokens.len())
            / SUPPORTED_PAGE_SIZE
            * SUPPORTED_PAGE_SIZE;
        if mat_len == 0 {
            return Ok(());
        }
        // Periodic sidecars for a future cross-conversation restore, each keyed at its
        // exact snapshot position. Full-attn KV is not in the blob: restore mirrors the
        // radix prefix's own device pages.
        for (pos, psnap) in periodic {
            if pos == 0 || pos > mat_len {
                continue;
            }
            let pkey = crate::qwen35::hash_prefix_tokens(&tokens[..pos]);
            self.store_sidecar_blob(pos, pkey, psnap.to_bytes(), prefix_pages);
        }
        // The L* prefill snapshot (full pair) is always restorable. A fresh
        // snapshot on a B2-live slot is not: the live state is the 1/cp decode
        // subset and the full pair is frozen at the scatter point, so a
        // tail-prefill resume would advance a stale full pair. Skip it; the
        // prefix recomputes on a future hit.
        let snap = match pending {
            Some((pos, snap)) if pos == boundary && boundary > 0 => Some(snap),
            _ if self.slots[slot].decode_recurrent_live => None,
            _ => Some(self.slots[slot].snapshot_recurrent(&self.model.ctx)?),
        };
        if let Some(snap) = snap {
            let key = crate::qwen35::hash_prefix_tokens(&tokens[..mat_len]);
            self.store_sidecar_blob(mat_len, key, snap.to_bytes(), prefix_pages);
        }
        Ok(())
    }

    /// Insert a sidecar blob and coordinate its eviction off the last radix page it
    /// covers: leaves evict deepest-first, so the blob drops as its own prefix erodes.
    fn store_sidecar_blob(&mut self, pos: usize, key: u64, bytes: Vec<u8>, prefix_pages: &[u32]) {
        if !self
            .slot_tier
            .insert_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key, &bytes)
        {
            return;
        }
        let cover_idx = (pos / SUPPORTED_PAGE_SIZE).saturating_sub(1);
        if let Some(&tail) = prefix_pages.get(cover_idx).or_else(|| prefix_pages.last())
            && let Some(old) = self.sidecar_page_key.insert(tail, key)
            && old != key
        {
            self.slot_tier
                .remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, old);
        }
    }

    /// Drop sidecar blobs keyed to evicted radix pages — eviction rides the radix,
    /// no independent sidecar LRU.
    pub(crate) fn release_sidecar_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            if let Some(key) = self.sidecar_page_key.remove(&page) {
                self.slot_tier
                    .remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key);
            }
        }
    }

    /// Return `slot`'s device pages NOW: a lazy free leaves the host admission pool
    /// over-reporting free pages, so the planner licenses prefill chunks the device
    /// pool
    /// cannot hold. Idempotent with the prefill-start mirror clear.
    pub(crate) fn release_kv_slot(&mut self, slot: usize) -> Result<()> {
        if slot >= self.num_slots {
            return Ok(());
        }
        let parked = std::mem::take(&mut self.recall_keepalive[slot]);
        if let Some(pool) = self.full_attn_kv.as_mut() {
            for (_logical, physical) in parked {
                pool.release_evicted_page(physical);
            }
            pool.mirror_slot(slot, &[], 0)?;
        }
        Ok(())
    }

    /// Restore the recurrent sidecar for a prefix hit, returning the ABSOLUTE token
    /// length restored — `matched_len`, or the largest stride boundary `B ≤
    /// matched_len`
    /// whose sidecar is present. Sets the device pool seq_len to `B`; the engine
    /// truncates the host pool to `B` and re-prefills `[B..prompt]`. A miss at every
    /// boundary returns `Err` so the caller full-recomputes.
    pub(crate) fn restore_recurrent_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> anyhow::Result<usize> {
        let matched_len = matched_len.min(tokens.len());
        let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
        // Reused slot: start_pos != 0 skips the normal release+acquire in
        // submit_prefill_row.
        self.slots[slot].release_recurrent(&mut self.recurrent_pool);
        self.slots[slot].acquire_recurrent(
            &self.model.ctx,
            num_linear,
            gdr_len,
            conv_len,
            &mut self.recurrent_pool,
        )?;

        // A sidecar-restored prefix has no draft ctx for the matched span; the tail
        // prefill rebases the draft ctx at start_pos.
        if let Some(df) = self.dspark.as_mut().and_then(|ds| ds.slots[slot].as_mut()) {
            df.reset();
        }

        // Probe largest-first; each boundary is page-aligned, so `hash(tokens[..B])`
        // rendezvous with the save keys.
        let stride = SIDECAR_SNAPSHOT_STRIDE_PAGES * SUPPORTED_PAGE_SIZE; // const, > 0
        let mut candidates: Vec<usize> = Vec::new();
        if matched_len > 0 {
            candidates.push(matched_len);
        }
        let mut b = matched_len / stride * stride;
        while b >= stride {
            if b != matched_len {
                candidates.push(b);
            }
            b -= stride;
        }
        // A corrupt/foreign payload deserializes to None and is skipped.
        let restored = candidates.into_iter().find_map(|b| {
            let key = crate::qwen35::hash_prefix_tokens(&tokens[..b]);
            self.slot_tier
                .read_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key)
                .ok()
                .and_then(|bytes| crate::qwen35::Qwen35RecurrentSnapshot::from_bytes(&bytes).ok())
                .map(|snap| (b, snap))
        });

        let Some((boundary, snap)) = restored else {
            // Clean up full_attn_kv and seq_len here — the caller's Err handler won't.
            if let Some(pool) = self.full_attn_kv.as_mut() {
                pool.mirror_slot(slot, &[], 0)?;
            }
            self.slots[slot].set_seq_len(0);
            return Err(anyhow::anyhow!(
                "no recurrent sidecar for prefix matched_len={matched_len} \
                 (probed stride={stride}); falling back to full recompute"
            ));
        };

        // restore_recurrent_from_snapshot doesn't advance seq_len.
        self.slots[slot].restore_recurrent_from_snapshot(&self.model.ctx, &snap)?;
        self.slots[slot].set_seq_len(boundary);

        // The sidecar carries recurrent state only; the prefix's own pages are already
        // attached and resident.
        if let Some(pool) = self.full_attn_kv.as_mut() {
            let need = boundary.div_ceil(SUPPORTED_PAGE_SIZE);
            ensure!(
                prefix_pages.len() >= need,
                "prefix restore: {} attached pages cover less than boundary {boundary} for slot {slot}",
                prefix_pages.len()
            );
            pool.mirror_slot(slot, &prefix_pages[..need], boundary)?;
        }
        // Drop any prior occupant's periodic snapshots so this request's saves are
        // clean.
        self.periodic_boundary_snapshots[slot].clear();
        Ok(boundary)
    }

    pub(crate) fn kv_tier_host_demoted_pages(&self) -> usize {
        self.slot_tier.host_demoted_pages()
    }

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        self.slot_tier.disk_pages()
    }

    pub(crate) fn kv_tier_io_stats(&self) -> infer_seam::KvTierIoStats {
        let slot = self.slot_tier.io_stats();
        let recall = self
            .recall_tier
            .as_ref()
            .map_or_else(kv_native_sys::TierIoStats::default, KvTierStore::io_stats);
        merge_tier_io_stats(&slot, &recall)
    }

    fn tp_min_usize(&self, value: usize, what: &str) -> Result<usize> {
        let capped = i32::try_from(value.min(i32::MAX as usize)).unwrap_or(i32::MAX);
        self.model
            .tp
            .all_reduce_min_scalar_i32(&self.model.ctx, capped)
            .map(|v| v.max(0) as usize)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 TP min-reduce {what} failed: {e}"))
    }

    /// See `BackendExecutor::tp_sync_min` (2026-07-05 TP=4 admission livelock).
    pub(crate) fn tp_sync_min(&self, local: usize) -> Result<usize> {
        self.tp_min_usize(local, "admission free pages")
    }

    /// 2D (attn_tp × cp) engages only when both partitions are real (pinned
    /// decision 4). Under it each rank's pool holds 1/cp of the sequence pages
    /// (block-cyclic: logical page `i` on shard `i % cp`).
    pub(crate) fn two_d_engaged(&self) -> bool {
        self.model.tp.two_d_engaged()
    }

    /// This rank's (cp_rank, cp_size) for the host pool's shard filter, or
    /// `None` when 2D is not engaged.
    pub(crate) fn kv_shard_spec(&self) -> Option<(usize, usize)> {
        self.two_d_engaged()
            .then(|| (self.model.tp.attn_cp_rank(), self.model.tp.attn_cp_size()))
    }

    /// Demote `slot`'s entire device state into `slot_tier` under `key`. The copy is
    /// complete before returning, so the engine may free the slot immediately. Returns
    /// `Ok(false)` when the tier is at budget on ANY rank. Exactly TWO collectives on
    /// every path, so the lockstep collective count is rank-invariant.
    pub(crate) fn demote_slot(&mut self, slot: usize, key: u64) -> Result<bool> {
        ensure!(
            slot < self.num_slots,
            "Qwen3.6 demote slot {slot} outside executor slots {}",
            self.num_slots
        );
        let image = {
            let Self {
                model,
                slots,
                full_attn_kv,
                recurrent_pool,
                ..
            } = &mut *self;
            let pool = full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (whole-slot demote)");
            slots[slot].swap_out_image(&model.ctx, slot, pool, recurrent_pool)
        };
        let capture_ok = usize::from(image.is_ok());
        if self.tp_min_usize(capture_ok, "slot demote capture")? == 0 {
            return Err(image.err().unwrap_or_else(|| {
                anyhow::anyhow!("peer rank failed Qwen3.6 slot demote capture")
            }));
        }
        // Chunked (16 MiB store pages): a whole image never fits one page. Per-rank
        // DRAM
        // headroom can diverge, so the verdict is min-reduced and locally-successful
        // ranks
        // roll their insert back on a mixed verdict.
        let bytes = image?.to_bytes();
        let inserted = self
            .slot_tier
            .insert_chunked(NS_SLOT, NS_SLOT_CHUNK, key, &bytes);
        if self.tp_min_usize(usize::from(inserted), "slot demote insert")? == 0 {
            if inserted {
                self.slot_tier.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
            }
            return Ok(false);
        }
        Ok(true)
    }

    /// Restore the whole-slot snapshot stored under `key` into `slot`. The entry stays
    /// in the tier — the engine drops it via [`Self::drop_kv_slot_entries`]. Exactly
    /// TWO
    /// collectives on every path; nothing rank-local errs between them.
    pub(crate) fn promote_slot(&mut self, key: u64, slot: usize, slot_pages: &[u32]) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "Qwen3.6 promote slot {slot} outside executor slots {}",
            self.num_slots
        );
        // The slot image carries no draft ctx cache; the first warm decode rebases it.
        if let Some(df) = self.dspark.as_mut().and_then(|ds| ds.slots[slot].as_mut()) {
            df.reset();
        }
        let image = self
            .slot_tier
            .read_chunked(NS_SLOT, NS_SLOT_CHUNK, key)
            .map_err(|err| anyhow::anyhow!("Qwen3.6 whole-slot tier read key {key}: {err}"))
            .and_then(|bytes| crate::qwen35::Qwen35SlotImage::from_bytes(&bytes));
        let image_ok = usize::from(image.is_ok());
        if self.tp_min_usize(image_ok, "slot promote read")? == 0 {
            return Err(image
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("peer rank failed Qwen3.6 slot promote read")));
        }
        let image = image?;
        // Infallible config data — safe between the collectives.
        let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
        let restored = {
            let Self {
                model,
                slots,
                full_attn_kv,
                recurrent_pool,
                ..
            } = &mut *self;
            let pool = full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (whole-slot promote)");
            slots[slot].swap_in_image(
                &model.ctx,
                slot,
                pool,
                recurrent_pool,
                num_linear,
                gdr_len,
                conv_len,
                &image,
                slot_pages,
            )
        };
        let restore_ok = usize::from(restored.is_ok());
        if self.tp_min_usize(restore_ok, "slot promote restore")? == 0 {
            return Err(restored.err().unwrap_or_else(|| {
                anyhow::anyhow!("peer rank failed Qwen3.6 slot promote restore")
            }));
        }
        restored
    }

    pub(crate) fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        for &key in keys {
            self.slot_tier.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
        }
    }

    pub(crate) fn from_qwen35_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        max_total_tokens: usize,
        kv_dtype: CudaKvCacheDtype,
        mem_fraction_static: f64,
        dspark_draft_model: Option<&Path>,
        dspark_sps_bias_ms: f32,
        dspark_sps_row_ms: f32,
        markov_head_rank: Option<usize>,
        dspark_block_size: Option<usize>,
        mtp_draft_tokens: Option<usize>,
        memory_budget_bytes: Option<usize>,
    ) -> Result<Self> {
        let total_t0 = Instant::now();
        ensure!(
            num_slots > 0,
            "Qwen35CudaExecutor requires at least one slot"
        );
        ensure!(
            total_pages > 0,
            "Qwen35CudaExecutor requires at least one KV page"
        );
        ensure!(
            dspark_draft_model.is_none() || mtp_draft_tokens.is_none(),
            "Qwen3.5/3.6 DSpark and MTP speculative decode are mutually exclusive"
        );
        // Per-request token ceiling: the model's positional budget + the host pool's
        // admission span.
        let max_seq_len = total_pages * SUPPORTED_PAGE_SIZE;
        let model_t0 = Instant::now();
        // `None`: the checkpoint-native NextN-MTP head stays unwired; DSpark is loaded
        // below from its own checkpoint dir.
        let mut model = crate::qwen35::Qwen35Model::from_safetensors(
            model_path.as_ref(),
            max_seq_len,
            mtp_draft_tokens,
        )?;
        // CP preconditions (T2.b replicated gather and the T3.2b 2D ring/decode
        // merge): a real NCCL cp sub-comm, and the BF16 pool (the quant-KV write
        // path does not cover remote cp slices).
        // DSpark is rejected below by its own single-GPU ensure; --kv-recall is
        // rejected at enable time (rank-local eviction scoring would diverge
        // the cp group's collective schedule).
        // attn_dp>1 would replicate attn_tp-sharded weights across dp peers
        // while the cp=1 reduce stays global — double-counted attention. No
        // engine dp routing exists; reject until it does.
        ensure!(
            model.tp.attn_tp_size() * model.tp.attn_cp_size() == model.tp.config().world_size
                || model.tp.config().is_single(),
            "attn_dp>1 is not supported by the qwen35 engine (attention would double-count)"
        );
        // 2D mutex: the MTP spec row builds a global-coverage PageMeta and
        // forwards with cp:None, neither shard-aware — the spec path is not
        // wired for the sharded pool. Reject at construction, like dspark.
        ensure!(
            !model.tp.two_d_engaged() || mtp_draft_tokens.is_none(),
            "--spec-type mtp is not supported under 2D (attn_tp>=2 && attn_cp>=2): the spec \
             row's page meta is not shard-aware"
        );
        if model.tp.attn_cp_size() > 1 {
            ensure!(
                model.tp.attn_cp().is_collective(),
                "attn_cp>1 requires the NCCL attn_cp sub-comm (build with --features nccl)"
            );
            ensure!(
                kv_dtype == CudaKvCacheDtype::Bf16,
                "attn_cp>1 requires --kv-cache-dtype bf16 (CP prefill KV all-gather writes \
                 raw BF16 pages; the quantized-pool path is not CP-aware)"
            );
        }
        cuda_startup_log(
            "executor.qwen35_model_load",
            model_t0,
            format_args!(
                "requested_slots={num_slots} total_pages={total_pages} max_seq_len={max_seq_len}"
            ),
        );
        // Loaded BEFORE the KV budget/pool profiling so its weights are subtracted from
        // the measured free VRAM.
        let dspark_head = dspark_draft_model
            .map(|dir| {
                ensure!(
                    model.tp.is_single(),
                    "--spec-type dspark is single-GPU only (draft lm_head/argmax are rank-local)"
                );
                let head = crate::qwen35::dspark::load_dspark_head(
                    &model.ctx,
                    dir,
                    max_seq_len,
                    max_total_tokens,
                    model.config.hidden_size,
                    model.config.num_hidden_layers,
                    model.config.vocab_size,
                    qwen35_spec::DsparkSps {
                        bias_ms: dspark_sps_bias_ms,
                        row_ms: dspark_sps_row_ms,
                        confidence_threshold: crate::runtime_flags::dspark_confidence_threshold(),
                    },
                    markov_head_rank,
                    dspark_block_size,
                )?;
                model.set_spec_draft_tokens(head.block_size());
                log::info!(
                    "CUDA Qwen3.6 DSpark drafter loaded from {}: mode={} block={} taps={:?}",
                    dir.display(),
                    head.mode_label(),
                    head.block_size(),
                    head.target_layer_ids(),
                );
                Ok(head)
            })
            .transpose()?;
        // Clamp num_slots to what post-weights free VRAM affords; deterministic and
        // NCCL min-reduced, so it stays TP-consistent.
        let budget_t0 = Instant::now();
        // Reclaim retained loading scratch before the budget measures free VRAM.
        if let Err(e) = model.ctx.trim_memory_pool() {
            log::warn!("pre-KV-budget trim_memory_pool failed (non-fatal): {e}");
        }
        let dspark_slot_bytes = dspark_head
            .as_ref()
            .map_or(0, |h| h.slot_state_bytes(model.config.vocab_size));
        let requested_pages = total_pages.max(1);
        let kv_format = kv_dtype.kv_format();
        // Joint (num_slots, pool_pages) solve (#182): slots and the shared pool
        // trade against the same free VRAM, so they are planned together.
        let plan = model.kv_budget_plan(
            num_slots,
            requested_pages,
            dspark_slot_bytes,
            memory_budget_bytes,
            mem_fraction_static,
            kv_format,
        )?;
        let num_slots = plan.num_slots;
        // Under 2D each rank stores 1/cp of the sequence pages (block-cyclic),
        // so the device pool is sized at plan.pool_pages / cp; the host
        // admission pool follows via effective_total_pages (1:1 id contract).
        let pool_pages = if model.tp.two_d_engaged() {
            plan.pool_pages / model.tp.attn_cp_size()
        } else {
            plan.pool_pages
        };
        cuda_startup_log(
            "executor.qwen35_kv_budget",
            budget_t0,
            format_args!("effective_slots={num_slots} pool_pages={pool_pages}"),
        );
        let slots_t0 = Instant::now();
        // Empty slots: each slot's ~147 MiB recurrent block is drawn from
        // `recurrent_pool` on its first request and returned on finish.
        let slots: Vec<_> = (0..num_slots).map(|_| model.new_slot_state()).collect();
        cuda_startup_log(
            "executor.qwen35_slot_alloc",
            slots_t0,
            format_args!("slots={num_slots} max_seq_len={max_seq_len}"),
        );

        // Sized by the joint budget plan above (the ONLY profile — see
        // `ensure_kv_pool`).
        let pool_t0 = Instant::now();
        let full_attn_kv = Self::alloc_full_attn_kv_pool(&model, num_slots, pool_pages, kv_format)?;
        let kv_pool_sized_pages = full_attn_kv.max_total_pages;
        cuda_startup_log(
            "executor.qwen35_paged_pool_alloc",
            pool_t0,
            format_args!("built"),
        );

        // Reserve the recurrent VRAM for ALL slots at construction: allocating each
        // ~144 MiB block on first request lets a concurrent burst OOM mid-request.
        let (num_linear, gdr_state_len, conv_len) = model.recurrent_dims();
        let (gdr_per_layer, conv_per_layer) = model.linear_state_bytes();
        let gdr_bytes = num_linear * gdr_per_layer;
        let conv_bytes = num_linear * conv_per_layer;
        log::info!(
            "Qwen3.5 pre-alloc recurrent pool: {num_slots} slots × \
             ({gdr_bytes} B gdr + {conv_bytes} B conv) = {} MiB",
            num_slots * (gdr_bytes + conv_bytes) / (1 << 20)
        );
        let recurrent_pool = (0..num_slots)
            .map(|_| alloc_recurrent_block(&model.ctx, num_linear, gdr_state_len, conv_len))
            .collect::<Result<Vec<_>>>()?;

        // Whole-slot spill: snapshots stored as 16 MiB chunked blobs — a whole image
        // never fits one fixed page, and the store's size contract is per-page.
        let tier_budget_bytes = default_t1_budget_per_rank();
        let slot_tier = KvTierStore::with_budget(tier_budget_bytes, BLOB_CHUNK_BYTES);

        // Single-GPU only: NCCL all-reduce is not graph-capturable on this stack.
        let decode_graph_armed = crate::runtime_flags::qwen35_decode_graph()
            && model.tp.is_single()
            && model.decode_graph_unsupported_reason().is_none();
        let model_path_buf = model_path.as_ref().to_path_buf();
        let weights_epoch = kv_native_sys::weights_epoch_tag(&model_path_buf);
        let executor = Self {
            model,
            slots,
            slot_tier,
            recurrent_pool,
            workspace: crate::qwen35::Qwen35Workspace::new(),
            num_slots,
            decode_graph_armed,
            decode_graph: None,
            paged_decode_meta: Vec::new(),
            batch_decode: None,
            kv_recall: false,
            recall_cfg: crate::recall::default_recall_config(),
            recall: (0..num_slots)
                .map(|_| crate::recall::CudaRecallState::default())
                .collect(),
            full_attn_kv: Some(full_attn_kv),
            kv_format,
            recall_tier: None,
            recall_keepalive: (0..num_slots).map(|_| Vec::new()).collect(),
            recall_budget_bytes: tier_budget_bytes,
            weights_epoch,
            disk_root: None,
            disk_budget: None,
            kv_pool_sized_pages,
            sidecar_page_key: std::collections::HashMap::new(),
            prefill_boundary_snapshot: (0..num_slots).map(|_| None).collect(),
            periodic_boundary_snapshots: (0..num_slots).map(|_| Vec::new()).collect(),
            dspark: dspark_head.map(|h| crate::qwen35::dspark::Qwen35DsparkExec::new(h, num_slots)),
            mtp: mtp_draft_tokens.map(|_| MtpExec {
                slots: (0..num_slots).map(|_| None).collect(),
                accepts: 0,
                rejects: 0,
                chains: 0,
            }),
        };
        cuda_startup_log(
            "executor.qwen35_executor_total",
            total_t0,
            format_args!("slots={num_slots} max_seq_len={max_seq_len}"),
        );
        Ok(executor)
    }

    /// Allocate the pool at an exact page count, no profiling.
    fn alloc_full_attn_kv_pool(
        model: &crate::qwen35::Qwen35Model,
        num_slots: usize,
        total_pool_pages: usize,
        kv_format: KVFormat,
    ) -> Result<PagedKVPool> {
        let num_full = model.config.num_full_attention_layers();
        let local_kv_heads = model.local_kv_heads();
        let head_dim = model.config.head_dim;
        let pool_token_budget = total_pool_pages * SUPPORTED_PAGE_SIZE;
        let pool_budget_bytes = PagedKVPool::budget_bytes_for_tokens(
            num_full,
            local_kv_heads,
            head_dim,
            pool_token_budget,
            kv_format,
        );
        let full_attn_kv = PagedKVPool::with_format(
            &model.ctx,
            num_full,
            local_kv_heads,
            head_dim,
            num_slots,
            pool_budget_bytes,
            kv_format,
        )?;
        ensure!(
            full_attn_kv.page_size == SUPPORTED_PAGE_SIZE,
            "Qwen3.6 full-attn paged pool page_size={} != {SUPPORTED_PAGE_SIZE}",
            full_attn_kv.page_size
        );
        Ok(full_attn_kv)
    }

    /// Drop the paged pool and trim the device pool once the enqueued frees
    /// complete, so a co-resident allocator never races an in-flight free.
    /// `ensure_kv_pool` re-acquires it.
    pub(crate) fn release_kv_pool(&mut self) -> Result<()> {
        let Some(pool) = self.full_attn_kv.as_ref() else {
            return Ok(());
        };
        let freed = pool.device_bytes();
        self.full_attn_kv = None;
        let event = self.model.ctx.stream.record_event(None)?;
        event.synchronize()?;
        self.model.ctx.trim_memory_pool()?;
        log::info!(
            "Qwen3.6 released full-attn KV pool: freed {}MB (agent-OPD writeback headroom)",
            freed >> 20
        );
        Ok(())
    }

    /// Re-acquire the pool at `kv_pool_sized_pages`; no-op if resident. Deliberately
    /// does NOT re-profile: `release_kv_pool` handed this VRAM to the co-resident
    /// student, so a profile here shrinks the pool every round until it hits the floor.
    pub(crate) fn ensure_kv_pool(&mut self) -> Result<()> {
        if self.full_attn_kv.is_some() {
            return Ok(());
        }
        let pool = Self::alloc_full_attn_kv_pool(
            &self.model,
            self.num_slots,
            self.kv_pool_sized_pages,
            self.kv_format,
        )?;
        let mb = pool.device_bytes() >> 20;
        // Construction warns once; the replay would otherwise be silent.
        let floor_pages = infer_seam::PROFILE_KV_TOKENS_FLOOR as usize / SUPPORTED_PAGE_SIZE;
        if self.kv_pool_sized_pages <= floor_pages {
            log::warn!(
                "Qwen3.6 re-acquired full-attn KV pool at the {}-token floor over {} slots \
                 ({mb}MB): admission stays capped and every longer prompt aborts until \
                 mem_fraction_static is raised",
                infer_seam::PROFILE_KV_TOKENS_FLOOR,
                self.num_slots,
            );
        }
        log::info!(
            "Qwen3.6 re-acquired full-attn KV pool: {mb}MB / {} pages (agent-OPD next-round \
             rollout)",
            self.kv_pool_sized_pages,
        );
        self.full_attn_kv = Some(pool);
        Ok(())
    }

    /// Boot-time decode-graph verdict log. Capture itself is lazy — one per slot on its
    /// first gated decode, so unused slots never pay capture/instantiation memory.
    pub(crate) fn warmup(&mut self) -> Result<()> {
        let warmup_t0 = Instant::now();
        let dense_t0 = Instant::now();
        let (warmed_shapes, warm_m) = self.model.warm_fp8_deepgemm_dense_prefill()?;
        cuda_startup_log(
            "executor.qwen35_warm_dense_deepgemm",
            dense_t0,
            format_args!("shapes={warmed_shapes} warm_m={warm_m}"),
        );
        if warmed_shapes > 0 {
            info!(
                "Qwen3.5 FP8 dense DeepGEMM warmed {warmed_shapes} projection shape(s) at M={warm_m}"
            );
        }
        let grouped_t0 = Instant::now();
        let (grouped_shapes, grouped_tokens, grouped_min_rows, grouped_max_rows) =
            self.model.warm_fp8_deepgemm_grouped_prefill()?;
        cuda_startup_log(
            "executor.qwen35_warm_grouped_deepgemm",
            grouped_t0,
            format_args!(
                "shapes={grouped_shapes} tokens={grouped_tokens} rows={grouped_min_rows}..{grouped_max_rows}"
            ),
        );
        if grouped_shapes > 0 {
            info!(
                "Qwen3.5 FP8 grouped DeepGEMM warmed {grouped_shapes} GEMM shape(s) at tokens<={grouped_tokens} rows={grouped_min_rows}..{grouped_max_rows}"
            );
        }
        if !crate::runtime_flags::qwen35_decode_graph() {
            info!(
                "Qwen3.5 whole-step decode graph disabled \
                 (set --qwen35-decode-graph to enable)"
            );
            cuda_startup_log(
                "executor.qwen35_warmup_total",
                warmup_t0,
                format_args!("graph=disabled"),
            );
            return Ok(());
        }
        if !self.model.tp.is_single() {
            info!(
                "Qwen3.5 whole-step decode graph disabled under tensor parallelism \
                 (world_size>1, NCCL collectives are not graph-capturable); \
                 using eager forward"
            );
            cuda_startup_log(
                "executor.qwen35_warmup_total",
                warmup_t0,
                format_args!("graph=tp_disabled"),
            );
            return Ok(());
        }
        if let Some(reason) = self.model.decode_graph_unsupported_reason() {
            info!("Qwen3.5 whole-step decode graph disabled: {reason}; using eager forward");
            cuda_startup_log(
                "executor.qwen35_warmup_total",
                warmup_t0,
                format_args!("graph=unsupported"),
            );
            return Ok(());
        }
        debug_assert!(self.decode_graph_armed);
        info!(
            "Qwen3.5 whole-step decode graph ARMED ({} slots; lazy capture per slot, \
             one eager warm run before each first capture; eager fallback on any failure)",
            self.num_slots
        );
        cuda_startup_log(
            "executor.qwen35_warmup_total",
            warmup_t0,
            format_args!("graph=armed"),
        );
        Ok(())
    }

    /// Per-rank L2 byte cap (`--kv-dram` ÷ world size). Pre-serve only (drops any
    /// existing entries).
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        self.recall_budget_bytes = bytes;
        self.slot_tier = KvTierStore::with_budget(bytes, BLOB_CHUNK_BYTES);
    }

    /// Attach NVMe spill (`--kv-disk`): ephemeral under `slot_tier`, durable for the
    /// recall tier (attached now if built, else stashed for `set_kv_recall`). Pre-serve
    /// only. The budget is a per-store cap, not a reservation — both stores are sparse
    /// mmaps, so disk is consumed only by actual spill.
    pub(crate) fn set_kv_tier_disk(
        &mut self,
        root: std::path::PathBuf,
        budget_bytes: usize,
    ) -> bool {
        self.disk_root = Some(root.clone());
        self.disk_budget = Some(budget_bytes);
        let recall_attached = match self.recall_tier.as_mut() {
            Some(tier) => {
                let page_bytes = tier.page_bytes();
                tier.load(
                    root.clone(),
                    budget_bytes,
                    page_bytes,
                    self.weights_epoch.clone(),
                    self.kv_format
                        .stable_tag()
                        .expect("persisted KV format must have a stable tag"),
                    self.model.tp.config().world_size,
                    self.model.tp.config().rank,
                )
            }
            None => false,
        };
        self.slot_tier
            .set_disk(root, budget_bytes, BLOB_CHUNK_BYTES)
            || recall_attached
    }

    /// Opt into the recall eviction/scoring cycle (`--kv-recall`): a working-set
    /// restriction on the SAME always-resident paged pool, with its L3 tier built
    /// lazily on the first enable.
    pub(crate) fn set_kv_recall(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            !(enabled && self.dspark.is_some()),
            "--kv-recall is not supported with --spec-type dspark (the verify \
             forward would race the recall eviction cycle)"
        );
        ensure!(
            !(enabled && self.model.tp.attn_cp_size() > 1),
            "--kv-recall is not supported with attn_cp>1 (rank-local recall scoring \
             diverges the cp group's collective schedule)"
        );
        self.kv_recall = enabled;
        if enabled && self.recall_tier.is_none() {
            // One entry == one pool page image (all `num_full` layers, K+V).
            let page_bytes = self
                .full_attn_kv
                .as_ref()
                .map(|p| p.storage_bytes_per_page())
                .ok_or_else(|| {
                    anyhow::anyhow!("--kv-recall: full-attn paged pool not allocated")
                })?;
            let mut tier = KvTierStore::with_budget(self.recall_budget_bytes, page_bytes);
            // Falls through to set_disk_durable on first run or epoch mismatch.
            if let (Some(root), Some(budget)) = (self.disk_root.as_ref(), self.disk_budget) {
                let format_tag = self
                    .kv_format
                    .stable_tag()
                    .expect("persisted KV format must have a stable tag");
                let rank = self.model.tp.config().rank;
                let world_size = self.model.tp.config().world_size;
                let loaded = tier.load(
                    root.clone(),
                    budget,
                    page_bytes,
                    self.weights_epoch.clone(),
                    format_tag,
                    world_size,
                    rank,
                );
                if !loaded {
                    tier.set_disk_durable(
                        root.clone(),
                        budget,
                        page_bytes,
                        self.weights_epoch.clone(),
                        format_tag,
                        world_size,
                        rank,
                    );
                }
            }
            self.recall_tier = Some(tier);
        }
        Ok(())
    }

    /// `Option` is `None` only after an OPD weight offload dropped the pool.
    fn full_attn_paged(&self) -> bool {
        self.full_attn_kv.is_some()
    }

    /// Gates the whole-step decode graph (persistent page table is BF16-only) and the
    /// DSpark batched draft.
    fn paged_kv_bf16(&self) -> bool {
        self.full_attn_kv
            .as_ref()
            .is_some_and(|p| p.format == KVFormat::BF16)
    }

    /// Actual pool page count, so the host admission pool mirrors the device 1:1.
    /// `None` only if the pool was dropped (OPD offload).
    pub(crate) fn full_attn_pool_pages(&self) -> Option<usize> {
        self.full_attn_kv.as_ref().map(|p| p.max_total_pages)
    }

    fn recall_active(&self) -> bool {
        self.kv_recall && self.recall_tier.is_some() && self.full_attn_kv.is_some()
    }

    /// One paged prefill row with no recall cycle: full attention over every resident
    /// page, no eviction/scoring/prefetch.
    fn prefill_row_paged_default(
        &mut self,
        row: &infer_plan::PrefillRow,
        position: u64,
        host_kv: &dyn KvPool,
    ) -> Result<(u32, Option<f32>)> {
        let slot = row.slot;
        {
            // A mismatch means an upstream restore left the two pools' cursors apart.
            let pool = self
                .full_attn_kv
                .as_ref()
                .expect("full_attn_kv present (full_attn_paged)");
            ensure!(
                pool.seq_len(slot) == row.start_pos,
                "Qwen3.6 default-paged prefill: device pool seq_len {} != start_pos {} for slot {}",
                pool.seq_len(slot),
                row.start_pos,
                slot
            );
        }
        self.mirror_host_slot(host_kv, slot, row.start_pos + row.tokens.len())?;
        let (meta, cp) = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv present");
            let len = row.tokens.len();
            let cp_size = self.model.tp.attn_cp_size();
            // 2D ring prefill (T3.2b Part D): one ring pass over the whole
            // prompt. Each rank preps its q-slice, rotates KV around the cp
            // ring, and scatters its owned pages (block-cyclic). DSpark taps
            // need the full chunk on every rank. The decision is a pure
            // function of the (rank-identical) plan, so the cp group's
            // collective schedule stays lockstep.
            if self.two_d_engaged() && self.dspark.is_none() {
                let per = len.div_ceil(cp_size);
                let slices: Vec<(usize, usize)> = (0..cp_size)
                    .map(|p| (p * per, ((p + 1) * per).min(len) - p * per))
                    .collect();
                let (off, my_len) = slices[self.model.tp.attn_cp_rank()];
                let meta = crate::loader::PageMeta::for_ring_prefill(
                    &self.model.ctx,
                    row.start_pos + off,
                    my_len,
                )?;
                // Local shard pages in local-index order (entry j backs global
                // page cp_rank + j*cp).
                let kv_indices: Vec<i32> =
                    pool.page_indices(slot).iter().map(|&p| p as i32).collect();
                let q_pos: Vec<usize> = (0..my_len).map(|i| row.start_pos + off + i).collect();
                let k_pos: Vec<Vec<usize>> = slices
                    .iter()
                    .map(|&(o, l)| (0..l).map(|i| row.start_pos + o + i).collect())
                    .collect();
                let q_pos_f32 = crate::ops::upload_f32(
                    &self.model.ctx,
                    &q_pos.iter().map(|&p| p as f32).collect::<Vec<_>>(),
                )?;
                let k_pos_f32 = k_pos
                    .iter()
                    .map(|kp| {
                        crate::ops::upload_f32(
                            &self.model.ctx,
                            &kp.iter().map(|&p| p as f32).collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let cp = crate::qwen35::Qwen35CpPrefill {
                    slices,
                    pad: per,
                    kv_indices: crate::ops::upload_i32(&self.model.ctx, &kv_indices)?,
                    q_pos,
                    k_pos,
                    q_pos_f32,
                    k_pos_f32,
                };
                (meta, Some(cp))
            } else {
                let meta = crate::loader::PageMeta::for_slot(
                    &self.model.ctx,
                    pool,
                    slot,
                    row.start_pos,
                    len,
                )?;
                (meta, None)
            }
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            dspark,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: None,
            cp,
            cp_decode: None,
        };
        let Some(ds) = dspark.as_mut() else {
            return model.forward_tokens_recall(
                &mut slots[slot],
                workspace,
                &row.tokens,
                row.start_pos,
                &row.params,
                position,
                penalty_of(&row.penalty_history, row.penalty_prompt_len),
                &mut rc,
            );
        };
        ds.taps.prepare(
            ds.head.target_layer_ids(),
            model.config.hidden_size,
            row.tokens.len(),
        );
        let (token, logprob) = model.forward_tokens_recall_tapped(
            &mut slots[slot],
            workspace,
            &row.tokens,
            row.start_pos,
            &row.params,
            position,
            penalty_of(&row.penalty_history, row.penalty_prompt_len),
            &mut rc,
            Some(&mut ds.taps),
        )?;
        if ds.slots[slot].is_none() {
            ds.slots[slot] = Some(crate::qwen35::dspark::Qwen35DsparkSlotState::new(
                &model.ctx, &ds.head,
            )?);
        }
        let df = ds.slots[slot].as_mut().expect("built above");
        if df.ctx_end != row.start_pos {
            // Gap (prefix-cache / sidecar-restored prefix): rebase at the suffix rather
            // than degrade to plain decode — approximate only for the full-attn layer.
            df.rebase(row.start_pos);
        }
        model.dspark_tap_features(&ds.head, &mut ds.taps, &mut ds.scratch)?;
        model.dspark_append_ctx(
            &ds.head,
            df,
            &mut ds.scratch,
            0,
            row.tokens.len(),
            row.start_pos,
        )?;
        let is_final = row.start_pos + row.tokens.len() == row.total_tokens;
        df.pending = (is_final && df.ctx_end == row.total_tokens).then_some(token);
        Ok((token, logprob))
    }

    /// One paged decode row with no recall cycle: append and attend the full resident
    /// page set, no tier I/O.
    fn decode_row_paged_default(
        &mut self,
        row: &DecodeRow,
        position: u64,
        host_kv: &dyn KvPool,
    ) -> Result<(u32, Option<f32>)> {
        let slot = row.slot;
        {
            let pool = self
                .full_attn_kv
                .as_ref()
                .expect("full_attn_kv present (full_attn_paged)");
            // Catch a slot whose pool seq_len drifted from the engine's kv_seq_len
            // here,
            // not via `PageMeta::for_slot` math downstream.
            ensure!(
                pool.seq_len(slot) == row.kv_seq_len,
                "Qwen3.6 default-paged decode: pool seq_len {} != kv_seq_len {} for slot {}",
                pool.seq_len(slot),
                row.kv_seq_len,
                slot
            );
        }
        self.mirror_host_slot(host_kv, slot, row.kv_seq_len + 1)?;
        let meta = {
            let pool = self
                .full_attn_kv
                .as_ref()
                .expect("full_attn_kv present (full_attn_paged)");
            if self.two_d_engaged() {
                Self::sharded_decode_meta(
                    &self.model.ctx,
                    pool,
                    slot,
                    row.kv_seq_len,
                    self.model.tp.attn_cp_rank(),
                    self.model.tp.attn_cp_size(),
                )?
            } else {
                crate::loader::PageMeta::for_slot(&self.model.ctx, pool, slot, row.kv_seq_len, 1)?
            }
        };
        // B2 CP decode (T3.1): head-shard across the cp group once the KV is
        // long enough to amortize the cp all-reduce. DSpark taps need the full
        // head set on every rank, like prefill CP. Under 2D the pool is already
        // cp-sharded and the GDN pair's trade is length-independent (recurrent
        // state is O(1) per step), so it engages from the first decode token.
        let cp_decode = if self.dspark.is_none()
            && (self.two_d_engaged() || row.kv_seq_len + 1 >= CP_DECODE_MIN_KV_TOKENS)
        {
            self.model.cp_decode_handle()
        } else {
            None
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: None,
            cp: None,
            cp_decode,
        };
        model.forward_tokens_recall(
            &mut slots[slot],
            workspace,
            &[row.last_token],
            row.kv_seq_len,
            &row.params,
            position,
            penalty_of(&row.penalty_history, row.penalty_prompt_len),
            &mut rc,
        )
    }

    /// 2D decode page meta: covers only this cp shard's local pages (logical
    /// page `i` on shard `i % cp`). `kv_lens`/`max_kv_len` are the shard-local
    /// token count; the new token's page appears only on the owning shard's
    /// table, so a non-owner's prep write is skipped and its FA3 partial
    /// covers exactly its resident pages. BF16 only (2D inherits the bf16
    /// mutex).
    fn sharded_decode_meta(
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        kv_seq_len: usize,
        cp_rank: usize,
        cp_size: usize,
    ) -> Result<PageMeta> {
        ensure!(
            pool.format == KVFormat::BF16,
            "2D decode requires the BF16 KV pool (got {:?})",
            pool.format
        );
        let total_len = kv_seq_len + 1;
        let page_size = pool.page_size;
        ensure!(
            pool.seq_len(slot) == total_len,
            "sharded decode meta: pool seq_len {} != total_len {total_len} for slot {slot}",
            pool.seq_len(slot)
        );
        let global_pages = total_len.div_ceil(page_size);
        let local_pages: Vec<i32> = pool.page_indices(slot).iter().map(|&p| p as i32).collect();
        let local_num_pages = local_pages.len();
        // The new token lands in the last global page; only its owner writes it.
        let shard = infer_seam::ShardSpec::new(cp_rank, cp_size);
        let owns_last = shard.owns_page(global_pages - 1);
        let overshoot = global_pages * page_size - total_len;
        let local_last_fill = if owns_last {
            // The last page is partial unless total_len is page-aligned.
            page_size - overshoot
        } else {
            page_size
        };
        let local_token_count = (local_num_pages)
            .saturating_sub(1)
            .saturating_mul(page_size)
            + if local_num_pages == 0 {
                0
            } else {
                local_last_fill
            };
        // A shard with no resident pages (total_len < page_size * cp) still
        // needs a 1-entry table: FA3 dereferences page_table[0], and kv_lens=0
        // bounds the read to zero tokens.
        let kv_indices_dev = crate::ops::upload_i32(ctx, &local_pages)?;
        let (page_table_rect, stride) = if local_num_pages == 0 {
            (crate::ops::upload_i32(ctx, &[0])?, 1usize)
        } else {
            (kv_indices_dev.clone(), local_num_pages)
        };
        let zero = crate::ops::upload_i32(ctx, &[0])?;
        Ok(PageMeta {
            q_indptr: crate::ops::upload_i32(ctx, &[0, 1])?,
            kv_indptr: crate::ops::upload_i32(ctx, &[0, local_num_pages as i32])?,
            kv_indices: kv_indices_dev,
            kv_last_page_len: crate::ops::upload_i32(ctx, &[local_last_fill as i32])?,
            page_table_offsets: zero.clone(),
            start_positions: crate::ops::upload_i32(ctx, &[kv_seq_len as i32])?,
            positions: crate::ops::upload_i32(ctx, &[kv_seq_len as i32])?,
            q_offsets: vec![0, 1],
            page_offsets: vec![0, local_num_pages],
            kv_lens: vec![local_token_count],
            kv_lens_dev: crate::ops::upload_i32(ctx, &[local_token_count as i32])?,
            page_table_rect,
            page_table_stride: stride,
            seq_len: 1,
            total_q: 1,
            num_pages: local_num_pages,
            batch: 1,
            start_pos: kv_seq_len,
            new_token_rows: None,
            prefix_token_rows: None,
            quant_decode_meta: None,
            seqlen_k_capture: None,
            write_kv: i32::from(owns_last),
        })
    }

    /// One MTP spec-decode row: the spec step when the slot is seeded, else a warm
    /// forward that captures the seed. Emits 1..=depth+1 tokens. Greedy rows verify by
    /// argmax match (token-exact to no-spec greedy), sampling rows by rejection
    /// sampling.
    fn mtp_decode_row(
        &mut self,
        row: &DecodeRow,
        host_kv: &mut dyn KvPool,
    ) -> Result<Vec<SlotToken>> {
        ensure!(
            row.slot < self.num_slots,
            "decode slot {} outside Qwen3.5 executor slots {}",
            row.slot,
            self.num_slots
        );
        let depth = self.model.spec_draft_tokens().max(1);
        // Seeded iff the stored pending matches the token the scheduler will feed.
        // A logprobs capture vetoes spec: the verify commits tokens without full
        // per-position distributions, so the row stays on the warm path.
        let seeded = row.params.top_logprobs.is_none()
            && speculative_chain_fits(row.kv_seq_len, depth, self.model.max_seq_len())
            && matches!(
                self.mtp.as_ref().and_then(|m| m.slots[row.slot].as_ref()),
                Some(s) if s.pending == row.last_token
            );
        if !seeded {
            let (token, logprob) = self.mtp_warm_decode_row(row, host_kv)?;
            return Ok(vec![SlotToken {
                slot: row.slot,
                token,
                logprob,
                top_logprobs: self.take_top_logprobs(&row.params),
                finish: None,
            }]);
        }
        self.mtp_spec_row(row, depth, host_kv)
    }

    /// Warm step: one forward that also captures the seed (pending token + hidden) for
    /// the next spec step, and returns the pending token as this step's output.
    fn mtp_warm_decode_row(
        &mut self,
        row: &DecodeRow,
        host_kv: &mut dyn KvPool,
    ) -> Result<(u32, Option<f32>)> {
        let slot = row.slot;
        let start = row.kv_seq_len;
        if self.full_attn_paged() {
            {
                let pool = self.full_attn_kv.as_ref().expect("paged (gated)");
                ensure!(
                    pool.seq_len(slot) == start,
                    "MTP warm decode: pool seq_len {} != start {start} for slot {slot}",
                    pool.seq_len(slot),
                );
            }
            set_host_slot_to(host_kv, slot, start + 1)?;
            self.mirror_host_slot(host_kv, slot, start + 1)?;
            let meta = {
                let pool = self.full_attn_kv.as_ref().expect("paged (gated)");
                crate::loader::PageMeta::for_slot(&self.model.ctx, pool, slot, start, 1)?
            };
            let Self {
                model,
                slots,
                workspace,
                full_attn_kv,
                mtp,
                ..
            } = self;
            let pool = full_attn_kv.as_mut().expect("paged (gated)");
            let mut rc = crate::qwen35::Qwen35RecallForward {
                pool,
                meta: &meta,
                layer0_query: None,
                cp: None,
                cp_decode: None,
            };
            let (logits, dims, hidden) = model.forward_tokens_with_hidden(
                &mut slots[slot],
                workspace,
                &[row.last_token],
                start,
                Some(&mut rc),
            )?;
            let vocab = dims[1];
            let mut spec = model.new_spec_slot_state()?;
            let (pending, logprob) = if row.params.is_raw_argmax() {
                let tok = crate::ops::argmax_row_into(
                    &model.ctx,
                    &logits,
                    dims[0] - 1,
                    vocab,
                    spec.argmax_scratch_mut(),
                )?;
                (tok, None)
            } else {
                // Sampled seed so the pending token is policy-distributed.
                crate::executor::sample_cuda_token_captured(
                    &model.ctx,
                    &logits,
                    &row.params,
                    start.saturating_add(1) as u64,
                    spec.argmax_scratch_mut(),
                    penalty_of(&row.penalty_history, row.penalty_prompt_len),
                    &mut workspace.top_logprobs,
                )?
            };
            mtp.as_mut().expect("mtp (gated)").slots[slot] = Some(MtpSlotState {
                spec,
                pending,
                hidden,
            });
            Ok((pending, logprob))
        } else {
            let Self {
                model,
                slots,
                workspace,
                mtp,
                ..
            } = self;
            let (logits, dims, hidden) = model.forward_tokens_with_hidden(
                &mut slots[slot],
                workspace,
                &[row.last_token],
                start,
                None,
            )?;
            let vocab = dims[1];
            let mut spec = model.new_spec_slot_state()?;
            let (pending, logprob) = if row.params.is_raw_argmax() {
                let tok = crate::ops::argmax_row_into(
                    &model.ctx,
                    &logits,
                    dims[0] - 1,
                    vocab,
                    spec.argmax_scratch_mut(),
                )?;
                (tok, None)
            } else {
                // Sampled seed so the pending token is policy-distributed.
                crate::executor::sample_cuda_token_captured(
                    &model.ctx,
                    &logits,
                    &row.params,
                    start.saturating_add(1) as u64,
                    spec.argmax_scratch_mut(),
                    penalty_of(&row.penalty_history, row.penalty_prompt_len),
                    &mut workspace.top_logprobs,
                )?
            };
            mtp.as_mut().expect("mtp (gated)").slots[slot] = Some(MtpSlotState {
                spec,
                pending,
                hidden,
            });
            Ok((pending, logprob))
        }
    }

    /// Spec step: draft a depth-token chain, verify in ONE trunk forward, accept the
    /// longest matching prefix. Partial accept rolls the trunk linear state back and
    /// truncates the paged pool. Returns the accepted drafts plus the bonus token.
    fn mtp_spec_row(
        &mut self,
        row: &DecodeRow,
        depth: usize,
        host_kv: &mut dyn KvPool,
    ) -> Result<Vec<SlotToken>> {
        let slot = row.slot;
        let start = row.kv_seq_len;
        // Verify appends depth+1 and the engine pre-allocated one; grow the host pool
        // by
        // the rest, truncate both back on partial accept.
        if self.full_attn_paged() {
            {
                let pool = self.full_attn_kv.as_ref().expect("paged (gated)");
                ensure!(
                    pool.seq_len(slot) == start,
                    "MTP spec decode: pool seq_len {} != start {start} for slot {slot}",
                    pool.seq_len(slot),
                );
            }
            set_host_slot_to(host_kv, slot, start + depth + 1)?;
            self.mirror_host_slot(host_kv, slot, start + depth + 1)?;
        }
        let meta = if self.full_attn_paged() {
            let pool = self.full_attn_kv.as_ref().expect("paged (gated)");
            Some(crate::loader::PageMeta::for_slot(
                &self.model.ctx,
                pool,
                slot,
                start,
                depth + 1,
            )?)
        } else {
            None
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            mtp,
            ..
        } = self;
        let (emitted, next_pending, next_hidden) = {
            let mtp_exec = mtp.as_mut().expect("mtp (gated)");
            let st = mtp_exec.slots[slot].as_mut().expect("seeded (gated)");
            let mut rc = full_attn_kv
                .as_mut()
                .map(|pool| crate::qwen35::Qwen35RecallForward {
                    pool,
                    meta: meta.as_ref().expect("paged (gated)"),
                    layer0_query: None,
                    cp: None,
                    cp_decode: None,
                });
            model.spec_step(
                &mut slots[slot],
                &mut st.spec,
                workspace,
                st.pending,
                &st.hidden,
                start,
                depth,
                &row.params,
                rc.as_mut(),
            )?
        };
        let truncate_to = (emitted.len() < depth + 1).then(|| start + emitted.len());
        let mtp_exec = mtp.as_mut().expect("mtp (gated)");
        let accepted = emitted.len() - 1;
        mtp_exec.accepts += accepted;
        mtp_exec.rejects += depth - accepted;
        mtp_exec.chains += 1;
        if let Some(st) = mtp_exec.slots[slot].as_mut() {
            st.pending = next_pending;
            st.hidden = next_hidden;
        }
        if let Some(len) = truncate_to
            && let Some(pool) = full_attn_kv.as_mut()
        {
            host_kv.truncate_slot(slot, len)?;
            let need = len.div_ceil(pool.page_size);
            pool.mirror_slot(slot, &host_kv.page_indices(slot)[..need], len)?;
        }
        Ok(emitted
            .into_iter()
            .map(|(token, logprob)| SlotToken {
                slot,
                token,
                logprob,
                // Spec is vetoed for logprobs requests, so no capture here.
                top_logprobs: Vec::new(),
                finish: None,
            })
            .collect())
    }

    /// Warm-decode one DSpark row with tap capture. Non-paged rows run plain and clear
    /// the draft state, so speculation re-seeds at the next paged step.
    fn dspark_warm_decode_row(
        &mut self,
        row: &DecodeRow,
        position: u64,
        host_kv: &mut dyn KvPool,
    ) -> Result<(u32, Option<f32>)> {
        let slot = row.slot;
        if !self.full_attn_paged() {
            if let Some(df) = self.dspark.as_mut().and_then(|ds| ds.slots[slot].as_mut()) {
                df.pending = None;
            }
            return self.submit_decode_row(row, false, host_kv);
        }
        {
            let pool = self.full_attn_kv.as_ref().expect("paged (checked)");
            ensure!(
                pool.seq_len(slot) == row.kv_seq_len,
                "Qwen3.6 dspark warm decode: pool seq_len {} != kv_seq_len {} for slot {}",
                pool.seq_len(slot),
                row.kv_seq_len,
                slot
            );
        }
        set_host_slot_to(host_kv, slot, row.kv_seq_len + 1)?;
        self.mirror_host_slot(host_kv, slot, row.kv_seq_len + 1)?;
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("paged (checked)");
            crate::loader::PageMeta::for_slot(&self.model.ctx, pool, slot, row.kv_seq_len, 1)?
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            dspark,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("paged (checked)");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: None,
            cp: None,
            cp_decode: None,
        };
        let ds = dspark.as_mut().expect("dspark warm without dspark");
        // Gap (whole-slot promote / restored prefix): rebase and rebuild the
        // suffix-only ctx from this step.
        if let Some(df) = ds.slots[slot].as_mut()
            && df.ctx_end != row.kv_seq_len
        {
            df.rebase(row.kv_seq_len);
        }
        let taps = if ds.slots[slot].is_some() {
            ds.taps
                .prepare(ds.head.target_layer_ids(), model.config.hidden_size, 1);
            Some(&mut ds.taps)
        } else {
            None
        };
        let (token, logprob) = model.forward_tokens_recall_tapped(
            &mut slots[slot],
            workspace,
            &[row.last_token],
            row.kv_seq_len,
            &row.params,
            position,
            penalty_of(&row.penalty_history, row.penalty_prompt_len),
            &mut rc,
            taps,
        )?;
        if ds.slots[slot].is_some() {
            model.dspark_tap_features(&ds.head, &mut ds.taps, &mut ds.scratch)?;
            let df = ds.slots[slot].as_mut().expect("checked above");
            model.dspark_append_ctx(&ds.head, df, &mut ds.scratch, 0, 1, row.kv_seq_len)?;
            df.pending = Some(token);
        }
        Ok((token, logprob))
    }

    /// One DSpark tick: draft per slot, verify EVERY chain in ONE trunk forward, then
    /// accept/roll back per row. Greedy stays token-exact to no-spec decode. A row
    /// falls
    /// back to its own warm step without stopping the rest of the tick.
    fn dspark_decode_batch(
        &mut self,
        decode_rows: &[DecodeRow],
        host_kv: &mut dyn KvPool,
    ) -> Result<Vec<SlotToken>> {
        let mut out: Vec<Vec<SlotToken>> = (0..decode_rows.len()).map(|_| Vec::new()).collect();
        let mut batch: Vec<DsparkChain> = Vec::with_capacity(decode_rows.len());
        let mut seeded = Vec::with_capacity(decode_rows.len());
        for row in decode_rows {
            ensure!(
                row.slot < self.num_slots,
                "decode slot {} outside Qwen3.5 executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                self.slots[row.slot].seq_len() == row.kv_seq_len,
                "Qwen3.5 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
                self.slots[row.slot].seq_len(),
                row.kv_seq_len,
                row.slot
            );
            // The verify appends the whole chain, so it must fit the trunk cap.
            let ds = self
                .dspark
                .as_ref()
                .expect("dspark_decode_batch without dspark");
            // A logprobs capture vetoes spec (no full per-position distributions
            // in the verify); the row falls to its warm step below.
            seeded.push(
                row.params.top_logprobs.is_none()
                    && self.full_attn_paged()
                    && speculative_chain_fits(
                        row.kv_seq_len,
                        ds.head.block_size(),
                        self.model.max_seq_len(),
                    )
                    && matches!(
                        ds.slots[row.slot].as_ref(),
                        Some(s) if s.pending == Some(row.last_token) && s.ctx_end == row.kv_seq_len
                    ),
            );
        }

        // Draft: no trunk/pool state touched, and weight-bound at block rows, so every
        // seeded slot shares one forward.
        let mut pre: Vec<Option<Vec<u32>>> = vec![None; decode_rows.len()];
        let mut idx: Vec<usize> = (0..decode_rows.len()).filter(|&i| seeded[i]).collect();
        if idx.len() >= 2 && decode_rows.iter().all(|r| r.params.is_greedy()) {
            idx.sort_by_key(|&i| decode_rows[i].slot);
            let anchors: Vec<u32> = idx.iter().map(|&i| decode_rows[i].last_token).collect();
            let starts: Vec<usize> = idx.iter().map(|&i| decode_rows[i].kv_seq_len).collect();
            let Self { model, dspark, .. } = self;
            let ds = dspark.as_mut().expect("dspark");
            let mut pick = vec![false; ds.slots.len()];
            for &i in &idx {
                pick[decode_rows[i].slot] = true;
            }
            let mut dfs: Vec<&mut crate::qwen35::dspark::Qwen35DsparkSlotState> = ds
                .slots
                .iter_mut()
                .enumerate()
                .filter(|(s, _)| pick[*s])
                .map(|(_, st)| st.as_mut().expect("seeded slot"))
                .collect();
            let sp: Vec<&SamplingParams> = idx.iter().map(|&i| &decode_rows[i].params).collect();
            let chains = model.dspark_draft_blocks(
                &ds.head,
                &mut dfs,
                &mut ds.scratch,
                &anchors,
                &starts,
                &sp,
            )?;
            for (n, &i) in idx.iter().enumerate() {
                pre[i] = Some(chains[n].clone());
            }
        }

        for (i, row) in decode_rows.iter().enumerate() {
            let start = row.kv_seq_len;
            let drafted = match pre[i].take() {
                Some(chain) => {
                    let ds = self.dspark.as_ref().expect("dspark");
                    let partial_ctx = ds.slots[row.slot].as_ref().expect("seeded").ctx_base > 0;
                    Some((chain, partial_ctx))
                }
                None => seeded[i]
                    .then(|| {
                        let Self { model, dspark, .. } = self;
                        let ds = dspark.as_mut().expect("dspark");
                        let df = ds.slots[row.slot].as_mut().expect("seeded slot");
                        let partial_ctx = df.ctx_base > 0;
                        model
                            .dspark_draft_block(
                                &ds.head,
                                df,
                                &mut ds.scratch,
                                row.last_token,
                                start,
                                &row.params,
                            )
                            .map(|chain| (chain, partial_ctx))
                    })
                    .transpose()?,
            }
            // A bare-anchor chain stays in the batched verify as one decode row
            // (min_verify_len=1); a per-slot fallback would cost a serial trunk
            // forward.
            // Sampled chains keep the fallback: the rejection kernel's depth-0 path is
            // unexercised.
            .filter(|(chain, _)| chain.len() >= 2 || row.params.is_greedy());
            let Some((chain, partial_ctx)) = drafted else {
                let (token, logprob) =
                    self.dspark_warm_decode_row(row, start.saturating_add(1) as u64, host_kv)?;
                out[i] = vec![SlotToken {
                    slot: row.slot,
                    token,
                    logprob,
                    top_logprobs: self.take_top_logprobs(&row.params),
                    finish: None,
                }];
                continue;
            };
            batch.push(DsparkChain {
                out: i,
                slot: row.slot,
                start,
                row0: 0,
                chain,
                partial_ctx,
            });
        }
        if batch.is_empty() {
            return Ok(out.into_iter().flatten().collect());
        }
        let mut total_rows = 0usize;
        for c in &mut batch {
            c.row0 = total_rows;
            total_rows += c.chain.len();
        }
        let chains: Vec<u32> = batch.iter().flat_map(|c| c.chain.iter().copied()).collect();

        // Snapshot every trunk's linear state as the partial-accept rollback base.
        for c in &batch {
            if self.dspark.as_ref().expect("dspark").spec[c.slot].is_none() {
                let st = self.model.new_spec_slot_state()?;
                self.dspark.as_mut().expect("dspark").spec[c.slot] = Some(st);
            }
        }
        {
            let Self {
                model,
                slots,
                dspark,
                ..
            } = self;
            let ds = dspark.as_mut().expect("dspark");
            let mut pick = vec![false; slots.len()];
            for c in &batch {
                pick[c.slot] = true;
            }
            let bytes = model.linear_state_bytes();
            let (mut gdr, mut conv) = ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()));
            for ((_, slot), (_, spec)) in slots
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| pick[*i])
                .zip(ds.spec.iter_mut().enumerate().filter(|(i, _)| pick[*i]))
            {
                spec.as_mut()
                    .expect("built above")
                    .linear_state_addrs(&model.ctx, slot, bytes, &mut gdr, &mut conv)?;
            }
            model.batched_copy(&mut ds.copy, &gdr.0, &gdr.1, &[bytes.0])?;
            model.batched_copy(&mut ds.copy, &conv.0, &conv.1, &[bytes.1])?;
        }

        // All-or-nothing: seq_lens advance only on success, so any failure must give
        // every reserved row back.
        let logits = match self.dspark_verify_forward(&batch, &chains, total_rows, host_kv) {
            Ok(logits) => logits,
            Err(e) => {
                for c in &batch {
                    if host_kv.seq_len(c.slot) > c.start {
                        host_kv.truncate_slot(c.slot, c.start)?;
                    }
                    self.mirror_host_slot(host_kv, c.slot, c.start)?;
                }
                return Err(e);
            }
        };

        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            dspark,
            ..
        } = self;
        let ds = dspark.as_mut().expect("dspark");
        // The tap fc projection is batch-wide; only the K/V append is per slot.
        model.dspark_tap_features(&ds.head, &mut ds.taps, &mut ds.scratch)?;
        // One argmax over every chain's rows: the accept scan is host arithmetic from
        // here, so the loop below adds no device syncs.
        let argmax = match decode_rows.iter().any(|r| r.params.is_greedy()) {
            true => model.argmax_rows(&logits, &mut ds.scratch)?,
            false => Vec::new(),
        };

        // Nothing in this loop reads the trunk linear state or `seq_len`, so the greedy
        // rollback batches below.
        let mut rollback: Vec<(usize, usize, usize)> = Vec::with_capacity(batch.len());
        for c in &batch {
            let params = &decode_rows[c.out].params;
            let spec = ds.spec[c.slot].as_mut().expect("built above");
            let (emitted, bonus, k) = if params.is_greedy() {
                // Greedy: no behavior logprob.
                let (tokens, bonus, k) = model.dspark_accept_commit(&c.chain, &argmax, c.row0)?;
                if k + 1 < c.chain.len() {
                    rollback.push((c.slot, c.start, k));
                }
                (
                    tokens.into_iter().map(|t| (t, None)).collect::<Vec<_>>(),
                    bonus,
                    k,
                )
            } else {
                let df = ds.slots[c.slot].as_mut().expect("seeded slot");
                model.dspark_accept_commit_sampled(
                    &mut slots[c.slot],
                    spec,
                    workspace,
                    &ds.head,
                    df,
                    &mut ds.scratch,
                    &c.chain,
                    &logits,
                    c.row0,
                    c.start,
                    params,
                )?
            };
            // Draft logits live in the SLOT: a tick drafts every row before
            // verifying any, so a shared buffer would pair the wrong slot.
            let df = ds.slots[c.slot].as_mut().expect("seeded slot");
            if k + 1 < c.chain.len() {
                let len = c.start + k + 1;
                let pool = full_attn_kv.as_mut().expect("paged (gated by seeded)");
                host_kv.truncate_slot(c.slot, len)?;
                let need = len.div_ceil(pool.page_size);
                pool.mirror_slot(c.slot, &host_kv.page_indices(c.slot)[..need], len)?;
            }
            model.dspark_append_ctx(&ds.head, df, &mut ds.scratch, c.row0, k + 1, c.start)?;
            df.pending = Some(bonus);
            ds.accepts += k;
            ds.rejects += c.chain.len() - 1 - k;
            ds.chains += 1;
            ds.partial_ctx_chains += usize::from(c.partial_ctx);
            out[c.out] = emitted
                .into_iter()
                .map(|(token, logprob)| SlotToken {
                    slot: c.slot,
                    token,
                    logprob,
                    // Spec is vetoed for logprobs requests, so no capture here.
                    top_logprobs: Vec::new(),
                    finish: None,
                })
                .collect();
        }
        if !rollback.is_empty() {
            rollback.sort_by_key(|r| r.0);
            let mut pick = vec![false; slots.len()];
            for r in &rollback {
                pick[r.0] = true;
            }
            let mut rolls: Vec<crate::qwen35::dspark::DsparkRollback<'_>> = slots
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| pick[*i])
                .zip(ds.spec.iter_mut().enumerate().filter(|(i, _)| pick[*i]))
                .zip(rollback.iter())
                .map(
                    |(((_, slot), (_, spec)), r)| crate::qwen35::dspark::DsparkRollback {
                        slot,
                        spec: spec.as_mut().expect("spec state built above"),
                        start_pos: r.1,
                        k: r.2,
                    },
                )
                .collect();
            model.dspark_rollback_batch(
                &mut rolls,
                &mut ds.replay_tables,
                &mut ds.copy,
                workspace,
            )?;
        }
        Ok(out.into_iter().flatten().collect())
    }

    /// Chain `i` owns logits rows `[c.row0, +len)`. The caller rolls the pool back on
    /// any error.
    fn dspark_verify_forward(
        &mut self,
        batch: &[DsparkChain],
        chains: &[u32],
        total_rows: usize,
        host_kv: &mut dyn KvPool,
    ) -> Result<cuda_kernels::prelude::HiddenStates> {
        for c in batch {
            {
                let pool = self.full_attn_kv.as_ref().expect("paged (gated by seeded)");
                ensure!(
                    pool.seq_len(c.slot) == c.start,
                    "Qwen3.6 dspark verify: pool seq_len {} != start {} for slot {}",
                    pool.seq_len(c.slot),
                    c.start,
                    c.slot
                );
            }
            set_host_slot_to(host_kv, c.slot, c.start + c.chain.len())?;
            self.mirror_host_slot(host_kv, c.slot, c.start + c.chain.len())?;
        }
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            dspark,
            ..
        } = self;
        let ds = dspark.as_mut().expect("dspark");
        let pool = full_attn_kv.as_mut().expect("paged (gated by seeded)");
        let rows: Vec<_> = batch
            .iter()
            .map(|c| (c.slot, c.start, c.chain.len()))
            .collect();
        let meta = crate::loader::PageMeta::for_rows(&model.ctx, pool, &rows)?;
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: None,
            cp: None,
            cp_decode: None,
        };
        ds.taps.prepare(
            ds.head.target_layer_ids(),
            model.config.hidden_size,
            total_rows,
        );
        let mut free_slots: Vec<Option<&mut crate::qwen35::Qwen35SlotState>> =
            slots.iter_mut().map(Some).collect();
        let mut free_caps: Vec<Option<&mut crate::qwen35::Qwen35LinearCapture>> = ds
            .spec
            .iter_mut()
            .map(|s| s.as_mut().map(|st| &mut st.capture))
            .collect();
        let mut fwd: Vec<crate::qwen35::LinearRow<'_>> = batch
            .iter()
            .map(|c| crate::qwen35::LinearRow {
                slot: free_slots[c.slot]
                    .take()
                    .expect("one row per slot per tick"),
                len: c.chain.len(),
                capture: free_caps[c.slot].take(),
            })
            .collect();
        model.dspark_verify_logits(&mut fwd, workspace, chains, &mut rc, &mut ds.taps)
    }

    /// One prefill row over the recall pool (`--kv-recall`) — the ONLY place the whole
    /// recall cycle runs: decode never recalls, prefetch happens only here. After
    /// return,
    /// `recall[slot].recall_pages()` is the immutable working set for decode.
    fn prefill_row_recall(
        &mut self,
        row: &infer_plan::PrefillRow,
        position: u64,
        host_kv: &mut dyn KvPool,
    ) -> Result<(u32, Option<f32>)> {
        let slot = row.slot;
        let cfg = self.recall_cfg;
        self.mirror_host_slot(host_kv, slot, row.start_pos + row.tokens.len())?;
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv present");
            crate::loader::PageMeta::for_slot(
                &self.model.ctx,
                pool,
                slot,
                row.start_pos,
                row.tokens.len(),
            )?
        };
        // `rc` carries back the layer-0 query used for scoring below.
        let (token, layer0_query) = {
            let Self {
                model,
                slots,
                workspace,
                full_attn_kv,
                ..
            } = self;
            let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
            let mut rc = crate::qwen35::Qwen35RecallForward {
                pool,
                meta: &meta,
                layer0_query: Some(Vec::new()),
                cp: None,
                cp_decode: None,
            };
            let token = model.forward_tokens_recall(
                &mut slots[slot],
                workspace,
                &row.tokens,
                row.start_pos,
                &row.params,
                position,
                penalty_of(&row.penalty_history, row.penalty_prompt_len),
                &mut rc,
            )?;
            (token, rc.layer0_query.expect("opted in above"))
        };

        let cache_len = row.start_pos + row.tokens.len();
        let ps = self.full_attn_kv.as_ref().expect("full_attn_kv").page_size;
        ensure!(
            cfg.n_init.is_multiple_of(ps)
                && cfg.n_local.is_multiple_of(ps)
                && cfg.l_bs.is_multiple_of(ps),
            "KV-recall config (n_init {}, n_local {}, l_bs {}) must be multiples of page_size {}",
            cfg.n_init,
            cfg.n_local,
            cfg.l_bs,
            ps
        );

        // `allow_prefetch=true` lets tier-resident blocks re-enter the working set.
        let num_q_heads = self.model.local_q_heads();
        let num_kv_heads = self.model.local_kv_heads();
        let head_dim = self.model.config.head_dim;
        let (evict_pages, prefetch_pages) = {
            let Self {
                recall,
                full_attn_kv,
                model,
                ..
            } = self;
            let pool = full_attn_kv.as_ref().expect("full_attn_kv");
            if let Some(state) = recall.get_mut(slot) {
                state.recompute_recall_plan(
                    &model.ctx,
                    pool,
                    slot,
                    cache_len,
                    &cfg,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    &layer0_query,
                    /* allow_prefetch = */ true,
                )?;
                (state.take_evict_pages(), state.take_prefetch_pages())
            } else {
                (Vec::new(), Vec::new())
            }
        };

        // Decode never prefetches — one batched sync point, here.
        for logical in prefetch_pages {
            let key = tier_block_u64(slot as u64, logical as u64);
            let Some(tier) = self.recall_tier.as_mut() else {
                break;
            };
            let payload = match tier.read(key) {
                Ok(p) => p.into_owned(),
                Err(_) => continue,
            };
            // Host owns the free stack: reinstate there, mirror down, refill.
            let Some(new_page) = host_kv.reinstate_slot_page(slot, logical) else {
                continue;
            };
            let seq_len = host_kv.seq_len(slot);
            self.mirror_host_slot(host_kv, slot, seq_len)?;
            if let Some(pool) = self.full_attn_kv.as_mut() {
                pool.copy_pages_from_host(&self.model.ctx, &[new_page], &payload)?;
            }
        }

        if let Some(state) = self.recall.get_mut(slot)
            && let Some(pool) = self.full_attn_kv.as_ref()
        {
            state.resolve_recall_pages(pool, slot);
        }

        // Prefill drained the compute stream, so freeing physical pages here cannot
        // race
        // an in-flight attention.
        for logical in evict_pages {
            let physical = {
                let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
                pool.page_indices(slot)
                    .get(logical)
                    .copied()
                    .filter(|&p| p != cuda_kernels::prelude::EVICTED_PAGE)
            };
            let Some(physical) = physical else {
                continue;
            };
            let key = tier_block_u64(slot as u64, logical as u64);
            let mirrored = {
                let payload = {
                    let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
                    pool.copy_pages_to_host(&self.model.ctx, &[physical])?
                };
                match self.recall_tier.as_mut() {
                    Some(tier) if !tier.is_full() => tier.insert(key, payload),
                    _ => false,
                }
            };
            if !mirrored {
                continue; // tier full → keep page resident (no KV loss)
            }
            // Host first — it owns the free stack.
            host_kv.evict_slot_page(slot, logical);
            if let Some(pool) = self.full_attn_kv.as_mut() {
                pool.evict_slot_page(slot, logical);
            }
        }
        Ok(token)
    }

    /// One decode row over the recall pool: ZERO tier I/O, working set fixed at
    /// prefill.
    fn decode_row_recall(
        &mut self,
        row: &DecodeRow,
        position: u64,
        host_kv: &mut dyn KvPool,
    ) -> Result<(u32, Option<f32>)> {
        let slot = row.slot;
        self.mirror_host_slot(host_kv, slot, row.kv_seq_len + 1)?;
        let cache_len = row.kv_seq_len + 1;
        let recall_pages: Vec<u32> = match self.recall.get(slot).and_then(|s| s.recall_pages()) {
            Some(p) => p.to_vec(),
            None => {
                let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
                let num_pages = cache_len.div_ceil(pool.page_size);
                pool.page_indices(slot)[..num_pages].to_vec()
            }
        };
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
            crate::loader::PageMeta::for_recall_decode(
                &self.model.ctx,
                pool,
                cache_len,
                &recall_pages,
            )?
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: None,
            cp: None,
            cp_decode: None,
        };
        model.forward_tokens_recall(
            &mut slots[slot],
            workspace,
            &[row.last_token],
            row.kv_seq_len,
            &row.params,
            position,
            penalty_of(&row.penalty_history, row.penalty_prompt_len),
            &mut rc,
        )
    }

    /// Stage per-step device scalars into the graph workspace and drop the
    /// slot's capture when any baked address drifted (release → re-alloc).
    fn stage_graph_step(
        model: &crate::qwen35::Qwen35Model,
        dg: &mut Qwen35DecodeGraph,
        slot: usize,
        last_token: u32,
        start_pos: usize,
        label: &str,
    ) -> Result<()> {
        let Qwen35DecodeGraph { ws, graphs, baked } = dg;
        let (token_ids_ptr, start_pos_ptr) =
            model.stage_step_inputs(ws, &[last_token], start_pos)?;
        let logits_ptr = model.workspace_logits_ptr(ws)?;
        let bake = Qwen35GraphBake {
            token_ids_ptr,
            start_pos_ptr,
            logits_ptr,
            ws_epoch: ws.epoch(),
        };
        match baked[slot] {
            Some(prev) if prev != bake => {
                info!(
                    "[qwen35-decode-graph] {label}slot {slot}: workspace addresses changed; \
                     dropping stale capture and recapturing"
                );
                graphs[slot] = crate::graph::CudaGraphState::new(model.ctx.stream.clone());
                baked[slot] = Some(bake);
            }
            None => baked[slot] = Some(bake),
            _ => {}
        }
        Ok(())
    }

    /// Shared graph-lane epilogue: advance the slot, bump the replay counters, then
    /// sample OUTSIDE the graph from the logits the run just wrote.
    fn finish_graph_step(
        &mut self,
        slot: usize,
        was_captured: bool,
        will_replay: bool,
        label: &str,
        row: &DecodeRow,
        position: u64,
    ) -> Result<(u32, Option<f32>)> {
        // Host-side state advance happens here — captured closure is host-state-free.
        self.slots[slot].advance_seq_len(1);
        let dg = self.decode_graph.as_ref().expect("still present");
        if !was_captured && dg.graphs[slot].is_captured() {
            let captures =
                QWEN35_GRAPH_CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let keys = dg.graphs.iter().filter(|g| g.is_captured()).count();
            info!(
                "[qwen35-decode-graph] captured {label}slot {slot} \
                 (captures_total={captures}, live_keys={keys}, max_keys={})",
                self.num_slots
            );
        }
        if will_replay {
            let replays =
                QWEN35_GRAPH_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if replays.is_multiple_of(100) {
                info!(
                    "[qwen35-decode-graph] {label}replay_total={replays} captures_total={}",
                    QWEN35_GRAPH_CAPTURES.load(std::sync::atomic::Ordering::Relaxed)
                );
            }
        }
        let dg = self.decode_graph.as_mut().expect("still present");
        self.model.sample_workspace_logits(
            &mut dg.ws,
            &row.params,
            position,
            penalty_of(&row.penalty_history, row.penalty_prompt_len),
        )
    }

    /// Whole-step decode graph over the PAGED pool: the growing page table is absorbed
    /// by a fixed-capacity per-slot [`crate::loader::PageMeta::persistent_decode`]
    /// refreshed outside the graph, with FA3's scheduling ceiling pinned via
    /// `seqlen_k_capture`. `Ok(None)` on any gate miss.
    fn try_graph_decode_paged(
        &mut self,
        row: &DecodeRow,
        position: u64,
        host_kv: &dyn KvPool,
    ) -> Result<Option<(u32, Option<f32>)>> {
        if !self.decode_graph_armed
            || !self.paged_kv_bf16()
            || !self.model.paged_decode_fa3_active()
        {
            return Ok(None);
        }
        if row.kv_seq_len + 1 > self.model.max_seq_len() {
            return Ok(None);
        }
        let slot = row.slot;
        {
            let pool = self
                .full_attn_kv
                .as_ref()
                .expect("full_attn_kv present (full_attn_paged)");
            ensure!(
                pool.seq_len(slot) == row.kv_seq_len,
                "Qwen3.6 paged decode graph: pool seq_len {} != kv_seq_len {} for slot {}",
                pool.seq_len(slot),
                row.kv_seq_len,
                slot
            );
        }
        // Idempotent, so the eager fallback may re-run it.
        self.mirror_host_slot(host_kv, slot, row.kv_seq_len + 1)?;
        if self.decode_graph.is_none() {
            self.decode_graph = Some(Qwen35DecodeGraph::new(
                self.num_slots,
                &self.model.ctx.stream,
            ));
        }
        if self.paged_decode_meta.is_empty() {
            self.paged_decode_meta = (0..self.num_slots).map(|_| None).collect();
        }
        {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv present");
            let capacity = self.model.max_seq_len().div_ceil(pool.page_size);
            let meta = match &mut self.paged_decode_meta[slot] {
                Some(meta) => meta,
                none => none.insert(crate::loader::PageMeta::persistent_decode(
                    &self.model.ctx,
                    pool.page_size,
                    capacity,
                )?),
            };
            meta.refresh_decode(&self.model.ctx, pool, slot, row.kv_seq_len)?;
        }
        let Self {
            model,
            slots,
            decode_graph,
            paged_decode_meta,
            full_attn_kv,
            ..
        } = self;
        let dg = decode_graph
            .as_mut()
            .expect("decode_graph built above when armed");
        Self::stage_graph_step(model, dg, slot, row.last_token, row.kv_seq_len, "paged ")?;
        let Qwen35DecodeGraph { ws, graphs, .. } = dg;
        let state = &mut graphs[slot];
        let was_captured = state.is_captured();
        let will_replay = was_captured && !state.is_armed_warm();
        let slot_state = &mut slots[slot];
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let meta = paged_decode_meta[slot]
            .as_ref()
            .expect("persistent meta built above");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta,
            layer0_query: None,
            cp: None,
            cp_decode: None,
        };
        let run = state.run_or_capture(|| {
            model.forward_decode_step_paged_captured(slot_state, ws, row.kv_seq_len, &mut rc)
        });
        if let Err(e) = run {
            warn!(
                "Qwen3.5 paged whole-step decode graph failed (slot {slot}), \
                 downgrading to eager forward: {e}"
            );
            self.decode_graph_armed = false;
            self.decode_graph = None;
            self.paged_decode_meta.clear();
            return Ok(None);
        }
        let out = self.finish_graph_step(slot, was_captured, will_replay, "paged ", row, position);
        out.map(Some)
    }

    /// Offload the model's device weights to host RAM, returning the bytes freed;
    /// per-slot KV / recurrent state stays resident. The forward workspace is released
    /// AFTER the offload's device sync, so no in-flight kernel references it.
    pub(crate) fn offload_engine_weights(&mut self) -> Result<usize> {
        self.ensure_not_collective("offload_engine_weights")?;
        let freed = self.model.offload_engine_weights()?;
        self.workspace.release();
        // The pointer TABLES survive: they address per-slot state, which the offload
        // leaves resident.
        if let Some(bd) = self.batch_decode.as_mut() {
            bd.release();
        }
        // Captured graphs bake the now-freed weight addresses.
        self.decode_graph = None;
        // Leaving N KV pools resident OOMs the co-resident student forward.
        self.release_kv_pool()?;
        // Trim AFTER releasing the scratch so the autograd store sees the freed VRAM.
        self.model.ctx.trim_memory_pool()?;
        let (free, total) = self.model.ctx.mem_info_bytes()?;
        eprintln!("[executor-offload] free={free} total={total}");
        Ok(freed)
    }

    /// Release the inference forward scratch WITHOUT offloading weights or touching KV:
    /// the freed blocks return to the shared async pool the co-resident OPD writeback
    /// reuses, which would otherwise OOM.
    pub(crate) fn release_inference_scratch(&mut self) -> Result<()> {
        self.workspace.release();
        if let Some(bd) = self.batch_decode.as_mut() {
            bd.release();
        }
        self.decode_graph = None;
        Ok(())
    }

    pub(crate) fn reload_engine_weights(&mut self) -> Result<()> {
        self.ensure_not_collective("reload_engine_weights")?;
        self.model.reload_engine_weights()?;
        self.ensure_kv_pool()?;
        Ok(())
    }

    /// OPD surfaces are rank-0 control-seam calls: under multi-rank TP they would run
    /// on
    /// one rank only, desyncing the per-step NCCL collective sequence.
    fn ensure_not_collective(&self, what: &str) -> Result<()> {
        ensure!(
            !self.model.tp.is_collective(),
            "{what} is single-GPU only: the Qwen3.5/3.6 OPD surfaces are not \
             wired for multi-rank tensor parallelism (world_size={})",
            self.model.tp.config().world_size
        );
        Ok(())
    }

    /// KV tokens a decode row reaches in one submit: MTP verifies depth+1,
    /// DSpark verifies a block_size chain (the anchor included). The engine
    /// pre-allocates this many tokens through its reclaim path, so the
    /// `set_host_slot_to` calls below only truncate a warm/short row's
    /// over-budget and never grow into an empty pool (#197).
    pub(crate) fn spec_row_tokens(&self) -> usize {
        let depth = self.model.spec_draft_tokens().max(1);
        if self.mtp.is_some() {
            depth + 1
        } else if self.dspark.is_some() {
            depth
        } else {
            1
        }
    }

    pub(crate) fn submit(
        &mut self,
        plan: &ForwardPlan,
        host_kv: &mut dyn KvPool,
    ) -> Result<StepOutput> {
        ensure!(
            host_kv.page_size() == SUPPORTED_PAGE_SIZE,
            "host CudaKvPool page_size={} does not match Qwen3.5 device page_size={SUPPORTED_PAGE_SIZE}",
            host_kv.page_size()
        );
        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        if rows == 0 {
            return Ok(StepOutput { tokens: Vec::new() });
        }

        // Gated to rows==1 PLANS so batched/mixed steps never capture or replay.
        let allow_graph = plan.prefill_rows.is_empty() && plan.decode_rows.len() == 1;

        // Plan rows always address disjoint slots, so the sequential prefill sub-steps
        // below are math-identical to consecutive single-mode ticks.
        let mut seen_slots = std::collections::BTreeSet::new();
        for slot in plan
            .prefill_rows
            .iter()
            .map(|row| row.slot)
            .chain(plan.decode_rows.iter().map(|row| row.slot))
        {
            ensure!(
                seen_slots.insert(slot),
                "Qwen3.5 plan schedules slot {slot} more than once per tick"
            );
        }

        let mut tokens = Vec::with_capacity(rows);
        for row in &plan.prefill_rows {
            let (token, logprob) = self.submit_prefill_row(row, host_kv)?;
            tokens.push(SlotToken {
                slot: row.slot,
                token,
                logprob,
                top_logprobs: self.take_top_logprobs(&row.params),
                finish: None,
            });
        }
        tokens.extend(self.dispatch_decode_rows(&plan.decode_rows, allow_graph, host_kv)?);
        Ok(StepOutput { tokens })
    }

    /// The spec scheme this executor is configured for (`--spec-type`).
    fn spec_kind(&self) -> super::spec_decode::SpecKind {
        if self.dspark.is_some() {
            super::spec_decode::SpecKind::Dspark
        } else if self.mtp.is_some() {
            super::spec_decode::SpecKind::Mtp
        } else {
            super::spec_decode::SpecKind::None
        }
    }

    /// The single `dspark → mtp → plain` dispatch ladder. At or below
    /// `--spec-max-batch`
    /// a spec scheme drafts per row; above it spec is a compute-bound loss, so decode
    /// falls to the plain batched path that scales.
    fn dispatch_decode_rows(
        &mut self,
        decode_rows: &[DecodeRow],
        allow_graph: bool,
        host_kv: &mut dyn KvPool,
    ) -> Result<Vec<SlotToken>> {
        use super::spec_decode::{DecodeRoute, SpecKind};
        let kind = self.spec_kind();
        // Only a batched greedy DSpark draft pays above c=1: sampling loses −15.5% at
        // c=8 and −26.4% at c=16.
        // ponytail: batched DSpark is gated to BF16 KV; upgrade path is a quant-KV
        // parity entry (needle gate ×3 same-config vs the BF16 baseline).
        let batched = kind == SpecKind::Dspark
            && self.paged_kv_bf16()
            && decode_rows.iter().all(|r| r.params.is_greedy());
        let any_penalty = decode_rows.iter().any(|r| r.params.has_penalty());
        let gate = match batched {
            true => crate::runtime_flags::spec_max_batch(),
            false => 1,
        };
        match super::spec_decode::route_decode(kind, decode_rows.len(), gate, any_penalty) {
            DecodeRoute::Dspark => self.dspark_decode_batch(decode_rows, host_kv),
            DecodeRoute::Mtp => {
                let mut tokens = Vec::with_capacity(decode_rows.len());
                for row in decode_rows {
                    tokens.extend(self.mtp_decode_row(row, host_kv)?);
                }
                Ok(tokens)
            }
            DecodeRoute::Plain => match decode_rows {
                [] => Ok(Vec::new()),
                [row] => {
                    let (token, logprob) = self.submit_decode_row(row, allow_graph, host_kv)?;
                    Ok(vec![SlotToken {
                        slot: row.slot,
                        token,
                        logprob,
                        top_logprobs: self.take_top_logprobs(&row.params),
                        finish: None,
                    }])
                }
                rows => self.submit_decode_batch(rows, host_kv),
            },
        }
    }

    /// Drain the workspace's OpenAI logprobs capture for a row that asked for
    /// it (written by the model's LAST host sampling in the preceding call).
    fn take_top_logprobs(&mut self, params: &SamplingParams) -> Vec<(u32, f32)> {
        if params.top_logprobs.is_some() {
            std::mem::take(&mut self.workspace.top_logprobs)
        } else {
            Vec::new()
        }
    }

    fn submit_prefill_row(
        &mut self,
        row: &infer_plan::PrefillRow,
        host_kv: &mut dyn KvPool,
    ) -> Result<(u32, Option<f32>)> {
        ensure!(
            row.slot < self.num_slots,
            "prefill slot {} outside Qwen3.5 executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
        // A fresh prefill rewinds this slot's recurrent + conv state. The captured
        // graph
        // stays valid (state buffers are memset, not re-allocated), but the new
        // occupant's
        // FIRST decode runs one eager warm step, so capture cost stays once per slot.
        if row.start_pos == 0 {
            // The prior occupant is finished, so return its recurrent block to the
            // free-list, then acquire and zero one; it MUST be resident before the
            // forward.
            self.slots[row.slot].release_recurrent(&mut self.recurrent_pool);
            // A stale snapshot must never key this new request's sidecar.
            self.prefill_boundary_snapshot[row.slot] = None;
            self.periodic_boundary_snapshots[row.slot].clear();
            // New occupant: the prior request's draft ctx cache is dead.
            if let Some(df) = self
                .dspark
                .as_mut()
                .and_then(|ds| ds.slots[row.slot].as_mut())
            {
                df.reset();
            }
            // New occupant: drop the prior request's MTP seed; the warm step re-seeds.
            if let Some(mtp) = self.mtp.as_mut() {
                mtp.slots[row.slot] = None;
            }
            let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
            self.slots[row.slot].acquire_recurrent(
                &self.model.ctx,
                num_linear,
                gdr_len,
                conv_len,
                &mut self.recurrent_pool,
            )?;
            // The new block's addresses differ from the prior occupant's, so the
            // pointer-table cache (keyed on slot_indices only) must restage.
            if let Some(bd) = self.batch_decode.as_mut() {
                bd.invalidate_staged_pointers();
            }
            // The capture bakes this slot's recurrent-block addresses and `baked`
            // tracks
            // only workspace ptrs, so drop it — `rearm_warm` alone would replay freed
            // mem.
            if let Some(dg) = self.decode_graph.as_mut() {
                dg.graphs[row.slot] =
                    crate::graph::CudaGraphState::new(self.model.ctx.stream.clone());
                dg.baked[row.slot] = None;
            }
            // Free the prior occupant's pages so a fresh prefill starts at logical page
            // 0.
            if self.recall_active() {
                self.recall[row.slot].reset();
                // Release keepalive-parked pages BEFORE freeing the slot: they are
                // sentinels in the table, so `free_slot` alone would not recycle them.
                let parked = std::mem::take(&mut self.recall_keepalive[row.slot]);
                if let Some(pool) = self.full_attn_kv.as_mut() {
                    for (_logical, physical) in parked {
                        pool.release_evicted_page(physical);
                    }
                    pool.mirror_slot(row.slot, &[], 0)?;
                }
                // Stale L3 entries are left to the store's LRU: `reset()` cleared all
                // reps, so a fresh occupant can never read a prior occupant's block
                // before overwriting that key.
            } else if let Some(pool) = self.full_attn_kv.as_mut() {
                // Drop the mirror; the host pool owns these pages.
                pool.mirror_slot(row.slot, &[], 0)?;
            }
        }
        let position = (row.start_pos + row.tokens.len()) as u64;
        if self.recall_active() {
            return self.prefill_row_recall(row, position, host_kv);
        }
        if self.slots[row.slot].has_recurrent() {
            return self.prefill_row_snapshotted(row, position, host_kv);
        }
        self.prefill_row_paged_default(row, position, host_kv)
    }

    /// Prefill a hybrid row, splitting the forward at recurrent-snapshot boundaries so
    /// a
    /// later prefix hit can restore the linear-attn state.
    ///
    /// Recurrent state is only observable at a forward's END, so a snapshot keyed at
    /// `S`
    /// requires a forward that ends at `S` — otherwise it bakes the residue `[S..end]`
    /// and double-advances it on restore. Cuts are `L*` (the exact-resend target) and
    /// the
    /// stride multiples crossed in `(start, end)`. Every cut is `< end`, so a real tail
    /// always remains, and all cuts are page-aligned so `hash(tokens[..S])` rendezvous
    /// with the restore probe's boundaries.
    fn prefill_row_snapshotted(
        &mut self,
        row: &infer_plan::PrefillRow,
        position: u64,
        host_kv: &dyn KvPool,
    ) -> Result<(u32, Option<f32>)> {
        let start = row.start_pos;
        let end = start + row.tokens.len();
        let is_final = end == row.total_tokens;
        let lstar = row.total_tokens.saturating_sub(1) / SUPPORTED_PAGE_SIZE * SUPPORTED_PAGE_SIZE;
        let stride = SIDECAR_SNAPSHOT_STRIDE_PAGES * SUPPORTED_PAGE_SIZE; // const, > 0

        // Ordered snapshot cuts in `[start, end)`; `bool` = is-`L*`.
        let mut cuts: Vec<(usize, bool)> = Vec::new();
        // A prior chunk ending exactly on a stride multiple leaves a boundary that is
        // never `< end` of any chunk, so snapshot the already-materialized state here.
        if start > 0 && start.is_multiple_of(stride) {
            cuts.push((start, is_final && start == lstar));
        }
        let mut s = (start / stride + 1) * stride;
        while s < end {
            cuts.push((s, is_final && s == lstar));
            s += stride;
        }
        // `L*` on the final chunk (non-aligned residue can land on `start`).
        if is_final
            && lstar > 0
            && lstar >= start
            && lstar < end
            && !cuts.iter().any(|&(p, _)| p == lstar)
        {
            cuts.push((lstar, true));
        }
        cuts.sort_unstable_by_key(|&(p, _)| p);

        let mut cursor = start;
        let mut did_cut = false;
        for (cut, is_lstar) in cuts {
            if cut > cursor {
                let seg = infer_plan::PrefillRow {
                    slot: row.slot,
                    tokens: row.tokens[cursor - start..cut - start].to_vec(),
                    start_pos: cursor,
                    total_tokens: row.total_tokens,
                    params: row.params.clone(),
                    penalty_history: row.penalty_history.clone(),
                    penalty_prompt_len: row.penalty_prompt_len,
                };
                self.prefill_row_paged_default(&seg, cut as u64, host_kv)?; // token discarded
                cursor = cut;
            }
            // State is now materialized at exactly `cut`.
            let snap = self.slots[row.slot].snapshot_recurrent(&self.model.ctx)?;
            if is_lstar {
                self.prefill_boundary_snapshot[row.slot] = Some((cut, snap));
            } else {
                self.periodic_boundary_snapshots[row.slot].push((cut, snap));
            }
            did_cut = true;
        }
        if !did_cut {
            return self.prefill_row_paged_default(row, position, host_kv);
        }
        // Cuts are all `< end`, so this tail is non-empty.
        let tail = infer_plan::PrefillRow {
            slot: row.slot,
            tokens: row.tokens[cursor - start..].to_vec(),
            start_pos: cursor,
            total_tokens: row.total_tokens,
            params: row.params.clone(),
            penalty_history: row.penalty_history.clone(),
            penalty_prompt_len: row.penalty_prompt_len,
        };
        self.prefill_row_paged_default(&tail, position, host_kv)
    }

    /// `allow_graph` admits the whole-step B=1 decode-graph lane — true only for
    /// rows==1 plans.
    fn submit_decode_row(
        &mut self,
        row: &DecodeRow,
        allow_graph: bool,
        host_kv: &mut dyn KvPool,
    ) -> Result<(u32, Option<f32>)> {
        ensure!(
            row.slot < self.num_slots,
            "decode slot {} outside Qwen3.5 executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(
            self.slots[row.slot].seq_len() == row.kv_seq_len,
            "Qwen3.5 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
            self.slots[row.slot].seq_len(),
            row.kv_seq_len,
            row.slot
        );
        let position = row.kv_seq_len.saturating_add(1) as u64;
        // Recall decode reads the fixed working set chosen at prefill; the seq_len
        // invariant above holds because the recall forward advances it in lockstep.
        if self.recall_active() {
            return self.decode_row_recall(row, position, host_kv);
        }
        // The graph lane runs first when armed; the eager paged forward is the
        // correctness floor and the fallback for every gate miss.
        if allow_graph && let Some(token) = self.try_graph_decode_paged(row, position, host_kv)? {
            return Ok(token);
        }
        self.decode_row_paged_default(row, position, host_kv)
    }

    /// A rows>1 pure-decode sub-batch: ONE batched forward over all rows. With
    /// `--qwen35-batched-decode false`, runs the rows sequentially instead.
    ///
    /// Batched steps never capture or replay, and cannot invalidate existing B=1
    /// captures: they mutate per-slot state strictly IN PLACE through pointer tables
    /// and
    /// own a separate workspace, and the per-step position is read from a staged device
    /// scalar at replay.
    fn submit_decode_batch(
        &mut self,
        rows: &[DecodeRow],
        host_kv: &mut dyn KvPool,
    ) -> Result<Vec<SlotToken>> {
        debug_assert!(rows.len() > 1);
        // Validate BEFORE any device mutation (the dup-slot ensure ran in `submit`).
        for row in rows {
            ensure!(
                row.slot < self.num_slots,
                "decode slot {} outside Qwen3.5 executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                self.slots[row.slot].seq_len() == row.kv_seq_len,
                "Qwen3.5 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
                self.slots[row.slot].seq_len(),
                row.kv_seq_len,
                row.slot
            );
        }

        if crate::runtime_flags::qwen35_batched_decode()
            && !self.recall_active()
            && self.model.tp.is_single()
        {
            return self.submit_decode_batch_paged(rows, host_kv);
        }
        let mut tokens = Vec::with_capacity(rows.len());
        for row in rows {
            let (token, logprob) =
                self.submit_decode_row(row, /* allow_graph = */ false, host_kv)?;
            tokens.push(SlotToken {
                slot: row.slot,
                token,
                logprob,
                top_logprobs: self.take_top_logprobs(&row.params),
                finish: None,
            });
        }
        Ok(tokens)
    }

    /// A rows>1 decode sub-batch over the shared-paged lane: ONE B-row page table and a
    /// single batched-paged forward. Each row attends only its own slot's pages via its
    /// `kv_indptr` slice, so a B-row batch is equivalent to B sequential single-row
    /// paged
    /// decodes. Single-GPU, no recall (gated by the caller).
    fn submit_decode_batch_paged(
        &mut self,
        rows: &[DecodeRow],
        host_kv: &mut dyn KvPool,
    ) -> Result<Vec<SlotToken>> {
        debug_assert!(rows.len() > 1);
        // Append before building the page table: the meta encodes POST-append lengths,
        // and the pool seq_len must equal the engine's kv_seq_len pre-append.
        for row in rows {
            {
                let pool = self
                    .full_attn_kv
                    .as_ref()
                    .expect("full_attn_kv present (full_attn_paged)");
                ensure!(
                    pool.seq_len(row.slot) == row.kv_seq_len,
                    "Qwen3.6 paged batched decode: pool seq_len {} != kv_seq_len {} for slot {}",
                    pool.seq_len(row.slot),
                    row.kv_seq_len,
                    row.slot
                );
            }
            self.mirror_host_slot(host_kv, row.slot, row.kv_seq_len + 1)?;
        }

        let slot_indices: Vec<usize> = rows.iter().map(|r| r.slot).collect();
        let tokens_in: Vec<u32> = rows.iter().map(|r| r.last_token).collect();
        let kv_seq_lens: Vec<usize> = rows.iter().map(|r| r.kv_seq_len).collect();
        let params: Vec<SamplingParams> = rows.iter().map(|r| r.params.clone()).collect();
        let sample_positions: Vec<u64> = rows
            .iter()
            .map(|r| r.kv_seq_len.saturating_add(1) as u64)
            .collect();
        let batch_rows: Vec<(usize, usize)> =
            rows.iter().map(|r| (r.slot, r.kv_seq_len + 1)).collect();
        let penalties: Vec<infer_plan::PenaltyHistory<'_>> = rows
            .iter()
            .map(|r| penalty_of(&r.penalty_history, r.penalty_prompt_len))
            .collect();

        if self.batch_decode.is_none() {
            let num_linear =
                self.model.config.num_hidden_layers - self.model.config.num_full_attention_layers();
            self.batch_decode = Some(crate::qwen35::Qwen35BatchDecodeState::new(
                &self.model.ctx,
                num_linear,
                self.num_slots,
            )?);
        }

        let Self {
            model,
            slots,
            batch_decode,
            full_attn_kv,
            ..
        } = self;
        let bd = batch_decode.as_mut().expect("batch_decode built above");
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let meta = PageMeta::for_decode_batch(&model.ctx, pool, &batch_rows)?;
        let sampled = model.forward_decode_batch_paged(
            slots,
            bd,
            pool,
            &meta,
            &slot_indices,
            &tokens_in,
            &kv_seq_lens,
            &params,
            &sample_positions,
            &penalties,
        )?;
        ensure!(
            sampled.len() == rows.len(),
            "Qwen3.6 paged batched decode returned {} tokens for {} rows",
            sampled.len(),
            rows.len()
        );
        Ok(slot_indices
            .into_iter()
            .zip(sampled)
            .map(|(slot, (token, logprob, top_logprobs))| SlotToken {
                slot,
                token,
                logprob,
                top_logprobs,
                finish: None,
            })
            .collect())
    }

    /// OPD teacher raw-logits forward on a FRESH transient slot: returns the FULL
    /// `[seq_len, vocab]` logits without sampling and never touches the serving slots.
    /// `positions` must be the contiguous absolute positions of `input_ids`.
    pub(crate) fn forward_token_logits(
        &mut self,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<(DeviceVec, [usize; 2])> {
        self.ensure_not_collective("forward_token_logits")?;
        ensure!(
            !input_ids.is_empty(),
            "forward_token_logits requires a non-empty token sequence"
        );
        ensure!(
            input_ids.len() == positions.len(),
            "forward_token_logits token/position length mismatch: tokens={} positions={}",
            input_ids.len(),
            positions.len()
        );
        let start_pos = positions[0] as usize;
        for (i, &p) in positions.iter().enumerate() {
            ensure!(
                p as usize == start_pos + i,
                "forward_token_logits requires contiguous positions; positions[{i}]={p} != {}",
                start_pos + i
            );
        }
        // The shared workspace IS reused (forwards are serial). The recurrent block
        // goes
        // to a throwaway local pool — this path is rare and transient.
        let mut slot = self.model.new_slot_state();
        let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
        let mut scratch_pool = Vec::new();
        slot.acquire_recurrent(
            &self.model.ctx,
            num_linear,
            gdr_len,
            conv_len,
            &mut scratch_pool,
        )?;
        // The transient forward borrows a FREE pool slot for its KV pages and returns
        // them before this call completes.
        let Some(pool_probe) = self.full_attn_kv.as_ref() else {
            return self.model.forward_token_logits_full(
                &mut slot,
                &mut self.workspace,
                input_ids,
                start_pos,
                None,
            );
        };
        ensure!(
            start_pos == 0,
            "forward_token_logits on the paged full-attn path requires start_pos 0, got {start_pos}"
        );
        let kv_slot = (0..self.slots.len())
            .find(|&s| pool_probe.seq_len(s) == 0 && self.slots[s].seq_len() == 0)
            .ok_or_else(|| {
                anyhow::anyhow!("forward_token_logits: no free KV slot for the transient forward")
            })?;
        let Self {
            model,
            workspace,
            full_attn_kv,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        // No host pool to lower from: detached pages, not the slot allocator (which
        // must not be mixed with `mirror_slot`).
        let scratch = pool.alloc_detached_pages(input_ids.len().div_ceil(pool.page_size))?;
        pool.mirror_slot(kv_slot, &scratch, input_ids.len())?;
        let result =
            crate::loader::PageMeta::for_slot(&model.ctx, pool, kv_slot, 0, input_ids.len())
                .and_then(|meta| {
                    let mut rc = crate::qwen35::Qwen35RecallForward {
                        pool,
                        meta: &meta,
                        layer0_query: None,
                        cp: None,
                        cp_decode: None,
                    };
                    model.forward_token_logits_full(
                        &mut slot,
                        workspace,
                        input_ids,
                        0,
                        Some(&mut rc),
                    )
                });
        let pool = self.full_attn_kv.as_mut().expect("full_attn_kv present");
        pool.mirror_slot(kv_slot, &[], 0)?;
        pool.release_pages(&scratch);
        result
    }

    /// One-shot trunk forward for offline DSpark draft training: the raw taps at
    /// `target_layer_ids` as `[seq, taps·hidden]` and the final-normed hidden states as
    /// `[seq, hidden]`, both host-side. Same transient-slot discipline as
    /// [`Self::forward_token_logits`].
    pub(crate) fn forward_training_taps(
        &mut self,
        input_ids: &[u32],
        target_layer_ids: &[i64],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.ensure_not_collective("forward_training_taps")?;
        ensure!(
            !input_ids.is_empty(),
            "forward_training_taps requires a non-empty token sequence"
        );
        let mut taps = crate::qwen35::dspark::Qwen35DsparkTaps::default();
        let mut slot = self.model.new_slot_state();
        let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
        let mut scratch_pool = Vec::new();
        slot.acquire_recurrent(
            &self.model.ctx,
            num_linear,
            gdr_len,
            conv_len,
            &mut scratch_pool,
        )?;
        if self.full_attn_kv.is_none() {
            return self.model.forward_training_taps(
                &mut slot,
                &mut self.workspace,
                &mut taps,
                target_layer_ids,
                input_ids,
                None,
            );
        }
        let pool_probe = self.full_attn_kv.as_ref().expect("full_attn_kv present");
        let kv_slot = (0..self.slots.len())
            .find(|&s| pool_probe.seq_len(s) == 0 && self.slots[s].seq_len() == 0)
            .ok_or_else(|| {
                anyhow::anyhow!("forward_training_taps: no free KV slot for the transient forward")
            })?;
        let Self {
            model,
            workspace,
            full_attn_kv,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let scratch = pool.alloc_detached_pages(input_ids.len().div_ceil(pool.page_size))?;
        pool.mirror_slot(kv_slot, &scratch, input_ids.len())?;
        let result =
            crate::loader::PageMeta::for_slot(&model.ctx, pool, kv_slot, 0, input_ids.len())
                .and_then(|meta| {
                    let mut rc = crate::qwen35::Qwen35RecallForward {
                        pool,
                        meta: &meta,
                        layer0_query: None,
                        cp: None,
                        cp_decode: None,
                    };
                    model.forward_training_taps(
                        &mut slot,
                        workspace,
                        &mut taps,
                        target_layer_ids,
                        input_ids,
                        Some(&mut rc),
                    )
                });
        let pool = self.full_attn_kv.as_mut().expect("full_attn_kv present");
        pool.mirror_slot(kv_slot, &[], 0)?;
        pool.release_pages(&scratch);
        result
    }

    pub(crate) fn device(&self) -> &DeviceContext {
        &self.model.ctx
    }

    /// Fold a fresh student LoRA update into the resident projection weights.
    pub(crate) fn remerge_student_lora(
        &mut self,
        update: crate::qwen35::StudentLoraUpdate,
    ) -> Result<()> {
        self.ensure_not_collective("remerge_student_lora")?;
        // The merge REPLACES `DeviceMatrix` buffers; captured graphs bake the old ones.
        self.decode_graph = None;
        // Weight epoch changed: drop every tracked sidecar blob so a skipped capture
        // never serves old-epoch state.
        for (_, key) in self.sidecar_page_key.drain() {
            self.slot_tier
                .remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key);
        }
        self.model.remerge_student_lora(update)
    }

    /// Read-only borrow of resident FP8 block-scaled base projection pointers
    /// (`--share-frozen-base`); no decode-graph invalidation is needed.
    pub(crate) fn frozen_base_fp8_pointers(
        &self,
    ) -> Result<Vec<crate::qwen35::SharedFp8BaseProjection>> {
        self.ensure_not_collective("frozen_base_fp8_pointers")?;
        self.model.frozen_base_fp8_pointers()
    }

    /// Non-owning views of every resident dense-BF16 base projection's device
    /// pointer, for refreshing the train student's frozen base AFTER a LoRA
    /// re-merge.
    pub(crate) fn frozen_base_bf16_pointers(
        &self,
    ) -> Result<Vec<crate::qwen35::SharedBf16BaseProjection>> {
        self.ensure_not_collective("frozen_base_bf16_pointers")?;
        self.model.frozen_base_bf16_pointers()
    }

    /// Hot-swap the DSpark Markov head weights from a host f32 snapshot; invalidates
    /// the
    /// decode graph, which bakes the old weight pointers.
    pub(crate) fn update_dspark_markov_weights(&mut self, w1: &[f32], w2: &[f32]) -> Result<()> {
        self.ensure_not_collective("update_dspark_markov_weights")?;
        self.decode_graph = None;
        let dspark = self
            .dspark
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("DSpark head not loaded"))?;
        dspark.head.update_markov_weights(&self.model.ctx, w1, w2)
    }
}

#[cfg(test)]
mod tier_io_tests {
    use super::*;
    #[test]
    fn speculative_chain_boundary_falls_back_before_verify_exceeds_max_seq_len() {
        assert!(speculative_chain_fits(12, 3, 16));
        assert!(!speculative_chain_fits(13, 3, 16));
        assert!(!speculative_chain_fits(usize::MAX, 1, usize::MAX));
    }

    #[test]
    fn merge_prefers_direct_and_saturates_counters() {
        let slot = kv_native_sys::TierIoStats {
            mode: kv_native_sys::DiskIoMode::Mmap,
            useful_read_bytes: u64::MAX,
            failures: 2,
            ..Default::default()
        };
        let recall = kv_native_sys::TierIoStats {
            mode: kv_native_sys::DiskIoMode::Direct,
            useful_read_bytes: 1,
            failures: 3,
            ..Default::default()
        };
        let merged = merge_tier_io_stats(&slot, &recall);
        assert_eq!(merged.mode, infer_seam::KvTierIoMode::Direct);
        assert_eq!(merged.useful_read_bytes, u64::MAX);
        assert_eq!(merged.failures, 5);
    }
}
