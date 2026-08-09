use super::*;

pub(crate) struct QwenCudaExecutor {
    pub(crate) model: CudaModel,
    pub(crate) kv: PagedKVPool,
    tier: KvTierStore,
    pub(crate) num_slots: usize,
    /// Per-slot (occupant epoch, materialized token count) continuity guard.
    ///
    /// The host `CudaKvPool` is the single page allocator; this executor only
    /// mirrors the host page table into the device pool per scheduled row
    /// (`TokenKVPool::mirror_slot`), so the device pool no longer proves that
    /// the KV rows behind a resumed position were actually written. This
    /// watermark restores that loud-error contract: within one occupant epoch,
    /// rows must append contiguously; a NEW epoch may start at `append_pos > 0`
    /// only because the engine attached a radix prefix whose retained pages
    /// still hold the publishing request's KV rows.
    slot_progress: Vec<SlotProgress>,
    /// Fixed device buffers for the B=1 captured decode path. Built lazily at
    /// warmup; `None` until then / on capture failure (capture is never
    /// load-bearing for correctness — eager is the floor).
    decode_ctx: Option<DecodeGraphContext>,
    /// Per-shape captured decode graphs, keyed by page-table length: batch is
    /// fixed at [`DECODE_GRAPH_BATCH`], so `num_pages` is the only varying capture
    /// scalar. A new page count recaptures.
    graphs: Option<GraphBucket>,
    /// Session KV-recall opt-in (`--kv-recall`, default off). When off the decode
    /// hot path does no scoring and the page table is the full contiguous cache →
    /// baseline byte-identical (CUDA is the Stable backend). When on, recall is
    /// BF16-only and **eager-only** (the captured decode graph bakes `num_pages`,
    /// and recall needs a host query read-back + restricted page table between
    /// steps), so a recall-active slot skips the graph.
    kv_recall: bool,
    /// Recall budget regions (validated defaults): sink 32 + local 256 + top-k 8
    /// blocks of 32. The working set is bounded regardless of history length.
    recall_cfg: infer_core::RecallConfig,
    /// Per-slot resident mean-key reps + next-step recall page plan. Indexed by
    /// slot; only mutated when `kv_recall` is on.
    recall: Vec<crate::recall::CudaRecallState>,
    /// One-time non-BF16-KV-with-recall fallback log latch.
    recall_quant_warned: bool,
    tier_budget_bytes: usize,
    weights_epoch: String,
    disk_root: Option<std::path::PathBuf>,
    disk_budget: Option<usize>,
}

/// One slot's executor-side materialization watermark (see
/// [`QwenCudaExecutor::slot_progress`]).
#[derive(Clone, Copy)]
struct SlotProgress {
    /// Host-pool occupant epoch the watermark belongs to.
    epoch: u64,
    /// Tokens materialized (KV rows written or prefix-attached) for that epoch.
    len: usize,
}

impl Default for SlotProgress {
    fn default() -> Self {
        // u64::MAX never collides with a real host epoch (epochs start at 0 and
        // bump by 1-2 per occupancy), so the first real row always takes the
        // fresh-occupant branch.
        Self {
            epoch: u64::MAX,
            len: 0,
        }
    }
}

impl std::fmt::Debug for QwenCudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenCudaExecutor")
            .field("model", &self.model)
            .field("num_slots", &self.num_slots)
            .field("page_size", &self.kv.page_size)
            .field("max_total_pages", &self.kv.max_total_pages)
            .field("decode_graph", &self.decode_ctx.is_some())
            .field(
                "captured_decode_shapes",
                &self.graphs.as_ref().map_or(0, GraphBucket::len),
            )
            .finish()
    }
}

impl QwenCudaExecutor {
    /// `BackendExecutor::tp_sync_min` — see there for why the scheduler needs
    /// this (2026-07-05 TP=4 admission livelock). Dense Qwen3 has no existing
    /// `tp_min_usize` helper (unlike DSv4/Qwen3.6, which already use one for
    /// KV-budget clamping), so this inlines the same all-reduce.
    pub(crate) fn tp_sync_min(&self, local: usize) -> Result<usize> {
        let capped = i32::try_from(local.min(i32::MAX as usize)).unwrap_or(i32::MAX);
        self.model
            .tp
            .all_reduce_min_scalar_i32(&self.model.ctx, capped)
            .map(|v| v.max(0) as usize)
            .map_err(|e| anyhow::anyhow!("Qwen3 TP min-reduce admission free pages failed: {e}"))
    }

    /// `mem_fraction_static` (default 0.9): the dense shared paged pool is sized
    /// from MEASURED free VRAM after weights load (`infer_seam::profile_kv_pool_tokens`,
    /// SGLang-style), NOT the requested `total_pages`. `total_pages` becomes a
    /// minimum-capacity floor: the profiled pool is the larger of the two so the
    /// internal `total_pages` default never shrinks the pool below it, but a large
    /// card gets the extra capacity for more concurrency.
    pub(crate) fn from_qwen3_bf16_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        kv_dtype: CudaKvCacheDtype,
        mem_fraction_static: f64,
    ) -> Result<Self> {
        ensure!(num_slots > 0, "CudaExecutor requires at least one slot");
        ensure!(
            total_pages > 0,
            "CudaExecutor requires at least one KV page"
        );

        let weights_epoch = kv_native_sys::weights_epoch_tag(model_path.as_ref());
        let model = CudaModel::from_safetensors(model_path.as_ref())?;
        let kv_format = kv_dtype.kv_format();

        // Profile the shared paged pool from MEASURED free VRAM now that the
        // weights are resident (SGLang's mem_fraction_static). Per-token cell cost
        // = `budget_bytes_for_tokens(.., 1 token)` (storage + work bytes the pool
        // actually charges). On a successful read the profiled budget IS the
        // sizing; if the VRAM probe fails (no active context / driver error),
        // fall back to the requested `total_pages` exactly.
        let cell_bytes_per_token = PagedKVPool::budget_bytes_for_tokens(
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            1,
            kv_format,
        ) as u64;
        let requested_pages = total_pages;
        let total_pages = match model.ctx.mem_info_bytes() {
            Ok((free, total)) => {
                let profiled_tokens = infer_seam::profile_kv_pool_tokens(
                    free as u64,
                    total as u64,
                    cell_bytes_per_token,
                    mem_fraction_static,
                );
                let profiled_pages = (profiled_tokens / SUPPORTED_PAGE_SIZE as u64) as usize;
                // #178: flooring the profile at a constant books HBM the card lacks.
                let sized = profiled_pages.max(1);
                log::info!(
                    "CUDA dense Qwen3 KV pool profiled from measured VRAM: free {}MB / total \
                     {}MB, mem_fraction_static {mem_fraction_static}, cell {cell_bytes_per_token}B/tok \
                     -> max_total_tokens {profiled_tokens} ({profiled_pages} pages); requested \
                     {requested_pages} pages (advisory) -> sizing {sized} pages",
                    free >> 20,
                    total >> 20,
                );
                sized
            }
            Err(e) => {
                log::warn!(
                    "CUDA dense Qwen3 KV pool: free-VRAM probe failed ({e}); falling back to \
                     requested total_pages={requested_pages}"
                );
                requested_pages
            }
        };

        let token_budget = total_pages * SUPPORTED_PAGE_SIZE;
        let budget_bytes = PagedKVPool::budget_bytes_for_tokens(
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            token_budget,
            kv_format,
        );
        let kv = PagedKVPool::with_format(
            &model.ctx,
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            num_slots,
            budget_bytes,
            kv_format,
        )?;
        ensure!(
            kv.page_size == SUPPORTED_PAGE_SIZE,
            "R6 paged Qwen3 expects cuda-kernels page_size={SUPPORTED_PAGE_SIZE}, got {}",
            kv.page_size
        );

        let slot_progress = vec![SlotProgress::default(); num_slots];
        let tier_budget_bytes = default_t1_budget_bytes(DEFAULT_DRAM_FRACTION);
        let tier = KvTierStore::with_budget(tier_budget_bytes, kv.storage_bytes_per_page());
        let recall = (0..num_slots)
            .map(|_| crate::recall::CudaRecallState::default())
            .collect();
        Ok(Self {
            model,
            kv,
            tier,
            num_slots,
            slot_progress,
            decode_ctx: None,
            graphs: None,
            kv_recall: false,
            recall_cfg: crate::recall::default_recall_config(),
            recall,
            recall_quant_warned: false,
            tier_budget_bytes,
            weights_epoch,
            disk_root: None,
            disk_budget: None,
        })
    }

    pub(crate) fn kv_tier_capacity_pages(&self) -> usize {
        self.tier.capacity_pages()
    }

    pub(crate) fn kv_tier_page_bytes(&self) -> usize {
        self.tier.page_bytes()
    }

    pub(crate) fn kv_tier_host_demoted_pages(&self) -> usize {
        self.tier.host_demoted_pages()
    }

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        self.tier.disk_pages()
    }

    pub(crate) fn kv_tier_location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        self.tier.location(key)
    }

    pub(crate) fn kv_tier_io_stats(&self) -> infer_seam::KvTierIoStats {
        let stats = self.tier.io_stats();
        infer_seam::KvTierIoStats {
            mode: match stats.mode {
                kv_native_sys::DiskIoMode::Disabled => infer_seam::KvTierIoMode::Disabled,
                kv_native_sys::DiskIoMode::Mmap => infer_seam::KvTierIoMode::Mmap,
                kv_native_sys::DiskIoMode::Direct => infer_seam::KvTierIoMode::Direct,
            },
            useful_read_bytes: stats.useful_read_bytes,
            useful_write_bytes: stats.useful_write_bytes,
            submitted_read_bytes: stats.submitted_read_bytes,
            submitted_write_bytes: stats.submitted_write_bytes,
            metadata_write_bytes: stats.metadata_write_bytes,
            failures: stats.failures,
            completion_wait_ns: stats.completion_wait_ns,
        }
    }

    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        pages_only_reusable_prefix_blocks(blocks, |key| self.tier.contains(key))
    }

    /// Re-budget the host-demoted tier store (`0` disables). Pre-serve only: any
    /// existing entries are dropped, so callers configure this right after
    /// construction, before the engine demotes anything.
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        self.tier_budget_bytes = bytes;
        self.tier = KvTierStore::with_budget(bytes, self.kv.storage_bytes_per_page());
    }

    /// Opt into session KV-recall (`--kv-recall`, default off). Mirrors the Metal
    /// `set_kv_recall`: a post-construction setter so the constructor signature
    /// stays stable. With recall off the decode hot path is unchanged
    /// (byte-identical baseline — CUDA is the Stable backend).
    pub(crate) fn set_kv_recall(&mut self, enabled: bool) {
        self.kv_recall = enabled;
        if !enabled || self.tier.host_demoted_pages() != 0 || self.tier.disk_pages() != 0 {
            return;
        }
        let (Some(root), Some(budget)) = (self.disk_root.clone(), self.disk_budget) else {
            return;
        };
        self.attach_recall_disk(root, budget);
    }

    fn attach_recall_disk(&mut self, root: std::path::PathBuf, budget_bytes: usize) -> bool {
        let page_bytes = self.kv.storage_bytes_per_page();
        let mut tier = KvTierStore::with_budget(self.tier_budget_bytes, page_bytes);
        if !tier.load(
            root.clone(),
            budget_bytes,
            page_bytes,
            self.weights_epoch.clone(),
            self.kv
                .format
                .stable_tag()
                .expect("persisted KV format must have a stable tag"),
            self.model.tp.config().world_size,
            self.model.tp.config().rank,
        ) {
            log::warn!("durable KV recall disk unavailable; using ephemeral disk spill");
            if !tier.set_disk(root, budget_bytes, page_bytes) {
                return false;
            }
        }
        self.tier = tier;
        true
    }

    /// Attach the opt-in disk spill level (`--kv-disk`). Pre-serve only.
    pub(crate) fn set_kv_tier_disk(
        &mut self,
        root: std::path::PathBuf,
        budget_bytes: usize,
    ) -> bool {
        self.disk_root = Some(root.clone());
        self.disk_budget = Some(budget_bytes);
        let page_bytes = self.kv.storage_bytes_per_page();
        if self.kv_recall {
            return self.attach_recall_disk(root, budget_bytes);
        }
        self.tier.set_disk(root, budget_bytes, page_bytes)
    }

    /// Copy device pages into the host tier store (synchronous: the copy is
    /// complete when this returns, so the engine may free the pages). Stops at
    /// capacity and reports the accepted prefix length.
    pub(crate) fn demote_prefix_pages(&mut self, entries: &[(u32, u64)]) -> Result<usize> {
        if entries.is_empty() || self.tier.is_full() {
            return Ok(0);
        }
        let accepted = entries.len().min(self.tier.available_pages());
        let entries = &entries[..accepted];
        let pages = entries.iter().map(|&(page, _)| page).collect::<Vec<_>>();
        let payload = self.kv.copy_pages_to_host(&self.model.ctx, &pages)?;
        let page_bytes = self.kv.storage_bytes_per_page();
        ensure!(
            payload.len() == accepted * page_bytes,
            "batched KV demote returned {} bytes for {accepted} pages of {page_bytes} bytes",
            payload.len()
        );
        let batch = entries
            .iter()
            .zip(payload.chunks_exact(page_bytes))
            .map(|(&(_, key), bytes)| (key, bytes.to_vec()))
            .collect::<Vec<_>>();
        Ok(if self.tier.insert_many(batch) {
            accepted
        } else {
            0
        })
    }

    /// Copy host tier entries back into freshly allocated device pages. The
    /// engine attaches the pages right after, so sync before returning.
    pub(crate) fn promote_prefix_pages(&mut self, entries: &[(u64, u32)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let page_bytes = self.kv.storage_bytes_per_page();
        let keys = entries.iter().map(|&(key, _)| key).collect::<Vec<_>>();
        let buf = self
            .tier
            .read_many_concat(&keys)
            .map_err(|err| anyhow::anyhow!("KV tier promote: {err}"))?;
        ensure!(
            buf.len() == entries.len() * page_bytes,
            "batched KV promote returned {} bytes, expected {}",
            buf.len(),
            entries.len() * page_bytes
        );
        let pages = entries.iter().map(|&(_, page)| page).collect::<Vec<_>>();
        self.kv
            .copy_pages_from_host_on_copy_stream(&self.model.ctx, &pages, &buf)?;
        self.model.ctx.sync_copy()?;
        Ok(())
    }

    pub(crate) fn drop_kv_tier_entries(&mut self, keys: &[u64]) {
        self.tier.remove(keys);
    }

    /// **Write-through**: mirror a filled device `page` into the host tier under
    /// `key`, so a later evict-drop of that page is free (the tier keeps the
    /// source of truth). Reuses the same `KvTierStore` as the prefix tier —
    /// there is ONE session-keyed store (R5), not a parallel one. `key` is a
    /// `(session, block)` pair flattened to the store's `u64` namespace by
    /// [`tier_block_u64`].
    ///
    /// Synchronous today (the copy is complete on return, matching
    /// `demote_prefix_pages`); R4's side-stream async mirror is the remaining perf
    /// step — see the pending-remote wins entry. Returns `false` (page not
    /// mirrored, MUST NOT be evict-dropped) when the tier is full.
    pub(crate) fn write_through(&mut self, key: u64, page: u32) -> Result<bool> {
        if self.tier.is_full() {
            return Ok(false);
        }
        let payload = self.kv.copy_pages_to_host(&self.model.ctx, &[page])?;
        Ok(self.tier.insert(key, payload))
    }

    /// **Prefetch**: load blocks from the host tier back into freshly allocated
    /// device pages (`(key, page)`), complete on return. Identical transport to
    /// `promote_prefix_pages`; the difference is the entry point
    /// (relevance-prefetch at prefill vs prefix-hit promote), per R5.
    #[allow(dead_code)] // WIP: R5 relevance-prefetch entry point, not yet wired
    pub(crate) fn prefetch_pages(&mut self, entries: &[(u64, u32)]) -> Result<()> {
        self.promote_prefix_pages(entries)
    }

    pub(crate) fn submit(
        &mut self,
        plan: &ForwardPlan,
        host_kv: &mut dyn KvPool,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<StepOutput> {
        ensure!(
            host_kv.page_size() == SUPPORTED_PAGE_SIZE,
            "host CudaKvPool page_size={} does not match CUDA BF16 page_size={SUPPORTED_PAGE_SIZE}",
            host_kv.page_size()
        );

        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        if rows == 0 {
            return Ok(StepOutput { tokens: Vec::new() });
        }
        // Multi-row plans (continuous batching): N prefill rows + M decode rows,
        // total > 1. Includes the MIXED case (a new request prefilling while
        // another decodes). Prefill rows run sequentially as single-row sub-steps,
        // then the M decode rows run as ONE batch. total == 1 falls through to the
        // single-row fast path below (byte-identical, incl. captured-graph/recall).
        if rows > 1 {
            return self.submit_multi_row(plan, host_kv, kv_batch);
        }
        ensure!(
            kv_batch.rows.len() == 1,
            "KV batch descriptor carries {} rows for a single-row plan",
            kv_batch.rows.len()
        );
        let kv_row = &kv_batch.rows[0];
        // Host page table covering [0, append_end) for this row — the engine
        // already allocated the append span in the host pool.
        let pages = &kv_batch.flat_page_ids[kv_row.page_range.clone()];

        let token = if let Some(row) = plan.prefill_rows.first() {
            ensure!(
                row.slot < self.num_slots,
                "prefill slot {} outside CUDA executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
            let expected_len = row.start_pos + row.tokens.len();
            ensure!(
                kv_row.slot == row.slot
                    && kv_row.append_pos == row.start_pos
                    && kv_row.append_len == row.tokens.len(),
                "KV batch row (slot {}, append {}+{}) does not match prefill row (slot {}, {}+{})",
                kv_row.slot,
                kv_row.append_pos,
                kv_row.append_len,
                row.slot,
                row.start_pos,
                row.tokens.len()
            );
            self.advance_slot_progress(row.slot, kv_row.slot_epoch, row.start_pos, expected_len)?;
            self.kv.mirror_slot(row.slot, pages, expected_len)?;
            let position = expected_len as u64;
            self.model.forward_tokens(
                row.slot,
                &row.tokens,
                row.start_pos,
                &mut self.kv,
                &row.params,
                position,
            )?
        } else {
            let row = &plan.decode_rows[0];
            ensure!(
                row.slot < self.num_slots,
                "decode slot {} outside CUDA executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                kv_row.slot == row.slot
                    && kv_row.append_pos == row.kv_seq_len
                    && kv_row.append_len == 1,
                "KV batch row (slot {}, append {}+{}) does not match decode row (slot {}, kv_seq_len {})",
                kv_row.slot,
                kv_row.append_pos,
                kv_row.append_len,
                row.slot,
                row.kv_seq_len
            );
            self.advance_slot_progress(
                row.slot,
                kv_row.slot_epoch,
                row.kv_seq_len,
                row.kv_seq_len + 1,
            )?;
            self.kv.mirror_slot(row.slot, pages, row.kv_seq_len + 1)?;
            let position = row.kv_seq_len.saturating_add(1) as u64;
            // Session KV-recall (eager-only, BF16-only): when on and this slot has
            // an active recall plan, attend the restricted page table + rescore
            // for the next step, then evict-drop the non-working-set middle pages
            // out of HBM (write-through tiered KV — the flat-VRAM win). Off / no
            // plan → the byte-identical default below.
            if let Some(token) = self.try_recall_decode(row, position, host_kv)? {
                token
            } else {
                match self.try_captured_decode(row.slot, row.last_token, row.kv_seq_len)? {
                    Some(()) => self.sample_decode_logits(&row.params, position)?,
                    None => self.model.forward_tokens(
                        row.slot,
                        &[row.last_token],
                        row.kv_seq_len,
                        &mut self.kv,
                        &row.params,
                        position,
                    )?,
                }
            }
        };

        Ok(StepOutput {
            tokens: vec![SlotToken {
                slot: plan
                    .prefill_rows
                    .first()
                    .map(|r| r.slot)
                    .unwrap_or_else(|| plan.decode_rows[0].slot),
                token,
                logprob: None,
                finish: None,
            }],
        })
    }

    /// Multi-row plan (continuous batching), total rows > 1. Handles pure-decode
    /// (N=0, M>1) and MIXED (N>=1 prefill + M>=0 decode) alike: the M decode
    /// rows run as ONE batch FIRST, then prefill rows run SEQUENTIALLY as
    /// single-row sub-steps. Decode-first minimises TTFT and ITL for in-flight
    /// requests — prefill sub-steps are expensive and must not stall decode.
    ///
    /// Plan rows always address disjoint slots (a request is either Prefilling or
    /// Decoding), so execution order is irrelevant for KV correctness. The
    /// KvBatchDescriptor layout is unchanged (prefill rows at 0..n_prefill, decode
    /// rows at n_prefill..total); only the GPU submission order is swapped.
    fn submit_multi_row(
        &mut self,
        plan: &ForwardPlan,
        host_kv: &mut dyn KvPool,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<StepOutput> {
        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        ensure!(
            kv_batch.rows.len() == rows,
            "KV batch descriptor has {} rows for a {rows}-row plan",
            kv_batch.rows.len()
        );
        let mut seen_slots = std::collections::BTreeSet::new();
        for slot in plan
            .prefill_rows
            .iter()
            .map(|row| row.slot)
            .chain(plan.decode_rows.iter().map(|row| row.slot))
        {
            ensure!(
                seen_slots.insert(slot),
                "CUDA plan schedules slot {slot} more than once per tick"
            );
        }

        let n_prefill = plan.prefill_rows.len();
        let mut tokens = Vec::with_capacity(rows);
        if !plan.decode_rows.is_empty() {
            let sub_batch = kv_batch.subset(n_prefill..kv_batch.rows.len())?;
            tokens.extend(self.submit_decode_batch(&plan.decode_rows, host_kv, &sub_batch)?);
        }
        for (idx, row) in plan.prefill_rows.iter().enumerate() {
            let sub_batch = kv_batch.subset(idx..idx + 1)?;
            tokens.push(self.submit_prefill_row(row, &sub_batch)?);
        }
        Ok(StepOutput { tokens })
    }

    /// One prefill row as its own single-row sub-step (mirror + forward_tokens).
    /// `kv_batch` is the row's single-row (sub-)descriptor — indistinguishable from
    /// what a prefill-only tick delivers. Byte-identical to the single-row prefill
    /// arm of [`Self::submit`].
    fn submit_prefill_row(
        &mut self,
        row: &infer_plan::PrefillRow,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<SlotToken> {
        ensure!(
            kv_batch.rows.len() == 1,
            "prefill sub-batch carries {} rows (expected 1)",
            kv_batch.rows.len()
        );
        let kv_row = &kv_batch.rows[0];
        let pages = &kv_batch.flat_page_ids[kv_row.page_range.clone()];
        ensure!(
            row.slot < self.num_slots,
            "prefill slot {} outside CUDA executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
        let expected_len = row.start_pos + row.tokens.len();
        ensure!(
            kv_row.slot == row.slot
                && kv_row.append_pos == row.start_pos
                && kv_row.append_len == row.tokens.len(),
            "KV batch row (slot {}, append {}+{}) does not match prefill row (slot {}, {}+{})",
            kv_row.slot,
            kv_row.append_pos,
            kv_row.append_len,
            row.slot,
            row.start_pos,
            row.tokens.len()
        );
        self.advance_slot_progress(row.slot, kv_row.slot_epoch, row.start_pos, expected_len)?;
        self.kv.mirror_slot(row.slot, pages, expected_len)?;
        let position = expected_len as u64;
        let token = self.model.forward_tokens(
            row.slot,
            &row.tokens,
            row.start_pos,
            &mut self.kv,
            &row.params,
            position,
        )?;
        Ok(SlotToken {
            slot: row.slot,
            token,
            logprob: None,
            finish: None,
        })
    }

    /// Run `decode_rows` (M >= 1) as ONE batched decode. `kv_batch` is the
    /// decode-only (sub-)descriptor whose rows align by index with `decode_rows`.
    ///
    /// BF16 + single-GPU + recall-inactive → the true batched path (one prep +
    /// one batched paged attention over `(M, M, 1)`, MLP/norm batched over M rows,
    /// per-row sample). Any of TP-collective / quant KV / active recall falls back
    /// to per-row sequential decode (the single-row machinery, correctness floor)
    /// rather than crash. Output order matches `decode_rows`.
    fn submit_decode_batch(
        &mut self,
        decode_rows: &[DecodeRow],
        host_kv: &mut dyn KvPool,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        let batch = decode_rows.len();
        ensure!(
            kv_batch.rows.len() == batch,
            "KV batch descriptor carries {} rows for a {}-row decode plan",
            kv_batch.rows.len(),
            batch
        );

        // Per-row validate + mirror is shared by both lanes (the device KV pool
        // must hold every row's just-appended token before any forward reads it).
        for (row, kv_row) in decode_rows.iter().zip(&kv_batch.rows) {
            ensure!(
                row.slot < self.num_slots,
                "decode slot {} outside CUDA executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                kv_row.slot == row.slot
                    && kv_row.append_pos == row.kv_seq_len
                    && kv_row.append_len == 1,
                "KV batch row (slot {}, append {}+{}) does not match decode row (slot {}, kv_seq_len {})",
                kv_row.slot,
                kv_row.append_pos,
                kv_row.append_len,
                row.slot,
                row.kv_seq_len
            );
            let pages = &kv_batch.flat_page_ids[kv_row.page_range.clone()];
            self.advance_slot_progress(
                row.slot,
                kv_row.slot_epoch,
                row.kv_seq_len,
                row.kv_seq_len + 1,
            )?;
            self.kv.mirror_slot(row.slot, pages, row.kv_seq_len + 1)?;
        }

        // Correctness floor: the batched paged kernels are BF16 single-GPU only,
        // and recall needs the per-row restricted page table the batched meta does
        // not carry. Any of these → per-row sequential decode.
        let recall_on = self.kv_recall && self.kv.format == KVFormat::BF16;
        let can_batch =
            self.kv.format == KVFormat::BF16 && !self.model.tp.is_collective() && !recall_on;
        if !can_batch {
            let mut tokens = Vec::with_capacity(batch);
            for row in decode_rows {
                let position = row.kv_seq_len.saturating_add(1) as u64;
                // Recall reuses its restricted-table path when active.
                let token = if let Some(token) = self.try_recall_decode(row, position, host_kv)? {
                    token
                } else {
                    self.model.forward_tokens(
                        row.slot,
                        &[row.last_token],
                        row.kv_seq_len,
                        &mut self.kv,
                        &row.params,
                        position,
                    )?
                };
                tokens.push(SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    finish: None,
                });
            }
            return Ok(tokens);
        }

        let last_tokens: Vec<u32> = decode_rows.iter().map(|r| r.last_token).collect();
        let params: Vec<SamplingParams> = decode_rows.iter().map(|r| r.params.clone()).collect();
        let positions: Vec<u64> = decode_rows
            .iter()
            .map(|r| r.kv_seq_len.saturating_add(1) as u64)
            .collect();
        let batch_rows: Vec<(usize, usize)> = decode_rows
            .iter()
            .map(|r| (r.slot, r.kv_seq_len + 1))
            .collect();
        let meta = PageMeta::for_decode_batch(&self.model.ctx, &self.kv, &batch_rows)?;
        let out_tokens = self.model.forward_decode_batch(
            &last_tokens,
            &mut self.kv,
            &meta,
            &params,
            &positions,
        )?;
        ensure!(
            out_tokens.len() == batch,
            "batched decode returned {} tokens for {batch} rows",
            out_tokens.len()
        );
        Ok(decode_rows
            .iter()
            .zip(out_tokens)
            .map(|(row, token)| SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// Warmup the B=1 decode graph before serving.
    ///
    /// Captures the smallest shape (`num_pages = 1`) so the machinery is proven
    /// before the first request; later page counts capture lazily on first decode
    /// (capture-once, replay-after, per key). Opt-in via
    /// `--cuda-graph`; any capture failure downgrades to eager-only
    /// (never fatal — eager is the correctness floor).
    pub(crate) fn warmup(&mut self) -> Result<()> {
        // NCCL all-reduce is not graph-capturable, so multi-rank TP stays eager
        // (a captured graph would silently skip the collective → wrong logits).
        // CUDA-graph decode is a single-GPU optimization.
        if self.model.tp.is_collective() {
            info!(
                "CUDA decode graph disabled under tensor parallelism \
                 (world_size>1, NCCL collectives are not graph-capturable); \
                 using eager forward"
            );
            return Ok(());
        }
        if self.kv.format != KVFormat::BF16 {
            info!(
                "CUDA decode graph disabled for quantized KV (format {:?}); decode runs eager \
                 through the fused-dequant kernels (#68 T3 V1)",
                self.kv.format
            );
            return Ok(());
        }
        if !decode_graph_enabled() {
            info!("CUDA decode graph disabled (set --cuda-graph to enable)");
            return Ok(());
        }
        if self.decode_ctx.is_some() {
            return Ok(()); // idempotent
        }
        match self.build_and_capture_warmup() {
            Ok(()) => {
                info!("CUDA B=1 decode graph captured (warmup, num_pages=1)");
                Ok(())
            }
            Err(e) => {
                warn!("CUDA decode graph warmup capture failed, falling back to eager: {e}");
                // Drop any half-built state so submit stays on the eager path.
                self.decode_ctx = None;
                self.graphs = None;
                Ok(())
            }
        }
    }

    /// Allocate the fixed buffers + capture `num_pages = 1` on a dummy slot.
    /// Isolated so a capture failure leaves the executor eager-only.
    fn build_and_capture_warmup(&mut self) -> Result<()> {
        let cfg = &self.model.config;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let ctx = DecodeGraphContext::new(
            &self.model.ctx,
            cfg.hidden_size,
            q_dim,
            kv_dim,
            cfg.intermediate_size,
            self.model.logits_dim(),
            self.kv.page_size,
            DECODE_GRAPH_MAX_SEQ_LEN,
        )?;
        self.decode_ctx = Some(ctx);
        self.graphs = Some(GraphBucket::new(self.model.ctx.stream.clone()));

        // Mirror dummy slot 0 onto page 0 for a single token so the page table
        // is valid for num_pages = 1, capture, then clear the view so serving
        // starts clean. The capture writes throwaway KV into page 0's rows; the
        // first real occupant of that page overwrites them. The device
        // allocator is never used — the host pool is the single allocator and
        // serving rows arrive via `mirror_slot`.
        ensure!(self.num_slots > 0, "warmup needs at least one slot");
        let dummy_slot = 0usize;
        self.kv.mirror_slot(dummy_slot, &[0], 1)?;
        // Two passes: `CudaGraphState` warms one eager run before its first
        // capture (universal JIT-in-capture guard), so a single call here
        // would only warm and silently push the capture onto the first real
        // request. First call = eager warm (also flushes lazy module loads
        // outside capture), second = the boot-time capture this warmup
        // promises.
        let capture_result = self
            .capture_decode_for_current_state(dummy_slot, 0, 0)
            .and_then(|()| self.capture_decode_for_current_state(dummy_slot, 0, 0));
        self.kv.mirror_slot(dummy_slot, &[], 0)?;
        capture_result?;
        self.model.ctx.sync()?;
        Ok(())
    }

    /// Session KV-recall decode step (#3/#4/#5), eager-only and BF16-only.
    ///
    /// Returns `Ok(Some(token))` when recall handled the step (restricted page
    /// table + rescore for the next step), `Ok(None)` to fall through to the
    /// default graph/eager decode. No-op (`None`) unless `--kv-recall` is on AND
    /// the session has exceeded the working-set budget — below the budget recall
    /// is a strict no-op, so the default path stays byte-identical. Non-BF16 KV
    /// falls back (logged once); recall is BF16-gated.
    fn try_recall_decode(
        &mut self,
        row: &DecodeRow,
        position: u64,
        host_kv: &mut dyn KvPool,
    ) -> Result<Option<u32>> {
        if !self.kv_recall {
            return Ok(None);
        }
        if self.kv.format != KVFormat::BF16 {
            if !self.recall_quant_warned {
                warn!(
                    "--kv-recall requested with a {:?} KV pool; recall is BF16-only — \
                     falling back to full attention (use --kv-cache-dtype bf16 to enable recall)",
                    self.kv.format
                );
                self.recall_quant_warned = true;
            }
            return Ok(None);
        }
        let cfg = self.recall_cfg;
        let cache_len = row.kv_seq_len + 1; // includes this step's appended token
        // Below the working-set budget recall is a strict no-op (mirrors
        // `plan_recall` returning the full contiguous range) → default path.
        if cache_len <= cfg.working_set_tokens() {
            return Ok(None);
        }

        // Correctness invariant for the restricted page table: the decode kernel
        // treats every page EXCEPT the last in `kv_indices` as FULL (`page_size`
        // tokens) and only the last as partial (`kv_last_page_len`). With all the
        // recall region boundaries (`n_init`, `l_bs`, `n_local`) multiples of
        // `page_size`, every range start/end is page-aligned except the final
        // local-window end (= `cache_len`), whose last page IS the current partial
        // page — so the only partial page is the last selected one, matching the
        // kernel. A future config that breaks this alignment would silently
        // mis-attend, so fail loud here.
        let ps = self.kv.page_size;
        ensure!(
            cfg.n_init.is_multiple_of(ps)
                && cfg.n_local.is_multiple_of(ps)
                && cfg.l_bs.is_multiple_of(ps),
            "KV-recall config (n_init {}, n_local {}, l_bs {}) must be multiples of \
             the KV page_size {} so the restricted page table has only its LAST page partial",
            cfg.n_init,
            cfg.n_local,
            cfg.l_bs,
            ps
        );

        // Page list for this step's attention: the slot's recall plan from the
        // previous step (stale-Q), or the full page list on the first recall step
        // (no plan yet) so the forward is still correct while we seed scoring.
        let recall_pages: Vec<u32> = match self.recall.get(row.slot).and_then(|s| s.recall_pages())
        {
            Some(p) => p.to_vec(),
            None => {
                let num_pages = cache_len.div_ceil(self.kv.page_size);
                self.kv.page_indices(row.slot)[..num_pages].to_vec()
            }
        };
        let recall_meta =
            PageMeta::for_recall_decode(&self.model.ctx, &self.kv, cache_len, &recall_pages)?;
        let (token, layer0_query) = self.model.forward_decode_recall(
            row.last_token,
            &mut self.kv,
            &recall_meta,
            &row.params,
            position,
        )?;

        // Score this step's query against the resident reps and plan the NEXT
        // step's recall (stale-Q, licensed).
        let num_q_heads = self.model.local_q_heads;
        let num_kv_heads = self.model.local_kv_heads;
        let head_dim = self.model.config.head_dim;
        let evict_pages = if let Some(state) = self.recall.get_mut(row.slot) {
            state.recompute_recall_plan(
                &self.model.ctx,
                &self.kv,
                row.slot,
                cache_len,
                &cfg,
                num_q_heads,
                num_kv_heads,
                head_dim,
                &layer0_query,
                // Dense arm has no L3 recall tier wired into the decode path:
                // evicted blocks stay -inf (no mid-decode re-recall). Byte-identical
                // to the prior behavior.
                /* allow_prefetch = */
                false,
            )?;
            state.take_evict_pages()
        } else {
            Vec::new()
        };

        // Write-through evict-drop (THE flat-VRAM win): free the cold middle pages
        // outside the working set out of HBM. For each, mirror it to the tier
        // (write_through, async D2H — the page already has a durable copy after
        // this) and only then return the physical page to BOTH the device pool and
        // the host single-allocator. The logical page table keeps its length (an
        // evict sentinel marks the slot), so `mirror_slot`/`SlotProgress` stay
        // valid. If the tier is full we skip that page (never lose KV).
        self.evict_drop_recall_pages(row.slot, &evict_pages, host_kv)?;

        Ok(Some(token))
    }

    /// Free the given logical pages of `slot` out of HBM (write-through tiered KV).
    ///
    /// Reused by [`Self::try_recall_decode`]. For each logical page: read its
    /// physical id, mirror it to the tier (so the drop is free), then evict-drop it
    /// from the device pool (`PagedKVPool::evict_slot_page` → recycles the HBM page)
    /// and the host single-allocator (`KvAllocator::evict_slot_page` → returns it to
    /// the free stack). Both pools' logical page tables stay the same length (an
    /// `EVICTED_PAGE` sentinel marks the freed slot), keeping the slot sparse
    /// without breaking the `mirror_slot` page-count or `SlotProgress` contiguity
    /// contracts. A tier-full `write_through` skips that page (KV must not be lost).
    fn evict_drop_recall_pages(
        &mut self,
        slot: usize,
        evict_pages: &[usize],
        host_kv: &mut dyn KvPool,
    ) -> Result<()> {
        for &logical in evict_pages {
            let table = self.kv.page_indices(slot);
            let Some(&physical) = table.get(logical) else {
                continue;
            };
            if physical == cuda_kernels::prelude::EVICTED_PAGE {
                continue; // already freed
            }
            // Mirror to the tier first; only drop if it took a durable copy.
            let key = tier_block_u64(slot as u64, logical as u64);
            if !self.write_through(key, physical)? {
                continue; // tier full → keep page resident (no KV loss)
            }
            // Free the physical page from BOTH pools (the real free). Host first so
            // the device pool's `page_indices` (read above) is still valid for the
            // device call; both replace the logical slot with the evict sentinel.
            let host_freed = host_kv.evict_slot_page(slot, logical);
            let dev_freed = self.kv.evict_slot_page(slot, logical);
            debug_assert_eq!(
                host_freed, dev_freed,
                "host/device pools disagreed on evicted page for slot {slot} logical {logical}"
            );
        }
        Ok(())
    }

    /// Write Stage-1 metadata and replay (or lazily capture) the decode graph for
    /// this step's page count. Returns `Ok(Some(()))` when the graph wrote
    /// `decode_ctx.logits` (sample from there), `Ok(None)` for eager fallback
    /// (disabled, or page count over budget). The step's KV token must already be
    /// allocated.
    fn try_captured_decode(
        &mut self,
        slot: usize,
        token: u32,
        kv_seq_len: usize,
    ) -> Result<Option<()>> {
        if self.decode_ctx.is_none() || self.graphs.is_none() {
            return Ok(None);
        }
        // Page count over budget → eager (never a stale replay).
        let total_len = kv_seq_len + 1;
        let num_pages = total_len.div_ceil(self.kv.page_size);
        if num_pages > DECODE_GRAPH_MAX_SEQ_LEN.div_ceil(self.kv.page_size) {
            return Ok(None);
        }
        self.capture_decode_for_current_state(slot, token, kv_seq_len)?;
        Ok(Some(()))
    }

    /// Stage-1 write + `run_or_capture` for the current decode state. Captures on
    /// first sight of a `num_pages` shape, replays after; a new page count gets
    /// its own graph entry.
    fn capture_decode_for_current_state(
        &mut self,
        slot: usize,
        token: u32,
        kv_seq_len: usize,
    ) -> Result<()> {
        // Split borrows: stage1 needs &kv + &mut decode_ctx; replay needs &model +
        // &mut kv + &mut decode_ctx + &mut graphs.
        let key: DecodeGraphKey = {
            let decode_ctx = self
                .decode_ctx
                .as_mut()
                .expect("decode_ctx present (checked by caller / warmup)");
            decode_ctx.stage1_write(&self.model.ctx, &self.kv, slot, token, kv_seq_len)?
        };
        debug_assert_eq!(key.batch_size, DECODE_GRAPH_BATCH);

        let model = &self.model;
        let kv = &mut self.kv;
        let decode_ctx = self
            .decode_ctx
            .as_mut()
            .expect("decode_ctx present (checked by caller / warmup)");
        let graphs = self
            .graphs
            .as_mut()
            .expect("graphs present (checked by caller / warmup)");
        let graph = graphs.entry(key.num_pages);
        graph.state.run_or_capture(|| {
            model.forward_decode_captured(kv, decode_ctx)?;
            Ok(())
        })
    }

    /// Sample the next token from the captured graph's fixed logits buffer.
    /// Sampling stays outside the graph (replay ends at `decode_ctx.logits`).
    fn sample_decode_logits(&self, params: &SamplingParams, position: u64) -> Result<u32> {
        let decode_ctx = self
            .decode_ctx
            .as_ref()
            .expect("decode_ctx present when sampling captured logits");
        sample_cuda_token(&self.model.ctx, &decode_ctx.logits, params, position)
    }

    /// Loud-error continuity guard replacing the old device-allocator
    /// materialized checks (see [`QwenCudaExecutor::slot_progress`]).
    ///
    /// Same occupant epoch ⇒ this executor must already have materialized
    /// exactly `append_pos` tokens (chunked prefill and decode append
    /// contiguously). A new epoch is a fresh occupant: `append_pos == 0` is a
    /// normal fresh prefill; `append_pos > 0` means the engine attached a radix
    /// prefix whose retained pages still hold the publishing request's KV rows
    /// (the host pool never recycled them), so the occupant starts materialized
    /// at that length.
    fn advance_slot_progress(
        &mut self,
        slot: usize,
        epoch: u64,
        append_pos: usize,
        end: usize,
    ) -> Result<()> {
        let progress = &mut self.slot_progress[slot];
        if progress.epoch == epoch {
            ensure!(
                progress.len == append_pos,
                "CUDA slot {slot} epoch {epoch} materialized {} tokens but the plan resumes at {append_pos} (non-contiguous append)",
                progress.len
            );
        } else if self.kv_recall {
            // New occupant: drop the prior session's recall reps + plan so the
            // fresh request starts from the byte-identical default page table.
            if let Some(state) = self.recall.get_mut(slot) {
                state.reset();
            }
        }
        self.slot_progress[slot] = SlotProgress { epoch, len: end };
        Ok(())
    }
}
