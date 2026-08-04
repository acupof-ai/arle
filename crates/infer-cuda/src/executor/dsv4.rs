use super::*;

const MAX_PENDING_PREFIX_CAPTURES: usize = 2;

struct PendingPrefixPage {
    source_page: u32,
    target_page: u32,
    confirmed: bool,
    cancelled: bool,
    entry: crate::attention::Dsv4PrefixPageEntry,
    frontier_tail: Option<Vec<u32>>,
}

impl PendingPrefixPage {
    fn confirm(&mut self, pages: &[u32]) {
        self.confirmed |= pages.contains(&self.target_page);
    }

    fn cancel_provisional(&mut self, pages: &[u32]) {
        self.cancelled |= !self.confirmed && pages.contains(&self.source_page);
    }

    fn repair(&mut self, canonical: u32, own: u32, canonical_exists: bool) {
        if self.source_page != own || self.cancelled {
            return;
        }
        if canonical_exists {
            self.cancelled = true;
        } else {
            self.target_page = canonical;
            self.confirmed = true;
        }
    }
}

fn capture_epoch_matches(captured: u64, current: Option<u64>) -> bool {
    current == Some(captured)
}

fn rekey_target_conflicts(source_page: u32, target_page: u32, target_exists: bool) -> bool {
    source_page != target_page && target_exists
}

struct PendingPrefixCapture {
    slot: usize,
    slot_epoch: u64,
    slot_pages: Vec<u32>,
    fence: CudaPipelineFence,
    pages: Vec<PendingPrefixPage>,
}

#[cfg(test)]
mod pending_prefix_tests {
    use super::*;

    fn page(source_page: u32) -> PendingPrefixPage {
        PendingPrefixPage {
            source_page,
            target_page: source_page,
            confirmed: false,
            cancelled: false,
            entry: crate::attention::Dsv4PrefixPageEntry {
                page_index: 0,
                boundary: false,
                layers: Vec::new(),
            },
            frontier_tail: None,
        }
    }

    #[test]
    fn pending_capture_rekeys_or_cancels_before_page_recycle() {
        let mut rekeyed = page(7);
        rekeyed.repair(9, 7, false);
        assert_eq!(rekeyed.target_page, 9);
        assert!(rekeyed.confirmed);
        rekeyed.cancel_provisional(&[7]);
        assert!(!rekeyed.cancelled);
        assert!(rekey_target_conflicts(7, 9, true));
        assert!(!rekey_target_conflicts(7, 9, false));
        assert!(!rekey_target_conflicts(9, 9, true));

        let mut recycled = page(11);
        recycled.cancel_provisional(&[11]);
        assert!(recycled.cancelled);
        assert!(capture_epoch_matches(3, Some(3)));
        assert!(!capture_epoch_matches(3, Some(4)));
    }
}

/// DSv4-Flash executor: drives [`crate::dsv4::Dsv4Model::forward_tokens`].
/// Prefill/mixed still run one scheduled row. Pure decode uses B=1 as the
/// single-row reference and B>1 as the canonical layer-major batched lane for
/// MODEL1 FlashMLA decode. DSv4 owns its MLA KV state inside the forward (bf16
/// SW rings + compressor pending/compressed pools), so it does NOT use a
/// [`PagedKVPool`]. The decode graph is disabled (MLA host-routing per step).
pub(crate) struct Dsv4CudaExecutor {
    pub(crate) model: crate::dsv4::Dsv4Model,
    pub(crate) slots: Vec<crate::dsv4::Dsv4SlotState>,
    pub(crate) kv_adapter: crate::attention::Dsv4KvAdapter,
    pub(crate) spec_slots: Vec<Dsv4SpecSlotState>,
    /// `Some(n)` = config-driven MTP spec decode on (draft depth `n`, from the
    /// serve path's `--spec-type mtp`); `None` = no MTP head.
    pub(crate) spec_draft_tokens: Option<usize>,
    /// MTP draft candidate width. `None`/`Some(1)` keeps chain-only candidates;
    /// `Some(k>1)` widens the draft matrix while verifier rows stay chain-shaped.
    pub(crate) spec_draft_topk: Option<usize>,
    pub(crate) num_slots: usize,
    pub(crate) mtp_accepts: usize,
    pub(crate) mtp_rejects: usize,
    /// Verified MTP draft chains (one per spec row that reached verify).
    pub(crate) mtp_chains: usize,
    /// Adaptive MTP gate (B=1): EMA of per-step accepted/depth, plus the count of
    /// consecutive gated skips since the last real spec step (drives the periodic
    /// probe). MTP only beats no-spec when it emits > t_mtp/t_nospec tok/step, so
    /// below that acceptance the gate runs a warm no-spec step instead. Init
    /// optimistic (1.0) so MTP runs until the running acceptance proves it loses.
    /// See `mtp_should_speculate`. Opt-in via `--mtp-adaptive` (bring-up;
    /// promote to a `--mtp-adaptive` CLI flag once pod-calibrated).
    pub(crate) mtp_accept_ema: f32,
    pub(crate) mtp_skip_streak: usize,
    /// Content-keyed cross-request prefix-state pool (#154 Phase 2): one
    /// host-resident entry per completed host page, written from the
    /// post-forward choke point, consumed by the prefix restore path. Also
    /// the preemption store (Phase 2b): published pages are already captured
    /// here, so preempt = free pages + requeue, resume = prefix-attach.
    prefix_state: crate::attention::Dsv4PrefixStatePool,
    pending_prefix_captures: VecDeque<PendingPrefixCapture>,
    /// DSpark block-drafter state (`--spec-type dspark`). `Some` = the draft is
    /// loaded and greedy B=1 decode routes through the DSpark
    /// draft→verify→accept loop instead of MTP; `None` = MTP or no spec. The
    /// per-slot draft caches ride here (serial spec state, like `spec_slots`).
    pub(crate) dspark: Option<Dsv4DsparkExec>,
}

/// Loaded DSpark drafter + its per-slot runtime caches. Present only under
/// `--spec-type dspark`; off the DSpark path this whole struct is absent (zero
/// cost). `draft` is the 3-stage block drafter weights; `slots` holds one
/// per-request latent cache + attention state set, indexed by slot.
pub(crate) struct Dsv4DsparkExec {
    draft: crate::dsv4::Dsv4DsparkDraft,
    /// Verify-step cost model driving the goodput budget (`--dspark-sps-*-ms`).
    sps: qwen35_spec::DsparkSps,
    /// Per-request token ceiling — the block spec step falls back to a single
    /// non-spec token when the worst-case chain would cross it.
    max_seq_len: usize,
    slots: Vec<Dsv4DsparkRuntime>,
}

/// One slot's DSpark draft runtime: the per-stage latent KV cache (`df`), the
/// pooled block-shaped scratch, and one attention state per draft stage
/// (`config.dspark_num_stages()`). Pre-allocated at construction — spec steps
/// are serial, so exact-shape reuse (no per-step alloc) holds.
struct Dsv4DsparkRuntime {
    df: crate::dsv4::dspark::Dsv4DsparkSlotState,
    scratch: crate::dsv4::dspark::Dsv4DsparkScratch,
    attn_states: Vec<crate::attention::Dsv4LayerAttentionState>,
}

/// One demoted slot: the device-state image plus the executor-level MTP spec
/// chain. `spec_pending`/`spec_hidden` MUST ride along (not reset): under
/// `--spec-type mtp` the resumed decode hard-requires the pending token and
/// the previous MTP stream (`forward_decode_tokens` errors on a missing
/// pending), and the slot's spec state is overwritten by whichever request
/// occupies the slot while this one is demoted.
#[derive(Default)]
pub(crate) struct Dsv4SpecSlotState {
    pub(crate) pending: Option<u32>,
    pub(crate) hidden: Option<DeviceVec>,
}

#[derive(Clone)]
struct Dsv4DecodeBatchRow {
    slot: usize,
    last_token: u32,
    start_pos: usize,
    position: u64,
    params: SamplingParams,
}

struct Dsv4DecodeBatch {
    slot_ids: Vec<usize>,
    tokens: Vec<u32>,
    start_positions: Vec<usize>,
    positions: Vec<u64>,
    rows: Vec<Dsv4DecodeBatchRow>,
}

impl Dsv4DecodeBatch {
    fn from_rows(
        rows: &[DecodeRow],
        slots: &[crate::dsv4::Dsv4SlotState],
        num_slots: usize,
    ) -> Result<Self> {
        let mut seen = vec![false; num_slots];
        let mut slot_ids = Vec::with_capacity(rows.len());
        let mut tokens = Vec::with_capacity(rows.len());
        let mut start_positions = Vec::with_capacity(rows.len());
        let mut positions = Vec::with_capacity(rows.len());
        let mut batch_rows = Vec::with_capacity(rows.len());
        for row in rows {
            ensure!(
                row.slot < num_slots,
                "decode slot {} outside DSv4 executor slots {}",
                row.slot,
                num_slots
            );
            ensure!(
                !seen[row.slot],
                "DSv4 decode batch contains duplicate slot {}",
                row.slot
            );
            seen[row.slot] = true;
            ensure!(
                slots[row.slot].seq_len() == row.kv_seq_len,
                "DSv4 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
                slots[row.slot].seq_len(),
                row.kv_seq_len,
                row.slot
            );
            let position = row.kv_seq_len.saturating_add(1) as u64;
            slot_ids.push(row.slot);
            tokens.push(row.last_token);
            start_positions.push(row.kv_seq_len);
            positions.push(position);
            batch_rows.push(Dsv4DecodeBatchRow {
                slot: row.slot,
                last_token: row.last_token,
                start_pos: row.kv_seq_len,
                position,
                params: row.params.clone(),
            });
        }
        Ok(Self {
            slot_ids,
            tokens,
            start_positions,
            positions,
            rows: batch_rows,
        })
    }
}

fn validate_dsv4_decode_kv_batch(rows: &[DecodeRow], kv_batch: &KvBatchDescriptor) -> Result<()> {
    ensure!(
        kv_batch.rows.len() == rows.len(),
        "DSv4 decode KV batch row count {} != plan rows {}",
        kv_batch.rows.len(),
        rows.len()
    );
    for (idx, (plan_row, kv_row)) in rows.iter().zip(&kv_batch.rows).enumerate() {
        ensure!(
            kv_row.kind == KvBatchRowKind::Decode,
            "DSv4 decode KV batch row {idx} has kind {:?}",
            kv_row.kind
        );
        ensure!(
            kv_row.slot == plan_row.slot,
            "DSv4 decode KV batch row {idx} slot {} != plan slot {}",
            kv_row.slot,
            plan_row.slot
        );
        ensure!(
            kv_row.seq_len == plan_row.kv_seq_len && kv_row.append_pos == plan_row.kv_seq_len,
            "DSv4 decode KV batch row {idx} seq/append ({},{}) != plan kv_seq_len {}",
            kv_row.seq_len,
            kv_row.append_pos,
            plan_row.kv_seq_len
        );
        ensure!(
            kv_row.append_len == 1,
            "DSv4 decode KV batch row {idx} append_len {} != 1",
            kv_row.append_len
        );
        ensure!(
            kv_row.page_range.start < kv_row.page_range.end,
            "DSv4 decode KV batch row {idx} has empty page range"
        );
        let tokens = &kv_batch.flat_token_ids[kv_row.token_range.clone()];
        ensure!(
            tokens == [plan_row.last_token],
            "DSv4 decode KV batch row {idx} tokens {:?} != plan token {}",
            tokens,
            plan_row.last_token
        );
    }
    Ok(())
}

fn validate_dsv4_prefill_kv_batch(
    row: &infer_plan::PrefillRow,
    kv_batch: &KvBatchDescriptor,
) -> Result<()> {
    ensure!(
        kv_batch.rows.len() == 1,
        "DSv4 prefill KV batch row count {} != 1",
        kv_batch.rows.len()
    );
    let kv_row = &kv_batch.rows[0];
    ensure!(
        kv_row.kind == KvBatchRowKind::Prefill,
        "DSv4 prefill KV batch row has kind {:?}",
        kv_row.kind
    );
    ensure!(
        kv_row.slot == row.slot,
        "DSv4 prefill KV batch slot {} != plan slot {}",
        kv_row.slot,
        row.slot
    );
    ensure!(
        kv_row.seq_len == row.start_pos && kv_row.append_pos == row.start_pos,
        "DSv4 prefill KV batch seq/append ({},{}) != plan start_pos {}",
        kv_row.seq_len,
        kv_row.append_pos,
        row.start_pos
    );
    ensure!(
        kv_row.append_len == row.tokens.len(),
        "DSv4 prefill KV batch append_len {} != plan token count {}",
        kv_row.append_len,
        row.tokens.len()
    );
    ensure!(
        kv_row.page_range.start < kv_row.page_range.end,
        "DSv4 prefill KV batch has empty page range"
    );
    let tokens = &kv_batch.flat_token_ids[kv_row.token_range.clone()];
    ensure!(
        tokens == row.tokens.as_slice(),
        "DSv4 prefill KV batch tokens do not match plan tokens"
    );
    Ok(())
}

fn validate_dsv4_decode_kv_view(
    rows: &[DecodeRow],
    view: &crate::attention::Dsv4KvBatchView,
) -> Result<()> {
    ensure!(
        view.rows.len() == rows.len(),
        "DSv4 decode KV adapter view row count {} != plan rows {}",
        view.rows.len(),
        rows.len()
    );
    for (idx, (plan_row, view_row)) in rows.iter().zip(&view.rows).enumerate() {
        ensure!(
            view_row.kind == KvBatchRowKind::Decode,
            "DSv4 decode KV adapter row {idx} has kind {:?}",
            view_row.kind
        );
        ensure!(
            view_row.slot == plan_row.slot,
            "DSv4 decode KV adapter row {idx} slot {} != plan slot {}",
            view_row.slot,
            plan_row.slot
        );
        ensure!(
            view_row.seq_len == plan_row.kv_seq_len
                && view_row.append_pos == plan_row.kv_seq_len
                && view_row.append_len == 1,
            "DSv4 decode KV adapter row {idx} seq/append ({},{},{}) != plan kv_seq_len {}",
            view_row.seq_len,
            view_row.append_pos,
            view_row.append_len,
            plan_row.kv_seq_len
        );
        ensure!(
            view_row.page_range.end <= view.flat_page_ids.len(),
            "DSv4 decode KV adapter row {idx} page range {:?} outside flat page len {}",
            view_row.page_range,
            view.flat_page_ids.len()
        );
        let pages = &view.flat_page_ids[view_row.page_range.clone()];
        ensure!(
            !pages.is_empty(),
            "DSv4 decode KV adapter row {idx} has no page ids"
        );
        ensure!(
            view_row.slot_page_range.end <= view.flat_slot_page_ids.len(),
            "DSv4 decode KV adapter row {idx} slot page range {:?} outside flat slot page len {}",
            view_row.slot_page_range,
            view.flat_slot_page_ids.len()
        );
        let slot_pages = &view.flat_slot_page_ids[view_row.slot_page_range.clone()];
        ensure!(
            !slot_pages.is_empty(),
            "DSv4 decode KV adapter row {idx} has no slot page ids"
        );
    }
    Ok(())
}

fn validate_dsv4_prefill_kv_view(
    row: &infer_plan::PrefillRow,
    view: &crate::attention::Dsv4KvBatchView,
) -> Result<()> {
    ensure!(
        view.rows.len() == 1,
        "DSv4 prefill KV adapter view row count {} != 1",
        view.rows.len()
    );
    let view_row = &view.rows[0];
    ensure!(
        view_row.kind == KvBatchRowKind::Prefill,
        "DSv4 prefill KV adapter row has kind {:?}",
        view_row.kind
    );
    ensure!(
        view_row.slot == row.slot
            && view_row.seq_len == row.start_pos
            && view_row.append_pos == row.start_pos
            && view_row.append_len == row.tokens.len(),
        "DSv4 prefill KV adapter row slot/seq/append ({},{},{},{}) != plan ({},{},{})",
        view_row.slot,
        view_row.seq_len,
        view_row.append_pos,
        view_row.append_len,
        row.slot,
        row.start_pos,
        row.tokens.len()
    );
    ensure!(
        view_row.page_range.end <= view.flat_page_ids.len(),
        "DSv4 prefill KV adapter row page range {:?} outside flat page len {}",
        view_row.page_range,
        view.flat_page_ids.len()
    );
    let pages = &view.flat_page_ids[view_row.page_range.clone()];
    ensure!(
        !pages.is_empty(),
        "DSv4 prefill KV adapter row has no page ids"
    );
    ensure!(
        view_row.slot_page_range.end <= view.flat_slot_page_ids.len(),
        "DSv4 prefill KV adapter row slot page range {:?} outside flat slot page len {}",
        view_row.slot_page_range,
        view.flat_slot_page_ids.len()
    );
    let slot_pages = &view.flat_slot_page_ids[view_row.slot_page_range.clone()];
    ensure!(
        !slot_pages.is_empty(),
        "DSv4 prefill KV adapter row has no slot page ids"
    );
    Ok(())
}

impl std::fmt::Debug for Dsv4CudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dsv4CudaExecutor")
            .field("model", &self.model)
            .field("num_slots", &self.num_slots)
            .finish()
    }
}

impl Dsv4CudaExecutor {
    /// Gate active only for demand-paged bands; identity/V32 draw the full
    /// band at slot alloc.
    pub(crate) fn kv_device_gate_active(&self) -> bool {
        self.kv_adapter.flashmla_demand_paged()
    }

    /// Device-budget plan gate (#160): per-layer, per-row band fit.
    /// Lockstep-safe without a collective: band pools mutate only from the
    /// rank-identical plan stream, so every rank computes the same fit.
    pub(crate) fn kv_device_fit(
        &self,
        rows: &[infer_seam::DeviceRowDemand],
        unfit: &mut Vec<usize>,
    ) {
        self.kv_adapter.flashmla_demand_fit(rows, unfit);
    }

    fn tp_min_usize(&self, value: usize, what: &str) -> Result<usize> {
        let capped = i32::try_from(value.min(i32::MAX as usize)).unwrap_or(i32::MAX);
        self.model
            .tp
            .all_reduce_min_scalar_i32(&self.model.ctx, capped)
            .map(|v| v.max(0) as usize)
            .map_err(|e| anyhow::anyhow!("DSv4 TP min-reduce {what} failed: {e}"))
    }

    pub(crate) fn from_dsv4_fp8_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        max_seq_len: usize,
        mtp_draft_tokens: Option<usize>,
        mtp_draft_topk: Option<usize>,
        dspark_draft_model: Option<&Path>,
        dspark_sps_bias_ms: f32,
        dspark_sps_row_ms: f32,
    ) -> Result<Self> {
        ensure!(num_slots > 0, "Dsv4CudaExecutor requires at least one slot");
        ensure!(max_seq_len > 0, "Dsv4CudaExecutor requires max_seq_len > 0");
        let mtp_draft_tokens_for_load = mtp_draft_tokens
            .or_else(|| mtp_draft_topk.map(|_| crate::dsv4::DEFAULT_SPEC_DRAFT_DEPTH));
        // Pass the draft dir through: the builder reads its config for DSpark
        // metadata and derives `spec_decode_on` (which allocates the per-slot
        // spec-ring snapshots MTP and DSpark share).
        let mut model = crate::dsv4::Dsv4Model::from_dsv4_fp8_safetensors(
            model_path.as_ref(),
            mtp_draft_tokens_for_load,
            dspark_draft_model,
        )?;
        // [vram-probe] TEMP bit-exact VRAM attribution (remove after budget unification).
        // Returns measured-used bytes (or None when mem_get_info fails) so the
        // ledger can reconcile predicted device_bytes() against the measured used.
        let mem_dbg = |tag: &str| -> Option<usize> {
            match cudarc::driver::result::mem_get_info() {
                Ok((free, total)) => {
                    let used = total - free;
                    log::info!(
                        "[vram-probe] {tag}: used {}MB free {}MB",
                        used >> 20,
                        free >> 20
                    );
                    Some(used)
                }
                Err(_) => None,
            }
        };
        let base_weights_used = mem_dbg("after base model load (weights+experts)");
        let dspark_draft = dspark_draft_model
            .map(|draft_dir| -> Result<_> {
                let loader = crate::loader::SafetensorLoader::new(draft_dir)?;
                loader.prefetch_shards_rank0(&model.ctx, &model.tp)?;
                let draft = crate::dsv4::load_dspark_draft(
                    &loader,
                    &model.ctx,
                    &model.config,
                    &model.split,
                    model.tp.config(),
                )?;
                model.ctx.sync()?;
                let after = mem_dbg("after DSpark draft weights load");
                if let (Some(before), Some(after)) = (base_weights_used, after) {
                    log::info!(
                        "[vram-ledger] DSpark weights: +{}MB ({}MB -> {}MB)",
                        (after as i64 - before as i64) >> 20,
                        before >> 20,
                        after >> 20,
                    );
                }
                Ok(draft)
            })
            .transpose()?;
        // Reclaim the cuMemAllocAsync pool BEFORE measuring free VRAM: weight
        // loading allocs+frees large device scratch (FP8 dequant, DeepGEMM cache
        // build, staging), and the retain-threshold=MAX pool holds it — so
        // `mem_get_info` in the budget would count freed loading scratch as USED
        // and starve the KV slot count. Trim returns it to the OS so the budget
        // sees the true free (recovers the KV slots those GB should fund).
        if let Err(e) = model.ctx.trim_memory_pool() {
            log::warn!("pre-KV-budget trim_memory_pool failed (non-fatal): {e}");
        }
        let weights_used_at_model_load = mem_dbg("after weight-load pool trim");
        // DSpark per-slot runtime (latent KV + draft attention states) is allocated
        // after kv_budget_plan; count it here so num_slots doesn't over-commit.
        let dspark_per_slot_bytes = if dspark_draft.is_some() {
            let num_stages = model.config.dspark_num_stages();
            let block_size = model.config.dspark_block_size;
            let head_dim = model.config.head_dim;
            // Sliding-window draft latent: fixed `window + block`, no prompt growth.
            let draft_span = model.config.sliding_window + block_size;
            // latent_kv: num_stages buffers, each [draft_span * head_dim] bf16.
            let latent_kv = num_stages
                .saturating_mul(draft_span)
                .saturating_mul(head_dim)
                .saturating_mul(std::mem::size_of::<half::bf16>());
            // Draft attention states: num_stages KV caches at draft_span. Rough
            // estimate; the adapter's per_slot already covers the trunk layers.
            let attn = num_stages
                .saturating_mul(draft_span)
                .saturating_mul(head_dim)
                .saturating_mul(2);
            latent_kv.saturating_add(attn)
        } else {
            0
        };
        let budget = model.kv_budget_plan(num_slots, max_seq_len, dspark_per_slot_bytes)?;
        let num_slots = budget.num_slots;
        let kv_adapter = model.new_kv_adapter(max_seq_len, budget)?;
        mem_dbg("after new_kv_adapter (KV pools)");
        // [vram-ledger] PREDICTED adapter device bytes + per-component breakdown.
        log::info!(
            "[vram-ledger] adapter predicted {}MB; breakdown {:?}",
            kv_adapter.device_bytes() >> 20,
            kv_adapter
                .device_bytes_breakdown()
                .iter()
                .map(|(name, bytes)| (*name, bytes >> 20))
                .collect::<Vec<_>>()
        );
        let mut slots = Vec::with_capacity(num_slots);
        for slot_idx in 0..num_slots {
            slots.push(model.new_slot_state(max_seq_len, slot_idx, &kv_adapter)?);
            if slot_idx == 0 {
                mem_dbg("after slot 0 (per-slot state)");
                // [vram-ledger] PREDICTED slot-0 device bytes + per-component breakdown.
                log::info!(
                    "[vram-ledger] slot0 predicted {}MB; breakdown {:?}",
                    slots[0].device_bytes() >> 20,
                    slots[0]
                        .device_bytes_breakdown()
                        .iter()
                        .map(|(name, bytes)| (*name, bytes >> 20))
                        .collect::<Vec<_>>()
                );
                // [vram-ledger] Attribute the per-slot attention bulk to the exact
                // buffer family, summed across all layers (every bit's source named).
                log::info!(
                    "[vram-ledger] slot0 attention sub-totals (Σ layers) {:?}",
                    slots[0]
                        .attention_breakdown_total()
                        .iter()
                        .map(|(name, bytes)| (*name, bytes >> 20))
                        .collect::<Vec<_>>()
                );
                // Drift guard: the KV budget divides free VRAM by the STATIC
                // `per_slot_device_bytes`; if it drifts from the real slot alloc,
                // `affordable` mis-clamps num_slots and engine build OOMs (the
                // 43→382 MB under-count). Warn on >5% so it can't silently return.
                let predicted = model.per_slot_device_bytes(max_seq_len)?;
                let actual = slots[0].device_bytes();
                let drift = (predicted as i64 - actual as i64).unsigned_abs() as usize;
                if drift.saturating_mul(20) > actual {
                    log::warn!(
                        "[vram-ledger] DSv4 per-slot budget drift {}%: static per_slot_device_bytes {}MB vs \
                         slot0 device_bytes {}MB — reconcile per_slot_device_bytes with Dsv4SlotState::new",
                        drift.saturating_mul(100) / actual.max(1),
                        predicted >> 20,
                        actual >> 20,
                    );
                }
            }
        }
        let measured_used_after_all = mem_dbg("after all slots (build complete)");
        // [vram-ledger] Reconcile PREDICTED cumulative device_bytes against the
        // measured used. residual = measured_used
        //   - (weights_used_at_model_load + adapter.device_bytes() + Σ slot.device_bytes()).
        // The residual is everything NOT in the named-buffer ledger: CUDA context
        // + library reservations + per-cudaMalloc allocation rounding across the
        // ~258 tiny per-layer allocs/slot. A large residual points the gap there,
        // a small one means the named buffers fully account for the slot cost.
        let adapter_bytes = kv_adapter.device_bytes();
        let slots_bytes: usize = slots.iter().map(|s| s.device_bytes()).sum();
        log::info!(
            "[vram-ledger] cumulative predicted: weights {}MB + adapter {}MB + Σ {} slots {}MB = {}MB",
            weights_used_at_model_load.map_or(0, |b| b >> 20),
            adapter_bytes >> 20,
            num_slots,
            slots_bytes >> 20,
            (weights_used_at_model_load.unwrap_or(0) + adapter_bytes + slots_bytes) >> 20
        );
        if let (Some(measured), Some(weights)) =
            (measured_used_after_all, weights_used_at_model_load)
        {
            let predicted_total = weights + adapter_bytes + slots_bytes;
            // Saturating signed residual: usually positive (ctx/libs/rounding),
            // but guard the rare measured < predicted (measurement skew).
            let residual_mb = (measured as i64 - predicted_total as i64) >> 20;
            log::info!(
                "[vram-ledger] residual (ctx+libs+cudaMalloc rounding) = {residual_mb}MB \
                 (measured used {}MB - predicted {}MB)",
                measured >> 20,
                predicted_total >> 20
            );
        }
        let spec_slots = (0..num_slots)
            .map(|_| Dsv4SpecSlotState::default())
            .collect();
        let layer_specs: Vec<_> = model
            .layers
            .iter()
            .map(|layer| (layer.mode, layer.compress_ratio))
            .collect();
        let prefix_entry_bytes = crate::attention::dsv4_prefix_entry_max_bytes(
            &model.config,
            &layer_specs,
            model.kv_arena.page_block_size,
        );
        // DSpark weights were loaded before KV budgeting; only per-slot runtime
        // state is allocated here.
        let dspark = dspark_draft
            .map(|draft| {
                Self::load_dspark_exec(
                    &model,
                    &kv_adapter,
                    draft,
                    dspark_sps_bias_ms,
                    dspark_sps_row_ms,
                    max_seq_len,
                    num_slots,
                )
            })
            .transpose()?;
        model.boot_mega_moe(num_slots.max(256))?;
        let exec = Self {
            model,
            slots,
            kv_adapter,
            spec_slots,
            spec_draft_tokens: mtp_draft_tokens_for_load,
            spec_draft_topk: mtp_draft_topk,
            num_slots,
            mtp_accepts: 0,
            mtp_rejects: 0,
            mtp_chains: 0,
            mtp_accept_ema: 1.0,
            mtp_skip_streak: 0,
            prefix_state: crate::attention::Dsv4PrefixStatePool::new(
                default_t1_budget_bytes(DEFAULT_DRAM_FRACTION),
                prefix_entry_bytes,
            ),
            pending_prefix_captures: VecDeque::new(),
            dspark,
        };
        log::info!(
            "DSv4 prefill chunk capability: {} tokens (ARLE_DSV4_PREFILL_CHUNK {:?}, deepep \
             per-forward cap {:?})",
            exec.max_prefill_chunk(),
            crate::runtime_flags::dsv4_prefill_chunk(),
            exec.model.max_tokens_per_step(),
        );
        Ok(exec)
    }

    /// Provision per-slot caches for the preloaded DSpark drafter. The
    /// main checkpoint's config MUST be DSpark-capable (`is_dspark()`, i.e. it
    /// carries `dspark_block_size` + target-layer ids so the T3 taps exist); the
    /// isolated MLA attention (`compress_ratio == 0`) gets its own attention state;
    /// the draft latent cache is sized to the request token ceiling.
    fn load_dspark_exec(
        model: &crate::dsv4::Dsv4Model,
        kv_adapter: &crate::attention::Dsv4KvAdapter,
        draft: crate::dsv4::Dsv4DsparkDraft,
        sps_bias_ms: f32,
        sps_row_ms: f32,
        max_seq_len: usize,
        num_slots: usize,
    ) -> Result<Dsv4DsparkExec> {
        ensure!(
            model.config.is_dspark(),
            "--spec-type dspark needs a DSpark-capable DSv4 checkpoint (config carries \
             dspark_block_size + dspark_target_layer_ids); this checkpoint is not one"
        );
        let num_stages = model.config.dspark_num_stages();
        let block_size = model.config.dspark_block_size;
        // Draft-stage attention is pure MLA (`compress_ratio == 0`, no CSA/HCA):
        // the same mode the trunk uses for a ratio-0 layer. The stage attends its
        // OWN `latent_kv`, not the trunk KV pool, so the pool arg is a bookkeeping
        // handle only (layer 0's), and `max_seq_len` sizes the draft's own span.
        let stage_mode = model.config.attention_mode_for_compress_ratio(0);
        let stage_pool = kv_adapter.layer(0)?;
        // Draft latent KV cache is sliding-window: fixed `sliding_window + block`
        // ring, independent of prompt length. Full-context draft latent OOMs on
        // long prompts (linear growth → router disabled DSpark on >64 tok).
        let draft_span = model.config.sliding_window + block_size;
        let slots = (0..num_slots)
            .map(|slot_idx| -> Result<_> {
                let context_capacity = model.config.sliding_window;
                let df = crate::dsv4::dspark::Dsv4DsparkSlotState::new(
                    &model.ctx,
                    &model.config,
                    num_stages,
                    context_capacity,
                    block_size,
                )?;
                let attn_states = draft
                    .stages
                    .iter()
                    .map(|stage| {
                        let local_width = stage.layer.attention.wq_b.rows;
                        crate::attention::Dsv4LayerAttentionState::new(
                            &model.ctx,
                            &model.config,
                            stage_mode,
                            0,
                            draft_span,
                            &model.kv_arena,
                            local_width / model.config.head_dim,
                            model.tp.config().world_size,
                            slot_idx,
                            stage_pool,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Dsv4DsparkRuntime {
                    df,
                    scratch: crate::dsv4::dspark::Dsv4DsparkScratch::default(),
                    attn_states,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sps = qwen35_spec::DsparkSps {
            bias_ms: sps_bias_ms,
            row_ms: sps_row_ms,
        };
        log::info!(
            "CUDA DSv4 DSpark runtime initialized: stages={num_stages} block={block_size} \
             sps={sps:?} target_layers={:?}",
            model.config.dspark_target_layer_ids,
        );
        Ok(Dsv4DsparkExec {
            draft,
            sps,
            max_seq_len,
            slots,
        })
    }

    /// Publish every host page this forward completed for `slot` into the
    /// content-keyed prefix-state pool (#154 Phase 2 write hook). Best-effort:
    /// a publish failure only forfeits a future reuse, never the forward.
    fn enqueue_prefix_capture(
        &mut self,
        slot: usize,
        slot_pages: &[u32],
        pages: Vec<PendingPrefixPage>,
    ) -> Result<()> {
        if pages.is_empty() {
            return Ok(());
        }
        let slot_epoch = self.kv_adapter.slot_epoch(slot).unwrap_or(u64::MAX);
        // A fence-record failure DROPS the capture (its pages are host-only
        // clones, released by plain drop — same as poll's cancelled path): a
        // fence-less entry at the front would never poll Ready, pinning
        // `prefix_capture_queue_full` true forever.
        let fence = self
            .model
            .ctx
            .record_pipeline_fence(CudaPipelineStreamKind::Compute)?;
        self.pending_prefix_captures
            .push_back(PendingPrefixCapture {
                slot,
                slot_epoch,
                slot_pages: slot_pages.to_vec(),
                fence,
                pages,
            });
        Ok(())
    }

    fn prefix_capture_queue_full(&self) -> bool {
        self.pending_prefix_captures.len() >= MAX_PENDING_PREFIX_CAPTURES
    }

    pub(crate) fn poll_prefix_captures(&mut self) {
        loop {
            let status = match self.pending_prefix_captures.front() {
                Some(capture) => capture.fence.query(),
                None => return,
            };
            match status {
                Ok(CudaPipelineFenceStatus::NotReady) => return,
                Err(err) => {
                    // Sticky query error: drop the capture so the queue drains
                    // instead of re-warning every tick behind a stuck front.
                    warn!("DSv4 prefix capture fence failed; dropping capture: {err:#}");
                    self.pending_prefix_captures.pop_front();
                    continue;
                }
                Ok(CudaPipelineFenceStatus::Ready) => {}
            }
            let capture = self
                .pending_prefix_captures
                .pop_front()
                .expect("front observed");
            let epoch_matches =
                capture_epoch_matches(capture.slot_epoch, self.kv_adapter.slot_epoch(capture.slot));
            let protected_pages = if epoch_matches {
                capture.slot_pages
            } else {
                capture
                    .pages
                    .iter()
                    .filter(|page| page.confirmed && !page.cancelled)
                    .map(|page| page.target_page)
                    .collect()
            };
            for page in capture.pages {
                let target_exists = self.prefix_state.page_meta(page.target_page).is_some();
                if page.cancelled
                    || (!page.confirmed && !epoch_matches)
                    || rekey_target_conflicts(page.source_page, page.target_page, target_exists)
                {
                    continue;
                }
                if !self
                    .prefix_state
                    .publish(page.target_page, &page.entry, &protected_pages)
                {
                    continue;
                }
                if page.confirmed {
                    self.prefix_state.confirm_pages(&[page.target_page]);
                }
                if let Some(tokens) = page.frontier_tail {
                    self.prefix_state
                        .set_frontier_tail(page.target_page, tokens);
                }
            }
        }
    }

    fn publish_completed_prefix_pages(
        &mut self,
        slot: usize,
        slot_pages: &[u32],
        start_pos: usize,
        end_pos: usize,
    ) {
        if self.prefix_state.is_inactive() {
            return;
        }
        self.poll_prefix_captures();
        if self.prefix_capture_queue_full() {
            return;
        }
        let page_tokens = self.model.kv_arena.page_block_size;
        if page_tokens == 0 || end_pos <= start_pos {
            return;
        }
        let align = self.model.config.sliding_window.max(1);
        // Phase 1: enqueue every page's D2H clones (stream-ordered, no sync).
        let mut captured = Vec::new();
        let mut capture_failed = false;
        for page_index in (start_pos / page_tokens)..(end_pos / page_tokens) {
            let page_end = (page_index + 1) * page_tokens;
            let Some(&page_id) = slot_pages.get(page_index) else {
                warn!(
                    "DSv4 prefix publish: slot {slot} page {page_index} outside host table ({} pages)",
                    slot_pages.len()
                );
                capture_failed = true;
                break;
            };
            // Boundary sections exist only when the forward ended exactly here;
            // restore commits only at `align` multiples, so skip odd page ends.
            // An MTP multi-token commit that CROSSES a boundary (page_end <
            // end_pos) also skips: the SW ring advances every token, so the
            // boundary-instant ring (and, once a compress block completes, the
            // overlap registers) is unrecoverable — correctness over coverage;
            // the presence-check (`reusable_prefix_blocks`) simply won't
            // commit there.
            let boundary = page_end == end_pos && page_end.is_multiple_of(align);
            match self.slots[slot].capture_prefix_page(
                &self.model.ctx,
                &self.model.layers,
                &self.kv_adapter,
                self.model.config.index_head_dim,
                page_tokens,
                page_index,
                boundary,
            ) {
                Ok(entry) => captured.push(PendingPrefixPage {
                    source_page: page_id,
                    target_page: page_id,
                    confirmed: false,
                    cancelled: false,
                    entry,
                    frontier_tail: None,
                }),
                Err(err) => {
                    warn!("DSv4 prefix publish failed for slot {slot} page {page_index}: {err:#}");
                    capture_failed = true;
                    break;
                }
            }
        }
        if captured.is_empty() {
            return;
        }
        if capture_failed {
            captured.iter_mut().for_each(|page| page.cancelled = true);
        }
        if let Err(err) = self.enqueue_prefix_capture(slot, slot_pages, captured) {
            warn!("DSv4 prefix publish fence failed for slot {slot}: {err:#}");
        }
    }

    /// Finish write-through (`--dsv4-decode-reuse`): publish content-keyed pool
    /// entries for the finished slot's whole sealed region + the frontier tail,
    /// so a later turn restores to the EXACT finish position instead of flooring
    /// at a page boundary. Publication waits on the capture's compute-stream
    /// fence; decode-lane per-tick publish is a no-op under this flag. Publishes
    /// PROVISIONAL onto the slot's own page ids exactly like the prefill publish;
    /// the finish's `save_prefix_sidecar` confirm/repair (running right after)
    /// reconciles them to the radix's canonical ids. Best-effort: a failure only
    /// forfeits a future reuse, never the finish.
    pub(crate) fn capture_finish_frontier(
        &mut self,
        slot: usize,
        tokens: &[u32],
        slot_pages: &[u32],
    ) -> Result<()> {
        if !crate::runtime_flags::dsv4_decode_reuse_enabled() || self.prefix_state.is_inactive() {
            return Ok(());
        }
        self.poll_prefix_captures();
        if self.prefix_capture_queue_full() {
            return Ok(());
        }
        let page_tokens = self.model.kv_arena.page_block_size;
        if page_tokens == 0 {
            return Ok(());
        }
        let finish_len = tokens.len().min(self.slots[slot].seq_len());
        let sealed_pages = finish_len / page_tokens;
        if sealed_pages == 0 {
            return Ok(()); // < one page: no radix anchor to reuse
        }
        let matched_len = sealed_pages * page_tokens;
        let index_head_dim = self.model.config.index_head_dim;
        let frontier = sealed_pages - 1;
        // Phase 1: enqueue D2H clones (stream-ordered, no sync). A content page
        // the prefill publish already stored is skipped; the frontier is always
        // (re)captured to attach its tail + carry at finish_len.
        let mut captured = Vec::new();
        let mut capture_failed = false;
        for page_index in 0..sealed_pages {
            let Some(&page_id) = slot_pages.get(page_index) else {
                warn!(
                    "DSv4 finish frontier: slot {slot} page {page_index} outside host table ({} pages)",
                    slot_pages.len()
                );
                capture_failed = true;
                break;
            };
            let is_frontier = page_index == frontier;
            if !is_frontier && self.prefix_state.page_meta(page_id).is_some() {
                continue;
            }
            // A tail-less chunk-end boundary entry licenses ANY continuation; a
            // finish-frontier recapture (tail-gated at finish_len) would veto every
            // diverging suffix (#166). Keep it — the finish forfeits only the sub-page tail.
            if is_frontier
                && self
                    .prefix_state
                    .page_meta(page_id)
                    .is_some_and(|m| m.boundary)
                && self.prefix_state.frontier_tail_tokens(page_id).is_none()
            {
                continue;
            }
            let entry = if is_frontier {
                self.slots[slot].capture_frontier_page(
                    &self.model.ctx,
                    &self.model.layers,
                    &self.kv_adapter,
                    index_head_dim,
                    page_tokens,
                    page_index,
                    matched_len,
                    finish_len,
                )
            } else {
                self.slots[slot].capture_prefix_page(
                    &self.model.ctx,
                    &self.model.layers,
                    &self.kv_adapter,
                    index_head_dim,
                    page_tokens,
                    page_index,
                    false,
                )
            };
            match entry {
                Ok(entry) => captured.push(PendingPrefixPage {
                    source_page: page_id,
                    target_page: page_id,
                    confirmed: false,
                    cancelled: false,
                    entry,
                    frontier_tail: (is_frontier && finish_len > matched_len)
                        .then(|| tokens[matched_len..finish_len].to_vec()),
                }),
                Err(err) => {
                    warn!(
                        "DSv4 finish frontier capture failed for slot {slot} page {page_index}: {err:#}"
                    );
                    capture_failed = true;
                    break;
                }
            }
        }
        if captured.is_empty() {
            return Ok(());
        }
        if capture_failed {
            captured.iter_mut().for_each(|page| page.cancelled = true);
        }
        self.enqueue_prefix_capture(slot, slot_pages, captured)?;
        Ok(())
    }

    /// Radix evicted these host pages: drop their pool entries (the pool's
    /// lifetime rides the radix, no independent lifetime above the DRAM budget).
    pub(crate) fn release_prefix_pages(&mut self, pages: &[u32]) {
        for capture in &mut self.pending_prefix_captures {
            for page in &mut capture.pages {
                page.cancelled |= pages.contains(&page.target_page);
            }
        }
        self.prefix_state.remove_pages(pages);
    }

    /// Slot free/abort returned these pages to the pool: drop provisional
    /// entries only (see `Dsv4PrefixStatePool::remove_provisional_pages`).
    pub(crate) fn release_provisional_prefix_pages(&mut self, pages: &[u32]) {
        for capture in &mut self.pending_prefix_captures {
            for page in &mut capture.pages {
                page.cancel_provisional(pages);
            }
        }
        self.prefix_state.remove_provisional_pages(pages);
    }

    /// Radix publish confirmed these pages (engine `save_prefix_sidecar` arm):
    /// flip their pool entries from provisional to readable.
    pub(crate) fn confirm_prefix_pages(&mut self, pages: &[u32]) {
        for capture in &mut self.pending_prefix_captures {
            for page in &mut capture.pages {
                page.confirm(pages);
            }
        }
        self.prefix_state.confirm_pages(pages);
    }

    /// #157 confirm-time repair. `canonical` is the radix chain the sidecar
    /// rides; `slot_pages` is the finishing slot's own chain at the same
    /// positions. Where dedup diverged them and the canonical entry is
    /// missing, adopt the slot's provisional entry under the canonical id.
    pub(crate) fn repair_prefix_pool_chain(&mut self, canonical: &[u32], slot_pages: &[u32]) {
        for (&canon, &own) in canonical.iter().zip(slot_pages) {
            if canon != own {
                let canonical_exists = self.prefix_state.page_meta(canon).is_some();
                for capture in &mut self.pending_prefix_captures {
                    for page in &mut capture.pages {
                        page.repair(canon, own, canonical_exists);
                    }
                }
                self.prefix_state.adopt_canonical(canon, own);
            }
        }
        self.poll_prefix_captures();
    }

    /// Attach the opt-in NVMe disk spill level (pre-serve only). The whole
    /// `--kv-disk` cap funds the prefix pool (Phase 2b deleted the whole-slot
    /// park store). The cap is soft: over it, publish drops the pool's oldest
    /// entries — capacity never blocks a forward and never enters the reuse
    /// license.
    pub(crate) fn set_kv_tier_disk(
        &mut self,
        root: std::path::PathBuf,
        budget_bytes: usize,
    ) -> bool {
        self.prefix_state.set_disk(root, budget_bytes)
    }

    /// Pre-serve re-budget: the whole `--kv-dram` share funds the prefix pool.
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        self.prefix_state.set_budget_bytes(bytes);
    }

    pub(crate) fn kv_tier_host_demoted_pages(&self) -> usize {
        self.prefix_state.host_pages()
    }

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        self.prefix_state.disk_pages()
    }

    pub(crate) fn kv_tier_read_hits(&self) -> infer_seam::KvTierReadHits {
        self.prefix_state.read_hits()
    }

    pub(crate) fn kv_tier_io_stats(&self) -> infer_seam::KvTierIoStats {
        let stats = self.prefix_state.io_stats();
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

    /// Content-keyed reuse license (#154 Phase 2): a leading page is
    /// attachable only while every page before and including it has a pool
    /// entry; the COMMIT point additionally requires the page to carry the
    /// carry (boundary) sections. Pages between commit points keep the scan
    /// alive without advancing it. Pool presence covers host DRAM and disk
    /// alike — licensing never does capacity math. Finish write-through
    /// (`--dsv4-decode-reuse`) captures the carry at the exact finish position
    /// (rides the frontier entry, not a ring boundary), so it commits at any
    /// carry-bearing page; the default path keeps the `sliding_window` gate.
    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        let page_tokens = self.model.kv_arena.page_block_size;
        if page_tokens == 0 {
            return 0;
        }
        let align = self.model.config.sliding_window.max(1);
        let reuse = crate::runtime_flags::dsv4_decode_reuse_enabled();
        let mut committed = 0usize;
        for (idx, block) in blocks.iter().enumerate() {
            // Fail closed on demoted keys: DSv4 pages never demote through
            // the radix tier (demote_prefix_pages returns 0).
            let PrefixBlock::ResidentPage(page_id) = *block else {
                break;
            };
            let Some(meta) = self.prefix_state.page_meta(page_id) else {
                break;
            };
            let page_end = (idx + 1) * page_tokens;
            if meta.boundary && (reuse || page_end.is_multiple_of(align)) {
                committed = idx + 1;
            }
        }
        committed
    }

    /// Request-admission reuse length: like [`Self::reusable_prefix_blocks`] but
    /// a finish-write-through frontier (a boundary page carrying a sub-page tail)
    /// commits ONLY when `prompt` is a verified continuation through the finish
    /// position — `prompt[page_end..page_end+tail] == entry.tail_tokens`. The
    /// radix proves content identity to the page boundary; the tail is a prior
    /// turn's continuation and a divergent/shorter prompt must not restore it
    /// (else a page-prefix collision injects a different request's KV, or the
    /// over-restore crashes on `seq_len > append_pos`). A tail-less boundary page
    /// (carry at its own page boundary) commits unconditionally. Flag OFF →
    /// strict page-aligned count, unchanged.
    pub(crate) fn reusable_prefix_blocks_for_prompt(
        &self,
        blocks: &[PrefixBlock],
        tokens: &[u32],
    ) -> usize {
        if !crate::runtime_flags::dsv4_decode_reuse_enabled() {
            return self.reusable_prefix_blocks(blocks);
        }
        let page_tokens = self.model.kv_arena.page_block_size;
        if page_tokens == 0 {
            return 0;
        }
        let mut committed = 0usize;
        for (idx, block) in blocks.iter().enumerate() {
            let PrefixBlock::ResidentPage(page_id) = *block else {
                break;
            };
            let Some(meta) = self.prefix_state.page_meta(page_id) else {
                break;
            };
            if !meta.boundary {
                continue;
            }
            let page_end = (idx + 1) * page_tokens;
            let commit = match self.prefix_state.frontier_tail_tokens(page_id) {
                // Carry at this page's own boundary — always valid.
                None => true,
                // Frontier tail: valid only for a verified continuation.
                Some(tail) => {
                    let finish = page_end + tail.len();
                    tokens.len() >= finish && tokens[page_end..finish] == *tail
                }
            };
            if commit {
                committed = idx + 1;
            }
        }
        committed
    }

    /// The boundary sections only exist at a forward's own end position —
    /// without this, a single-call prefill never visits an earlier boundary.
    pub(crate) fn prefill_restore_boundary_alignment(&self) -> usize {
        self.model.config.sliding_window.max(1)
    }

    /// Largest single prefill forward; also the restore-snapshot grain when it
    /// exceeds the boundary alignment. Default 2048 (plan Phase 3 flip,
    /// 2026-07-17 gates: c1 TTFT 3031→1088 ms, c16 out +138%, ITL p99 −293 ms,
    /// needle zero-miss ×2 passes). `ARLE_DSV4_PREFILL_CHUNK` overrides:
    /// clamped to `[128, DSV4_PREFILL_QUERY_CHUNK]` (the shared prefill
    /// scratch M bound), min'd with the deepep_ll per-forward token cap, and
    /// rounded down to a 128 multiple so chunk ends stay on the ring/snapshot
    /// alignment.
    pub(crate) fn max_prefill_chunk(&self) -> usize {
        const DEFAULT_PREFILL_CHUNK: usize = 2048;
        let grain = self.model.config.sliding_window.max(1);
        let requested = crate::runtime_flags::dsv4_prefill_chunk().unwrap_or(DEFAULT_PREFILL_CHUNK);
        let cap = requested
            .clamp(128, crate::attention::DSV4_PREFILL_QUERY_CHUNK)
            .min(self.model.max_tokens_per_step().unwrap_or(usize::MAX));
        (cap / 128 * 128).max(grain)
    }

    /// Restore a radix-matched prefix into `slot` from the content-keyed pool
    /// (#154 Phase 2 read hook; engine calls via the restore_prefix_sidecar
    /// seam AFTER attaching the host pages). Every entry decodes to owned
    /// host state first, so a missing/undecodable page aborts before any
    /// device byte moves; an `Err` propagates to the engine's fallback
    /// (free slot, full recompute) — never a partial restore.
    /// Returns the EXTRA tokens restored beyond `matched_len` (`0` = restored
    /// exactly the page-aligned match). Finish write-through (`--dsv4-decode-
    /// reuse`) restores the sub-page tail carried on the last entry, so the slot
    /// advances to the exact finish position and the engine prefills only past it.
    pub(crate) fn restore_prefix_state(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> Result<usize> {
        self.poll_prefix_captures();
        ensure!(
            slot < self.num_slots,
            "DSv4 prefix restore slot {slot} outside executor slots {}",
            self.num_slots
        );
        let page_tokens = self.model.kv_arena.page_block_size;
        let align = self.model.config.sliding_window.max(1);
        let reuse = crate::runtime_flags::dsv4_decode_reuse_enabled();
        ensure!(
            page_tokens > 0 && matched_len > 0,
            "DSv4 prefix restore needs a non-empty match (matched_len {matched_len})"
        );
        // D4 hard error: an unaligned match means the engine trim/clamp
        // ordering regressed — fail loud, never silently skip the ring restore.
        // Finish write-through reuses at any page boundary (the ring alignment
        // rides the frontier entry's captured carry, not `matched_len`), so it
        // only requires page alignment.
        ensure!(
            matched_len.is_multiple_of(page_tokens) && (reuse || matched_len.is_multiple_of(align)),
            "DSv4 prefix restore matched_len {matched_len} not aligned to ring {align} / page {page_tokens}"
        );
        ensure!(
            prefix_pages.len() == matched_len / page_tokens,
            "DSv4 prefix restore {} pages != matched_len {matched_len} / page {page_tokens}",
            prefix_pages.len()
        );
        let entries = prefix_pages
            .iter()
            .map(|&page_id| self.prefix_state.read_entry(page_id))
            .collect::<Result<Vec<_>>>()?;
        // The frontier entry may carry a sub-page tail (a finish write-through
        // that landed off a page boundary). Reuse it ONLY for a verified
        // continuation: the tail `[matched_len, finish_len)` has no radix key,
        // so the prompt must actually contain those exact tokens. A divergent or
        // shorter prompt falls back to the page-aligned match (`tail_len 0`) —
        // never restore a prior turn's tail. The carry lives only at the finish
        // position, so restore is atomic to `finish_len`.
        let frontier = *prefix_pages.last().expect("matched_len > 0 ⇒ ≥1 page");
        let tail_len = match self.prefix_state.frontier_tail_tokens(frontier) {
            Some(tail)
                if reuse
                    && tokens.len() >= matched_len + tail.len()
                    && tokens[matched_len..matched_len + tail.len()] == *tail =>
            {
                tail.len()
            }
            _ => 0,
        };
        let finish_len = matched_len + tail_len;
        self.kv_adapter
            .mirror_full_band(&self.model.ctx, slot, finish_len)?;
        // A17: a restored occupant enters Decoding without a tail warm step
        // for its MTP chain; stale spec state belongs to the prior occupant. The
        // DSpark draft latent cache is the same story — rebase it to the restored
        // frontier so the first block drafts against fresh, in-request features.
        self.spec_slots[slot] = Dsv4SpecSlotState::default();
        self.reset_dspark_slot(slot, finish_len);
        self.slots[slot].restore_prefix_state(
            &self.model.ctx,
            &self.model.layers,
            &mut self.kv_adapter,
            self.model.config.index_head_dim,
            &entries,
            matched_len,
            finish_len,
            page_tokens,
        )?;
        Ok(tail_len)
    }

    /// `BackendExecutor::tp_sync_min` — see there for why the scheduler needs
    /// this (2026-07-05 TP=4 admission livelock).
    pub(crate) fn tp_sync_min(&self, local: usize) -> Result<usize> {
        self.tp_min_usize(local, "admission free pages")
    }

    /// One no-spec greedy forward that ALSO stages the MTP draft state (pending
    /// token + stream hidden) so a subsequent `spec_step` can resume. Shared by
    /// final-prefill (seeds the first chain) and the adaptive gate's fallback (a
    /// low-acceptance step runs at no-spec cost but keeps the draft head warm).
    fn forward_mtp_warm_step(
        &mut self,
        slot_idx: usize,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        let (token, hidden) = self.model.forward_tokens_with_hidden(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            tokens,
            start_pos,
            params,
            position,
        )?;
        self.spec_slots[slot_idx].pending = Some(token);
        self.spec_slots[slot_idx].hidden = Some(hidden);
        Ok(token)
    }

    fn forward_prefill_tokens(
        &mut self,
        slot_idx: usize,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        final_prefill: bool,
    ) -> Result<Vec<u32>> {
        let spec_on = self.spec_requested();
        if spec_on && params.is_greedy() {
            let token = if final_prefill {
                self.forward_mtp_warm_step(slot_idx, tokens, start_pos, params, position)?
            } else {
                // Non-final chunk: emit the token, drop any stale draft state.
                let (token, _hidden) = self.model.forward_tokens_with_hidden(
                    &mut self.slots[slot_idx],
                    &mut self.kv_adapter,
                    tokens,
                    start_pos,
                    params,
                    position,
                )?;
                self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
                token
            };
            Ok(vec![token])
        } else {
            if spec_on {
                self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
            }
            Ok(vec![self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                tokens,
                start_pos,
                params,
                position,
            )?])
        }
    }

    fn forward_decode_tokens(
        &mut self,
        slot_idx: usize,
        last_token: u32,
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<Vec<u32>> {
        if !self.spec_requested() {
            let token = self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &[last_token],
                start_pos,
                params,
                position,
            )?;
            return Ok(vec![token]);
        }

        if !params.is_greedy() {
            self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
            let token = self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &[last_token],
                start_pos,
                params,
                position,
            )?;
            return Ok(vec![token]);
        }
        // DSpark block spec (`--spec-type dspark`): the draft→verify→accept loop
        // replaces MTP for greedy decode. Off the DSpark path (`dspark` is
        // `None`), MTP runs below byte-identically.
        if self.dspark.is_some() {
            return self.dspark_decode_tokens(slot_idx, last_token, start_pos, params, position);
        }
        // Self-heal an un-seeded / desynced MTP stream (#140). A request that
        // enters Decoding WITHOUT a tail prefill — a full position-0 prefix-cache
        // hit or a whole-slot promote, both of which reset `spec_slots` to
        // default (pending=None) and rely on a tail warm-step that never runs on
        // a full hit — reaches its first decode spec_step with no pending token.
        // The old code bailed the whole TP group (crash observed ~613 verify
        // ticks into sustained serving once the prefix cache warmed). Instead
        // re-seed via one warm no-spec step for `last_token`: it stages
        // pending + hidden and emits the token, so the NEXT step runs real MTP.
        // A pending that disagrees with `last_token` is a stream desync (the
        // staged hidden belongs to a different token) — same recovery, warned.
        let seeded = match self.spec_slots[slot_idx].pending {
            Some(pending) if pending == last_token => true,
            Some(pending) => {
                log::warn!(
                    "DSv4 MTP stream desync (slot {slot_idx}): pending {pending} != last_token \
                     {last_token}; re-seeding via a warm step"
                );
                false
            }
            None => false,
        };
        if !seeded {
            let token =
                self.forward_mtp_warm_step(slot_idx, &[last_token], start_pos, params, position)?;
            return Ok(vec![token]);
        }
        // Adaptive gate (B=1): when the running acceptance EMA predicts MTP would
        // lose to no-spec, run a warm no-spec step instead — keeps the draft head
        // staged so MTP resumes the moment acceptance recovers (a periodic probe
        // forces one real step to refresh the EMA). The warm step needs the MTP
        // hidden so it takes the eager forward (not the decode-graph fast path):
        // it costs eager-no-spec, a touch above t_nospec, which only narrows the
        // win — calibrate MIN_ACCEPT against that. NOTE: the EMA + skip_streak are
        // executor-global (fine for this B=1 bring-up flag); make them per-slot
        // before promoting the gate to a default.
        if self.mtp_adaptive_skip() {
            self.mtp_skip_streak += 1;
            let token =
                self.forward_mtp_warm_step(slot_idx, &[last_token], start_pos, params, position)?;
            return Ok(vec![token]);
        }
        self.spec_step(slot_idx, start_pos, position)
    }

    /// Reset the DSpark draft latent cache for `slot`, rebasing it to absolute
    /// trunk position `pos` (a request boundary: fresh prefill → 0, restored
    /// prefix → the restored frontier). No-op off the DSpark path.
    fn reset_dspark_slot(&mut self, slot: usize, pos: usize) {
        if let Some(ds) = self.dspark.as_mut() {
            ds.slots[slot].df.rebase(pos);
        }
    }

    /// Hot-swap the DSpark Markov head weights from a host f32 snapshot.
    ///
    /// Called by the train sidecar after each acceptance-weighted step. The new
    /// weights are converted to bf16 and uploaded; the device `data` pointer is
    /// flipped under the stream's ordering guarantee. `w1` is `[vocab * rank]`,
    /// `w2` is `[rank * vocab]`.
    pub(crate) fn update_dspark_markov_weights(&mut self, w1: &[f32], w2: &[f32]) -> Result<()> {
        let dspark = self
            .dspark
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("DSpark head not loaded"))?;
        let stage = dspark
            .draft
            .stages
            .last_mut()
            .ok_or_else(|| anyhow::anyhow!("DSpark draft has no stages"))?;
        let (Some(mw1), Some(mw2)) = (&mut stage.markov_w1, &mut stage.markov_w2) else {
            anyhow::bail!("DSpark exit stage has no Markov head weights");
        };
        let w1_len = mw1.rows * mw1.cols;
        let w2_len = mw2.rows * mw2.cols;
        ensure!(
            w1.len() == w1_len,
            "DSpark markov w1 size mismatch: got {}, expected {w1_len}",
            w1.len()
        );
        ensure!(
            w2.len() == w2_len,
            "DSpark markov w2 size mismatch: got {}, expected {w2_len}",
            w2.len()
        );
        let w1_bf16: Vec<half::bf16> = w1.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let w2_bf16: Vec<half::bf16> = w2.iter().map(|&x| half::bf16::from_f32(x)).collect();
        mw1.data = self
            .model
            .ctx
            .stream
            .clone_htod(&w1_bf16)
            .map_err(|e| anyhow::anyhow!("DSpark markov w1 H2D upload failed: {e}"))?;
        mw2.data = self
            .model
            .ctx
            .stream
            .clone_htod(&w2_bf16)
            .map_err(|e| anyhow::anyhow!("DSpark markov w2 H2D upload failed: {e}"))?;
        self.model.ctx.sync()?;
        Ok(())
    }

    /// Seed a freshly-prefilled prompt chunk into the DSpark draft context: pull
    /// the transient multi-row taps the prefill forward stashed on the slot and
    /// append them at absolute trunk positions `start_abs..` (the canonical
    /// `dspark_append_context` path). No-op off the DSpark path or when the
    /// forward captured no multi-row taps (single-token prompt / chunk).
    fn seed_dspark_prompt(&mut self, slot: usize, start_abs: usize) -> Result<()> {
        let Self {
            model,
            slots,
            dspark,
            ..
        } = self;
        let Some(ds) = dspark.as_mut() else {
            return Ok(());
        };
        let Some(taps) = slots[slot].take_dspark_prompt_taps() else {
            return Ok(());
        };
        model.dspark_append_context(
            &ds.draft,
            &mut ds.slots[slot].df,
            &taps.bufs,
            taps.rows,
            start_abs,
        )
    }

    fn tp_lockstep_proposal(
        model: &crate::dsv4::Dsv4Model,
        proposal: &mut crate::dsv4::dspark::Dsv4DsparkProposal,
    ) -> Result<()> {
        if model.tp.config().world_size <= 1 {
            return Ok(());
        }
        let block = model.config.dspark_block_size;
        let mut payload = Vec::with_capacity(2 + block);
        payload.push(proposal.draft_len as i32);
        payload.extend(proposal.chain.iter().map(|&t| t as i32));
        payload.resize(2 + block, 0);
        let r0 = model.tp.broadcast_rank0_i32(&model.ctx, &payload)?;
        let r0_chain: Vec<u32> = r0[1..2 + r0[0] as usize]
            .iter()
            .map(|&v| v as u32)
            .collect();
        if r0_chain != proposal.chain {
            proposal.draft_len = r0[0] as usize;
            proposal.chain = r0_chain;
        }
        Ok(())
    }

    fn tp_lockstep_accept(
        model: &crate::dsv4::Dsv4Model,
        accepted: usize,
        bonus: u32,
    ) -> Result<(usize, u32)> {
        if model.tp.config().world_size <= 1 {
            return Ok((accepted, bonus));
        }
        let r0 = model
            .tp
            .broadcast_rank0_i32(&model.ctx, &[accepted as i32, bonus as i32])?;
        Ok((r0[0] as usize, r0[1] as u32))
    }

    /// One DSpark block decode step (`--spec-type dspark`, greedy). Forward the
    /// last token to emit the block anchor + capture the T3 trunk taps, draft a
    /// whole block from those taps in ONE non-causal pass, verify the chain in one
    /// contiguous target forward, accept the longest greedy-matching prefix, and
    /// roll the rejected tail back. Mirrors the MTP `spec_step` bookkeeping
    /// (`capture_spec_rings` / `truncate_slot` / `restore_spec_ring_tail`),
    /// swapping the autoregressive MTP draft for the single-pass block drafter and
    /// the sparse frozen verify for the contiguous commit verify + truncate.
    ///
    /// Every committed token is a target greedy argmax (anchor + verified
    /// accepted drafts + bonus).
    fn dspark_decode_tokens(
        &mut self,
        slot_idx: usize,
        last_token: u32,
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<Vec<u32>> {
        // Sequence-cap fallback: the block verify appends the anchor + up to
        // `block_size` drafts (chain rows at `verify_pos .. verify_pos + block`),
        // plus a bonus row. Request normalization caps EMITTED tokens but NOT the
        // speculative chain, so near `max_seq_len` a full block would overflow the
        // KV/ring and abort an otherwise-valid request. Mirror the Qwen DSpark
        // guard: when the worst-case chain does not fit, take one non-spec token.
        let ds = self
            .dspark
            .as_ref()
            .expect("dspark_decode_tokens without dspark");
        let block = self.model.config.dspark_block_size;
        if start_pos + 1 + block + 1 > ds.max_seq_len {
            let token = self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &[last_token],
                start_pos,
                params,
                position,
            )?;
            return Ok(vec![token]);
        }
        // Anchor forward: a normal eager decode step commits `last_token`, emits
        // the block anchor (a real target greedy token), and — under `is_dspark()`
        // — populates `slot.dspark_taps()` at the target layers (no arming flag).
        let anchor = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[last_token],
            start_pos,
            params,
            position,
        )?;
        let verify_pos = start_pos + 1;

        // Draft the block, then build the proposal (tied-head logits → Markov
        // semi-AR sampling → confidence truncation). Split-borrow: model, slot
        // taps, and the draft + this slot's caches are disjoint fields.
        let mut proposal = {
            let Self {
                model,
                slots,
                dspark,
                ..
            } = self;
            let Dsv4DsparkExec {
                draft,
                sps,
                slots: ds_slots,
                ..
            } = dspark
                .as_mut()
                .expect("dspark_decode_tokens without dspark");
            let rt = &mut ds_slots[slot_idx];
            // Append the anchor-forward token to the accumulated draft context
            // (committed at absolute position `start_pos`, i.e. `verify_pos - 1`)
            // BEFORE the block forward — the canonical context path (prompt prefix
            // seeded at prefill, per-step anchor here).
            model.dspark_append_context(
                draft,
                &mut rt.df,
                slots[slot_idx].dspark_taps(),
                1,
                start_pos,
            )?;
            let block_hidden = model.dspark_forward_block(
                draft,
                &mut rt.df,
                &mut rt.scratch,
                &mut rt.attn_states,
                anchor,
                verify_pos,
            )?;
            model.dspark_build_proposal(draft, &block_hidden, anchor, params.temperature, *sps)?
        };
        ensure!(
            proposal.chain.len() == proposal.draft_len + 1 && proposal.chain[0] == anchor,
            "DSpark proposal chain {} != draft_len {} + 1 (anchor {anchor})",
            proposal.chain.len(),
            proposal.draft_len,
        );

        // TP lockstep: the rank-local proposal (confidence-truncated draft_len)
        // drifts by FP across ranks; a divergent length feeds verify a different
        // token count per rank → per-forward collective-count mismatch → deadlock.
        // Adopt rank 0's `[draft_len, chain..]` so every rank verifies one shape.
        Self::tp_lockstep_proposal(&self.model, &mut proposal)?;
        let draft_len = proposal.draft_len;

        self.model.capture_spec_rings(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            verify_pos,
            draft_len,
        )?;
        let verify = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &proposal.chain,
            verify_pos,
            position,
        )?;
        ensure!(
            verify.argmax.len() == proposal.chain.len(),
            "DSpark verify rows {} != chain {}",
            verify.argmax.len(),
            proposal.chain.len()
        );

        // Accept scan: `argmax[i]` is the target greedy token AFTER chain[i];
        // accept chain[i+1] while it matches. Longest matching prefix.
        let mut accepted = 0usize;
        while accepted < draft_len && verify.argmax[accepted] == proposal.chain[accepted + 1] {
            accepted += 1;
        }
        let mut bonus = verify.argmax[accepted];

        // TP lockstep: accepted/bonus are rank-local FP; adopt rank 0's so every
        // rank commits an identical KV tail (next tick's combined attention reads
        // all ranks' shards — an inconsistent commit corrupts it).
        (accepted, bonus) = Self::tp_lockstep_accept(&self.model, accepted, bonus)?;

        // Return to the real frontier, restore the aliased ring boundary, then
        // fold anchor + accepted drafts exactly once from persisted verify rows.
        self.model
            .truncate_slot(&mut self.slots[slot_idx], &mut self.kv_adapter, verify_pos)?;
        self.model.restore_spec_ring_tail(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            verify_pos,
            accepted,
            draft_len,
        )?;
        self.model.commit_accepted_fold(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            0..accepted + 1,
            verify_pos,
        )?;

        // The verify taps cover [anchor, drafts..]. Only its committed prefix
        // belongs in the next block's context; the rejected tail must not leak.
        {
            let Self {
                model,
                slots,
                dspark,
                ..
            } = self;
            let Dsv4DsparkExec {
                draft,
                slots: ds_slots,
                ..
            } = dspark
                .as_mut()
                .expect("dspark_decode_tokens without dspark");
            model.dspark_append_context(
                draft,
                &mut ds_slots[slot_idx].df,
                slots[slot_idx].dspark_taps(),
                accepted + 1,
                verify_pos,
            )?;
        }

        self.mtp_accepts += accepted;
        self.mtp_rejects += draft_len - accepted;
        self.mtp_chains += 1;
        if self.model.tp.config().rank == 0 {
            log::debug!(
                "[dsv4-dspark] slot={slot_idx} block={draft_len} accepted={accepted} \
                 bonus={bonus} accept_total={} reject_total={}",
                self.mtp_accepts,
                self.mtp_rejects
            );
            // Token-level accept attribution (INFER_DSPARK_DEBUG): the drafted
            // ids vs the target greedy argmax at each block position. `argmax[i]`
            // is the target token AFTER chain[i]; accept requires
            // argmax[i] == chain[i+1]. Reveals de-RoPE (plausible-but-shifted
            // drafts) vs alignment off-by-one (drafts match a shifted argmax) vs
            // geometry garbage (unrelated drafts).
            if std::env::var_os("INFER_DSPARK_DEBUG").is_some() {
                eprintln!(
                    "[dspark-dbg] anchor={anchor} base_argmax={:?} drafts={:?} target_argmax={:?} accepted={accepted}",
                    &proposal.base_argmax,
                    &proposal.chain[1..],
                    &verify.argmax,
                );
            }
        }

        // Emit the anchor (committed this step) + accepted drafts + the bonus (the
        // next block's anchor, committed by the next step's anchor forward). The
        // DSpark step is self-contained — clear any MTP spec carry.
        self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
        let mut out = Vec::with_capacity(accepted + 2);
        out.push(anchor);
        out.extend_from_slice(&proposal.chain[1..accepted + 1]);
        out.push(bonus);
        Ok(out)
    }

    /// Batched DSpark block decode for B>1. Drafts each slot's block independently
    /// (the 3-stage draft forward is per-slot), then verifies ALL chains in ONE
    /// batched target forward (`forward_decode_batch_verify`) instead of N serial
    /// verifies. Accept/commit remain per-slot (the proven fold path).
    ///
    /// This amortizes the heaviest phase (target-model verify) across the batch,
    /// matching the no-spec baseline's batched decode at B>1.
    fn dspark_decode_tokens_batched(
        &mut self,
        rows: &[Dsv4DecodeBatchRow],
    ) -> Result<Vec<Vec<u32>>> {
        let n = rows.len();
        let max_seq_len = self
            .dspark
            .as_ref()
            .expect("dspark_decode_tokens_batched without dspark")
            .max_seq_len;
        let block = self.model.config.dspark_block_size;

        // ── Phase 1a: separate fallback slots (sequence-cap) from DSpark slots.
        let mut proposals: Vec<Option<crate::dsv4::dspark::Dsv4DsparkProposal>> =
            (0..n).map(|_| None).collect();
        let mut verify_positions: Vec<usize> = vec![0; n];
        let mut fallback_tokens: Vec<Option<Vec<u32>>> = (0..n).map(|_| None).collect();
        let mut dspark_idxs: Vec<usize> = Vec::with_capacity(n);
        for (i, row) in rows.iter().enumerate() {
            if row.start_pos + 1 + block + 1 > max_seq_len {
                let token = self.model.forward_tokens(
                    &mut self.slots[row.slot],
                    &mut self.kv_adapter,
                    &[row.last_token],
                    row.start_pos,
                    &row.params,
                    row.position,
                )?;
                fallback_tokens[i] = Some(vec![token]);
            } else {
                dspark_idxs.push(i);
                verify_positions[i] = row.start_pos + 1;
            }
        }

        // ── Phase 1b: batched anchor forward (ONE target forward over N slots),
        // capturing DSpark taps in the batched decode lane. Replaces N serial
        // single-token target forwards.
        if !dspark_idxs.is_empty() {
            let slot_ids: Vec<usize> = dspark_idxs.iter().map(|&i| rows[i].slot).collect();
            let tokens: Vec<u32> = dspark_idxs.iter().map(|&i| rows[i].last_token).collect();
            let starts: Vec<usize> = dspark_idxs.iter().map(|&i| rows[i].start_pos).collect();
            let positions: Vec<u64> = dspark_idxs.iter().map(|&i| rows[i].position).collect();
            let params: Vec<SamplingParams> = dspark_idxs
                .iter()
                .map(|&i| rows[i].params.clone())
                .collect();
            let anchors = self.model.forward_decode_batch(
                &mut self.slots,
                &mut self.kv_adapter,
                &slot_ids,
                &tokens,
                &starts,
                &positions,
                &params,
                true, // anchor: the draft reads these taps
            )?;
            // ── Phase 1c: per-slot draft + proposal (draft is a small 3-stage
            // model; each slot has independent latent_kv context, so batching
            // the draft would require unifying variable-length context — not
            // worth the complexity vs the already-amortized target verify).
            for (k, &i) in dspark_idxs.iter().enumerate() {
                let anchor = anchors[k];
                let verify_pos = verify_positions[i];
                let proposal = {
                    let Self {
                        model,
                        slots,
                        dspark,
                        ..
                    } = self;
                    let Dsv4DsparkExec {
                        draft,
                        sps,
                        slots: ds_slots,
                        ..
                    } = dspark
                        .as_mut()
                        .expect("dspark_decode_tokens_batched without dspark");
                    let rt = &mut ds_slots[rows[i].slot];
                    model.dspark_append_context(
                        draft,
                        &mut rt.df,
                        slots[rows[i].slot].dspark_taps(),
                        1,
                        rows[i].start_pos,
                    )?;
                    let block_hidden = model.dspark_forward_block(
                        draft,
                        &mut rt.df,
                        &mut rt.scratch,
                        &mut rt.attn_states,
                        anchor,
                        verify_pos,
                    )?;
                    model.dspark_build_proposal(
                        draft,
                        &block_hidden,
                        anchor,
                        rows[i].params.temperature,
                        *sps,
                    )?
                };
                proposals[i] = Some(proposal);
            }
        }

        // ── Phase 2: TP lockstep on proposals (rank 0's draft_len + chain wins,
        // same as the B=1 path — divergent lengths deadlock the verify collective).
        for prop in proposals.iter_mut() {
            let Some(p) = prop else { continue };
            Self::tp_lockstep_proposal(&self.model, p)?;
        }

        // ── Phase 3: batched verify over all chains (the amortized phase).
        // If every slot hit the sequence-cap fallback, there are no chains to
        // verify — return the fallback tokens directly.
        let (mut slot_ids, mut chains, mut starts) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            if fallback_tokens[i].is_some() {
                continue;
            }
            slot_ids.push(rows[i].slot);
            chains.push(proposals[i].as_ref().unwrap().chain.clone());
            starts.push(verify_positions[i]);
        }
        if slot_ids.is_empty() {
            return Ok(fallback_tokens.into_iter().map(|t| t.unwrap()).collect());
        }
        let scheds: Vec<crate::dsv4::SpecVerifySchedule> = chains
            .iter()
            .zip(&starts)
            .map(|(chain, &start)| {
                let positions: Vec<usize> = (0..chain.len()).map(|j| start + j).collect();
                let ancestors: Vec<Vec<usize>> =
                    (0..chain.len()).map(|j| (0..j).collect()).collect();
                crate::dsv4::SpecVerifySchedule {
                    positions,
                    ancestors,
                }
            })
            .collect();

        // capture_spec_rings per slot (cheap host snapshot, must run before verify
        // overwrites the speculative KV band).
        for (s, &slot_idx) in slot_ids.iter().enumerate() {
            self.model.capture_spec_rings(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                starts[s],
                chains[s].len() - 1,
            )?;
        }

        let (verified, _verify_logits) = self.model.forward_decode_batch_verify(
            &mut self.slots,
            &mut self.kv_adapter,
            &slot_ids,
            &chains,
            &starts,
            &scheds,
        )?;

        // ── Phase 4: per-slot accept + rollback + commit (proven fold path).
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut verified_iter = verified.into_iter();
        for i in 0..n {
            if let Some(toks) = &fallback_tokens[i] {
                out[i] = toks.clone();
                continue;
            }
            let slot_idx = rows[i].slot;
            let verify_pos = verify_positions[i];
            let proposal = proposals[i].as_ref().unwrap();
            let draft_len = proposal.draft_len;
            let (argmax, _hiddens) = verified_iter.next().unwrap();
            ensure!(
                argmax.len() == proposal.chain.len(),
                "DSpark batched verify slot {slot_idx} argmax {} != chain {}",
                argmax.len(),
                proposal.chain.len()
            );
            let mut accepted = 0usize;
            while accepted < draft_len && argmax[accepted] == proposal.chain[accepted + 1] {
                accepted += 1;
            }
            let mut bonus = argmax[accepted];

            // TP lockstep on accepted/bonus (rank 0 wins — KV tail must be identical
            // across ranks, else next tick's combined attention corrupts).
            (accepted, bonus) = Self::tp_lockstep_accept(&self.model, accepted, bonus)?;

            self.model.truncate_slot(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                verify_pos,
            )?;
            self.model.restore_spec_ring_tail(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                verify_pos,
                accepted,
                draft_len,
            )?;
            self.model.commit_accepted_fold(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                0..accepted + 1,
                verify_pos,
            )?;

            // Append only the committed prefix (anchor + accepted drafts) to the
            // draft context; the rejected tail must not leak into the next block.
            {
                let Self {
                    model,
                    slots,
                    dspark,
                    ..
                } = self;
                let Dsv4DsparkExec {
                    draft,
                    slots: ds_slots,
                    ..
                } = dspark
                    .as_mut()
                    .expect("dspark_decode_tokens_batched without dspark");
                model.dspark_append_context(
                    draft,
                    &mut ds_slots[slot_idx].df,
                    slots[slot_idx].dspark_taps(),
                    accepted + 1,
                    verify_pos,
                )?;
            }

            self.mtp_accepts += accepted;
            self.mtp_rejects += draft_len - accepted;
            self.mtp_chains += 1;
            self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
            let mut slot_out = proposal.chain[..accepted + 1].to_vec();
            slot_out.push(bonus);
            out[i] = slot_out;
        }
        Ok(out)
    }

    fn forward_decode_row(&mut self, row: &Dsv4DecodeBatchRow) -> Result<Vec<SlotToken>> {
        let tokens = self.forward_decode_tokens(
            row.slot,
            row.last_token,
            row.start_pos,
            &row.params,
            row.position,
        )?;
        Ok(tokens
            .into_iter()
            .map(|token| SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// One decode sub-batch, then the prefix-state publish hook: any row that
    /// crossed a host-page boundary this tick publishes its completed pages.
    /// The hook runs AFTER the inner forward's MTP verify+commit+truncate, so
    /// `seq_len` reflects only committed tokens — rejected drafts never enter
    /// the pool.
    fn forward_decode_batch(
        &mut self,
        rows: &[DecodeRow],
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        let out = self.forward_decode_batch_inner(rows, kv_batch)?;
        // Finish write-through owns the decode region: per-tick D2H capture is
        // graph-incompatible, so under --dsv4-decode-reuse it's a no-op and the
        // whole generated region is captured once at finish.
        if !crate::runtime_flags::dsv4_decode_reuse_enabled() {
            for (row, kv_row) in rows.iter().zip(&kv_batch.rows) {
                let slot_pages = &kv_batch.flat_slot_page_ids[kv_row.slot_page_range.clone()];
                let end_pos = self.slots[row.slot].seq_len();
                self.publish_completed_prefix_pages(row.slot, slot_pages, row.kv_seq_len, end_pos);
            }
        }
        Ok(out)
    }

    fn forward_decode_batch_inner(
        &mut self,
        rows: &[DecodeRow],
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        validate_dsv4_decode_kv_batch(rows, kv_batch)?;
        let kv_view = self.kv_adapter.prepare_kv_batch(kv_batch)?;
        validate_dsv4_decode_kv_view(rows, &kv_view)?;
        // Host band changed since last sync (page growth at a page boundary) —
        // refresh the graph-referenced device tables BEFORE the forward. Decode
        // growth previously never resynced the device table (#154 Phase 0).
        for row in rows {
            if self.kv_adapter.take_device_table_dirty(row.slot) {
                self.slots[row.slot]
                    .refresh_flashmla_device_page_tables(&self.model.ctx, &self.kv_adapter)?;
            }
        }
        let batch = Dsv4DecodeBatch::from_rows(rows, &self.slots, self.num_slots)?;
        ensure!(
            batch.slot_ids.len() == batch.rows.len()
                && batch.tokens.len() == batch.rows.len()
                && batch.start_positions.len() == batch.rows.len()
                && batch.positions.len() == batch.rows.len(),
            "DSv4 decode batch surface length mismatch"
        );
        if batch.rows.len() == 1 {
            let out = self.forward_decode_row(&batch.rows[0])?;
            // B=1 bypasses `Dsv4Model::forward_decode_batch` (the CUDA-graph
            // decode-graph lane, `forward_tokens_decode_graph`) — mirror its
            // trace call here so the diagnostic covers both lanes uniformly.
            // See ARLE_DSV4_DECODE_TRACE, crate::dsv4::dsv4_decode_trace.
            crate::dsv4::dsv4_decode_trace(
                self.model.tp.config().rank,
                &batch.slot_ids,
                &batch.start_positions,
                &out.iter().map(|t| t.token).collect::<Vec<_>>(),
            );
            return Ok(out);
        }

        // B>1 path. The concurrency gate (`--spec-max-batch`, default 1) decides
        // the route: spec drafts only at or below the gate (the c=1 win); above
        // it spec is a compute-bound loss, so decode falls to the plain batched
        // path that scales. Same `route_decode` the qwen35 executor uses.
        use super::spec_decode::{DecodeRoute, SpecKind, route_decode};
        let spec_on = self.spec_requested();
        let all_greedy = batch.rows.iter().all(|row| row.params.is_greedy());
        let spec_kind = if self.dspark.is_some() {
            SpecKind::Dspark
        } else if spec_on {
            SpecKind::Mtp
        } else {
            SpecKind::None
        };
        // DSv4 still drafts per slot, so it keeps the c=1 gate the 2026-07-26
        // campaign licensed (−47.7% at c=16 with the gate open). Only Qwen3.5,
        // whose draft and rollback batch, honours a wider `--spec-max-batch`.
        let route = route_decode(
            spec_kind,
            batch.rows.len(),
            crate::runtime_flags::spec_max_batch().min(1),
        );

        // DSpark drafts per-slot (serial 3-stage forward) but verifies ALL chains
        // in ONE batched target forward (`dspark_decode_tokens_batched`),
        // amortizing the heaviest phase across the batch.
        if route == DecodeRoute::Dspark {
            // The batched verify path requires FlashMLA sparse prefill
            // (`mla_attention` chain-verify lane); without it, fall back to
            // per-slot dispatch (the pre-batch behavior — correct but slower).
            if crate::attention::dsv4_flashmla_prefill_enabled()? {
                let tokens = self.dspark_decode_tokens_batched(&batch.rows)?;
                let mut out = Vec::with_capacity(batch.rows.len());
                for (row, toks) in batch.rows.iter().zip(tokens) {
                    out.extend(toks.into_iter().map(|token| SlotToken {
                        slot: row.slot,
                        token,
                        logprob: None,
                        finish: None,
                    }));
                }
                return Ok(out);
            }
            let mut out = Vec::new();
            for row in &batch.rows {
                out.extend(self.forward_decode_row(row)?);
            }
            return Ok(out);
        }
        if route == DecodeRoute::Mtp && all_greedy {
            // Self-heal an un-seeded / desynced MTP stream (#140), the batched
            // twin of the B=1 path: a slot entering Decoding without a tail
            // prefill (full prefix-cache hit / whole-slot promote) has pending=
            // None. Rather than bail the TP group, warm-step EVERY row this one
            // tick to (re)seed pending+hidden, then batched MTP resumes next
            // tick. Cheaper than splitting the batch; a one-tick per-slot
            // degradation only when a slot joins un-seeded.
            let needs_seed = batch
                .rows
                .iter()
                .any(|row| self.spec_slots[row.slot].pending != Some(row.last_token));
            if needs_seed {
                let mut tokens = Vec::with_capacity(batch.rows.len());
                for row in &batch.rows {
                    if let Some(pending) = self.spec_slots[row.slot].pending
                        && pending != row.last_token
                    {
                        log::warn!(
                            "DSv4 MTP batched stream desync (slot {}): pending {pending} != \
                                 last_token {}; re-seeding",
                            row.slot,
                            row.last_token
                        );
                    }
                    let token = self.forward_mtp_warm_step(
                        row.slot,
                        &[row.last_token],
                        row.start_pos,
                        &row.params,
                        row.position,
                    )?;
                    tokens.push(SlotToken {
                        slot: row.slot,
                        token,
                        logprob: None,
                        finish: None,
                    });
                }
                return Ok(tokens);
            }
            let committed = self.spec_step_batched(&batch.slot_ids, &batch.start_positions)?;
            ensure!(
                committed.len() == batch.rows.len(),
                "DSv4 batched MTP returned {} chains for {} rows",
                committed.len(),
                batch.rows.len()
            );
            let tokens: Vec<SlotToken> = batch
                .slot_ids
                .iter()
                .zip(committed)
                .flat_map(|(&slot, chain)| {
                    chain.into_iter().map(move |token| SlotToken {
                        slot,
                        token,
                        logprob: None,
                        finish: None,
                    })
                })
                .collect();
            return Ok(tokens);
        }

        // True batched decode (layer-major driver, batched attention + grouped
        // MoE). B=1 is the single-row reference above; B>1 always batches.
        // Reached when spec was requested but this batch routed to Plain — the
        // concurrency gate (n_rows > spec_max_batch) or non-greedy sampling. Drop
        // the per-row spec state so a later c=1 tick re-seeds cleanly.
        if spec_on {
            for row in &batch.rows {
                self.spec_slots[row.slot] = Dsv4SpecSlotState::default();
            }
        }
        let params: Vec<SamplingParams> = batch.rows.iter().map(|r| r.params.clone()).collect();
        let out = self.model.forward_decode_batch(
            &mut self.slots,
            &mut self.kv_adapter,
            &batch.slot_ids,
            &batch.tokens,
            &batch.start_positions,
            &batch.positions,
            &params,
            false, // plain decode: no draft consumes taps here
        )?;
        ensure!(
            out.len() == batch.rows.len(),
            "DSv4 batched decode returned {} tokens for {} rows",
            out.len(),
            batch.rows.len()
        );
        Ok(batch
            .slot_ids
            .iter()
            .zip(out)
            .map(|(&slot, token)| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    pub(crate) fn submit(
        &mut self,
        plan: &ForwardPlan,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<StepOutput> {
        self.poll_prefix_captures();
        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        // Cross-rank lockstep debug surface: every rank logs a per-forward plan
        // fingerprint. Divergence at tick K (different rows on different ranks)
        // is the NCCL-deadlock root cause for multi-rank serve; diff the per-rank
        // streams with RUST_LOG=infer_cuda=debug.
        if rows > 0 && log::log_enabled!(log::Level::Debug) {
            static PLAN_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let tick = PLAN_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            log::debug!(
                "[dsv4-plan] rank={} tick={tick} prefill={:?} decode={:?}",
                self.model.tp.config().rank,
                plan.prefill_rows
                    .iter()
                    .map(|r| (r.slot, r.start_pos, r.tokens.len()))
                    .collect::<Vec<_>>(),
                plan.decode_rows
                    .iter()
                    .map(|r| (r.slot, r.kv_seq_len))
                    .collect::<Vec<_>>(),
            );
        }
        if rows == 0 {
            ensure!(
                kv_batch.rows.is_empty(),
                "DSv4 empty plan got non-empty KV batch descriptor"
            );
            return Ok(StepOutput { tokens: Vec::new() });
        }

        if plan.prefill_rows.is_empty() {
            return Ok(StepOutput {
                tokens: self.forward_decode_batch(&plan.decode_rows, kv_batch)?,
            });
        }

        // Mixed / multi-prefill plans split into per-prefill single-row
        // sub-steps plus one decode sub-batch. Plan rows always address
        // disjoint slots (a request is either Prefilling or Decoding), so the
        // sequential sub-steps are math-identical to consecutive single-mode
        // ticks. Descriptor commit order is prefill rows first, then decode
        // rows (`KvBatchDescriptor::from_plan`), mapping plan rows onto
        // descriptor rows by index.
        ensure!(
            kv_batch.rows.len() == rows,
            "DSv4 KV batch descriptor has {} rows for a {rows}-row plan",
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
                "DSv4 plan schedules slot {slot} more than once per tick"
            );
        }

        let n_prefill = plan.prefill_rows.len();
        let mut tokens = Vec::with_capacity(rows);
        for (idx, row) in plan.prefill_rows.iter().enumerate() {
            let sub_batch = kv_batch.subset(idx..idx + 1)?;
            tokens.extend(self.submit_prefill_row(row, &sub_batch)?);
        }
        if !plan.decode_rows.is_empty() {
            let sub_batch = kv_batch.subset(n_prefill..kv_batch.rows.len())?;
            tokens.extend(self.forward_decode_batch(&plan.decode_rows, &sub_batch)?);
        }
        Ok(StepOutput { tokens })
    }

    /// One prefill row as its own single-row sub-step. `kv_batch` must be the
    /// row's single-row (sub-)descriptor — indistinguishable from what a
    /// prefill-only tick delivers.
    fn submit_prefill_row(
        &mut self,
        row: &infer_plan::PrefillRow,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        validate_dsv4_prefill_kv_batch(row, kv_batch)?;
        ensure!(
            row.slot < self.num_slots,
            "prefill slot {} outside DSv4 executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
        // C2: free+reset BEFORE prepare_kv_batch (free-then-alloc, matching the
        // Qwen3.5 executor). prepare_kv_batch draws pages + advances seq_len; if
        // the reset ran after, fresh prefill wrote into an EMPTY page table and a
        // reused slot aborted the prepare seq_len==append_pos invariant.
        if row.start_pos == 0 {
            self.kv_adapter.flashmla_free_slot(row.slot)?;
            self.slots[row.slot].reset(&self.model.ctx, &mut self.kv_adapter)?;
            self.spec_slots[row.slot] = Dsv4SpecSlotState::default();
            self.reset_dspark_slot(row.slot, 0);
        }
        let kv_view = self.kv_adapter.prepare_kv_batch(kv_batch)?;
        validate_dsv4_prefill_kv_view(row, &kv_view)?;
        if row.start_pos == 0 {
            self.kv_adapter.zero_slot_band(&self.model.ctx, row.slot)?;
        }
        // Host band changed since last sync — refresh the graph-referenced device tables.
        if self.kv_adapter.take_device_table_dirty(row.slot) {
            self.slots[row.slot]
                .refresh_flashmla_device_page_tables(&self.model.ctx, &self.kv_adapter)?;
        }
        let position = (row.start_pos + row.tokens.len()) as u64;
        let final_prefill = row.start_pos + row.tokens.len() >= row.total_tokens;
        let tokens = self.forward_prefill_tokens(
            row.slot,
            &row.tokens,
            row.start_pos,
            &row.params,
            position,
            final_prefill,
        )?;
        // Seed THIS prompt chunk into the DSpark draft context (the canonical
        // context path). The forward captured the chunk's full multi-row taps;
        // append them at their absolute trunk positions `row.start_pos..`. Chunks
        // accumulate in `df.ctx_end`, so a chunked prefill seeds the whole prompt
        // across chunks; the frontier the first decode block reads is thus the
        // prompt context, not an empty rebase. No-op off the DSpark path or when
        // the chunk was single-row. A restored (cache-hit) prefix's context is
        // genuinely absent — its taps were never captured (partial-window, still
        // strictly better than the prior empty seed).
        self.seed_dspark_prompt(row.slot, row.start_pos)?;
        let slot = row.slot;
        // Prefix-state publish hook: this chunk completed pages
        // [start_pos/page, (start_pos+len)/page).
        let slot_pages = &kv_batch.flat_slot_page_ids[kv_batch.rows[0].slot_page_range.clone()];
        self.publish_completed_prefix_pages(
            slot,
            slot_pages,
            row.start_pos,
            row.start_pos + row.tokens.len(),
        );
        Ok(tokens
            .into_iter()
            .map(|token| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// Selftest band prep: free + reset, then materialize the band and
    /// refresh the device page tables — the selftest drives `forward_tokens`
    /// directly (no `prepare_kv_batch`), which used to run kernels against
    /// never-refreshed all-zero device tables (#154 Phase 3b fix).
    fn selftest_reset_band(&mut self, slot_idx: usize, prompt_len: usize) -> Result<()> {
        self.kv_adapter.flashmla_free_slot(slot_idx)?;
        self.slots[slot_idx].reset(&self.model.ctx, &mut self.kv_adapter)?;
        self.kv_adapter.prepare_direct_forward(
            &self.model.ctx,
            slot_idx,
            prompt_len + crate::dsv4::MAX_SPEC_DRAFT_DEPTH + 2,
        )?;
        self.kv_adapter.zero_slot_band(&self.model.ctx, slot_idx)?;
        if self.kv_adapter.take_device_table_dirty(slot_idx) {
            self.slots[slot_idx]
                .refresh_flashmla_device_page_tables(&self.model.ctx, &self.kv_adapter)?;
        }
        Ok(())
    }
    pub(crate) fn verify_forward_selftest(&mut self, prompt: &[u32]) -> Result<()> {
        ensure!(
            !prompt.is_empty(),
            "DSv4 verify-forward selftest requires a non-empty prompt"
        );
        let slot_idx = 0;
        let params = SamplingParams::default();
        let start_pos = prompt.len();

        self.selftest_reset_band(slot_idx, prompt.len())?;
        let token_a = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        let verify_one = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a],
            start_pos,
            (start_pos + 1) as u64,
        )?;

        self.selftest_reset_band(slot_idx, prompt.len())?;
        let token_a_again = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        ensure!(
            token_a == token_a_again,
            "DSv4 verify selftest prefill token drifted: {token_a} != {token_a_again}"
        );
        let normal_one = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a],
            start_pos,
            &params,
            (start_pos + 1) as u64,
        )?;
        ensure!(
            verify_one.argmax.first().copied() == Some(normal_one),
            "DSv4 verify selftest one-token mismatch: verify={:?} normal={normal_one}",
            verify_one.argmax
        );

        self.selftest_reset_band(slot_idx, prompt.len())?;
        let token_a = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        let verify_one = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a],
            start_pos,
            (start_pos + 1) as u64,
        )?;
        let token_b = verify_one.argmax[0];
        let mut wrong_b = token_b.wrapping_add(2);
        if wrong_b == token_b {
            wrong_b = token_b.wrapping_add(3);
        }

        self.selftest_reset_band(slot_idx, prompt.len())?;
        let token_a = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        let verify_two = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a, wrong_b],
            start_pos,
            (start_pos + 1) as u64,
        )?;
        ensure!(
            verify_two.argmax.first() == verify_one.argmax.first(),
            "DSv4 verify selftest two-token row0 mismatch: one={:?} two={:?}",
            verify_one.argmax,
            verify_two.argmax
        );

        // NOTE: no col1/bonus byte-identity gate here. The 2-token verify's col1
        // (bonus) on a FORCED-WRONG draft is DISCARDED in real decode (rejects emit
        // only base_next), and any byte-identity check is confounded by the M=2-vs-M=1
        // FP8 kernel path (SWA-prefill + prefill-DeepGEMM vs FlashMLA-decode +
        // decode-DeepGEMM). The real correctness gate is full-decode byte-identity vs
        // non-spec (validated 2026-06-08: batched MTP byte-identical on needle+capital,
        // +61/+70%). See errors/2026-06-08-dsv4-batched-verify-col1-wrong.md.

        // Depth-2 sequential MTP probe REMOVED 2026-06-08 (killed): measured depth-1 3/3,
        // depth-2-top1 1/3 (~33% accept) → sequential chain is only ~+15%, not the 6ms
        // path. The single MTP head caps multi-step drafts; tree-EAGLE (top-K) is the
        // amortization lever beyond depth-1. See the consolidated decode-6ms report.

        self.kv_adapter.flashmla_free_slot(slot_idx)?;
        self.slots[slot_idx].reset(&self.model.ctx, &mut self.kv_adapter)?;
        self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
        if self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp-selftest] PASS token_a={token_a} token_b={token_b} wrong_b={wrong_b} verify_two={:?}",
                verify_two.argmax
            );
        }
        Ok(())
    }
}
