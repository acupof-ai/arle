use super::*;

#[path = "dsv4/build.rs"]
mod build;
#[path = "dsv4/dspark.rs"]
mod dspark;
#[path = "dsv4/kv_contract.rs"]
mod kv_contract;
#[path = "dsv4/prefix_pool.rs"]
mod prefix_pool;
#[path = "dsv4/selftest.rs"]
mod selftest;
#[path = "dsv4/slot_tier.rs"]
mod slot_tier;

use kv_contract::{
    Dsv4DecodeBatch, Dsv4DecodeBatchRow, KvRowExpect, validate_dsv4_kv_batch, validate_dsv4_kv_view,
};
use prefix_pool::PendingPrefixCapture;

/// DSv4-Flash executor: drives [`crate::dsv4::Dsv4Model::forward_tokens`].
/// DSv4 owns its MLA KV state inside the forward, so it does NOT use a
/// [`PagedKVPool`]; the decode graph is disabled (MLA host-routing per step).
pub(crate) struct Dsv4CudaExecutor {
    pub(crate) model: crate::dsv4::Dsv4Model,
    pub(crate) slots: Vec<crate::dsv4::Dsv4SlotState>,
    pub(crate) kv_adapter: crate::attention::Dsv4KvAdapter,
    pub(crate) spec_slots: Vec<Dsv4SpecSlotState>,
    /// `Some(n)` = MTP spec decode on at draft depth `n`; `None` = no MTP head.
    pub(crate) spec_draft_tokens: Option<usize>,
    /// MTP draft candidate width; `None`/`Some(1)` keeps chain-only candidates.
    pub(crate) spec_draft_topk: Option<usize>,
    pub(crate) num_slots: usize,
    pub(crate) mtp_accepts: usize,
    pub(crate) mtp_rejects: usize,
    /// Verified MTP draft chains (one per spec row that reached verify).
    pub(crate) mtp_chains: usize,
    /// Content-keyed cross-request prefix-state pool: one host-resident entry
    /// per completed host page, written from the post-forward choke point.
    pub(super) prefix_state: crate::attention::Dsv4PrefixStatePool,
    pending_prefix_captures: VecDeque<PendingPrefixCapture>,
    /// `Some` = the DSpark drafter is loaded and greedy decode routes through
    /// its draft→verify→accept loop instead of MTP.
    pub(crate) dspark: Option<Dsv4DsparkExec>,
    /// Whole-slot spill store: a demoted slot's complete device image, restored
    /// on promote (without it, KV overflow falls back to recompute). NVMe spill
    /// is opt-in via `set_kv_tier_disk`.
    pub(super) slot_tier: KvTierStore,
    /// Per-slot c=1 decode graph: the capture bakes the slot's device buffers
    /// (start_pos, page table), so one graph per slot.
    decode_graphs: Vec<Option<crate::graph::CudaGraphState>>,
}

pub(crate) struct Dsv4DsparkExec {
    pub(super) draft: crate::dsv4::Dsv4DsparkDraft,
    /// Verify-step cost model driving the goodput budget (`--dspark-sps-*-ms`).
    pub(super) sps: qwen35_spec::DsparkSps,
    /// Per-request token ceiling; a block whose worst-case chain crosses it
    /// falls back to one non-spec token.
    pub(super) max_seq_len: usize,
    slots: Vec<Dsv4DsparkRuntime>,
}

/// Pre-allocated at construction — spec steps are serial, so exact-shape reuse
/// (no per-step alloc) holds.
struct Dsv4DsparkRuntime {
    pub(super) df: crate::dsv4::dspark::Dsv4DsparkSlotState,
    pub(super) scratch: crate::dsv4::dspark::Dsv4DsparkScratch,
    pub(super) attn_states: Vec<crate::attention::Dsv4LayerAttentionState>,
}

/// MTP spec carry; must ride along a demote/promote, since a resumed decode
/// requires the pending token and the hidden staged with it.
#[derive(Default)]
pub(crate) struct Dsv4SpecSlotState {
    pub(crate) pending: Option<u32>,
    pub(crate) hidden: Option<DeviceVec>,
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
    /// Active only for demand-paged bands; identity/V32 draw the full band at
    /// slot alloc.
    pub(crate) fn kv_device_gate_active(&self) -> bool {
        self.kv_adapter.flashmla_demand_paged()
    }

    /// Lockstep-safe without a collective: band pools mutate only from the
    /// rank-identical plan stream, so every rank computes the same fit.
    pub(crate) fn kv_device_fit(
        &self,
        rows: &[infer_seam::DeviceRowDemand],
        unfit: &mut Vec<usize>,
    ) {
        self.kv_adapter.flashmla_demand_fit(rows, unfit);
    }

    pub(super) fn tp_min_usize(&self, value: usize, what: &str) -> Result<usize> {
        let capped = i32::try_from(value.min(i32::MAX as usize)).unwrap_or(i32::MAX);
        self.model
            .tp
            .all_reduce_min_scalar_i32(&self.model.ctx, capped)
            .map(|v| v.max(0) as usize)
            .map_err(|e| anyhow::anyhow!("DSv4 TP min-reduce {what} failed: {e}"))
    }

    /// Largest single prefill forward; also the restore-snapshot grain when it
    /// exceeds the boundary alignment. 2048 licensed by c1 TTFT 3031→1088 ms,
    /// c16 out +138%. Rounded down to a 128 multiple so chunk ends stay on the
    /// ring/snapshot alignment.
    pub(crate) fn max_prefill_chunk(&self) -> usize {
        const DEFAULT_PREFILL_CHUNK: usize = 2048;
        let grain = self.model.config.sliding_window.max(1);
        let cap = DEFAULT_PREFILL_CHUNK
            .clamp(128, crate::attention::DSV4_PREFILL_QUERY_CHUNK)
            .min(self.model.max_tokens_per_step().unwrap_or(usize::MAX));
        (cap / 128 * 128).max(grain)
    }

    pub(crate) fn tp_sync_min(&self, local: usize) -> Result<usize> {
        self.tp_min_usize(local, "admission free pages")
    }

    /// Build a plain `SlotToken` (no logprob / finish) — the DSv4 executor never
    /// emits per-token logprobs or finish-reasons from the forward.
    #[inline]
    fn slot_token(slot: usize, token: u32) -> SlotToken {
        SlotToken {
            slot,
            token,
            logprob: None,
            top_logprobs: Vec::new(),
            finish: None,
        }
    }

    /// Whether `slot`'s staged MTP pending token matches `last_token`. A
    /// mismatch (or absence) means the staged hidden belongs to another token
    /// and the stream must be re-seeded via a warm step.
    fn mtp_pending_matches(&self, slot: usize, last_token: u32) -> bool {
        matches!(self.spec_slots[slot].pending, Some(pending) if pending == last_token)
    }

    /// One no-spec greedy forward that ALSO stages the MTP draft state (pending
    /// token + stream hidden) so a subsequent `spec_step` can resume.
    fn forward_mtp_warm_step(
        &mut self,
        slot_idx: usize,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
    ) -> Result<u32> {
        let (token, hidden) = self.model.forward_tokens_with_hidden(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            tokens,
            start_pos,
            params,
            position,
            penalty,
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
        penalty: infer_plan::PenaltyHistory<'_>,
        final_prefill: bool,
    ) -> Result<Vec<u32>> {
        let spec_on = self.spec_requested();
        if spec_on && params.is_raw_argmax() {
            let token = if final_prefill {
                self.forward_mtp_warm_step(slot_idx, tokens, start_pos, params, position, penalty)?
            } else {
                self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
                self.model.forward_tokens(
                    &mut self.slots[slot_idx],
                    &mut self.kv_adapter,
                    tokens,
                    start_pos,
                    params,
                    position,
                    penalty,
                )?
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
                penalty,
            )?])
        }
    }

    /// Try the c=1 decode graph. Returns `Ok(Some(tokens))` on success,
    /// `Ok(None)` if the graph is unavailable (conditions not met, capture
    /// failed, or replay not yet reached).
    fn try_graph_decode_c1(
        &mut self,
        slot_idx: usize,
        last_token: u32,
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
    ) -> Result<Option<Vec<u32>>> {
        if !params.is_raw_argmax() || params.has_penalty() {
            return Ok(None);
        }
        if self.spec_requested() || self.dspark.is_some() {
            return Ok(None);
        }
        #[cfg(feature = "deepep")]
        if self.model.deepep.is_some() {
            return Ok(None);
        }

        let graph = self.decode_graphs[slot_idx].get_or_insert_with(|| {
            crate::graph::CudaGraphState::new(self.model.ctx.stream.clone())
        });

        let model = &self.model;
        let slot = &mut self.slots[slot_idx];
        let kv_adapter = &mut self.kv_adapter;

        // Pre-graph: upload token_ids + start_pos to persistent buffers.
        model.graph_upload_token_ids(&[last_token])?;
        let start_pos_i32 = i32::try_from(start_pos)
            .map_err(|_| anyhow::anyhow!("DSv4 start_pos {start_pos} overflows i32"))?;
        model
            .ctx
            .stream
            .memcpy_htod(&[start_pos_i32], &mut slot.start_pos_device)
            .map_err(|e| anyhow::anyhow!("DSv4 graph start_pos H2D failed: {e}"))?;

        model
            .graph_mode
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // Graph: capture/replay transformer layers. The closure stores the
        // output stream so the LM head can read it after replay.
        let result = std::cell::RefCell::new(None);
        let capture_result = graph.run_or_capture(|| {
            let (stream, keepalive) = model.forward_tokens_stream_impl(
                slot,
                kv_adapter,
                &[last_token],
                start_pos,
                false,
                None,
            )?;
            *result.borrow_mut() = Some(stream);
            drop(keepalive);
            Ok(())
        });

        model
            .graph_mode
            .store(false, std::sync::atomic::Ordering::Relaxed);

        if let Err(e) = capture_result {
            log::warn!("DSv4 c=1 decode graph failed, falling back to eager: {e}");
            self.decode_graphs[slot_idx] = None;
            return Ok(None);
        }

        // During warm-up/capture the closure ran and produced the stream.
        // During replay the closure did NOT run; clone the persistent buffer.
        let was_replay = result.borrow().is_none();
        let stream = if let Some(s) = result.into_inner() {
            s
        } else {
            model.graph_stream_clone()?
        };

        // LM head + sampling (outside the graph).
        let mut keepalive = crate::dsv4::Dsv4ForwardKeepalive::new(false);
        let token = model.forward_stream_last_token(
            &stream,
            1,
            params,
            position,
            penalty,
            None,
            &mut keepalive,
        )?;
        drop(keepalive);

        if was_replay {
            slot.advance_after_replay(&model.layers);
        }

        Ok(Some(vec![token]))
    }

    fn forward_decode_tokens(
        &mut self,
        slot_idx: usize,
        last_token: u32,
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
    ) -> Result<Vec<u32>> {
        if !self.spec_requested() {
            if let Some(tokens) = self
                .try_graph_decode_c1(slot_idx, last_token, start_pos, params, position, penalty)?
            {
                return Ok(tokens);
            }
            let token = self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &[last_token],
                start_pos,
                params,
                position,
                penalty,
            )?;
            return Ok(vec![token]);
        }

        if !params.is_raw_argmax() {
            self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
            let token = self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &[last_token],
                start_pos,
                params,
                position,
                penalty,
            )?;
            return Ok(vec![token]);
        }
        if self.dspark.is_some() {
            return self.dspark_decode_tokens(slot_idx, last_token, start_pos, params, position);
        }
        // Self-heal an un-seeded / desynced MTP stream (#140): a request that
        // enters Decoding without a tail prefill (full prefix-cache hit or
        // whole-slot promote) has no pending token, and a pending disagreeing
        // with `last_token` means the staged hidden belongs to another token.
        // Re-seed with one warm step instead of bailing the TP group.
        if !self.mtp_pending_matches(slot_idx, last_token) {
            if let Some(pending) = self.spec_slots[slot_idx].pending {
                log::warn!(
                    "DSv4 MTP stream desync (slot {slot_idx}): pending {pending} != last_token \
                     {last_token}; re-seeding via a warm step"
                );
            }
            let token = self.forward_mtp_warm_step(
                slot_idx,
                &[last_token],
                start_pos,
                params,
                position,
                penalty,
            )?;
            return Ok(vec![token]);
        }
        self.spec_step(slot_idx, start_pos, position)
    }

    fn forward_decode_row(&mut self, row: &Dsv4DecodeBatchRow) -> Result<Vec<SlotToken>> {
        let tokens = self.forward_decode_tokens(
            row.slot,
            row.last_token,
            row.start_pos,
            &row.params,
            row.position,
            penalty_of(&row.penalty_history, row.penalty_prompt_len),
        )?;
        Ok(tokens
            .into_iter()
            .map(|token| Self::slot_token(row.slot, token))
            .collect())
    }

    /// The publish hook runs AFTER the inner forward's verify+commit+truncate,
    /// so `seq_len` reflects only committed tokens — rejected drafts never
    /// enter the pool.
    fn forward_decode_batch(
        &mut self,
        rows: &[DecodeRow],
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        let out = self.forward_decode_batch_inner(rows, kv_batch)?;
        // Per-tick D2H capture is graph-incompatible, so finish write-through
        // captures the whole generated region once at finish instead.
        Ok(out)
    }

    fn forward_decode_batch_inner(
        &mut self,
        rows: &[DecodeRow],
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        let expect: Vec<KvRowExpect<'_>> = rows.iter().map(KvRowExpect::decode).collect();
        validate_dsv4_kv_batch(&expect, kv_batch)?;
        let kv_view = self.kv_adapter.prepare_kv_batch(kv_batch)?;
        validate_dsv4_kv_view(&expect, &kv_view)?;
        // Host band changed since last sync (page growth at a page boundary) —
        // refresh the graph-referenced device tables BEFORE the forward.
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
            return Ok(out);
        }

        // The concurrency gate drafts spec only at or below `--spec-max-batch`;
        // above it spec is a compute-bound loss and decode takes the plain
        // batched path that scales.
        use super::spec_decode::{DecodeRoute, SpecKind, route_decode};
        let spec_on = self.spec_requested();
        let any_penalty = batch.rows.iter().any(|row| row.params.has_penalty());
        // The batched MTP verify commits a device argmax over raw target logits.
        let all_raw_argmax = batch.rows.iter().all(|row| row.params.is_raw_argmax());
        let spec_kind = if self.dspark.is_some() {
            SpecKind::Dspark
        } else if spec_on {
            SpecKind::Mtp
        } else {
            SpecKind::None
        };
        // DSv4 drafts per slot: the sequential draft tax at c>1 exceeds the
        // batched-verify savings (−32% at c=8, −47.7% at c=16). Pin to c=1.
        let route = route_decode(
            spec_kind,
            batch.rows.len(),
            crate::runtime_flags::spec_max_batch().min(1),
            any_penalty,
        );

        if route == DecodeRoute::Dspark {
            // The batched verify needs FlashMLA's sparse-prefill chain-verify
            // lane; without it, per-slot dispatch is correct but slower.
            if cuda_kernels::HAS_FLASHMLA {
                let tokens = self.dspark_decode_tokens_batched(&batch.rows)?;
                let mut out = Vec::with_capacity(batch.rows.len());
                for (row, toks) in batch.rows.iter().zip(tokens) {
                    out.extend(
                        toks.into_iter()
                            .map(|token| Self::slot_token(row.slot, token)),
                    );
                }
                return Ok(out);
            }
            let mut out = Vec::new();
            for row in &batch.rows {
                out.extend(self.forward_decode_row(row)?);
            }
            return Ok(out);
        }
        if route == DecodeRoute::Mtp && all_raw_argmax {
            // Self-heal an un-seeded / desynced MTP stream (#140): warm-step
            // EVERY row for one tick to re-seed pending+hidden — cheaper than
            // splitting the batch, and batched MTP resumes next tick.
            let needs_seed = batch
                .rows
                .iter()
                .any(|row| !self.mtp_pending_matches(row.slot, row.last_token));
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
                        penalty_of(&row.penalty_history, row.penalty_prompt_len),
                    )?;
                    tokens.push(Self::slot_token(row.slot, token));
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
                    chain
                        .into_iter()
                        .map(move |token| Self::slot_token(slot, token))
                })
                .collect();
            return Ok(tokens);
        }

        // Reached when spec was requested but this batch routed to Plain; drop
        // the per-row spec state so a later c=1 tick re-seeds cleanly.
        if spec_on {
            for row in &batch.rows {
                self.spec_slots[row.slot] = Dsv4SpecSlotState::default();
            }
        }
        let params: Vec<SamplingParams> = batch.rows.iter().map(|r| r.params.clone()).collect();
        let penalties: Vec<infer_plan::PenaltyHistory<'_>> = batch
            .rows
            .iter()
            .map(|r| penalty_of(&r.penalty_history, r.penalty_prompt_len))
            .collect();
        let out = self.model.forward_decode_batch(
            &mut self.slots,
            &mut self.kv_adapter,
            &batch.slot_ids,
            &batch.tokens,
            &batch.start_positions,
            &batch.positions,
            &params,
            &penalties,
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
            .map(|(&slot, token)| Self::slot_token(slot, token))
            .collect())
    }

    pub(crate) fn submit(
        &mut self,
        plan: &ForwardPlan,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<StepOutput> {
        self.poll_prefix_captures();
        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        // Per-rank plan fingerprint: divergence at tick K is the NCCL-deadlock
        // root cause for multi-rank serve.
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

        // Plan rows always address disjoint slots, so splitting a mixed plan into
        // per-prefill sub-steps plus one decode sub-batch is math-identical to
        // consecutive single-mode ticks. Descriptor order is prefill rows first,
        // then decode rows, mapping plan rows onto descriptor rows by index.
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

    /// `kv_batch` must be the row's own single-row (sub-)descriptor.
    fn submit_prefill_row(
        &mut self,
        row: &infer_plan::PrefillRow,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        let expect = [KvRowExpect::prefill(row)];
        validate_dsv4_kv_batch(&expect, kv_batch)?;
        ensure!(
            row.slot < self.num_slots,
            "prefill slot {} outside DSv4 executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
        // Free+reset must precede prepare_kv_batch, which draws pages and
        // advances seq_len: resetting after leaves fresh prefill writing into an
        // EMPTY page table and breaks the seq_len==append_pos invariant.
        if row.start_pos == 0 {
            self.kv_adapter.flashmla_free_slot(row.slot)?;
            self.slots[row.slot].reset(&self.model.ctx, &mut self.kv_adapter)?;
            self.spec_slots[row.slot] = Dsv4SpecSlotState::default();
            self.reset_dspark_slot(row.slot, 0);
        }
        let kv_view = self.kv_adapter.prepare_kv_batch(kv_batch)?;
        validate_dsv4_kv_view(&expect, &kv_view)?;
        if row.start_pos == 0 {
            self.kv_adapter.zero_slot_band(&self.model.ctx, row.slot)?;
        }
        // Host band changed since last sync — refresh the graph-referenced device
        // tables.
        if self.kv_adapter.take_device_table_dirty(row.slot) {
            self.slots[row.slot]
                .refresh_flashmla_device_page_tables(&self.model.ctx, &self.kv_adapter)?;
        }
        let position = row.end_pos() as u64;
        let final_prefill = row.is_final_chunk();
        let tokens = self.forward_prefill_tokens(
            row.slot,
            &row.tokens,
            row.start_pos,
            &row.params,
            position,
            penalty_of(&row.penalty_history, row.penalty_prompt_len),
            final_prefill,
        )?;
        // Chunks accumulate in `df.ctx_end`, so a chunked prefill seeds the whole
        // prompt; a restored (cache-hit) prefix's taps were never captured, so
        // its context is genuinely absent.
        self.seed_dspark_prompt(row.slot, row.start_pos)?;
        let slot = row.slot;
        let slot_pages = &kv_batch.flat_slot_page_ids[kv_batch.rows[0].slot_page_range.clone()];
        self.publish_completed_prefix_pages(slot, slot_pages, row.start_pos, row.end_pos());
        Ok(tokens
            .into_iter()
            .map(|token| Self::slot_token(slot, token))
            .collect())
    }
}
