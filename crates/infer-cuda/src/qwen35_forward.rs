use super::*;
use anyhow::bail;

/// Joint slot/pool capacity split, mirroring DSv4's `Dsv4KvBudgetPlan`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Qwen35KvBudgetPlan {
    pub(crate) num_slots: usize,
    /// Shared full-attn paged-pool size the executor allocates verbatim.
    pub(crate) pool_pages: usize,
}

impl Qwen35Model {
    pub(crate) fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    pub(crate) fn local_full_attn_q_dim(&self) -> usize {
        self.local_q_heads * self.config.head_dim
    }

    /// This rank's GATED q_proj output width: the projection interleaves
    /// `[query; gate]` per head, so each local head contributes `2*head_dim` rows.
    pub(crate) fn local_full_attn_q_proj_dim(&self) -> usize {
        self.local_q_heads * self.config.head_dim * 2
    }

    /// Single source for every full-attn K/V cache / pool size on this rank.
    /// CACHE-IDENTITY INVARIANT (KV replication, kv_heads < world_size): each
    /// rank holds `local_kv_heads == 1` KV head, and ranks in the same replica
    /// group loaded IDENTICAL k/v projection weights ([`infer_topo::kv_load_block_index`])
    /// AND see the same `normed` hidden states, so their per-rank-local cache rows
    /// are bit-identical. GQA then runs independently per rank against its own
    /// copy — no cross-replica KV exchange is needed or done.
    pub(crate) fn local_full_attn_kv_dim(&self) -> usize {
        self.local_kv_heads * self.config.head_dim
    }

    pub(crate) fn local_linear_qkv_dim(&self) -> usize {
        let qk = 2 * self.local_linear_k_heads * self.config.linear_key_head_dim;
        qk + self.local_linear_v_heads * self.config.linear_value_head_dim
    }

    pub(crate) fn local_linear_z_dim(&self) -> usize {
        self.local_linear_v_heads * self.config.linear_value_head_dim
    }

    /// B2 CP decode per-rank head counts: 1/cp of the local attn_tp shard.
    /// Equal to the local counts when cp=1 or a head count is indivisible
    /// (decode then runs replicated, `cp_decode_handle` is None).
    pub(crate) fn decode_q_heads(&self) -> usize {
        self.local_q_heads / self.cp_decode_divisor()
    }

    pub(crate) fn decode_kv_heads(&self) -> usize {
        self.local_kv_heads / self.cp_decode_divisor()
    }

    pub(crate) fn decode_linear_k_heads(&self) -> usize {
        self.local_linear_k_heads / self.cp_decode_divisor()
    }

    pub(crate) fn decode_linear_v_heads(&self) -> usize {
        self.local_linear_v_heads / self.cp_decode_divisor()
    }

    fn cp_decode_divisor(&self) -> usize {
        self.cp_decode_handle().map_or(1, |h| h.cp_size)
    }

    /// `(num_linear, gdr_state_len, conv_len)` for this rank's recurrent state,
    /// sized from LOCAL shard widths (each rank carries only its own v-head
    /// recurrent slabs / qkv conv channels). Single source for both
    /// [`Qwen35SlotState::acquire_recurrent`] and the spec snapshot scratch.
    pub(crate) fn recurrent_dims(&self) -> (usize, usize, usize) {
        let c = &self.config;
        let num_linear = c.num_hidden_layers - c.num_full_attention_layers();
        let gdr_state_len =
            self.local_linear_v_heads * c.linear_key_head_dim * c.linear_value_head_dim;
        let conv_len = self.local_linear_qkv_dim() * (c.linear_conv_kernel_dim - 1);
        (num_linear, gdr_state_len, conv_len)
    }

    /// B2 CP decode dims for the 1/cp-head recurrent pair: same layer count,
    /// 1/cp the gdr/conv widths (the engage guard makes both head counts
    /// divisible by cp).
    pub(crate) fn recurrent_dims_decode(&self, cp_size: usize) -> (usize, usize, usize) {
        let c = &self.config;
        let num_linear = c.num_hidden_layers - c.num_full_attention_layers();
        let gdr_state_len =
            (self.local_linear_v_heads / cp_size) * c.linear_key_head_dim * c.linear_value_head_dim;
        let conv_len = (self.local_linear_qkv_dim() / cp_size) * (c.linear_conv_kernel_dim - 1);
        (num_linear, gdr_state_len, conv_len)
    }

    /// A fresh idle slot — allocates nothing. The recurrent state is drawn from
    /// the executor's free-list on activation ([`Qwen35SlotState::acquire_recurrent`]);
    /// the full-attn K/V lives in the shared `PagedKVPool`.
    pub(crate) fn new_slot_state(&self) -> Qwen35SlotState {
        Qwen35SlotState::new_linear_only()
    }

    /// Allocate per-slot speculative-decode state (draft head KV + trunk linear
    /// snapshot scratch), sized from this rank's local shard widths and the
    /// requested draft depth. Snapshot scratch matches [`Self::new_slot_state`]'s
    /// linear-state dims exactly so a snapshot/restore is a straight D2D copy.
    #[allow(dead_code)] // called by the executor spec-slot init in a later increment
    pub(crate) fn linear_state_bytes(&self) -> (usize, usize) {
        let c = &self.config;
        (
            self.local_linear_v_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4,
            self.local_linear_qkv_dim() * (c.linear_conv_kernel_dim - 1) * 2,
        )
    }

    pub(crate) fn new_spec_slot_state(&self) -> Result<Qwen35SpecSlotState> {
        let c = &self.config;
        let num_full = c.num_full_attention_layers();
        let num_linear = c.num_hidden_layers - num_full;
        // depth >= 1: the head cache holds the draft chain (level positions
        // 0..depth); +1 leaves room for the seed row at position 0.
        let depth = self.spec_draft_tokens.max(1);
        let kv_dim = self.local_full_attn_kv_dim();
        let gdr_state_len =
            self.local_linear_v_heads * c.linear_key_head_dim * c.linear_value_head_dim;
        let conv_len = self.local_linear_qkv_dim() * (c.linear_conv_kernel_dim - 1);
        let qkv_dim = self.local_linear_qkv_dim();
        // b/a are one scalar per LOCAL value head, sharded identically on every
        // linear layer (`in_proj_b`/`in_proj_a` rows == local_linear_v_heads),
        // so the capture stride is uniform across layers.
        let ba_dim = self.local_linear_v_heads;
        let cap_rows = depth + 1;
        let mut gdr_snap = Vec::with_capacity(num_linear);
        let mut conv_snap = Vec::with_capacity(num_linear);
        let mut cap_qkv = Vec::with_capacity(num_linear);
        let mut cap_b = Vec::with_capacity(num_linear);
        let mut cap_a = Vec::with_capacity(num_linear);
        for _ in 0..num_linear {
            gdr_snap.push(
                self.ctx
                    .stream
                    .alloc_zeros::<f32>(gdr_state_len)
                    .map_err(|e| anyhow!("alloc spec gated-delta snapshot failed: {e}"))?,
            );
            conv_snap.push(DeviceVec::zeros(&self.ctx, conv_len)?);
            cap_qkv.push(DeviceVec::zeros(&self.ctx, cap_rows * qkv_dim)?);
            cap_b.push(DeviceVec::zeros(&self.ctx, cap_rows * ba_dim)?);
            cap_a.push(DeviceVec::zeros(&self.ctx, cap_rows * ba_dim)?);
        }
        Ok(Qwen35SpecSlotState {
            head_k: DeviceVec::zeros(&self.ctx, (depth + 1) * kv_dim)?,
            head_v: DeviceVec::zeros(&self.ctx, (depth + 1) * kv_dim)?,
            gdr_snap,
            conv_snap,
            capture: Qwen35LinearCapture {
                rows: cap_rows,
                qkv: cap_qkv,
                b_proj: cap_b,
                a_proj: cap_a,
            },
            argmax_scratch: self
                .ctx
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| anyhow!("alloc spec argmax scratch failed: {e}"))?,
            q_probs: SliceSlot::default(),
            p_probs: SliceSlot::default(),
            sample_tok: SliceSlot::default(),
            accept_out: SliceSlot::default(),
            chain_draft: SliceSlot::default(),
            u_accept: SliceSlot::default(),
            u_residual: SliceSlot::default(),
        })
    }

    pub(crate) fn per_slot_kv_bytes(&self) -> (usize, usize, usize, usize) {
        let c = &self.config;
        let num_full = c.num_full_attention_layers();
        let num_linear = c.num_hidden_layers - num_full;
        let bf16 = std::mem::size_of::<half::bf16>();
        let f32sz = std::mem::size_of::<f32>();
        // Full-attn K/V is paged (shared pool), not per-slot → 0 here.
        let kv_bytes = 0usize;
        let gdr_len = self.local_linear_v_heads * c.linear_key_head_dim * c.linear_value_head_dim;
        let gdr_bytes = num_linear.saturating_mul(gdr_len).saturating_mul(f32sz);
        let conv_len = self.local_linear_qkv_dim() * (c.linear_conv_kernel_dim - 1);
        let conv_bytes = num_linear.saturating_mul(conv_len).saturating_mul(bf16);
        let per_slot = kv_bytes
            .saturating_add(gdr_bytes)
            .saturating_add(conv_bytes);
        (per_slot, kv_bytes, gdr_bytes, conv_bytes)
    }

    /// Jointly pick `(num_slots, pool_pages)` — the DSv4
    /// [`crate::dsv4::Dsv4Model::kv_budget_plan`] pattern (#182): the per-slot recurrent state and the shared full-attn
    /// paged pool trade against the same post-weights free VRAM, so solving
    /// slots alone let the recurrent reservation eat the headroom and starve
    /// the pool (86 slots × 47 tokens each on a 32 GB card). The plan is the
    /// largest state-affordable slot count whose pool remainder still funds
    /// one full-length (`max_seq_len`) request; feasibility is monotone in
    /// decreasing n, so the first feasible n scanning down is the max. When
    /// the pool is easily funded (H20-class VRAM) the scan accepts n at the
    /// old slots-only clamp on the first probe and the pool profiles the same
    /// remainder — the joint solve degenerates to the previous answer.
    /// NCCL min-reduced (slots, then pages at the reduced slot count) for
    /// TP-consistent capacity. Call AFTER weights load so `mem_get_info().free`
    /// already excludes them.
    /// `memory_budget_bytes`: a startup-fixed grant (co-resident allocators,
    /// e.g. the OPD driver's VRAM plan) capping the free-VRAM input, so the
    /// budget is decided by the grant rather than instantaneous free.
    /// `None` keeps the measured-free serve behavior.
    /// `requested == 1` (single-sequence engines, e.g. the OPD teacher scorer)
    /// keeps its exact `requested_pages` — the fraction probe over-provisions
    /// tens of GB for one ~8K-token sequence.
    pub(crate) fn kv_budget_plan(
        &self,
        requested: usize,
        requested_pages: usize,
        extra_per_slot_bytes: usize,
        memory_budget_bytes: Option<usize>,
        mem_fraction_static: f64,
        kv_format: cuda_kernels::KVFormat,
    ) -> Result<Qwen35KvBudgetPlan> {
        const MEM_FRACTION: f64 = 0.7;
        const PAGE: usize = crate::executor::SUPPORTED_PAGE_SIZE;
        let (per_slot, kv_bytes, gdr_bytes, conv_bytes) = self.per_slot_kv_bytes();
        let per_slot = per_slot.saturating_add(extra_per_slot_bytes);
        let num_full = self.config.num_full_attention_layers();
        let local_kv_heads = self.local_kv_heads();
        let head_dim = self.config.head_dim;
        let cell_bytes_per_token =
            PagedKVPool::budget_bytes_for_tokens(num_full, local_kv_heads, head_dim, 1, kv_format)
                as u64;
        let pool_tokens_at = |free: usize, total: usize, n: usize| -> u64 {
            infer_seam::profile_kv_pool_tokens(
                (free as u64).saturating_sub(per_slot.saturating_mul(n) as u64),
                total as u64,
                cell_bytes_per_token,
                mem_fraction_static,
            )
        };
        let (n_local, probe): (i32, Option<(usize, usize)>) =
            match cudarc::driver::result::mem_get_info() {
                Ok((free, total)) => {
                    let free = match memory_budget_bytes {
                        Some(grant) => {
                            let capped = free.min(grant);
                            log::info!(
                                "Qwen3.5 KV budget: fixed grant {}MB caps measured free {}MB -> {}MB",
                                grant >> 20,
                                free >> 20,
                                capped >> 20,
                            );
                            capped
                        }
                        None => free,
                    };
                    // Same neutral kernel as DSv4: floor(free × fraction) / per_slot.
                    let budget = infer_seam::SlotBudget::from_free(free, MEM_FRACTION, 0, per_slot);
                    log::info!(
                        "Qwen3.5 KV budget: free {}MB, per_slot {}MB (K+V {}MB + gdr {}MB + conv {}MB + draft {}MB)",
                        free >> 20,
                        per_slot >> 20,
                        kv_bytes >> 20,
                        gdr_bytes >> 20,
                        conv_bytes >> 20,
                        extra_per_slot_bytes >> 20,
                    );
                    let affordable = budget.affordable().unwrap_or(usize::MAX);
                    // Joint solve: shed slots until the pool remainder funds one
                    // full-length request. n=1 is the best-effort floor — the
                    // pool collapse warning below reports it, matching the old
                    // profile path instead of DSv4's hard reject.
                    let mut n = requested.max(1).min(affordable);
                    while n > 1 && pool_tokens_at(free, total, n) < self.max_seq_len as u64 {
                        n -= 1;
                    }
                    if affordable == 0 {
                        n = 0;
                    }
                    (i32::try_from(n).unwrap_or(i32::MAX), Some((free, total)))
                }
                // Can't query (no active context / driver error) → don't bind the
                // min; the other ranks' budgets still apply.
                Err(_) => (i32::MAX, None),
            };
        let affordable = self.tp.all_reduce_min_scalar_i32(&self.ctx, n_local)? as usize;
        // Reject-below-fixed guard (parity with Metal's fits_fixed + DSv4): a
        // cross-rank-min affordable of 0 means post-weights free VRAM cannot
        // hold even one slot at this max_seq_len. Fail closed uniformly
        // (lockstep-safe — same reduced scalar on every rank) instead of the
        // former `max(1)` admitting one slot and OOMing at slot allocation.
        anyhow::ensure!(
            affordable > 0,
            "Qwen3.5 KV budget rejected startup: post-weights free VRAM affords 0 slots at \
             max_seq_len {} (per_slot ~{}MB exceeds {MEM_FRACTION} of free). Free VRAM or \
             lower --max-total-tokens.",
            self.max_seq_len,
            per_slot >> 20,
        );
        let (planned, clamped) = infer_seam::clamp_to_affordable(requested, affordable);
        if clamped {
            log::warn!(
                "Qwen3.5 KV budget: requested {requested} slots × ~{}MB/slot exceeds the \
                 cross-rank-min joint-affordable {affordable} (local {n_local}, {MEM_FRACTION} \
                 of post-weights free minus the shared-pool funding for one full-length \
                 request); clamping num_slots to {affordable}. Lower --max-total-tokens \
                 (max_seq_len {}) to raise concurrency.",
                per_slot >> 20,
                self.max_seq_len,
            );
        }
        if requested == 1 {
            return Ok(Qwen35KvBudgetPlan {
                num_slots: planned,
                pool_pages: requested_pages,
            });
        }
        // Pool pages at the REDUCED slot count: n is rank-identical, and pages
        // shrink as n grows, so min-reducing the per-rank profile at the same n
        // yields a capacity every rank can hold.
        let pages_local: i32 = match probe {
            Some((free, total)) => {
                let profiled_tokens = pool_tokens_at(free, total, planned);
                if profiled_tokens == infer_seam::PROFILE_KV_TOKENS_FLOOR {
                    // `mem_fraction_static` bounds the engine's share of TOTAL, so a
                    // value under the weights' own share caps admission at 4096
                    // tokens even at num_slots 1.
                    let reserve = (total as f64 * (1.0 - mem_fraction_static)) as u64;
                    log::warn!(
                        "KV pool collapsed to the {}-token floor even at num_slots {planned}: \
                         free {}MB − recurrent {}MB − reserve {}MB (= total {}MB × (1 − \
                         mem_fraction_static {mem_fraction_static})) leaves nothing for \
                         {cell_bytes_per_token}B/tok cells. Raise mem_fraction_static, or free \
                         VRAM: every prompt over {} tokens will abort.",
                        infer_seam::PROFILE_KV_TOKENS_FLOOR,
                        free >> 20,
                        per_slot.saturating_mul(planned) >> 20,
                        reserve >> 20,
                        total >> 20,
                        infer_seam::PROFILE_KV_TOKENS_FLOOR,
                    );
                }
                let profiled_pages = (profiled_tokens / PAGE as u64).max(1) as usize;
                log::info!(
                    "CUDA Qwen3.6 full-attn KV pool profiled from measured VRAM: free {}MB / \
                     total {}MB, recurrent reservation {}MB ({planned} slots × {}MB), \
                     mem_fraction_static {mem_fraction_static}, cell {cell_bytes_per_token}B/tok \
                     ({num_full} full-attn layers × {local_kv_heads} kv-heads × {head_dim} hd) \
                     -> max_total_tokens {profiled_tokens} ({profiled_pages} pages); requested \
                     {requested_pages} pages (advisory)",
                    free >> 20,
                    total >> 20,
                    per_slot.saturating_mul(planned) >> 20,
                    per_slot >> 20,
                );
                i32::try_from(profiled_pages).unwrap_or(i32::MAX)
            }
            None => i32::MAX,
        };
        let pool_pages = self.tp.all_reduce_min_scalar_i32(&self.ctx, pages_local)? as usize;
        let pool_pages = if pool_pages == i32::MAX as usize {
            // Every rank's free-VRAM probe failed → the requested floor, matching
            // the old profile-failure fallback.
            log::warn!(
                "CUDA Qwen3.6 full-attn KV pool: free-VRAM probe failed on every rank; \
                 falling back to requested floor {requested_pages} pages"
            );
            requested_pages
        } else {
            pool_pages
        };
        Ok(Qwen35KvBudgetPlan {
            num_slots: planned,
            pool_pages,
        })
    }

    /// Per-linear-layer gated-delta input capture (spec verify only). When
    /// `capture` is `Some`, each linear layer copies its post-in_proj `qkv`
    /// (PRE-conv1d) + `b`/`a` projections for all rows into the capture so a
    /// partial-accept replay can re-run only conv1d + GDR (see
    /// [`Qwen35LinearCapture`]).
    pub(crate) fn forward_hidden_capture(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        capture: Option<&mut Qwen35LinearCapture>,
        recall: Option<&mut Qwen35PagedForward>,
        taps: Option<&mut dspark::Qwen35DsparkTaps>,
    ) -> Result<()> {
        ensure!(
            !tokens.is_empty(),
            "Qwen3.5 hybrid forward requires at least one token"
        );
        ensure!(
            slot.seq_len == start_pos,
            "Qwen3.5 hybrid slot seq_len {} != start_pos {start_pos} (uncached full-prefix \
             path requires contiguous appends)",
            slot.seq_len
        );
        let seq_len = tokens.len();
        ensure!(
            start_pos + seq_len <= self.max_seq_len,
            "Qwen3.5 hybrid sequence {} exceeds KV cache budget {}",
            start_pos + seq_len,
            self.max_seq_len
        );
        self.stage_step_inputs(ws, tokens, start_pos)?;
        let mut rows = [LinearRow {
            slot,
            len: seq_len,
            capture,
        }];
        self.forward_hidden_staged(&mut rows, ws, start_pos, recall, taps)?;
        rows[0].slot.seq_len += seq_len;
        Ok(())
    }

    /// Stage the per-step HOST inputs (token ids, start_pos) into their
    /// persistent device slots. This is the ONLY H2D traffic of a decode step;
    /// the captured decode graph runs it OUTSIDE the capture/replay (the dense
    /// `stage1_write` pattern), so the graph body below is a pure GPU kernel
    /// sequence. Returns the `(token_ids, start_pos)` device addresses for the
    /// graph-bake fingerprint (length-matched slot reuse keeps them stable; a
    /// change means the captured graph reads stale memory and must recapture).
    pub(crate) fn stage_step_inputs(
        &self,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(u64, u64)> {
        let seq_len = tokens.len();
        crate::profile::profile_op(&self.ctx, "stage_inputs", None, seq_len, || {
            let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let token_ids = ws.token_ids.upload(&self.ctx, &token_ids_host)?;
            let (token_ids_ptr, _g0) = token_ids.device_ptr(&self.ctx.stream);
            let start_pos_dev = ws.start_pos.upload(&self.ctx, &[start_pos as i32])?;
            let (start_pos_ptr, _g1) = start_pos_dev.device_ptr(&self.ctx.stream);
            Ok((token_ids_ptr, start_pos_ptr))
        })
    }

    /// The pure-GPU layer stack over already-staged inputs: embeds the staged
    /// token ids, runs every layer over the workspace buffers, and advances
    /// each row's recurrent/conv/KV device state in place. Does NOT advance
    /// `slot.seq_len` and performs NO H2D/D2H/sync — at one row of one token
    /// this is the CUDA-graph-capturable decode body (every per-step scalar is
    /// read from the staged device buffers).
    ///
    /// `rows` are the ragged per-slot token spans, token-major and in the same
    /// order as the staged ids; the full-attn layers get their identity from
    /// `recall`'s page table instead. `start_pos` (host) is consumed only by
    /// the multi-token NON-paged attention launch, which is single-row.
    pub(crate) fn forward_hidden_staged(
        &self,
        rows: &mut [LinearRow<'_>],
        ws: &mut Qwen35Workspace,
        _start_pos: usize,
        mut recall: Option<&mut Qwen35PagedForward>,
        mut taps: Option<&mut dspark::Qwen35DsparkTaps>,
    ) -> Result<()> {
        // Single chokepoint for every recurrent-reading forward (prefill /
        // decode / spec-capture / OPD): each row's recurrent block MUST be
        // resident — a missed `acquire_recurrent` hook would otherwise read an
        // empty `gdr_states` as a silent no-op.
        ensure!(
            rows.iter().all(|r| r.slot.has_recurrent()),
            "Qwen3.6 forward: slot recurrent state not acquired (missing \
             acquire_recurrent at the start_pos==0 prefill)"
        );
        let seq_len: usize = rows.iter().map(|r| r.len).sum();
        ensure!(
            recall.is_some() || rows.len() == 1,
            "Qwen3.6 forward: the contiguous full-attn cache is single-row, got {} rows",
            rows.len()
        );
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden_size = c.hidden_size;

        // Destructure once: each binding borrows its own workspace field, so
        // the residual-stream buffers, the per-block scratch, and the MoE
        // scratch stay simultaneously borrowable across the layer loop.
        let Qwen35Workspace {
            token_ids,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            ..
        } = ws;

        // Shape-matched re-gets return the SAME buffers `stage_step_inputs`
        // just wrote (no H2D here — a mismatch would mean staging was skipped,
        // which the two call paths above make impossible).
        let token_ids = &*token_ids.get(&self.ctx, seq_len)?;

        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        crate::profile::profile_op(&self.ctx, "embedding", None, seq_len, || {
            embedding_batch(&self.ctx, &self.embed_tokens, token_ids, hidden)
        })?;
        // DSpark tap: `-1` = the embedding output (residual stream pre-layer-0).
        if let Some(t) = taps.as_deref_mut() {
            t.capture(&self.ctx, -1, hidden)?;
        }
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        let hidden_mid = hidden_mid.get(&self.ctx, hidden_size, seq_len)?;
        let attn_out = attn_out.get(&self.ctx, hidden_size, seq_len)?;
        let mlp_out = mlp_out.get(&self.ctx, hidden_size, seq_len)?;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            crate::profile::profile_op(&self.ctx, "input_norm", Some(layer_idx), seq_len, || {
                rms_norm_offset(&self.ctx, hidden, &layer.input_layernorm, eps, normed)
            })?;

            match &layer.attn {
                Qwen35Attn::Full(full_attn) => {
                    crate::profile::profile_op(
                        &self.ctx,
                        "full_attention",
                        Some(layer_idx),
                        seq_len,
                        || {
                            if let Some(rc) = recall.as_deref_mut() {
                                self.full_attention_paged(
                                    full_attn,
                                    normed,
                                    full_idx,
                                    rc.pool,
                                    rc.meta,
                                    rc.cp.as_ref(),
                                    rc.cp_decode.as_ref(),
                                    full,
                                    attn_out,
                                )
                            } else {
                                bail!(
                                    "Qwen3.5 full attention without a paged recall pool is unsupported \
                                     (legacy contiguous KV lane removed)"
                                )
                            }
                        },
                    )?;
                    full_idx += 1;
                }
                Qwen35Attn::Linear(lin) => {
                    let cp = recall.as_deref().and_then(|rc| rc.cp.as_ref());
                    let cp_decode = recall.as_deref().and_then(|rc| rc.cp_decode.as_ref());
                    crate::profile::profile_op(
                        &self.ctx,
                        "linear_attention",
                        Some(layer_idx),
                        seq_len,
                        || {
                            self.linear_attention(
                                lin,
                                normed,
                                LinearCore::Rows(&mut *rows),
                                linear_idx,
                                linear,
                                cp,
                                cp_decode,
                                attn_out,
                            )
                        },
                    )?;
                    linear_idx += 1;
                }
            }

            crate::profile::profile_op(
                &self.ctx,
                "post_attn_norm",
                Some(layer_idx),
                seq_len,
                || {
                    add_batch(&self.ctx, hidden, attn_out, hidden_mid)?;
                    rms_norm_offset(
                        &self.ctx,
                        hidden_mid,
                        &layer.post_attention_layernorm,
                        eps,
                        normed,
                    )
                },
            )?;
            let mlp_in: &HiddenStates = normed;
            if let Some(moe_weights) = &layer.moe {
                let cfg = self
                    .moe_config
                    .as_ref()
                    .ok_or_else(|| anyhow!("MoE layer present but model has no moe_config"))?;
                crate::profile::profile_op(&self.ctx, "moe_ffn", Some(layer_idx), seq_len, || {
                    moe_forward_into(
                        &self.ctx,
                        moe_weights,
                        mlp_in,
                        cfg,
                        &self.expert_split,
                        moe,
                        mlp_out,
                    )
                })?;
            } else {
                let mlp = layer
                    .mlp
                    .as_ref()
                    .ok_or_else(|| anyhow!("dense layer missing both mlp and moe weights"))?;
                crate::profile::profile_op(
                    &self.ctx,
                    "dense_ffn",
                    Some(layer_idx),
                    seq_len,
                    || self.dense_mlp(mlp, mlp_in, dense, mlp_out),
                )?;
            }
            // ONE all-reduce covers the whole FFN partial: the MoE buffer already
            // sums this rank's routed experts (non-local routes contribute zero)
            // + the column/row-sharded shared expert; the dense branch is a
            // row-parallel down_proj partial. No-op on a single GPU.
            crate::profile::profile_op(
                &self.ctx,
                "ffn_allreduce",
                Some(layer_idx),
                seq_len,
                || self.tp.all_reduce_sum(&self.ctx, mlp_out),
            )?;

            // MLP residual add producing the next layer's residual stream.
            // The post-attn sum lives in `hidden_mid`; add_batch reads
            // hidden_mid/mlp_out and writes `hidden` (whose previous value is
            // dead).
            crate::profile::profile_op(
                &self.ctx,
                "ffn_residual",
                Some(layer_idx),
                seq_len,
                || add_batch(&self.ctx, hidden_mid, mlp_out, hidden),
            )?;
            if let Some(t) = taps.as_deref_mut() {
                t.capture(&self.ctx, layer_idx as i64, hidden)?;
            }
        }

        Ok(())
    }

    /// Forward over the opt-in paged recall KV pool (`--kv-recall`): the
    /// full-attn layers read/write `recall.pool` over `recall.meta`.
    /// `slot.seq_len` is still advanced in lockstep (the executor's decode
    /// invariant reads it); the pool's own `seq_len` is advanced separately
    /// via `alloc_tokens`.
    pub(crate) fn forward_tokens_recall(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
        recall: &mut Qwen35PagedForward,
    ) -> Result<(u32, Option<f32>)> {
        self.forward_tokens_recall_tapped(
            slot, ws, tokens, start_pos, params, position, penalty, recall, None,
        )
    }

    /// [`Self::forward_tokens_recall`] with an optional DSpark trunk-tap
    /// capture (`--spec-type dspark` prefill/warm steps). `None` is
    /// byte-identical to the untapped path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tokens_recall_tapped(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
        recall: &mut Qwen35PagedForward,
        taps: Option<&mut dspark::Qwen35DsparkTaps>,
    ) -> Result<(u32, Option<f32>)> {
        ensure!(
            !tokens.is_empty(),
            "Qwen3.5 recall forward requires at least one token"
        );
        let seq_len = tokens.len();
        self.stage_step_inputs(ws, tokens, start_pos)?;
        let mut rows = [LinearRow {
            slot,
            len: seq_len,
            capture: None,
        }];
        crate::profile::profile_op(&self.ctx, "forward_hidden", None, seq_len, || {
            self.forward_hidden_staged(&mut rows, ws, start_pos, Some(recall), taps)
        })?;
        rows[0].slot.seq_len += seq_len;
        crate::profile::profile_op(&self.ctx, "lm_head", None, seq_len, || {
            self.lm_head_logits(ws, seq_len)
        })?;
        crate::profile::profile_op(&self.ctx, "sample", None, seq_len, || {
            self.sample_workspace_logits(ws, params, position, penalty)
        })
    }

    /// Final norm (offset) + LM head on the last token only, into `ws.logits`.
    /// Last stage of the captured decode graph (the capture ends at the logits
    /// buffer, dense-style); sampling stays outside.
    ///
    /// TP invariant: embed/lm_head are replicated and `hidden` is
    /// post-all-reduce (every row-parallel output above was summed), so the
    /// logits — and therefore the sampled token — are identical on every
    /// rank. No rank ever needs to broadcast its sample.
    pub(crate) fn lm_head_logits(&self, ws: &mut Qwen35Workspace, seq_len: usize) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let hidden_size = self.config.hidden_size;
        let Qwen35Workspace {
            hidden,
            last_hidden,
            last_normed,
            logits,
            ..
        } = ws;
        // Shape-matched re-get returns the SAME buffer forward_hidden filled.
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let last_hidden = last_hidden.get(&self.ctx, hidden_size)?;
        crate::profile::profile_op(&self.ctx, "lm_head_slice", None, seq_len, || {
            copy_row_to_vec(&self.ctx, hidden, seq_len - 1, last_hidden)
        })?;
        let last_normed = last_normed.get(&self.ctx, hidden_size)?;
        crate::profile::profile_op(&self.ctx, "final_norm", None, seq_len, || {
            rms_norm_offset_vec(&self.ctx, last_hidden, &self.norm, eps, last_normed)
        })?;
        let logits = logits.get(&self.ctx, self.output_projection().rows)?;
        crate::profile::profile_op(&self.ctx, "lm_head_gemv", None, seq_len, || {
            gemv(&self.ctx, self.output_projection(), last_normed, logits)
        })?;
        Ok(())
    }

    /// Sample the next token from `ws.logits` (written by
    /// [`Self::lm_head_logits`] — eagerly or by a decode-graph replay). Greedy
    /// uses the persistent `argmax_out` slot (zero per-token allocations);
    /// non-greedy reads the logits to host. Always OUTSIDE any capture (syncs).
    pub(crate) fn sample_workspace_logits(
        &self,
        ws: &mut Qwen35Workspace,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
    ) -> Result<(u32, Option<f32>)> {
        let Qwen35Workspace {
            logits,
            argmax_out,
            top_logprobs,
            ..
        } = ws;
        let logits = logits.get(&self.ctx, self.output_projection().rows)?;
        let argmax_out = argmax_out.get(&self.ctx, 1)?;
        crate::executor::sample_cuda_token_captured(
            &self.ctx,
            logits,
            params,
            position,
            argmax_out,
            penalty,
            top_logprobs,
        )
    }

    /// Device address of the workspace logits buffer (allocating it at vocab
    /// size if absent) — the decode-graph bake fingerprint's output anchor.
    pub(crate) fn workspace_logits_ptr(&self, ws: &mut Qwen35Workspace) -> Result<u64> {
        let logits = ws.logits.get(&self.ctx, self.output_projection().rows)?;
        let (ptr, _g) = logits.data.device_ptr(&self.ctx.stream);
        Ok(ptr)
    }

    pub(crate) fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// MTP spec-decode draft depth this model was built for (`0` = spec off).
    /// The executor's `mtp_decode_row` uses this to size `spec_step`'s depth.
    pub(crate) fn spec_draft_tokens(&self) -> usize {
        self.spec_draft_tokens
    }

    pub(crate) fn local_kv_heads(&self) -> usize {
        self.local_kv_heads
    }

    pub(crate) fn local_q_heads(&self) -> usize {
        self.local_q_heads
    }

    /// Run the full forward over `tokens` and return the FULL `[seq_len, vocab]`
    /// logits (every row, not just the last) WITHOUT sampling. Mirrors
    /// [`Self::forward_tokens`]'s layer stack but, instead of slicing the last
    /// row + a single `gemv`, applies the final offset-RMSNorm over the whole
    /// batch and a batched lm-head GEMM, returning the device logits buffer plus
    /// its `[seq_len, vocab]` shape.
    ///
    /// This is the OPD teacher raw-logits path: the distillation loss needs the
    /// teacher's per-position logit distribution over the prompt, so it cannot
    /// use the sampling tail. `slot` carries the per-slot KV + recurrent state
    /// exactly like [`Self::forward_tokens`].
    pub(crate) fn forward_token_logits_full(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        recall: Option<&mut Qwen35PagedForward>,
    ) -> Result<(DeviceVec, [usize; 2])> {
        self.forward_hidden_capture(slot, ws, tokens, start_pos, None, recall, None)?;
        let seq_len = tokens.len();
        let eps = self.config.rms_norm_eps;
        let hidden_size = self.config.hidden_size;

        // Final norm (offset) over the WHOLE batch, then the batched lm-head GEMM
        // produces every row's logits — no last-row slice, no sampling.
        // (TP invariant as in `forward_tokens`: replicated lm_head over
        // post-all-reduce hidden ⇒ identical logits on every rank.)
        let Qwen35Workspace { hidden, normed, .. } = ws;
        // Shape-matched re-gets return the SAME buffers forward_hidden used;
        // rms_norm fully overwrites `normed` before the lm-head GEMM reads it.
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        crate::profile::profile_op(&self.ctx, "final_norm", None, seq_len, || {
            rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)
        })?;
        let vocab = self.output_projection().rows;
        // The logits buffer is RETURNED to the OPD caller (ownership leaves the
        // forward), so it stays a per-call allocation — not a workspace slot.
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, seq_len)?;
        crate::profile::profile_op(&self.ctx, "lm_head_gemm", None, seq_len, || {
            gemm_batch(&self.ctx, self.output_projection(), normed, &mut logits)
        })?;
        self.ctx.sync()?;

        // `HiddenStates` is a `[vocab, seq_len]` column-batch over a flat device
        // buffer; reinterpret it as a `[seq_len, vocab]` row-major `DeviceVec`
        // (the train OPD bridge reads it as `[1, seq_len, vocab]`).
        let logits_vec = DeviceVec {
            data: logits.data,
            len: seq_len * vocab,
            label: "qwen35_token_logits[seq,vocab]",
        };
        Ok((logits_vec, [seq_len, vocab]))
    }

    /// Like [`Self::forward_token_logits_full`] but ALSO returns the LAST token's
    /// RAW trunk hidden (pre-final-norm) in a freshly-owned `DeviceVec`. The
    /// NextN-MTP draft head consumes that hidden as its level-0 input
    /// (`fc(concat[embed(tok), h])`), so spec-decode needs both the verify logits
    /// and the trunk hidden from a single forward. Byte-identical to
    /// `forward_token_logits_full` for the logits; the extra `copy_row_to_vec`
    /// captures the hidden BEFORE the lm-head norm reuses the workspace.
    /// (Spec-decode port increment 3a; gated path, no default-decode change.)
    #[allow(dead_code)] // consumed by mtp_forward_level / spec_step (next increment)
    pub(crate) fn forward_tokens_with_hidden(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        recall: Option<&mut Qwen35PagedForward>,
    ) -> Result<(DeviceVec, [usize; 2], DeviceVec)> {
        self.forward_hidden_capture(slot, ws, tokens, start_pos, None, recall, None)?;
        let seq_len = tokens.len();
        let eps = self.config.rms_norm_eps;
        let hidden_size = self.config.hidden_size;
        // Capture the last row's raw trunk hidden into an OWNED buffer first —
        // the lm-head norm below overwrites `ws.normed` from `ws.hidden`.
        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        {
            let hidden = ws.hidden.get(&self.ctx, hidden_size, seq_len)?;
            crate::profile::profile_op(&self.ctx, "lm_head_slice", None, seq_len, || {
                copy_row_to_vec(&self.ctx, hidden, seq_len - 1, &mut last_hidden)
            })?;
        }
        let Qwen35Workspace { hidden, normed, .. } = ws;
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        crate::profile::profile_op(&self.ctx, "final_norm", None, seq_len, || {
            rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)
        })?;
        let vocab = self.output_projection().rows;
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, seq_len)?;
        crate::profile::profile_op(&self.ctx, "lm_head_gemm", None, seq_len, || {
            gemm_batch(&self.ctx, self.output_projection(), normed, &mut logits)
        })?;
        self.ctx.sync()?;
        let logits_vec = DeviceVec {
            data: logits.data,
            len: seq_len * vocab,
            label: "qwen35_token_logits[seq,vocab]",
        };
        Ok((logits_vec, [seq_len, vocab], last_hidden))
    }

    /// Like [`Self::forward_token_logits_full`] but ALSO returns EVERY row's raw
    /// trunk hidden (pre-final-norm) as one owned `[seq_len, hidden]` (token-major)
    /// `DeviceVec`. The depth>1 spec-decode verify needs the ACCEPTED row's hidden
    /// (not just the last) to seed the next block's level-0 draft — the accepted
    /// prefix length is only known after host-argmax, so all rows are captured.
    /// Byte-identical logits to `forward_token_logits_full`; the extra D2D copy of
    /// `ws.hidden` happens BEFORE the lm-head norm reuses the workspace.
    #[allow(dead_code)] // consumed by spec_step (next increment)
    pub(crate) fn forward_tokens_verify(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        tokens: &[u32],
        start_pos: usize,
        capture: Option<&mut Qwen35LinearCapture>,
        recall: Option<&mut Qwen35PagedForward>,
    ) -> Result<(DeviceVec, [usize; 2], DeviceVec)> {
        self.forward_hidden_capture(slot, ws, tokens, start_pos, capture, recall, None)?;
        let seq_len = tokens.len();
        let eps = self.config.rms_norm_eps;
        let hidden_size = self.config.hidden_size;
        // Capture ALL rows' raw (pre-final-norm) trunk hidden into an owned
        // [seq_len, hidden] (token-major) buffer — the spec step seeds the next
        // block's level-0 draft from the accepted row's hidden.
        let mut hiddens = DeviceVec::zeros(&self.ctx, seq_len * hidden_size)?;
        {
            let hidden = ws.hidden.get(&self.ctx, hidden_size, seq_len)?;
            crate::profile::profile_op(&self.ctx, "verify_hidden_capture", None, seq_len, || {
                self.ctx
                    .stream
                    .memcpy_dtod(&hidden.data, &mut hiddens.data)
                    .map_err(|e| anyhow!("verify hidden batch capture failed: {e}"))
            })?;
        }
        let Qwen35Workspace { hidden, normed, .. } = ws;
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        crate::profile::profile_op(&self.ctx, "final_norm", None, seq_len, || {
            rms_norm_offset(&self.ctx, hidden, &self.norm, eps, normed)
        })?;
        let vocab = self.output_projection().rows;
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, seq_len)?;
        crate::profile::profile_op(&self.ctx, "lm_head_gemm", None, seq_len, || {
            gemm_batch(&self.ctx, self.output_projection(), normed, &mut logits)
        })?;
        // No sync: the caller's argmax_row_into reads these logits on the same
        // stream — stream ordering already guarantees the GEMM completed (#198).
        let logits_vec = DeviceVec {
            data: logits.data,
            len: seq_len * vocab,
            label: "qwen35_verify_logits[seq,vocab]",
        };
        Ok((logits_vec, [seq_len, vocab], hiddens))
    }

    /// Trunk forward for offline DSpark draft training: the raw residual-stream
    /// taps at `target_layer_ids` as `[seq, taps·hidden]`, and the final-normed
    /// hidden states as `[seq, hidden]`, both on the host.
    ///
    /// Serving fuses the taps on device and never materializes them; training
    /// needs them raw because `fc`/`hidden_norm` are the parameters being
    /// learned. No lm-head GEMM here — the draft's distillation target is
    /// computed from `last_hidden` against the same weights on the train side.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_training_taps(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        taps: &mut dspark::Qwen35DsparkTaps,
        target_layer_ids: &[i64],
        tokens: &[u32],
        recall: Option<&mut Qwen35PagedForward>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(!tokens.is_empty(), "training forward needs tokens");
        ensure!(
            !target_layer_ids.is_empty(),
            "training forward needs target_layer_ids"
        );
        let seq_len = tokens.len();
        let hidden_size = self.config.hidden_size;
        dspark::Qwen35DsparkTaps::validate(target_layer_ids, self.config.num_hidden_layers)?;
        taps.prepare(target_layer_ids, hidden_size, seq_len);
        self.forward_hidden_capture(slot, ws, tokens, 0, None, recall, Some(taps))?;

        // Taps come off first: the final norm below reuses `ws.normed`, and a
        // tap buffer is the draft's only view of the trunk's interior.
        let ntaps = target_layer_ids.len();
        let mut features = vec![0.0f32; seq_len * ntaps * hidden_size];
        for t in 0..ntaps {
            let tap = taps.tap_to_host(&self.ctx, t)?;
            for r in 0..seq_len {
                let dst = r * ntaps * hidden_size + t * hidden_size;
                features[dst..dst + hidden_size]
                    .copy_from_slice(&tap[r * hidden_size..][..hidden_size]);
            }
        }
        taps.release();

        let Qwen35Workspace { hidden, normed, .. } = ws;
        let hidden = hidden.get(&self.ctx, hidden_size, seq_len)?;
        let normed = normed.get(&self.ctx, hidden_size, seq_len)?;
        crate::profile::profile_op(&self.ctx, "final_norm", None, seq_len, || {
            rms_norm_offset(
                &self.ctx,
                hidden,
                &self.norm,
                self.config.rms_norm_eps,
                normed,
            )
        })?;
        self.ctx.sync()?;
        let last_hidden = normed.to_host(&self.ctx)?;
        Ok((features, last_hidden))
    }
}
