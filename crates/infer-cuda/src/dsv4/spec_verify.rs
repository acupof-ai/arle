//! Single-sequence speculative verify: the row schedule, its validation, and the
//! verify entry points.

use super::layer_block::HcHalf;
use super::*;

/// Max depth for the per-slot spec-ring snapshot. `topk` widens candidate
/// matching only; verifier rows remain chain-shaped.
pub(crate) const MAX_SPEC_DRAFT_DEPTH: usize = 8;
/// Bounded chain verifier rows per slot. MTP uses `depth + 1`; `topk` adds none.
pub(crate) const MAX_SPEC_VERIFY_ROWS: usize = 64;
pub(crate) const DEFAULT_SPEC_DRAFT_DEPTH: usize = 2;

pub(crate) const DEFAULT_SPEC_DRAFT_TOPK: usize = 1;

/// Row schedule for one speculative verify forward. `ancestors` is the prefix
/// metadata the batched FlashMLA sparse verify lane reads: row `r` attends
/// committed KV plus the listed earlier chunk rows and self.
pub(crate) struct SpecVerifySchedule {
    /// Per row: absolute position (`start_pos + node depth`).
    pub(crate) positions: Vec<usize>,
    /// Per row: chunk-row ancestors, shallow to deep, self excluded.
    pub(crate) ancestors: Vec<Vec<usize>>,
}

impl SpecVerifySchedule {
    pub(crate) fn validate_sparse_at(&self, start_pos: usize) -> Result<()> {
        ensure!(
            !self.positions.is_empty() && self.positions.len() == self.ancestors.len(),
            "DSv4 sparse verify schedule shape mismatch: positions={} ancestors={}",
            self.positions.len(),
            self.ancestors.len()
        );
        ensure!(
            self.positions.len() <= MAX_SPEC_VERIFY_ROWS,
            "DSv4 sparse verify rows {} exceed fold cache rows {MAX_SPEC_VERIFY_ROWS}",
            self.positions.len()
        );
        for (row, &pos) in self.positions.iter().enumerate() {
            ensure!(
                pos >= start_pos,
                "DSv4 sparse verify row {row} position {pos} precedes start_pos {start_pos}"
            );
            for &ancestor in &self.ancestors[row] {
                ensure!(
                    ancestor < row,
                    "DSv4 sparse verify row {row} has non-causal ancestor row {ancestor}"
                );
                ensure!(
                    self.positions[ancestor] < pos,
                    "DSv4 sparse verify row {row} position {pos} ancestor {ancestor} position {} is not earlier",
                    self.positions[ancestor]
                );
            }
        }
        Ok(())
    }
}

/// Target verify output for an MTP verify chunk: the full `[rows, vocab]` logits
/// matrix plus the greedy top-1 view and per-row MTP stream hiddens.
pub(crate) struct SpecVerifyResult {
    pub(crate) logits: HiddenStates,
    pub(crate) argmax: Vec<u32>,
    pub(crate) hiddens: Vec<DeviceVec>,
}

impl Dsv4Model {
    /// Build a [`SpecVerifyResult`] from a forward's output stream: capture the
    /// per-row MTP stream hiddens, project the logits, and take the greedy
    /// argmax. Shared by the commit-verify and the frozen scheduled-verify
    /// lanes.
    fn spec_verify_result_from_stream(
        &self,
        stream: &HiddenStates,
        n: usize,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<SpecVerifyResult> {
        let stream_dim = self.config.hidden_size * self.config.hc_mult;
        let mut hiddens = Vec::with_capacity(n);
        for i in 0..n {
            let mut h = DeviceVec::zeros(&self.ctx, stream_dim)?;
            self.capture_mtp_stream_hidden(stream, i, 1, &mut h, keepalive)?;
            hiddens.push(h);
        }
        let logits = self.verify_logits_from_stream(stream, n, keepalive)?;
        let argmax = self.mtp_argmax_batch(&logits)?;
        Ok(SpecVerifyResult {
            logits,
            argmax,
            hiddens,
        })
    }

    /// Commit/selftest forward for a contiguous token prefix. Not the frozen MTP
    /// verifier: it writes the slot KV state like a normal forward.
    pub(crate) fn forward_tokens_verify(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        _position: u64,
    ) -> Result<SpecVerifyResult> {
        ensure!(
            !tokens.is_empty(),
            "DSv4 verify forward requires at least one token"
        );
        let n = tokens.len();
        let persist_spec_normed = slot.spec_normed.is_some();
        let (stream, mut keepalive) = self.forward_tokens_stream_impl(
            slot,
            kv_adapter,
            tokens,
            start_pos,
            persist_spec_normed,
            None,
        )?;
        let result = self.spec_verify_result_from_stream(&stream, n, &mut keepalive)?;
        std::hint::black_box(keepalive.len());
        drop(keepalive);
        Ok(result)
    }

    /// Verify `tokens` in ONE scheduled sparse forward under `sched`'s per-row
    /// positions. MTP schedules are chain-shaped (`depth + 1` rows), so this
    /// requires at least two rows.
    pub(crate) fn forward_tokens_verify_scheduled(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        _position: u64,
        sched: &SpecVerifySchedule,
    ) -> Result<SpecVerifyResult> {
        ensure!(
            !tokens.is_empty(),
            "DSv4 verify forward requires at least one token"
        );
        ensure!(
            tokens.len() >= 2,
            "DSv4 scheduled sparse verify requires at least 2 rows; use ordinary decode for single-token verify"
        );
        ensure!(
            sched.positions.len() == tokens.len() && sched.ancestors.len() == tokens.len(),
            "DSv4 verify schedule rows ({}, {}) != tokens {}",
            sched.positions.len(),
            sched.ancestors.len(),
            tokens.len()
        );
        // `hiddens[j]` = row j's MTP stream; `argmax[j]` = the target's argmax
        // AFTER `tokens[j]`. The frozen sparse verify lane commits no slot KV.
        let seq_len = tokens.len();
        let was_frozen = crate::attention::dsv4_verify_frozen();
        if !was_frozen {
            crate::attention::set_dsv4_verify_frozen(true);
        }
        let result = crate::profile::profile_op(&self.ctx, "lm_head", None, seq_len, || {
            let (stream, mut keepalive) = self.forward_tokens_stream_impl(
                slot,
                kv_adapter,
                tokens,
                start_pos,
                true,
                Some(sched),
            )?;
            let result = self.spec_verify_result_from_stream(&stream, seq_len, &mut keepalive)?;
            std::hint::black_box(keepalive.len());
            drop(keepalive);
            Ok(result)
        });
        if !was_frozen {
            crate::attention::set_dsv4_verify_frozen(false);
        }
        result
    }

    pub(super) fn forward_tokens_verify_stream_persistent(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        sched: &SpecVerifySchedule,
    ) -> Result<(HiddenStates, Dsv4ForwardKeepalive)> {
        ensure!(
            !tokens.is_empty(),
            "DSv4 persistent verify forward requires at least one token"
        );
        ensure!(
            slot.seq_len == start_pos,
            "DSv4 slot seq_len {} != start_pos {start_pos}; decode requires contiguous appends",
            slot.seq_len
        );
        ensure!(
            start_pos + tokens.len() <= slot.max_seq_len,
            "DSv4 sequence {} exceeds slot max_seq_len {}",
            start_pos + tokens.len(),
            slot.max_seq_len
        );
        let seq_len = tokens.len();
        let mega_epoch = self.begin_mega_moe_forward(seq_len)?;
        ensure!(
            seq_len <= MAX_SPEC_VERIFY_ROWS,
            "DSv4 persistent verify rows {seq_len} exceed capacity {MAX_SPEC_VERIFY_ROWS}"
        );
        ensure!(
            sched.positions.len() == seq_len && sched.ancestors.len() == seq_len,
            "DSv4 sparse verify schedule rows ({}, {}) != token rows {seq_len}",
            sched.positions.len(),
            sched.ancestors.len()
        );
        sched.validate_sparse_at(start_pos)?;
        ensure!(
            !crate::runtime_flags::dsv4_moe_transport()?.is_deepep(),
            "DSv4 persistent MTP verify scratch currently supports allreduce transport"
        );

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let ctx = &self.ctx;
        let mut keepalive = Dsv4ForwardKeepalive::new(false);
        let scratch = slot
            .spec_verify
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 scheduled verify missing persistent scratch"))?;
        scratch.set_rows(seq_len)?;

        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        crate::profile::profile_op(ctx, "embedding", None, seq_len, || {
            let token_ids = crate::ops::upload_i32(ctx, &token_ids_host)?;
            crate::ops::embedding_batch(
                ctx,
                &self.embed_tokens,
                &token_ids,
                &mut scratch.embeddings,
            )?;
            crate::hc::initial_stream_from_embeddings(
                ctx,
                &scratch.embeddings,
                hidden_size,
                hc_mult,
                &mut scratch.initial_stream,
            )
        })?;

        let sparse_verify_meta = crate::attention::Dsv4ChainVerifyAttnMeta::new(
            ctx,
            &sched.positions,
            &sched.ancestors,
        )?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (prev_layers, current_and_rest) = scratch.layers.split_at_mut(layer_idx);
            let current = current_and_rest
                .first_mut()
                .ok_or_else(|| anyhow!("DSv4 spec-verify layer scratch {layer_idx} missing"))?;

            {
                let stream = if layer_idx == 0 {
                    &scratch.initial_stream
                } else {
                    &prev_layers[layer_idx - 1].ffn_stream
                };
                let normed = &mut current.attn_normed;
                let attn_mhc = self.hc_pre_norm(
                    layer,
                    HcHalf::Attn,
                    layer_idx,
                    seq_len,
                    stream,
                    normed,
                    &mut keepalive,
                )?;

                if let Some(cache) = slot.spec_normed.as_mut() {
                    let rows = seq_len * hidden_size;
                    let src = normed.data.slice(0..rows);
                    let mut dst = cache[layer_idx].data.slice_mut(0..rows);
                    ctx.stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| anyhow!("DSv4 commit-fold normed persist failed: {e}"))?;
                }

                crate::profile::profile_op(ctx, "attention", Some(layer_idx), seq_len, || {
                    let (layer_pool, dsa_shared, flashmla_scratch, prefill_shared, fp32) =
                        kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                    crate::attention::mla_attention(
                        ctx,
                        &self.config,
                        &layer.attention,
                        layer.mode,
                        layer.compress_ratio,
                        layer_idx,
                        normed,
                        &mut slot.attention[layer_idx],
                        layer_pool,
                        dsa_shared,
                        flashmla_scratch,
                        prefill_shared,
                        fp32,
                        crate::attention::Dsv4Position {
                            start: start_pos,
                            device: None,
                        },
                        Some(&sparse_verify_meta),
                        &self.tp,
                        &mut current.attn_out,
                        &mut keepalive,
                    )
                })?;

                crate::profile::profile_op(
                    ctx,
                    "attn_allreduce",
                    Some(layer_idx),
                    seq_len,
                    || self.tp.all_reduce_sum(ctx, &mut current.attn_out),
                )?;

                self.hc_post_fold(
                    attn_mhc.as_ref(),
                    HcHalf::Attn,
                    layer_idx,
                    seq_len,
                    &current.attn_out,
                    stream,
                    &mut current.attn_stream,
                )?;
            }

            {
                let stream = &current.attn_stream;
                let normed = &mut current.ffn_normed;
                let ffn_mhc = self.hc_pre_norm(
                    layer,
                    HcHalf::Ffn,
                    layer_idx,
                    seq_len,
                    stream,
                    normed,
                    &mut keepalive,
                )?;

                if let Some(dense) = layer.dense_mlp.as_ref() {
                    crate::profile::profile_op(ctx, "mlp", Some(layer_idx), seq_len, || {
                        dsv4_dense_mlp_forward(
                            ctx,
                            dense,
                            normed,
                            &mut current.moe_with_shared,
                            self.config.swiglu_limit,
                            &mut keepalive,
                        )
                    })?;
                } else {
                    let needs_moe_allreduce = crate::profile::profile_op(
                        ctx,
                        "moe_route",
                        Some(layer_idx),
                        seq_len,
                        || {
                            crate::moe::dsv4_moe_forward(
                                self,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                tokens,
                                normed,
                                &mut current.moe_out,
                                &mut keepalive,
                                None,
                                mega_epoch,
                            )
                        },
                    )?;

                    if needs_moe_allreduce {
                        crate::profile::profile_op(
                            ctx,
                            "moe_allreduce",
                            Some(layer_idx),
                            seq_len,
                            || self.tp.all_reduce_sum(ctx, &mut current.moe_out),
                        )?;
                    }

                    crate::profile::profile_op(ctx, "shared_hc", Some(layer_idx), seq_len, || {
                        let (shared_out, shared_scratch) = kv_adapter.shared_expert_decode_mut();
                        let shared = shared_out;
                        shared.seq_len = seq_len;
                        ensure!(
                            shared.hidden_dim == hidden_size,
                            "DSv4 shared verify scratch hidden {} != {}",
                            shared.hidden_dim,
                            hidden_size
                        );
                        let shared_scratch = shared_scratch
                            .ok_or_else(|| anyhow!("DSv4 verify requires shared-expert scratch"))?;
                        crate::moe::dsv4_shared_expert_forward_decode_scratch(
                            ctx,
                            &ctx.stream,
                            layer.moe.as_ref().expect("DSv4 layer.moe"),
                            normed,
                            shared,
                            self.config.swiglu_limit,
                            shared_scratch,
                        )?;
                        crate::ops::add_batch(
                            ctx,
                            &current.moe_out,
                            shared,
                            &mut current.moe_with_shared,
                        )
                    })?;
                }

                self.hc_post_fold(
                    ffn_mhc.as_ref(),
                    HcHalf::Ffn,
                    layer_idx,
                    seq_len,
                    &current.moe_with_shared,
                    stream,
                    &mut current.ffn_stream,
                )?;

                if self.config.is_dspark()
                    && let Some(tap_idx) = self
                        .config
                        .dspark_target_layer_ids
                        .iter()
                        .position(|&l| l == layer_idx)
                {
                    let elems = stream_dim * seq_len;
                    let src = current.ffn_stream.data.slice(0..elems);
                    let mut dst = slot.dspark_taps[tap_idx].data.slice_mut(0..elems);
                    ctx.stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| anyhow!("DSpark verify tap capture D2D failed: {e}"))?;
                }
            }
        }

        // SAFETY: uninit device scratch; fully written before first read.
        let mut stream_out = unsafe { HiddenStates::uninit(ctx, stream_dim, seq_len)? };
        let elems = stream_dim * seq_len;
        let final_stream = scratch
            .layers
            .last()
            .ok_or_else(|| anyhow!("DSv4 spec-verify scratch has no layers"))?;
        let src = final_stream.ffn_stream.data.slice(0..elems);
        ctx.stream
            .memcpy_dtod(&src, &mut stream_out.data)
            .map_err(|e| anyhow!("DSv4 persistent verify stream export failed: {e}"))?;
        slot.seq_len += seq_len;
        Ok((stream_out, keepalive))
    }
}
