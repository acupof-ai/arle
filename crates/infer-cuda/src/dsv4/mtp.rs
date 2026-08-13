//! The MTP draft head: one frozen-KV draft level and its candidate selection.

use super::*;

/// One draft row to expand in an MTP head pass ([`Dsv4Model::mtp_forward_level`]).
pub(crate) struct MtpDraftRow {
    pub token: u32,
}

impl Dsv4Model {
    /// Draft one MTP matrix level. `rows` is the draft batch for this level
    /// (`m == 1` single-slot; one row per slot for the cross-slot batched draft).
    /// `slot_ids[r]` selects row `r`'s KV slot; `positions[r]` is its draft
    /// position. Returns per row: top-k candidate tokens (highest first) + the
    /// wide MTP stream the next draft row branches from. Candidates come from k
    /// rounds of device argmax + mask — no full-vocab D2H.
    pub(crate) fn mtp_forward_level(
        &self,
        slots: &mut [Dsv4SlotState],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_ids: &[usize],
        rows: &[MtpDraftRow],
        h_prevs: &[&DeviceVec],
        positions: &[u64],
        top_k: usize,
    ) -> Result<Vec<(Vec<u32>, DeviceVec)>> {
        ensure!(
            self.spec_decode_on,
            "DSv4 MTP forward called while spec decode is off (need --spec-type mtp / \
             --mtp-draft-tokens)"
        );
        ensure!(
            !crate::runtime_flags::dsv4_moe_transport()?.is_deepep(),
            "DSv4 MTP Phase 1 supports allreduce transport only"
        );
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 MTP requested but the draft head is not loaded"))?;
        let m = rows.len();
        ensure!(
            m > 0 && h_prevs.len() == m && slot_ids.len() == m && positions.len() == m,
            "DSv4 MTP level shape mismatch (rows {m}, h_prevs {}, slot_ids {}, positions {})",
            h_prevs.len(),
            slot_ids.len(),
            positions.len()
        );
        let mega_epoch = self.begin_mega_moe_forward(m)?;
        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        for h in h_prevs {
            ensure!(
                h.len == stream_dim,
                "DSv4 MTP h_prev len {} != stream dim {stream_dim}",
                h.len
            );
        }
        let eps = self.config.rms_norm_eps;
        let ctx = &self.ctx;
        let mut keepalive = Dsv4ForwardKeepalive::new(false);

        // ── h' = e_proj(enorm(emb(token))) + h_proj(hnorm(h_prev)), batched.
        let token_ids_host: Vec<i32> = rows.iter().map(|r| r.token as i32).collect();
        let token_ids = crate::ops::upload_i32(ctx, &token_ids_host)?;
        // SAFETY: embedding_batch writes the full [m, hidden_size] buffer.
        let mut emb = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::profile::profile_op(ctx, "embedding", None, m, || {
            crate::ops::embedding_batch(ctx, &self.embed_tokens, &token_ids, &mut emb)
        })?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut emb_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::profile::profile_op(ctx, "mtp_enorm", None, m, || {
            crate::ops::rms_norm_batch(ctx, &emb, &mtp.enorm, eps, &mut emb_normed)
        })?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut e_proj = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::profile::profile_op(ctx, "mtp_e_proj", None, m, || {
            crate::attention::dsv4_linear(ctx, &mtp.e_proj, &emb_normed, &mut e_proj)
        })?;

        // Gather h_prev streams into [m * hc_mult, hidden] (a stream is
        // hc_mult consecutive hidden rows, token-major).
        // SAFETY: uninit device scratch; fully written before first read.
        let mut h_prev_batch = unsafe { HiddenStates::uninit(ctx, hidden_size, m * hc_mult)? };
        for (r, h) in h_prevs.iter().enumerate() {
            let mut dst = h_prev_batch
                .data
                .slice_mut(r * stream_dim..(r + 1) * stream_dim);
            ctx.stream
                .memcpy_dtod(&h.data, &mut dst)
                .map_err(|e| anyhow!("DSv4 MTP h_prev D2D gather failed: {e}"))?;
        }
        // SAFETY: uninit device scratch; fully written before first read.
        let mut h_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, m * hc_mult)? };
        crate::profile::profile_op(ctx, "mtp_hnorm", None, m, || {
            crate::ops::rms_norm_batch(ctx, &h_prev_batch, &mtp.hnorm, eps, &mut h_normed)
        })?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut h_proj = unsafe { HiddenStates::uninit(ctx, hidden_size, m * hc_mult)? };
        crate::profile::profile_op(ctx, "mtp_h_proj", None, m, || {
            crate::attention::dsv4_linear(ctx, &mtp.h_proj, &h_normed, &mut h_proj)
        })?;

        // SAFETY: uninit device scratch; fully written before first read.
        let mut stream = unsafe { HiddenStates::uninit(ctx, stream_dim, m)? };
        {
            let (e_ptr, _ge) = e_proj.data.device_ptr(&ctx.stream);
            let (h_ptr, _gh) = h_proj.data.device_ptr(&ctx.stream);
            let (out_ptr, _go) = stream.data.device_ptr_mut(&ctx.stream);
            let row_h = (hidden_size * 2) as u64;
            let row_s = (stream_dim * 2) as u64;
            for r in 0..m as u64 {
                // SAFETY: per-row slices of buffers sized above.
                unsafe {
                    ffi::dsv4_mtp_add_eproj_hproj_cuda(
                        (e_ptr + r * row_h) as *const ffi::Half,
                        (h_ptr + r * row_s) as *const ffi::Half,
                        (out_ptr + r * row_s) as *mut ffi::Half,
                        hidden_size as i32,
                        hc_mult as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }
        keepalive.keep_hidden(&e_proj);
        keepalive.keep_hidden(&h_proj);
        keepalive.keep_hidden(&stream);

        // ONE MTP transformer layer over the batch; attention per row at the draft
        // position. Top-k widens this draft readout only.
        let layer = &mtp.layer;
        let target_layer_idx = self.mtp_frozen_target_layer_idx(mtp)?;
        for &sid in slot_ids {
            ensure!(
                target_layer_idx < slots[sid].attention.len(),
                "DSv4 MTP frozen-KV target layer {target_layer_idx} outside slot {sid} attention len {}",
                slots[sid].attention.len()
            );
        }
        let local_width = layer.attention.wq_b.rows;
        ensure!(
            local_width.is_multiple_of(self.config.head_dim),
            "DSv4 MTP attention local width {local_width} is not a multiple of head_dim {}",
            self.config.head_dim
        );

        let attn_mhc =
            crate::profile::profile_op(ctx, "attn_hc_params", Some(target_layer_idx), m, || {
                crate::hc::gen_mhc_params(ctx, &self.config, &layer.hc_attn, &stream)
            })?;
        // SAFETY: scratch fully written before read.
        let mut attn_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::profile::profile_op(ctx, "attn_hc_pre_norm", Some(target_layer_idx), m, || {
            crate::hc::mhc_pre_rms_norm(
                ctx,
                &stream,
                &attn_mhc.pre,
                &layer.attn_norm,
                eps,
                hidden_size,
                hc_mult,
                &mut attn_normed,
            )
        })?;
        keepalive.keep_hidden(&attn_normed);
        // SAFETY: scratch fully written before read.
        let mut attn_out = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::profile::profile_op(ctx, "attention", Some(target_layer_idx), m, || {
            // SAFETY: scratch fully written before read.
            let mut normed_row = unsafe { HiddenStates::uninit(ctx, hidden_size, 1)? };
            // SAFETY: scratch fully written before read.
            let mut attn_row = unsafe { HiddenStates::uninit(ctx, hidden_size, 1)? };
            keepalive.keep_hidden(&normed_row);
            keepalive.keep_hidden(&attn_row);
            let (layer_pool, mut dsa_shared, mut flashmla_scratch, mut prefill_shared, _fp32) =
                kv_adapter.layer_and_dsa_shared_mut(target_layer_idx)?;
            for r in 0..rows.len() {
                let src = attn_normed
                    .data
                    .slice(r * hidden_size..(r + 1) * hidden_size);
                ctx.stream
                    .memcpy_dtod(&src, &mut normed_row.data)
                    .map_err(|e| anyhow!("DSv4 MTP attn copy-in failed: {e}"))?;
                let pos_dev = ctx
                    .stream
                    .clone_htod(&[positions[r] as i32])
                    .map_err(|e| anyhow!("DSv4 MTP start_pos H2D failed: {e}"))?;
                crate::attention::mla_attention(
                    ctx,
                    &self.config,
                    &layer.attention,
                    layer.mode,
                    layer.compress_ratio,
                    target_layer_idx,
                    &normed_row,
                    &mut slots[slot_ids[r]].attention[target_layer_idx],
                    layer_pool,
                    dsa_shared.as_deref_mut(),
                    flashmla_scratch.as_deref_mut(),
                    prefill_shared.as_deref_mut(),
                    None,
                    positions[r] as usize,
                    Some(&pos_dev),
                    None,
                    &self.tp,
                    &mut attn_row,
                    &mut keepalive,
                )?;
                let mut dst = attn_out
                    .data
                    .slice_mut(r * hidden_size..(r + 1) * hidden_size);
                ctx.stream
                    .memcpy_dtod(&attn_row.data, &mut dst)
                    .map_err(|e| anyhow!("DSv4 MTP attn copy-out failed: {e}"))?;
            }
            Ok(())
        })?;
        crate::profile::profile_op(ctx, "attn_allreduce", Some(target_layer_idx), m, || {
            self.tp.all_reduce_sum(ctx, &mut attn_out)
        })?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut attn_stream = unsafe { HiddenStates::uninit(ctx, stream_dim, m)? };
        crate::profile::profile_op(ctx, "attn_hc_post", Some(target_layer_idx), m, || {
            crate::hc::hc_post(
                ctx,
                &attn_out,
                &stream,
                &attn_mhc.post,
                &attn_mhc.comb,
                hidden_size,
                hc_mult,
                &mut attn_stream,
            )
        })?;
        keepalive.keep_hidden(&attn_out);
        keepalive.keep_hidden(&attn_stream);

        let ffn_mhc =
            crate::profile::profile_op(ctx, "ffn_hc_params", Some(target_layer_idx), m, || {
                crate::hc::gen_mhc_params(ctx, &self.config, &layer.hc_ffn, &attn_stream)
            })?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut ffn_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::profile::profile_op(ctx, "ffn_hc_pre_norm", Some(target_layer_idx), m, || {
            crate::hc::mhc_pre_rms_norm(
                ctx,
                &attn_stream,
                &ffn_mhc.pre,
                &layer.ffn_norm,
                eps,
                hidden_size,
                hc_mult,
                &mut ffn_normed,
            )
        })?;
        keepalive.keep_hidden(&ffn_normed);
        let level_tokens: Vec<u32> = rows.iter().map(|r| r.token).collect();
        // SAFETY: uninit device scratch; fully written before first read.
        let mut moe_out = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        let needs_moe_allreduce =
            crate::profile::profile_op(ctx, "moe_route", Some(target_layer_idx), m, || {
                crate::moe::dsv4_moe_forward(
                    self,
                    layer.moe.as_ref().expect("DSv4 layer.moe"),
                    &level_tokens,
                    &ffn_normed,
                    &mut moe_out,
                    &mut keepalive,
                    None,
                    mega_epoch,
                )
            })?;
        if needs_moe_allreduce {
            crate::profile::profile_op(ctx, "moe_allreduce", Some(target_layer_idx), m, || {
                self.tp.all_reduce_sum(ctx, &mut moe_out)
            })?;
        }
        // SAFETY: uninit device scratch; fully written before first read.
        let mut shared = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::profile::profile_op(ctx, "shared_expert", Some(target_layer_idx), m, || {
            crate::moe::dsv4_shared_expert_forward(
                ctx,
                &ctx.stream,
                layer.moe.as_ref().expect("DSv4 layer.moe"),
                &ffn_normed,
                &mut shared,
                self.config.swiglu_limit,
                &mut keepalive,
            )
        })?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut moe_with_shared = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::profile::profile_op(ctx, "shared_add", Some(target_layer_idx), m, || {
            crate::ops::add_batch(ctx, &moe_out, &shared, &mut moe_with_shared)
        })?;
        // SAFETY: uninit device scratch; fully written before first read.
        let mut ffn_stream = unsafe { HiddenStates::uninit(ctx, stream_dim, m)? };
        crate::profile::profile_op(ctx, "ffn_hc_post", Some(target_layer_idx), m, || {
            crate::hc::hc_post(
                ctx,
                &moe_with_shared,
                &attn_stream,
                &ffn_mhc.post,
                &ffn_mhc.comb,
                hidden_size,
                hc_mult,
                &mut ffn_stream,
            )
        })?;
        keepalive.keep_hidden(&moe_out);
        keepalive.keep_hidden(&shared);
        keepalive.keep_hidden(&moe_with_shared);
        keepalive.keep_hidden(&ffn_stream);

        // SAFETY: uninit device scratch; fully written before first read.
        let mut head_normed = unsafe { HiddenStates::uninit(ctx, hidden_size, m)? };
        crate::profile::profile_op(ctx, "head_hc", Some(target_layer_idx), m, || {
            let mut last_hidden = DeviceVec::zeros(ctx, hidden_size)?;
            let mut last_normed = DeviceVec::zeros(ctx, hidden_size)?;
            for r in 0..m {
                crate::hc::head_hidden_from_stream(
                    ctx,
                    &self.config,
                    &mtp.head_hc,
                    &ffn_stream,
                    r,
                    &mut last_hidden,
                )?;
                crate::ops::rms_norm_vec(ctx, &last_hidden, &mtp.norm, eps, &mut last_normed)?;
                let mut dst = head_normed
                    .data
                    .slice_mut(r * hidden_size..(r + 1) * hidden_size);
                ctx.stream
                    .memcpy_dtod(&last_normed.data, &mut dst)
                    .map_err(|e| anyhow!("DSv4 MTP head row copy failed: {e}"))?;
            }
            Ok(())
        })?;
        keepalive.keep_hidden(&head_normed);
        // SAFETY: uninit device scratch; fully written before first read.
        let mut logits = unsafe { HiddenStates::uninit(ctx, self.lm_head.rows, m)? };
        crate::profile::profile_op(ctx, "lm_head_project", Some(target_layer_idx), m, || {
            self.lm_head_project_batch(&head_normed, &mut logits)
        })?;
        keepalive.keep_hidden(&logits);
        let candidates = self.mtp_topk_device(&mut logits, top_k.max(1))?;
        std::hint::black_box(keepalive.len());
        drop(keepalive);

        // Split the level stream into per-row owned vecs (next draft rows
        // branch from their own parent row only).
        let mut out = Vec::with_capacity(m);
        for (r, cand) in candidates.into_iter().enumerate() {
            let mut row_stream = DeviceVec::zeros(ctx, stream_dim)?;
            let src = ffn_stream.data.slice(r * stream_dim..(r + 1) * stream_dim);
            ctx.stream
                .memcpy_dtod(&src, &mut row_stream.data)
                .map_err(|e| anyhow!("DSv4 MTP stream row split failed: {e}"))?;
            out.push((cand, row_stream));
        }
        Ok(out)
    }

    /// Batched device argmax over `[m, vocab]` verifier logits — one launch,
    /// one D2H of m target top-1 ids.
    pub(super) fn mtp_argmax_batch(&self, logits: &HiddenStates) -> Result<Vec<u32>> {
        let ctx = &self.ctx;
        let m = logits.seq_len;
        let vocab = logits.hidden_dim;
        let mut ids_dev = ctx
            .stream
            .alloc_zeros::<i32>(m)
            .map_err(|e| anyhow!("DSv4 MTP argmax ids alloc failed: {e}"))?;
        {
            let (logits_ptr, _lg) = logits.data.device_ptr(&ctx.stream);
            let (ids_ptr, _ig) = ids_dev.device_ptr_mut(&ctx.stream);
            // SAFETY: logits [m, vocab] and ids [m] sized above.
            unsafe {
                ffi::argmax_batch_cuda(
                    logits_ptr as *const ffi::Half,
                    ids_ptr as *mut i32,
                    m as i32,
                    vocab as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        let ids: Vec<i32> = ctx
            .stream
            .clone_dtoh(&ids_dev)
            .map_err(|e| anyhow!("DSv4 MTP argmax D2H failed: {e}"))?;
        ids.into_iter()
            .map(|id| {
                ensure!(
                    (0..vocab as i32).contains(&id),
                    "DSv4 MTP argmax id {id} out of vocab {vocab}"
                );
                Ok(id as u32)
            })
            .collect()
    }

    /// Batched device top-k over `[m, vocab]` logits, highest-first per row.
    /// Masks each selected id to `-inf` in the caller's logits scratch.
    pub(super) fn mtp_topk_device(
        &self,
        logits: &mut HiddenStates,
        k: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let ctx = &self.ctx;
        let m = logits.seq_len;
        let vocab = logits.hidden_dim;
        ensure!(
            k >= 1 && k < vocab,
            "DSv4 MTP top-k {k} out of range for vocab {vocab}"
        );
        let mut ids_dev = ctx
            .stream
            .alloc_zeros::<i32>(m)
            .map_err(|e| anyhow!("DSv4 MTP top-k ids alloc failed: {e}"))?;
        let mut out = vec![Vec::with_capacity(k); m];
        for round in 0..k {
            {
                let (logits_ptr, _lg) = logits.data.device_ptr(&ctx.stream);
                let (ids_ptr, _ig) = ids_dev.device_ptr_mut(&ctx.stream);
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    ffi::argmax_batch_cuda(
                        logits_ptr as *const ffi::Half,
                        ids_ptr as *mut i32,
                        m as i32,
                        vocab as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
            let ids: Vec<i32> = ctx
                .stream
                .clone_dtoh(&ids_dev)
                .map_err(|e| anyhow!("DSv4 MTP top-k D2H failed: {e}"))?;
            for (r, &id) in ids.iter().enumerate() {
                ensure!(
                    (0..vocab as i32).contains(&id),
                    "DSv4 MTP top-k id {id} out of vocab {vocab}"
                );
                out[r].push(id as u32);
            }
            if round + 1 < k {
                for (r, &id) in ids.iter().enumerate() {
                    let offset = r * vocab + id as usize;
                    let mut dst = logits.data.slice_mut(offset..offset + 1);
                    ctx.stream
                        .memcpy_htod(&[half::bf16::NEG_INFINITY], &mut dst)
                        .map_err(|e| anyhow!("DSv4 MTP top-k mask failed: {e}"))?;
                }
            }
        }
        Ok(out)
    }

    pub(super) fn mtp_frozen_target_layer_idx(&self, mtp: &Dsv4MtpLayer) -> Result<usize> {
        // DSv4's shipped MTP layer is forced to compress_ratio=0 (SW-only), so the
        // draft reads target physical layer 0's committed SW ring instead of a
        // fresh one-token attention state.
        let idx = 0;
        let layer = self.layers.get(idx).ok_or_else(|| {
            anyhow!(
                "DSv4 MTP frozen-KV target layer {idx} outside base layer count {}",
                self.layers.len()
            )
        })?;
        ensure!(
            mtp.layer.mode == DeepSeekV4AttentionMode::SlidingWindow,
            "DSv4 MTP frozen-KV path expects the MTP layer to be SlidingWindow, got {:?}",
            mtp.layer.mode
        );
        ensure!(
            layer.mode == DeepSeekV4AttentionMode::SlidingWindow,
            "DSv4 MTP frozen-KV target layer {idx} must be SlidingWindow for the current MTP layer, got {:?}",
            layer.mode
        );
        Ok(idx)
    }
}
