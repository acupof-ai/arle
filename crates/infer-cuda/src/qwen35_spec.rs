use super::*;

impl Qwen35Model {
    pub(crate) fn mtp_forward_level(
        &self,
        spec: &mut Qwen35SpecSlotState,
        ws: &mut Qwen35Workspace,
        token: u32,
        h_prev: &DeviceVec,
        level: usize,
        params: &SamplingParams,
        start_pos: usize,
    ) -> Result<(u32, DeviceVec)> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow!("mtp_forward_level called without a loaded MTP head"))?;
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let hidden = c.hidden_size;
        ensure!(
            h_prev.len == hidden,
            "mtp_forward_level h_prev len {} != hidden {hidden}",
            h_prev.len
        );

        let token_ids = upload_i32(&self.ctx, &[token as i32])?;
        let mut emb = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut emb)?;

        // Matches the Metal `qwen3_5_mtp` loader order.
        let mut concat = HiddenStates::zeros(&self.ctx, 2 * hidden, 1)?;
        let mut emb_n = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        rms_norm_offset(&self.ctx, &emb, &mtp.pre_fc_norm_embedding, eps, &mut emb_n)?;
        {
            let mut dst = concat.data.slice_mut(0..hidden);
            self.ctx
                .stream
                .memcpy_dtod(&emb_n.data, &mut dst)
                .map_err(|e| anyhow!("mtp concat emb half failed: {e}"))?;
        }
        let mut h_n = DeviceVec::zeros(&self.ctx, hidden)?;
        rms_norm_offset_vec(&self.ctx, h_prev, &mtp.pre_fc_norm_hidden, eps, &mut h_n)?;
        {
            let mut dst = concat.data.slice_mut(hidden..2 * hidden);
            self.ctx
                .stream
                .memcpy_dtod(&h_n.data, &mut dst)
                .map_err(|e| anyhow!("mtp concat hidden half failed: {e}"))?;
        }
        let mut h_fc = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        gemm_batch(&self.ctx, &mtp.fc, &concat, &mut h_fc)?;

        // Mirrors the trunk layer body.
        let layer = &mtp.layer;
        let Qwen35Attn::Full(full_attn) = &layer.attn else {
            unreachable!("MTP head layer is always full attention");
        };

        let mut normed = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        rms_norm_offset(&self.ctx, &h_fc, &layer.input_layernorm, eps, &mut normed)?;

        // level == head-KV row == RoPE position within the draft block.
        let start_pos_dev = upload_i32(&self.ctx, &[level as i32])?;
        let mut attn_out = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        self.full_attention_into(
            full_attn,
            &normed,
            &mut spec.head_k,
            &mut spec.head_v,
            0, // profiling label
            level,
            &start_pos_dev,
            &mut ws.full,
            &mut attn_out,
        )?;

        let mut hidden_mid = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        add_batch(&self.ctx, &h_fc, &attn_out, &mut hidden_mid)?;
        rms_norm_offset(
            &self.ctx,
            &hidden_mid,
            &layer.post_attention_layernorm,
            eps,
            &mut normed,
        )?;
        let mut mlp_out = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        if let Some(moe) = &layer.moe {
            let moe_cfg = self
                .moe_config
                .as_ref()
                .ok_or_else(|| anyhow!("MTP MoE layer but no moe_config"))?;
            crate::moe::moe_forward_into(
                &self.ctx,
                moe,
                &normed,
                moe_cfg,
                &self.expert_split,
                &mut ws.moe,
                &mut mlp_out,
            )?;
        } else {
            let mlp = layer
                .mlp
                .as_ref()
                .ok_or_else(|| anyhow!("MTP head layer missing MLP"))?;
            self.dense_mlp(mlp, &normed, &mut ws.dense, &mut mlp_out, None)?;
        }
        let mut h_layer = HiddenStates::zeros(&self.ctx, hidden, 1)?;
        add_batch(&self.ctx, &hidden_mid, &mlp_out, &mut h_layer)?;

        // Same weights as the trunk lm_head.
        rms_norm_offset(&self.ctx, &h_layer, &mtp.norm, eps, &mut normed)?;
        let vocab = self.output_projection().rows;
        let mut logits = HiddenStates::zeros(&self.ctx, vocab, 1)?;
        gemm_batch(&self.ctx, self.output_projection(), &normed, &mut logits)?;
        let logits_vec = DeviceVec {
            data: logits.data,
            len: vocab,
            label: "qwen35_mtp_draft_logits",
        };
        let next = if params.is_greedy() {
            argmax_into(&self.ctx, &logits_vec, &mut spec.argmax_scratch)?
        } else {
            // Filter + q-row retain + multinomial draw in one device call; the
            // uniform is the salted (seed, position) stream plain decode would
            // consume at this position (mirrors the DSpark draft draw).
            let u =
                dspark::unit_uniform(params.seed, dspark::SALT_DRAW, (start_pos + level) as u64);
            let cap = self.spec_draft_tokens.max(1);
            let q_all = spec.q_probs.get(&self.ctx, cap * vocab)?;
            let tok_out = spec.sample_tok.get(&self.ctx, 1)?;
            {
                let (l_ptr, _gl) = logits_vec.data.device_ptr(&self.ctx.stream);
                let (q_ptr, _gq) = q_all.device_ptr_mut(&self.ctx.stream);
                let (t_ptr, _gt) = tok_out.device_ptr_mut(&self.ctx.stream);
                // SAFETY: logits_vec holds `vocab` bf16; q row `level` < cap
                // (spec_step's depth guard bounds every level).
                unsafe {
                    ffi::dspark_draft_sample_cuda(
                        l_ptr as *const ffi::Half,
                        (q_ptr + (level * vocab * 4) as u64) as *mut f32,
                        t_ptr as *mut i32,
                        vocab as i32,
                        1.0 / params.temperature,
                        params.top_k,
                        params.top_p,
                        params.min_p,
                        u,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
            self.ctx.sync()?;
            self.ctx
                .stream
                .clone_dtoh(tok_out)
                .map_err(|e| anyhow!("D2H mtp draft token failed: {e}"))?[0] as u32
        };

        // The head's own output hidden seeds the next level (autoregressive chain,
        // per `dflash.rs` `step_hidden = flat`).
        let mut h_out = DeviceVec::zeros(&self.ctx, hidden)?;
        copy_row_to_vec(&self.ctx, &h_layer, 0, &mut h_out)?;
        Ok((next, h_out))
    }

    /// NextN-MTP speculative decode step (single chain).
    ///
    /// Correctness: greedy rows accept the longest prefix whose draft equals the
    /// trunk's argmax — token-exact to no-spec greedy. Sampled rows accept by
    /// chain rejection sampling; committed tokens distribute exactly as filtered
    /// target sampling.
    ///
    /// State contract:
    /// - `pending`: last emitted token, written into KV at `start_pos` by verify.
    /// - `hidden`: trunk hidden that produced `pending` — the head's level-0 seed.
    /// - entry invariant: `slot.seq_len() == start_pos`.
    ///
    /// Returns `(emitted, bonus, next_hidden)`: emitted = `[d1..dk, bonus]`,
    /// `k` = accepted draft count. On partial accept the trunk linear state
    /// rolls back via the pre-verify snapshot + a `(k+1)`-token linear replay;
    /// the full-attn KV self-heals under the seq_len rewind.
    pub(crate) fn spec_step(
        &self,
        slot: &mut Qwen35SlotState,
        spec: &mut Qwen35SpecSlotState,
        ws: &mut Qwen35Workspace,
        pending: u32,
        hidden: &DeviceVec,
        start_pos: usize,
        depth: usize,
        params: &SamplingParams,
        recall: Option<&mut Qwen35RecallForward>,
    ) -> Result<(Vec<CommittedToken>, u32, DeviceVec)> {
        ensure!(depth >= 1, "spec_step requires depth >= 1, got {depth}");
        // The MTP head KV (spec.head_k/head_v) was sized (spec_draft_tokens+1)
        // rows by new_spec_slot_state; a depth beyond that would overflow the
        // head KV in mtp_forward_level (row = level, 0..depth-1). Guard it.
        ensure!(
            depth <= self.spec_draft_tokens.max(1),
            "spec_step depth {depth} exceeds the MTP head KV capacity (model built for {} draft tokens)",
            self.spec_draft_tokens.max(1)
        );
        ensure!(
            slot.seq_len() == start_pos,
            "spec_step entry seq_len {} != start_pos {start_pos}",
            slot.seq_len()
        );
        let vocab = self.output_projection().rows;
        let hidden_size = self.config.hidden_size;
        let mut pt = mtp_phase_start(&self.ctx);

        let mut h_prev = DeviceVec::zeros(&self.ctx, hidden_size)?;
        self.ctx
            .stream
            .memcpy_dtod(&hidden.data, &mut h_prev.data)
            .map_err(|e| anyhow!("spec seed hidden copy failed: {e}"))?;
        let mut chain: Vec<u32> = Vec::with_capacity(depth + 1);
        chain.push(pending);
        for level in 0..depth {
            let last_tok = *chain.last().unwrap();
            let (tok, h_out) =
                self.mtp_forward_level(spec, ws, last_tok, &h_prev, level, params, start_pos)?;
            chain.push(tok);
            h_prev = h_out;
        }
        let draft_ms = mtp_phase_lap(&self.ctx, &mut pt);

        spec.snapshot_trunk(&self.ctx, slot)?;
        let snap_ms = mtp_phase_lap(&self.ctx, &mut pt);

        // Captures each linear layer's gated-delta inputs for all depth+1 rows
        // so the partial-accept replay can re-advance only the linear state.
        let (logits, dims, hiddens) = self.forward_tokens_verify(
            slot,
            ws,
            &chain,
            start_pos,
            Some(&mut spec.capture),
            recall,
        )?;
        ensure!(
            dims == [depth + 1, vocab],
            "spec verify dims {dims:?} != [{}, {vocab}]",
            depth + 1
        );
        let verify_ms = mtp_phase_lap(&self.ctx, &mut pt);

        let (emitted, bonus, k) = if params.is_greedy() {
            let mut k = 0usize;
            let bonus;
            loop {
                let am = argmax_row_into(&self.ctx, &logits, k, vocab, &mut spec.argmax_scratch)?;
                if k < depth && am == chain[k + 1] {
                    k += 1;
                } else {
                    bonus = am;
                    break;
                }
            }
            // Greedy: delta policy, no behavior logprob.
            let mut emitted: Vec<CommittedToken> =
                chain[1..=k].iter().map(|&t| (t, None)).collect();
            emitted.push((bonus, None));
            if k < depth {
                spec.restore_trunk(&self.ctx, slot)?;
                // LINEAR-ONLY replay: restore_trunk just rewound the 48 gated-delta
                // recurrent + conv rings to S_{start_pos}; re-advance ONLY them over
                // the accepted prefix `[pending, d1..dk]` (k+1 rows) from the verify
                // capture, skipping the full-attn blocks, MLP/MoE, final norm, and
                // lm_head — the dominant avoidable cost of the old full replay. The
                // 16 full-attn KV caches self-heal via position-indexing under the
                // explicit seq_len rewind below; MLP/MoE/lm_head leave no state.
                self.replay_linear_only(slot, ws, &spec.capture, k)?;
                slot.set_seq_len(start_pos + k + 1);
            }
            (emitted, bonus, k)
        } else {
            self.mtp_accept_commit_sampled(slot, spec, ws, &chain, &logits, start_pos, params)?
        };
        let accept_ms = mtp_phase_lap(&self.ctx, &mut pt);

        let mut next_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        {
            let src = hiddens.data.slice(k * hidden_size..(k + 1) * hidden_size);
            self.ctx
                .stream
                .memcpy_dtod(&src, &mut next_hidden.data)
                .map_err(|e| anyhow!("spec next-hidden copy failed: {e}"))?;
        }

        if pt.is_some() {
            eprintln!(
                "[mtp-phase] depth={depth} accept={k} draft={draft_ms:.2} snap={snap_ms:.2} verify={verify_ms:.2} accept_commit={accept_ms:.2} ms"
            );
        }
        Ok((emitted, bonus, next_hidden))
    }

    /// Chain rejection sampling for the sampled spec path.
    ///
    /// Exactness invariant: p and q pass the same engine-sampler filter, so
    /// committed tokens distribute exactly as filtered target sampling.
    pub(crate) fn mtp_accept_commit_sampled(
        &self,
        slot: &mut Qwen35SlotState,
        spec: &mut Qwen35SpecSlotState,
        ws: &mut Qwen35Workspace,
        chain: &[u32],
        logits: &DeviceVec,
        start_pos: usize,
        params: &SamplingParams,
    ) -> Result<(Vec<CommittedToken>, u32, usize)> {
        let ctx = &self.ctx;
        let depth = chain.len() - 1;
        let cap = self.spec_draft_tokens.max(1);
        ensure!(
            depth <= cap,
            "mtp sampled verify: depth {depth} > head cap {cap}"
        );
        let vocab = self.output_projection().rows;
        // Position-salted: identical to the host path's per-step draws.
        let pos = |j: usize| (start_pos + j + 1) as u64;
        let u_acc: Vec<f32> = (0..depth)
            .map(|j| dspark::unit_uniform(params.seed, dspark::SALT_ACCEPT, pos(j)))
            .collect();
        let u_res: Vec<f32> = (0..=depth)
            .map(|j| dspark::unit_uniform(params.seed, dspark::SALT_RESIDUAL, pos(j)))
            .collect();
        let draft: Vec<i32> = chain[1..].iter().map(|&t| t as i32).collect();

        let p_all = spec.p_probs.get(ctx, (cap + 1) * vocab)?;
        let q_all = spec.q_probs.get(ctx, cap * vocab)?;
        let draft_dev = spec.chain_draft.get(ctx, cap)?;
        let ua_dev = spec.u_accept.get(ctx, cap)?;
        let ur_dev = spec.u_residual.get(ctx, cap + 1)?;
        let out_dev = spec.accept_out.get(ctx, 2)?;
        ctx.stream
            .memcpy_htod(&draft, &mut draft_dev.slice_mut(0..depth))
            .and_then(|()| {
                ctx.stream
                    .memcpy_htod(&u_acc, &mut ua_dev.slice_mut(0..depth))
            })
            .and_then(|()| {
                ctx.stream
                    .memcpy_htod(&u_res, &mut ur_dev.slice_mut(0..=depth))
            })
            .map_err(|e| anyhow!("H2D mtp chain inputs failed: {e}"))?;
        {
            let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
            let (p_ptr, _gp) = p_all.device_ptr_mut(&ctx.stream);
            let (q_ptr, _gq) = q_all.device_ptr(&ctx.stream);
            let (d_ptr, _gd) = draft_dev.device_ptr(&ctx.stream);
            let (ua_ptr, _gua) = ua_dev.device_ptr(&ctx.stream);
            let (ur_ptr, _gur) = ur_dev.device_ptr(&ctx.stream);
            let (o_ptr, _go) = out_dev.device_ptr_mut(&ctx.stream);
            // SAFETY: logits holds chain.len()*vocab bf16; p/q scratches hold
            // (cap+1)/cap vocab-rows and depth <= cap (ensured above); the q
            // rows were written by this step's draft; draft/u prefixes uploaded
            // just above.
            unsafe {
                ffi::dspark_filter_probs_cuda(
                    l_ptr as *const ffi::Half,
                    p_ptr as *mut f32,
                    chain.len() as i32,
                    vocab as i32,
                    1.0 / params.temperature,
                    params.top_k,
                    params.top_p,
                    params.min_p,
                    ctx.stream.cu_stream(),
                )
                .result()?;
                ffi::dspark_chain_accept_cuda(
                    q_ptr as *const f32,
                    p_ptr as *const f32,
                    d_ptr as *const i32,
                    ua_ptr as *const f32,
                    ur_ptr as *const f32,
                    o_ptr as *mut i32,
                    depth as i32,
                    vocab as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        ctx.sync()?;
        let out = ctx
            .stream
            .clone_dtoh(out_dev)
            .map_err(|e| anyhow!("D2H mtp chain verdict failed: {e}"))?;
        let (k, bonus) = (out[0] as usize, out[1] as u32);
        ensure!(
            k <= depth,
            "mtp chain kernel returned k {k} > depth {depth}"
        );
        let mut tokens: Vec<u32> = chain[1..=k].to_vec();
        tokens.push(bonus);
        // Committed token j distributes as the filtered target dist p_j;
        // p rows are final (verdict D2H synced above).
        let logprobs = chain_commit_logprobs(ctx, p_all, vocab, &tokens)?;
        let emitted = tokens
            .into_iter()
            .zip(logprobs)
            .map(|(t, lp)| (t, Some(lp)))
            .collect();
        if k < depth {
            spec.restore_trunk(ctx, slot)?;
            self.replay_linear_only(slot, ws, &spec.capture, k)?;
            slot.set_seq_len(start_pos + k + 1);
        }
        Ok((emitted, bonus, k))
    }
}

/// Opt-in phase timing (set `ARLE_MTP_PHASE`); the per-phase sync needed for
/// accurate GPU timing is off by default so the hot path pays nothing.
pub(crate) fn mtp_phase_start(ctx: &DeviceContext) -> Option<std::time::Instant> {
    phase_start(ctx, "ARLE_MTP_PHASE")
}

pub(crate) fn dspark_phase_start(ctx: &DeviceContext) -> Option<std::time::Instant> {
    phase_start(ctx, "ARLE_DSPARK_PHASE")
}

pub(crate) fn phase_start(ctx: &DeviceContext, var: &str) -> Option<std::time::Instant> {
    if std::env::var(var).is_ok() {
        let _ = ctx.sync();
        Some(std::time::Instant::now())
    } else {
        None
    }
}

pub(crate) fn mtp_phase_lap(ctx: &DeviceContext, t: &mut Option<std::time::Instant>) -> f64 {
    match t {
        Some(prev) => {
            let _ = ctx.sync();
            let now = std::time::Instant::now();
            let ms = now.duration_since(*prev).as_secs_f64() * 1000.0;
            *t = Some(now);
            ms
        }
        None => 0.0,
    }
}

/// log p_filtered of each committed chain token from the materialized `p` rows.
/// Caller must have synced the accept verdict D2H so the rows are final.
/// The floor clamp guards f32 underflow at `ln`.
pub(crate) fn chain_commit_logprobs(
    ctx: &DeviceContext,
    p_all: &CudaSlice<f32>,
    vocab: usize,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    tokens
        .iter()
        .enumerate()
        .map(|(j, &tok)| {
            let off = j * vocab + tok as usize;
            let p = ctx
                .stream
                .clone_dtoh(&p_all.slice(off..off + 1))
                .map_err(|e| anyhow!("D2H chain commit prob failed: {e}"))?[0];
            Ok(p.max(f32::MIN_POSITIVE).ln())
        })
        .collect()
}
