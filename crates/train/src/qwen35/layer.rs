use autograd::ops::elementwise::add_consuming_rhs;

use super::*;

impl Qwen35Layer {
    /// This layer's trainable param ids — fed to `checkpoint_sequential`'s
    /// `layer_params` so each group's saved inputs carry every layer's params
    /// (that's how `requires_grad` is true and param grads come back).
    pub(super) fn checkpoint_param_ids(
        &self,
        skip_experts: bool,
        store: &TensorStore,
    ) -> Vec<TensorId> {
        let mut params = Vec::new();
        params.push(self.input_layernorm);
        params.push(self.post_attention_layernorm);
        collect_mlp_ids(&self.mlp, skip_experts, &mut params);
        match &self.self_attn {
            Qwen35Attention::Full(attn) => {
                collect_linear_ids(&attn.q_proj, &mut params);
                collect_linear_ids(&attn.k_proj, &mut params);
                collect_linear_ids(&attn.v_proj, &mut params);
                collect_linear_ids(&attn.o_proj, &mut params);
                params.push(attn.q_norm);
                params.push(attn.k_norm);
            }
            Qwen35Attention::Linear(attn) => {
                collect_linear_ids(&attn.in_proj_qkv, &mut params);
                collect_linear_ids(&attn.in_proj_z, &mut params);
                collect_linear_ids(&attn.in_proj_b, &mut params);
                collect_linear_ids(&attn.in_proj_a, &mut params);
                collect_linear_ids(&attn.out_proj, &mut params);
                params.push(attn.conv1d_weight);
                params.push(attn.dt_bias);
                params.push(attn.a_log);
                params.push(attn.norm);
            }
        }
        params.sort_unstable();
        params.dedup();
        params.retain(|&id| store.get(id).is_some_and(|tensor| tensor.requires_grad));
        params
    }

    pub(super) fn forward(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cp: crate::context_parallel::CpContext,
        cos: TensorId,
        sin: TensorId,
        cp_positions: Option<&[usize]>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];
        checkpoint_replay_mem_stage(tape, store, "layer_enter");

        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        checkpoint_replay_mem_stage(tape, store, "post_input_norm");
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => self.forward_full_attention(
                h,
                attn,
                cfg,
                tp,
                cp,
                cos,
                sin,
                batch,
                seq_len,
                cp_positions,
                store,
                tape,
            )?,
            Qwen35Attention::Linear(attn) => {
                self.forward_linear_attention(h, attn, cfg, cp, batch, seq_len, store, tape)?
            }
        };
        checkpoint_replay_mem_stage(tape, store, "post_attention");
        let x = add_consuming_rhs(x, attn_out, store, tape)?;
        checkpoint_replay_mem_stage(tape, store, "post_attention_residual");

        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        checkpoint_replay_mem_stage(tape, store, "post_mlp_norm");
        // MLP is position-wise — chunking is exact.
        let mlp_out = {
            let mut param_ids = Vec::new();
            collect_mlp_ids(&self.mlp, false, &mut param_ids);
            param_ids.retain(|&id| store.get(id).is_some_and(|t| t.requires_grad));
            let layer = self.clone();
            let cfg_owned = cfg.clone();
            autograd::ops::checkpoint_seq_chunked(
                h,
                param_ids,
                crate::runtime_flags::opd_seq_chunk(),
                store,
                tape,
                move |st, tp_tape, _start, inp| {
                    let shape = st
                        .get(inp[0])
                        .ok_or(AutogradError::InvalidTensorId(inp[0]))?
                        .shape
                        .clone();
                    layer
                        .forward_mlp(
                            inp[0],
                            &cfg_owned,
                            tp,
                            shape[0],
                            shape[1],
                            &mut MoeRouteMode::Free,
                            st,
                            tp_tape,
                        )
                        .map_err(qwen35_to_autograd)
                },
            )?
        };
        checkpoint_replay_mem_stage(tape, store, "post_mlp");
        let out = add_consuming_rhs(x, mlp_out, store, tape)?;
        checkpoint_replay_mem_stage(tape, store, "layer_exit");
        Ok(out)
    }

    /// OPD frozen-prompt-KV phase 1 (off-tape): capture this layer's prompt
    /// prefix K/V (full attention) or boundary state (linear attention) AND
    /// produce the layer's attention+MLP output so the next layer's capture
    /// sees the correct residual stream.
    ///
    /// `prefix_kv` = accumulated K/V from earlier prompt chunks (None for the
    /// first chunk). `chunk_start` = absolute row index of this chunk's first
    /// token (0 for the first chunk). The returned `LayerPrefix` holds the FULL
    /// K/V (prefix + this chunk) for the gen-segment pass. The caller passes a
    /// DISABLED tape.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_capture_prefix(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        chunk_start: usize,
        prefix_kv: Option<&PrefixKv>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(TensorId, LayerPrefix)> {
        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let (attn_out, prefix) = match &self.self_attn {
            Qwen35Attention::Full(attn) => {
                let chunk_kv = self.forward_full_attention_capture_prefix_kv(
                    h, attn, cfg, tp, cos, sin, batch, seq_len, store, tape,
                )?;
                let full_kv = if let Some(prefix) = prefix_kv {
                    PrefixKv {
                        k: cat_seq(prefix.k, chunk_kv.k, store, tape)?,
                        v: cat_seq(prefix.v, chunk_kv.v, store, tape)?,
                    }
                } else {
                    chunk_kv
                };
                // q_start = chunk_start so causal masking is by absolute position.
                let attn_out = self.forward_full_attention_with_kv(
                    h,
                    attn,
                    cfg,
                    tp,
                    cos,
                    sin,
                    &full_kv,
                    batch,
                    seq_len,
                    chunk_start,
                    store,
                    tape,
                )?;
                (attn_out, LayerPrefix::Full(full_kv))
            }
            Qwen35Attention::Linear(attn) => {
                let prefix = self.forward_linear_attention_capture_prefix_state(
                    h, attn, cfg, batch, seq_len, store, tape,
                )?;
                let attn_out = self.forward_linear_attention(
                    h,
                    attn,
                    cfg,
                    crate::context_parallel::CpContext::single(),
                    batch,
                    seq_len,
                    store,
                    tape,
                )?;
                (attn_out, LayerPrefix::Linear(prefix))
            }
        };
        let x = add_consuming_rhs(x, attn_out, store, tape)?;
        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let mlp_out = self.forward_mlp(
            h,
            cfg,
            tp,
            batch,
            seq_len,
            &mut MoeRouteMode::Free,
            store,
            tape,
        )?;
        let next = add_consuming_rhs(x, mlp_out, store, tape)?;
        Ok((next, prefix))
    }

    /// OPD frozen-prompt-KV phase 2 (per layer, TAPED): residual block over the
    /// gen rows only (`x` = `[batch, gen_len, hidden]`), where attention consumes
    /// this layer's captured prefix. RMSNorm + MLP are position-local, so the gen
    /// residual stream is exact given the seeded attention.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_gen_segment(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos_gen: TensorId,
        sin_gen: TensorId,
        layer_prefix: &LayerPrefix,
        batch: usize,
        gen_start: usize,
        gen_len: usize,
        cp: crate::context_parallel::CpContext,
        cp_positions: Option<&[usize]>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let attn_out = match (&self.self_attn, layer_prefix) {
            (Qwen35Attention::Full(attn), LayerPrefix::Full(prefix_kv)) => self
                .forward_full_attention_gen_segment(
                    h,
                    attn,
                    cfg,
                    tp,
                    cos_gen,
                    sin_gen,
                    prefix_kv,
                    batch,
                    gen_start,
                    gen_len,
                    cp,
                    cp_positions,
                    store,
                    tape,
                )?,
            (Qwen35Attention::Linear(attn), LayerPrefix::Linear(prefix_state)) => {
                if cp.is_enabled() {
                    return Err(Qwen35Error::InvalidConfig(
                        "frozen-prompt-KV + CP is not yet implemented for linear attention layers",
                    ));
                }
                self.forward_linear_attention_gen_segment(
                    h,
                    attn,
                    cfg,
                    prefix_state,
                    batch,
                    gen_len,
                    store,
                    tape,
                )?
            }
            _ => {
                return Err(Qwen35Error::InvalidConfig(
                    "frozen-prompt-KV layer prefix kind does not match the layer attention kind",
                ));
            }
        };
        let x = add_consuming_rhs(x, attn_out, store, tape)?;
        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let mlp_out = self.forward_mlp(
            h,
            cfg,
            tp,
            batch,
            gen_len,
            &mut MoeRouteMode::Free,
            store,
            tape,
        )?;
        Ok(add_consuming_rhs(x, mlp_out, store, tape)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_profiled(
        &self,
        layer_index: usize,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        trace: bool,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(TensorId, Qwen35LayerForwardProfile)> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];
        let mut profile = Qwen35LayerForwardProfile::default();

        let started = Instant::now();
        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        profile.input_rmsnorm += started.elapsed();
        trace_forward_component(trace, layer_index, "input_rmsnorm", profile.input_rmsnorm);

        let started = Instant::now();
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => {
                let mut attention_profile = Qwen35AttentionForwardProfile::default();
                let out = self.forward_full_attention_profiled(
                    layer_index,
                    h,
                    attn,
                    cfg,
                    tp,
                    cos,
                    sin,
                    batch,
                    seq_len,
                    trace,
                    store,
                    tape,
                    &mut attention_profile,
                )?;
                profile.attention_detail = attention_profile;
                out
            }
            Qwen35Attention::Linear(attn) => {
                let mut attention_profile = Qwen35AttentionForwardProfile::default();
                let out = self.forward_linear_attention_profiled(
                    layer_index,
                    h,
                    attn,
                    cfg,
                    batch,
                    seq_len,
                    trace,
                    store,
                    tape,
                    &mut attention_profile,
                )?;
                profile.attention_detail = attention_profile;
                out
            }
        };
        profile.attention += started.elapsed();
        trace_forward_component(trace, layer_index, "attention_total", profile.attention);

        let started = Instant::now();
        let x = add_consuming_rhs(x, attn_out, store, tape)?;
        profile.attention_residual += started.elapsed();
        trace_forward_component(
            trace,
            layer_index,
            "attention_residual",
            profile.attention_residual,
        );

        let started = Instant::now();
        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        profile.post_attention_rmsnorm += started.elapsed();
        trace_forward_component(
            trace,
            layer_index,
            "post_attention_rmsnorm",
            profile.post_attention_rmsnorm,
        );

        let started = Instant::now();
        let mlp_out = self.forward_mlp(
            h,
            cfg,
            tp,
            batch,
            seq_len,
            &mut MoeRouteMode::Free,
            store,
            tape,
        )?;
        profile.mlp += started.elapsed();
        trace_forward_component(trace, layer_index, "mlp", profile.mlp);

        let started = Instant::now();
        let out = add_consuming_rhs(x, mlp_out, store, tape)?;
        profile.mlp_residual += started.elapsed();
        trace_forward_component(trace, layer_index, "mlp_residual", profile.mlp_residual);

        Ok((out, profile))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_routes(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        mode: &mut MoeRouteMode<'_>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];

        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => self.forward_full_attention(
                h,
                attn,
                cfg,
                tp,
                crate::context_parallel::CpContext::single(),
                cos,
                sin,
                batch,
                seq_len,
                None,
                store,
                tape,
            )?,
            Qwen35Attention::Linear(attn) => self.forward_linear_attention(
                h,
                attn,
                cfg,
                crate::context_parallel::CpContext::single(),
                batch,
                seq_len,
                store,
                tape,
            )?,
        };
        let x = add_consuming_rhs(x, attn_out, store, tape)?;

        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let mlp_out = self.forward_mlp(h, cfg, tp, batch, seq_len, mode, store, tape)?;
        Ok(add_consuming_rhs(x, mlp_out, store, tape)?)
    }

    pub(super) fn forward_with_kv_cache(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        cos: TensorId,
        sin: TensorId,
        layer_cache: &mut Qwen35LayerKvCache,
        q_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];

        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => self.forward_full_attention_with_kv_cache(
                h,
                attn,
                cfg,
                cos,
                sin,
                batch,
                seq_len,
                layer_cache,
                q_start,
                store,
                tape,
            )?,
            Qwen35Attention::Linear(_) => {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache requires full-attention layers",
                ));
            }
        };
        let x = add_consuming_rhs(x, attn_out, store, tape)?;

        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let mlp_out = self.forward_mlp(
            h,
            cfg,
            TpContext::single(),
            batch,
            seq_len,
            &mut MoeRouteMode::Free,
            store,
            tape,
        )?;
        Ok(add_consuming_rhs(x, mlp_out, store, tape)?)
    }

    pub(super) fn forward_with_kv_cache_profiled(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        cos: TensorId,
        sin: TensorId,
        layer_cache: &mut Qwen35LayerKvCache,
        q_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(TensorId, Qwen35LayerForwardProfile)> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];
        let mut profile = Qwen35LayerForwardProfile::default();

        let started = Instant::now();
        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        profile.input_rmsnorm += started.elapsed();

        let started = Instant::now();
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => {
                let mut attention_profile = Qwen35AttentionForwardProfile::default();
                let out = self.forward_full_attention_with_kv_cache_profiled(
                    h,
                    attn,
                    cfg,
                    cos,
                    sin,
                    batch,
                    seq_len,
                    layer_cache,
                    q_start,
                    store,
                    tape,
                    &mut attention_profile,
                )?;
                profile.attention_detail = attention_profile;
                out
            }
            Qwen35Attention::Linear(_) => {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache requires full-attention layers",
                ));
            }
        };
        profile.attention += started.elapsed();

        let started = Instant::now();
        let x = add_consuming_rhs(x, attn_out, store, tape)?;
        profile.attention_residual += started.elapsed();

        let started = Instant::now();
        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        profile.post_attention_rmsnorm += started.elapsed();

        let started = Instant::now();
        let mlp_out = self.forward_mlp(
            h,
            cfg,
            TpContext::single(),
            batch,
            seq_len,
            &mut MoeRouteMode::Free,
            store,
            tape,
        )?;
        profile.mlp += started.elapsed();

        let started = Instant::now();
        let out = add_consuming_rhs(x, mlp_out, store, tape)?;
        profile.mlp_residual += started.elapsed();

        Ok((out, profile))
    }
}
