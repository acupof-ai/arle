use super::*;

impl Qwen35Layer {
    pub(super) fn forward_linear_attention(
        &self,
        h: TensorId,
        attn: &Qwen35LinearAttention,
        cfg: &Qwen35Config,
        cp: crate::context_parallel::CpContext,
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        // Projections are position-wise — chunking keeps only the full-seq
        // output resident instead of every matmul/LoRA intermediate.
        let qkv = chunked_proj(&attn.in_proj_qkv, h, store, tape)?;
        let z = chunked_proj(&attn.in_proj_z, h, store, tape)?;
        let b_proj = chunked_proj(&attn.in_proj_b, h, store, tape)?;
        let a_proj = chunked_proj(&attn.in_proj_a, h, store, tape)?;
        // CP shards sequence; linear_attention_core_cp runs the recurrence on this
        // rank's rows and carries the state across ranks in global order (taped, so
        // the state gradient crosses back). cp.size==1 is the single-card core.
        //
        // Its own checkpoint sub-group. A layer-sized group frees nothing until the
        // whole layer's backward is done, so the core's transport chain and scan
        // scratch stay resident through the projection and out_proj backwards —
        // the layer peak is the SUM over its stages. A nested boundary makes it the
        // MAX. Inert in the forward: `checkpoint` passes straight through while the
        // outer group has the tape disabled, so this only splits the replay.
        let params = la_params(cfg, batch, seq_len);
        let (cp_size, cp_rank) = (cp.size, cp.rank);
        let linear = autograd::ops::checkpoint(
            vec![
                qkv,
                z,
                b_proj,
                a_proj,
                attn.conv1d_weight,
                attn.dt_bias,
                attn.a_log,
                attn.norm,
            ],
            store,
            tape,
            move |st, tp, inp| {
                let [qkv, z, b_proj, a_proj, conv1d_weight, dt_bias, a_log, norm] = inp else {
                    return Err(autograd::AutogradError::TapeInvariant(
                        "linear-attention core checkpoint expects 8 saved inputs",
                    ));
                };
                autograd::ops::linear_attention_core_cp(
                    *qkv,
                    *z,
                    *b_proj,
                    *a_proj,
                    *conv1d_weight,
                    *dt_bias,
                    *a_log,
                    *norm,
                    params,
                    cp_size,
                    cp_rank,
                    st,
                    tp,
                )
            },
        )?;
        chunked_proj(&attn.out_proj, linear, store, tape)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_linear_attention_capture_prefix_state(
        &self,
        h_prefix: TensorId,
        attn: &Qwen35LinearAttention,
        cfg: &Qwen35Config,
        batch: usize,
        gen_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<PrefixState> {
        let qkv = attn.in_proj_qkv.forward(h_prefix, store, tape)?;
        let b_proj = attn.in_proj_b.forward(h_prefix, store, tape)?;
        let a_proj = attn.in_proj_a.forward(h_prefix, store, tape)?;
        let params = la_params(cfg, batch, gen_start);
        let (state, conv_window) = if store.backend().device() == Device::Cuda
            && params.seq_len >= params.conv_kernel - 1
        {
            linear_attention_boundary(
                qkv,
                b_proj,
                a_proj,
                attn.conv1d_weight,
                attn.dt_bias,
                attn.a_log,
                params,
                None,
                None,
                store,
            )?
        } else {
            let z = attn.in_proj_z.forward(h_prefix, store, tape)?;
            let (_, state, conv) = linear_attention_core_with_carry(
                qkv,
                z,
                b_proj,
                a_proj,
                attn.conv1d_weight,
                attn.dt_bias,
                attn.a_log,
                attn.norm,
                params,
                None,
                None,
                true,
                store,
            )?;
            (
                state.ok_or(Qwen35Error::InvalidConfig("missing linear-attention state"))?,
                conv.ok_or(Qwen35Error::InvalidConfig(
                    "missing linear-attention conv tail",
                ))?,
            )
        };
        Ok(PrefixState { state, conv_window })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_linear_attention_gen_segment(
        &self,
        h_gen: TensorId,
        attn: &Qwen35LinearAttention,
        cfg: &Qwen35Config,
        prefix_state: &PrefixState,
        batch: usize,
        gen_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let qkv = attn.in_proj_qkv.forward(h_gen, store, tape)?;
        let z = attn.in_proj_z.forward(h_gen, store, tape)?;
        let b_proj = attn.in_proj_b.forward(h_gen, store, tape)?;
        let a_proj = attn.in_proj_a.forward(h_gen, store, tape)?;
        let linear = linear_attention_core_with_carry_taped(
            qkv,
            z,
            b_proj,
            a_proj,
            attn.conv1d_weight,
            attn.dt_bias,
            attn.a_log,
            attn.norm,
            la_params(cfg, batch, gen_len),
            Some(prefix_state.state),
            Some(prefix_state.conv_window),
            store,
            tape,
        )?;
        Ok(attn.out_proj.forward(linear, store, tape)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_linear_attention_profiled(
        &self,
        layer_index: usize,
        h: TensorId,
        attn: &Qwen35LinearAttention,
        cfg: &Qwen35Config,
        batch: usize,
        seq_len: usize,
        trace: bool,
        store: &mut TensorStore,
        tape: &mut Tape,
        profile: &mut Qwen35AttentionForwardProfile,
    ) -> Result<TensorId> {
        let started = Instant::now();
        let qkv = attn.in_proj_qkv.forward(h, store, tape)?;
        profile.linear_qkv_proj += started.elapsed();
        trace_attention_component(
            trace,
            layer_index,
            "linear_qkv_proj",
            profile.linear_qkv_proj,
        );

        let started = Instant::now();
        let z = attn.in_proj_z.forward(h, store, tape)?;
        profile.linear_z_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "linear_z_proj", profile.linear_z_proj);

        let started = Instant::now();
        let b_proj = attn.in_proj_b.forward(h, store, tape)?;
        profile.linear_b_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "linear_b_proj", profile.linear_b_proj);

        let started = Instant::now();
        let a_proj = attn.in_proj_a.forward(h, store, tape)?;
        profile.linear_a_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "linear_a_proj", profile.linear_a_proj);

        let started = Instant::now();
        let linear = linear_attention_core(
            qkv,
            z,
            b_proj,
            a_proj,
            attn.conv1d_weight,
            attn.dt_bias,
            attn.a_log,
            attn.norm,
            la_params(cfg, batch, seq_len),
            store,
            tape,
        )?;
        profile.linear_core += started.elapsed();
        trace_attention_component(trace, layer_index, "linear_core", profile.linear_core);

        let started = Instant::now();
        let out = attn.out_proj.forward(linear, store, tape)?;
        profile.linear_out_proj += started.elapsed();
        trace_attention_component(
            trace,
            layer_index,
            "linear_out_proj",
            profile.linear_out_proj,
        );
        Ok(out)
    }
}

/// Position-wise projection through `checkpoint_seq_chunked`: backward replays
/// each chunk, so LoRA/base intermediates never sit resident at full seq.
fn chunked_proj(
    proj: &LinearWithLora,
    x: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let mut param_ids = Vec::new();
    collect_linear_ids(proj, &mut param_ids);
    param_ids.retain(|&id| store.get(id).is_some_and(|t| t.requires_grad));
    let proj = proj.clone();
    Ok(autograd::ops::checkpoint_seq_chunked(
        x,
        param_ids,
        crate::runtime_flags::opd_seq_chunk(),
        store,
        tape,
        move |st, tp, _start, inp| proj.forward(inp[0], st, tp),
    )?)
}
