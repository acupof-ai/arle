use super::layer_block::HcHalf;
use super::*;
use cuda_kernels::tensor::CudaPipelineStreamKind;

impl Dsv4Model {
    /// The residual is the `hidden_size * hc_mult`-wide hyper-connection STREAM,
    /// not a plain hidden vector; the head HC folds it to one hidden row before
    /// the final RMSNorm + lm_head + sample.
    pub(crate) fn forward_tokens(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
    ) -> Result<u32> {
        self.forward_tokens_impl(
            slot, kv_adapter, tokens, start_pos, params, position, penalty, None,
        )
    }

    pub(crate) fn forward_tokens_with_hidden(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
    ) -> Result<(u32, DeviceVec)> {
        let mut last_hidden =
            DeviceVec::zeros(&self.ctx, self.config.hidden_size * self.config.hc_mult)?;
        let token = self.forward_tokens_impl(
            slot,
            kv_adapter,
            tokens,
            start_pos,
            params,
            position,
            penalty,
            Some(&mut last_hidden),
        )?;
        Ok((token, last_hidden))
    }

    pub(super) fn forward_tokens_impl(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
        last_hidden_out: Option<&mut DeviceVec>,
    ) -> Result<u32> {
        let seq_len = tokens.len();
        let (stream, mut keepalive) =
            self.forward_tokens_stream_impl(slot, kv_adapter, tokens, start_pos, false, None)?;
        if let Some(out) = last_hidden_out {
            self.capture_mtp_stream_hidden(&stream, seq_len - 1, 1, out, &mut keepalive)?;
        }
        let token = self.forward_stream_last_token(
            &stream,
            seq_len,
            params,
            position,
            penalty,
            None,
            &mut keepalive,
        )?;
        std::hint::black_box(keepalive.len());
        drop(keepalive);
        Ok(token)
    }

    pub(crate) fn forward_tokens_stream_impl(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        tokens: &[u32],
        start_pos: usize,
        persist_spec_normed: bool,
        // `Some`: verify a scheduled row chunk. Ancestor metadata routes attention
        // through one FlashMLA sparse verify call per layer; point-wise and MoE
        // stay batched for weight-read amortization.
        verify: Option<&SpecVerifySchedule>,
    ) -> Result<(super::StepBuf, Dsv4ForwardKeepalive)> {
        ensure!(
            !tokens.is_empty(),
            "DSv4 forward requires at least one token"
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
        if let Some(sched) = verify
            && tokens.len() <= MAX_SPEC_VERIFY_ROWS
            && !crate::runtime_flags::dsv4_moe_transport()?.is_deepep()
        {
            return self
                .forward_tokens_verify_stream_persistent(slot, kv_adapter, tokens, start_pos, sched)
                .map(|(h, k)| (super::StepBuf::Owned(h), k));
        }

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let seq_len = tokens.len();
        let mega_epoch = self.begin_mega_moe_forward(seq_len)?;
        let use_deepep_transport = crate::runtime_flags::dsv4_moe_transport()?.is_deepep();
        let dspark = self.config.is_dspark();
        let mut keepalive = Dsv4ForwardKeepalive::new(false);
        let ctx = &self.ctx;
        // DSpark prefix seed: a PREFILL forward (`seq_len > 1`) additionally
        // captures ALL rows per target layer into transient `[stream_dim, seq_len]`
        // buffers, so the executor can seed the full committed context (the
        // per-slot single-row `dspark_taps` only holds `seq_len-1`).
        let mut dspark_prompt_bufs: Option<Vec<DeviceVec>> = if dspark && seq_len > 1 {
            let n_taps = self.config.dspark_target_layer_ids.len();
            Some(
                (0..n_taps)
                    .map(|_| DeviceVec::zeros(ctx, stream_dim * seq_len))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        // Catch any deferred CUDA error from init/boot before the first forward op.
        if use_deepep_transport {
            self.ctx.stream.synchronize().map_err(|e| {
                anyhow!("DSv4 deepep: stream error at forward entry (before embed): {e}")
            })?;
        }
        let graph_mode = self.graph_mode() && seq_len == 1;
        let start_pos_device = if seq_len == 1 {
            if !graph_mode {
                let start_pos_i32 = i32::try_from(start_pos)
                    .map_err(|_| anyhow!("DSv4 start_pos {start_pos} overflows i32"))?;
                self.ctx
                    .stream
                    .memcpy_htod(&[start_pos_i32], &mut slot.start_pos_device)
                    .map_err(|e| anyhow!("DSv4 start_pos H2D failed: {e}"))?;
            }
            Some(&slot.start_pos_device)
        } else {
            None
        };

        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = if graph_mode {
            self.graph_token_ids_i32()?
        } else {
            super::StepSlice::Owned(crate::ops::upload_i32(&self.ctx, &token_ids_host)?)
        };
        keepalive.keep_i32(&token_ids);
        // SAFETY: embed_stream writes the full stream buffer.
        let mut stream = self.step_hidden(
            graph_mode,
            (super::GRAPH_MODEL_LAYER, super::GraphSlot::Stream),
            stream_dim,
            seq_len,
        )?;
        self.embed_stream(&token_ids, seq_len, &mut stream, &mut keepalive)?;
        let sparse_verify_meta = match verify {
            Some(sched) => {
                ensure!(
                    sched.positions.len() == seq_len && sched.ancestors.len() == seq_len,
                    "DSv4 sparse verify schedule rows ({}, {}) != token rows {seq_len}",
                    sched.positions.len(),
                    sched.ancestors.len()
                );
                sched.validate_sparse_at(start_pos)?;
                Some(crate::attention::Dsv4ChainVerifyAttnMeta::new(
                    &self.ctx,
                    &sched.positions,
                    &sched.ancestors,
                )?)
            }
            _ => None,
        };
        let probe_positions: Vec<usize> = if self.probe.borrow().is_some() {
            (0..seq_len).map(|i| start_pos + i).collect()
        } else {
            Vec::new()
        };
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // SAFETY: fused hc_pre+rms_norm / plain rms_norm writes the full
            // [seq_len, hidden_size] buffer.
            let mut normed = self.step_hidden(
                graph_mode,
                (layer_idx, super::GraphSlot::NormedAttn),
                hidden_size,
                seq_len,
            )?;
            let attn_mhc = self.hc_pre_norm(
                layer,
                HcHalf::Attn,
                layer_idx,
                seq_len,
                &stream,
                &mut normed,
                &mut keepalive,
                graph_mode,
            )?;
            keepalive.keep_hidden(&normed);
            if persist_spec_normed {
                let cache = slot
                    .spec_normed
                    .as_mut()
                    .ok_or_else(|| anyhow!("DSv4 verify without spec_normed cache"))?;
                let rows = seq_len * hidden_size;
                let src = normed.data.slice(0..rows);
                let mut dst = cache[layer_idx].data.slice_mut(0..rows);
                self.ctx
                    .stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| anyhow!("DSv4 commit-fold normed persist failed: {e}"))?;
            }
            // SAFETY: mla_attention writes the full [seq_len, hidden_size] buffer.
            let mut attn_out = self.step_hidden(
                graph_mode,
                (layer_idx, super::GraphSlot::AttnOut),
                hidden_size,
                seq_len,
            )?;
            if let Some(meta) = sparse_verify_meta.as_ref() {
                // Scheduled verify: one sparse FlashMLA attention over the whole
                // row chunk. Ancestors carry row `r`'s chain dependency, so no row
                // replay or slot-ring mutation is needed.
                crate::profile::profile_op(
                    &self.ctx,
                    "attention",
                    Some(layer_idx),
                    seq_len,
                    || {
                        let (layer_pool, dsa_shared, flashmla_scratch, prefill_shared, fp32) =
                            kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                        crate::attention::mla_attention(
                            &self.attn_ctx(
                                layer,
                                layer_idx,
                                crate::attention::Dsv4Position {
                                    start: start_pos,
                                    device: None,
                                },
                                Some(meta),
                            ),
                            &normed,
                            &mut slot.attention[layer_idx],
                            layer_pool,
                            dsa_shared,
                            flashmla_scratch,
                            prefill_shared,
                            fp32,
                            &mut attn_out,
                            &mut keepalive,
                        )
                    },
                )?;
            } else {
                crate::profile::profile_op(
                    &self.ctx,
                    "attention",
                    Some(layer_idx),
                    seq_len,
                    || {
                        let model1_decode = seq_len == 1
                            && start_pos_device.is_some()
                            && layer.attention.w_kc.is_none()
                            && layer.attention.w_vc.is_none()
                            && layer.attention.o_proj.is_none();
                        if model1_decode {
                            let (layer_pool, dsa_shared, flashmla_scratch, mla_scratch) =
                                kv_adapter.layer_flashmla_and_mla_decode_mut(layer_idx)?;
                            let mla_scratch = mla_scratch.ok_or_else(|| {
                                anyhow!("DSv4 MODEL1 layer {layer_idx} missing MLA decode scratch")
                            })?;
                            crate::attention::mla_attention_decode(
                                &self.attn_ctx(
                                    layer,
                                    layer_idx,
                                    crate::attention::Dsv4Position {
                                        start: start_pos,
                                        device: start_pos_device,
                                    },
                                    None,
                                ),
                                &normed,
                                &mut slot.attention[layer_idx],
                                layer_pool,
                                dsa_shared,
                                flashmla_scratch,
                                mla_scratch,
                                &mut attn_out,
                                &mut keepalive,
                            )
                        } else {
                            let (layer_pool, dsa_shared, flashmla_scratch, prefill_shared, fp32) =
                                kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                            crate::attention::mla_attention(
                                &self.attn_ctx(
                                    layer,
                                    layer_idx,
                                    crate::attention::Dsv4Position {
                                        start: start_pos,
                                        device: start_pos_device,
                                    },
                                    None,
                                ),
                                &normed,
                                &mut slot.attention[layer_idx],
                                layer_pool,
                                dsa_shared,
                                flashmla_scratch,
                                prefill_shared,
                                fp32,
                                &mut attn_out,
                                &mut keepalive,
                            )
                        }
                    },
                )?;
            }
            keepalive.keep_hidden(&attn_out);
            // Row-parallel O-LoRA: sum the per-rank partials (no-op single-GPU).
            crate::profile::profile_op(
                &self.ctx,
                "attn_allreduce",
                Some(layer_idx),
                seq_len,
                || self.tp.all_reduce_sum(&self.ctx, &mut attn_out),
            )?;
            // SAFETY: hc_post / add_batch writes the full stream buffer.
            let mut attn_stream = self.step_hidden(
                graph_mode,
                (layer_idx, super::GraphSlot::AttnStream),
                stream_dim,
                seq_len,
            )?;
            self.hc_post_fold(
                attn_mhc.as_ref(),
                HcHalf::Attn,
                layer_idx,
                seq_len,
                &attn_out,
                &stream,
                &mut attn_stream,
            )?;
            keepalive.keep_hidden(&attn_stream);
            stream = attn_stream;

            // SAFETY: fused hc_pre+rms_norm / plain rms_norm writes the full
            // [seq_len, hidden_size] buffer.
            let mut normed = self.step_hidden(
                graph_mode,
                (layer_idx, super::GraphSlot::NormedFfn),
                hidden_size,
                seq_len,
            )?;
            let ffn_mhc = self.hc_pre_norm(
                layer,
                HcHalf::Ffn,
                layer_idx,
                seq_len,
                &stream,
                &mut normed,
                &mut keepalive,
                graph_mode,
            )?;
            keepalive.keep_hidden(&normed);
            // GLM dense layer (`per_layer_dense_mlp[i]`): a plain SwiGLU FFN
            // replaces the routed-expert + shared-expert MoE entirely.
            let mut moe_with_shared = self.step_hidden(
                graph_mode,
                (layer_idx, super::GraphSlot::MoeWithShared),
                hidden_size,
                seq_len,
            )?;
            if let Some(dense) = layer.dense_mlp.as_ref() {
                crate::profile::profile_op(ctx, "mlp", Some(layer_idx), seq_len, || {
                    dsv4_dense_mlp_forward(
                        &self.ctx,
                        dense,
                        &normed,
                        &mut moe_with_shared,
                        self.config.swiglu_limit,
                        &mut keepalive,
                    )
                })?;
                keepalive.keep_hidden(&moe_with_shared);
            } else {
                let use_comm_overlap = seq_len == 1 && !use_deepep_transport && !graph_mode;
                let normed_ready = if use_comm_overlap {
                    let fence = ctx.record_pipeline_fence(CudaPipelineStreamKind::Compute)?;
                    ctx.wait_on_pipeline_fence(&fence, CudaPipelineStreamKind::Comm)?;
                    Some(fence)
                } else {
                    None
                };
                // SAFETY: the MoE forward writes the full routed output buffer.
                let mut moe_out = self.step_hidden(
                    graph_mode,
                    (layer_idx, super::GraphSlot::MoeOut),
                    hidden_size,
                    seq_len,
                )?;
                // DeepEP combine already reduces the EP-sharded routed output; the
                // non-deepep path needs the explicit TP all-reduce below.
                let (shared_out, mut shared_scratch, moe_tail) =
                    kv_adapter.moe_decode_scratch_mut();
                let needs_moe_allreduce = if use_deepep_transport {
                    #[cfg(feature = "deepep")]
                    {
                        let transport = self.deepep.as_ref().ok_or_else(|| {
                            anyhow!(
                                "ARLE_DSV4_MOE_TRANSPORT=deepep but DeepEP transport is not booted"
                            )
                        })?;
                        if transport.is_low_latency() {
                            // deepep_ll token-owned path: TP8 replicates `normed`
                            // and HiddenStates is token-major, so this rank's owned
                            // shard is a CONTIGUOUS byte range. LL dispatch + combine
                            // are COLLECTIVES: every rank must participate every step
                            // or the symmetric protocol deadlocks, so the helper runs
                            // `compute_fn` even at owned_n == 0. This REPLACES the
                            // moe all-reduce.
                            let scratch = slot.deepep_ll_scratch.as_mut().ok_or_else(|| {
                                anyhow!("deepep_ll selected but slot LL scratch not allocated")
                            })?;
                            self.tp.shard_rows_and_allreduce(
                                &self.ctx,
                                &normed,
                                &mut moe_out,
                                hidden_size,
                                seq_len,
                                |owned_in, owned_out, start, owned_n| {
                                    keepalive.keep_hidden(owned_in);
                                    keepalive.keep_hidden(owned_out);
                                    crate::moe::dsv4_moe_forward_deepep_ll(
                                        self,
                                        transport,
                                        scratch,
                                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                                        &tokens[start..start + owned_n],
                                        tokens.len(),
                                        owned_in,
                                        owned_out,
                                        &mut keepalive,
                                    )
                                },
                            )?;
                        } else {
                            crate::moe::dsv4_moe_forward_deepep(
                                self,
                                transport,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                tokens,
                                &normed,
                                &mut moe_out,
                                &mut keepalive,
                            )?;
                        }
                        false
                    }
                    #[cfg(not(feature = "deepep"))]
                    {
                        anyhow::bail!(
                            "ARLE_DSV4_MOE_TRANSPORT=deepep requires infer-cuda feature deepep"
                        );
                    }
                } else {
                    crate::profile::profile_op(ctx, "moe_route", Some(layer_idx), seq_len, || {
                        crate::moe::dsv4_moe_forward(
                            self,
                            layer.moe.as_ref().expect("DSv4 layer.moe"),
                            tokens,
                            &normed,
                            &mut moe_out,
                            &mut keepalive,
                            moe_tail,
                            mega_epoch,
                        )
                    })?
                };
                keepalive.keep_hidden(&moe_out);
                crate::profile::profile_op(
                    &self.ctx,
                    "shared_hc",
                    Some(layer_idx),
                    seq_len,
                    || {
                        let mut shared_owned = None;
                        let shared_ready = if use_comm_overlap {
                            shared_out.seq_len = seq_len;
                            ensure!(
                                shared_out.hidden_dim == hidden_size
                                    && shared_out.seq_len == seq_len,
                                "DSv4 shared decode scratch shape {}x{} != {}x{}",
                                shared_out.hidden_dim,
                                shared_out.seq_len,
                                hidden_size,
                                seq_len
                            );
                            let _normed_ready = normed_ready
                                .as_ref()
                                .expect("comm-overlap path records normed fence");
                            let scratch = shared_scratch.as_deref_mut().ok_or_else(|| {
                                anyhow!("DSv4 decode requires shared-expert scratch")
                            })?;
                            crate::moe::dsv4_shared_expert_forward_decode_scratch(
                                &self.ctx,
                                &self.ctx.comm_stream,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                &normed,
                                shared_out,
                                self.config.swiglu_limit,
                                scratch,
                            )?;
                            Some(ctx.record_pipeline_fence(CudaPipelineStreamKind::Comm)?)
                        } else {
                            None
                        };
                        // Routed experts are EP-sharded; sum them first, then add the
                        // replicated shared expert once per rank. In the comm-overlap
                        // path the two depend on different inputs and run concurrently.
                        if needs_moe_allreduce {
                            crate::profile::profile_op(
                                &self.ctx,
                                "moe_allreduce",
                                Some(layer_idx),
                                seq_len,
                                || {
                                    self.tp.all_reduce_sum_on(
                                        &self.ctx,
                                        &mut moe_out,
                                        CudaPipelineStreamKind::Compute,
                                    )
                                },
                            )?;
                        }
                        if use_comm_overlap {
                            // Already launched on the comm stream above.
                        } else if seq_len == 1 {
                            shared_out.seq_len = seq_len;
                            ensure!(
                                shared_out.hidden_dim == hidden_size
                                    && shared_out.seq_len == seq_len,
                                "DSv4 shared decode scratch shape {}x{} != {}x{}",
                                shared_out.hidden_dim,
                                shared_out.seq_len,
                                hidden_size,
                                seq_len
                            );
                            let scratch = shared_scratch.as_deref_mut().ok_or_else(|| {
                                anyhow!("DSv4 decode requires shared-expert scratch")
                            })?;
                            crate::moe::dsv4_shared_expert_forward_decode_scratch(
                                &self.ctx,
                                &self.ctx.stream,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                &normed,
                                shared_out,
                                self.config.swiglu_limit,
                                scratch,
                            )?;
                        } else if !use_deepep_transport
                            && sparse_verify_meta.is_some()
                            && seq_len <= MAX_SPEC_VERIFY_ROWS
                        {
                            shared_out.seq_len = seq_len;
                            ensure!(
                                shared_out.hidden_dim == hidden_size,
                                "DSv4 shared verify scratch hidden {} != {}",
                                shared_out.hidden_dim,
                                hidden_size
                            );
                            let scratch = shared_scratch.ok_or_else(|| {
                                anyhow!("DSv4 verify requires shared-expert scratch")
                            })?;
                            crate::moe::dsv4_shared_expert_forward_decode_scratch(
                                &self.ctx,
                                &self.ctx.stream,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                &normed,
                                shared_out,
                                self.config.swiglu_limit,
                                scratch,
                            )?;
                        } else {
                            // Keep the default masked path order: routed MoE →
                            // all-reduce → allocate/run shared expert. Allocating
                            // shared scratch before the all-reduce can reuse in-flight
                            // helper buffers under disabled cudarc event tracking.
                            // SAFETY: dsv4_shared_expert_forward writes the full shared
                            // output.
                            let mut shared =
                                // SAFETY: fully written before first read.
                                unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                            crate::moe::dsv4_shared_expert_forward(
                                &self.ctx,
                                &self.ctx.stream,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                &normed,
                                &mut shared,
                                self.config.swiglu_limit,
                                &mut keepalive,
                            )?;
                            shared_owned = Some(shared);
                        };
                        let shared = if seq_len == 1
                            || (!use_deepep_transport
                                && sparse_verify_meta.is_some()
                                && seq_len <= MAX_SPEC_VERIFY_ROWS)
                        {
                            &*shared_out
                        } else {
                            shared_owned
                                .as_ref()
                                .expect("multi-token shared expert allocates owned output")
                        };
                        keepalive.keep_hidden(shared);
                        if let Some(fence) = shared_ready.as_ref() {
                            ctx.wait_on_pipeline_fence(fence, CudaPipelineStreamKind::Compute)?;
                        }
                        // SAFETY: add_batch writes the full [seq_len, hidden_size]
                        // buffer.
                        crate::ops::add_batch(&self.ctx, &moe_out, shared, &mut moe_with_shared)?;
                        keepalive.keep_hidden(&moe_with_shared);
                        Ok(())
                    },
                )?;
            }
            // SAFETY: hc_post / add_batch writes the full stream buffer.
            let mut ffn_stream = self.step_hidden(
                graph_mode,
                (layer_idx, super::GraphSlot::FfnStream),
                stream_dim,
                seq_len,
            )?;
            self.hc_post_fold(
                ffn_mhc.as_ref(),
                HcHalf::Ffn,
                layer_idx,
                seq_len,
                &moe_with_shared,
                &stream,
                &mut ffn_stream,
            )?;
            keepalive.keep_hidden(&ffn_stream);
            stream = ffn_stream;
            self.probe_capture(&stream, layer_idx, &probe_positions)?;
            // D2D copy only (graph-safe, no readback).
            if dspark
                && let Some(tap_idx) = self
                    .config
                    .dspark_target_layer_ids
                    .iter()
                    .position(|&l| l == layer_idx)
            {
                let tap = &mut slot.dspark_taps[tap_idx];
                let elems = stream_dim * seq_len;
                if tap.len >= elems {
                    let src = stream.data.slice(0..elems);
                    let mut dst = tap.data.slice_mut(0..elems);
                    ctx.stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| anyhow!("DSpark tap capture D2D failed: {e}"))?;
                    keepalive.keep_vec(tap);
                } else {
                    self.capture_mtp_stream_hidden(&stream, seq_len - 1, 1, tap, &mut keepalive)?;
                }
                if let Some(bufs) = dspark_prompt_bufs.as_mut() {
                    ctx.stream
                        .memcpy_dtod(&stream.data, &mut bufs[tap_idx].data)
                        .map_err(|e| anyhow!("DSpark prompt tap capture D2D failed: {e}"))?;
                    keepalive.keep_vec(&bufs[tap_idx]);
                }
            }
        }

        self.probe_flush()?;
        if let Some(bufs) = dspark_prompt_bufs.take() {
            slot.dspark_prompt_taps = Some(Dsv4DsparkPromptTaps {
                bufs,
                rows: seq_len,
            });
        }

        slot.seq_len += seq_len;
        Ok((stream, keepalive))
    }
}
