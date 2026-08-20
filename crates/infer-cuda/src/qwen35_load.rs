use super::*;

impl Qwen35Model {
    pub(crate) fn warm_fp8_deepgemm_dense_prefill(&self) -> Result<(usize, usize)> {
        let warm_m = self.max_seq_len.min(2048);
        // sm_120 dense FP8 runs the dequant→BF16 fallback (no DeepGEMM); nothing to warm.
        if self.ctx.is_sm120() {
            return Ok((0, warm_m));
        }
        if warm_m < 1024 {
            return Ok((0, warm_m));
        }
        let mut seen = HashSet::new();
        let mut warmed = 0usize;
        // The quant layout is part of the key, not just the dims: this
        // checkpoint's NVFP4 and per-channel FP8 MLP weights share
        // [34816, 5120] and [5120, 17408], and on dims alone the NVFP4 one
        // claims the shape, declines the warm, and leaves the FP8 one to
        // compile in-request.
        let mut warm = |weight: &DeviceMatrix| -> Result<()> {
            let key = (
                weight.rows,
                weight.cols,
                weight.weight_format,
                weight.quant_block_m,
            );
            if seen.insert(key) && warm_fp8_deepgemm_dense(&self.ctx, weight, warm_m)? {
                // Also JIT-warm the spec-verify row count so the first DSpark
                // block step doesn't compile DeepGEMM M=16 kernels in-request.
                warm_fp8_deepgemm_dense(&self.ctx, weight, 16)?;
                warmed += 1;
            }
            Ok(())
        };

        for layer in &self.layers {
            match &layer.attn {
                Qwen35Attn::Full(full) => {
                    warm(&full.qkv_proj)?;
                    warm(&full.o_proj)?;
                }
                Qwen35Attn::Linear(linear) => {
                    warm(&linear.in_proj_qkvz)?;
                    warm(&linear.in_proj_ba)?;
                    warm(&linear.out_proj)?;
                }
            }
            if let Some(mlp) = &layer.mlp {
                warm(&mlp.gate_up_proj)?;
                warm(&mlp.down_proj)?;
            }
            if let Some(moe) = &layer.moe {
                warm(&moe.router_gate)?;
                warm(&moe.shared_gate)?;
                warm(&moe.shared_up)?;
                warm(&moe.shared_down)?;
                warm(&moe.shared_gate_router)?;
            }
        }
        if warmed > 0 {
            self.ctx.sync()?;
        }
        Ok((warmed, warm_m))
    }

    pub(crate) fn warm_fp8_deepgemm_grouped_prefill(&self) -> Result<(usize, usize, usize, usize)> {
        let warm_tokens = self.max_seq_len.min(2048);
        // sm_120 routes grouped FP8 to the AOT CUTLASS collective — no DeepGEMM
        // JIT kernels to warm (the preflight below is Hopper-only).
        if self.ctx.is_sm120() {
            return Ok((0, warm_tokens, 0, 0));
        }
        let topk = self.config.num_experts_per_tok;
        let mut seen = HashSet::new();
        let mut warmed = 0usize;
        let mut min_rows = usize::MAX;
        let mut max_rows = 0usize;
        for warm_tokens in [warm_tokens, warm_tokens.saturating_sub(16)] {
            let warm_routes = warm_tokens.saturating_mul(topk);
            if warm_routes < QWEN35_DEEPGEMM_MIN_ROUTES {
                continue;
            }
            for layer in &self.layers {
                let Some(moe) = &layer.moe else {
                    continue;
                };
                let (Some(w13), Some(down)) = (&moe.w13_fp8_grouped, &moe.down_fp8_grouped) else {
                    continue;
                };
                let rows = deepgemm_contig_rows_cap(warm_routes, w13.groups, DEEPGEMM_CONTIG_ALIGN);
                let key = (w13.groups, w13.rows, w13.cols, down.rows, down.cols, rows);
                if seen.insert(key) {
                    Self::warm_fp8_deepgemm_grouped_pair(&self.ctx, w13, down, rows)?;
                    min_rows = min_rows.min(rows);
                    max_rows = max_rows.max(rows);
                    warmed += 2;
                }
            }
        }
        if warmed > 0 {
            self.ctx.sync()?;
        }
        if warmed == 0 {
            min_rows = 0;
        }
        Ok((warmed, self.max_seq_len.min(2048), min_rows, max_rows))
    }

    pub(crate) fn warm_fp8_deepgemm_grouped_pair(
        ctx: &DeviceContext,
        w13: &crate::loader::MoeFp8ExpertGroup,
        down: &crate::loader::MoeFp8ExpertGroup,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            w13.groups == down.groups && w13.cols == down.rows && w13.rows == 2 * down.cols,
            "Qwen FP8 grouped DeepGEMM warm shape mismatch: w13={}x{} g={} down={}x{} g={}",
            w13.rows,
            w13.cols,
            w13.groups,
            down.rows,
            down.cols,
            down.groups
        );
        ensure!(
            rows.is_multiple_of(DEEPGEMM_CONTIG_ALIGN),
            "Qwen FP8 grouped DeepGEMM warm rows {rows} not aligned to {DEEPGEMM_CONTIG_ALIGN}"
        );
        cuda_moe::dsv4_deepgemm_native_preflight()?;

        let hidden = w13.cols;
        let intermediate = down.cols;
        let scale_stride_m = rows.div_ceil(4) * 4;
        let hidden_scale_cols = hidden.div_ceil(128);
        let inter_scale_cols = intermediate.div_ceil(128);
        let input_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(rows * hidden)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm input alloc failed: {e}"))?;
        let input_scales = ctx
            .stream
            .alloc_zeros::<f32>(scale_stride_m * hidden_scale_cols)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm input scale alloc failed: {e}"))?;
        let w13_out = ctx
            .stream
            .alloc_zeros::<bf16>(rows * w13.rows)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm w13 output alloc failed: {e}"))?;
        let act_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(rows * intermediate)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm act alloc failed: {e}"))?;
        let act_scales = ctx
            .stream
            .alloc_zeros::<f32>(scale_stride_m * inter_scale_cols)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm act scale alloc failed: {e}"))?;
        let out = ctx
            .stream
            .alloc_zeros::<bf16>(rows * hidden)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm output alloc failed: {e}"))?;
        let m_indices = ctx
            .stream
            .alloc_zeros::<i32>(rows)
            .map_err(|e| anyhow!("Qwen FP8 grouped DeepGEMM warm m_indices alloc failed: {e}"))?;
        let stream = ctx.stream.cu_stream();

        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            cuda_moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                cache_ptr(&input_fp8, ctx),
                cache_ptr(&input_scales, ctx),
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                cache_ptr(&w13_out, ctx),
                cache_ptr(&m_indices, ctx),
                w13.groups,
                rows,
                w13.rows,
                hidden,
                scale_stride_m,
                DEEPGEMM_CONTIG_ALIGN,
                stream,
            )?;
            cuda_moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                cache_ptr(&act_fp8, ctx),
                cache_ptr(&act_scales, ctx),
                cache_ptr(&down.weight, ctx),
                cache_ptr(&down.scales, ctx),
                cache_ptr(&out, ctx),
                cache_ptr(&m_indices, ctx),
                down.groups,
                rows,
                hidden,
                intermediate,
                scale_stride_m,
                DEEPGEMM_CONTIG_ALIGN,
                stream,
            )?;
        }
        Ok(())
    }

    pub(crate) fn from_safetensors(
        model_path: &Path,
        max_seq_len: usize,
        mtp_draft_tokens: Option<usize>,
    ) -> Result<Self> {
        let tp = crate::loader::build_tp_runtime(false)?;
        Self::from_safetensors_with_tp(model_path, max_seq_len, tp, mtp_draft_tokens)
    }

    pub(crate) fn from_safetensors_with_tp(
        model_path: &Path,
        max_seq_len: usize,
        #[cfg_attr(not(feature = "nccl"), allow(unused_mut))] mut tp: crate::tp::TpRuntime,
        mtp_draft_tokens: Option<usize>,
    ) -> Result<Self> {
        let total_t0 = Instant::now();
        let config_t0 = Instant::now();
        let m = Qwen35Config::from_model_dir(model_path)
            .map_err(|e| anyhow!("load Qwen3.5 config from {}: {e}", model_path.display()))?;
        validate_qwen35_cuda_config(&m)?;
        crate::executor::cuda_startup_log(
            "qwen35.config",
            config_t0,
            format_args!(
                "layers={} hidden={} moe={} model_path={}",
                m.num_hidden_layers,
                m.hidden_size,
                m.is_moe(),
                model_path.display()
            ),
        );
        // Full attention here is the GATED q_proj variant (Qwen3.5/3.6); the
        // prep+gate kernels assume it. Vanilla un-gated Qwen3 would need
        // the dense path, not this loader.
        ensure!(
            m.full_attn_gated,
            "clean CUDA Qwen3.5 hybrid path expects the gated full-attention q_proj \
             (Qwen3.5/3.6); un-gated Qwen3 uses from_qwen3_bf16_safetensors"
        );
        ensure!(
            m.rope_scaling.is_none(),
            "Qwen3.5 rope_scaling is set but the YaRN bridge is not wired into the \
             clean hybrid RoPE precompute; refusing to silently drop it (pod follow-up)"
        );
        ensure!(
            max_seq_len > 0,
            "Qwen3.5 hybrid model requires a non-zero KV cache budget"
        );

        let tp_cfg = *tp.config();
        let world = tp_cfg.world_size;
        // Attention (full + linear) weights shard over the mesh's attn_tp axis
        // and REPLICATE across attn_cp (cp peers need identical head shards for
        // the CP-prefill KV all-gather + GDN state relay). attn_cp=1 makes this
        // exactly `tp_cfg`, so raw-TP sharding is byte-identical.
        let attn_cfg = infer_topo::TpConfig {
            world_size: tp.attn_tp_size(),
            rank: tp.attn_tp_rank(),
        };
        // Per-rank full-attn GQA head counts. `head_shard` shards KV when
        // num_kv_heads >= world (e.g. Qwen3.6-35B kv=8 @ TP8 -> 1/rank) and
        // REPLICATES when num_kv_heads < world (Qwen3.5-122B kv=2 @ TP4 -> every
        // rank holds 1 replicated KV head + its Q-head shard). Replicas load
        // identical K/V weights (`kv_load_block_index`) so each computes GQA
        // independently; the divisible case stays byte-identical.
        let (local_q_heads, local_kv_heads) = if attn_cfg.is_single() {
            (m.num_attention_heads, m.num_key_value_heads)
        } else {
            infer_topo::head_shard(m.num_attention_heads, m.num_key_value_heads, &attn_cfg)
                .map_err(|e| anyhow!("Qwen3.5 TP full-attention head shard failed: {e}"))?
        };
        // KV-head block index this rank loads (== attn-tp rank in the shard
        // regime; shared within a replica group in the replication regime). Q
        // always partitions by `attn_cfg.rank`.
        let kv_block = if attn_cfg.is_single() {
            0
        } else {
            infer_topo::kv_load_block_index(m.num_key_value_heads, &attn_cfg)
                .map_err(|e| anyhow!("Qwen3.5 TP full-attention KV block index: {e}"))?
        };
        // Gated-delta head counts. The linear (gated-delta) heads are large
        // (Qwen3.5/3.6: Kh=16, Vh=32), so they SHARD cleanly at the TP sizes that
        // need full-attn KV replication (122B @ TP4: 16->4, 32->8). We keep the
        // strict divisibility contract here — a contiguous head-major block shard
        // preserves the v-per-k grouping (gated_delta_rule.cu maps
        // k_head = v_head * Kh / Vh): each rank's v-head range reads exactly its
        // own k-head range. Linear-head replication would need a different shard
        // (the k/v grouping can't be split by replica), so reject it loudly
        // rather than silently mis-shard.
        let attn_world = attn_cfg.world_size;
        ensure!(
            m.linear_num_key_heads.is_multiple_of(attn_world),
            "Qwen3.5 TP: linear_num_key_heads ({}) not divisible by attn_tp ({attn_world}) \
             — gated-delta linear heads must shard (replication unsupported on the linear path)",
            m.linear_num_key_heads
        );
        ensure!(
            m.linear_num_value_heads.is_multiple_of(attn_world),
            "Qwen3.5 TP: linear_num_value_heads ({}) not divisible by attn_tp ({attn_world}) \
             — gated-delta linear heads must shard (replication unsupported on the linear path)",
            m.linear_num_value_heads
        );
        let local_linear_k_heads = m.linear_num_key_heads / attn_world;
        let local_linear_v_heads = m.linear_num_value_heads / attn_world;
        // Shared expert is column/row-sharded like a dense MLP (its partial
        // joins the routed partial in one post-MoE all-reduce).
        ensure!(
            m.shared_expert_intermediate_size.is_multiple_of(world),
            "Qwen3.5 TP: shared_expert_intermediate_size ({}) not divisible by world_size ({world})",
            m.shared_expert_intermediate_size
        );
        // Dense-MLP layers (mlp_only_layers / sparse-step gaps) shard their
        // intermediate dim; only constrain it when such a layer exists.
        if (0..m.num_hidden_layers).any(|i| !m.is_moe_layer(i)) {
            ensure!(
                m.intermediate_size.is_multiple_of(world),
                "Qwen3.5 TP: dense intermediate_size ({}) not divisible by world_size ({world})",
                m.intermediate_size
            );
        }

        let moe_config = if m.is_moe() {
            Some(crate::moe_config::moe_config_from_qwen35(&m)?)
        } else {
            None
        };
        // EP mirrors TP for MoE: each rank owns `num_experts / world` whole
        // experts (`ExpertSplit::new` rejects an indivisible expert count
        // loudly). Dense Qwen3.5 has no expert-owned buffers; keep an inert
        // split so the struct layout stays uniform.
        let split = if !m.is_moe() {
            ExpertSplit::single(0)
        } else if tp_cfg.is_single() {
            ExpertSplit::single(m.num_experts)
        } else {
            ExpertSplit::new(m.num_experts, world, tp_cfg.rank)
                .map_err(|e| anyhow!("Qwen3.5 TP expert split: {e}"))?
        };

        let loader_t0 = Instant::now();
        let ctx = DeviceContext::new()?;
        // One-shot small-message collectives (default-on, loud auto-degrade).
        // COLLECTIVE boot — identical construction point on every rank.
        #[cfg(feature = "nccl")]
        tp.init_oneshot_comm(&ctx);
        let loader = SafetensorLoader::new(model_path)?;
        crate::executor::cuda_startup_log("qwen35.ctx_loader", loader_t0, format_args!(""));

        let embed_t0 = Instant::now();
        let embed_tokens = loader.load_matrix(&ctx, m.embed_tokens_tensor_name())?;
        let lm_head = if m.tie_word_embeddings {
            None
        } else {
            Some(loader.load_output_head_quant_aware(&ctx, m.lm_head_tensor_name())?)
        };
        crate::executor::cuda_startup_log(
            "qwen35.embeddings",
            embed_t0,
            format_args!("tie_word_embeddings={}", m.tie_word_embeddings),
        );

        let mut layers = Vec::with_capacity(m.num_hidden_layers);
        for layer_idx in 0..m.num_hidden_layers {
            let layer_t0 = Instant::now();
            let names = m.layer_tensor_names(layer_idx);
            let attn_t0 = Instant::now();
            let attn = match &names.attention {
                Qwen35AttentionTensorNames::Full(full) if attn_cfg.is_single() => {
                    Qwen35Attn::Full(Box::new(FullAttn {
                        qkv_proj: loader.load_matrices_row_fused(
                            &ctx,
                            &[
                                (full.q_proj.as_str(), None),
                                (full.k_proj.as_str(), None),
                                (full.v_proj.as_str(), None),
                            ],
                        )?,
                        o_proj: loader.load_dense_matrix_quant_aware(&ctx, &full.o_proj)?,
                        q_norm: loader.load_vec(&ctx, &full.q_norm)?,
                        k_norm: loader.load_vec(&ctx, &full.k_norm)?,
                        decode: None,
                    }))
                }
                Qwen35AttentionTensorNames::Full(full) => Qwen35Attn::Full(Box::new(FullAttn {
                    // The GATED q_proj interleaves [query(HD); gate(HD)] PER
                    // HEAD (prefill_attention_hd256.cu reads q at
                    // `head*2*HD + d`, the gate kernel at `head*2*HD + HD + d`),
                    // so a whole-head slice with per-head row block 2*head_dim
                    // carries each head's query rows AND its matching gate rows.
                    // Q partitions by rank; K/V load the replica-aware KV block
                    // (== rank in the shard regime; shared within a replica group
                    // when kv_heads < world_size, giving identical K/V weights).
                    qkv_proj: {
                        let head_spec = |name: &str, local_rows: usize, block: usize| {
                            let total = loader.logical_rows(name)?;
                            Ok::<_, anyhow::Error>(infer_topo::ShardingSpec {
                                offset: block * local_rows,
                                size: local_rows,
                                total,
                            })
                        };
                        let q_spec =
                            head_spec(&full.q_proj, local_q_heads * m.head_dim * 2, attn_cfg.rank)?;
                        let k_spec =
                            head_spec(&full.k_proj, local_kv_heads * m.head_dim, kv_block)?;
                        let v_spec =
                            head_spec(&full.v_proj, local_kv_heads * m.head_dim, kv_block)?;
                        loader.load_matrices_row_fused(
                            &ctx,
                            &[
                                (full.q_proj.as_str(), Some(q_spec)),
                                (full.k_proj.as_str(), Some(k_spec)),
                                (full.v_proj.as_str(), Some(v_spec)),
                            ],
                        )?
                    },
                    o_proj: loader.load_matrix_sharded_quant_aware(
                        &ctx,
                        &full.o_proj,
                        infer_topo::ParallelLinearKind::Row,
                        &attn_cfg,
                    )?,
                    // q/k_norm are `[head_dim]`, broadcast across heads by the
                    // full-attention prep kernel — replicated.
                    q_norm: loader.load_vec(&ctx, &full.q_norm)?,
                    k_norm: loader.load_vec(&ctx, &full.k_norm)?,
                    decode: None,
                })),
                Qwen35AttentionTensorNames::Linear(lin) if attn_cfg.is_single() => {
                    Qwen35Attn::Linear(Box::new(LinearAttn {
                        in_proj_qkvz: loader.load_matrix_pair_fused(
                            &ctx,
                            &lin.in_proj_qkv,
                            &lin.in_proj_z,
                        )?,
                        in_proj_ba: loader.load_matrix_pair_fused(
                            &ctx,
                            &lin.in_proj_b,
                            &lin.in_proj_a,
                        )?,
                        conv1d_weight: loader.load_conv1d_vec(&ctx, &lin.conv1d_weight)?,
                        dt_bias: loader.load_vec_any(&ctx, &lin.dt_bias)?,
                        a_log: loader.load_f32_vec(&ctx, &lin.a_log)?,
                        norm_weight: loader.load_f32_vec(&ctx, &lin.norm)?,
                        out_proj: loader.load_dense_matrix_quant_aware(&ctx, &lin.out_proj)?,
                        decode: None,
                    }))
                }
                Qwen35AttentionTensorNames::Linear(lin) => {
                    Qwen35Attn::Linear(Box::new(LinearAttn {
                        // Fused [q | k | v] blocks: shard EACH block on whole-head
                        // boundaries and re-stack this rank's three slices (a flat
                        // column shard would cut across the block boundaries).
                        in_proj_qkvz: {
                            let qkv = load_linear_qkv_sharded(
                                &loader,
                                &ctx,
                                &lin.in_proj_qkv,
                                &m,
                                &attn_cfg,
                            )?;
                            // z gate is v-head-major `[Vh*Vd]` (rms_norm_gated
                            // reads the gate at `head*Vd + d`).
                            let z = loader.load_qkv_head_sharded_quant_aware(
                                &ctx,
                                &lin.in_proj_z,
                                local_linear_v_heads,
                                m.linear_value_head_dim,
                                attn_cfg.rank,
                            )?;
                            DeviceMatrix::fuse_rows(&ctx, &qkv, &z)
                                .map_err(|e| anyhow!("fuse TP in_proj_qkv + in_proj_z: {e}"))?
                        },
                        // b/a are ONE SCALAR PER V HEAD (gated_delta_rule.cu reads
                        // `b_proj[token*Vh + v_head]`) → per-head row count 1;
                        // the local head shards row-fuse into one `[2*Vh, H]`.
                        in_proj_ba: {
                            let b = loader.load_qkv_head_sharded(
                                &ctx,
                                &lin.in_proj_b,
                                local_linear_v_heads,
                                1,
                                attn_cfg.rank,
                            )?;
                            let a = loader.load_qkv_head_sharded(
                                &ctx,
                                &lin.in_proj_a,
                                local_linear_v_heads,
                                1,
                                attn_cfg.rank,
                            )?;
                            DeviceMatrix::fuse_rows(&ctx, &b, &a)?
                        },
                        conv1d_weight: load_conv1d_sharded(
                            &loader,
                            &ctx,
                            &lin.conv1d_weight,
                            &m,
                            &attn_cfg,
                        )?,
                        dt_bias: load_v_head_vec_sharded(
                            &loader,
                            &ctx,
                            &lin.dt_bias,
                            m.linear_num_value_heads,
                            &attn_cfg,
                        )?,
                        a_log: load_v_head_f32_sharded(
                            &loader,
                            &ctx,
                            &lin.a_log,
                            m.linear_num_value_heads,
                            &attn_cfg,
                        )?,
                        // Gated-norm scale is `[Vd]`, broadcast across heads by
                        // rms_norm_gated (norm.cu `weight[tid]`) — replicated,
                        // matching the qwen35-spec Shard contract.
                        norm_weight: loader.load_f32_vec(&ctx, &lin.norm)?,
                        out_proj: loader.load_matrix_sharded_quant_aware(
                            &ctx,
                            &lin.out_proj,
                            infer_topo::ParallelLinearKind::Row,
                            &attn_cfg,
                        )?,
                        decode: None,
                    }))
                }
            };
            crate::executor::cuda_startup_log(
                "qwen35.layer.attn",
                attn_t0,
                format_args!("layer={layer_idx} type={:?}", m.layer_types[layer_idx]),
            );

            let ffn_t0 = Instant::now();
            let (mlp, moe) = if m.is_moe_layer(layer_idx) {
                let moe = loader.load_moe_layer_experts(
                    &ctx,
                    &names.common.moe_tensor_names(),
                    &split,
                    &tp_cfg,
                    m.moe_intermediate_size,
                    m.hidden_size,
                )?;
                (None, Some(moe))
            } else if tp_cfg.is_single() {
                (
                    Some(DenseMlp {
                        gate_up_proj: loader.load_matrix_pair_fused(
                            &ctx,
                            &names.common.mlp_gate_proj,
                            &names.common.mlp_up_proj,
                        )?,
                        down_proj: loader
                            .load_dense_matrix_quant_aware(&ctx, &names.common.mlp_down_proj)?,
                    }),
                    None,
                )
            } else {
                (
                    Some(DenseMlp {
                        gate_up_proj: loader.load_matrix_pair_fused_column_sharded(
                            &ctx,
                            &names.common.mlp_gate_proj,
                            &names.common.mlp_up_proj,
                            &tp_cfg,
                        )?,
                        down_proj: loader.load_matrix_sharded_quant_aware(
                            &ctx,
                            &names.common.mlp_down_proj,
                            infer_topo::ParallelLinearKind::Row,
                            &tp_cfg,
                        )?,
                    }),
                    None,
                )
            };
            crate::executor::cuda_startup_log(
                "qwen35.layer.ffn",
                ffn_t0,
                format_args!("layer={layer_idx} moe={}", m.is_moe_layer(layer_idx)),
            );

            layers.push(Qwen35Layer {
                input_layernorm: loader.load_vec(&ctx, &names.common.input_layernorm)?,
                attn,
                post_attention_layernorm: loader
                    .load_vec(&ctx, &names.common.post_attention_layernorm)?,
                mlp,
                moe,
            });
            crate::executor::cuda_startup_log(
                "qwen35.layer.total",
                layer_t0,
                format_args!("layer={layer_idx} moe={}", m.is_moe_layer(layer_idx)),
            );
        }
        // B2 CP decode (T3.1): a second, finer-sharded copy of the attention
        // weights (1/(attn_tp x cp) heads per rank) for the decode head-shard.
        // The primary load above is untouched; when cp=1 the gate is false and
        // `decode` stays None, so the baseline is byte-identical.
        let cp_size = tp.attn_cp_size();
        if cp_size > 1
            && local_q_heads.is_multiple_of(cp_size)
            && local_kv_heads.is_multiple_of(cp_size)
            && local_linear_k_heads.is_multiple_of(cp_size)
            && local_linear_v_heads.is_multiple_of(cp_size)
        {
            // decode_rank = attn_tp_rank*cp + attn_cp_rank enumerates the
            // attn_tp x cp product (attn_dp=1 under CP) as the head-block index
            // each rank computes: its attn_tp shard's cp sub-block, contiguous.
            let decode_attn_cfg = infer_topo::TpConfig {
                world_size: attn_cfg.world_size * cp_size,
                rank: attn_cfg.rank * cp_size + tp.attn_cp_rank(),
            };
            let decode_q = local_q_heads / cp_size;
            let decode_kv = local_kv_heads / cp_size;
            let decode_lk = local_linear_k_heads / cp_size;
            let decode_lv = local_linear_v_heads / cp_size;
            debug_assert_eq!(
                m.linear_num_key_heads / decode_attn_cfg.world_size,
                decode_lk
            );
            let decode_t0 = Instant::now();
            for (layer_idx, layer) in layers.iter_mut().enumerate() {
                let names = m.layer_tensor_names(layer_idx);
                match (&mut layer.attn, &names.attention) {
                    (Qwen35Attn::Full(full), Qwen35AttentionTensorNames::Full(fnames)) => {
                        full.decode = Some(load_full_attn_decode(
                            &loader,
                            &ctx,
                            fnames,
                            &m,
                            &decode_attn_cfg,
                            decode_q,
                            decode_kv,
                        )?);
                    }
                    (Qwen35Attn::Linear(lin), Qwen35AttentionTensorNames::Linear(lnames)) => {
                        lin.decode = Some(load_linear_attn_decode(
                            &loader,
                            &ctx,
                            lnames,
                            &m,
                            &decode_attn_cfg,
                            decode_lv,
                            &lin.conv1d_weight,
                            local_linear_k_heads,
                            decode_lk,
                            tp.attn_cp_rank(),
                        )?);
                    }
                    _ => unreachable!("layer attn kind matches its tensor names"),
                }
            }
            crate::executor::cuda_startup_log(
                "qwen35.cp_decode_weights",
                decode_t0,
                format_args!(
                    "cp={cp_size} decode_world={} decode_q={decode_q} decode_kv={decode_kv} \
                     decode_lv={decode_lv}",
                    decode_attn_cfg.world_size
                ),
            );
        }
        let tail_t0 = Instant::now();
        let norm = loader.load_vec(&ctx, m.norm_tensor_name())?;

        let rope_len = m
            .rope_cache_len_hint()
            .unwrap_or(DEFAULT_ROPE_CACHE_LEN)
            .max(DEFAULT_ROPE_CACHE_LEN);
        ensure!(
            max_seq_len <= rope_len,
            "Qwen3.5 max_seq_len ({max_seq_len}) exceeds the RoPE cache length ({rope_len}); \
             positions beyond the table would read out of bounds"
        );
        // PARTIAL RoPE: the table must be built over `rotary_dim` (= head_dim ×
        // partial_rotary_factor, 64 on Qwen3.6), not head_dim — the prep
        // kernel indexes `cos_cache[pos * rotary_dim + d]` and expects inv_freq
        // computed over rotary_dim dims (`precompute_rope` is generic over its
        // dim arg and emits the half-duplicated stride-dim layout it reads).
        let (cos_cache, sin_cache) =
            crate::ops::precompute_rope(&ctx, m.rotary_dim, rope_len, m.rope_theta, None)?;
        ctx.sync()?;
        crate::executor::cuda_startup_log(
            "qwen35.tail_norm_rope_sync",
            tail_t0,
            format_args!("rope_len={rope_len} max_seq_len={max_seq_len}"),
        );
        crate::executor::cuda_startup_log(
            "qwen35.total",
            total_t0,
            format_args!("layers={} max_seq_len={max_seq_len}", m.num_hidden_layers),
        );

        // NextN-MTP draft head (speculative decode). Loaded only on request; the
        // default decode path never touches it, so the baseline stays
        // byte-identical when off. Single-GPU only for now — the head shares the
        // base lm_head/embed and the 27B-FP8 fits on one H20; TP-sharded MTP is a
        // follow-up once spec-decode proves out single-GPU.
        let mtp = if mtp_draft_tokens.is_some() {
            let mtp_t0 = Instant::now();
            ensure!(
                tp_cfg.is_single(),
                "Qwen3.5 MTP spec-decode is single-GPU only for now \
                 (TP-sharded MTP draft head not yet wired)"
            );
            let head = load_qwen35_mtp_head(&loader, &ctx, &m, &split, &tp_cfg)?;
            crate::executor::cuda_startup_log(
                "qwen35.mtp_head",
                mtp_t0,
                format_args!("draft_tokens={}", mtp_draft_tokens.unwrap_or(0)),
            );
            Some(head)
        } else {
            None
        };
        let spec_draft_tokens = mtp_draft_tokens.unwrap_or(0);

        Ok(Self {
            ctx,
            config: m,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            moe_config,
            tp,
            local_q_heads,
            local_kv_heads,
            local_linear_k_heads,
            local_linear_v_heads,
            expert_split: split,
            max_seq_len,
            mtp,
            spec_draft_tokens,
            offloaded: None,
            frozen_base_ptrs_exported: AtomicBool::new(false),
            lora_delta_scratch: None,
            lora_dirty: HashSet::new(),
            lora_base_dev: HashMap::new(),
            weights_epoch: kv_native_sys::weights_epoch_tag(model_path),
        })
    }

    #[allow(dead_code)] // WIP: durable KV-recall manifest weight-version stamp, not yet wired
    pub(crate) fn weights_epoch(&self) -> &str {
        &self.weights_epoch
    }

    pub(crate) fn offload_engine_weights(&mut self) -> Result<usize> {
        if self.offloaded.is_some() {
            return Ok(0);
        }
        ensure!(
            !self.frozen_base_ptrs_exported.load(Ordering::Relaxed),
            "cannot offload engine weights: frozen-base device pointers are exported to the \
             trainer; offloading would free aliased memory"
        );
        let ctx = self.ctx.clone();
        // Drain ALL in-flight GPU work before snapshotting. The OPD step has
        // co-resident allocators (infer-teacher + train autograd) sharing one
        // device/pool on separate streams; a full synchronize quiesces every
        // stream so the D2H snapshot and the subsequent block frees do not race
        // other-stream allocations from the shared async pool.
        ctx.sync()?;
        let mut freed = 0usize;

        let embed_tokens = self.embed_tokens.offload_to_host(&ctx)?;
        freed += embed_tokens.freed_bytes();
        let lm_head = match self.lm_head.as_mut() {
            Some(head) => {
                let snap = head.offload_to_host(&ctx)?;
                freed += snap.freed_bytes();
                Some(snap)
            }
            None => None,
        };
        let (norm, norm_n) = self.norm.offload_to_host(&ctx)?;
        freed += norm_n;

        let mut blocks = Vec::with_capacity(self.layers.len());
        for layer in &mut self.layers {
            let (input_layernorm, in_ln_n) = layer.input_layernorm.offload_to_host(&ctx)?;
            let (post_attention_layernorm, post_ln_n) =
                layer.post_attention_layernorm.offload_to_host(&ctx)?;
            freed += in_ln_n + post_ln_n;

            let mlp = match layer.mlp.as_mut() {
                Some(dense) => {
                    let gate_up_proj = dense.gate_up_proj.offload_to_host(&ctx)?;
                    let down_proj = dense.down_proj.offload_to_host(&ctx)?;
                    freed += gate_up_proj.freed_bytes() + down_proj.freed_bytes();
                    Some(OffloadedDenseMlp {
                        gate_up_proj,
                        down_proj,
                    })
                }
                None => None,
            };
            let moe = match layer.moe.as_mut() {
                Some(moe) => {
                    let snap = moe.offload_to_host(&ctx)?;
                    freed += snap.freed_bytes();
                    Some(snap)
                }
                None => None,
            };

            let attn = match &mut layer.attn {
                Qwen35Attn::Full(full) => {
                    let qkv_proj = full.qkv_proj.offload_to_host(&ctx)?;
                    let o_proj = full.o_proj.offload_to_host(&ctx)?;
                    let (q_norm, qn) = full.q_norm.offload_to_host(&ctx)?;
                    let (k_norm, kn) = full.k_norm.offload_to_host(&ctx)?;
                    freed += qkv_proj.freed_bytes() + o_proj.freed_bytes() + qn + kn;
                    OffloadedAttn::Full(Box::new(OffloadedFullAttn {
                        qkv_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                    }))
                }
                Qwen35Attn::Linear(lin) => {
                    let in_proj_qkvz = lin.in_proj_qkvz.offload_to_host(&ctx)?;
                    let in_proj_ba = lin.in_proj_ba.offload_to_host(&ctx)?;
                    let (conv1d_weight, conv_n) = lin.conv1d_weight.offload_to_host(&ctx)?;
                    let (dt_bias, dt_n) = lin.dt_bias.offload_to_host(&ctx)?;
                    let (a_log, al) = offload_raw_slice(&ctx, &mut lin.a_log)?;
                    let (norm_weight, nw) = offload_raw_slice(&ctx, &mut lin.norm_weight)?;
                    let out_proj = lin.out_proj.offload_to_host(&ctx)?;
                    freed += in_proj_qkvz.freed_bytes()
                        + in_proj_ba.freed_bytes()
                        + out_proj.freed_bytes()
                        + conv_n
                        + dt_n
                        + al
                        + nw;
                    OffloadedAttn::Linear(Box::new(OffloadedLinearAttn {
                        in_proj_qkvz,
                        in_proj_ba,
                        conv1d_weight,
                        dt_bias,
                        a_log,
                        norm_weight,
                        out_proj,
                    }))
                }
            };

            blocks.push(OffloadedBlock {
                input_layernorm,
                post_attention_layernorm,
                attn,
                mlp,
                moe,
            });
        }

        // Quiesce again after the block frees so reload (or a co-resident
        // backward) sees a settled pool. Trim the pool to the OS so the freed
        // VRAM is reusable for the co-resident autograd student forward (which
        // allocates from the same device default async pool).
        ctx.sync()?;
        let (free_before, _) = ctx.mem_info_bytes()?;
        ctx.trim_memory_pool()?;
        let (free_after, _) = ctx.mem_info_bytes()?;
        eprintln!(
            "[offload] freed={freed} bytes, free_before={free_before}, free_after={free_after}"
        );

        self.offloaded = Some(Box::new(OffloadedWeights {
            embed_tokens,
            lm_head,
            norm,
            blocks,
        }));
        Ok(freed)
    }

    pub(crate) fn reload_engine_weights(&mut self) -> Result<()> {
        let Some(snapshot) = self.offloaded.take() else {
            return Ok(());
        };
        let ctx = self.ctx.clone();
        // Quiesce the whole device before re-allocating weight VRAM so the H2D
        // restores do not race the train/optimizer allocations still draining
        // from the shared async pool (see offload note).
        ctx.sync()?;
        let OffloadedWeights {
            embed_tokens,
            lm_head,
            norm,
            blocks,
        } = *snapshot;

        self.embed_tokens.reload_from_host(&ctx, &embed_tokens)?;
        match (self.lm_head.as_mut(), &lm_head) {
            (Some(head), Some(snap)) => head.reload_from_host(&ctx, snap)?,
            (None, None) => {}
            _ => anyhow::bail!("offload/reload lm_head presence mismatch"),
        }
        self.norm.reload_from_host(&ctx, &norm)?;

        ensure!(
            blocks.len() == self.layers.len(),
            "offload/reload layer count mismatch: snapshot {} vs model {}",
            blocks.len(),
            self.layers.len()
        );
        for (layer, block) in self.layers.iter_mut().zip(blocks) {
            let OffloadedBlock {
                input_layernorm,
                post_attention_layernorm,
                attn,
                mlp,
                moe,
            } = block;
            layer
                .input_layernorm
                .reload_from_host(&ctx, &input_layernorm)?;
            layer
                .post_attention_layernorm
                .reload_from_host(&ctx, &post_attention_layernorm)?;

            match (layer.mlp.as_mut(), layer.moe.as_mut(), mlp, moe) {
                (Some(dense), None, Some(snap), None) => {
                    dense
                        .gate_up_proj
                        .reload_from_host(&ctx, &snap.gate_up_proj)?;
                    dense.down_proj.reload_from_host(&ctx, &snap.down_proj)?;
                }
                (None, Some(moe), None, Some(snap)) => {
                    moe.reload_from_host(&ctx, &snap)?;
                }
                _ => anyhow::bail!("offload/reload MLP/MoE presence mismatch"),
            }

            match (&mut layer.attn, attn) {
                (Qwen35Attn::Full(full), OffloadedAttn::Full(snap)) => {
                    let OffloadedFullAttn {
                        qkv_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                    } = *snap;
                    full.qkv_proj.reload_from_host(&ctx, &qkv_proj)?;
                    full.o_proj.reload_from_host(&ctx, &o_proj)?;
                    full.q_norm.reload_from_host(&ctx, &q_norm)?;
                    full.k_norm.reload_from_host(&ctx, &k_norm)?;
                }
                (Qwen35Attn::Linear(lin), OffloadedAttn::Linear(snap)) => {
                    let OffloadedLinearAttn {
                        in_proj_qkvz,
                        in_proj_ba,
                        conv1d_weight,
                        dt_bias,
                        a_log,
                        norm_weight,
                        out_proj,
                    } = *snap;
                    lin.in_proj_qkvz.reload_from_host(&ctx, &in_proj_qkvz)?;
                    lin.in_proj_ba.reload_from_host(&ctx, &in_proj_ba)?;
                    lin.conv1d_weight.reload_from_host(&ctx, &conv1d_weight)?;
                    lin.dt_bias.reload_from_host(&ctx, &dt_bias)?;
                    reload_raw_slice(&ctx, &mut lin.a_log, &a_log)?;
                    reload_raw_slice(&ctx, &mut lin.norm_weight, &norm_weight)?;
                    lin.out_proj.reload_from_host(&ctx, &out_proj)?;
                }
                _ => anyhow::bail!("offload/reload attention-kind mismatch"),
            }
        }
        ctx.sync()?;
        Ok(())
    }
}

fn linear_qkv_head_blocks(m: &Qwen35Config) -> [crate::shard_slice::HeadBlock; 3] {
    let k_block = crate::shard_slice::HeadBlock {
        heads: m.linear_num_key_heads,
        head_rows: m.linear_key_head_dim,
    };
    let v_block = crate::shard_slice::HeadBlock {
        heads: m.linear_num_value_heads,
        head_rows: m.linear_value_head_dim,
    };
    [k_block, k_block, v_block]
}

fn load_linear_qkv_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    m: &Qwen35Config,
    tp: &TpConfig,
) -> Result<DeviceMatrix> {
    let head_blocks = linear_qkv_head_blocks(m);
    // FP8 block-scaled checkpoints (e.g. Qwen3.6-27B-FP8) carry the fused qkv as
    // F8_E4M3 + a `weight_scale_inv` sidecar; shard both with the same head-block
    // helper as the BF16 path. `None` → no quant view, keep the BF16 path below
    // byte-for-byte. head_dim == block_m makes head-block boundaries land on
    // scale-row boundaries, so the 3-block re-stack mirrors 1:1 in scale units.
    if let Some(matrix) = loader.load_linear_qkv_fp8_head_sharded(ctx, name, &head_blocks, tp)? {
        return Ok(matrix);
    }
    // GPTQ/W4A16 and other packed-quant formats: the fused qkv weight lives at
    // `{name}.qweight` (no `.weight`). For TP=1 the whole matrix loads as one
    // quant view; head-block sharding for TP>1 is not yet implemented for
    // packed quant (falls through to the BF16 path, which errors clearly).
    if tp.world_size == 1 && loader.quant_view_for(name)?.is_some() {
        return loader.load_dense_matrix_quant_aware(ctx, name);
    }
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.dtype == Dtype::BF16,
        "{name}: expected BF16 fused qkv projection, got {:?}",
        tensor.dtype
    );
    ensure!(
        tensor.shape.len() == 2 && tensor.shape[0] == m.linear_attn_qkv_dim(),
        "{name}: expected [{}, hidden] fused qkv projection, got shape {:?}",
        m.linear_attn_qkv_dim(),
        tensor.shape
    );
    let sharded = crate::shard_slice::shard_head_blocks_column_parallel(
        &tensor.bytes,
        tensor.shape[1],
        2,
        &head_blocks,
        tp,
    )?;
    DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
        .map_err(|e| anyhow!("upload sharded fused qkv {name}: {e}"))
}

/// B2 CP decode subset of a full-attention layer: the q/k/v rows and o_proj
/// cols for this rank's 1/(attn_tp x cp) head block, via the same quant-aware
/// sharded path as the primary TP load (W8A16/Marlin preserved).
fn load_full_attn_decode(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    full: &qwen35_spec::Qwen35FullAttentionTensorNames,
    m: &Qwen35Config,
    decode_cfg: &TpConfig,
    decode_q: usize,
    decode_kv: usize,
) -> Result<FullAttnDecode> {
    let head_spec =
        |name: &str, local_rows: usize, block: usize| -> Result<infer_topo::ShardingSpec> {
            let total = loader.logical_rows(name)?;
            Ok(infer_topo::ShardingSpec {
                offset: block * local_rows,
                size: local_rows,
                total,
            })
        };
    // The engage guard (local_kv % cp == 0) puts the decode world in the KV-shard
    // regime, so kv_block == decode_cfg.rank; kv_load_block_index is the robust form.
    let kv_block = infer_topo::kv_load_block_index(m.num_key_value_heads, decode_cfg)?;
    let q_spec = head_spec(&full.q_proj, decode_q * m.head_dim * 2, decode_cfg.rank)?;
    let k_spec = head_spec(&full.k_proj, decode_kv * m.head_dim, kv_block)?;
    let v_spec = head_spec(&full.v_proj, decode_kv * m.head_dim, kv_block)?;
    let qkv_proj = loader.load_matrices_row_fused(
        ctx,
        &[
            (full.q_proj.as_str(), Some(q_spec)),
            (full.k_proj.as_str(), Some(k_spec)),
            (full.v_proj.as_str(), Some(v_spec)),
        ],
    )?;
    let o_proj = loader.load_matrix_sharded_quant_aware(
        ctx,
        &full.o_proj,
        infer_topo::ParallelLinearKind::Row,
        decode_cfg,
    )?;
    Ok(FullAttnDecode { qkv_proj, o_proj })
}

/// B2 CP decode subset of a linear-attention layer: the qkvz/ba rows,
/// out_proj cols, and a compact `[qkv_dim', K]` conv1d weight for this rank's
/// 1/(attn_tp x cp) v-head block. dt_bias / a_log are head-indexed by the
/// decode kernel, which offsets into the primary buffers, so they need no
/// second copy; the conv weight's subset channels are three disjoint blocks,
/// so it gets a compact copy.
#[allow(clippy::too_many_arguments)]
fn load_linear_attn_decode(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    lin: &qwen35_spec::Qwen35LinearAttentionTensorNames,
    m: &Qwen35Config,
    decode_cfg: &TpConfig,
    decode_lv: usize,
    full_conv1d: &DeviceVec,
    local_lk: usize,
    decode_lk: usize,
    cp_rank: usize,
) -> Result<LinearAttnDecode> {
    let qkv = load_linear_qkv_sharded(loader, ctx, &lin.in_proj_qkv, m, decode_cfg)?;
    let z = loader.load_qkv_head_sharded_quant_aware(
        ctx,
        &lin.in_proj_z,
        decode_lv,
        m.linear_value_head_dim,
        decode_cfg.rank,
    )?;
    let in_proj_qkvz = DeviceMatrix::fuse_rows(ctx, &qkv, &z)
        .map_err(|e| anyhow!("fuse CP-decode in_proj_qkv + in_proj_z: {e}"))?;
    let b = loader.load_qkv_head_sharded(ctx, &lin.in_proj_b, decode_lv, 1, decode_cfg.rank)?;
    let a = loader.load_qkv_head_sharded(ctx, &lin.in_proj_a, decode_lv, 1, decode_cfg.rank)?;
    let in_proj_ba = DeviceMatrix::fuse_rows(ctx, &b, &a)?;
    let out_proj = loader.load_matrix_sharded_quant_aware(
        ctx,
        &lin.out_proj,
        infer_topo::ParallelLinearKind::Row,
        decode_cfg,
    )?;
    // Compact conv weight: the subset's q/k/v channel blocks are disjoint in
    // the primary `[local_qkv, K]` weight, so three D2D copies restack them
    // into `[qkv_dim', K]` (channel-major, each channel's K rows contiguous).
    let (kd, vd, k) = (
        m.linear_key_head_dim,
        m.linear_value_head_dim,
        m.linear_conv_kernel_dim,
    );
    let qk_ch = decode_lk * kd;
    let v_ch = decode_lv * vd;
    let mut conv1d_weight = DeviceVec::zeros(ctx, 2 * qk_ch * k + v_ch * k)?;
    let copies: [(usize, usize, usize); 3] = [
        (cp_rank * qk_ch, 0, qk_ch),
        (local_lk * kd + cp_rank * qk_ch, qk_ch, qk_ch),
        (2 * local_lk * kd + cp_rank * v_ch, 2 * qk_ch, v_ch),
    ];
    for (src_ch, dst_ch, cnt) in copies {
        ctx.stream
            .memcpy_dtod(
                &full_conv1d.data.slice(src_ch * k..(src_ch + cnt) * k),
                &mut conv1d_weight.data.slice_mut(dst_ch * k..(dst_ch + cnt) * k),
            )
            .map_err(|e| anyhow!("compact B2 conv1d weight copy failed: {e}"))?;
    }
    Ok(LinearAttnDecode {
        in_proj_qkvz,
        in_proj_ba,
        out_proj,
        conv1d_weight,
    })
}

fn load_conv1d_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    m: &Qwen35Config,
    tp: &TpConfig,
) -> Result<DeviceVec> {
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.dtype == Dtype::BF16,
        "{name}: expected BF16 conv1d weight, got {:?}",
        tensor.dtype
    );
    let channels = tensor.shape.first().copied().unwrap_or(0);
    ensure!(
        channels == m.linear_attn_qkv_dim(),
        "{name}: conv1d channels {channels} != qkv_dim {} (shape {:?})",
        m.linear_attn_qkv_dim(),
        tensor.shape
    );
    // `[channels, 1, kernel]` (HF) or `[channels, kernel]`: the singleton middle
    // dim is squeezed by treating each channel's row as `kernel` elements.
    let kernel: usize = tensor.shape[1..].iter().product();
    ensure!(
        kernel == m.linear_conv_kernel_dim,
        "{name}: conv1d kernel {kernel} != linear_conv_kernel_dim {} (shape {:?})",
        m.linear_conv_kernel_dim,
        tensor.shape
    );
    let sharded = crate::shard_slice::shard_head_blocks_column_parallel(
        &tensor.bytes,
        kernel,
        2,
        &linear_qkv_head_blocks(m),
        tp,
    )?;
    DeviceVec::from_safetensors(ctx, &sharded.bytes)
        .map_err(|e| anyhow!("upload sharded conv1d {name}: {e}"))
}

fn load_v_head_vec_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    total_v_heads: usize,
    tp: &TpConfig,
) -> Result<DeviceVec> {
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.shape.len() == 1 && tensor.shape[0] == total_v_heads,
        "{name}: expected 1D [{total_v_heads}] per-v-head vector, got shape {:?}",
        tensor.shape
    );
    let bf16_bytes = SafetensorLoader::dsv4_bytes_to_bf16(name, &tensor)?;
    let (start, len) = v_head_shard_range(name, total_v_heads, tp)?;
    DeviceVec::from_safetensors(ctx, &bf16_bytes[start * 2..(start + len) * 2])
        .map_err(|e| anyhow!("upload sharded per-v-head vec {name}: {e}"))
}

fn load_v_head_f32_sharded(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    name: &str,
    total_v_heads: usize,
    tp: &TpConfig,
) -> Result<CudaSlice<f32>> {
    let tensor = loader.load_raw_tensor(name)?;
    ensure!(
        tensor.shape.len() == 1 && tensor.shape[0] == total_v_heads,
        "{name}: expected 1D [{total_v_heads}] per-v-head tensor, got shape {:?}",
        tensor.shape
    );
    let host: Vec<f32> = match tensor.dtype {
        Dtype::F32 => tensor
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::BF16 => tensor
            .bytes
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => anyhow::bail!("{name}: expected F32/BF16 1D tensor, got {other:?}"),
    };
    let (start, len) = v_head_shard_range(name, total_v_heads, tp)?;
    ctx.stream
        .clone_htod(&host[start..start + len])
        .map_err(|e| anyhow!("upload sharded per-v-head f32 {name}: {e}"))
}

fn load_qwen35_mtp_head(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    m: &Qwen35Config,
    split: &ExpertSplit,
    tp: &TpConfig,
) -> Result<Qwen35MtpHead> {
    let names = m.mtp_tensor_names();
    let Qwen35AttentionTensorNames::Full(full) = &names.layer.attention else {
        unreachable!("MTP head layer is always full attention");
    };
    let attn = Qwen35Attn::Full(Box::new(FullAttn {
        qkv_proj: loader.load_matrices_row_fused(
            ctx,
            &[
                (full.q_proj.as_str(), None),
                (full.k_proj.as_str(), None),
                (full.v_proj.as_str(), None),
            ],
        )?,
        o_proj: loader.load_dense_matrix_quant_aware(ctx, &full.o_proj)?,
        q_norm: loader.load_vec(ctx, &full.q_norm)?,
        k_norm: loader.load_vec(ctx, &full.k_norm)?,
        decode: None,
    }));
    let (mlp, moe) = if m.is_moe() {
        let moe = loader.load_moe_layer_experts(
            ctx,
            &names.layer.common.moe_tensor_names(),
            split,
            tp,
            m.moe_intermediate_size,
            m.hidden_size,
        )?;
        (None, Some(moe))
    } else {
        let mlp = DenseMlp {
            gate_up_proj: loader.load_matrix_pair_fused(
                ctx,
                &names.layer.common.mlp_gate_proj,
                &names.layer.common.mlp_up_proj,
            )?,
            down_proj: loader
                .load_dense_matrix_quant_aware(ctx, &names.layer.common.mlp_down_proj)?,
        };
        (Some(mlp), None)
    };
    let layer = Qwen35Layer {
        input_layernorm: loader.load_vec(ctx, &names.layer.common.input_layernorm)?,
        attn,
        post_attention_layernorm: loader
            .load_vec(ctx, &names.layer.common.post_attention_layernorm)?,
        mlp,
        moe,
    };
    Ok(Qwen35MtpHead {
        pre_fc_norm_embedding: loader.load_vec(ctx, &names.pre_fc_norm_embedding)?,
        pre_fc_norm_hidden: loader.load_vec(ctx, &names.pre_fc_norm_hidden)?,
        fc: loader.load_dense_matrix_quant_aware(ctx, &names.fc)?,
        layer,
        norm: loader.load_vec(ctx, &names.norm)?,
    })
}

fn v_head_shard_range(name: &str, total_v_heads: usize, tp: &TpConfig) -> Result<(usize, usize)> {
    ensure!(
        total_v_heads.is_multiple_of(tp.world_size),
        "{name}: {total_v_heads} v heads not divisible by world_size {}",
        tp.world_size
    );
    let local = total_v_heads / tp.world_size;
    Ok((tp.rank * local, local))
}
