use anyhow::{Context as _, bail};
use safetensors::SafeTensors;

use crate::loader::{float_elem_size, tensor_bytes_to_f32};
use crate::quant_format::{QuantFormat, ScaleApply};

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
            if !seen.insert(key) {
                return Ok(());
            }
            // NVFP4 has its own arm and its own kernel; without this it relies on
            // an FP8 weight of the same (m, n, k) having warmed a cubin it can
            // reuse, which is a property of this checkpoint, not of the format.
            if warm_fp4_deepgemm_dense(&self.ctx, weight, warm_m)? {
                warmed += 1;
                return Ok(());
            }
            if warm_fp8_deepgemm_dense(&self.ctx, weight, warm_m)? {
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
        // prep+gate kernels assume it. Vanilla un-gated Qwen3 has no CUDA path.
        ensure!(
            m.full_attn_gated,
            "clean CUDA Qwen3.5 hybrid path expects the gated full-attention q_proj \
             (Qwen3.5/3.6); un-gated Qwen3 dense is not supported on CUDA"
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
                            // Fuse first, repack once: fuse_rows reads the
                            // pre-repack buffers the Marlin repack releases.
                            let fused = DeviceMatrix::fuse_rows(&ctx, &qkv, &z)
                                .map_err(|e| anyhow!("fuse TP in_proj_qkv + in_proj_z: {e}"))?;
                            marlin_repack_dense(&ctx, &lin.in_proj_qkv, fused, true)?
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
        // Norm convention auto-detect: Qwen3.5/3.6 stores `weight - 1` (RMS ≈ 0),
        // standard RMSNorm stores `weight` (RMS ≈ 1). Check the first in-layer
        // norm — the final norm's weights are near 1 either way, so it can't
        // distinguish the conventions.
        if let Some(first_layer) = layers.first() {
            let host = first_layer.input_layernorm.to_host(&ctx)?;
            let rms = (host.iter().map(|&x| x * x).sum::<f32>() / host.len() as f32).sqrt();
            if rms > 0.5 {
                log::warn!(
                    "Qwen3.5 in-layer norm RMS={rms:.4} — looks like standard RMSNorm (w), \
                     but the kernel applies (1+w). Check the model's norm convention."
                );
            } else {
                log::info!(
                    "Qwen3.5 in-layer norm RMS={rms:.4} — (1+w) offset convention confirmed"
                );
            }
        }

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
        })
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

/// Returns the shard un-repacked — both callers row-fuse it with `in_proj_z`
/// and repack the fused matrix, because `fuse_rows` reads the buffers a repack
/// releases.
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
        return loader.load_matrix_quant_aware(ctx, name);
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
    // Fuse first, repack once: fuse_rows reads the pre-repack buffers the
    // Marlin repack releases.
    let in_proj_qkvz = DeviceMatrix::fuse_rows(ctx, &qkv, &z)
        .map_err(|e| anyhow!("fuse CP-decode in_proj_qkv + in_proj_z: {e}"))?;
    let in_proj_qkvz = marlin_repack_dense(ctx, &lin.in_proj_qkv, in_proj_qkvz, true)?;
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::BF16 => tensor
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
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

struct Fp8BlockProjectionView {
    weight_name: String,
    scale_name: String,
    rows: usize,
    cols: usize,
    scale_rows: usize,
    scale_cols: usize,
    scale_apply: ScaleApply,
}

struct DirectFp8MoeRouted {
    w13: MoeFp8ExpertGroup,
    down: MoeFp8ExpertGroup,
    gate_up_quant_signature: ExpertQuantDispatchSignature,
    down_quant_signature: ExpertQuantDispatchSignature,
}

/// Transpose each group's `[rows, cols]` row-major block-scale slab to `[cols, rows]`
/// in
/// place: maps the checkpoint's K-contiguous `weight_scale_inv` to the CUTLASS sm_120
/// N-contiguous SFB layout.
fn transpose_group_block_scales(scales: &mut [f32], groups: usize, rows: usize, cols: usize) {
    let per = rows * cols;
    debug_assert_eq!(scales.len(), groups * per);
    let mut tmp = vec![0f32; per];
    for g in 0..groups {
        let block = &mut scales[g * per..(g + 1) * per];
        for r in 0..rows {
            for c in 0..cols {
                tmp[c * rows + r] = block[r * cols + c];
            }
        }
        block.copy_from_slice(&tmp);
    }
}

impl SafetensorLoader {
    /// Quant-aware twin of the BF16 fused-qkv head shard: shard the F8_E4M3 weight AND
    /// its
    /// block-scale sidecar with the SAME head-block helper, or return None so the
    /// caller
    /// keeps its BF16 path.
    ///
    /// Scale rows map 1:1 to head-block rows only because `head_rows` is a whole
    /// multiple of
    /// `block_m`, so the blocks can be re-expressed in scale units and fed through the
    /// identical helper.
    pub(crate) fn load_linear_qkv_fp8_head_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        blocks: &[crate::shard_slice::HeadBlock],
        tp: &TpConfig,
    ) -> Result<Option<DeviceMatrix>> {
        let Some(view) = self.quant_view_for(name)? else {
            return Ok(None);
        };
        let QuantFormat::Fp8BlockScaled {
            block_m,
            block_k,
            scale_apply,
        } = view.format
        else {
            // A non-FP8 quant sidecar on the fused qkv would silently mis-shard here.
            return Ok(None);
        };
        ensure!(
            view.logical_shape.len() == 2,
            "{name}: expected 2D fused qkv FP8 matrix, got {:?}",
            view.logical_shape
        );
        let rows = view.logical_shape[0];
        let cols = view.logical_shape[1];

        let weight = self.borrow_raw_tensor(&view.name)?;
        ensure!(
            weight.dtype == Dtype::F8_E4M3 && weight.shape == view.logical_shape,
            "{name}: expected F8_E4M3 {:?}, got {:?} {:?}",
            view.logical_shape,
            weight.dtype,
            weight.shape
        );
        let scale = self.borrow_raw_tensor(&view.scale_names[0])?;
        let scale_elem = float_elem_size(&view.scale_names[0], scale.dtype)?;
        let scale_rows = rows.div_ceil(block_m);
        let scale_cols = cols.div_ceil(block_k);
        ensure!(
            scale.shape == [scale_rows, scale_cols],
            "{}: scale shape {:?} != [{scale_rows}, {scale_cols}]",
            view.scale_names[0],
            scale.shape
        );

        // A head block's rows must tile the scale-block grid, or a scale row straddles
        // two.
        let scale_blocks = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| {
                ensure!(
                    b.head_rows.is_multiple_of(block_m),
                    "{name}: fused block {i} head_rows {} not a multiple of block_m {block_m} \
                     (FP8 head shard requires head rows to tile the scale grid)",
                    b.head_rows
                );
                Ok(crate::shard_slice::HeadBlock {
                    heads: b.heads,
                    head_rows: b.head_rows / block_m,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            cols.is_multiple_of(block_k),
            "{name}: cols {cols} not a multiple of block_k {block_k}"
        );

        let weight_shard = crate::shard_slice::shard_head_blocks_column_parallel(
            weight.bytes(),
            cols,
            1,
            blocks,
            tp,
        )?;
        let scale_shard = crate::shard_slice::shard_head_blocks_column_parallel(
            scale.bytes(),
            scale_cols,
            scale_elem,
            &scale_blocks,
            tp,
        )?;
        let scales = tensor_bytes_to_f32(
            &view.scale_names[0],
            scale.dtype,
            &scale_shard.bytes,
            scale_apply,
        )?;
        let matrix = DeviceMatrix::from_fp8_block_scaled(
            ctx,
            &weight_shard.bytes,
            &scales,
            weight_shard.rows,
            weight_shard.cols,
            block_m,
            block_k,
        )
        .with_context(|| format!("upload sharded FP8 fused qkv {name}"))?;
        Ok(Some(matrix))
    }

    /// Load this EP rank's MoE weights for one layer (routed gate/up/down + router gate
    /// +
    /// shared expert) and build the per-expert weight-pointer tables. Only the experts
    /// in
    /// `split.local_expert_start..local_expert_end()` are loaded.
    ///
    /// Routed experts ship either per-expert (`experts.{i}.{gate,up,down}_proj.weight`)
    /// or
    /// stacked+fused (`experts.gate_up_proj` `[E, 2*moe_inter, hidden]`, gate rows
    /// first, plus
    /// `experts.down_proj` `[E, hidden, moe_inter]`), auto-detected per layer.
    ///
    /// Under TP the router gate and the shared-expert sigmoid gate stay replicated —
    /// routing
    /// must be computed identically on every rank — while the shared expert is sharded
    /// like a
    /// dense MLP so its partial lands in the same post-MoE all-reduce.
    pub(crate) fn load_moe_layer_experts(
        &self,
        ctx: &DeviceContext,
        names: &qwen35_spec::Qwen35MoeTensorNames,
        split: &crate::moe_config::ExpertSplit,
        tp: &TpConfig,
        moe_intermediate_size: usize,
        hidden_size: usize,
    ) -> Result<MoeLayerWeights> {
        let layer_t0 = Instant::now();
        const BF16_ELEM_SIZE: usize = 2;
        let mut gate = Vec::with_capacity(split.experts_per_rank);
        let mut up = Vec::with_capacity(split.experts_per_rank);
        let mut down = Vec::with_capacity(split.experts_per_rank);
        let per_expert_probe = names.expert_gate_proj(split.local_expert_start);
        let per_expert_quant_probe = self.quant_view_for(&per_expert_probe)?.is_some();
        let deepgemm_native_ready = match cuda_kernels::moe::dsv4_deepgemm_native_preflight() {
            Ok(_) => true,
            Err(err) => {
                log::warn!(
                    "Qwen3.5 DeepGEMM MoE disabled: native bridge unavailable ({err}); \
                             falling back to the hand grouped kernels"
                );
                false
            }
        };
        // sm_120 has no DeepGEMM native bridge, but the CUTLASS sm_120a grouped
        // collective
        // consumes the SAME contiguous grouped FP8 caches — build them regardless of
        // the
        // Hopper-only preflight, with weight scales transposed to N-contiguous SFB.
        let sm120 = ctx.is_sm120();
        let mut direct_fp8_routed = None;
        // The OPD rollout student re-merges LoRA into experts each step, which needs a
        // mutable
        // per-expert BF16 `DeviceMatrix` — suppress the fused grouped-FP8 path.
        let experts_bf16_resident = crate::runtime_flags::qwen35_moe_experts_bf16_resident();
        // The stacked tensors are HF `nn.Parameter`s (no `.weight` suffix), but accept
        // a
        // `.weight`-suffixed export too.
        let resolve_stacked = |base: &str| -> Option<String> {
            [base.to_string(), format!("{base}.weight")]
                .into_iter()
                .find(|name| self.has_tensor(name))
        };
        if !experts_bf16_resident && per_expert_quant_probe && (deepgemm_native_ready || sm120) {
            direct_fp8_routed = self.load_fp8_moe_groups_direct(
                ctx,
                names,
                split,
                moe_intermediate_size,
                hidden_size,
                sm120,
            )?;
        }
        if direct_fp8_routed.is_some() {
            // The direct FP8 path already filled the grouped caches; no per-expert
            // list.
        } else if self.has_tensor(&per_expert_probe) || per_expert_quant_probe {
            for e in split.local_expert_start..split.local_expert_end() {
                gate.push(self.load_matrix_quant_aware(ctx, &names.expert_gate_proj(e))?);
                up.push(self.load_matrix_quant_aware(ctx, &names.expert_up_proj(e))?);
                down.push(self.load_matrix_quant_aware(ctx, &names.expert_down_proj(e))?);
            }
        } else if let Some(gate_up_name) = resolve_stacked(&names.experts_stacked_gate_up_proj) {
            let routed_t0 = Instant::now();
            let down_name = resolve_stacked(&names.experts_stacked_down_proj).ok_or_else(|| {
                anyhow!(
                    "MoE layer `{}`: found stacked `{gate_up_name}` but no `{}` \
                     (expected [{}, {hidden_size}, {moe_intermediate_size}])",
                    names.mlp_prefix,
                    names.experts_stacked_down_proj,
                    split.num_experts
                )
            })?;
            ensure!(
                moe_intermediate_size > 0 && hidden_size > 0,
                "MoE layer `{}`: stacked expert load needs non-zero config dims \
                 (moe_intermediate_size={moe_intermediate_size}, hidden_size={hidden_size})",
                names.mlp_prefix
            );
            let stacked_rows = 2 * moe_intermediate_size;
            // Borrow each stacked tensor ONCE and slice every local expert out of the
            // cached
            // bytes — an owned load costs ~1 GiB + 512 MiB of host memcpy per MoE
            // layer.
            let gate_up_t = self.borrow_bf16_tensor(&gate_up_name)?;
            ensure!(
                gate_up_t.shape == [split.num_experts, stacked_rows, hidden_size],
                "{gate_up_name}: expected stacked fused gate‖up tensor \
                 [{}, {stacked_rows}, {hidden_size}] \
                 ([num_experts, 2*moe_intermediate_size, hidden_size]), got {:?}",
                split.num_experts,
                gate_up_t.shape
            );
            let down_t = self.borrow_bf16_tensor(&down_name)?;
            ensure!(
                down_t.shape == [split.num_experts, hidden_size, moe_intermediate_size],
                "{down_name}: expected stacked down tensor \
                 [{}, {hidden_size}, {moe_intermediate_size}] \
                 ([num_experts, hidden_size, moe_intermediate_size]), got {:?}",
                split.num_experts,
                down_t.shape
            );
            for e in split.local_expert_start..split.local_expert_end() {
                // gate = rows [0, mi), up = rows [mi, 2*mi) of expert e's contiguous
                // block.
                let gate_bytes = crate::shard_slice::slice_stacked_expert(
                    gate_up_t.bytes(),
                    split.num_experts,
                    stacked_rows,
                    hidden_size,
                    BF16_ELEM_SIZE,
                    e,
                    0,
                    moe_intermediate_size,
                )?;
                gate.push(
                    DeviceMatrix::from_safetensors(
                        ctx,
                        gate_bytes,
                        moe_intermediate_size,
                        hidden_size,
                    )
                    .with_context(|| format!("upload expert {e} gate slice of {gate_up_name}"))?,
                );
                let up_bytes = crate::shard_slice::slice_stacked_expert(
                    gate_up_t.bytes(),
                    split.num_experts,
                    stacked_rows,
                    hidden_size,
                    BF16_ELEM_SIZE,
                    e,
                    moe_intermediate_size,
                    moe_intermediate_size,
                )?;
                up.push(
                    DeviceMatrix::from_safetensors(
                        ctx,
                        up_bytes,
                        moe_intermediate_size,
                        hidden_size,
                    )
                    .with_context(|| format!("upload expert {e} up slice of {gate_up_name}"))?,
                );
                // down_proj [E, hidden, mi]: the whole expert block.
                let down_bytes = crate::shard_slice::slice_stacked_expert(
                    down_t.bytes(),
                    split.num_experts,
                    hidden_size,
                    moe_intermediate_size,
                    BF16_ELEM_SIZE,
                    e,
                    0,
                    hidden_size,
                )?;
                down.push(
                    DeviceMatrix::from_safetensors(
                        ctx,
                        down_bytes,
                        hidden_size,
                        moe_intermediate_size,
                    )
                    .with_context(|| format!("upload expert {e} down slice of {down_name}"))?,
                );
            }
            crate::executor::cuda_startup_log(
                "loader.moe.stacked_routed_load",
                routed_t0,
                format_args!(
                    "layer={} local_experts={} gate={} up={} down={}",
                    names.mlp_prefix,
                    split.experts_per_rank,
                    gate.len(),
                    up.len(),
                    down.len()
                ),
            );
        } else {
            let legacy_switch_mlp =
                resolve_stacked(&format!("{}.switch_mlp.gate_proj", names.mlp_prefix)).is_some();
            bail!(
                "MoE layer `{}`: no recognized routed-expert layout — need per-expert \
                 `{per_expert_probe}` (+ up/down siblings) or stacked+fused \
                 `{}` [{}, {}, {hidden_size}] + `{}` [{}, {hidden_size}, {moe_intermediate_size}]{}",
                names.mlp_prefix,
                names.experts_stacked_gate_up_proj,
                split.num_experts,
                2 * moe_intermediate_size,
                names.experts_stacked_down_proj,
                split.num_experts,
                if legacy_switch_mlp {
                    " (found unsupported legacy `switch_mlp.*`)"
                } else {
                    ""
                }
            );
        }
        crate::executor::cuda_startup_log(
            "loader.moe.routed_load",
            layer_t0,
            format_args!(
                "layer={} local_experts={} gate={} up={} down={} direct_fp8_grouped={}",
                names.mlp_prefix,
                split.experts_per_rank,
                gate.len(),
                up.len(),
                down.len(),
                direct_fp8_routed.is_some()
            ),
        );
        let shared_t0 = Instant::now();
        let router_gate = self.load_matrix(ctx, &names.router_gate)?;
        let (shared_gate, shared_up, shared_down) = if tp.is_single() {
            (
                self.load_dense_matrix_quant_aware(ctx, &names.shared_expert_gate_proj)?,
                self.load_dense_matrix_quant_aware(ctx, &names.shared_expert_up_proj)?,
                self.load_dense_matrix_quant_aware(ctx, &names.shared_expert_down_proj)?,
            )
        } else {
            (
                self.load_matrix_sharded_quant_aware(
                    ctx,
                    &names.shared_expert_gate_proj,
                    infer_topo::ParallelLinearKind::Column,
                    tp,
                )?,
                self.load_matrix_sharded_quant_aware(
                    ctx,
                    &names.shared_expert_up_proj,
                    infer_topo::ParallelLinearKind::Column,
                    tp,
                )?,
                self.load_matrix_sharded_quant_aware(
                    ctx,
                    &names.shared_expert_down_proj,
                    infer_topo::ParallelLinearKind::Row,
                    tp,
                )?,
            )
        };
        let shared_gate_router = self.load_matrix(ctx, &names.shared_expert_gate)?;
        crate::executor::cuda_startup_log(
            "loader.moe.shared_load",
            shared_t0,
            format_args!("layer={}", names.mlp_prefix),
        );

        // Concat the per-expert matrices into one contiguous [G, n, k] buffer per
        // projection
        // and DROP the per-expert copies — keeping both doubles routed-expert VRAM (~2x
        // model
        // weights on Qwen3.6-35B). An unavailable native bridge must skip the grouped
        // caches
        // so `use_deepgemm` self-disables instead of erroring at the first MoE forward.
        let (expert_weight_format, gate_sig, down_sig) =
            if let Some(direct) = direct_fp8_routed.as_ref() {
                (
                    WeightFormat::Fp8BlockScaled,
                    Some(direct.gate_up_quant_signature),
                    Some(direct.down_quant_signature),
                )
            } else {
                routed_expert_weight_format(&gate, &up, &down)?
            };
        // BF16-resident student: dequantize the per-expert FP8 experts in place so the
        // layer
        // is one BF16 kernel with a stable ptr table the LoRA re-merge can fold into.
        // Must run
        // before the grouped-cache decision and pointer-table build below.
        let mut expert_weight_format = expert_weight_format;
        let mut gate_sig = gate_sig;
        let mut down_sig = down_sig;
        if experts_bf16_resident && expert_weight_format == WeightFormat::Fp8BlockScaled {
            for (proj, experts) in [("gate", &mut gate), ("up", &mut up), ("down", &mut down)] {
                for (e, m) in experts.iter_mut().enumerate() {
                    dequantize_fp8_expert_to_bf16_in_place(ctx, m).with_context(|| {
                        format!("dequantize FP8 {proj} expert {e} of `{}`", names.mlp_prefix)
                    })?;
                }
            }
            expert_weight_format = WeightFormat::DenseBf16;
            // FP8 quant signatures are stale now the experts are dense BF16.
            gate_sig = None;
            down_sig = None;
        }
        let routed_quant = expert_weight_format.is_quantized();
        let grouped_t0 = Instant::now();
        // BF16 grouped DeepGEMM is Hopper-only: the contiguous kernel reads m_indices,
        // which
        // the sm_120 path leaves None, so building the caches there would panic on
        // first
        // prefill. Also skipped when BF16-resident: the concat clears the per-expert
        // Vecs the
        // LoRA re-merge needs mutable.
        let deepgemm_ready =
            !routed_quant && deepgemm_native_ready && !sm120 && !experts_bf16_resident;
        let fp8_deepgemm_ready =
            expert_weight_format == WeightFormat::Fp8BlockScaled && deepgemm_native_ready;
        let (gate_grouped, up_grouped, down_grouped) = if deepgemm_ready {
            let gate_g = MoeExpertGroup::concat(ctx, &gate)?;
            let up_g = MoeExpertGroup::concat(ctx, &up)?;
            let down_g = MoeExpertGroup::concat(ctx, &down)?;
            // Event tracking is disabled: dropping the per-expert sources
            // frees device memory at Rust last-use, so the async D2D concats
            // MUST have completed first.
            ctx.sync()?;
            gate.clear();
            up.clear();
            down.clear();
            (Some(gate_g), Some(up_g), Some(down_g))
        } else {
            (None, None, None)
        };
        let (w13_fp8_grouped, down_fp8_grouped) = if let Some(direct) = direct_fp8_routed.take() {
            (Some(direct.w13), Some(direct.down))
        } else if fp8_deepgemm_ready {
            let w13_g = MoeFp8ExpertGroup::concat_pair_rows(
                ctx,
                &gate,
                &up,
                moe_intermediate_size,
                hidden_size,
            )?;
            let down_g = MoeFp8ExpertGroup::concat(ctx, &down, hidden_size, moe_intermediate_size)?;
            // Event tracking is disabled: sync before dropping the sources, whose bytes
            // the
            // async D2D concats above may still be reading.
            ctx.sync()?;
            (Some(w13_g), Some(down_g))
        } else {
            (None, None)
        };

        let ptr_tables = build_moe_layer_pointer_tables(
            ctx,
            expert_weight_format,
            &gate,
            &up,
            &down,
            gate_grouped.as_ref(),
            up_grouped.as_ref(),
            down_grouped.as_ref(),
            w13_fp8_grouped.as_ref(),
            down_fp8_grouped.as_ref(),
        )?;
        if fp8_deepgemm_ready {
            gate.clear();
            up.clear();
            down.clear();
        }
        crate::executor::cuda_startup_log(
            "loader.moe.grouped_cache",
            grouped_t0,
            format_args!(
                "layer={} format={expert_weight_format:?} fp8_deepgemm_ready={} routed_quant={} retained_gate={} retained_up={} retained_down={}",
                names.mlp_prefix,
                fp8_deepgemm_ready,
                routed_quant,
                gate.len(),
                up.len(),
                down.len()
            ),
        );

        Ok(MoeLayerWeights {
            gate,
            up,
            down,
            expert_weight_format,
            gate_up_quant_signature: gate_sig,
            down_quant_signature: down_sig,
            gate_ptrs: ptr_tables.gate_ptrs,
            up_ptrs: ptr_tables.up_ptrs,
            down_ptrs: ptr_tables.down_ptrs,
            gate_scale_ptrs: ptr_tables.gate_scale_ptrs,
            up_scale_ptrs: ptr_tables.up_scale_ptrs,
            down_scale_ptrs: ptr_tables.down_scale_ptrs,
            gate_global_ptrs: ptr_tables.gate_global_ptrs,
            up_global_ptrs: ptr_tables.up_global_ptrs,
            down_global_ptrs: ptr_tables.down_global_ptrs,
            gate_grouped,
            up_grouped,
            down_grouped,
            w13_fp8_grouped,
            down_fp8_grouped,
            router_gate,
            shared_gate,
            shared_up,
            shared_down,
            shared_gate_router,
        })
    }

    fn load_fp8_moe_groups_direct(
        &self,
        ctx: &DeviceContext,
        names: &qwen35_spec::Qwen35MoeTensorNames,
        split: &crate::moe_config::ExpertSplit,
        moe_intermediate_size: usize,
        hidden_size: usize,
        // Transpose each expert's block scales from the checkpoint's K-contiguous
        // `[n_blocks, k_blocks]` to CUTLASS's N-contiguous SFB. Hopper keeps the raw
        // layout.
        transpose_sfb: bool,
    ) -> Result<Option<DirectFp8MoeRouted>> {
        let t0 = Instant::now();
        ensure!(
            moe_intermediate_size.is_multiple_of(128) && hidden_size.is_multiple_of(128),
            "Qwen3.6 FP8 direct grouped MoE needs 128-aligned dims, got mi={moe_intermediate_size} hidden={hidden_size}"
        );
        let groups = split.experts_per_rank;
        let w13_rows = 2 * moe_intermediate_size;
        let w13_scale_rows = w13_rows / 128;
        let w13_scale_cols = hidden_size / 128;
        let down_scale_rows = hidden_size / 128;
        let down_scale_cols = moe_intermediate_size / 128;
        let mut expert_views = Vec::with_capacity(groups);
        let mut shard_idx = None;
        let gate_up_sig = ExpertQuantDispatchSignature {
            rows: moe_intermediate_size,
            cols: hidden_size,
            quant_scale_rows: moe_intermediate_size / 128,
            quant_scale_cols: hidden_size / 128,
            quant_block_m: 128,
            quant_block_k: 128,
            group_size: 0,
        };
        let down_sig = ExpertQuantDispatchSignature {
            rows: hidden_size,
            cols: moe_intermediate_size,
            quant_scale_rows: hidden_size / 128,
            quant_scale_cols: moe_intermediate_size / 128,
            quant_block_m: 128,
            quant_block_k: 128,
            group_size: 0,
        };

        for e in split.local_expert_start..split.local_expert_end() {
            let gate = match self.fp8_block_projection_view(
                &names.expert_gate_proj(e),
                moe_intermediate_size,
                hidden_size,
            )? {
                Some(view) => view,
                None => return Ok(None),
            };
            let up = match self.fp8_block_projection_view(
                &names.expert_up_proj(e),
                moe_intermediate_size,
                hidden_size,
            )? {
                Some(view) => view,
                None => return Ok(None),
            };
            let down = match self.fp8_block_projection_view(
                &names.expert_down_proj(e),
                hidden_size,
                moe_intermediate_size,
            )? {
                Some(view) => view,
                None => return Ok(None),
            };
            for view in [&gate, &up, &down] {
                let Some(weight_idx) = self.weight_map.get(&view.weight_name).copied() else {
                    return Ok(None);
                };
                let Some(scale_idx) = self.weight_map.get(&view.scale_name).copied() else {
                    return Ok(None);
                };
                if weight_idx != scale_idx {
                    return Ok(None);
                }
                match shard_idx {
                    Some(idx) if idx != weight_idx => return Ok(None),
                    Some(_) => {}
                    None => shard_idx = Some(weight_idx),
                }
            }
            expert_views.push((gate, up, down));
        }
        let Some(shard_idx) = shard_idx else {
            return Ok(None);
        };

        let mut w13_weight = vec![0u8; groups * w13_rows * hidden_size];
        let mut w13_scales = vec![0f32; groups * w13_scale_rows * w13_scale_cols];
        let mut down_weight = vec![0u8; groups * hidden_size * moe_intermediate_size];
        let mut down_scales = vec![0f32; groups * down_scale_rows * down_scale_cols];
        let shard = self.shard_bytes(shard_idx)?;
        let tensors = SafeTensors::deserialize(&shard)
            .with_context(|| format!("deserialize {}", self.shards[shard_idx].display()))?;

        for (g, (gate, up, down)) in expert_views.iter().enumerate() {
            let w13_weight_base = g * w13_rows * hidden_size;
            let gate_weight = &mut w13_weight
                [w13_weight_base..w13_weight_base + moe_intermediate_size * hidden_size];
            self.copy_fp8_projection_from_shard(&tensors, gate, gate_weight)?;
            let up_weight_start = w13_weight_base + moe_intermediate_size * hidden_size;
            let up_weight = &mut w13_weight
                [up_weight_start..up_weight_start + moe_intermediate_size * hidden_size];
            self.copy_fp8_projection_from_shard(&tensors, up, up_weight)?;

            let w13_scale_base = g * w13_scale_rows * w13_scale_cols;
            let gate_scales =
                &mut w13_scales[w13_scale_base..w13_scale_base + gate.scale_rows * gate.scale_cols];
            self.copy_fp8_scales_from_shard(&tensors, gate, gate_scales)?;
            let up_scale_start = w13_scale_base + (moe_intermediate_size / 128) * w13_scale_cols;
            let up_scales =
                &mut w13_scales[up_scale_start..up_scale_start + up.scale_rows * up.scale_cols];
            self.copy_fp8_scales_from_shard(&tensors, up, up_scales)?;

            let down_weight_base = g * hidden_size * moe_intermediate_size;
            let down_weight_dst = &mut down_weight
                [down_weight_base..down_weight_base + hidden_size * moe_intermediate_size];
            self.copy_fp8_projection_from_shard(&tensors, down, down_weight_dst)?;
            let down_scale_base = g * down_scale_rows * down_scale_cols;
            let down_scales_dst = &mut down_scales
                [down_scale_base..down_scale_base + down.scale_rows * down.scale_cols];
            self.copy_fp8_scales_from_shard(&tensors, down, down_scales_dst)?;
        }

        // CUTLASS SFB is N-contiguous per group; the checkpoint is K-contiguous.
        if transpose_sfb {
            transpose_group_block_scales(&mut w13_scales, groups, w13_scale_rows, w13_scale_cols);
            transpose_group_block_scales(
                &mut down_scales,
                groups,
                down_scale_rows,
                down_scale_cols,
            );
        }

        // `transpose_sfb` (== sm_120) also records the layout on the cache, so the
        // executor's
        // dispatch reads one source of truth.
        let w13 = MoeFp8ExpertGroup::from_host(
            ctx,
            &w13_weight,
            &w13_scales,
            groups,
            w13_rows,
            hidden_size,
            transpose_sfb,
        )?;
        let down = MoeFp8ExpertGroup::from_host(
            ctx,
            &down_weight,
            &down_scales,
            groups,
            hidden_size,
            moe_intermediate_size,
            transpose_sfb,
        )?;
        crate::executor::cuda_startup_log(
            "loader.moe.direct_fp8_grouped_load",
            t0,
            format_args!(
                "layer={} shard_idx={} local_experts={} w13_bytes={} down_bytes={}",
                names.mlp_prefix,
                shard_idx,
                groups,
                w13_weight.len(),
                down_weight.len()
            ),
        );
        Ok(Some(DirectFp8MoeRouted {
            w13,
            down,
            gate_up_quant_signature: gate_up_sig,
            down_quant_signature: down_sig,
        }))
    }

    fn fp8_block_projection_view(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Option<Fp8BlockProjectionView>> {
        let Some(view) = self.quant_view_for(name)? else {
            return Ok(None);
        };
        let QuantFormat::Fp8BlockScaled {
            block_m,
            block_k,
            scale_apply,
        } = view.format
        else {
            return Ok(None);
        };
        ensure!(
            block_m == 128 && block_k == 128,
            "{}: direct FP8 grouped MoE supports 128x128 block scales, got {block_m}x{block_k}",
            view.name
        );
        ensure!(
            view.storage_dtype == Dtype::F8_E4M3 && view.logical_shape == [rows, cols],
            "{}: expected FP8 projection [{rows}, {cols}], got {:?} {:?}",
            view.name,
            view.storage_dtype,
            view.logical_shape
        );
        let scale_name = view
            .scale_names
            .first()
            .ok_or_else(|| anyhow!("{}: FP8 projection missing scale tensor", view.name))?
            .clone();
        Ok(Some(Fp8BlockProjectionView {
            weight_name: view.name,
            scale_name,
            rows,
            cols,
            scale_rows: rows / 128,
            scale_cols: cols / 128,
            scale_apply,
        }))
    }

    fn copy_fp8_projection_from_shard(
        &self,
        tensors: &SafeTensors<'_>,
        view: &Fp8BlockProjectionView,
        dst: &mut [u8],
    ) -> Result<()> {
        let tensor = tensors
            .tensor(&view.weight_name)
            .with_context(|| format!("find tensor {}", view.weight_name))?;
        ensure!(
            tensor.dtype() == Dtype::F8_E4M3 && tensor.shape() == [view.rows, view.cols],
            "{}: expected F8_E4M3 [{}, {}], got {:?} {:?}",
            view.weight_name,
            view.rows,
            view.cols,
            tensor.dtype(),
            tensor.shape()
        );
        let data = tensor.data();
        ensure!(
            data.len() == dst.len(),
            "{}: FP8 weight bytes {} != destination {}",
            view.weight_name,
            data.len(),
            dst.len()
        );
        dst.copy_from_slice(data);
        Ok(())
    }

    fn copy_fp8_scales_from_shard(
        &self,
        tensors: &SafeTensors<'_>,
        view: &Fp8BlockProjectionView,
        dst: &mut [f32],
    ) -> Result<()> {
        let tensor = tensors
            .tensor(&view.scale_name)
            .with_context(|| format!("find tensor {}", view.scale_name))?;
        ensure!(
            (tensor.dtype() == Dtype::BF16 || tensor.dtype() == Dtype::F32)
                && tensor.shape() == [view.scale_rows, view.scale_cols],
            "{}: expected BF16/F32 scale [{}, {}], got {:?} {:?}",
            view.scale_name,
            view.scale_rows,
            view.scale_cols,
            tensor.dtype(),
            tensor.shape()
        );
        let scales = tensor_bytes_to_f32(
            &view.scale_name,
            tensor.dtype(),
            tensor.data(),
            view.scale_apply,
        )?;
        ensure!(
            scales.len() == dst.len(),
            "{}: FP8 scale values {} != destination {}",
            view.scale_name,
            scales.len(),
            dst.len()
        );
        dst.copy_from_slice(&scales);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExpertQuantDispatchSignature {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) quant_scale_rows: usize,
    pub(crate) quant_scale_cols: usize,
    pub(crate) quant_block_m: usize,
    pub(crate) quant_block_k: usize,
    pub(crate) group_size: usize,
}

impl ExpertQuantDispatchSignature {
    fn from_matrix(matrix: &DeviceMatrix) -> Self {
        Self {
            rows: matrix.rows,
            cols: matrix.cols,
            quant_scale_rows: matrix.quant_scale_rows,
            quant_scale_cols: matrix.quant_scale_cols,
            quant_block_m: matrix.quant_block_m,
            quant_block_k: matrix.quant_block_k,
            group_size: matrix.group_size,
        }
    }
}

/// Dequantize one FP8-block-scaled routed expert to dense BF16 in place. Runs at load
/// for the
/// whole layer because grouped MoE dispatches one kernel per layer off a static ptr
/// table,
/// so the per-expert lazy promote cannot apply.
fn dequantize_fp8_expert_to_bf16_in_place(
    ctx: &DeviceContext,
    matrix: &mut DeviceMatrix,
) -> Result<()> {
    if matrix.weight_format == WeightFormat::DenseBf16 {
        return Ok(());
    }
    ensure!(
        matrix.weight_format == WeightFormat::Fp8BlockScaled
            && matrix.quant_block_m > 0
            && matrix.quant_block_k > 0
            && matrix.quant_scale_rows > 0
            && matrix.quant_scale_cols > 0,
        "BF16-resident expert dequant needs FP8 block-scaled metadata; got {:?}",
        matrix.weight_format
    );
    let dense = ctx
        .stream
        .alloc_zeros::<half::bf16>(matrix.rows * matrix.cols)
        .map_err(|e| anyhow!("expert BF16 dequant alloc failed: {e}"))?;
    {
        let qweight = matrix
            .qweight_u8
            .as_ref()
            .ok_or_else(|| anyhow!("FP8 expert missing qweight"))?;
        let scales = matrix
            .scale_f32
            .as_ref()
            .ok_or_else(|| anyhow!("FP8 expert missing f32 scales"))?;
        ensure!(
            qweight.len() == matrix.rows * matrix.cols,
            "FP8 expert qweight len {} != rows*cols {}",
            qweight.len(),
            matrix.rows * matrix.cols
        );
        let shape = cuda_kernels::quant_linear::Fp8ScaleShape {
            scale_rows: matrix.quant_scale_rows as i32,
            scale_cols: matrix.quant_scale_cols as i32,
            block_m: matrix.quant_block_m as i32,
            block_k: matrix.quant_block_k as i32,
        };
        // SAFETY: `dense` covers rows*cols and lives across the launch (the
        // sync below outlives it).
        unsafe {
            cuda_kernels::quant_linear::dequantize_fp8_block_scaled_to_bf16(
                ctx,
                qweight,
                scales,
                cuda_kernels::tensor::cache_ptr(&dense, ctx),
                matrix.rows,
                matrix.cols,
                shape,
            )
        }
        .map_err(|e| anyhow!("FP8->BF16 expert dequant failed: {e}"))?;
    }
    // Event tracking is disabled: sync before dropping the FP8 source so the
    // async dequant kernel has finished reading it (mirrors the grouped path).
    ctx.sync()?;
    matrix.data = dense;
    matrix.weight_format = WeightFormat::DenseBf16;
    matrix.qweight_u8 = None;
    matrix.scale_f32 = None;
    matrix.quant_scale_rows = 0;
    matrix.quant_scale_cols = 0;
    matrix.quant_block_m = 0;
    matrix.quant_block_k = 0;
    Ok(())
}

fn validate_expert_projection_dispatch_signature(
    name: &str,
    experts: &[DeviceMatrix],
    format: WeightFormat,
) -> Result<Option<ExpertQuantDispatchSignature>> {
    let first = experts
        .first()
        .ok_or_else(|| anyhow!("MoE layer has no local {name} experts"))?;
    let first_sig = ExpertQuantDispatchSignature::from_matrix(first);
    for (idx, expert) in experts.iter().enumerate() {
        ensure!(
            expert.weight_format() == format,
            "Qwen3.6 MoE {name} expert {idx} format {} != {format}",
            expert.weight_format()
        );
        if format.is_quantized() {
            let sig = ExpertQuantDispatchSignature::from_matrix(expert);
            ensure!(
                sig == first_sig,
                "Qwen3.6 MoE {name} expert {idx} quant dispatch signature {sig:?} != {first_sig:?}"
            );
        }
    }
    Ok(format.is_quantized().then_some(first_sig))
}

fn routed_expert_weight_format(
    gate: &[DeviceMatrix],
    up: &[DeviceMatrix],
    down: &[DeviceMatrix],
) -> Result<(
    WeightFormat,
    Option<ExpertQuantDispatchSignature>,
    Option<ExpertQuantDispatchSignature>,
)> {
    let first = gate
        .first()
        .ok_or_else(|| anyhow!("MoE layer has no local gate experts"))?
        .weight_format();
    ensure!(
        matches!(
            first,
            WeightFormat::DenseBf16
                | WeightFormat::Fp8BlockScaled
                | WeightFormat::Fp8PerShard
                | WeightFormat::Fp4E2M1Group
                | WeightFormat::W4A16
        ),
        "Qwen3.6 MoE routed expert format {first} is not supported"
    );
    let gate_sig = validate_expert_projection_dispatch_signature("gate", gate, first)?;
    let up_sig = validate_expert_projection_dispatch_signature("up", up, first)?;
    let down_sig = validate_expert_projection_dispatch_signature("down", down, first)?;
    if let (Some(gate_sig), Some(up_sig)) = (gate_sig, up_sig) {
        ensure!(
            gate_sig == up_sig,
            "Qwen3.6 MoE gate/up quant dispatch signature mismatch: gate={gate_sig:?} up={up_sig:?}"
        );
    }
    Ok((first, gate_sig, down_sig))
}

/// This EP rank's loaded MoE weights for one sparse layer. Built by
/// [`SafetensorLoader::load_moe_layer_experts`], consumed by
/// [`crate::moe::moe_forward_into`].
pub(crate) struct MoeLayerWeights {
    /// Per-expert weight matrices (hand grouped-GEMM path). EMPTY when the grouped
    /// caches
    /// below are built — the grouped buffer then owns the only copy of the bytes and
    /// the
    /// `*_ptrs` tables point into it, so the hand kernels stay runnable.
    pub(crate) gate: Vec<DeviceMatrix>,
    pub(crate) up: Vec<DeviceMatrix>,
    pub(crate) down: Vec<DeviceMatrix>,
    pub(crate) expert_weight_format: WeightFormat,
    pub(crate) gate_up_quant_signature: Option<ExpertQuantDispatchSignature>,
    pub(crate) down_quant_signature: Option<ExpertQuantDispatchSignature>,
    pub(crate) gate_ptrs: CudaSlice<u64>,
    pub(crate) up_ptrs: CudaSlice<u64>,
    pub(crate) down_ptrs: CudaSlice<u64>,
    pub(crate) gate_scale_ptrs: Option<CudaSlice<u64>>,
    pub(crate) up_scale_ptrs: Option<CudaSlice<u64>>,
    pub(crate) down_scale_ptrs: Option<CudaSlice<u64>>,
    pub(crate) gate_global_ptrs: Option<CudaSlice<u64>>,
    pub(crate) up_global_ptrs: Option<CudaSlice<u64>>,
    pub(crate) down_global_ptrs: Option<CudaSlice<u64>>,
    /// DeepGEMM grouped-B caches (`[groups, n, k]` contiguous row-major BF16, this
    /// rank's EP
    /// experts only).
    pub(crate) gate_grouped: Option<MoeExpertGroup>,
    pub(crate) up_grouped: Option<MoeExpertGroup>,
    pub(crate) down_grouped: Option<MoeExpertGroup>,
    /// DeepGEMM FP8 grouped-B cache for quantized routed experts. `w13`
    /// fuses gate rows followed by up rows per expert, so the DeepGEMM
    /// prefill lane can run one FP8 GEMM then SwiGLU+requantize.
    pub(crate) w13_fp8_grouped: Option<MoeFp8ExpertGroup>,
    pub(crate) down_fp8_grouped: Option<MoeFp8ExpertGroup>,
    pub(crate) router_gate: DeviceMatrix,
    pub(crate) shared_gate: DeviceMatrix,
    pub(crate) shared_up: DeviceMatrix,
    pub(crate) shared_down: DeviceMatrix,
    pub(crate) shared_gate_router: DeviceMatrix,
}

pub(crate) struct MoeLayerHostSnapshot {
    gate: Vec<HostMatrixSnapshot>,
    up: Vec<HostMatrixSnapshot>,
    down: Vec<HostMatrixSnapshot>,
    gate_grouped: Option<MoeExpertGroupHostSnapshot>,
    up_grouped: Option<MoeExpertGroupHostSnapshot>,
    down_grouped: Option<MoeExpertGroupHostSnapshot>,
    w13_fp8_grouped: Option<MoeFp8ExpertGroupHostSnapshot>,
    down_fp8_grouped: Option<MoeFp8ExpertGroupHostSnapshot>,
    router_gate: HostMatrixSnapshot,
    shared_gate: HostMatrixSnapshot,
    shared_up: HostMatrixSnapshot,
    shared_down: HostMatrixSnapshot,
    shared_gate_router: HostMatrixSnapshot,
    freed_bytes: usize,
}

impl MoeLayerHostSnapshot {
    #[must_use]
    pub(crate) fn freed_bytes(&self) -> usize {
        self.freed_bytes
    }
}

struct MoeExpertGroupHostSnapshot {
    data: Vec<half::bf16>,
    groups: usize,
    rows: usize,
    cols: usize,
}

struct MoeFp8ExpertGroupHostSnapshot {
    weight: Vec<u8>,
    scales: Vec<f32>,
    groups: usize,
    rows: usize,
    cols: usize,
    sfb_n_contiguous: bool,
}

impl MoeLayerWeights {
    pub(crate) fn offload_to_host(&mut self, ctx: &DeviceContext) -> Result<MoeLayerHostSnapshot> {
        let mut freed = 0usize;
        let gate = offload_matrix_vec(ctx, &mut self.gate, "moe.gate", &mut freed)?;
        let up = offload_matrix_vec(ctx, &mut self.up, "moe.up", &mut freed)?;
        let down = offload_matrix_vec(ctx, &mut self.down, "moe.down", &mut freed)?;
        let gate_grouped =
            offload_group_opt(ctx, &mut self.gate_grouped, "moe.gate_grouped", &mut freed)?;
        let up_grouped =
            offload_group_opt(ctx, &mut self.up_grouped, "moe.up_grouped", &mut freed)?;
        let down_grouped =
            offload_group_opt(ctx, &mut self.down_grouped, "moe.down_grouped", &mut freed)?;
        let w13_fp8_grouped = offload_fp8_group_opt(
            ctx,
            &mut self.w13_fp8_grouped,
            "moe.w13_fp8_grouped",
            &mut freed,
        )?;
        let down_fp8_grouped = offload_fp8_group_opt(
            ctx,
            &mut self.down_fp8_grouped,
            "moe.down_fp8_grouped",
            &mut freed,
        )?;
        let router_gate = self.router_gate.offload_to_host(ctx)?;
        freed += router_gate.freed_bytes();
        let shared_gate = self.shared_gate.offload_to_host(ctx)?;
        freed += shared_gate.freed_bytes();
        let shared_up = self.shared_up.offload_to_host(ctx)?;
        freed += shared_up.freed_bytes();
        let shared_down = self.shared_down.offload_to_host(ctx)?;
        freed += shared_down.freed_bytes();
        let shared_gate_router = self.shared_gate_router.offload_to_host(ctx)?;
        freed += shared_gate_router.freed_bytes();

        Ok(MoeLayerHostSnapshot {
            gate,
            up,
            down,
            gate_grouped,
            up_grouped,
            down_grouped,
            w13_fp8_grouped,
            down_fp8_grouped,
            router_gate,
            shared_gate,
            shared_up,
            shared_down,
            shared_gate_router,
            freed_bytes: freed,
        })
    }

    pub(crate) fn reload_from_host(
        &mut self,
        ctx: &DeviceContext,
        snapshot: &MoeLayerHostSnapshot,
    ) -> Result<()> {
        reload_matrix_vec(ctx, &mut self.gate, &snapshot.gate, "moe.gate")?;
        reload_matrix_vec(ctx, &mut self.up, &snapshot.up, "moe.up")?;
        reload_matrix_vec(ctx, &mut self.down, &snapshot.down, "moe.down")?;
        self.gate_grouped = reload_group_opt(ctx, &snapshot.gate_grouped, "moe.gate_grouped")?;
        self.up_grouped = reload_group_opt(ctx, &snapshot.up_grouped, "moe.up_grouped")?;
        self.down_grouped = reload_group_opt(ctx, &snapshot.down_grouped, "moe.down_grouped")?;
        self.w13_fp8_grouped =
            reload_fp8_group_opt(ctx, &snapshot.w13_fp8_grouped, "moe.w13_fp8_grouped")?;
        self.down_fp8_grouped =
            reload_fp8_group_opt(ctx, &snapshot.down_fp8_grouped, "moe.down_fp8_grouped")?;
        self.router_gate
            .reload_from_host(ctx, &snapshot.router_gate)?;
        self.shared_gate
            .reload_from_host(ctx, &snapshot.shared_gate)?;
        self.shared_up.reload_from_host(ctx, &snapshot.shared_up)?;
        self.shared_down
            .reload_from_host(ctx, &snapshot.shared_down)?;
        self.shared_gate_router
            .reload_from_host(ctx, &snapshot.shared_gate_router)?;
        self.rebuild_pointer_tables(ctx)
    }

    fn rebuild_pointer_tables(&mut self, ctx: &DeviceContext) -> Result<()> {
        let ptr_tables = build_moe_layer_pointer_tables(
            ctx,
            self.expert_weight_format,
            &self.gate,
            &self.up,
            &self.down,
            self.gate_grouped.as_ref(),
            self.up_grouped.as_ref(),
            self.down_grouped.as_ref(),
            self.w13_fp8_grouped.as_ref(),
            self.down_fp8_grouped.as_ref(),
        )?;
        self.gate_ptrs = ptr_tables.gate_ptrs;
        self.up_ptrs = ptr_tables.up_ptrs;
        self.down_ptrs = ptr_tables.down_ptrs;
        self.gate_scale_ptrs = ptr_tables.gate_scale_ptrs;
        self.up_scale_ptrs = ptr_tables.up_scale_ptrs;
        self.down_scale_ptrs = ptr_tables.down_scale_ptrs;
        self.gate_global_ptrs = ptr_tables.gate_global_ptrs;
        self.up_global_ptrs = ptr_tables.up_global_ptrs;
        self.down_global_ptrs = ptr_tables.down_global_ptrs;
        Ok(())
    }
}

struct MoeLayerPointerTables {
    gate_ptrs: CudaSlice<u64>,
    up_ptrs: CudaSlice<u64>,
    down_ptrs: CudaSlice<u64>,
    gate_scale_ptrs: Option<CudaSlice<u64>>,
    up_scale_ptrs: Option<CudaSlice<u64>>,
    down_scale_ptrs: Option<CudaSlice<u64>>,
    gate_global_ptrs: Option<CudaSlice<u64>>,
    up_global_ptrs: Option<CudaSlice<u64>>,
    down_global_ptrs: Option<CudaSlice<u64>>,
}

struct MoeExpertRefs<'a> {
    gate: Vec<&'a DeviceMatrix>,
    up: Vec<&'a DeviceMatrix>,
    down: Vec<&'a DeviceMatrix>,
}

fn moe_expert_refs<'a>(
    gate: &'a [DeviceMatrix],
    up: &'a [DeviceMatrix],
    down: &'a [DeviceMatrix],
) -> Result<MoeExpertRefs<'a>> {
    ensure!(
        !gate.is_empty() && gate.len() == up.len() && gate.len() == down.len(),
        "MoE pointer-table rebuild requires matching non-empty per-expert matrices: gate={} up={} down={}",
        gate.len(),
        up.len(),
        down.len()
    );
    Ok(MoeExpertRefs {
        gate: gate.iter().collect(),
        up: up.iter().collect(),
        down: down.iter().collect(),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_moe_layer_pointer_tables(
    ctx: &DeviceContext,
    expert_weight_format: WeightFormat,
    gate: &[DeviceMatrix],
    up: &[DeviceMatrix],
    down: &[DeviceMatrix],
    gate_grouped: Option<&MoeExpertGroup>,
    up_grouped: Option<&MoeExpertGroup>,
    down_grouped: Option<&MoeExpertGroup>,
    w13_fp8_grouped: Option<&MoeFp8ExpertGroup>,
    down_fp8_grouped: Option<&MoeFp8ExpertGroup>,
) -> Result<MoeLayerPointerTables> {
    let routed_quant = expert_weight_format.is_quantized();
    let bf16_grouped = match (gate_grouped, up_grouped, down_grouped) {
        (Some(g), Some(u), Some(d)) => Some((g, u, d)),
        (None, None, None) => None,
        _ => bail!("MoE pointer-table rebuild found partial BF16 grouped cache"),
    };
    let fp8_grouped = match (w13_fp8_grouped, down_fp8_grouped) {
        (Some(w13), Some(down_g)) => Some((w13, down_g)),
        (None, None) => None,
        _ => bail!("MoE pointer-table rebuild found partial FP8 grouped cache"),
    };
    ensure!(
        bf16_grouped.is_none() || fp8_grouped.is_none(),
        "MoE pointer-table rebuild cannot use both BF16 and FP8 grouped caches"
    );

    let (gate_ptrs, up_ptrs, down_ptrs) = if let Some((g, u, d)) = bf16_grouped {
        (g.ptr_table(ctx)?, u.ptr_table(ctx)?, d.ptr_table(ctx)?)
    } else if let Some((w13, down_g)) = fp8_grouped {
        let up_offset = w13.rows / 2;
        (
            w13.qweight_ptr_table(ctx, 0)?,
            w13.qweight_ptr_table(ctx, up_offset)?,
            down_g.qweight_ptr_table(ctx, 0)?,
        )
    } else {
        let refs = moe_expert_refs(gate, up, down)?;
        if routed_quant {
            // W4A16 packs INT4 nibbles into `i8` qweight; FP8/FP4 use `u8`.
            if expert_weight_format == WeightFormat::W4A16 {
                (
                    cuda_kernels::moe::build_expert_qweight_i8_ptr_table(ctx, &refs.gate)?,
                    cuda_kernels::moe::build_expert_qweight_i8_ptr_table(ctx, &refs.up)?,
                    cuda_kernels::moe::build_expert_qweight_i8_ptr_table(ctx, &refs.down)?,
                )
            } else {
                (
                    cuda_kernels::moe::build_expert_qweight_u8_ptr_table(ctx, &refs.gate)?,
                    cuda_kernels::moe::build_expert_qweight_u8_ptr_table(ctx, &refs.up)?,
                    cuda_kernels::moe::build_expert_qweight_u8_ptr_table(ctx, &refs.down)?,
                )
            }
        } else {
            (
                cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &refs.gate)?,
                cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &refs.up)?,
                cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &refs.down)?,
            )
        }
    };

    let (gate_scale_ptrs, up_scale_ptrs, down_scale_ptrs) = if let Some((w13, down_g)) = fp8_grouped
    {
        let up_offset = w13.rows / 2;
        (
            Some(w13.scale_ptr_table(ctx, 0)?),
            Some(w13.scale_ptr_table(ctx, up_offset)?),
            Some(down_g.scale_ptr_table(ctx, 0)?),
        )
    } else if routed_quant
        && matches!(
            expert_weight_format,
            WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard
        )
    {
        let refs = moe_expert_refs(gate, up, down)?;
        (
            Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                ctx, &refs.gate,
            )?),
            Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                ctx, &refs.up,
            )?),
            Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                ctx, &refs.down,
            )?),
        )
    } else if routed_quant && expert_weight_format == WeightFormat::Fp4E2M1Group {
        let refs = moe_expert_refs(gate, up, down)?;
        (
            Some(cuda_kernels::moe::build_expert_qscale_fp8_ptr_table(
                ctx, &refs.gate,
            )?),
            Some(cuda_kernels::moe::build_expert_qscale_fp8_ptr_table(
                ctx, &refs.up,
            )?),
            Some(cuda_kernels::moe::build_expert_qscale_fp8_ptr_table(
                ctx, &refs.down,
            )?),
        )
    } else if routed_quant && expert_weight_format == WeightFormat::W4A16 {
        let refs = moe_expert_refs(gate, up, down)?;
        (
            Some(cuda_kernels::moe::build_expert_qscale_bf16_ptr_table(
                ctx, &refs.gate,
            )?),
            Some(cuda_kernels::moe::build_expert_qscale_bf16_ptr_table(
                ctx, &refs.up,
            )?),
            Some(cuda_kernels::moe::build_expert_qscale_bf16_ptr_table(
                ctx, &refs.down,
            )?),
        )
    } else {
        (None, None, None)
    };

    let (gate_global_ptrs, up_global_ptrs, down_global_ptrs) =
        if routed_quant && expert_weight_format == WeightFormat::Fp4E2M1Group {
            let refs = moe_expert_refs(gate, up, down)?;
            (
                Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                    ctx, &refs.gate,
                )?),
                Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                    ctx, &refs.up,
                )?),
                Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                    ctx, &refs.down,
                )?),
            )
        } else {
            (None, None, None)
        };

    Ok(MoeLayerPointerTables {
        gate_ptrs,
        up_ptrs,
        down_ptrs,
        gate_scale_ptrs,
        up_scale_ptrs,
        down_scale_ptrs,
        gate_global_ptrs,
        up_global_ptrs,
        down_global_ptrs,
    })
}

fn offload_matrix_vec(
    ctx: &DeviceContext,
    matrices: &mut [DeviceMatrix],
    label: &str,
    freed: &mut usize,
) -> Result<Vec<HostMatrixSnapshot>> {
    matrices
        .iter_mut()
        .enumerate()
        .map(|(idx, matrix)| {
            let snapshot = matrix
                .offload_to_host(ctx)
                .with_context(|| format!("offload {label}[{idx}]"))?;
            *freed += snapshot.freed_bytes();
            Ok(snapshot)
        })
        .collect()
}

fn reload_matrix_vec(
    ctx: &DeviceContext,
    matrices: &mut [DeviceMatrix],
    snapshots: &[HostMatrixSnapshot],
    label: &str,
) -> Result<()> {
    ensure!(
        matrices.len() == snapshots.len(),
        "reload {label}: matrix count {} != snapshot count {}",
        matrices.len(),
        snapshots.len()
    );
    for (idx, (matrix, snapshot)) in matrices.iter_mut().zip(snapshots).enumerate() {
        matrix
            .reload_from_host(ctx, snapshot)
            .with_context(|| format!("reload {label}[{idx}]"))?;
    }
    Ok(())
}

fn offload_group_opt(
    ctx: &DeviceContext,
    group: &mut Option<MoeExpertGroup>,
    label: &str,
    freed: &mut usize,
) -> Result<Option<MoeExpertGroupHostSnapshot>> {
    match group {
        Some(group) => {
            let (snapshot, bytes) = group
                .offload_to_host(ctx)
                .with_context(|| format!("offload {label}"))?;
            *freed += bytes;
            Ok(Some(snapshot))
        }
        None => Ok(None),
    }
}

fn reload_group_opt(
    ctx: &DeviceContext,
    snapshot: &Option<MoeExpertGroupHostSnapshot>,
    label: &str,
) -> Result<Option<MoeExpertGroup>> {
    snapshot
        .as_ref()
        .map(|snapshot| {
            MoeExpertGroup::from_host(ctx, snapshot).with_context(|| format!("reload {label}"))
        })
        .transpose()
}

fn offload_fp8_group_opt(
    ctx: &DeviceContext,
    group: &mut Option<MoeFp8ExpertGroup>,
    label: &str,
    freed: &mut usize,
) -> Result<Option<MoeFp8ExpertGroupHostSnapshot>> {
    match group {
        Some(group) => {
            let (snapshot, bytes) = group
                .offload_to_host(ctx)
                .with_context(|| format!("offload {label}"))?;
            *freed += bytes;
            Ok(Some(snapshot))
        }
        None => Ok(None),
    }
}

fn reload_fp8_group_opt(
    ctx: &DeviceContext,
    snapshot: &Option<MoeFp8ExpertGroupHostSnapshot>,
    label: &str,
) -> Result<Option<MoeFp8ExpertGroup>> {
    snapshot
        .as_ref()
        .map(|snapshot| {
            MoeFp8ExpertGroup::from_host(
                ctx,
                &snapshot.weight,
                &snapshot.scales,
                snapshot.groups,
                snapshot.rows,
                snapshot.cols,
                snapshot.sfb_n_contiguous,
            )
            .with_context(|| format!("reload {label}"))
        })
        .transpose()
}

/// One contiguous `[groups, rows, cols]` row-major BF16 expert-weight buffer —
/// DeepGEMM's grouped-B layout (group `g` starts at `g * rows * cols`).
pub(crate) struct MoeExpertGroup {
    pub(crate) data: CudaSlice<half::bf16>,
    pub(crate) groups: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl MoeExpertGroup {
    fn from_host(ctx: &DeviceContext, snapshot: &MoeExpertGroupHostSnapshot) -> Result<Self> {
        ensure!(
            snapshot.groups > 0,
            "MoE expert group: groups must be non-zero"
        );
        ensure!(
            snapshot.data.len() == snapshot.groups * snapshot.rows * snapshot.cols,
            "MoE grouped host data len {} != expected {}",
            snapshot.data.len(),
            snapshot.groups * snapshot.rows * snapshot.cols
        );
        Ok(Self {
            data: ctx
                .stream
                .clone_htod(snapshot.data.as_slice())
                .map_err(|e| anyhow!("MoE expert group H2D failed: {e}"))?,
            groups: snapshot.groups,
            rows: snapshot.rows,
            cols: snapshot.cols,
        })
    }

    fn offload_to_host(
        &mut self,
        ctx: &DeviceContext,
    ) -> Result<(MoeExpertGroupHostSnapshot, usize)> {
        let data = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("MoE expert group D2H failed: {e}"))?;
        let freed = data.len() * std::mem::size_of::<half::bf16>();
        ctx.sync()?;
        self.data = ctx
            .stream
            .alloc_zeros::<half::bf16>(1)
            .map_err(|e| anyhow!("MoE expert group placeholder alloc failed: {e}"))?;
        Ok((
            MoeExpertGroupHostSnapshot {
                data,
                groups: self.groups,
                rows: self.rows,
                cols: self.cols,
            },
            freed,
        ))
    }

    /// Concatenate per-expert `[rows, cols]` matrices into one contiguous group-major
    /// buffer
    /// (D2D). The source matrices may be dropped afterwards **only after a stream
    /// sync** —
    /// event tracking is disabled, so a Rust drop frees device memory immediately.
    fn concat(ctx: &DeviceContext, experts: &[DeviceMatrix]) -> Result<Self> {
        let first = experts
            .first()
            .ok_or_else(|| anyhow!("MoE expert group concat: no local experts"))?;
        let (rows, cols) = (first.rows, first.cols);
        let stride = rows * cols;
        let groups = experts.len();
        let mut data = ctx
            .stream
            .alloc_zeros::<half::bf16>(groups * stride)
            .map_err(|e| anyhow!("MoE expert group alloc failed: {e}"))?;
        for (g, expert) in experts.iter().enumerate() {
            ensure!(
                expert.rows == rows && expert.cols == cols && expert.data.len() == stride,
                "MoE expert group {g} non-uniform: {}x{} (data len {}) != {rows}x{cols}",
                expert.rows,
                expert.cols,
                expert.data.len()
            );
            ensure!(
                expert.qweight.is_none() && expert.group_size == 0,
                "MoE expert group {g} is quantized — DeepGEMM BF16 grouped cache needs dense BF16"
            );
            let mut dst = data.slice_mut(g * stride..(g + 1) * stride);
            ctx.stream
                .memcpy_dtod(&expert.data, &mut dst)
                .map_err(|e| anyhow!("MoE expert group {g} D2D failed: {e}"))?;
        }
        Ok(Self {
            data,
            groups,
            rows,
            cols,
        })
    }

    /// Device table of per-group base pointers in the same `*const u64` format as
    /// [`cuda_kernels::moe::build_expert_weight_ptr_table`], so the hand kernels run
    /// unchanged.
    fn ptr_table(&self, ctx: &DeviceContext) -> Result<CudaSlice<u64>> {
        let (base, _guard) = self.data.device_ptr(&ctx.stream);
        let stride_bytes = (self.rows * self.cols * std::mem::size_of::<half::bf16>()) as u64;
        let host: Vec<u64> = (0..self.groups as u64)
            .map(|g| base + g * stride_bytes)
            .collect();
        ctx.stream
            .clone_htod(&host)
            .map_err(|e| anyhow!("MoE expert group ptr table H2D failed: {e}"))
    }
}

/// One contiguous `[groups, rows, cols]` row-major FP8 expert-weight buffer
/// with DeepGEMM-compatible FP32 `[groups, rows/128, cols/128]` block scales.
pub(crate) struct MoeFp8ExpertGroup {
    pub(crate) weight: CudaSlice<u8>,
    pub(crate) scales: CudaSlice<f32>,
    pub(crate) groups: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) scale_rows: usize,
    pub(crate) scale_cols: usize,
    /// SFB scale layout, decided once at load: `true` = N-contiguous per group (CUTLASS
    /// sm_120a), `false` = the checkpoint's K-contiguous packing (Hopper DeepGEMM). The
    /// executor dispatches on this field and never re-derives the SM, so loader and
    /// executor
    /// cannot silently disagree on the operand layout.
    pub(crate) sfb_n_contiguous: bool,
}

impl MoeFp8ExpertGroup {
    fn from_host(
        ctx: &DeviceContext,
        weight: &[u8],
        scales: &[f32],
        groups: usize,
        rows: usize,
        cols: usize,
        sfb_n_contiguous: bool,
    ) -> Result<Self> {
        ensure!(groups > 0, "FP8 MoE expert group: groups must be non-zero");
        ensure!(
            rows.is_multiple_of(128) && cols.is_multiple_of(128),
            "FP8 MoE DeepGEMM group needs rows/cols 128-aligned, got {rows}x{cols}"
        );
        let scale_rows = rows / 128;
        let scale_cols = cols / 128;
        ensure!(
            weight.len() == groups * rows * cols,
            "FP8 MoE grouped host weight bytes {} != expected {}",
            weight.len(),
            groups * rows * cols
        );
        ensure!(
            scales.len() == groups * scale_rows * scale_cols,
            "FP8 MoE grouped host scale values {} != expected {}",
            scales.len(),
            groups * scale_rows * scale_cols
        );
        Ok(Self {
            weight: ctx
                .stream
                .clone_htod(weight)
                .map_err(|e| anyhow!("FP8 MoE grouped weight H2D failed: {e}"))?,
            scales: ctx
                .stream
                .clone_htod(scales)
                .map_err(|e| anyhow!("FP8 MoE grouped scales H2D failed: {e}"))?,
            groups,
            rows,
            cols,
            scale_rows,
            scale_cols,
            sfb_n_contiguous,
        })
    }

    fn offload_to_host(
        &mut self,
        ctx: &DeviceContext,
    ) -> Result<(MoeFp8ExpertGroupHostSnapshot, usize)> {
        let weight = ctx
            .stream
            .clone_dtoh(&self.weight)
            .map_err(|e| anyhow!("FP8 MoE grouped weight D2H failed: {e}"))?;
        let scales = ctx
            .stream
            .clone_dtoh(&self.scales)
            .map_err(|e| anyhow!("FP8 MoE grouped scales D2H failed: {e}"))?;
        let freed =
            weight.len() * std::mem::size_of::<u8>() + scales.len() * std::mem::size_of::<f32>();
        ctx.sync()?;
        self.weight = ctx
            .stream
            .alloc_zeros::<u8>(1)
            .map_err(|e| anyhow!("FP8 MoE grouped weight placeholder alloc failed: {e}"))?;
        self.scales = ctx
            .stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| anyhow!("FP8 MoE grouped scales placeholder alloc failed: {e}"))?;
        Ok((
            MoeFp8ExpertGroupHostSnapshot {
                weight,
                scales,
                groups: self.groups,
                rows: self.rows,
                cols: self.cols,
                sfb_n_contiguous: self.sfb_n_contiguous,
            },
            freed,
        ))
    }

    fn concat(
        ctx: &DeviceContext,
        experts: &[DeviceMatrix],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let groups = experts.len();
        ensure!(groups > 0, "FP8 MoE expert group concat: no local experts");
        let mut group = Self::empty(ctx, groups, rows, cols)?;
        for (g, expert) in experts.iter().enumerate() {
            group.copy_matrix_rows(ctx, g, 0, expert)?;
        }
        Ok(group)
    }

    fn concat_pair_rows(
        ctx: &DeviceContext,
        first: &[DeviceMatrix],
        second: &[DeviceMatrix],
        rows_each: usize,
        cols: usize,
    ) -> Result<Self> {
        ensure!(
            first.len() == second.len() && !first.is_empty(),
            "FP8 MoE fused group needs matching non-empty gate/up experts"
        );
        ensure!(
            rows_each.is_multiple_of(128),
            "FP8 MoE fused group first half rows must be 128-aligned, got {rows_each}"
        );
        let mut group = Self::empty(ctx, first.len(), rows_each * 2, cols)?;
        for (g, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            group.copy_matrix_rows(ctx, g, 0, a)?;
            group.copy_matrix_rows(ctx, g, rows_each, b)?;
        }
        Ok(group)
    }

    fn empty(ctx: &DeviceContext, groups: usize, rows: usize, cols: usize) -> Result<Self> {
        ensure!(groups > 0, "FP8 MoE expert group: groups must be non-zero");
        ensure!(
            rows.is_multiple_of(128) && cols.is_multiple_of(128),
            "FP8 MoE DeepGEMM group needs rows/cols 128-aligned, got {rows}x{cols}"
        );
        let scale_rows = rows.div_ceil(128);
        let scale_cols = cols.div_ceil(128);
        Ok(Self {
            weight: ctx
                .stream
                .alloc_zeros::<u8>(groups * rows * cols)
                .map_err(|e| anyhow!("FP8 MoE grouped weight alloc failed: {e}"))?,
            scales: ctx
                .stream
                .alloc_zeros::<f32>(groups * scale_rows * scale_cols)
                .map_err(|e| anyhow!("FP8 MoE grouped scale alloc failed: {e}"))?,
            groups,
            rows,
            cols,
            scale_rows,
            scale_cols,
            // `empty` backs the Hopper DeepGEMM concat path — K-contiguous SFB.
            sfb_n_contiguous: false,
        })
    }

    fn copy_matrix_rows(
        &mut self,
        ctx: &DeviceContext,
        group: usize,
        row_offset: usize,
        matrix: &DeviceMatrix,
    ) -> Result<()> {
        ensure!(
            group < self.groups,
            "FP8 MoE group index {group} outside groups {}",
            self.groups
        );
        ensure!(
            matrix.weight_format() == WeightFormat::Fp8BlockScaled,
            "FP8 MoE grouped cache needs FP8 block-scaled experts, got {}",
            matrix.weight_format()
        );
        ensure!(
            matrix.rows + row_offset <= self.rows && matrix.cols == self.cols,
            "FP8 MoE grouped cache shape mismatch: matrix {}x{} at row_offset {} into group {}x{}",
            matrix.rows,
            matrix.cols,
            row_offset,
            self.rows,
            self.cols
        );
        ensure!(
            row_offset.is_multiple_of(128)
                && matrix.quant_block_m == 128
                && matrix.quant_block_k == 128,
            "FP8 MoE grouped cache needs 128x128 block metadata, row_offset={} block={}x{}",
            row_offset,
            matrix.quant_block_m,
            matrix.quant_block_k
        );
        let matrix_scale_rows = matrix.rows.div_ceil(128);
        let matrix_scale_cols = matrix.cols.div_ceil(128);
        ensure!(
            matrix.quant_scale_rows == matrix_scale_rows
                && matrix.quant_scale_cols == matrix_scale_cols
                && matrix_scale_cols == self.scale_cols,
            "FP8 MoE grouped cache scale shape {}x{} != expected {}x{}",
            matrix.quant_scale_rows,
            matrix.quant_scale_cols,
            matrix_scale_rows,
            self.scale_cols
        );
        let qweight = matrix
            .qweight_u8
            .as_ref()
            .ok_or_else(|| anyhow!("FP8 MoE grouped cache source missing weight bytes"))?;
        let scales = matrix
            .scale_f32
            .as_ref()
            .ok_or_else(|| anyhow!("FP8 MoE grouped cache source missing f32 scales"))?;
        ensure!(
            qweight.len() == matrix.rows * matrix.cols
                && scales.len() == matrix_scale_rows * self.scale_cols,
            "FP8 MoE grouped cache source lengths mismatch: weight={} scale={}",
            qweight.len(),
            scales.len()
        );

        {
            let src = qweight.slice(0..qweight.len());
            let group_weight_base = group * self.rows * self.cols;
            let start = group_weight_base + row_offset * self.cols;
            let mut dst = self.weight.slice_mut(start..start + qweight.len());
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow!("FP8 MoE grouped weight D2D failed: {e}"))?;
        }
        {
            let src = scales.slice(0..scales.len());
            let group_scale_base = group * self.scale_rows * self.scale_cols;
            let start = group_scale_base + (row_offset / 128) * self.scale_cols;
            let mut dst = self.scales.slice_mut(start..start + scales.len());
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow!("FP8 MoE grouped scale D2D failed: {e}"))?;
        }
        Ok(())
    }

    /// Device weight+scale pointers for the `[num_rows, cols]` sub-matrix of expert
    /// `group`
    /// starting at `row_offset` rows, letting a borrower alias an expert's FP8 slice
    /// zero-copy. `row_offset` and `num_rows` must be 128-aligned (block grid). Returns
    /// (weight_ptr, scale_ptr, rows, cols, block_m=128, block_k=128).
    pub(crate) fn expert_slice_fp8_ptrs(
        &self,
        ctx: &DeviceContext,
        group: usize,
        row_offset: usize,
        num_rows: usize,
    ) -> Option<(u64, u64, usize, usize, usize, usize)> {
        if group >= self.groups || !row_offset.is_multiple_of(128) || !num_rows.is_multiple_of(128)
        {
            return None;
        }
        if row_offset + num_rows > self.rows {
            return None;
        }
        let (wbase, _wg) = self.weight.device_ptr(&ctx.stream);
        let (sbase, _sg) = self.scales.device_ptr(&ctx.stream);
        let group_stride = self.rows * self.cols;
        let weight_ptr = wbase + (group * group_stride + row_offset * self.cols) as u64;
        let scale_group_stride = self.scale_rows * self.scale_cols;
        let scale_elem_size = std::mem::size_of::<f32>() as u64;
        let scale_ptr = sbase
            + ((group * scale_group_stride + (row_offset / 128) * self.scale_cols) as u64
                * scale_elem_size);
        Some((weight_ptr, scale_ptr, num_rows, self.cols, 128, 128))
    }

    fn qweight_ptr_table(&self, ctx: &DeviceContext, row_offset: usize) -> Result<CudaSlice<u64>> {
        ensure!(
            row_offset < self.rows && row_offset.is_multiple_of(128),
            "FP8 MoE qweight ptr row offset {row_offset} invalid for rows {}",
            self.rows
        );
        let (base, _guard) = self.weight.device_ptr(&ctx.stream);
        let group_stride = self.rows * self.cols;
        let row_offset_elems = row_offset * self.cols;
        let host: Vec<u64> = (0..self.groups)
            .map(|g| base + (g * group_stride + row_offset_elems) as u64)
            .collect();
        ctx.stream
            .clone_htod(&host)
            .map_err(|e| anyhow!("FP8 MoE qweight ptr table H2D failed: {e}"))
    }

    fn scale_ptr_table(&self, ctx: &DeviceContext, row_offset: usize) -> Result<CudaSlice<u64>> {
        ensure!(
            row_offset < self.rows && row_offset.is_multiple_of(128),
            "FP8 MoE scale ptr row offset {row_offset} invalid for rows {}",
            self.rows
        );
        let (base, _guard) = self.scales.device_ptr(&ctx.stream);
        let group_stride = self.scale_rows * self.scale_cols;
        let row_offset_elems = (row_offset / 128) * self.scale_cols;
        let elem_size = std::mem::size_of::<f32>() as u64;
        let host: Vec<u64> = (0..self.groups)
            .map(|g| base + ((g * group_stride + row_offset_elems) as u64 * elem_size))
            .collect();
        ctx.stream
            .clone_htod(&host)
            .map_err(|e| anyhow!("FP8 MoE scale ptr table H2D failed: {e}"))
    }
}
