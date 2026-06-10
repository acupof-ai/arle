//! Single-GPU BF16 MoE forward — Qwen3.5/3.6 SparseMoeBlock (all experts local).
//!
//! ```text
//!  router gemm  → logits[T, E]
//!    → infer_moe::route (HOST)         → per-token (expert, weight)
//!    → flatten_routing (HOST)          → route_indices[T*topk], route_weights[T*topk]
//!    → dsv4_count_local_experts        → counts[E]
//!    → dsv4_exclusive_scan_i32         → offsets[E]
//!    → dsv4_pack_local_experts_with_slots
//!                                      → packed_hidden[R,H], packed_route_slot[R],
//!                                        packed_weight[R]   (R = T*topk)
//!    → moe_bf16_grouped_gemm_pair_batch (gate+up)
//!    → silu_mul (unclamped SwiGLU — Qwen3.6 has no clamp)
//!    → moe_bf16_grouped_gemm_batch (down)
//!    → dsv4_scatter_all_route_slots    → route_out[R,H]   (weight·expert_out, by slot)
//!    → dsv4_combine_route_slot_outputs → routed[T,H]      (sum over topk)
//!    → shared expert dense SwiGLU + qwen36_add_shared_expert_gated
//! ```
//!
//! Routing runs on the host (the small `[T, E]` logits are cheap to round-trip
//! through the verified `infer_moe::route`); a device router is a perf follow-up.
//! W4/4-bit is a separate follow-up: the two `moe_bf16_grouped_gemm_*` call sites
//! are the swap points; everything else is dtype-agnostic on BF16 activations.

use infer_moe::RoutingDecision;

/// Host hash routing for a DSv4 hash-routed MoE layer: each token's experts come
/// straight from the `tid2eid` table (`token_id * topk .. + topk`), weighted by
/// the layer's router scores (sqrtsoftplus + normalize + `routed_scaling_factor`,
/// via `config.moe_routes_from_scores`). No learned router gate / bias.
///
/// `logits` is the row-major `[num_tokens, n_routed_experts]` router gemm output
/// (used only for the SCORES that weight the hash-picked experts). Returns one
/// [`RoutingDecision`] per token with exactly `topk` experts, matching the bias
/// path's contract so `flatten_routing` is shared. DSv4-only, so gated on `cuda`
/// (the `deepseek-spec` dep is a `cuda`-feature dependency).
#[cfg(feature = "cuda")]
pub(crate) fn hash_route(
    config: &deepseek_spec::DeepSeekV4Config,
    tid2eid: &[i64],
    tokens: &[u32],
    logits: &[f32],
) -> anyhow::Result<Vec<RoutingDecision>> {
    use infer_moe::ExpertWeight;

    let num_experts = config.n_routed_experts;
    let topk = config.num_experts_per_tok;
    anyhow::ensure!(
        logits.len() == tokens.len() * num_experts,
        "DSv4 hash route logits len {} != tokens {} * experts {num_experts}",
        logits.len(),
        tokens.len()
    );
    let mut decisions = Vec::with_capacity(tokens.len());
    for (token_idx, &token) in tokens.iter().enumerate() {
        let token = token as usize;
        anyhow::ensure!(
            token < config.vocab_size,
            "DSv4 hash route token {token} exceeds vocab_size {}",
            config.vocab_size
        );
        let start = token * topk;
        let end = start + topk;
        anyhow::ensure!(
            end <= tid2eid.len(),
            "DSv4 tid2eid too short: need {end} for token {token}, have {}",
            tid2eid.len()
        );
        let experts: Vec<usize> = tid2eid[start..end]
            .iter()
            .map(|&e| {
                anyhow::ensure!(e >= 0, "DSv4 tid2eid has negative expert id {e}");
                usize::try_from(e)
                    .map_err(|_| anyhow::anyhow!("DSv4 tid2eid expert id {e} overflow"))
            })
            .collect::<anyhow::Result<_>>()?;

        let row = &logits[token_idx * num_experts..(token_idx + 1) * num_experts];
        let scores = config
            .router_scores_from_logits(row)
            .map_err(|e| anyhow::anyhow!("DSv4 hash route scores: {e}"))?;
        let routes = config
            .moe_routes_from_scores(0, token_idx, &scores, None, Some(&experts))
            .map_err(|e| anyhow::anyhow!("DSv4 hash route weights: {e}"))?;
        decisions.push(RoutingDecision {
            experts: routes
                .into_iter()
                .map(|r| ExpertWeight {
                    expert: r.expert_idx,
                    weight: r.weight,
                })
                .collect(),
        });
    }
    Ok(decisions)
}

/// Flatten per-token [`RoutingDecision`]s into the token-major flat buffers the
/// `dsv4_*` kernels read at `route = token * topk + k`: `route_indices` (expert
/// id), `route_weights` (gate weight), each length `num_tokens * topk`. Each
/// decision must carry exactly `topk` experts; selection order is preserved.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn flatten_routing(
    decisions: &[RoutingDecision],
    topk: usize,
) -> anyhow::Result<(Vec<i32>, Vec<f32>)> {
    let total = decisions.len() * topk;
    let mut indices = Vec::with_capacity(total);
    let mut weights = Vec::with_capacity(total);
    for (token, decision) in decisions.iter().enumerate() {
        anyhow::ensure!(
            decision.experts.len() == topk,
            "routing decision for token {token} has {} experts, expected topk {topk}",
            decision.experts.len()
        );
        for ew in &decision.experts {
            indices.push(
                i32::try_from(ew.expert).map_err(|_| {
                    anyhow::anyhow!("expert id {} exceeds i32 kernel ABI", ew.expert)
                })?,
            );
            weights.push(ew.weight);
        }
    }
    Ok((indices, weights))
}

#[cfg(feature = "cuda")]
mod gpu {
    use anyhow::{Result, ensure};
    use cuda_kernels::moe;
    use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, HiddenStates};
    use cuda_kernels::tensor::cache_ptr;
    use half::bf16;
    use infer_moe::MoeConfig;

    use crate::loader::MoeLayerWeights;
    use crate::ops::{gemm_batch, silu_mul};

    /// Single-GPU BF16 MoE forward for one sparse layer (all experts local).
    /// `normed` is the post-LN hidden `[num_tokens, hidden]`; returns the block
    /// output (routed + sigmoid-gated shared expert) as a fresh `HiddenStates`.
    pub(crate) fn moe_forward(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        cfg: &MoeConfig,
    ) -> Result<HiddenStates> {
        let num_tokens = normed.seq_len;
        let hidden_dim = normed.hidden_dim;
        let num_experts = cfg.num_experts;
        let topk = cfg.top_k;
        ensure!(
            weights.router_gate.cols == hidden_dim && weights.router_gate.rows == num_experts,
            "MoE router shape mismatch: gate={}x{} hidden_dim={} num_experts={}",
            weights.router_gate.rows,
            weights.router_gate.cols,
            hidden_dim,
            num_experts
        );
        ensure!(
            weights.gate.len() == num_experts
                && weights.up.len() == num_experts
                && weights.down.len() == num_experts,
            "MoE expert count mismatch: gate={} up={} down={} num_experts={}",
            weights.gate.len(),
            weights.up.len(),
            weights.down.len(),
            num_experts
        );

        // ── 1. Router gemm → logits[T, E] (token-major). ────────────────────
        let mut logits = HiddenStates::zeros(ctx, num_experts, num_tokens)?;
        gemm_batch(ctx, &weights.router_gate, normed, &mut logits)?;

        // ── 2. HOST route (verified infer_moe reference). ───────────────────
        // Sync so the gemm has landed before the D2H read.
        ctx.sync()?;
        let logits_bf16: Vec<bf16> = ctx
            .stream
            .clone_dtoh(&logits.data)
            .map_err(|e| anyhow::anyhow!("MoE router logits D2H failed: {e}"))?;
        let logits_host: Vec<f32> = logits_bf16.iter().map(|&v| v.to_f32()).collect();
        let decisions = infer_moe::route(&logits_host, &[], cfg)
            .map_err(|e| anyhow::anyhow!("MoE host route failed: {e}"))?;
        let (indices_host, weights_host) = super::flatten_routing(&decisions, topk)?;
        let total_routes = num_tokens * topk;
        ensure!(
            indices_host.len() == total_routes && weights_host.len() == total_routes,
            "flattened routing length mismatch"
        );

        let route_indices = ctx
            .stream
            .clone_htod(&indices_host)
            .map_err(|e| anyhow::anyhow!("MoE route-index H2D failed: {e}"))?;
        let route_weights = ctx
            .stream
            .clone_htod(&weights_host)
            .map_err(|e| anyhow::anyhow!("MoE route-weight H2D failed: {e}"))?;

        // ── 3. Per-expert route counts → group offsets. ─────────────────────
        // Kernels write these through raw device pointers, so no Rust `mut`.
        let counts = ctx
            .stream
            .alloc_zeros::<i32>(num_experts)
            .map_err(|e| anyhow::anyhow!("MoE count alloc failed: {e}"))?;
        let offsets = ctx
            .stream
            .alloc_zeros::<i32>(num_experts)
            .map_err(|e| anyhow::anyhow!("MoE offset alloc failed: {e}"))?;
        let scan_total = ctx
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!("MoE scan total alloc failed: {e}"))?;
        // SAFETY: all buffers are valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_count_local_experts(
                cache_ptr(&route_indices, ctx),
                cache_ptr(&counts, ctx),
                num_tokens,
                topk,
                0,
                num_experts,
                ctx.stream.cu_stream(),
            )?;
            moe::dsv4_exclusive_scan_i32(
                cache_ptr(&counts, ctx),
                cache_ptr(&offsets, ctx),
                cache_ptr(&scan_total, ctx),
                num_experts,
                ctx.stream.cu_stream(),
            )?;
        }

        // ── 4. Pack routed tokens grouped-by-expert (with route slots). ─────
        let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, total_routes)?;
        let packed_route_slot = ctx
            .stream
            .alloc_zeros::<i32>(total_routes.max(1))
            .map_err(|e| anyhow::anyhow!("MoE packed_route_slot alloc failed: {e}"))?;
        let packed_weight = ctx
            .stream
            .alloc_zeros::<f32>(total_routes.max(1))
            .map_err(|e| anyhow::anyhow!("MoE packed_weight alloc failed: {e}"))?;
        let cursors = ctx
            .stream
            .alloc_zeros::<i32>(num_experts)
            .map_err(|e| anyhow::anyhow!("MoE cursors alloc failed: {e}"))?;
        // SAFETY: buffers valid on ctx.stream; shapes checked by the kernel.
        unsafe {
            moe::dsv4_pack_local_experts_with_slots(
                cache_ptr(&normed.data, ctx),
                cache_ptr(&route_indices, ctx),
                cache_ptr(&route_weights, ctx),
                cache_ptr(&offsets, ctx),
                cache_ptr(&cursors, ctx),
                cache_ptr(&packed_hidden.data, ctx),
                cache_ptr(&packed_route_slot, ctx),
                cache_ptr(&packed_weight, ctx),
                num_tokens,
                hidden_dim,
                topk,
                0,
                num_experts,
                ctx.stream.cu_stream(),
            )?;
        }

        // Identity compact→global remap: single GPU runs every expert, so the
        // 0..E table is the identity walk (explicit table keeps the RawDevicePtr
        // contract — no null hack).
        let expert_index_table: Vec<i32> = (0..num_experts as i32).collect();
        let expert_indices = ctx
            .stream
            .clone_htod(&expert_index_table)
            .map_err(|e| anyhow::anyhow!("MoE expert-index table H2D failed: {e}"))?;

        // ── 5. Grouped expert GEMM (gate + up paired). ──────────────────────
        let moe_inter = weights.gate[0].rows;
        expert_shape_ok(&weights.gate[0], &weights.up[0], hidden_dim, moe_inter)?;
        let gate_out = HiddenStates::zeros(ctx, moe_inter, total_routes)?;
        let up_out = HiddenStates::zeros(ctx, moe_inter, total_routes)?;
        // `max_count` only sizes grid Y; total_routes is a safe upper bound.
        let max_count = total_routes.max(1);
        // SAFETY: weight-ptr tables + packed buffers valid on ctx.stream.
        unsafe {
            moe::moe_bf16_grouped_gemm_pair_batch(
                &weights.gate_ptrs,
                &weights.up_ptrs,
                cache_ptr(&packed_hidden.data, ctx),
                cache_ptr(&gate_out.data, ctx),
                cache_ptr(&up_out.data, ctx),
                cache_ptr(&offsets, ctx),
                cache_ptr(&counts, ctx),
                cache_ptr(&expert_indices, ctx),
                num_experts,
                max_count,
                moe_inter,
                hidden_dim,
                ctx,
                ctx.stream.cu_stream(),
            )?;
        }

        // ── 6. SwiGLU (UNCLAMPED — Qwen3.6 MoE has no swiglu clamp). ────────
        let mut act = HiddenStates::zeros(ctx, moe_inter, total_routes)?;
        silu_mul(ctx, &gate_out, &up_out, &mut act)?;

        // ── 7. Grouped down GEMM → expert_out[R, H]. ────────────────────────
        ensure!(
            weights.down[0].cols == moe_inter && weights.down[0].rows == hidden_dim,
            "MoE expert down shape {}x{} != hidden_dim {} / moe_inter {}",
            weights.down[0].rows,
            weights.down[0].cols,
            hidden_dim,
            moe_inter
        );
        let expert_out = HiddenStates::zeros(ctx, hidden_dim, total_routes)?;
        // SAFETY: down weight table + act/expert_out valid on ctx.stream.
        unsafe {
            moe::moe_bf16_grouped_gemm_batch(
                &weights.down_ptrs,
                cache_ptr(&act.data, ctx),
                cache_ptr(&expert_out.data, ctx),
                cache_ptr(&offsets, ctx),
                cache_ptr(&counts, ctx),
                cache_ptr(&expert_indices, ctx),
                num_experts,
                max_count,
                hidden_dim,
                moe_inter,
                ctx,
                ctx.stream.cu_stream(),
            )?;
        }

        // ── 8. Scatter weighted expert outputs to route slots, combine topk. ─
        // route_out[slot] = weight · expert_out[slot] (zero-init covers unwritten
        // slots); combine sums over topk.
        let route_out = HiddenStates::zeros(ctx, hidden_dim, total_routes)?;
        let routed = HiddenStates::zeros(ctx, hidden_dim, num_tokens)?;
        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_scatter_all_route_slots(
                cache_ptr(&expert_out.data, ctx),
                cache_ptr(&route_out.data, ctx),
                cache_ptr(&packed_route_slot, ctx),
                cache_ptr(&packed_weight, ctx),
                total_routes,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
            moe::dsv4_combine_route_slot_outputs(
                cache_ptr(&route_out.data, ctx),
                cache_ptr(&routed.data, ctx),
                num_tokens,
                topk,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
        }

        // ── 9. Shared expert: dense SwiGLU · sigmoid(x @ shared_gate_router).
        let shared = shared_expert_forward(ctx, weights, normed)?;
        let mut gate_logit = HiddenStates::zeros(ctx, 1, num_tokens)?;
        gemm_batch(ctx, &weights.shared_gate_router, normed, &mut gate_logit)?;
        // SAFETY: routed/shared/gate_logit valid on ctx.stream.
        unsafe {
            moe::qwen36_add_shared_expert_gated(
                cache_ptr(&routed.data, ctx),
                cache_ptr(&shared.data, ctx),
                cache_ptr(&gate_logit.data, ctx),
                num_tokens,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
        }

        Ok(routed)
    }

    /// Dense shared-expert SwiGLU: `down(silu(gate(x)) * up(x))`.
    fn shared_expert_forward(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
    ) -> Result<HiddenStates> {
        let shared_inter = weights.shared_gate.rows;
        let hidden_dim = normed.hidden_dim;
        let mut gate = HiddenStates::zeros(ctx, shared_inter, normed.seq_len)?;
        let mut up = HiddenStates::zeros(ctx, shared_inter, normed.seq_len)?;
        gemm_batch(ctx, &weights.shared_gate, normed, &mut gate)?;
        gemm_batch(ctx, &weights.shared_up, normed, &mut up)?;
        let mut act = HiddenStates::zeros(ctx, shared_inter, normed.seq_len)?;
        silu_mul(ctx, &gate, &up, &mut act)?;
        let mut out = HiddenStates::zeros(ctx, hidden_dim, normed.seq_len)?;
        gemm_batch(ctx, &weights.shared_down, &act, &mut out)?;
        Ok(out)
    }

    fn expert_shape_ok(
        gate: &DeviceMatrix,
        up: &DeviceMatrix,
        hidden_dim: usize,
        moe_inter: usize,
    ) -> Result<()> {
        ensure!(
            gate.cols == hidden_dim && up.cols == hidden_dim && up.rows == moe_inter,
            "MoE expert gate/up shape mismatch: gate={}x{} up={}x{} hidden_dim={} moe_inter={}",
            gate.rows,
            gate.cols,
            up.rows,
            up.cols,
            hidden_dim,
            moe_inter
        );
        Ok(())
    }
}

// The DSv4 FP8 MoE forward is consumed by the Piece 2 `model.rs` layer loop
// (the `RealCudaExecutor` DSv4 branch); until that integration edit lands there
// is no in-crate caller, matching the `dsv4.rs` pending-consumer gate. The
// `dead_code`/`unused_imports` allowance marks pending-consumer infra, not cruft
// (see `feedback_necessity_not_callers`).
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod dsv4_gpu {
    use anyhow::{Result, ensure};
    use cuda_kernels::moe;
    use cuda_kernels::prelude::{DeviceContext, HiddenStates};
    use cuda_kernels::tensor::{Dsv4Fp8DeepGemmWeightCache, RawDevicePtr, cache_ptr};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
    use half::bf16;
    use std::sync::Arc;

    use crate::dsv4::{Dsv4ForwardKeepalive, Dsv4Model, Dsv4MoeLayer};
    use crate::moe_config::ExpertSplit;
    use crate::ops::gemm_batch;

    const CONTIG_ROUTE_ALIGN: usize = 128;

    struct DeviceRouting {
        indices: CudaSlice<i32>,
        weights: CudaSlice<f32>,
    }

    pub(crate) struct Dsv4MoeDecodeScratch {
        topk: usize,
        num_groups: usize,
        hidden_dim: usize,
        intermediate: usize,
        shared_intermediate: usize,
        route_indices: CudaSlice<i32>,
        route_weights: CudaSlice<f32>,
        token_ids: CudaSlice<u32>,
        router_logits: HiddenStates,
        counts: CudaSlice<i32>,
        offsets: CudaSlice<i32>,
        scan_total: CudaSlice<i32>,
        packed_hidden: HiddenStates,
        packed_route_slot: CudaSlice<i32>,
        packed_weight: CudaSlice<f32>,
        cursors: CudaSlice<i32>,
        route_out: HiddenStates,
        grouped: Dsv4GroupedDecodeScratch,
        grouped_contig: Dsv4GroupedContiguousDecodeScratch,
        shared: Dsv4SharedDecodeScratch,
    }

    struct Dsv4GroupedDecodeScratch {
        max_m: usize,
        scale_stride_m: usize,
        input_fp8: CudaSlice<u8>,
        input_scales: CudaSlice<f32>,
        w13_out: HiddenStates,
        act_fp8: CudaSlice<u8>,
        act_scales: CudaSlice<f32>,
        out_padded: HiddenStates,
        out_compact: HiddenStates,
        masked_m: CudaSlice<i32>,
        active_experts: CudaSlice<i32>,
    }

    struct Dsv4GroupedContiguousDecodeScratch {
        rows: usize,
        scale_stride_m: usize,
        input_fp8: CudaSlice<u8>,
        input_scales: CudaSlice<f32>,
        w13_out: HiddenStates,
        act_fp8: CudaSlice<u8>,
        act_scales: CudaSlice<f32>,
        out: HiddenStates,
        packed_hidden: HiddenStates,
        packed_route_slot: CudaSlice<i32>,
        packed_weight: CudaSlice<f32>,
        m_indices: CudaSlice<i32>,
        active_experts: CudaSlice<i32>,
        active_offsets: CudaSlice<i32>,
        active_counts: CudaSlice<i32>,
    }

    struct Dsv4SharedDecodeScratch {
        max_m: usize,
        scale_stride_m: usize,
        input_fp8: CudaSlice<u8>,
        input_scales: CudaSlice<f32>,
        w13_out: HiddenStates,
        act_fp8: CudaSlice<u8>,
        act_scales: CudaSlice<f32>,
        out: HiddenStates,
        active_experts: CudaSlice<i32>,
        active_offsets: CudaSlice<i32>,
        counts: CudaSlice<i32>,
        masked_m: CudaSlice<i32>,
    }

    impl Dsv4MoeDecodeScratch {
        pub(crate) fn new(
            ctx: &DeviceContext,
            cfg: &infer_moe::MoeConfig,
            split: &ExpertSplit,
            layer: &Dsv4MoeLayer,
        ) -> Result<Self> {
            let topk = cfg.top_k;
            let num_groups = layer.num_groups;
            let hidden_dim = layer.hidden_dim;
            let intermediate = layer.intermediate;
            let shared_intermediate = layer.shared_w2.cols;
            ensure!(
                topk > 0 && num_groups == split.experts_per_rank,
                "DSv4 decode scratch shape mismatch: topk={topk} groups={num_groups} experts_per_rank={}",
                split.experts_per_rank
            );
            ensure!(
                hidden_dim.is_multiple_of(128)
                    && intermediate.is_multiple_of(128)
                    && shared_intermediate.is_multiple_of(128),
                "DSv4 decode scratch needs 128-aligned dims: H={hidden_dim} I={intermediate} shared_I={shared_intermediate}"
            );

            let route_indices = ctx.stream.alloc_zeros::<i32>(topk).map_err(|e| {
                anyhow::anyhow!("DSv4 decode route-index scratch alloc failed: {e}")
            })?;
            let route_weights = ctx.stream.alloc_zeros::<f32>(topk).map_err(|e| {
                anyhow::anyhow!("DSv4 decode route-weight scratch alloc failed: {e}")
            })?;
            let token_ids = ctx
                .stream
                .alloc_zeros::<u32>(1)
                .map_err(|e| anyhow::anyhow!("DSv4 decode token-id scratch alloc failed: {e}"))?;
            let router_logits = unsafe { HiddenStates::uninit(ctx, cfg.num_experts, 1)? };
            let counts = ctx
                .stream
                .alloc_zeros::<i32>(num_groups)
                .map_err(|e| anyhow::anyhow!("DSv4 decode count scratch alloc failed: {e}"))?;
            let offsets = ctx
                .stream
                .alloc_zeros::<i32>(num_groups)
                .map_err(|e| anyhow::anyhow!("DSv4 decode offset scratch alloc failed: {e}"))?;
            let scan_total = ctx
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| anyhow::anyhow!("DSv4 decode scan-total scratch alloc failed: {e}"))?;
            let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, topk)?;
            let packed_route_slot = ctx
                .stream
                .alloc_zeros::<i32>(topk)
                .map_err(|e| anyhow::anyhow!("DSv4 decode route-slot scratch alloc failed: {e}"))?;
            let packed_weight = ctx.stream.alloc_zeros::<f32>(topk).map_err(|e| {
                anyhow::anyhow!("DSv4 decode packed-weight scratch alloc failed: {e}")
            })?;
            let cursors = ctx
                .stream
                .alloc_zeros::<i32>(num_groups)
                .map_err(|e| anyhow::anyhow!("DSv4 decode cursor scratch alloc failed: {e}"))?;
            let route_out = HiddenStates::zeros(ctx, hidden_dim, topk)?;
            let grouped =
                Dsv4GroupedDecodeScratch::new(ctx, num_groups, hidden_dim, intermediate, topk)?;
            let grouped_contig =
                Dsv4GroupedContiguousDecodeScratch::new(ctx, hidden_dim, intermediate, topk)?;
            let shared = Dsv4SharedDecodeScratch::new(ctx, hidden_dim, shared_intermediate)?;

            Ok(Self {
                topk,
                num_groups,
                hidden_dim,
                intermediate,
                shared_intermediate,
                route_indices,
                route_weights,
                token_ids,
                router_logits,
                counts,
                offsets,
                scan_total,
                packed_hidden,
                packed_route_slot,
                packed_weight,
                cursors,
                route_out,
                grouped,
                grouped_contig,
                shared,
            })
        }

        fn reset_routed(&mut self, ctx: &DeviceContext) -> Result<()> {
            ctx.stream
                .memset_zeros(&mut self.counts)
                .map_err(|e| anyhow::anyhow!("DSv4 decode count scratch reset failed: {e}"))?;
            ctx.stream
                .memset_zeros(&mut self.cursors)
                .map_err(|e| anyhow::anyhow!("DSv4 decode cursor scratch reset failed: {e}"))?;
            ctx.stream
                .memset_zeros(&mut self.packed_weight)
                .map_err(|e| {
                    anyhow::anyhow!("DSv4 decode packed-weight scratch reset failed: {e}")
                })?;
            ctx.stream
                .memset_zeros(&mut self.route_out.data)
                .map_err(|e| anyhow::anyhow!("DSv4 decode route-out scratch reset failed: {e}"))?;
            memset_i32_minus_one(ctx, &mut self.packed_route_slot)?;
            // grouped_contig.{packed_route_slot,m_indices} are consumed ONLY on the
            // contiguous decode path (the `use_contiguous` branch reads them); on the
            // default masked path they are never read, so skip the per-layer memset
            // there — pure waste (review #6, the measured 10.4% memset bucket).
            if use_contiguous_decode_moe() {
                memset_i32_minus_one(ctx, &mut self.grouped_contig.packed_route_slot)?;
                memset_i32_minus_one(ctx, &mut self.grouped_contig.m_indices)?;
            }
            Ok(())
        }

        fn validate_routed(
            &self,
            hidden_dim: usize,
            topk: usize,
            layer: &Dsv4MoeLayer,
        ) -> Result<()> {
            ensure!(
                self.topk == topk
                    && self.num_groups == layer.num_groups
                    && self.hidden_dim == hidden_dim
                    && self.intermediate == layer.intermediate,
                "DSv4 decode scratch mismatch: scratch topk={} groups={} H={} I={} vs topk={topk} groups={} H={hidden_dim} I={}",
                self.topk,
                self.num_groups,
                self.hidden_dim,
                self.intermediate,
                layer.num_groups,
                layer.intermediate
            );
            Ok(())
        }
    }

    impl Dsv4GroupedDecodeScratch {
        fn new(
            ctx: &DeviceContext,
            num_groups: usize,
            hidden_dim: usize,
            intermediate: usize,
            route_capacity: usize,
        ) -> Result<Self> {
            let max_m = route_capacity.max(128);
            let scale_stride_m = max_m.div_ceil(4) * 4;
            let rows = num_groups * max_m;
            let hidden_scale_cols = hidden_dim.div_ceil(128);
            let inter_scale_cols = intermediate.div_ceil(128);
            let input_fp8 = alloc_u8(ctx, rows * hidden_dim)?;
            let input_scales =
                alloc_zeros_f32(ctx, num_groups * scale_stride_m * hidden_scale_cols)?;
            let w13_out = HiddenStates::zeros(ctx, 2 * intermediate, rows)?;
            let act_fp8 = alloc_u8(ctx, rows * intermediate)?;
            let act_scales = alloc_zeros_f32(ctx, num_groups * scale_stride_m * inter_scale_cols)?;
            let out_padded = HiddenStates::zeros(ctx, hidden_dim, rows)?;
            let out_compact = HiddenStates::zeros(ctx, hidden_dim, route_capacity.max(1))?;
            let masked_m = ctx
                .stream
                .alloc_zeros::<i32>(num_groups)
                .map_err(|e| anyhow::anyhow!("DSv4 decode masked-m scratch alloc failed: {e}"))?;
            let active_experts = ctx
                .stream
                .clone_htod(&(0..num_groups as i32).collect::<Vec<i32>>())
                .map_err(|e| {
                    anyhow::anyhow!("DSv4 decode active-expert scratch H2D failed: {e}")
                })?;
            Ok(Self {
                max_m,
                scale_stride_m,
                input_fp8,
                input_scales,
                w13_out,
                act_fp8,
                act_scales,
                out_padded,
                out_compact,
                masked_m,
                active_experts,
            })
        }
    }

    impl Dsv4GroupedContiguousDecodeScratch {
        fn new(
            ctx: &DeviceContext,
            hidden_dim: usize,
            intermediate: usize,
            route_capacity: usize,
        ) -> Result<Self> {
            // DeepGEMM's contiguous grouped layout expects each expert segment
            // to be M-tile aligned, with invalid padding rows marked -1. Decode
            // has at most `topk` active routes, so give each route one aligned
            // tile instead of materialising every local expert group.
            let rows = route_capacity.max(1) * CONTIG_ROUTE_ALIGN;
            let scale_stride_m = rows.div_ceil(4) * 4;
            let hidden_scale_cols = hidden_dim.div_ceil(128);
            let inter_scale_cols = intermediate.div_ceil(128);
            let input_fp8 = alloc_u8(ctx, rows * hidden_dim)?;
            let input_scales = alloc_zeros_f32(ctx, scale_stride_m * hidden_scale_cols)?;
            let w13_out = HiddenStates::zeros(ctx, 2 * intermediate, rows)?;
            let act_fp8 = alloc_u8(ctx, rows * intermediate)?;
            let act_scales = alloc_zeros_f32(ctx, scale_stride_m * inter_scale_cols)?;
            let out = HiddenStates::zeros(ctx, hidden_dim, rows)?;
            let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, rows)?;
            let packed_route_slot = ctx.stream.alloc_zeros::<i32>(rows).map_err(|e| {
                anyhow::anyhow!("DSv4 contiguous route-slot scratch alloc failed: {e}")
            })?;
            let packed_weight = ctx.stream.alloc_zeros::<f32>(rows).map_err(|e| {
                anyhow::anyhow!("DSv4 contiguous packed-weight scratch alloc failed: {e}")
            })?;
            let m_indices = ctx
                .stream
                .alloc_zeros::<i32>(rows)
                .map_err(|e| anyhow::anyhow!("DSv4 decode contiguous m-index alloc failed: {e}"))?;
            let active_experts = ctx
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| anyhow::anyhow!("DSv4 contiguous active scratch H2D failed: {e}"))?;
            let active_offsets = ctx
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| anyhow::anyhow!("DSv4 contiguous offset scratch H2D failed: {e}"))?;
            let active_counts = ctx
                .stream
                .clone_htod(&[i32::try_from(rows)?])
                .map_err(|e| anyhow::anyhow!("DSv4 contiguous count scratch H2D failed: {e}"))?;
            Ok(Self {
                rows,
                scale_stride_m,
                input_fp8,
                input_scales,
                w13_out,
                act_fp8,
                act_scales,
                out,
                packed_hidden,
                packed_route_slot,
                packed_weight,
                m_indices,
                active_experts,
                active_offsets,
                active_counts,
            })
        }
    }

    impl Dsv4SharedDecodeScratch {
        fn new(ctx: &DeviceContext, hidden_dim: usize, shared_inter: usize) -> Result<Self> {
            let max_m = 128usize;
            let scale_stride_m = max_m.div_ceil(4) * 4;
            let hidden_scale_cols = hidden_dim.div_ceil(128);
            let inter_scale_cols = shared_inter.div_ceil(128);
            let input_fp8 = alloc_u8(ctx, max_m * hidden_dim)?;
            let input_scales = alloc_zeros_f32(ctx, scale_stride_m * hidden_scale_cols)?;
            let w13_out = HiddenStates::zeros(ctx, 2 * shared_inter, max_m)?;
            let act_fp8 = alloc_u8(ctx, max_m * shared_inter)?;
            let act_scales = alloc_zeros_f32(ctx, scale_stride_m * inter_scale_cols)?;
            let out = HiddenStates::zeros(ctx, hidden_dim, max_m)?;
            let active_experts = ctx
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| anyhow::anyhow!("DSv4 shared active scratch H2D failed: {e}"))?;
            let active_offsets = ctx
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| anyhow::anyhow!("DSv4 shared offset scratch H2D failed: {e}"))?;
            let counts = ctx
                .stream
                .clone_htod(&[1i32])
                .map_err(|e| anyhow::anyhow!("DSv4 shared count scratch H2D failed: {e}"))?;
            let masked_m = ctx
                .stream
                .clone_htod(&[1i32])
                .map_err(|e| anyhow::anyhow!("DSv4 shared masked-m scratch H2D failed: {e}"))?;
            Ok(Self {
                max_m,
                scale_stride_m,
                input_fp8,
                input_scales,
                w13_out,
                act_fp8,
                act_scales,
                out,
                active_experts,
                active_offsets,
                counts,
                masked_m,
            })
        }
    }

    fn memset_i32_minus_one(ctx: &DeviceContext, slice: &mut CudaSlice<i32>) -> Result<()> {
        let bytes = slice.len() * std::mem::size_of::<i32>();
        let (ptr, _record) = slice.device_ptr_mut(&ctx.stream);
        unsafe {
            cudarc::driver::result::memset_d8_async(ptr, 0xFF, bytes, ctx.stream.cu_stream())
                .map_err(|e| anyhow::anyhow!("DSv4 decode i32 -1 memset failed: {e}"))?;
        }
        Ok(())
    }

    fn use_gpu_router() -> bool {
        // Default ON: route fully on-device (no per-layer logits D2H + `ctx.sync()`).
        // DSv4-Flash ships no group-limited routing (`MoeConfig::dsv4` hardcodes
        // `n_group/topk_group = None`, config.rs:151), so the device kernel's plain
        // bias-corrected top-k is algorithmically identical to the host
        // `route_token`. The decode-graph path (`dsv4_moe_forward_decode_graph`)
        // already calls this same kernel unconditionally and is token-verified, so
        // the eager/deepep paths are just converging onto the proven path. Opt out
        // with `ARLE_DSV4_GPU_ROUTER=0`.
        !matches!(std::env::var("ARLE_DSV4_GPU_ROUTER").as_deref(), Ok("0"))
    }

    fn use_contiguous_decode_moe() -> bool {
        matches!(
            std::env::var("ARLE_DSV4_MOE_CONTIG_DECODE").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        )
    }

    fn dsv4_route_device(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        logits: &HiddenStates,
        decode_scratch: Option<&mut Dsv4MoeDecodeScratch>,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<DeviceRouting> {
        use deepseek_spec::DeepSeekV4MoeRoutingKind;

        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let num_tokens = logits.seq_len;
        let total_routes = num_tokens * cfg.top_k;
        ensure!(
            tokens.len() == num_tokens,
            "DSv4 device route token count {} != logits seq_len {num_tokens}",
            tokens.len()
        );
        ensure!(
            logits.hidden_dim == cfg.num_experts,
            "DSv4 device route logits hidden_dim {} != num_experts {}",
            logits.hidden_dim,
            cfg.num_experts
        );

        let mut decode_scratch = decode_scratch;
        let (route_indices, route_weights, token_ids) = if let Some(scratch) =
            decode_scratch.as_deref_mut()
        {
            ensure!(
                num_tokens == 1 && total_routes == scratch.topk,
                "DSv4 decode route scratch only supports one token: tokens={num_tokens} routes={total_routes} scratch_topk={}",
                scratch.topk
            );
            let token_ids = if matches!(layer.routing_kind, DeepSeekV4MoeRoutingKind::Hash) {
                ctx.stream
                    .memcpy_htod(tokens, &mut scratch.token_ids)
                    .map_err(|e| anyhow::anyhow!("DSv4 decode route token-id H2D failed: {e}"))?;
                Some(scratch.token_ids.clone())
            } else {
                None
            };
            (
                scratch.route_indices.clone(),
                scratch.route_weights.clone(),
                token_ids,
            )
        } else {
            let route_indices = ctx
                .stream
                .alloc_zeros::<i32>(total_routes)
                .map_err(|e| anyhow::anyhow!("DSv4 device route-index alloc failed: {e}"))?;
            let route_weights = ctx
                .stream
                .alloc_zeros::<f32>(total_routes)
                .map_err(|e| anyhow::anyhow!("DSv4 device route-weight alloc failed: {e}"))?;
            let token_ids = if matches!(layer.routing_kind, DeepSeekV4MoeRoutingKind::Hash) {
                let token_ids = ctx
                    .stream
                    .clone_htod(tokens)
                    .map_err(|e| anyhow::anyhow!("DSv4 device route token-id H2D failed: {e}"))?;
                keepalive.keep_route_u32(&token_ids);
                Some(token_ids)
            } else {
                None
            };
            (route_indices, route_weights, token_ids)
        };

        let routing_kind = match layer.routing_kind {
            DeepSeekV4MoeRoutingKind::Hash => 0,
            DeepSeekV4MoeRoutingKind::LearnedBias => 1,
        };
        let bias = layer
            .gate_bias
            .as_ref()
            .map(|bias| cache_ptr(&bias.data, ctx));
        let tid2eid = layer
            .hash_tid2eid_device
            .as_ref()
            .map(|table| cache_ptr(table, ctx));
        let token_ids_ptr = token_ids.as_ref().map(|ids| cache_ptr(ids, ctx));

        keepalive.keep_route_hidden(logits);
        keepalive.keep_route_i32(&route_indices);
        keepalive.keep_route_f32(&route_weights);
        // SAFETY: buffers are allocated for `[num_tokens * topk]`; optional
        // pointers are validated by the wrapper according to `routing_kind`.
        unsafe {
            moe::dsv4_route(
                cache_ptr(&logits.data, ctx),
                bias,
                tid2eid,
                token_ids_ptr,
                cache_ptr(&route_indices, ctx),
                cache_ptr(&route_weights, ctx),
                num_tokens,
                cfg.num_experts,
                cfg.top_k,
                routing_kind,
                cfg.scoring_func.scoring_kind(),
                cfg.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )?;
        }
        Ok(DeviceRouting {
            indices: route_indices,
            weights: route_weights,
        })
    }

    /// Decode-graph MoE path: all mutable buffers are fixed scratch addresses,
    /// and the token id is read from a caller-staged device buffer so graph
    /// replay does not bake the capture step's host token value.
    pub(crate) fn dsv4_moe_forward_decode_graph(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        token_ids: &CudaSlice<u32>,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        scratch: &mut Dsv4MoeDecodeScratch,
    ) -> Result<()> {
        use deepseek_spec::DeepSeekV4MoeRoutingKind;

        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let split = &model.split;
        let swiglu_limit = model.config.swiglu_limit;
        let topk = cfg.top_k;
        let hidden_dim = hidden.hidden_dim;
        ensure!(
            hidden.seq_len == 1 && token_ids.len() == 1,
            "DSv4 decode-graph MoE requires B=1, got hidden seq={} token_ids={}",
            hidden.seq_len,
            token_ids.len()
        );
        scratch.validate_routed(hidden_dim, topk, layer)?;
        scratch.reset_routed(ctx)?;
        moe::dsv4_deepgemm_native_preflight()?;

        gemm_batch(ctx, &layer.gate, hidden, &mut scratch.router_logits)?;

        let routing_kind = match layer.routing_kind {
            DeepSeekV4MoeRoutingKind::Hash => 0,
            DeepSeekV4MoeRoutingKind::LearnedBias => 1,
        };
        let bias = layer
            .gate_bias
            .as_ref()
            .map(|bias| cache_ptr(&bias.data, ctx));
        let tid2eid = layer
            .hash_tid2eid_device
            .as_ref()
            .map(|table| cache_ptr(table, ctx));
        let token_ids_ptr = if matches!(layer.routing_kind, DeepSeekV4MoeRoutingKind::Hash) {
            Some(cache_ptr(token_ids, ctx))
        } else {
            None
        };
        unsafe {
            moe::dsv4_route(
                cache_ptr(&scratch.router_logits.data, ctx),
                bias,
                tid2eid,
                token_ids_ptr,
                cache_ptr(&scratch.route_indices, ctx),
                cache_ptr(&scratch.route_weights, ctx),
                1,
                cfg.num_experts,
                cfg.top_k,
                routing_kind,
                cfg.scoring_func.scoring_kind(),
                cfg.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )?;
        }

        let route_indices = cache_ptr(&scratch.route_indices, ctx);
        let route_weights = cache_ptr(&scratch.route_weights, ctx);
        dsv4_moe_forward_decode_pooled(
            ctx,
            layer,
            split,
            route_indices,
            route_weights,
            hidden,
            out,
            topk,
            split.local_expert_start,
            swiglu_limit,
            scratch,
        )
    }

    /// FP8 DeepGEMM MoE forward for one DSv4 routed-MoE layer (this EP rank's
    /// experts only). Mirrors the BF16 [`super::gpu::moe_forward`] route/pack/
    /// scatter/combine plumbing but swaps the two BF16 grouped GEMMs for the
    /// native DeepGEMM 5-call FP8 pipeline (`f8f8bf16`, 128-block scale).
    ///
    /// Routing is per-layer: bias-routed layers run the learned router gemm +
    /// `noaux_tc` correction bias through `infer_moe::route`; hash-routed layers
    /// (`layer.hash_tid2eid` present) pick experts directly from the `tid2eid`
    /// table by token id (no router gate), weighting them by the router scores.
    ///
    /// `tokens` are the input token ids (needed for hash routing); `hidden` is
    /// the post-LN `[num_tokens, hidden]`; `out` receives routed experts only.
    /// Callers all-reduce this EP-sharded output, then add the replicated shared
    /// expert exactly once per rank via [`dsv4_shared_expert_forward`].
    pub(crate) fn dsv4_moe_forward(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        decode_scratch: Option<&mut Dsv4MoeDecodeScratch>,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let split = &model.split;
        let swiglu_limit = model.config.swiglu_limit;

        let num_tokens = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let topk = cfg.top_k;
        let experts_per_rank = split.experts_per_rank;
        let local_start = split.local_expert_start;

        ensure!(
            tokens.len() == num_tokens,
            "DSv4 MoE token count {} != hidden seq_len {num_tokens}",
            tokens.len()
        );
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == num_tokens,
            "DSv4 MoE out shape {}x{} != hidden {}x{}",
            out.hidden_dim,
            out.seq_len,
            hidden_dim,
            num_tokens
        );
        ensure!(
            layer.gate.cols == hidden_dim && layer.gate.rows == cfg.num_experts,
            "DSv4 router shape mismatch: gate={}x{} hidden={} num_experts={}",
            layer.gate.rows,
            layer.gate.cols,
            hidden_dim,
            cfg.num_experts
        );
        ensure!(
            layer.num_groups == experts_per_rank,
            "DSv4 expert group count {} != experts_per_rank {experts_per_rank}",
            layer.num_groups
        );
        ensure!(
            layer.hidden_dim == hidden_dim,
            "DSv4 expert hidden dim {} != runtime hidden dim {hidden_dim}",
            layer.hidden_dim
        );
        if let Some(scratch) = decode_scratch.as_ref() {
            scratch.validate_routed(hidden_dim, topk, layer)?;
            ensure!(
                num_tokens == 1,
                "DSv4 decode MoE scratch is only valid for one-token decode, got {num_tokens}"
            );
        }

        // Fail loud if the native DeepGEMM bridge is a build-time stub.
        moe::dsv4_deepgemm_native_preflight()?;

        // ── 1+2. Router gemm → logits[T, E] → route (host oracle or device). ───
        let total_routes = num_tokens * topk;
        let mut decode_scratch = decode_scratch;
        let (route_indices, route_weights) =
            crate::stage_profile::profile(ctx, "dsv4/stage/moe_route", || -> Result<_> {
                let _nvtx = crate::nvtx::range("dsv4/moe_route");
                // SAFETY: router gemm writes the full logits buffer.
                let mut logits = unsafe { HiddenStates::uninit(ctx, cfg.num_experts, num_tokens)? };
                gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
                if use_gpu_router() {
                    let routing = dsv4_route_device(
                        model,
                        layer,
                        tokens,
                        &logits,
                        decode_scratch.as_deref_mut(),
                        keepalive,
                    )?;
                    Ok((routing.indices, routing.weights))
                } else {
                    keepalive.keep_hidden(&logits);
                    ctx.sync()?;
                    let logits_bf16: Vec<bf16> = ctx
                        .stream
                        .clone_dtoh(&logits.data)
                        .map_err(|e| anyhow::anyhow!("DSv4 router logits D2H failed: {e}"))?;
                    let logits_host: Vec<f32> = logits_bf16.iter().map(|&v| v.to_f32()).collect();
                    let decisions =
                        dsv4_route(ctx, &model.config, cfg, layer, tokens, &logits_host)?;
                    let (indices_host, weights_host) = super::flatten_routing(&decisions, topk)?;

                    let route_indices = ctx
                        .stream
                        .clone_htod(&indices_host)
                        .map_err(|e| anyhow::anyhow!("DSv4 route-index H2D failed: {e}"))?;
                    let route_weights = ctx
                        .stream
                        .clone_htod(&weights_host)
                        .map_err(|e| anyhow::anyhow!("DSv4 route-weight H2D failed: {e}"))?;
                    keepalive.keep_i32(&route_indices);
                    keepalive.keep_f32(&route_weights);
                    Ok((route_indices, route_weights))
                }
            })?;

        if let Some(scratch) = decode_scratch.as_deref_mut() {
            let route_indices = cache_ptr(&route_indices, ctx);
            let route_weights = cache_ptr(&route_weights, ctx);
            return dsv4_moe_forward_decode_pooled(
                ctx,
                layer,
                split,
                route_indices,
                route_weights,
                hidden,
                out,
                topk,
                local_start,
                swiglu_limit,
                scratch,
            );
        }

        // ── 3. Per-local-expert counts → group offsets (EP-aware start/range). ──
        let counts = ctx
            .stream
            .alloc_zeros::<i32>(experts_per_rank)
            .map_err(|e| anyhow::anyhow!("DSv4 count alloc failed: {e}"))?;
        let offsets = ctx
            .stream
            .alloc_zeros::<i32>(experts_per_rank)
            .map_err(|e| anyhow::anyhow!("DSv4 offset alloc failed: {e}"))?;
        let scan_total = ctx
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!("DSv4 scan-total alloc failed: {e}"))?;
        keepalive.keep_i32(&counts);
        keepalive.keep_i32(&offsets);
        keepalive.keep_i32(&scan_total);
        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_count_local_experts(
                cache_ptr(&route_indices, ctx),
                cache_ptr(&counts, ctx),
                num_tokens,
                topk,
                local_start,
                experts_per_rank,
                ctx.stream.cu_stream(),
            )?;
            moe::dsv4_exclusive_scan_i32(
                cache_ptr(&counts, ctx),
                cache_ptr(&offsets, ctx),
                cache_ptr(&scan_total, ctx),
                experts_per_rank,
                ctx.stream.cu_stream(),
            )?;
        }

        // ── 4. Pack routed tokens grouped-by-local-expert (compact rows). ───────
        let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, total_routes.max(1))?;
        keepalive.keep_hidden(&packed_hidden);
        // Initialize to -1 (the invalid sentinel), NOT 0. The scatter kernel
        // treats only route_slot < 0 as invalid; zero-init left unfilled compact
        // rows looking like valid slot-0 rows, which in m=1 decode overwrote
        // route slot 0 with zero output (DeepGEMM-path divergence, fixed H20).
        let packed_route_slot = ctx
            .stream
            .clone_htod(&vec![-1i32; total_routes.max(1)])
            .map_err(|e| anyhow::anyhow!("DSv4 packed_route_slot H2D failed: {e}"))?;
        let packed_weight = ctx
            .stream
            .alloc_zeros::<f32>(total_routes.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 packed_weight alloc failed: {e}"))?;
        let cursors = ctx
            .stream
            .alloc_zeros::<i32>(experts_per_rank)
            .map_err(|e| anyhow::anyhow!("DSv4 cursors alloc failed: {e}"))?;
        keepalive.keep_i32(&packed_route_slot);
        keepalive.keep_f32(&packed_weight);
        keepalive.keep_i32(&cursors);
        // SAFETY: buffers valid on ctx.stream; shapes checked by the kernel.
        unsafe {
            moe::dsv4_pack_local_experts_with_slots(
                cache_ptr(&hidden.data, ctx),
                cache_ptr(&route_indices, ctx),
                cache_ptr(&route_weights, ctx),
                cache_ptr(&offsets, ctx),
                cache_ptr(&cursors, ctx),
                cache_ptr(&packed_hidden.data, ctx),
                cache_ptr(&packed_route_slot, ctx),
                cache_ptr(&packed_weight, ctx),
                num_tokens,
                hidden_dim,
                topk,
                local_start,
                experts_per_rank,
                ctx.stream.cu_stream(),
            )?;
        }

        // ── 5. FP8 DeepGEMM 5-call grouped expert pipeline → compact rows. ──────
        let intermediate = layer.intermediate;
        ensure!(
            hidden_dim.is_multiple_of(128) && intermediate.is_multiple_of(128),
            "DSv4 DeepGEMM needs H and I aligned to 128, got H={hidden_dim} I={intermediate}"
        );
        ensure!(
            layer.w13_grouped.rows == intermediate * 2 && layer.w13_grouped.cols == hidden_dim,
            "DSv4 grouped w13 cache shape {}x{} != [2*I={}, H={}]",
            layer.w13_grouped.rows,
            layer.w13_grouped.cols,
            intermediate * 2,
            hidden_dim
        );
        ensure!(
            layer.w2_grouped.rows == hidden_dim && layer.w2_grouped.cols == intermediate,
            "DSv4 grouped w2 cache shape {}x{} != [H={}, I={}]",
            layer.w2_grouped.rows,
            layer.w2_grouped.cols,
            hidden_dim,
            intermediate
        );

        let expert_out = {
            let _nvtx = crate::nvtx::range("dsv4/deepgemm_grouped");
            deepgemm_grouped_experts(
                ctx,
                layer,
                &packed_hidden,
                &counts,
                &offsets,
                swiglu_limit,
                keepalive,
            )?
        };
        keepalive.keep_hidden(&expert_out);

        // ── 6. Scatter weighted expert outputs to route slots, combine topk. ────
        let nvtx_combine = crate::nvtx::range("dsv4/combine_scatter");
        let route_out = HiddenStates::zeros(ctx, hidden_dim, total_routes.max(1))?;
        keepalive.keep_hidden(&route_out);
        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_scatter_all_route_slots(
                cache_ptr(&expert_out.data, ctx),
                cache_ptr(&route_out.data, ctx),
                cache_ptr(&packed_route_slot, ctx),
                cache_ptr(&packed_weight, ctx),
                total_routes,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
            moe::dsv4_combine_route_slot_outputs(
                cache_ptr(&route_out.data, ctx),
                cache_ptr(&out.data, ctx),
                num_tokens,
                topk,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
        }
        drop(nvtx_combine);

        // The shared expert is replicated on every rank. Callers must all-reduce
        // the routed local expert contribution first, then add the shared expert
        // exactly once per rank.
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dsv4_moe_forward_decode_pooled(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        split: &ExpertSplit,
        route_indices: RawDevicePtr<i32>,
        route_weights: RawDevicePtr<f32>,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        topk: usize,
        local_start: usize,
        swiglu_limit: f32,
        scratch: &mut Dsv4MoeDecodeScratch,
    ) -> Result<()> {
        let num_tokens = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let total_routes = num_tokens * topk;
        ensure!(
            num_tokens == 1 && total_routes == scratch.topk,
            "DSv4 pooled decode MoE expected one token/topk routes, got tokens={num_tokens} routes={total_routes} scratch_topk={}",
            scratch.topk
        );
        let use_contiguous = use_contiguous_decode_moe();
        crate::stage_profile::profile(ctx, "dsv4/stage/moe_pack", || -> Result<()> {
            scratch.reset_routed(ctx)?;
            unsafe {
                moe::dsv4_count_local_experts(
                    route_indices,
                    cache_ptr(&scratch.counts, ctx),
                    num_tokens,
                    topk,
                    local_start,
                    split.experts_per_rank,
                    ctx.stream.cu_stream(),
                )?;
                moe::dsv4_exclusive_scan_i32(
                    cache_ptr(&scratch.counts, ctx),
                    cache_ptr(&scratch.offsets, ctx),
                    cache_ptr(&scratch.scan_total, ctx),
                    split.experts_per_rank,
                    ctx.stream.cu_stream(),
                )?;
                if use_contiguous {
                    moe::dsv4_pack_local_experts_with_slots_and_indices(
                        cache_ptr(&hidden.data, ctx),
                        route_indices,
                        route_weights,
                        cache_ptr(&scratch.offsets, ctx),
                        cache_ptr(&scratch.cursors, ctx),
                        cache_ptr(&scratch.grouped_contig.packed_hidden.data, ctx),
                        cache_ptr(&scratch.grouped_contig.packed_route_slot, ctx),
                        cache_ptr(&scratch.grouped_contig.packed_weight, ctx),
                        cache_ptr(&scratch.grouped_contig.m_indices, ctx),
                        num_tokens,
                        hidden_dim,
                        topk,
                        local_start,
                        split.experts_per_rank,
                        ctx.stream.cu_stream(),
                    )?;
                } else {
                    moe::dsv4_pack_local_experts_with_slots(
                        cache_ptr(&hidden.data, ctx),
                        route_indices,
                        route_weights,
                        cache_ptr(&scratch.offsets, ctx),
                        cache_ptr(&scratch.cursors, ctx),
                        cache_ptr(&scratch.packed_hidden.data, ctx),
                        cache_ptr(&scratch.packed_route_slot, ctx),
                        cache_ptr(&scratch.packed_weight, ctx),
                        num_tokens,
                        hidden_dim,
                        topk,
                        local_start,
                        split.experts_per_rank,
                        ctx.stream.cu_stream(),
                    )?;
                }
            }
            Ok(())
        })?;

        let expert_out = {
            let _nvtx = crate::nvtx::range("dsv4/deepgemm_grouped");
            crate::stage_profile::profile(ctx, "dsv4/stage/moe_deepgemm_grouped", || {
                if use_contiguous {
                    deepgemm_grouped_experts_contiguous_pooled(
                        ctx,
                        layer,
                        swiglu_limit,
                        &mut scratch.grouped_contig,
                    )
                } else {
                    deepgemm_grouped_experts_pooled(
                        ctx,
                        layer,
                        &scratch.packed_hidden,
                        &scratch.counts,
                        &scratch.offsets,
                        swiglu_limit,
                        &mut scratch.grouped,
                    )
                }
            })?
        };

        let nvtx_combine = crate::nvtx::range("dsv4/combine_scatter");
        crate::stage_profile::profile(ctx, "dsv4/stage/moe_combine_scatter", || -> Result<()> {
            unsafe {
                if use_contiguous {
                    moe::dsv4_scatter_all_route_slots(
                        cache_ptr(&expert_out.data, ctx),
                        cache_ptr(&scratch.route_out.data, ctx),
                        cache_ptr(&scratch.grouped_contig.packed_route_slot, ctx),
                        cache_ptr(&scratch.grouped_contig.packed_weight, ctx),
                        expert_out.seq_len,
                        hidden_dim,
                        ctx.stream.cu_stream(),
                    )?;
                } else {
                    moe::dsv4_scatter_all_route_slots(
                        cache_ptr(&expert_out.data, ctx),
                        cache_ptr(&scratch.route_out.data, ctx),
                        cache_ptr(&scratch.packed_route_slot, ctx),
                        cache_ptr(&scratch.packed_weight, ctx),
                        total_routes,
                        hidden_dim,
                        ctx.stream.cu_stream(),
                    )?;
                }
                moe::dsv4_combine_route_slot_outputs(
                    cache_ptr(&scratch.route_out.data, ctx),
                    cache_ptr(&out.data, ctx),
                    num_tokens,
                    topk,
                    hidden_dim,
                    ctx.stream.cu_stream(),
                )?;
            }
            Ok(())
        })?;
        drop(nvtx_combine);
        Ok(())
    }

    #[cfg(feature = "deepep")]
    pub(crate) fn dsv4_moe_forward_deepep(
        model: &Dsv4Model,
        transport: &crate::deepep::DeepEpTransport,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let split = &model.split;
        let swiglu_limit = model.config.swiglu_limit;

        let num_tokens = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let topk = cfg.top_k;
        let experts_per_rank = split.experts_per_rank;

        ensure!(
            tokens.len() == num_tokens,
            "DSv4 DeepEP token count {} != hidden seq_len {num_tokens}",
            tokens.len()
        );
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == num_tokens,
            "DSv4 DeepEP out shape {}x{} != hidden {}x{}",
            out.hidden_dim,
            out.seq_len,
            hidden_dim,
            num_tokens
        );
        ensure!(
            layer.num_groups == experts_per_rank,
            "DSv4 DeepEP expert group count {} != experts_per_rank {experts_per_rank}",
            layer.num_groups
        );
        ensure!(
            layer.hidden_dim == hidden_dim,
            "DSv4 DeepEP expert hidden dim {} != runtime hidden dim {hidden_dim}",
            layer.hidden_dim
        );
        // Local DSv4 MoE is DeepGEMM-only (no scalar/native expert fallback exists
        // in this tree), so the production path is always the DeepGEMM backend. The
        // preflight below is the real guard against a build-time stub.
        moe::dsv4_deepgemm_native_preflight()?;

        // Route exactly like the allreduce path. DeepEP dispatch consumes global
        // expert ids as i64 and remaps received ids to rank-local expert ids.
        let mut logits = HiddenStates::zeros(ctx, cfg.num_experts, num_tokens)?;
        gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
        let total_routes = num_tokens * topk;
        let (topk_idx_i64, route_weights) = if use_gpu_router() {
            let routing = dsv4_route_device(model, layer, tokens, &logits, None, keepalive)?;
            let topk_idx_i64 = ctx
                .stream
                .alloc_zeros::<i64>(total_routes)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP route-index i64 alloc failed: {e}"))?;
            keepalive.keep_route_i64(&topk_idx_i64);
            unsafe {
                moe::dsv4_cast_i32_to_i64(
                    cache_ptr(&routing.indices, ctx),
                    cache_ptr(&topk_idx_i64, ctx),
                    total_routes,
                    ctx.stream.cu_stream(),
                )?;
            }
            (topk_idx_i64, routing.weights)
        } else {
            keepalive.keep_hidden(&logits);
            ctx.sync()?;
            let logits_bf16: Vec<bf16> = ctx
                .stream
                .clone_dtoh(&logits.data)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP router logits D2H failed: {e}"))?;
            let logits_host: Vec<f32> = logits_bf16.iter().map(|&v| v.to_f32()).collect();
            let decisions = dsv4_route(ctx, &model.config, cfg, layer, tokens, &logits_host)?;
            let (indices_host, weights_host) = super::flatten_routing(&decisions, topk)?;
            let indices_i64: Vec<i64> = indices_host.iter().map(|&v| i64::from(v)).collect();
            let topk_idx_i64 = ctx
                .stream
                .clone_htod(&indices_i64)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP route-index i64 H2D failed: {e}"))?;
            let route_weights = ctx
                .stream
                .clone_htod(&weights_host)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP route-weight H2D failed: {e}"))?;
            keepalive.keep_i64(&topk_idx_i64);
            keepalive.keep_f32(&route_weights);
            (topk_idx_i64, route_weights)
        };

        let num_sms = crate::deepep::DeepEpTransport::num_sms()?;
        let mut scratch =
            transport.alloc_scratch(ctx, hidden_dim, num_tokens, topk, cfg.num_experts, num_sms)?;
        keepalive.keep_hidden(&scratch.recv_x);
        keepalive.keep_i32(&scratch.recv_src_idx);
        keepalive.keep_i64(&scratch.recv_topk_idx);
        keepalive.keep_f32(&scratch.recv_topk_weights);
        keepalive.keep_i32(&scratch.rank_prefix);
        keepalive.keep_i32(&scratch.recv_channel_prefix);
        keepalive.keep_i32(&scratch.send_head);
        keepalive.keep_i32(&scratch.num_tokens_per_rank);
        keepalive.keep_i32(&scratch.num_tokens_per_expert);
        keepalive.keep_u8(&scratch.is_token_in_rank);
        keepalive.keep_i32(&scratch.channel_prefix_matrix);
        keepalive.keep_f32(&scratch.combined_topk_weights);
        let num_recv = transport.dispatch(
            ctx,
            &mut scratch,
            hidden,
            &topk_idx_i64,
            &route_weights,
            cfg.num_experts,
            topk,
            num_sms,
        )?;

        let recv_slots = num_recv.saturating_mul(topk);
        let local_routed = HiddenStates::zeros(ctx, hidden_dim, scratch.capacity_recv)?;
        if recv_slots > 0 {
            // DeepEP gives rank-local expert ids as i64. Convert the valid prefix
            // to i32 for the existing local count/pack kernels.
            let recv_i64 = ctx
                .stream
                .clone_dtoh(&scratch.recv_topk_idx)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP recv_topk_idx D2H failed: {e}"))?;
            let mut recv_i32 = vec![0i32; scratch.capacity_recv.saturating_mul(topk)];
            for (dst, &src) in recv_i32.iter_mut().zip(recv_i64.iter()).take(recv_slots) {
                *dst = i32::try_from(src).map_err(|_| {
                    anyhow::anyhow!("DSv4 DeepEP recv expert id {src} overflows i32")
                })?;
            }
            let recv_topk_i32 = ctx
                .stream
                .clone_htod(&recv_i32)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP recv_topk_i32 H2D failed: {e}"))?;
            let counts = ctx
                .stream
                .alloc_zeros::<i32>(experts_per_rank)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP local count alloc failed: {e}"))?;
            let offsets = ctx
                .stream
                .alloc_zeros::<i32>(experts_per_rank)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP local offset alloc failed: {e}"))?;
            let scan_total = ctx
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP scan-total alloc failed: {e}"))?;
            keepalive.keep_i32(&recv_topk_i32);
            keepalive.keep_i32(&counts);
            keepalive.keep_i32(&offsets);
            keepalive.keep_i32(&scan_total);
            unsafe {
                moe::dsv4_count_local_experts(
                    cache_ptr(&recv_topk_i32, ctx),
                    cache_ptr(&counts, ctx),
                    num_recv,
                    topk,
                    0,
                    experts_per_rank,
                    ctx.stream.cu_stream(),
                )?;
                moe::dsv4_exclusive_scan_i32(
                    cache_ptr(&counts, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&scan_total, ctx),
                    experts_per_rank,
                    ctx.stream.cu_stream(),
                )?;
            }

            let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, recv_slots.max(1))?;
            let packed_route_slot = ctx
                .stream
                .clone_htod(&vec![-1i32; recv_slots.max(1)])
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP packed_route_slot H2D failed: {e}"))?;
            let packed_weight = ctx
                .stream
                .alloc_zeros::<f32>(recv_slots.max(1))
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP packed_weight alloc failed: {e}"))?;
            let cursors = ctx
                .stream
                .alloc_zeros::<i32>(experts_per_rank)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepEP cursors alloc failed: {e}"))?;
            keepalive.keep_hidden(&packed_hidden);
            keepalive.keep_i32(&packed_route_slot);
            keepalive.keep_f32(&packed_weight);
            keepalive.keep_i32(&cursors);
            unsafe {
                moe::dsv4_pack_local_experts_with_slots(
                    cache_ptr(&scratch.recv_x.data, ctx),
                    cache_ptr(&recv_topk_i32, ctx),
                    cache_ptr(&scratch.recv_topk_weights, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&cursors, ctx),
                    cache_ptr(&packed_hidden.data, ctx),
                    cache_ptr(&packed_route_slot, ctx),
                    cache_ptr(&packed_weight, ctx),
                    num_recv,
                    hidden_dim,
                    topk,
                    0,
                    experts_per_rank,
                    ctx.stream.cu_stream(),
                )?;
            }

            let expert_out = deepgemm_grouped_experts(
                ctx,
                layer,
                &packed_hidden,
                &counts,
                &offsets,
                swiglu_limit,
                keepalive,
            )?;
            keepalive.keep_hidden(&expert_out);
            let route_out = HiddenStates::zeros(ctx, hidden_dim, recv_slots.max(1))?;
            keepalive.keep_hidden(&route_out);
            unsafe {
                moe::dsv4_scatter_all_route_slots(
                    cache_ptr(&expert_out.data, ctx),
                    cache_ptr(&route_out.data, ctx),
                    cache_ptr(&packed_route_slot, ctx),
                    cache_ptr(&packed_weight, ctx),
                    recv_slots,
                    hidden_dim,
                    ctx.stream.cu_stream(),
                )?;
                moe::dsv4_combine_route_slot_outputs(
                    cache_ptr(&route_out.data, ctx),
                    cache_ptr(&local_routed.data, ctx),
                    num_recv,
                    topk,
                    hidden_dim,
                    ctx.stream.cu_stream(),
                )?;
            }
        }

        transport.combine(
            ctx,
            &mut scratch,
            &local_routed,
            out,
            num_recv,
            num_tokens,
            topk,
            num_sms,
        )?;
        let _ = total_routes;
        Ok(())
    }

    pub(crate) fn dsv4_shared_expert_forward(
        ctx: &DeviceContext,
        stream: &Arc<CudaStream>,
        layer: &Dsv4MoeLayer,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        swiglu_limit: f32,
        decode_scratch: Option<&mut Dsv4MoeDecodeScratch>,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        if let Some(scratch) = decode_scratch {
            ensure!(
                hidden.seq_len == 1
                    && scratch.hidden_dim == hidden.hidden_dim
                    && scratch.shared_intermediate == layer.shared_w2.cols,
                "DSv4 shared decode scratch mismatch: tokens={} scratch_H={} hidden_H={} scratch_I={} layer_I={}",
                hidden.seq_len,
                scratch.hidden_dim,
                hidden.hidden_dim,
                scratch.shared_intermediate,
                layer.shared_w2.cols
            );
            return dsv4_shared_expert_pooled(
                ctx,
                stream,
                layer,
                hidden,
                out,
                swiglu_limit,
                &mut scratch.shared,
            );
        }
        let shared = dsv4_shared_expert(ctx, stream, layer, hidden, swiglu_limit, keepalive)?;
        ensure!(
            shared.hidden_dim == out.hidden_dim && shared.seq_len >= out.seq_len,
            "DSv4 shared expert shape {}x{} != output {}x{}",
            shared.hidden_dim,
            shared.seq_len,
            out.hidden_dim,
            out.seq_len
        );
        let elems = out.hidden_dim * out.seq_len;
        let src = shared.data.slice(0..elems);
        stream
            .memcpy_dtod(&src, &mut out.data)
            .map_err(|e| anyhow::anyhow!("DSv4 shared expert output D2D failed: {e}"))?;
        keepalive.keep_hidden(&shared);
        Ok(())
    }

    /// Dispatch DSv4 host routing for one layer: bias-routed → learned router +
    /// `noaux_tc` correction bias via `infer_moe::route` (the validated path);
    /// hash-routed → `tid2eid` table via [`super::hash_route`]. Returns one
    /// [`RoutingDecision`] per token.
    pub(crate) fn dsv4_route(
        ctx: &DeviceContext,
        config: &deepseek_spec::DeepSeekV4Config,
        moe_config: &infer_moe::MoeConfig,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        logits_host: &[f32],
    ) -> Result<Vec<infer_moe::RoutingDecision>> {
        use deepseek_spec::DeepSeekV4MoeRoutingKind;
        match layer.routing_kind {
            DeepSeekV4MoeRoutingKind::Hash => {
                let tid2eid = layer.hash_tid2eid.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("DSv4 hash-routed MoE layer missing tid2eid table")
                })?;
                super::hash_route(config, tid2eid, tokens, logits_host)
            }
            DeepSeekV4MoeRoutingKind::LearnedBias => {
                let gate_bias = layer.gate_bias.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("DSv4 bias-routed MoE layer missing gate bias")
                })?;
                // bias D2H is cheap (one [n_routed] vec); thread the REAL bias.
                let bias_bf16: Vec<bf16> = ctx
                    .stream
                    .clone_dtoh(&gate_bias.data)
                    .map_err(|e| anyhow::anyhow!("DSv4 gate bias D2H failed: {e}"))?;
                let bias_host: Vec<f32> = bias_bf16.iter().map(|&v| v.to_f32()).collect();
                infer_moe::route(logits_host, &bias_host, moe_config)
                    .map_err(|e| anyhow::anyhow!("DSv4 host route failed: {e}"))
            }
        }
    }

    /// Run the FP8 DeepGEMM 5-call grouped expert pipeline over this rank's local
    /// experts; returns the compact `[total_routes, hidden]` expert output.
    fn deepgemm_grouped_experts(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        packed_hidden: &HiddenStates,
        counts: &cudarc::driver::CudaSlice<i32>,
        offsets: &cudarc::driver::CudaSlice<i32>,
        swiglu_limit: f32,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<HiddenStates> {
        let num_groups = layer.num_groups;
        let hidden_dim = packed_hidden.hidden_dim;
        let intermediate = layer.intermediate;
        let w13 = &layer.w13_grouped;
        let w2 = &layer.w2_grouped;
        ensure!(
            layer.hidden_dim == hidden_dim,
            "DSv4 grouped expert hidden dim {} != packed hidden dim {hidden_dim}",
            layer.hidden_dim
        );
        ensure!(
            w13.groups == num_groups
                && w2.groups == num_groups
                && w13.rows == 2 * intermediate
                && w13.cols == hidden_dim
                && w2.rows == hidden_dim
                && w2.cols == intermediate,
            "DSv4 grouped expert cache metadata mismatch: groups={} w13={}x{} g={} w2={}x{} g={} hidden={hidden_dim} inter={intermediate}",
            num_groups,
            w13.rows,
            w13.cols,
            w13.groups,
            w2.rows,
            w2.cols,
            w2.groups,
        );

        // Prefill can have tens of thousands of routes. The old masked layout
        // materialized `num_groups * max_m` rows and overflowed the unpad kernel's
        // work-size at ~1.5K prompt tokens (32 * T * topk * H > i32::MAX), long
        // before the 8K/32K SLO shapes. Use DeepGEMM's contiguous grouped layout
        // over the compact route rows instead: `m_indices[row]` names the local
        // expert for each activation row, so no padded per-expert slab or unpad is
        // needed.
        let rows = packed_hidden.seq_len.max(1);
        let scale_stride_m = rows.div_ceil(4) * 4;
        let hidden_scale_cols = hidden_dim.div_ceil(128);
        let inter_scale_cols = intermediate.div_ceil(128);

        let input_fp8 = alloc_u8(ctx, rows * hidden_dim)?;
        let input_scales = alloc_zeros_f32(ctx, scale_stride_m * hidden_scale_cols)?;
        let w13_out = HiddenStates::zeros(ctx, 2 * intermediate, rows)?;
        let act_fp8 = alloc_u8(ctx, rows * intermediate)?;
        let act_scales = alloc_zeros_f32(ctx, scale_stride_m * inter_scale_cols)?;
        let mut out_compact = HiddenStates::zeros(ctx, hidden_dim, packed_hidden.seq_len.max(1))?;
        let m_indices = ctx
            .stream
            .alloc_zeros::<i32>(rows)
            .map_err(|e| anyhow::anyhow!("DSv4 contiguous m-index alloc failed: {e}"))?;
        let active_experts = ctx.stream.clone_htod(&[0i32]).map_err(|e| {
            anyhow::anyhow!("DSv4 DeepGEMM contiguous active-expert H2D failed: {e}")
        })?;
        let active_offsets = ctx
            .stream
            .clone_htod(&[0i32])
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM contiguous offset H2D failed: {e}"))?;
        let active_counts = ctx
            .stream
            .clone_htod(&[i32::try_from(rows)?])
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM contiguous count H2D failed: {e}"))?;
        keepalive.keep_u8(&input_fp8);
        keepalive.keep_f32(&input_scales);
        keepalive.keep_hidden(&w13_out);
        keepalive.keep_u8(&act_fp8);
        keepalive.keep_f32(&act_scales);
        keepalive.keep_hidden(&out_compact);
        keepalive.keep_i32(&m_indices);
        keepalive.keep_i32(&active_experts);
        keepalive.keep_i32(&active_offsets);
        keepalive.keep_i32(&active_counts);

        let p_hidden = cache_ptr(&packed_hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&input_fp8, ctx);
        let p_in_scales = cache_ptr(&input_scales, ctx);
        let p_active = cache_ptr(&active_experts, ctx);
        let p_active_offsets = cache_ptr(&active_offsets, ctx);
        let p_active_counts = cache_ptr(&active_counts, ctx);
        let p_m_indices = cache_ptr(&m_indices, ctx);
        let p_w13_out = cache_ptr(&w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&act_fp8, ctx);
        let p_act_scales = cache_ptr(&act_scales, ctx);
        let p_out_compact = cache_ptr(&out_compact.data, ctx);
        let stream = ctx.stream.cu_stream();

        // SAFETY: every buffer/ptr above is valid on ctx.stream for the shapes
        // checked against the loaded caches; the kernels bound rows by masked_m.
        unsafe {
            moe::dsv4_fill_m_indices_from_counts(
                cache_ptr(counts, ctx),
                cache_ptr(offsets, ctx),
                p_m_indices,
                num_groups,
                rows,
                stream,
            )?;
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_active_offsets,
                p_active_counts,
                1,
                rows,
                hidden_dim,
                scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                p_in_fp8,
                p_in_scales,
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                p_w13_out,
                p_m_indices,
                num_groups,
                rows,
                2 * intermediate,
                hidden_dim,
                scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_swiglu_quantize_w13(
                p_w13_out,
                p_act_fp8,
                p_act_scales,
                p_active,
                p_active_counts,
                1,
                rows,
                intermediate,
                scale_stride_m,
                swiglu_limit,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                p_act_fp8,
                p_act_scales,
                cache_ptr(&w2.weight, ctx),
                cache_ptr(&w2.scales, ctx),
                p_out_compact,
                p_m_indices,
                num_groups,
                rows,
                hidden_dim,
                intermediate,
                scale_stride_m,
                stream,
            )?;
        }

        out_compact.seq_len = packed_hidden.seq_len;
        Ok(out_compact)
    }

    fn deepgemm_grouped_experts_pooled(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        packed_hidden: &HiddenStates,
        counts: &CudaSlice<i32>,
        offsets: &CudaSlice<i32>,
        swiglu_limit: f32,
        scratch: &mut Dsv4GroupedDecodeScratch,
    ) -> Result<HiddenStates> {
        let num_groups = layer.num_groups;
        let hidden_dim = packed_hidden.hidden_dim;
        let intermediate = layer.intermediate;
        let w13 = &layer.w13_grouped;
        let w2 = &layer.w2_grouped;
        ensure!(
            scratch.max_m >= packed_hidden.seq_len
                && scratch.out_compact.seq_len >= packed_hidden.seq_len
                && scratch.out_padded.hidden_dim == hidden_dim
                && scratch.w13_out.hidden_dim == 2 * intermediate,
            "DSv4 pooled grouped scratch mismatch: max_m={} out_cap={} packed={} H={} I={}",
            scratch.max_m,
            scratch.out_compact.seq_len,
            packed_hidden.seq_len,
            hidden_dim,
            intermediate
        );
        ensure!(
            w13.groups == num_groups
                && w2.groups == num_groups
                && w13.rows == 2 * intermediate
                && w13.cols == hidden_dim
                && w2.rows == hidden_dim
                && w2.cols == intermediate,
            "DSv4 grouped expert cache metadata mismatch: groups={} w13={}x{} g={} w2={}x{} g={} hidden={hidden_dim} inter={intermediate}",
            num_groups,
            w13.rows,
            w13.cols,
            w13.groups,
            w2.rows,
            w2.cols,
            w2.groups,
        );
        // `masked_m` is the per-group valid-row count = `counts`; alias it directly
        // (the masked GEMM only reads it, and `counts` is already passed read-only to
        // pack_quantize + swiglu below) instead of a per-layer D2D copy into
        // `scratch.masked_m` — kills one cuMemcpyDtoD/layer (the 17.8% D2D bucket).
        let p_hidden = cache_ptr(&packed_hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&scratch.input_fp8, ctx);
        let p_in_scales = cache_ptr(&scratch.input_scales, ctx);
        let p_active = cache_ptr(&scratch.active_experts, ctx);
        let p_offsets = cache_ptr(offsets, ctx);
        let p_counts = cache_ptr(counts, ctx);
        // Alias masked_m to counts (same data, read-only in the masked GEMM).
        let p_masked = p_counts;
        let p_w13_out = cache_ptr(&scratch.w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&scratch.act_fp8, ctx);
        let p_act_scales = cache_ptr(&scratch.act_scales, ctx);
        let p_out_padded = cache_ptr(&scratch.out_padded.data, ctx);
        let p_out_compact = cache_ptr(&scratch.out_compact.data, ctx);
        let stream = ctx.stream.cu_stream();

        unsafe {
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_offsets,
                p_counts,
                num_groups,
                scratch.max_m,
                hidden_dim,
                scratch.scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_in_fp8,
                p_in_scales,
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                p_w13_out,
                p_masked,
                num_groups,
                scratch.max_m,
                2 * intermediate,
                hidden_dim,
                scratch.scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_swiglu_quantize_w13(
                p_w13_out,
                p_act_fp8,
                p_act_scales,
                p_active,
                p_counts,
                num_groups,
                scratch.max_m,
                intermediate,
                scratch.scale_stride_m,
                swiglu_limit,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_act_fp8,
                p_act_scales,
                cache_ptr(&w2.weight, ctx),
                cache_ptr(&w2.scales, ctx),
                p_out_padded,
                p_masked,
                num_groups,
                scratch.max_m,
                hidden_dim,
                intermediate,
                scratch.scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_unpad_grouped_bf16(
                p_out_padded,
                p_out_compact,
                p_active,
                p_offsets,
                p_counts,
                num_groups,
                scratch.max_m,
                hidden_dim,
                stream,
            )?;
        }

        Ok(HiddenStates {
            data: scratch.out_compact.data.clone(),
            hidden_dim,
            seq_len: packed_hidden.seq_len,
        })
    }

    fn deepgemm_grouped_experts_contiguous_pooled(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        swiglu_limit: f32,
        scratch: &mut Dsv4GroupedContiguousDecodeScratch,
    ) -> Result<HiddenStates> {
        let num_groups = layer.num_groups;
        let packed_hidden = &scratch.packed_hidden;
        let hidden_dim = packed_hidden.hidden_dim;
        let intermediate = layer.intermediate;
        let w13 = &layer.w13_grouped;
        let w2 = &layer.w2_grouped;
        ensure!(
            scratch.rows >= packed_hidden.seq_len
                && scratch.out.hidden_dim == hidden_dim
                && scratch.w13_out.hidden_dim == 2 * intermediate,
            "DSv4 contiguous grouped scratch mismatch: rows={} packed={} H={} I={}",
            scratch.rows,
            packed_hidden.seq_len,
            hidden_dim,
            intermediate
        );
        ensure!(
            w13.groups == num_groups
                && w2.groups == num_groups
                && w13.rows == 2 * intermediate
                && w13.cols == hidden_dim
                && w2.rows == hidden_dim
                && w2.cols == intermediate,
            "DSv4 contiguous grouped cache metadata mismatch: groups={} w13={}x{} g={} w2={}x{} g={} hidden={hidden_dim} inter={intermediate}",
            num_groups,
            w13.rows,
            w13.cols,
            w13.groups,
            w2.rows,
            w2.cols,
            w2.groups,
        );

        let p_hidden = cache_ptr(&packed_hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&scratch.input_fp8, ctx);
        let p_in_scales = cache_ptr(&scratch.input_scales, ctx);
        let p_active = cache_ptr(&scratch.active_experts, ctx);
        let p_offsets = cache_ptr(&scratch.active_offsets, ctx);
        let p_counts = cache_ptr(&scratch.active_counts, ctx);
        let p_m_indices = cache_ptr(&scratch.m_indices, ctx);
        let p_w13_out = cache_ptr(&scratch.w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&scratch.act_fp8, ctx);
        let p_act_scales = cache_ptr(&scratch.act_scales, ctx);
        let p_out = cache_ptr(&scratch.out.data, ctx);
        let stream = ctx.stream.cu_stream();

        unsafe {
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_offsets,
                p_counts,
                1,
                scratch.rows,
                hidden_dim,
                scratch.scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                p_in_fp8,
                p_in_scales,
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                p_w13_out,
                p_m_indices,
                num_groups,
                scratch.rows,
                2 * intermediate,
                hidden_dim,
                scratch.scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_swiglu_quantize_w13(
                p_w13_out,
                p_act_fp8,
                p_act_scales,
                p_active,
                p_counts,
                1,
                scratch.rows,
                intermediate,
                scratch.scale_stride_m,
                swiglu_limit,
                stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                p_act_fp8,
                p_act_scales,
                cache_ptr(&w2.weight, ctx),
                cache_ptr(&w2.scales, ctx),
                p_out,
                p_m_indices,
                num_groups,
                scratch.rows,
                hidden_dim,
                intermediate,
                scratch.scale_stride_m,
                stream,
            )?;
        }

        Ok(HiddenStates {
            data: scratch.out.data.clone(),
            hidden_dim,
            seq_len: packed_hidden.seq_len,
        })
    }

    /// DSv4 dense shared expert via a single-group FP8 DeepGEMM pass: w13 fused
    /// gate+up → clamped SwiGLU → w2 down, over every token. No routing/scatter.
    fn dsv4_shared_expert_pooled(
        ctx: &DeviceContext,
        stream: &Arc<CudaStream>,
        layer: &Dsv4MoeLayer,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        swiglu_limit: f32,
        scratch: &mut Dsv4SharedDecodeScratch,
    ) -> Result<()> {
        let hidden_dim = hidden.hidden_dim;
        let num_tokens = hidden.seq_len;
        let shared_inter = layer.shared_w2.cols;
        ensure!(
            num_tokens == 1
                && scratch.out.hidden_dim == hidden_dim
                && scratch.w13_out.hidden_dim == 2 * shared_inter,
            "DSv4 pooled shared scratch mismatch: tokens={num_tokens} H={hidden_dim} I={shared_inter}"
        );
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == num_tokens,
            "DSv4 pooled shared out shape {}x{} != hidden {}x{}",
            out.hidden_dim,
            out.seq_len,
            hidden_dim,
            num_tokens
        );
        let p_hidden = cache_ptr(&hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&scratch.input_fp8, ctx);
        let p_in_scales = cache_ptr(&scratch.input_scales, ctx);
        let p_active = cache_ptr(&scratch.active_experts, ctx);
        let p_offsets = cache_ptr(&scratch.active_offsets, ctx);
        let p_counts = cache_ptr(&scratch.counts, ctx);
        let p_masked = cache_ptr(&scratch.masked_m, ctx);
        let p_w13_out = cache_ptr(&scratch.w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&scratch.act_fp8, ctx);
        let p_act_scales = cache_ptr(&scratch.act_scales, ctx);
        let p_out = cache_ptr(&scratch.out.data, ctx);
        let cu_stream = stream.cu_stream();

        unsafe {
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_offsets,
                p_counts,
                1,
                scratch.max_m,
                hidden_dim,
                scratch.scale_stride_m,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_in_fp8,
                p_in_scales,
                cache_ptr(&layer.shared_w13.weight, ctx),
                cache_ptr(&layer.shared_w13.scales, ctx),
                p_w13_out,
                p_masked,
                1,
                scratch.max_m,
                2 * shared_inter,
                hidden_dim,
                scratch.scale_stride_m,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_swiglu_quantize_w13(
                p_w13_out,
                p_act_fp8,
                p_act_scales,
                p_active,
                p_counts,
                1,
                scratch.max_m,
                shared_inter,
                scratch.scale_stride_m,
                swiglu_limit,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_act_fp8,
                p_act_scales,
                cache_ptr(&layer.shared_w2.weight, ctx),
                cache_ptr(&layer.shared_w2.scales, ctx),
                p_out,
                p_masked,
                1,
                scratch.max_m,
                hidden_dim,
                shared_inter,
                scratch.scale_stride_m,
                cu_stream,
            )?;
        }

        let elems = hidden_dim * num_tokens;
        let src = scratch.out.data.slice(0..elems);
        stream
            .memcpy_dtod(&src, &mut out.data)
            .map_err(|e| anyhow::anyhow!("DSv4 pooled shared output D2D failed: {e}"))?;
        Ok(())
    }

    /// DSv4 dense shared expert via a single-group FP8 DeepGEMM pass: w13 fused
    /// gate+up → clamped SwiGLU → w2 down, over every token. No routing/scatter.
    fn dsv4_shared_expert(
        ctx: &DeviceContext,
        stream: &Arc<CudaStream>,
        layer: &Dsv4MoeLayer,
        hidden: &HiddenStates,
        swiglu_limit: f32,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<HiddenStates> {
        let hidden_dim = hidden.hidden_dim;
        let num_tokens = hidden.seq_len;
        let shared_inter = layer.shared_w2.cols;
        ensure!(
            layer.shared_w13.rows == shared_inter * 2 && layer.shared_w13.cols == hidden_dim,
            "DSv4 shared w13 shape {}x{} != [2*I={}, H={}]",
            layer.shared_w13.rows,
            layer.shared_w13.cols,
            shared_inter * 2,
            hidden_dim
        );
        ensure!(
            layer.shared_w2.rows == hidden_dim,
            "DSv4 shared w2 rows {} != hidden {hidden_dim}",
            layer.shared_w2.rows
        );
        ensure!(
            hidden_dim.is_multiple_of(128) && shared_inter.is_multiple_of(128),
            "DSv4 shared expert needs H and I aligned to 128, got H={hidden_dim} I={shared_inter}"
        );

        // Floor at 128 for the SAME reason as the routed grouped path: the shared
        // expert runs the identical `dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked`
        // kernel, whose small-m (m=1 decode) tile path diverges below 128. The
        // routed-only floor left this reachable (codex review P2); a prompt whose
        // next-token margin leans on the shared expert could flip at m<128.
        let max_m = num_tokens.max(128);
        let scale_stride_m = max_m.div_ceil(4) * 4;
        let hidden_scale_cols = hidden_dim.div_ceil(128);
        let inter_scale_cols = shared_inter.div_ceil(128);
        let use_ctx_stream = Arc::ptr_eq(stream, &ctx.stream);

        let input_fp8 = if use_ctx_stream {
            alloc_u8(ctx, max_m * hidden_dim)?
        } else {
            alloc_u8_on(stream, max_m * hidden_dim)?
        };
        let input_scales = if use_ctx_stream {
            alloc_zeros_f32(ctx, scale_stride_m * hidden_scale_cols)?
        } else {
            alloc_zeros_f32_on(stream, scale_stride_m * hidden_scale_cols)?
        };
        let w13_out = if use_ctx_stream {
            HiddenStates::zeros(ctx, 2 * shared_inter, max_m)?
        } else {
            hidden_zeros_on(stream, 2 * shared_inter, max_m)?
        };
        let act_fp8 = if use_ctx_stream {
            alloc_u8(ctx, max_m * shared_inter)?
        } else {
            alloc_u8_on(stream, max_m * shared_inter)?
        };
        let act_scales = if use_ctx_stream {
            alloc_zeros_f32(ctx, scale_stride_m * inter_scale_cols)?
        } else {
            alloc_zeros_f32_on(stream, scale_stride_m * inter_scale_cols)?
        };
        // DeepGEMM's TMA D descriptor is built with `m = max_m`, so the output
        // allocation must cover the padded row capacity even though downstream
        // consumers only read the first `num_tokens` rows.
        let out = if use_ctx_stream {
            HiddenStates::zeros(ctx, hidden_dim, max_m)?
        } else {
            hidden_zeros_on(stream, hidden_dim, max_m)?
        };
        keepalive.keep_u8(&input_fp8);
        keepalive.keep_f32(&input_scales);
        keepalive.keep_hidden(&w13_out);
        keepalive.keep_u8(&act_fp8);
        keepalive.keep_f32(&act_scales);
        keepalive.keep_hidden(&out);

        // Single group spanning all tokens: identity expert 0, offset 0, count T.
        let active_experts = if use_ctx_stream {
            ctx.stream.clone_htod(&[0i32])
        } else {
            stream.clone_htod(&[0i32])
        }
        .map_err(|e| anyhow::anyhow!("DSv4 shared active H2D failed: {e}"))?;
        let active_offsets = if use_ctx_stream {
            ctx.stream.clone_htod(&[0i32])
        } else {
            stream.clone_htod(&[0i32])
        }
        .map_err(|e| anyhow::anyhow!("DSv4 shared offset H2D failed: {e}"))?;
        let counts = if use_ctx_stream {
            ctx.stream.clone_htod(&[num_tokens as i32])
        } else {
            stream.clone_htod(&[num_tokens as i32])
        }
        .map_err(|e| anyhow::anyhow!("DSv4 shared count H2D failed: {e}"))?;
        let masked_m = if use_ctx_stream {
            ctx.stream.clone_htod(&[num_tokens as i32])
        } else {
            stream.clone_htod(&[num_tokens as i32])
        }
        .map_err(|e| anyhow::anyhow!("DSv4 shared masked_m H2D failed: {e}"))?;
        keepalive.keep_i32(&active_experts);
        keepalive.keep_i32(&active_offsets);
        keepalive.keep_i32(&counts);
        keepalive.keep_i32(&masked_m);

        let p_hidden = cache_ptr(&hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&input_fp8, ctx);
        let p_in_scales = cache_ptr(&input_scales, ctx);
        let p_active = cache_ptr(&active_experts, ctx);
        let p_offsets = cache_ptr(&active_offsets, ctx);
        let p_counts = cache_ptr(&counts, ctx);
        let p_masked = cache_ptr(&masked_m, ctx);
        let p_w13_out = cache_ptr(&w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&act_fp8, ctx);
        let p_act_scales = cache_ptr(&act_scales, ctx);
        let p_out = cache_ptr(&out.data, ctx);
        let cu_stream = if use_ctx_stream {
            ctx.stream.cu_stream()
        } else {
            stream.cu_stream()
        };

        // SAFETY: single-group buffers valid on the selected stream; masked_m bounds rows.
        unsafe {
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_offsets,
                p_counts,
                1,
                max_m,
                hidden_dim,
                scale_stride_m,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_in_fp8,
                p_in_scales,
                cache_ptr(&layer.shared_w13.weight, ctx),
                cache_ptr(&layer.shared_w13.scales, ctx),
                p_w13_out,
                p_masked,
                1,
                max_m,
                2 * shared_inter,
                hidden_dim,
                scale_stride_m,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_swiglu_quantize_w13(
                p_w13_out,
                p_act_fp8,
                p_act_scales,
                p_active,
                p_counts,
                1,
                max_m,
                shared_inter,
                scale_stride_m,
                swiglu_limit,
                cu_stream,
            )?;
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_act_fp8,
                p_act_scales,
                cache_ptr(&layer.shared_w2.weight, ctx),
                cache_ptr(&layer.shared_w2.scales, ctx),
                p_out,
                p_masked,
                1,
                max_m,
                hidden_dim,
                shared_inter,
                scale_stride_m,
                cu_stream,
            )?;
        }

        Ok(HiddenStates {
            data: out.data.clone(),
            hidden_dim,
            seq_len: num_tokens,
        })
    }

    /// One contiguous group-major FP8 weight + scale buffer the masked GEMM
    /// strides per group (`b + g * n * k`, `sfb + g * scale_rows * scale_cols`).
    pub(crate) struct GroupedCache {
        pub(crate) weight: cudarc::driver::CudaSlice<u8>,
        pub(crate) scales: cudarc::driver::CudaSlice<f32>,
        pub(crate) groups: usize,
        pub(crate) rows: usize,
        pub(crate) cols: usize,
    }

    /// Concatenate the per-expert FP8 caches into one contiguous group-major
    /// buffer (D2D), validating uniform `[rows, cols]` shape + 128-row alignment.
    ///
    /// The loader stores this grouped cache in [`crate::dsv4::Dsv4MoeLayer`] and
    /// drops the per-expert Vecs. The weights are static after load; rebuilding
    /// this concat during decode would copy hundreds of MiB per layer per token.
    pub(crate) fn build_grouped_cache(
        ctx: &DeviceContext,
        caches: &[Dsv4Fp8DeepGemmWeightCache],
        rows: usize,
        cols: usize,
    ) -> Result<GroupedCache> {
        let first = caches
            .first()
            .ok_or_else(|| anyhow::anyhow!("DSv4 DeepGEMM grouped cache has no local experts"))?;
        ensure!(
            first.rows == rows && first.cols == cols,
            "DSv4 grouped cache shape {}x{} != [{rows}, {cols}]",
            first.rows,
            first.cols
        );
        ensure!(
            rows.is_multiple_of(128),
            "DSv4 grouped cache rows {rows} must be 128-aligned for group-major scale concat"
        );
        let num_groups = caches.len();
        let weight_stride = first.weight.len();
        let scale_stride = first.scales.len();
        let mut weight = ctx
            .stream
            .alloc_zeros::<u8>(num_groups * weight_stride)
            .map_err(|e| anyhow::anyhow!("DSv4 grouped weight alloc failed: {e}"))?;
        let mut scales = ctx
            .stream
            .alloc_zeros::<f32>(num_groups * scale_stride)
            .map_err(|e| anyhow::anyhow!("DSv4 grouped scale alloc failed: {e}"))?;
        for (g, cache) in caches.iter().enumerate() {
            ensure!(
                cache.rows == rows
                    && cache.cols == cols
                    && cache.weight.len() == weight_stride
                    && cache.scales.len() == scale_stride,
                "DSv4 grouped cache group {g} non-uniform: {}x{}",
                cache.rows,
                cache.cols
            );
            {
                let mut dst = weight.slice_mut(g * weight_stride..(g + 1) * weight_stride);
                ctx.stream
                    .memcpy_dtod(&cache.weight, &mut dst)
                    .map_err(|e| anyhow::anyhow!("DSv4 grouped weight D2D failed: {e}"))?;
            }
            let mut dst = scales.slice_mut(g * scale_stride..(g + 1) * scale_stride);
            ctx.stream
                .memcpy_dtod(&cache.scales, &mut dst)
                .map_err(|e| anyhow::anyhow!("DSv4 grouped scale D2D failed: {e}"))?;
        }
        Ok(GroupedCache {
            weight,
            scales,
            groups: num_groups,
            rows,
            cols,
        })
    }

    fn alloc_u8(ctx: &DeviceContext, len: usize) -> Result<cudarc::driver::CudaSlice<u8>> {
        ctx.stream
            .alloc_zeros::<u8>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM u8 scratch alloc failed: {e}"))
    }

    fn alloc_u8_on(stream: &Arc<CudaStream>, len: usize) -> Result<cudarc::driver::CudaSlice<u8>> {
        stream
            .alloc_zeros::<u8>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM u8 scratch alloc failed: {e}"))
    }

    fn alloc_zeros_f32(ctx: &DeviceContext, len: usize) -> Result<cudarc::driver::CudaSlice<f32>> {
        ctx.stream
            .alloc_zeros::<f32>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM f32 scratch alloc failed: {e}"))
    }

    fn alloc_zeros_f32_on(
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        stream
            .alloc_zeros::<f32>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM f32 scratch alloc failed: {e}"))
    }

    fn hidden_zeros_on(
        stream: &Arc<CudaStream>,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let len = hidden_dim * seq_len;
        let data = stream
            .alloc_zeros::<bf16>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 hidden scratch alloc failed: {e}"))?;
        Ok(HiddenStates {
            data,
            hidden_dim,
            seq_len,
        })
    }

    /// NVSHMEM low-latency (token-owned) DSv4 MoE forward for THIS rank's owned
    /// token slice. Implements the EP pipeline over the owned `[hidden, owned_n]`
    /// activations: route → LL dispatch (FP8 pack) → masked grouped GEMM w13 →
    /// masked SwiGLU+requant → masked grouped GEMM w2 → LL combine → add shared
    /// expert. The caller (dsv4.rs) owns the token slicing + the final all-gather.
    ///
    /// `out` is this rank's owned routed+shared output `[hidden, owned_n]`. The
    /// LL packed recv / GEMM-output scratch is pre-allocated once in `scratch`;
    /// this path only overwrites it (no per-step alloc beyond the small route +
    /// topk-id buffers + the routed/shared temporaries).
    #[cfg(feature = "deepep")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dsv4_moe_forward_deepep_ll(
        model: &Dsv4Model,
        transport: &crate::deepep::DeepEpTransport,
        scratch: &mut crate::deepep::DeepEpLlScratch,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let swiglu_limit = model.config.swiglu_limit;

        let owned_n = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let topk = cfg.top_k;
        let intermediate = layer.intermediate;
        let num_local_experts = transport.ll_num_local_experts()?;

        ensure!(
            tokens.len() == owned_n,
            "deepep_ll token count {} != owned hidden seq_len {owned_n}",
            tokens.len()
        );
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == owned_n,
            "deepep_ll out shape {}x{} != owned {hidden_dim}x{owned_n}",
            out.hidden_dim,
            out.seq_len,
        );
        ensure!(
            layer.num_groups == num_local_experts,
            "deepep_ll expert group count {} != num_local_experts {num_local_experts}",
            layer.num_groups
        );
        ensure!(
            layer.hidden_dim == hidden_dim,
            "deepep_ll expert hidden dim {} != runtime hidden dim {hidden_dim}",
            layer.hidden_dim
        );
        moe::dsv4_deepgemm_native_preflight()?;

        // ── Step 2: route the OWNED tokens on device → topk_idx (i64, global
        // expert ids, kept on device — no host i32→i64 glue) + topk weights.
        // owned_n may be 0 (seq_len < world): this rank still participates in the
        // dispatch/combine COLLECTIVE with num_tokens=0 (sends nothing, receives
        // tokens routed to its local experts), so allocate 1-slot dummies for the
        // route buffers and skip the actual route compute.
        let total_routes = owned_n * topk;
        let topk_idx_i64 = ctx
            .stream
            .alloc_zeros::<i64>(total_routes.max(1))
            .map_err(|e| anyhow::anyhow!("deepep_ll route-index i64 alloc failed: {e}"))?;
        keepalive.keep_route_i64(&topk_idx_i64);
        let route_weights = if owned_n > 0 {
            let mut logits = HiddenStates::zeros(ctx, cfg.num_experts, owned_n)?;
            gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
            keepalive.keep_hidden(&logits);
            let routing = dsv4_route_device(model, layer, tokens, &logits, None, keepalive)?;
            keepalive.keep_route_f32(&routing.weights);
            // SAFETY: both buffers hold `total_routes` elements on `ctx.stream`.
            unsafe {
                moe::dsv4_cast_i32_to_i64(
                    cache_ptr(&routing.indices, ctx),
                    cache_ptr(&topk_idx_i64, ctx),
                    total_routes,
                    ctx.stream.cu_stream(),
                )?;
            }
            routing.weights
        } else {
            let w = ctx
                .stream
                .alloc_zeros::<f32>(1)
                .map_err(|e| anyhow::anyhow!("deepep_ll empty route-weight alloc failed: {e}"))?;
            keepalive.keep_route_f32(&w);
            w
        };

        // ── Step 3: LL dispatch the owned tokens → packed FP8 recv into scratch.
        // `scratch` is model-owned (outlives the forward), so it does not need
        // the forward-keepalive guard the transient buffers below get.
        let _expected_m = transport.ll_dispatch(ctx, scratch, hidden, &topk_idx_i64, topk)?;

        let m = scratch.m_padded;
        let sfa_aligned_m = scratch.sfa_aligned_m;
        let w13 = &layer.w13_grouped;
        let w2 = &layer.w2_grouped;
        ensure!(
            w13.groups == num_local_experts
                && w2.groups == num_local_experts
                && w13.rows == 2 * intermediate
                && w13.cols == hidden_dim
                && w2.rows == hidden_dim
                && w2.cols == intermediate,
            "deepep_ll grouped weight metadata mismatch: groups={} w13={}x{}g{} w2={}x{}g{} H={hidden_dim} I={intermediate}",
            num_local_experts,
            w13.rows,
            w13.cols,
            w13.groups,
            w2.rows,
            w2.cols,
            w2.groups,
        );

        let p_recv_x = cache_ptr(&scratch.recv_x_fp8, ctx);
        let p_recv_sc = cache_ptr(&scratch.recv_x_scales, ctx);
        let p_masked = cache_ptr(&scratch.recv_count, ctx);
        let p_w13_out = cache_ptr(&scratch.w13_out, ctx);
        let p_act_fp8 = cache_ptr(&scratch.act_fp8, ctx);
        let p_act_sc = cache_ptr(&scratch.act_scales, ctx);
        let p_expert_out = cache_ptr(&scratch.expert_out, ctx);
        let stream = ctx.stream.cu_stream();

        // ── Step 4: masked grouped GEMM w13 (fused gate+up): recv FP8 → bf16
        //    [E_local, m, 2*intermediate]. masked_m = recv_count (per expert).
        // SAFETY: all buffers are scratch sized for `[E_local, m, *]`; masked_m
        // bounds rows; sfa_aligned_m == m (TMA-aligned, asserted at scratch alloc).
        unsafe {
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_recv_x,
                p_recv_sc,
                cache_ptr(&w13.weight, ctx),
                cache_ptr(&w13.scales, ctx),
                p_w13_out,
                p_masked,
                num_local_experts,
                m,
                2 * intermediate,
                hidden_dim,
                sfa_aligned_m,
                stream,
            )?;
            // ── Step 5: masked SwiGLU(clamp) + per-128-block FP8 requant →
            //    [E_local, m, intermediate] FP8 + column-major scales.
            moe::dsv4_deepgemm_silu_mul_masked_quant(
                p_w13_out,
                p_act_fp8,
                p_act_sc,
                p_masked,
                num_local_experts,
                m,
                2 * intermediate,
                swiglu_limit,
                stream,
            )?;
            // ── Step 6: masked grouped GEMM w2 (down): act FP8 → bf16
            //    [E_local, m, hidden]. This IS the LL-combine input layout.
            moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                p_act_fp8,
                p_act_sc,
                cache_ptr(&w2.weight, ctx),
                cache_ptr(&w2.scales, ctx),
                p_expert_out,
                p_masked,
                num_local_experts,
                m,
                hidden_dim,
                intermediate,
                sfa_aligned_m,
                stream,
            )?;
        }

        // ── Step 7: LL combine → this rank's owned routed output [hidden, owned_n].
        // The shared expert is NOT added here: the dsv4.rs forward adds the
        // replicated shared expert on the FULL gathered `moe_out` afterward
        // (identical to the intranode `dsv4_moe_forward_deepep` contract), so
        // adding it here would double-count. `out` holds routed-only.
        transport.ll_combine(
            ctx,
            scratch,
            out,
            &topk_idx_i64,
            &route_weights,
            owned_n,
            topk,
        )?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
#[allow(unused_imports)] // consumed by the Piece 2 model.rs DSv4 branch
pub(crate) use dsv4_gpu::{
    Dsv4MoeDecodeScratch, GroupedCache, build_grouped_cache, dsv4_moe_forward,
    dsv4_moe_forward_decode_graph, dsv4_shared_expert_forward,
};
#[cfg(feature = "deepep")]
pub(crate) use dsv4_gpu::{dsv4_moe_forward_deepep, dsv4_moe_forward_deepep_ll};
#[cfg(feature = "cuda")]
pub(crate) use gpu::moe_forward;

#[cfg(test)]
mod tests {
    use super::*;
    use infer_moe::{MoeConfig, route};

    fn cfg(num_experts: usize, top_k: usize, norm: bool) -> MoeConfig {
        MoeConfig::qwen36(num_experts, top_k, norm, 8)
    }

    #[test]
    fn flatten_matches_route_token_major() {
        // 2 tokens, 4 experts, top-2: check the flat buffers mirror the
        // route → (token*topk + k) layout the kernels read.
        let c = cfg(4, 2, true);
        // token 0: experts 3,2 ; token 1: experts 0,1 (descending logit).
        let logits = vec![0.1f32, 0.2, 3.0, 4.0, 4.0, 3.0, 0.2, 0.1];
        let decisions = route(&logits, &[], &c).unwrap();
        let (idx, w) = flatten_routing(&decisions, c.top_k).unwrap();
        assert_eq!(idx.len(), 4);
        assert_eq!(w.len(), 4);
        // token 0 routes at flat slots [0,1]; token 1 at [2,3].
        assert_eq!(idx[0], decisions[0].experts[0].expert as i32);
        assert_eq!(idx[1], decisions[0].experts[1].expert as i32);
        assert_eq!(idx[2], decisions[1].experts[0].expert as i32);
        assert_eq!(idx[3], decisions[1].experts[1].expert as i32);
        assert_eq!(idx[0], 3);
        assert_eq!(idx[1], 2);
        assert_eq!(idx[2], 0);
        assert_eq!(idx[3], 1);
        // Weights mirror selection order and renormalize to sum 1 per token.
        let t0: f32 = w[0] + w[1];
        let t1: f32 = w[2] + w[3];
        assert!((t0 - 1.0).abs() < 1e-5, "token0 weights sum {t0}");
        assert!((t1 - 1.0).abs() < 1e-5, "token1 weights sum {t1}");
    }

    #[test]
    fn flatten_rejects_wrong_topk() {
        let c = cfg(4, 2, false);
        let logits = vec![0.1f32, 0.2, 3.0, 4.0];
        let decisions = route(&logits, &[], &c).unwrap();
        // Ask for the wrong topk → mismatch error.
        assert!(flatten_routing(&decisions, 3).is_err());
    }

    // ── DSv4 noaux_tc gate-bias plumbing (the Piece 3 router thread). ────────
    // `dsv4_moe_forward` reads the real per-expert correction bias and passes it
    // to `infer_moe::route` (not `&[]`). These CPU tests lock that the bias
    // changes selection while the weight stays the unbiased sqrtsoftplus score,
    // and that `flatten_routing` mirrors the selection token-major.

    fn dsv4_cfg(num_experts: usize, top_k: usize, scaling: f32) -> MoeConfig {
        MoeConfig::dsv4(num_experts, 1, top_k, scaling, 8)
    }

    #[test]
    fn dsv4_gate_bias_flips_selection_not_weight() {
        // 4 experts, top-1. Raw sqrtsoftplus scores rank expert 3 first; a big
        // positive bias on expert 0 must steer noaux_tc selection to expert 0,
        // but the emitted weight is expert 0's UNBIASED score × scaling.
        let c = dsv4_cfg(4, 1, 1.0);
        let logits = vec![0.1f32, 0.2, 0.3, 0.9];
        let no_bias = route(&logits, &[], &c).unwrap();
        assert_eq!(no_bias[0].experts[0].expert, 3, "raw top-1 is expert 3");

        let bias = vec![5.0f32, 0.0, 0.0, 0.0];
        let biased = route(&logits, &bias, &c).unwrap();
        assert_eq!(
            biased[0].experts[0].expert, 0,
            "bias steers noaux_tc selection to expert 0"
        );
        // Weight is the unbiased score of expert 0 (sum-normalized over the single
        // selected expert ⇒ ~1.0 × scaling), not the biased key.
        let (idx, w) = flatten_routing(&biased, c.top_k).unwrap();
        assert_eq!(idx, vec![0]);
        assert!(
            (w[0] - 1.0).abs() < 1e-4,
            "single-select weight ≈ 1.0, got {}",
            w[0]
        );
    }

    #[test]
    fn dsv4_routed_scaling_factor_threads_through() {
        // routed_scaling_factor multiplies the normalized weight; top-1 over a
        // single expert ⇒ weight ≈ scaling.
        let c = dsv4_cfg(4, 1, 2.5);
        let logits = vec![0.1f32, 0.2, 0.3, 0.9];
        let decisions = route(&logits, &[], &c).unwrap();
        let (_idx, w) = flatten_routing(&decisions, c.top_k).unwrap();
        assert!(
            (w[0] - 2.5).abs() < 1e-3,
            "weight ≈ routed_scaling 2.5, got {}",
            w[0]
        );
    }

    // ── DSv4 hash routing (`super::hash_route`). Hash layers pick experts from
    // the tid2eid table by token id (ignoring the router gate selection), then
    // weight them by the sqrtsoftplus router scores. cuda-gated: `deepseek_spec`
    // is a `cuda`-feature dep.
    #[cfg(feature = "cuda")]
    fn hash_test_config() -> deepseek_spec::DeepSeekV4Config {
        deepseek_spec::DeepSeekV4Config::from_json_str(
            r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4", "torch_dtype": "bfloat16",
            "vocab_size": 8, "hidden_size": 256, "num_hidden_layers": 3,
            "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 512,
            "hidden_act": "silu", "swiglu_limit": 10.0, "q_lora_rank": 256,
            "o_lora_rank": 256, "o_groups": 8, "qk_rope_head_dim": 64,
            "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "moe_intermediate_size": 256, "routed_scaling_factor": 1.0,
            "norm_topk_prob": true, "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc", "index_n_heads": 8, "index_head_dim": 64,
            "index_topk": 64, "num_hash_layers": 3, "sliding_window": 64,
            "compress_ratios": [0, 4, 0], "compress_rope_theta": 160000.0,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1.0e-6,
            "num_nextn_predict_layers": 0, "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rope_scaling": {"type": "yarn", "factor": 16.0,
                "original_max_position_embeddings": 2048, "beta_fast": 32.0, "beta_slow": 1.0},
            "rms_norm_eps": 1.0e-6, "initializer_range": 0.02,
            "tie_word_embeddings": false, "attention_bias": false,
            "attention_dropout": 0.0, "bos_token_id": 0, "eos_token_id": 1
        }"#,
        )
        .unwrap()
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn hash_route_picks_tid2eid_experts_ignoring_logits() {
        let cfg = hash_test_config();
        // tid2eid: token 0 → experts [1,2]; token 1 → experts [3,0]. topk=2.
        let tid2eid: Vec<i64> = vec![1, 2, /*tok0*/ 3, 0 /*tok1*/];
        let tokens = vec![0u32, 1u32];
        // Logits rank expert 3 highest for both tokens; hash must IGNORE that
        // for selection (experts come from tid2eid), using logits only to weight.
        let logits = vec![
            0.1f32, 0.2, 0.3, 0.9, // token 0
            0.1, 0.2, 0.3, 0.9, // token 1
        ];
        let decisions = super::hash_route(&cfg, &tid2eid, &tokens, &logits).unwrap();
        assert_eq!(decisions.len(), 2);
        assert_eq!(
            decisions[0].expert_ids(),
            vec![1, 2],
            "token 0 from tid2eid"
        );
        assert_eq!(
            decisions[1].expert_ids(),
            vec![3, 0],
            "token 1 from tid2eid"
        );
        // Weights are positive normalized scores (each token's two weights sum ~1).
        let (idx, w) = flatten_routing(&decisions, cfg.num_experts_per_tok).unwrap();
        assert_eq!(idx, vec![1, 2, 3, 0]);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-3, "token 0 weights sum ~1");
        assert!((w[2] + w[3] - 1.0).abs() < 1e-3, "token 1 weights sum ~1");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn hash_route_rejects_token_beyond_table() {
        let cfg = hash_test_config();
        // Only token 0's two experts are present; token 1 needs entries [2,3].
        let tid2eid: Vec<i64> = vec![0, 1];
        assert!(super::hash_route(&cfg, &tid2eid, &[0u32, 1u32], &[0.0; 8]).is_err());
    }
}
