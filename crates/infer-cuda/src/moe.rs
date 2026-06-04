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
    use cuda_kernels::tensor::{Dsv4Fp8DeepGemmWeightCache, cache_ptr};
    use half::bf16;

    use crate::dsv4::{Dsv4Model, Dsv4MoeLayer};
    use crate::ops::gemm_batch;

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
            layer.w13.len() == experts_per_rank && layer.w2.len() == experts_per_rank,
            "DSv4 expert cache count mismatch: w13={} w2={} experts_per_rank={}",
            layer.w13.len(),
            layer.w2.len(),
            experts_per_rank
        );

        // Fail loud if the native DeepGEMM bridge is a build-time stub.
        moe::dsv4_deepgemm_native_preflight()?;

        // ── 1+2. Router gemm → logits[T, E] → HOST route (bias or hash). ────────
        let mut logits = HiddenStates::zeros(ctx, cfg.num_experts, num_tokens)?;
        gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
        ctx.sync()?;
        let logits_bf16: Vec<bf16> = ctx
            .stream
            .clone_dtoh(&logits.data)
            .map_err(|e| anyhow::anyhow!("DSv4 router logits D2H failed: {e}"))?;
        let logits_host: Vec<f32> = logits_bf16.iter().map(|&v| v.to_f32()).collect();
        let decisions = dsv4_route(ctx, &model.config, cfg, layer, tokens, &logits_host)?;
        let (indices_host, weights_host) = super::flatten_routing(&decisions, topk)?;
        let total_routes = num_tokens * topk;

        let route_indices = ctx
            .stream
            .clone_htod(&indices_host)
            .map_err(|e| anyhow::anyhow!("DSv4 route-index H2D failed: {e}"))?;
        let route_weights = ctx
            .stream
            .clone_htod(&weights_host)
            .map_err(|e| anyhow::anyhow!("DSv4 route-weight H2D failed: {e}"))?;

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
        let intermediate = layer.w2[0].cols;
        ensure!(
            hidden_dim.is_multiple_of(128) && intermediate.is_multiple_of(128),
            "DSv4 DeepGEMM needs H and I aligned to 128, got H={hidden_dim} I={intermediate}"
        );
        ensure!(
            layer.w13[0].rows == intermediate * 2 && layer.w13[0].cols == hidden_dim,
            "DSv4 w13 cache shape {}x{} != [2*I={}, H={}]",
            layer.w13[0].rows,
            layer.w13[0].cols,
            intermediate * 2,
            hidden_dim
        );
        ensure!(
            layer.w2[0].rows == hidden_dim,
            "DSv4 w2 cache rows {} != hidden {}",
            layer.w2[0].rows,
            hidden_dim
        );

        let expert_out =
            deepgemm_grouped_experts(ctx, layer, &packed_hidden, &counts, &offsets, swiglu_limit)?;

        // ── 6. Scatter weighted expert outputs to route slots, combine topk. ────
        let route_out = HiddenStates::zeros(ctx, hidden_dim, total_routes.max(1))?;
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

        // The shared expert is replicated on every rank. Callers must all-reduce
        // the routed local expert contribution first, then add the shared expert
        // exactly once per rank.
        Ok(())
    }

    pub(crate) fn dsv4_shared_expert_forward(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        hidden: &HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        dsv4_shared_expert(ctx, layer, hidden, swiglu_limit)
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
    ) -> Result<HiddenStates> {
        let num_groups = layer.w13.len();
        let hidden_dim = packed_hidden.hidden_dim;
        let intermediate = layer.w2[0].cols;
        // Concatenate the per-expert FP8 caches into one contiguous group-major
        // buffer the masked GEMM strides per group (w13 = fused [gate|up], w2 =
        // down). See `build_grouped_cache` for the 128-aligned scale-concat note.
        let w13 = build_grouped_cache(ctx, layer.w13.as_slice(), 2 * intermediate, hidden_dim)?;
        let w2 = build_grouped_cache(ctx, layer.w2.as_slice(), hidden_dim, intermediate)?;

        // Upper-bound padded capacity per group = total routes (every route could
        // land on one expert). scale_stride_m mirrors legacy TMA-aligned padding.
        // Floor at 128: DeepGEMM's grouped GEMM small-m tile path diverges below
        // 128 (m=1 decode flips tight-margin tokens); the >=128 floor matches the
        // verified production path (H20 TP=8/EP=8 16/16). NOT a block-scale issue
        // (scale_stride_m=128 alone did not fix it).
        let max_m = packed_hidden.seq_len.max(128);
        let scale_stride_m = max_m.div_ceil(4) * 4;
        let rows = num_groups * max_m;
        let hidden_scale_cols = hidden_dim.div_ceil(128);
        let inter_scale_cols = intermediate.div_ceil(128);

        let input_fp8 = alloc_u8(ctx, rows * hidden_dim)?;
        let input_scales = alloc_zeros_f32(ctx, num_groups * scale_stride_m * hidden_scale_cols)?;
        let w13_out = HiddenStates::zeros(ctx, 2 * intermediate, rows)?;
        let act_fp8 = alloc_u8(ctx, rows * intermediate)?;
        let act_scales = alloc_zeros_f32(ctx, num_groups * scale_stride_m * inter_scale_cols)?;
        let out_padded = HiddenStates::zeros(ctx, hidden_dim, rows)?;
        let mut out_compact = HiddenStates::zeros(ctx, hidden_dim, packed_hidden.seq_len.max(1))?;

        // masked_m = per-group valid row count = the local-expert counts (D2D).
        let mut masked_m = ctx
            .stream
            .alloc_zeros::<i32>(num_groups)
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM masked_m alloc failed: {e}"))?;
        {
            let src = counts.slice(0..num_groups);
            let mut dst = masked_m.slice_mut(0..num_groups);
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM masked_m D2D failed: {e}"))?;
        }

        // Compact identity active-expert table 0..num_groups (dense local layout).
        let active_experts = ctx
            .stream
            .clone_htod(&(0..num_groups as i32).collect::<Vec<i32>>())
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM active-expert H2D failed: {e}"))?;

        let p_hidden = cache_ptr(&packed_hidden.data, ctx);
        let p_in_fp8 = cache_ptr(&input_fp8, ctx);
        let p_in_scales = cache_ptr(&input_scales, ctx);
        let p_active = cache_ptr(&active_experts, ctx);
        let p_offsets = cache_ptr(offsets, ctx);
        let p_counts = cache_ptr(counts, ctx);
        let p_masked = cache_ptr(&masked_m, ctx);
        let p_w13_out = cache_ptr(&w13_out.data, ctx);
        let p_act_fp8 = cache_ptr(&act_fp8, ctx);
        let p_act_scales = cache_ptr(&act_scales, ctx);
        let p_out_padded = cache_ptr(&out_padded.data, ctx);
        let p_out_compact = cache_ptr(&out_compact.data, ctx);
        let stream = ctx.stream.cu_stream();

        // SAFETY: every buffer/ptr above is valid on ctx.stream for the shapes
        // checked against the loaded caches; the kernels bound rows by masked_m.
        unsafe {
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                p_hidden,
                p_in_fp8,
                p_in_scales,
                p_active,
                p_offsets,
                p_counts,
                num_groups,
                max_m,
                hidden_dim,
                scale_stride_m,
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
                max_m,
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
                p_counts,
                num_groups,
                max_m,
                intermediate,
                scale_stride_m,
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
                max_m,
                hidden_dim,
                intermediate,
                scale_stride_m,
                stream,
            )?;
            moe::dsv4_deepgemm_unpad_grouped_bf16(
                p_out_padded,
                p_out_compact,
                p_active,
                p_offsets,
                p_counts,
                num_groups,
                max_m,
                hidden_dim,
                stream,
            )?;
        }

        out_compact.seq_len = packed_hidden.seq_len;
        Ok(out_compact)
    }

    /// DSv4 dense shared expert via a single-group FP8 DeepGEMM pass: w13 fused
    /// gate+up → clamped SwiGLU → w2 down, over every token. No routing/scatter.
    fn dsv4_shared_expert(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        hidden: &HiddenStates,
        swiglu_limit: f32,
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

        let input_fp8 = alloc_u8(ctx, max_m * hidden_dim)?;
        let input_scales = alloc_zeros_f32(ctx, scale_stride_m * hidden_scale_cols)?;
        let w13_out = HiddenStates::zeros(ctx, 2 * shared_inter, max_m)?;
        let act_fp8 = alloc_u8(ctx, max_m * shared_inter)?;
        let act_scales = alloc_zeros_f32(ctx, scale_stride_m * inter_scale_cols)?;
        let out = HiddenStates::zeros(ctx, hidden_dim, num_tokens)?;

        // Single group spanning all tokens: identity expert 0, offset 0, count T.
        let active_experts = ctx
            .stream
            .clone_htod(&[0i32])
            .map_err(|e| anyhow::anyhow!("DSv4 shared active H2D failed: {e}"))?;
        let active_offsets = ctx
            .stream
            .clone_htod(&[0i32])
            .map_err(|e| anyhow::anyhow!("DSv4 shared offset H2D failed: {e}"))?;
        let counts = ctx
            .stream
            .clone_htod(&[num_tokens as i32])
            .map_err(|e| anyhow::anyhow!("DSv4 shared count H2D failed: {e}"))?;
        let masked_m = ctx
            .stream
            .clone_htod(&[num_tokens as i32])
            .map_err(|e| anyhow::anyhow!("DSv4 shared masked_m H2D failed: {e}"))?;

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
        let stream = ctx.stream.cu_stream();

        // SAFETY: single-group buffers valid on ctx.stream; masked_m bounds rows.
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
                stream,
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
                stream,
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
                stream,
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
                stream,
            )?;
        }

        Ok(out)
    }

    /// One contiguous group-major FP8 weight + scale buffer the masked GEMM
    /// strides per group (`b + g * n * k`, `sfb + g * scale_rows * scale_cols`).
    struct GroupedCache {
        weight: cudarc::driver::CudaSlice<u8>,
        scales: cudarc::driver::CudaSlice<f32>,
    }

    /// Concatenate the per-expert FP8 caches into one contiguous group-major
    /// buffer (D2D), validating uniform `[rows, cols]` shape + 128-row alignment.
    ///
    /// Piece 1's loader stores each local expert as a separate
    /// [`Dsv4Fp8DeepGemmWeightCache`] allocation, but the masked grouped GEMM
    /// reads a single `[num_groups, rows, cols]` weight (and matching scale grid).
    /// Because `rows` (`2*intermediate` / `hidden`) is 128-aligned, per-expert
    /// scale grids concatenate exactly into the fused layout legacy builds at load
    /// (`from_dsv4_weight_pair_rows`). Persisting this concat is a Piece 4 / perf
    /// follow-up; the functional port packs per call.
    fn build_grouped_cache(
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
        Ok(GroupedCache { weight, scales })
    }

    fn alloc_u8(ctx: &DeviceContext, len: usize) -> Result<cudarc::driver::CudaSlice<u8>> {
        ctx.stream
            .alloc_zeros::<u8>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM u8 scratch alloc failed: {e}"))
    }

    fn alloc_zeros_f32(ctx: &DeviceContext, len: usize) -> Result<cudarc::driver::CudaSlice<f32>> {
        ctx.stream
            .alloc_zeros::<f32>(len.max(1))
            .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM f32 scratch alloc failed: {e}"))
    }
}

#[cfg(feature = "cuda")]
#[allow(unused_imports)] // consumed by the Piece 2 model.rs DSv4 branch
pub(crate) use dsv4_gpu::{dsv4_moe_forward, dsv4_shared_expert_forward};
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
        let tid2eid: Vec<i64> = vec![0, 1]; // only token 0's two experts present
        // token 1 needs entries [2,3] which the short table can't supply.
        assert!(super::hash_route(&cfg, &tid2eid, &[0u32, 1u32], &[0.0; 8]).is_err());
    }
}
