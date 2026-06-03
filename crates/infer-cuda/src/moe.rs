//! Single-GPU BF16 MoE forward (MoE-4) — Qwen3.5/3.6 SparseMoeBlock.
//!
//! Wires the committed MoE foundation into the CUDA executor's forward:
//!   - [`crate::moe_config::moe_config_from_qwen35`] / [`infer_moe::route`] —
//!     the device-independent router (plain softmax + greedy top-k + optional
//!     `norm_topk_prob` renorm), shared with the CPU reference.
//!   - [`crate::loader::MoeLayerWeights`] — this layer's per-expert gate/up/down
//!     stacks + weight-ptr tables + router gate + shared expert.
//!   - the `cuda_kernels::moe` safe wrappers over the grouped-GEMM + DSv4/Qwen3.6
//!     expert-dispatch kernels.
//!
//! ## Pipeline (single GPU, all experts local — `local_expert_start = 0`,
//! `experts_per_rank = num_experts`)
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
//! Host routing is the simplest correct choice for a first landing: the router
//! gemm output is a small `[T, E]` BF16 tensor, copied to host, run through the
//! CPU `infer_moe::route` (the verified reference), flattened, and re-uploaded.
//! A device router (`dsv4_route_cuda`) is a perf follow-up; correctness first.
//!
//! W4/4-bit (Qwen3.6-35B-A3B canonical is 4bit) is a SEPARATE follow-up: it needs
//! a W4 grouped GEMM. The two `moe_bf16_grouped_gemm_*` call sites below are the
//! exact swap points for a W4 variant; everything else (route, pack, scatter,
//! combine, shared expert) is dtype-agnostic on BF16 activations.

use infer_moe::RoutingDecision;

/// Flatten per-token [`RoutingDecision`]s into the token-major flat buffers the
/// `dsv4_*` route kernels consume: `route_indices[token * topk + k]` is the
/// selected expert id, `route_weights[token * topk + k]` its gate weight.
///
/// This is the host glue between the CPU router ([`infer_moe::route`]) and the
/// device pack/count kernels, which read `indices[route]` / `weights[route]`
/// with `route = token * topk + k` (see `dsv4_route.cu`
/// `dsv4_pack_local_experts_with_slots_kernel`: `token = route / topk`). Each
/// decision must carry exactly `topk` experts; selection order is preserved.
///
/// Returns `(route_indices, route_weights)`, each length `num_tokens * topk`.
/// Feature-agnostic + CPU-tested so the route→assignment plumbing is verified
/// without a GPU. Consumed by [`gpu::moe_forward`] under `cuda`; on a non-CUDA
/// build it is exercised only by the unit tests below.
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

    /// Single-GPU BF16 MoE forward for one sparse layer.
    ///
    /// `normed` is the post-attention-layernorm hidden (`[num_tokens, hidden]`,
    /// token-major). Returns the MoE block output (routed experts + sigmoid-gated
    /// shared expert) as a fresh `[num_tokens, hidden]` [`HiddenStates`].
    ///
    /// All experts are local (single GPU): `local_expert_start = 0`,
    /// `experts_per_rank = num_experts`.
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

        // ── 2. HOST route (the verified infer_moe reference). ───────────────
        // Copy the small [T, E] logits to host, route, flatten back to the
        // token-major route_indices/route_weights the kernels read. Sync first
        // so the gemm has landed before the D2H read.
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
        // These buffers are written by the kernels through raw device pointers
        // (cache_ptr takes `&`), so no Rust-level `mut` binding is needed.
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
        // Reads `normed` token rows; writes packed_hidden[R,H], the route slot
        // map packed_route_slot[R], and packed_weight[R]. cursors[E] scratch.
        // All written via raw device pointers, so no Rust-level `mut`.
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

        // Identity compact→global expert remap. Single GPU runs every expert, so
        // compact index == global index; the grouped GEMM treats a 0..E table as
        // the identity walk (it also accepts a null pointer, but an explicit
        // table keeps the safe-wrapper RawDevicePtr contract — no null hack).
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
        // route_out[route_slot] = weight · expert_out[slot]; unwritten slots
        // (none on single GPU — every route is local) read zero, so the buffer
        // is zero-initialised by HiddenStates::zeros. combine sums over topk.
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
        // Two tokens, 4 experts, top-2. Build logits so the selected experts and
        // their order are deterministic, then check the flat buffers mirror the
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
}
