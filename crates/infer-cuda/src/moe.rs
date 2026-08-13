//! Single-GPU BF16 MoE forward — Qwen3.5/3.6 SparseMoeBlock (all experts local).
//!
//! Routing runs on DEVICE: `dsv4_route` with a zero selection bias is exactly
//! greedy top-k. `--qwen35-gpu-router false` selects the host `infer_moe::route`
//! reference, which costs a full-stream `ctx.sync` + logits D2H + 2×H2D per layer
//! per step. DSv4 uses the device route kernel unconditionally.

use infer_moe::RoutingDecision;

/// Flatten per-token routing into the token-major flat buffers the `dsv4_*`
/// kernels read at `route = token * topk + k`, each of length `num_tokens *
/// topk`. Each decision must carry exactly `topk` experts; order is preserved.
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

/// DeepGEMM m-grouped-contiguous row alignment on SM90: the kernel resolves the
/// B-side expert group ONCE per BLOCK_M output tile from `m_indices[tile_start]`
/// and BLOCK_M is pinned to 128 upstream — so every group's row segment must
/// start at a multiple of 128, with `-1` on every pad row.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const DEEPGEMM_CONTIG_ALIGN: usize = 128;

/// DSv4 decode-band per-group row alignment. 64 is the smallest legal value:
/// the SM90 warpgroup MMA grants only block_m ∈ {64, 128}, and the native bridge
/// caps block_m at the alignment the caller packed with. At decode shapes
/// (R = topk·B ≤ 128) the row-linear kernels dominate, so halving pad rows halves
/// them (128-alignment inflated the lane 149×, −23% e2e B=1).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const DSV4_DECODE_CONTIG_ALIGN: usize = 64;

/// Routed-row ceiling for the 64-aligned decode band: `topk(8) × B_max(16)`.
/// Above it the GEMM tail dominates and block_m=128 wins — use
/// [`DEEPGEMM_CONTIG_ALIGN`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const DSV4_DECODE_CONTIG_MAX_ROUTES: usize = 128;

/// Host upper bound on the `align`-aligned packed row count for the DeepGEMM
/// contiguous grouped layout. The true total is device-resident while the GEMM's
/// `m` and the TMA descriptors are host values, so the launch uses this cap
/// instead of a per-layer D2H sync:
/// `Σ_g align(c_g, a) ≤ align(R, a) + a·min(R, G)`. Tiles past the true aligned
/// total carry `m_indices = -1`. Always a multiple of `align`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn deepgemm_contig_rows_cap(
    total_routes: usize,
    local_experts: usize,
    align: usize,
) -> usize {
    total_routes.div_ceil(align) * align + align * local_experts.min(total_routes)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
/// Routed-row floor below which the DeepGEMM grouped path loses to the hand
/// CUDA-core kernels (decode R=8 hand +8%, prefill R=16384 DeepGEMM -33% needle
/// wall; 1024 = 128-token chunk x top-8). The live value is
/// `--qwen35-deepgemm-min-routes`.
pub(crate) const QWEN35_DEEPGEMM_MIN_ROUTES: usize = 1024;

/// Routed-row ceiling for the decode-specialized weight-read-bound grouped
/// kernels: `256 = top_k(8) × B_max(32)`, the batched-decode envelope where each
/// expert receives ≤ B rows so weight traffic dominates (the batch kernels burn
/// 487 µs/layer at R=8 on scalar 2B loads, ≈3% of HBM bandwidth, vs a 25-60
/// µs/layer target).
///
/// Must stay `< QWEN35_DEEPGEMM_MIN_ROUTES` so the decode band never shadows the
/// DeepGEMM prefill dispatch.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const QWEN35_MOE_DECODE_MAX_ROUTES: usize = 256;

/// `--qwen35-deepgemm` (default on): DeepGEMM SM90 BF16 m-grouped GEMMs for the
/// expert GEMMs — decode neutral (40.86 vs 40.46 tok/s), prefill needle 3k wall
/// 9.10 -> 2.32 s (-74.5%). Also read at LOAD time: the loader builds the
/// contiguous grouped-B caches only when enabled, so flipping it requires a
/// process restart.
#[cfg(feature = "cuda")]
/// `--qwen35-moe-decode-kernel` (default on): the decode-band weight-read-bound
/// grouped kernels; `false` runs the hand batch kernels at every routed-row count
/// below the DeepGEMM floor. Read per call — inside a captured decode graph the
/// value read at capture time is what replays.
#[cfg(feature = "cuda")]
/// Allocate an i32 buffer pre-filled with `-1` ON DEVICE (memset 0xFF) — the
/// "route slot not packed on this rank" sentinel.
///
/// A `clone_htod(&vec![-1; n])` would record a CUDA-graph memcpy node whose HOST
/// source dies with this call, so replay reads freed memory.
#[cfg(feature = "cuda")]
fn alloc_neg1_i32(
    ctx: &cuda_kernels::prelude::DeviceContext,
    len: usize,
) -> anyhow::Result<cudarc::driver::CudaSlice<i32>> {
    use cudarc::driver::DevicePtrMut;

    let mut buf = ctx
        .stream
        .alloc_zeros::<i32>(len)
        .map_err(|e| anyhow::anyhow!("MoE -1 sentinel alloc failed: {e}"))?;
    let n_bytes = len * std::mem::size_of::<i32>();
    let (ptr, _guard) = buf.device_ptr_mut(&ctx.stream);
    // SAFETY: `ptr` is a live device allocation of `n_bytes` on this stream.
    unsafe {
        cudarc::driver::result::memset_d8_async(ptr, 0xFF, n_bytes, ctx.stream.cu_stream())
            .map_err(|e| anyhow::anyhow!("MoE -1 sentinel memset failed: {e}"))?;
    }
    drop(_guard);
    Ok(buf)
}

#[cfg(feature = "cuda")]
mod gpu {
    use anyhow::{Result, ensure};
    use cuda_kernels::moe;
    use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, HiddenStates, RawDevicePtr};
    use cuda_kernels::tensor::WeightFormat;
    use cuda_kernels::tensor::cache_ptr;
    use cudarc::driver::sys::CUevent_flags;
    use half::bf16;
    use infer_moe::{MoeConfig, ScoringFunc, TopkMethod};
    use std::sync::OnceLock;
    use std::time::Instant;

    use super::{
        DEEPGEMM_CONTIG_ALIGN, QWEN35_DEEPGEMM_MIN_ROUTES, QWEN35_MOE_DECODE_MAX_ROUTES,
        deepgemm_contig_rows_cap,
    };
    use crate::loader::{ExpertQuantDispatchSignature, MoeLayerWeights};
    use crate::moe_config::ExpertSplit;
    use crate::ops::{gemm_batch, silu_mul};
    use crate::workspace::{HiddenSlot, SliceSlot};

    /// DeepGEMM masked per-group band capacity (rows). Must be a multiple of
    /// 128: the masked GEMM's TMA store writes full BLOCK_M (64/128) output
    /// tiles, so a smaller band would let a tile cross into the next group's
    /// band. Host-shape-fixed (`[G, 128, K]`), hence CUDA-graph-safe.
    const DEEPGEMM_MASKED_BAND: usize = 128;

    fn qwen_moe_profile_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("ARLE_QWEN35_MOE_PROFILE").is_some())
    }

    fn qwen_moe_profile<T>(
        ctx: &DeviceContext,
        label: &'static str,
        rows: usize,
        n: usize,
        k: usize,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !qwen_moe_profile_enabled() {
            return f();
        }
        let start = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        start.record(&ctx.stream)?;
        let host_t0 = Instant::now();
        let result = f();
        let host_ms = host_t0.elapsed().as_secs_f64() * 1000.0;
        stop.record(&ctx.stream)?;
        stop.synchronize()?;
        let cuda_ms = start.elapsed_ms(&stop)? as f64;
        if std::env::var("INFER_TP_RANK")
            .map(|rank| rank == "0")
            .unwrap_or(true)
        {
            eprintln!(
                "[qwen-moe-profile] {label} rows={rows} n={n} k={k} cuda_ms={cuda_ms:.3} host_ms={host_ms:.3}"
            );
        }
        result
    }

    /// Persistent device scratch for [`moe_forward_into`], exact-shape reuse:
    /// decode (`R = topk`) hits the cache every step, prefill re-allocates only
    /// when the chunk shape changes.
    ///
    /// Only slots their writer does not fully overwrite need per-call init:
    /// `counts`/`cursors` → 0 (atomicAdd accumulators); under EP additionally
    /// `packed_route_slot` → -1 (the scatter skips `< 0`; a stale tail would alias
    /// a live slot) and `route_out` → 0 (the combine sums non-local slots as 0
    /// into the partial). The DeepGEMM path also needs `dg_m_indices` → -1 and
    /// `packed_route_slot` → -1 even single-GPU, since pad rows stay unwritten.
    #[derive(Default)]
    pub(crate) struct MoeForwardScratch {
        logits: HiddenSlot,
        route_indices: SliceSlot<i32>,
        route_weights: SliceSlot<f32>,
        router_bias_zero: SliceSlot<bf16>,
        counts: SliceSlot<i32>,
        offsets: SliceSlot<i32>,
        scan_total: SliceSlot<i32>,
        packed_hidden: HiddenSlot,
        packed_route_slot: SliceSlot<i32>,
        packed_weight: SliceSlot<f32>,
        cursors: SliceSlot<i32>,
        expert_indices: SliceSlot<i32>,
        dg_band_offsets: SliceSlot<i32>,
        dg_aligned_offsets: SliceSlot<i32>,
        dg_m_indices: SliceSlot<i32>,
        dg_input_fp8: SliceSlot<u8>,
        dg_input_scales: SliceSlot<f32>,
        dg_act_fp8: SliceSlot<u8>,
        dg_act_scales: SliceSlot<f32>,
        dg_active_experts: SliceSlot<i32>,
        dg_active_offsets: SliceSlot<i32>,
        dg_active_counts: SliceSlot<i32>,
        gate_out: HiddenSlot,
        up_out: HiddenSlot,
        act: HiddenSlot,
        expert_out: HiddenSlot,
        route_out: HiddenSlot,
        shared_gate: HiddenSlot,
        shared_up: HiddenSlot,
        shared_act: HiddenSlot,
        shared_out: HiddenSlot,
        gate_logit: HiddenSlot,
    }

    impl MoeForwardScratch {
        pub(crate) fn new() -> Self {
            Self::default()
        }

        pub(crate) fn release(&mut self) {
            let Self {
                logits,
                route_indices,
                route_weights,
                router_bias_zero,
                counts,
                offsets,
                scan_total,
                packed_hidden,
                packed_route_slot,
                packed_weight,
                cursors,
                expert_indices,
                dg_band_offsets,
                dg_aligned_offsets,
                dg_m_indices,
                dg_input_fp8,
                dg_input_scales,
                dg_act_fp8,
                dg_act_scales,
                dg_active_experts,
                dg_active_offsets,
                dg_active_counts,
                gate_out,
                up_out,
                act,
                expert_out,
                route_out,
                shared_gate,
                shared_up,
                shared_act,
                shared_out,
                gate_logit,
            } = self;
            logits.release();
            route_indices.release();
            route_weights.release();
            router_bias_zero.release();
            counts.release();
            offsets.release();
            scan_total.release();
            packed_hidden.release();
            packed_route_slot.release();
            packed_weight.release();
            cursors.release();
            expert_indices.release();
            dg_band_offsets.release();
            dg_aligned_offsets.release();
            dg_m_indices.release();
            dg_input_fp8.release();
            dg_input_scales.release();
            dg_act_fp8.release();
            dg_act_scales.release();
            dg_active_experts.release();
            dg_active_offsets.release();
            dg_active_counts.release();
            gate_out.release();
            up_out.release();
            act.release();
            expert_out.release();
            route_out.release();
            shared_gate.release();
            shared_up.release();
            shared_act.release();
            shared_out.release();
            gate_logit.release();
        }
    }

    fn device_route_eligible(cfg: &MoeConfig) -> bool {
        cfg.topk_method == TopkMethod::Greedy && cfg.n_group.is_none() && cfg.topk_group.is_none()
    }

    /// Decode-graph gate: TRUE iff a `seq_len == 1` MoE step is a pure
    /// device-kernel sequence — the device router (no host sync + D2H) and
    /// `R = top_k` below the DeepGEMM floor, whose JIT is not capture-safe.
    pub(crate) fn qwen35_decode_moe_graph_capturable(cfg: &MoeConfig) -> bool {
        crate::runtime_flags::qwen35_gpu_router()
            && device_route_eligible(cfg)
            && cfg.top_k < QWEN35_DEEPGEMM_MIN_ROUTES
    }

    /// Allocate-per-call wrapper around [`moe_forward_into`]; the hot loop passes
    /// a persistent scratch instead.
    pub(crate) fn moe_forward(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        cfg: &MoeConfig,
        split: &ExpertSplit,
    ) -> Result<HiddenStates> {
        let mut scratch = MoeForwardScratch::new();
        let mut out = HiddenStates::zeros(ctx, normed.hidden_dim, normed.seq_len)?;
        moe_forward_into(ctx, weights, normed, cfg, split, &mut scratch, &mut out)?;
        Ok(out)
    }

    /// BF16 MoE forward for one sparse layer. `normed` is the post-LN hidden
    /// `[num_tokens, hidden]`; the block output (routed + sigmoid-gated shared
    /// expert) fully overwrites `out` (`[hidden, num_tokens]`).
    ///
    /// Routing runs over ALL `cfg.num_experts`, but only routes landing on
    /// `split`'s local experts contribute — under `ep_size > 1` `out` is a
    /// PARTIAL sum the caller must `all_reduce_sum` before the residual add.
    pub(crate) fn moe_forward_into(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        cfg: &MoeConfig,
        split: &ExpertSplit,
        scratch: &mut MoeForwardScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let num_tokens = normed.seq_len;
        let hidden_dim = normed.hidden_dim;
        let num_experts = cfg.num_experts;
        let topk = cfg.top_k;
        ensure!(
            split.num_experts == num_experts,
            "MoE expert split covers {} experts but the router has {num_experts}",
            split.num_experts
        );
        let local_experts = split.experts_per_rank;
        ensure!(
            weights.router_gate.cols == hidden_dim && weights.router_gate.rows == num_experts,
            "MoE router shape mismatch: gate={}x{} hidden_dim={} num_experts={}",
            weights.router_gate.rows,
            weights.router_gate.cols,
            hidden_dim,
            num_experts
        );
        // Hybrid dispatch by routed-row count: at decode R=8 the masked grouped
        // path loses to the hand CUDA-core kernels (37.5 vs 40.8 tok/s, fixed
        // JIT/TMA overhead on tiny bands), at prefill R=16384 the contiguous
        // tensor-core path wins (needle 3k wall 9.07 -> 6.10 s).
        let has_deepgemm_grouped = weights.gate_grouped.is_some()
            || (weights.w13_fp8_grouped.is_some() && weights.down_fp8_grouped.is_some());
        // FP8 experts only have the CONTIGUOUS DeepGEMM layout; the masked band
        // is BF16-only and errors out mid-step. The routed-row floor is tunable,
        // so gate rather than assert — a lowered floor must fall back to the hand
        // kernels instead of killing the engine.
        let fp8_masked_unsupported = weights.expert_weight_format == WeightFormat::Fp8BlockScaled
            && num_tokens * topk <= DEEPGEMM_MASKED_BAND;
        let use_deepgemm = crate::runtime_flags::qwen35_deepgemm()
            && has_deepgemm_grouped
            && !fp8_masked_unsupported
            && num_tokens * topk >= crate::runtime_flags::qwen35_deepgemm_min_routes();
        if !use_deepgemm {
            // Grouped-mode loads cleared the per-expert Vecs (the hand kernels
            // run through the rebuilt ptr tables), so accept either weight form.
            ensure!(
                has_deepgemm_grouped
                    || (weights.gate.len() == local_experts
                        && weights.up.len() == local_experts
                        && weights.down.len() == local_experts),
                "MoE expert count mismatch: gate={} up={} down={} local_experts={local_experts} \
                 (ep_size={} ep_rank={})",
                weights.gate.len(),
                weights.up.len(),
                weights.down.len(),
                split.ep_size,
                split.ep_rank
            );
        }
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == num_tokens,
            "MoE output shape mismatch: out={}x{} expected {hidden_dim}x{num_tokens}",
            out.hidden_dim,
            out.seq_len
        );

        let logits = scratch.logits.get(ctx, num_experts, num_tokens)?;
        gemm_batch(ctx, &weights.router_gate, normed, logits)?;

        let total_routes = num_tokens * topk;
        let (route_indices, route_weights) =
            if crate::runtime_flags::qwen35_gpu_router() && device_route_eligible(cfg) {
                // `dsv4_route` with routing_kind=1 and an all-zero bias IS greedy
                // top-k: key = scores + 0.0 exactly (softmax scores are >= +0.0, so
                // `x + 0.0f == x` bitwise). The `norm_topk_prob` renorm is the
                // separate launch below. The bias table is a pure function of its
                // length, so `upload_const` uploads once.
                let bias_zero_host = vec![bf16::ZERO; num_experts];
                let bias_zero = scratch
                    .router_bias_zero
                    .upload_const(ctx, &bias_zero_host)?;
                let bias_zero_ptr = cache_ptr(bias_zero, ctx);
                let route_indices = scratch.route_indices.get(ctx, total_routes)?;
                let route_weights = scratch.route_weights.get(ctx, total_routes)?;
                // SAFETY: logits `[E, T]`, indices/weights `[T*topk]`, bias `[E]`
                // are live scratch buffers on ctx.stream. Phase 3 of the route
                // kernel writes EVERY indices/weights slot unconditionally, so
                // length-matched slot reuse needs no re-init; the renorm kernel
                // rewrites the same freshly written slots in place.
                unsafe {
                    moe::dsv4_route(
                        cache_ptr(&logits.data, ctx),
                        Some(bias_zero_ptr),
                        None,
                        None,
                        cache_ptr(route_indices, ctx),
                        cache_ptr(route_weights, ctx),
                        num_tokens,
                        num_experts,
                        topk,
                        1, // learned-bias selection key; zero bias ⇒ greedy
                        cfg.scoring_func.scoring_kind(),
                        cfg.routed_scaling_factor,
                        ctx.stream.cu_stream(),
                    )?;
                    // Non-softmax scoring already normalizes inside the route kernel.
                    if cfg.norm_topk_prob && cfg.scoring_func == ScoringFunc::Softmax {
                        moe::qwen36_renorm_topk_weights(
                            cache_ptr(route_weights, ctx),
                            num_tokens,
                            topk,
                            ctx.stream.cu_stream(),
                        )?;
                    }
                }
                (&*route_indices, &*route_weights)
            } else {
                // Sync so the router gemm has landed before the D2H read.
                ctx.sync()?;
                let logits_bf16: Vec<bf16> = ctx
                    .stream
                    .clone_dtoh(&logits.data)
                    .map_err(|e| anyhow::anyhow!("MoE router logits D2H failed: {e}"))?;
                let logits_host: Vec<f32> = logits_bf16.iter().map(|&v| v.to_f32()).collect();
                let decisions = infer_moe::route(&logits_host, &[], cfg)
                    .map_err(|e| anyhow::anyhow!("MoE host route failed: {e}"))?;
                let (indices_host, weights_host) = super::flatten_routing(&decisions, topk)?;
                ensure!(
                    indices_host.len() == total_routes && weights_host.len() == total_routes,
                    "flattened routing length mismatch"
                );

                let route_indices = scratch.route_indices.upload(ctx, &indices_host)?;
                let route_weights = scratch.route_weights.upload(ctx, &weights_host)?;
                (route_indices, route_weights)
            };

        // Raw-ptr conversion ends the scratch field borrows so the downstream
        // tails can take `&mut scratch`; the buffers persist in the scratch.
        let route_indices_ptr = cache_ptr(route_indices, ctx);
        let route_weights_ptr = cache_ptr(route_weights, ctx);

        // counts MUST be zeroed every call (atomicAdd accumulator).
        let counts_ptr = cache_ptr(scratch.counts.get_zeroed(ctx, local_experts)?, ctx);
        // SAFETY: all buffers are valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_count_local_experts(
                route_indices_ptr,
                counts_ptr,
                num_tokens,
                topk,
                split.local_expert_start,
                local_experts,
                ctx.stream.cu_stream(),
            )?;
        }

        if use_deepgemm {
            deepgemm_routed_tail(
                ctx,
                weights,
                normed,
                split,
                scratch,
                route_indices_ptr,
                route_weights_ptr,
                counts_ptr,
                out,
                topk,
            )?;
            return add_shared_expert_gated(ctx, weights, normed, scratch, out);
        }

        let offsets = scratch.offsets.get(ctx, local_experts)?;
        let scan_total = scratch.scan_total.get(ctx, 1)?;
        // SAFETY: counts/offsets/scan_total valid on ctx.stream.
        unsafe {
            moe::dsv4_exclusive_scan_i32(
                counts_ptr,
                cache_ptr(offsets, ctx),
                cache_ptr(scan_total, ctx),
                local_experts,
                ctx.stream.cu_stream(),
            )?;
        }

        // Under EP only routes hitting LOCAL experts get packed, so the tail of
        // `packed_route_slot` keeps a STALE slot id on reuse; the scatter skips
        // `< 0`, so EP re-fills the -1 sentinel every call. cursors MUST be
        // zeroed every call (atomicAdd slot allocator).
        let packed_hidden = scratch.packed_hidden.get(ctx, hidden_dim, total_routes)?;
        let packed_route_slot = if split.ep_size == 1 {
            scratch.packed_route_slot.get(ctx, total_routes.max(1))?
        } else {
            scratch
                .packed_route_slot
                .neg1_filled(ctx, total_routes.max(1))?
        };
        let packed_weight = scratch.packed_weight.get(ctx, total_routes.max(1))?;
        let cursors = scratch.cursors.get_zeroed(ctx, local_experts)?;
        // SAFETY: buffers valid on ctx.stream; shapes checked by the kernel.
        unsafe {
            moe::dsv4_pack_local_experts_with_slots(
                cache_ptr(&normed.data, ctx),
                route_indices_ptr,
                route_weights_ptr,
                cache_ptr(offsets, ctx),
                cache_ptr(cursors, ctx),
                cache_ptr(&packed_hidden.data, ctx),
                cache_ptr(packed_route_slot, ctx),
                cache_ptr(packed_weight, ctx),
                num_tokens,
                hidden_dim,
                topk,
                split.local_expert_start,
                local_experts,
                ctx.stream.cu_stream(),
            )?;
        }

        // The weight-pointer tables hold ONLY this rank's experts, so group g
        // indexes table entry g. Pure function of `local_experts`, so a
        // length-matched reuse skips the H2D.
        let expert_index_table: Vec<i32> = (0..local_experts as i32).collect();
        let expert_indices = scratch
            .expert_indices
            .upload_const(ctx, &expert_index_table)?;

        // Grouped-mode loads carry the shapes on the group (per-expert Vecs are
        // cleared); concat already enforced uniformity.
        let moe_inter = match (
            &weights.gate_grouped,
            &weights.w13_fp8_grouped,
            weights.gate.first(),
        ) {
            (Some(g), _, _) => g.rows,
            (None, Some(g), _) => {
                ensure!(
                    g.rows.is_multiple_of(2),
                    "FP8 MoE fused w13 grouped rows {} must be even",
                    g.rows
                );
                g.rows / 2
            }
            (None, None, Some(first)) => {
                let mi = first.rows;
                expert_shape_ok(first, &weights.up[0], hidden_dim, mi)?;
                mi
            }
            (None, None, None) => {
                anyhow::bail!("MoE weights carry neither grouped nor per-expert experts")
            }
        };
        // `max_count` only sizes grid Y; total_routes is a safe upper bound.
        let max_count = total_routes.max(1);
        let (down_rows, down_cols) = match (
            &weights.down_grouped,
            &weights.down_fp8_grouped,
            weights.down.first(),
        ) {
            (Some(g), _, _) => (g.rows, g.cols),
            (None, Some(g), _) => (g.rows, g.cols),
            (None, None, Some(first)) => (first.rows, first.cols),
            (None, None, None) => {
                anyhow::bail!("MoE weights carry neither grouped nor per-expert down")
            }
        };
        ensure!(
            down_cols == moe_inter && down_rows == hidden_dim,
            "MoE expert down shape {}x{} != hidden_dim {} / moe_inter {}",
            down_rows,
            down_cols,
            hidden_dim,
            moe_inter
        );

        // Decode-band dispatch: the weight-read-bound decode kernels replace
        // pair-GEMM + silu_mul + down-GEMM (3 launches -> 2). Both contraction
        // dims must be 16B-vector rows (gate/up k = H, down k = I); otherwise the
        // batch kernels keep the shape. Shape-constant, so capture-safe.
        let use_bf16_decode_kernels = weights.expert_weight_format == WeightFormat::DenseBf16
            && total_routes <= QWEN35_MOE_DECODE_MAX_ROUTES
            && hidden_dim.is_multiple_of(8)
            && moe_inter.is_multiple_of(8)
            && crate::runtime_flags::qwen35_moe_decode_kernel();
        let fp8_decode_scale_cols = if weights.expert_weight_format == WeightFormat::Fp8BlockScaled
            && total_routes <= QWEN35_MOE_DECODE_MAX_ROUTES
            && hidden_dim.is_multiple_of(16)
            && moe_inter.is_multiple_of(16)
            && crate::runtime_flags::qwen35_moe_decode_kernel()
        {
            let (gate_scale_rows, gate_scale_cols, gate_block_m, gate_block_k) =
                fp8_signature_shape(weights.gate_up_quant_signature, "gate/up")?;
            let (down_scale_rows, down_scale_cols, down_block_m, down_block_k) =
                fp8_signature_shape(weights.down_quant_signature, "down")?;
            ensure!(
                gate_block_m == 128
                    && gate_block_k == 128
                    && down_block_m == 128
                    && down_block_k == 128,
                "Qwen FP8 decode-fused path requires 128x128 block scales, got gate {gate_block_m}x{gate_block_k}, down {down_block_m}x{down_block_k}"
            );
            ensure!(
                gate_scale_cols == hidden_dim.div_ceil(128)
                    && down_scale_cols == moe_inter.div_ceil(128),
                "Qwen FP8 decode-fused scale cols mismatch: gate {gate_scale_cols} vs {}, down {down_scale_cols} vs {}",
                hidden_dim.div_ceil(128),
                moe_inter.div_ceil(128)
            );
            ensure!(
                gate_scale_rows == moe_inter.div_ceil(128)
                    && down_scale_rows == hidden_dim.div_ceil(128),
                "Qwen FP8 decode-fused scale rows mismatch: gate {gate_scale_rows} vs {}, down {down_scale_rows} vs {}",
                moe_inter.div_ceil(128),
                hidden_dim.div_ceil(128)
            );
            Some((gate_scale_cols, down_scale_cols))
        } else {
            None
        };

        // Gate+up GEMM + SwiGLU (UNCLAMPED — Qwen3.6 has no clamp).
        let act = scratch.act.get(ctx, moe_inter, total_routes)?;
        if use_bf16_decode_kernels {
            // SAFETY: weight-ptr tables + packed buffers valid on ctx.stream;
            // k = hidden_dim % 8 == 0 checked by the dispatch above.
            unsafe {
                moe::moe_bf16_grouped_gemm_swiglu_decode(
                    &weights.gate_ptrs,
                    &weights.up_ptrs,
                    cache_ptr(&packed_hidden.data, ctx),
                    cache_ptr(&act.data, ctx),
                    cache_ptr(offsets, ctx),
                    counts_ptr,
                    cache_ptr(expert_indices, ctx),
                    local_experts,
                    max_count,
                    moe_inter,
                    hidden_dim,
                    ctx,
                    ctx.stream.cu_stream(),
                )?;
            }
        } else if let Some((gate_scale_cols, _down_scale_cols)) = fp8_decode_scale_cols {
            // Same compact decode-band kernel family as the DSv4 FP8 lane with
            // Qwen's unclamped SwiGLU (`limit = inf`), avoiding the generic batch
            // GEMV's [num_experts, max_count] launch grid on the decode band.
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                moe::dsv4_fp8_grouped_swiglu_decode(
                    cache_ptr(&weights.gate_ptrs, ctx),
                    cache_ptr(opt_ptrs(&weights.gate_scale_ptrs, "gate FP8 scale")?, ctx),
                    cache_ptr(&weights.up_ptrs, ctx),
                    cache_ptr(opt_ptrs(&weights.up_scale_ptrs, "up FP8 scale")?, ctx),
                    cache_ptr(&packed_hidden.data, ctx),
                    cache_ptr(&act.data, ctx),
                    cache_ptr(offsets, ctx),
                    counts_ptr,
                    local_experts,
                    max_count,
                    moe_inter,
                    hidden_dim,
                    gate_scale_cols,
                    f32::INFINITY,
                    ctx.stream.cu_stream(),
                )?;
            }
        } else {
            let gate_out = scratch.gate_out.get(ctx, moe_inter, total_routes)?;
            let up_out = scratch.up_out.get(ctx, moe_inter, total_routes)?;
            // SAFETY: weight-ptr tables + packed buffers valid on ctx.stream.
            unsafe {
                grouped_pair_batch(
                    ctx,
                    weights,
                    packed_hidden,
                    gate_out,
                    up_out,
                    offsets,
                    counts_ptr,
                    expert_indices,
                    local_experts,
                    max_count,
                    moe_inter,
                    hidden_dim,
                )?;
            }
            silu_mul(ctx, gate_out, up_out, act)?;
        }

        let expert_out = scratch.expert_out.get(ctx, hidden_dim, total_routes)?;
        if use_bf16_decode_kernels {
            // SAFETY: down weight table + act/expert_out valid on ctx.stream;
            // k = moe_inter % 8 == 0 checked by the dispatch above.
            unsafe {
                moe::moe_bf16_grouped_gemm_decode(
                    &weights.down_ptrs,
                    cache_ptr(&act.data, ctx),
                    cache_ptr(&expert_out.data, ctx),
                    cache_ptr(offsets, ctx),
                    counts_ptr,
                    cache_ptr(expert_indices, ctx),
                    local_experts,
                    max_count,
                    hidden_dim,
                    moe_inter,
                    ctx,
                    ctx.stream.cu_stream(),
                )?;
            }
        } else if let Some((_gate_scale_cols, down_scale_cols)) = fp8_decode_scale_cols {
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                moe::dsv4_fp8_grouped_down_decode(
                    cache_ptr(&weights.down_ptrs, ctx),
                    cache_ptr(opt_ptrs(&weights.down_scale_ptrs, "down FP8 scale")?, ctx),
                    cache_ptr(&act.data, ctx),
                    cache_ptr(&expert_out.data, ctx),
                    cache_ptr(offsets, ctx),
                    counts_ptr,
                    local_experts,
                    max_count,
                    hidden_dim,
                    moe_inter,
                    down_scale_cols,
                    ctx.stream.cu_stream(),
                )?;
            }
        } else {
            // SAFETY: down weight table + act/expert_out valid on ctx.stream.
            unsafe {
                grouped_down_batch(
                    ctx,
                    weights,
                    act,
                    expert_out,
                    offsets,
                    counts_ptr,
                    expert_indices,
                    local_experts,
                    max_count,
                    hidden_dim,
                    moe_inter,
                )?;
            }
        }

        // Single-GPU the scatter writes ALL `total_routes` slots (route↔slot
        // bijection); under EP the unwritten non-local slots MUST read zero in
        // the combine (partial-sum contract), so EP re-zeros every call.
        let route_out = if split.ep_size == 1 {
            scratch.route_out.get(ctx, hidden_dim, total_routes)?
        } else {
            scratch
                .route_out
                .get_zeroed(ctx, hidden_dim, total_routes)?
        };
        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_scatter_all_route_slots(
                cache_ptr(&expert_out.data, ctx),
                cache_ptr(&route_out.data, ctx),
                cache_ptr(packed_route_slot, ctx),
                cache_ptr(packed_weight, ctx),
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

        add_shared_expert_gated(ctx, weights, normed, scratch, out)
    }

    /// Dense shared expert, sigmoid-gated and accumulated into the routed output
    /// (`out` is RMW'd; the scatter/combine stage fully wrote it).
    fn add_shared_expert_gated(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        scratch: &mut MoeForwardScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let num_tokens = normed.seq_len;
        let hidden_dim = normed.hidden_dim;
        let shared = shared_expert_forward(
            ctx,
            weights,
            normed,
            &mut scratch.shared_gate,
            &mut scratch.shared_up,
            &mut scratch.shared_act,
            &mut scratch.shared_out,
        )?;
        let gate_logit = scratch.gate_logit.get(ctx, 1, num_tokens)?;
        gemm_batch(ctx, &weights.shared_gate_router, normed, gate_logit)?;
        // SAFETY: out/shared/gate_logit valid on ctx.stream.
        unsafe {
            moe::qwen36_add_shared_expert_gated(
                cache_ptr(&out.data, ctx),
                cache_ptr(&shared.data, ctx),
                cache_ptr(&gate_logit.data, ctx),
                num_tokens,
                hidden_dim,
                ctx.stream.cu_stream(),
            )?;
        }

        Ok(())
    }

    /// Expert path on the DeepGEMM m-grouped GEMMs: pack → gate/up GEMM →
    /// silu_mul → down GEMM → scatter/combine into `out`.
    ///
    /// * **Masked** (`R <= 128`, decode): fixed per-group bands `[G, 128, *]`
    ///   with `masked_m = counts`. 128 is exactly the dispatch threshold because
    ///   the only host-provable capacity bound is `max_g count_g <= R`; shapes
    ///   are routing-independent (CUDA-graph-safe).
    /// * **Contiguous** (`R > 128`, prefill): 128-aligned per-group segments plus
    ///   `m_indices` row→group ids (-1 pads). Masked cannot serve this regime —
    ///   one hot expert can take all R rows, needing `G·R` rows of scratch. Pad
    ///   tiles resolve to group 0 and the route-slot -1 sentinel excludes them.
    #[allow(clippy::too_many_arguments)]
    fn deepgemm_routed_tail(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        split: &ExpertSplit,
        scratch: &mut MoeForwardScratch,
        route_indices: RawDevicePtr<i32>,
        route_weights: RawDevicePtr<f32>,
        counts: RawDevicePtr<i32>,
        out: &mut HiddenStates,
        topk: usize,
    ) -> Result<()> {
        let num_tokens = normed.seq_len;
        let hidden_dim = normed.hidden_dim;
        let total_routes = num_tokens * topk;
        let local_experts = split.experts_per_rank;
        let stream = ctx.stream.cu_stream();
        // sm_120 has no DeepGEMM native bridge (Hopper-only); its FP8 grouped
        // GEMMs use the CUTLASS sm_120a collective on the same buffers.
        let sm120 = ctx.is_sm120();

        let fp8_grouped = match (&weights.w13_fp8_grouped, &weights.down_fp8_grouped) {
            (Some(w13), Some(down)) => Some((w13, down)),
            _ => None,
        };
        let bf16_grouped = match (
            &weights.gate_grouped,
            &weights.up_grouped,
            &weights.down_grouped,
        ) {
            (Some(g), Some(u), Some(d)) => Some((g, u, d)),
            _ => None,
        };
        let moe_inter_for_abi = if let Some((w13, down_g)) = fp8_grouped {
            ensure!(
                w13.rows.is_multiple_of(2),
                "FP8 DeepGEMM MoE fused w13 rows {} must be even",
                w13.rows
            );
            let moe_inter = w13.rows / 2;
            ensure!(
                w13.groups == local_experts
                    && down_g.groups == local_experts
                    && w13.cols == hidden_dim
                    && down_g.rows == hidden_dim
                    && down_g.cols == moe_inter,
                "FP8 DeepGEMM MoE grouped cache shape mismatch: w13={}x{} g={} down={}x{} g={} H={hidden_dim}",
                w13.rows,
                w13.cols,
                w13.groups,
                down_g.rows,
                down_g.cols,
                down_g.groups
            );
            ensure!(
                hidden_dim.is_multiple_of(128) && moe_inter.is_multiple_of(128),
                "FP8 DeepGEMM MoE needs H and I aligned to 128, got H={hidden_dim} I={moe_inter}"
            );
            moe_inter
        } else if let Some((gate_g, up_g, down_g)) = bf16_grouped {
            let moe_inter = gate_g.rows;
            ensure!(
                gate_g.groups == local_experts
                    && up_g.groups == local_experts
                    && down_g.groups == local_experts,
                "DeepGEMM MoE group count mismatch: gate={} up={} down={} local_experts={local_experts}",
                gate_g.groups,
                up_g.groups,
                down_g.groups
            );
            ensure!(
                gate_g.cols == hidden_dim
                    && up_g.rows == moe_inter
                    && up_g.cols == hidden_dim
                    && down_g.rows == hidden_dim
                    && down_g.cols == moe_inter,
                "DeepGEMM MoE grouped cache shape mismatch: gate={}x{} up={}x{} down={}x{} H={hidden_dim} I={moe_inter}",
                gate_g.rows,
                gate_g.cols,
                up_g.rows,
                up_g.cols,
                down_g.rows,
                down_g.cols
            );
            // BF16 kernel constraints: K % 64 (BLOCK_K) and N % 8, in both GEMM
            // directions (gate/up: n=I k=H; down: n=H k=I) → both dims % 64.
            ensure!(
                hidden_dim.is_multiple_of(64) && moe_inter.is_multiple_of(64),
                "DeepGEMM BF16 MoE needs H and I aligned to 64, got H={hidden_dim} I={moe_inter}"
            );
            moe_inter
        } else {
            anyhow::bail!(
                "DeepGEMM MoE path requires grouped expert caches (load with --qwen35-deepgemm)"
            )
        };
        // Fail loud if the native DeepGEMM bridge is a build-time stub (sm_120
        // uses the CUTLASS collective instead).
        if !sm120 {
            moe::dsv4_deepgemm_native_preflight()?;
        }

        let use_masked = total_routes <= DEEPGEMM_MASKED_BAND;
        let rows = if use_masked {
            local_experts * DEEPGEMM_MASKED_BAND
        } else {
            deepgemm_contig_rows_cap(total_routes, local_experts, DEEPGEMM_CONTIG_ALIGN)
        };
        ensure!(
            rows.checked_mul(hidden_dim.max(moe_inter_for_abi))
                .is_some_and(|v| i32::try_from(v).is_ok()),
            "DeepGEMM MoE padded rows {rows} x max(H={hidden_dim}, I={moe_inter_for_abi}) exceeds the i32 kernel ABI"
        );

        // packed_route_slot MUST be -1-refilled every call even single-GPU: the
        // pack writes only the R real rows, and every PAD row has to read -1 so
        // the scatter skips its garbage GEMM output.
        let offsets_ptr = if use_masked {
            // Band bases `g * 128` — a pure function of the length.
            let band_table: Vec<i32> = (0..local_experts as i32)
                .map(|g| g * DEEPGEMM_MASKED_BAND as i32)
                .collect();
            cache_ptr(scratch.dg_band_offsets.upload_const(ctx, &band_table)?, ctx)
        } else {
            // 128-aligned segment starts; the total is never read back (`rows`
            // is the host cap).
            let aligned_offsets =
                cache_ptr(scratch.dg_aligned_offsets.get(ctx, local_experts)?, ctx);
            let scan_total = cache_ptr(scratch.scan_total.get(ctx, 1)?, ctx);
            // SAFETY: counts/offsets/total valid on ctx.stream for E_l groups.
            qwen_moe_profile(
                ctx,
                "qwen/dg/scan_offsets",
                rows,
                local_experts,
                0,
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                || unsafe {
                    moe::moe_exclusive_scan_aligned_i32(
                        counts,
                        aligned_offsets,
                        scan_total,
                        local_experts,
                        DEEPGEMM_CONTIG_ALIGN,
                        stream,
                    )
                },
            )?;
            aligned_offsets
        };
        let cursors = cache_ptr(scratch.cursors.get_zeroed(ctx, local_experts)?, ctx);
        let packed_route_slot = cache_ptr(scratch.packed_route_slot.neg1_filled(ctx, rows)?, ctx);
        let packed_weight = cache_ptr(scratch.packed_weight.get(ctx, rows)?, ctx);
        let packed_hidden = cache_ptr(&scratch.packed_hidden.get(ctx, hidden_dim, rows)?.data, ctx);
        // SAFETY: buffers valid on ctx.stream; the pack writes every route's
        // row at offsets[local] + cursor < rows (masked: count_g <= R <= 128
        // = band capacity by the dispatch threshold; contiguous: aligned
        // offsets + counts <= the rows cap by construction).
        qwen_moe_profile(
            ctx,
            "qwen/dg/pack_slots",
            rows,
            hidden_dim,
            topk,
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            || unsafe {
                moe::dsv4_pack_local_experts_with_slots(
                    cache_ptr(&normed.data, ctx),
                    route_indices,
                    route_weights,
                    offsets_ptr,
                    cursors,
                    packed_hidden,
                    packed_route_slot,
                    packed_weight,
                    num_tokens,
                    hidden_dim,
                    topk,
                    split.local_expert_start,
                    local_experts,
                    stream,
                )
            },
        )?;

        // Contiguous only: row → local-expert map (-1 pads). sm_120's CUTLASS
        // collective consumes offsets+counts directly and never reads it.
        let m_indices = if use_masked || sm120 {
            None
        } else {
            let m_indices = cache_ptr(scratch.dg_m_indices.neg1_filled(ctx, rows)?, ctx);
            // SAFETY: counts/offsets are per-group; m_indices has `rows` slots.
            qwen_moe_profile(
                ctx,
                "qwen/dg/fill_m_indices",
                rows,
                local_experts,
                0,
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                || unsafe {
                    moe::dsv4_fill_m_indices_from_counts(
                        counts,
                        offsets_ptr,
                        m_indices,
                        local_experts,
                        rows,
                        stream,
                    )
                },
            )?;
            Some(m_indices)
        };

        if let Some((w13, down_g)) = fp8_grouped {
            if qwen_moe_profile_enabled()
                && std::env::var("INFER_TP_RANK")
                    .map(|rank| rank == "0")
                    .unwrap_or(true)
            {
                eprintln!(
                    "[qwen-moe-profile] qwen/fp8/shape tokens={num_tokens} topk={topk} routes={total_routes} rows={rows} experts={local_experts} hidden={hidden_dim} intermediate={} masked={use_masked}",
                    w13.rows / 2
                );
            }
            ensure!(
                !use_masked,
                "Qwen FP8 DeepGEMM MoE is prefill-only and requires contiguous layout"
            );
            let moe_inter = w13.rows / 2;
            // The loader recorded whether SFB is N-contiguous (CUTLASS sm_120a)
            // or K-contiguous (Hopper DeepGEMM); dispatch on THAT, not a
            // re-derived SM, so the two can't disagree.
            let fp8_n_contiguous = w13.sfb_n_contiguous;
            ensure!(
                down_g.sfb_n_contiguous == fp8_n_contiguous,
                "FP8 MoE SFB layout mismatch: w13 n_contiguous={fp8_n_contiguous} down={}",
                down_g.sfb_n_contiguous
            );
            debug_assert_eq!(
                fp8_n_contiguous, sm120,
                "loader/executor disagree on sm_120 SFB layout"
            );
            let scale_stride_m = rows.div_ceil(4) * 4;
            let hidden_scale_cols = hidden_dim.div_ceil(128);
            let inter_scale_cols = moe_inter.div_ceil(128);
            let input_fp8 = scratch.dg_input_fp8.get(ctx, rows * hidden_dim)?;
            let input_scales = scratch
                .dg_input_scales
                .get(ctx, scale_stride_m * hidden_scale_cols)?;
            let w13_out = scratch.gate_out.get(ctx, 2 * moe_inter, rows)?;
            let act_fp8 = scratch.dg_act_fp8.get(ctx, rows * moe_inter)?;
            let act_scales = scratch
                .dg_act_scales
                .get(ctx, scale_stride_m * inter_scale_cols)?;
            let expert_out = scratch.expert_out.get(ctx, hidden_dim, rows)?;
            let active_experts = scratch.dg_active_experts.upload_const(ctx, &[0i32])?;
            let active_offsets = scratch.dg_active_offsets.upload_const(ctx, &[0i32])?;
            let rows_i32 = i32::try_from(rows)
                .map_err(|_| anyhow::anyhow!("FP8 DeepGEMM MoE rows overflow i32"))?;
            let active_counts = scratch.dg_active_counts.upload(ctx, &[rows_i32])?;
            let stream = ctx.stream.cu_stream();
            // The w13 + down GEMMs share group geometry, so D2H offsets/counts
            // ONCE here (one sync/layer). Hopper DeepGEMM consumes device
            // m_indices, so it skips the readback.
            let (host_offsets, host_counts) = if fp8_n_contiguous {
                moe::dtoh_i32_pair(ctx, offsets_ptr, counts, local_experts)?
            } else {
                (Vec::new(), Vec::new())
            };
            // One dispatch for both grouped FP8 GEMMs, differing only in
            // operands + (n,k).
            let run_grouped_gemm = |a: RawDevicePtr<u8>,
                                    sfa: RawDevicePtr<f32>,
                                    b: RawDevicePtr<u8>,
                                    sfb: RawDevicePtr<f32>,
                                    d: RawDevicePtr<bf16>,
                                    n: usize,
                                    k: usize|
             -> Result<()> {
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    if fp8_n_contiguous {
                        moe::arle_fp8_moe_grouped_gemm_nt_sm120(
                            a,
                            sfa,
                            b,
                            sfb,
                            d,
                            &host_offsets,
                            &host_counts,
                            local_experts,
                            n,
                            k,
                            scale_stride_m,
                            stream,
                        )
                    } else {
                        moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                            a,
                            sfa,
                            b,
                            sfb,
                            d,
                            m_indices.expect("contiguous non-sm120 FP8 path fills m_indices"),
                            local_experts,
                            rows,
                            n,
                            k,
                            scale_stride_m,
                            DEEPGEMM_CONTIG_ALIGN,
                            stream,
                        )
                    }
                }
            };
            qwen_moe_profile(
                ctx,
                "qwen/fp8/pack_quantize_hidden",
                rows,
                hidden_dim,
                0,
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                || unsafe {
                    moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                        packed_hidden,
                        cache_ptr(input_fp8, ctx),
                        cache_ptr(input_scales, ctx),
                        cache_ptr(active_experts, ctx),
                        cache_ptr(active_offsets, ctx),
                        cache_ptr(active_counts, ctx),
                        1,
                        rows,
                        hidden_dim,
                        scale_stride_m,
                        stream,
                    )
                },
            )?;
            qwen_moe_profile(
                ctx,
                "qwen/fp8/gemm_w13",
                rows,
                2 * moe_inter,
                hidden_dim,
                || {
                    run_grouped_gemm(
                        cache_ptr(input_fp8, ctx),
                        cache_ptr(input_scales, ctx),
                        cache_ptr(&w13.weight, ctx),
                        cache_ptr(&w13.scales, ctx),
                        cache_ptr(&w13_out.data, ctx),
                        2 * moe_inter,
                        hidden_dim,
                    )
                },
            )?;
            qwen_moe_profile(
                ctx,
                "qwen/fp8/swiglu_quantize",
                rows,
                moe_inter,
                0,
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                || unsafe {
                    moe::dsv4_deepgemm_swiglu_quantize_w13(
                        cache_ptr(&w13_out.data, ctx),
                        cache_ptr(act_fp8, ctx),
                        cache_ptr(act_scales, ctx),
                        cache_ptr(active_experts, ctx),
                        cache_ptr(active_counts, ctx),
                        1,
                        rows,
                        moe_inter,
                        scale_stride_m,
                        f32::INFINITY,
                        stream,
                    )
                },
            )?;
            qwen_moe_profile(
                ctx,
                "qwen/fp8/gemm_down",
                rows,
                hidden_dim,
                moe_inter,
                || {
                    run_grouped_gemm(
                        cache_ptr(act_fp8, ctx),
                        cache_ptr(act_scales, ctx),
                        cache_ptr(&down_g.weight, ctx),
                        cache_ptr(&down_g.scales, ctx),
                        cache_ptr(&expert_out.data, ctx),
                        hidden_dim,
                        moe_inter,
                    )
                },
            )?;

            let route_out = if split.ep_size == 1 {
                scratch.route_out.get(ctx, hidden_dim, total_routes)?
            } else {
                scratch
                    .route_out
                    .get_zeroed(ctx, hidden_dim, total_routes)?
            };
            qwen_moe_profile(
                ctx,
                "qwen/fp8/scatter_combine",
                rows,
                hidden_dim,
                topk,
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                || unsafe {
                    moe::dsv4_scatter_all_route_slots(
                        cache_ptr(&expert_out.data, ctx),
                        cache_ptr(&route_out.data, ctx),
                        packed_route_slot,
                        packed_weight,
                        rows,
                        hidden_dim,
                        stream,
                    )?;
                    moe::dsv4_combine_route_slot_outputs(
                        cache_ptr(&route_out.data, ctx),
                        cache_ptr(&out.data, ctx),
                        num_tokens,
                        topk,
                        hidden_dim,
                        stream,
                    )
                },
            )?;
            return Ok(());
        }

        let (gate_g, up_g, down_g) = bf16_grouped.ok_or_else(|| {
            anyhow::anyhow!("DeepGEMM BF16 MoE path requires gate/up/down grouped caches")
        })?;
        let moe_inter = gate_g.rows;

        // Heuristics-only hint: expected valid rows per group.
        let expected_m = total_routes.div_ceil(local_experts).max(1);
        let gate_out = scratch.gate_out.get(ctx, moe_inter, rows)?;
        let up_out = scratch.up_out.get(ctx, moe_inter, rows)?;
        // SAFETY: A `[rows, H]`, B `[G, I, H]`, D `[rows, I]` row-major BF16
        // on ctx.stream; masked_m = counts (read-only); m_indices contract
        // holds by the aligned scan (group segments start 128-aligned).
        for (label, grouped_b, d) in [
            ("qwen/bf16/gemm_gate", gate_g, &*gate_out),
            ("qwen/bf16/gemm_up", up_g, &*up_out),
        ] {
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            qwen_moe_profile(ctx, label, rows, moe_inter, hidden_dim, || unsafe {
                if use_masked {
                    moe::deepgemm_m_grouped_bf16_gemm_nt_masked(
                        packed_hidden,
                        cache_ptr(&grouped_b.data, ctx),
                        cache_ptr(&d.data, ctx),
                        counts,
                        local_experts,
                        DEEPGEMM_MASKED_BAND,
                        moe_inter,
                        hidden_dim,
                        expected_m,
                        stream,
                    )
                } else {
                    moe::deepgemm_m_grouped_bf16_gemm_nt_contiguous(
                        packed_hidden,
                        cache_ptr(&grouped_b.data, ctx),
                        cache_ptr(&d.data, ctx),
                        m_indices.expect("contiguous path fills m_indices"),
                        local_experts,
                        rows,
                        moe_inter,
                        hidden_dim,
                        stream,
                    )
                }
            })?;
        }
        // SwiGLU over the full padded buffers (UNCLAMPED — Qwen3.6 has no clamp).
        let act = scratch.act.get(ctx, moe_inter, rows)?;
        qwen_moe_profile(ctx, "qwen/bf16/silu_mul", rows, moe_inter, 0, || {
            silu_mul(ctx, gate_out, up_out, act)
        })?;
        let expert_out = scratch.expert_out.get(ctx, hidden_dim, rows)?;
        // SAFETY: A `[rows, I]`, B `[G, H, I]`, D `[rows, H]`; same contracts
        // as the gate/up GEMMs above.
        qwen_moe_profile(
            ctx,
            "qwen/bf16/gemm_down",
            rows,
            hidden_dim,
            moe_inter,
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            || unsafe {
                if use_masked {
                    moe::deepgemm_m_grouped_bf16_gemm_nt_masked(
                        cache_ptr(&act.data, ctx),
                        cache_ptr(&down_g.data, ctx),
                        cache_ptr(&expert_out.data, ctx),
                        counts,
                        local_experts,
                        DEEPGEMM_MASKED_BAND,
                        hidden_dim,
                        moe_inter,
                        expected_m,
                        stream,
                    )
                } else {
                    moe::deepgemm_m_grouped_bf16_gemm_nt_contiguous(
                        cache_ptr(&act.data, ctx),
                        cache_ptr(&down_g.data, ctx),
                        cache_ptr(&expert_out.data, ctx),
                        m_indices.expect("contiguous path fills m_indices"),
                        local_experts,
                        rows,
                        hidden_dim,
                        moe_inter,
                        stream,
                    )
                }
            },
        )?;

        // The scatter walks ALL `rows` padded rows and skips route_slot < 0.
        // Under EP the unwritten non-local slots MUST read zero in the combine
        // (partial-sum contract), so EP re-zeros.
        let route_out = if split.ep_size == 1 {
            scratch.route_out.get(ctx, hidden_dim, total_routes)?
        } else {
            scratch
                .route_out
                .get_zeroed(ctx, hidden_dim, total_routes)?
        };
        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        qwen_moe_profile(
            ctx,
            "qwen/bf16/scatter_combine",
            rows,
            hidden_dim,
            topk,
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            || unsafe {
                moe::dsv4_scatter_all_route_slots(
                    cache_ptr(&expert_out.data, ctx),
                    cache_ptr(&route_out.data, ctx),
                    packed_route_slot,
                    packed_weight,
                    rows,
                    hidden_dim,
                    stream,
                )?;
                moe::dsv4_combine_route_slot_outputs(
                    cache_ptr(&route_out.data, ctx),
                    cache_ptr(&out.data, ctx),
                    num_tokens,
                    topk,
                    hidden_dim,
                    stream,
                )
            },
        )?;
        Ok(())
    }

    /// Dense shared-expert SwiGLU into `out_slot`. Takes the four slots
    /// individually so the caller's other scratch fields stay borrowable while
    /// the returned reference (tied to `out_slot`) is alive.
    fn shared_expert_forward<'a>(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        normed: &HiddenStates,
        gate_slot: &mut HiddenSlot,
        up_slot: &mut HiddenSlot,
        act_slot: &mut HiddenSlot,
        out_slot: &'a mut HiddenSlot,
    ) -> Result<&'a HiddenStates> {
        let shared_inter = weights.shared_gate.rows;
        let hidden_dim = normed.hidden_dim;
        let gate = gate_slot.get(ctx, shared_inter, normed.seq_len)?;
        let up = up_slot.get(ctx, shared_inter, normed.seq_len)?;
        gemm_batch(ctx, &weights.shared_gate, normed, gate)?;
        gemm_batch(ctx, &weights.shared_up, normed, up)?;
        let act = act_slot.get(ctx, shared_inter, normed.seq_len)?;
        silu_mul(ctx, gate, up, act)?;
        let out = out_slot.get(ctx, hidden_dim, normed.seq_len)?;
        gemm_batch(ctx, &weights.shared_down, act, out)?;
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

    fn fp8_grouped_shape(weight: &DeviceMatrix) -> Result<(usize, usize, usize, usize)> {
        match weight.weight_format() {
            WeightFormat::Fp8BlockScaled => {
                ensure!(
                    weight.quant_scale_rows > 0
                        && weight.quant_scale_cols > 0
                        && weight.quant_block_m > 0
                        && weight.quant_block_k > 0,
                    "FP8 block-scaled MoE weight missing scale metadata"
                );
                Ok((
                    weight.quant_scale_rows,
                    weight.quant_scale_cols,
                    weight.quant_block_m,
                    weight.quant_block_k,
                ))
            }
            WeightFormat::Fp8PerShard => {
                ensure!(
                    weight.quant_scale_rows == 1 && weight.quant_scale_cols == 1,
                    "FP8 per-shard MoE currently requires scalar scale, got {}x{}",
                    weight.quant_scale_rows,
                    weight.quant_scale_cols
                );
                Ok((1, 1, weight.rows, weight.cols))
            }
            other => anyhow::bail!("expected FP8 MoE weight, got {other}"),
        }
    }

    fn fp8_signature_shape(
        sig: Option<ExpertQuantDispatchSignature>,
        label: &str,
    ) -> Result<(usize, usize, usize, usize)> {
        let sig =
            sig.ok_or_else(|| anyhow::anyhow!("FP8 MoE {label} missing dispatch signature"))?;
        ensure!(
            sig.quant_scale_rows > 0
                && sig.quant_scale_cols > 0
                && sig.quant_block_m > 0
                && sig.quant_block_k > 0,
            "FP8 MoE {label} signature missing scale metadata: {sig:?}"
        );
        Ok((
            sig.quant_scale_rows,
            sig.quant_scale_cols,
            sig.quant_block_m,
            sig.quant_block_k,
        ))
    }

    fn opt_ptrs<'a>(
        ptrs: &'a Option<cudarc::driver::CudaSlice<u64>>,
        label: &str,
    ) -> Result<&'a cudarc::driver::CudaSlice<u64>> {
        ptrs.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.6 MoE missing {label} pointer table"))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn grouped_pair_batch(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        packed_hidden: &HiddenStates,
        gate_out: &mut HiddenStates,
        up_out: &mut HiddenStates,
        offsets: &cudarc::driver::CudaSlice<i32>,
        counts: RawDevicePtr<i32>,
        expert_indices: &cudarc::driver::CudaSlice<i32>,
        local_experts: usize,
        max_count: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match weights.expert_weight_format {
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            WeightFormat::DenseBf16 => unsafe {
                moe::moe_bf16_grouped_gemm_pair_batch(
                    &weights.gate_ptrs,
                    &weights.up_ptrs,
                    cache_ptr(&packed_hidden.data, ctx),
                    cache_ptr(&gate_out.data, ctx),
                    cache_ptr(&up_out.data, ctx),
                    cache_ptr(offsets, ctx),
                    counts,
                    cache_ptr(expert_indices, ctx),
                    local_experts,
                    max_count,
                    n,
                    k,
                    ctx,
                    ctx.stream.cu_stream(),
                )
            },
            WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard => {
                let (scale_rows, scale_cols, block_m, block_k) =
                    if let Some(first) = weights.gate.first() {
                        fp8_grouped_shape(first)?
                    } else {
                        fp8_signature_shape(weights.gate_up_quant_signature, "gate/up")?
                    };
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    moe::moe_fp8_block_scaled_grouped_gemv_pair_batch(
                        &weights.gate_ptrs,
                        opt_ptrs(&weights.gate_scale_ptrs, "gate FP8 scale")?,
                        &weights.up_ptrs,
                        opt_ptrs(&weights.up_scale_ptrs, "up FP8 scale")?,
                        cache_ptr(&packed_hidden.data, ctx),
                        cache_ptr(&gate_out.data, ctx),
                        cache_ptr(&up_out.data, ctx),
                        cache_ptr(offsets, ctx),
                        counts,
                        cache_ptr(expert_indices, ctx),
                        local_experts,
                        max_count,
                        n,
                        k,
                        scale_rows,
                        scale_cols,
                        block_m,
                        block_k,
                        ctx,
                        ctx.stream.cu_stream(),
                    )
                }
            }
            WeightFormat::Fp4E2M1Group => {
                let first = weights
                    .gate
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("FP4 MoE pair batch has no gate experts"))?;
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    moe::moe_fp4_e2m1_grouped_gemv_pair_batch(
                        &weights.gate_ptrs,
                        opt_ptrs(&weights.gate_scale_ptrs, "gate FP4 scale")?,
                        opt_ptrs(&weights.gate_global_ptrs, "gate FP4 global scale")?,
                        &weights.up_ptrs,
                        opt_ptrs(&weights.up_scale_ptrs, "up FP4 scale")?,
                        opt_ptrs(&weights.up_global_ptrs, "up FP4 global scale")?,
                        cache_ptr(&packed_hidden.data, ctx),
                        cache_ptr(&gate_out.data, ctx),
                        cache_ptr(&up_out.data, ctx),
                        cache_ptr(offsets, ctx),
                        counts,
                        cache_ptr(expert_indices, ctx),
                        local_experts,
                        max_count,
                        n,
                        k,
                        first.group_size,
                        first.quant_scale_cols,
                        ctx,
                        ctx.stream.cu_stream(),
                    )
                }
            }
            WeightFormat::W4A16 => {
                let first = weights
                    .gate
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("W4A16 MoE pair batch has no gate experts"))?;
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    moe::moe_w4a16_grouped_gemv_pair_batch(
                        &weights.gate_ptrs,
                        opt_ptrs(&weights.gate_scale_ptrs, "gate W4A16 scale")?,
                        &weights.up_ptrs,
                        opt_ptrs(&weights.up_scale_ptrs, "up W4A16 scale")?,
                        cache_ptr(&packed_hidden.data, ctx),
                        cache_ptr(&gate_out.data, ctx),
                        cache_ptr(&up_out.data, ctx),
                        cache_ptr(offsets, ctx),
                        counts,
                        cache_ptr(expert_indices, ctx),
                        local_experts,
                        max_count,
                        n,
                        k,
                        first.group_size,
                        ctx,
                        ctx.stream.cu_stream(),
                    )
                }
            }
            other => anyhow::bail!("unsupported Qwen3.6 MoE pair format {other}"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn grouped_down_batch(
        ctx: &DeviceContext,
        weights: &MoeLayerWeights,
        act: &HiddenStates,
        expert_out: &mut HiddenStates,
        offsets: &cudarc::driver::CudaSlice<i32>,
        counts: RawDevicePtr<i32>,
        expert_indices: &cudarc::driver::CudaSlice<i32>,
        local_experts: usize,
        max_count: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match weights.expert_weight_format {
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            WeightFormat::DenseBf16 => unsafe {
                moe::moe_bf16_grouped_gemm_batch(
                    &weights.down_ptrs,
                    cache_ptr(&act.data, ctx),
                    cache_ptr(&expert_out.data, ctx),
                    cache_ptr(offsets, ctx),
                    counts,
                    cache_ptr(expert_indices, ctx),
                    local_experts,
                    max_count,
                    n,
                    k,
                    ctx,
                    ctx.stream.cu_stream(),
                )
            },
            WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard => {
                let (scale_rows, scale_cols, block_m, block_k) =
                    if let Some(first) = weights.down.first() {
                        fp8_grouped_shape(first)?
                    } else {
                        fp8_signature_shape(weights.down_quant_signature, "down")?
                    };
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    moe::moe_fp8_block_scaled_grouped_gemv_batch(
                        &weights.down_ptrs,
                        opt_ptrs(&weights.down_scale_ptrs, "down FP8 scale")?,
                        cache_ptr(&act.data, ctx),
                        cache_ptr(&expert_out.data, ctx),
                        cache_ptr(offsets, ctx),
                        counts,
                        cache_ptr(expert_indices, ctx),
                        local_experts,
                        max_count,
                        n,
                        k,
                        scale_rows,
                        scale_cols,
                        block_m,
                        block_k,
                        ctx,
                        ctx.stream.cu_stream(),
                    )
                }
            }
            WeightFormat::Fp4E2M1Group => {
                let first = weights
                    .down
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("FP4 MoE down batch has no down experts"))?;
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    moe::moe_fp4_e2m1_grouped_gemv_batch(
                        &weights.down_ptrs,
                        opt_ptrs(&weights.down_scale_ptrs, "down FP4 scale")?,
                        opt_ptrs(&weights.down_global_ptrs, "down FP4 global scale")?,
                        cache_ptr(&act.data, ctx),
                        cache_ptr(&expert_out.data, ctx),
                        cache_ptr(offsets, ctx),
                        counts,
                        cache_ptr(expert_indices, ctx),
                        local_experts,
                        max_count,
                        n,
                        k,
                        first.group_size,
                        first.quant_scale_cols,
                        ctx,
                        ctx.stream.cu_stream(),
                    )
                }
            }
            WeightFormat::W4A16 => {
                let first = weights
                    .down
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("W4A16 MoE down batch has no down experts"))?;
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    moe::moe_w4a16_grouped_gemv_batch(
                        &weights.down_ptrs,
                        opt_ptrs(&weights.down_scale_ptrs, "down W4A16 scale")?,
                        cache_ptr(&act.data, ctx),
                        cache_ptr(&expert_out.data, ctx),
                        cache_ptr(offsets, ctx),
                        counts,
                        cache_ptr(expert_indices, ctx),
                        local_experts,
                        max_count,
                        n,
                        k,
                        first.group_size,
                        ctx,
                        ctx.stream.cu_stream(),
                    )
                }
            }
            other => anyhow::bail!("unsupported Qwen3.6 MoE down format {other}"),
        }
    }
}

// `dead_code` marks pending-consumer infra (the `model.rs` DSv4 branch), not cruft.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod dsv4_gpu {
    use anyhow::{Result, ensure};
    use cuda_kernels::moe;
    use cuda_kernels::prelude::{DeviceContext, HiddenStates};
    use cuda_kernels::tensor::{Dsv4Fp8DeepGemmWeightCache, cache_ptr};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
    use half::bf16;
    use std::sync::Arc;

    use super::{
        DEEPGEMM_CONTIG_ALIGN, DSV4_DECODE_CONTIG_ALIGN, DSV4_DECODE_CONTIG_MAX_ROUTES,
        alloc_neg1_i32, deepgemm_contig_rows_cap,
    };
    use crate::dsv4::{Dsv4ForwardKeepalive, Dsv4Model, Dsv4MoeLayer};
    use crate::ops::gemm_batch;

    struct DeviceRouting {
        indices: CudaSlice<i32>,
        weights: CudaSlice<f32>,
    }

    pub(crate) struct Dsv4SharedDecodeScratch {
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

    /// Model-wide reusable scratch for the compact FP8 decode-band MoE tail,
    /// sized to `DSV4_DECODE_CONTIG_MAX_ROUTES`. Layers run sequentially, serial
    /// on `ctx.stream`, so one instance serves every layer with no aliasing. Six
    /// buffers are pure-output; the other four need per-step re-init.
    pub(crate) struct Dsv4MoeTailScratch {
        max_rows: usize,
        experts_per_rank: usize,
        counts: CudaSlice<i32>,
        offsets: CudaSlice<i32>,
        scan_total: CudaSlice<i32>,
        cursors: CudaSlice<i32>,
        packed_hidden: HiddenStates,
        packed_route_slot: CudaSlice<i32>,
        packed_weight: CudaSlice<f32>,
        act: HiddenStates,
        expert_out: HiddenStates,
        route_out: HiddenStates,
    }

    impl Dsv4MoeTailScratch {
        pub(crate) fn new(
            ctx: &DeviceContext,
            hidden_dim: usize,
            intermediate: usize,
            experts_per_rank: usize,
        ) -> Result<Self> {
            let max_rows = DSV4_DECODE_CONTIG_MAX_ROUTES;
            Ok(Self {
                max_rows,
                experts_per_rank,
                counts: ctx.stream.alloc_zeros::<i32>(experts_per_rank)?,
                offsets: ctx.stream.alloc_zeros::<i32>(experts_per_rank)?,
                scan_total: ctx.stream.alloc_zeros::<i32>(1)?,
                cursors: ctx.stream.alloc_zeros::<i32>(experts_per_rank)?,
                packed_hidden: HiddenStates::zeros(ctx, hidden_dim, max_rows)?,
                packed_route_slot: alloc_neg1_i32(ctx, max_rows)?,
                packed_weight: ctx.stream.alloc_zeros::<f32>(max_rows)?,
                act: HiddenStates::zeros(ctx, intermediate, max_rows)?,
                expert_out: HiddenStates::zeros(ctx, hidden_dim, max_rows)?,
                route_out: HiddenStates::zeros(ctx, hidden_dim, max_rows)?,
            })
        }

        pub(crate) fn device_bytes(
            hidden_dim: usize,
            intermediate: usize,
            experts_per_rank: usize,
        ) -> usize {
            let max_rows = DSV4_DECODE_CONTIG_MAX_ROUTES;
            let i32b = |n: usize| n * std::mem::size_of::<i32>();
            let f32b = |n: usize| n * std::mem::size_of::<f32>();
            let bf16 = |h: usize, s: usize| h * s * std::mem::size_of::<half::bf16>();
            i32b(experts_per_rank) * 3
                + i32b(1)
                + bf16(hidden_dim, max_rows) * 3
                + i32b(max_rows)
                + f32b(max_rows)
                + bf16(intermediate, max_rows)
        }

        /// `counts`/`cursors`/`route_out` → 0, `packed_route_slot` → -1.
        fn reinit(&mut self, ctx: &DeviceContext, rows: usize) -> Result<()> {
            use cudarc::driver::DevicePtrMut;
            ensure!(
                rows <= self.max_rows,
                "DSv4 MoE tail scratch rows {rows} > max_rows {}",
                self.max_rows
            );
            let zero_i32 = |buf: &mut CudaSlice<i32>, n: usize| -> Result<()> {
                let (ptr, _g) = buf.device_ptr_mut(&ctx.stream);
                // SAFETY: live device alloc of >= n i32 on this stream.
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        ptr,
                        0x00,
                        n * std::mem::size_of::<i32>(),
                        ctx.stream.cu_stream(),
                    )?;
                }
                Ok(())
            };
            zero_i32(&mut self.counts, self.experts_per_rank)?;
            zero_i32(&mut self.cursors, self.experts_per_rank)?;
            {
                let hidden_dim = self.route_out.hidden_dim;
                let (ptr, _g) = self.route_out.data.device_ptr_mut(&ctx.stream);
                // SAFETY: live device alloc of hidden_dim*max_rows bf16 on this stream.
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        ptr,
                        0x00,
                        hidden_dim * rows * std::mem::size_of::<half::bf16>(),
                        ctx.stream.cu_stream(),
                    )?;
                }
            }
            // packed_route_slot → -1 (0xFF bytes).
            {
                let (ptr, _g) = self.packed_route_slot.device_ptr_mut(&ctx.stream);
                // SAFETY: live device alloc of >= rows i32 on this stream.
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        ptr,
                        0xFF,
                        rows * std::mem::size_of::<i32>(),
                        ctx.stream.cu_stream(),
                    )?;
                }
            }
            Ok(())
        }
    }

    impl Dsv4SharedDecodeScratch {
        pub(crate) fn new(
            ctx: &DeviceContext,
            hidden_dim: usize,
            shared_inter: usize,
        ) -> Result<Self> {
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

        pub(crate) fn device_bytes(hidden_dim: usize, shared_inter: usize) -> usize {
            let max_m = 128usize;
            let scale_stride_m = max_m.div_ceil(4) * 4;
            let hidden_scale_cols = hidden_dim.div_ceil(128);
            let inter_scale_cols = shared_inter.div_ceil(128);
            let b = |elems: usize, elem_bytes: usize| elems.saturating_mul(elem_bytes);
            let bf16 = |hidden_dim: usize, seq_len: usize| {
                hidden_dim.saturating_mul(seq_len).saturating_mul(2)
            };
            b(max_m.saturating_mul(hidden_dim), 1)
                .saturating_add(b(
                    scale_stride_m.saturating_mul(hidden_scale_cols),
                    std::mem::size_of::<f32>(),
                ))
                .saturating_add(bf16(2usize.saturating_mul(shared_inter), max_m))
                .saturating_add(b(max_m.saturating_mul(shared_inter), 1))
                .saturating_add(b(
                    scale_stride_m.saturating_mul(inter_scale_cols),
                    std::mem::size_of::<f32>(),
                ))
                .saturating_add(bf16(hidden_dim, max_m))
                .saturating_add(4usize.saturating_mul(std::mem::size_of::<i32>()))
        }

        #[allow(dead_code)]
        pub(crate) fn device_bytes_live(&self) -> usize {
            let f32_sz = std::mem::size_of::<f32>();
            let i32_sz = std::mem::size_of::<i32>();
            self.input_fp8.len() // u8
                + self.input_scales.len() * f32_sz
                + self.w13_out.device_bytes()
                + self.act_fp8.len() // u8
                + self.act_scales.len() * f32_sz
                + self.out.device_bytes()
                + self.active_experts.len() * i32_sz
                + self.active_offsets.len() * i32_sz
                + self.counts.len() * i32_sz
                + self.masked_m.len() * i32_sz
        }
    }

    fn dsv4_route_device(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        logits: &HiddenStates,
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
            keepalive.keep_u32(&token_ids);
            Some(token_ids)
        } else {
            None
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

        keepalive.keep_hidden(logits);
        keepalive.keep_i32(&route_indices);
        keepalive.keep_f32(&route_weights);
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

    /// FP8 DeepGEMM MoE forward for one DSv4 routed-MoE layer (this EP rank's
    /// experts only).
    ///
    /// `tokens` are the input token ids, needed by hash-routed layers, which pick
    /// experts from the `tid2eid` table by token id instead of a router gate.
    /// `out` receives routed experts only: callers all-reduce this EP-sharded
    /// output, then add the replicated shared expert exactly once per rank via
    /// [`dsv4_shared_expert_forward`].
    pub(crate) fn dsv4_moe_forward(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        keepalive: &mut Dsv4ForwardKeepalive,
        tail: Option<&mut Dsv4MoeTailScratch>,
        _mega_epoch: Option<u64>,
    ) -> Result<bool> {
        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let num_tokens = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let experts_per_rank = model.split.experts_per_rank;

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
        // Fail loud if the native DeepGEMM bridge is a build-time stub.
        moe::dsv4_deepgemm_native_preflight()?;

        let (route_indices, route_weights) =
            crate::stage_profile::profile(ctx, "dsv4/stage/moe_route", || -> Result<_> {
                crate::profile::profile_op(ctx, "moe_route", None, num_tokens, || {
                    // SAFETY: router gemm writes the full logits buffer.
                    let mut logits =
                        unsafe { HiddenStates::uninit(ctx, cfg.num_experts, num_tokens)? };
                    gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
                    let routing = dsv4_route_device(model, layer, tokens, &logits, keepalive)?;
                    Ok((routing.indices, routing.weights))
                })
            })?;

        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(mega_moe) = &model.mega_moe {
            mega_moe.assert_forward_epoch(
                _mega_epoch.ok_or_else(|| anyhow::anyhow!("DSv4 MegaMoE forward epoch missing"))?,
                num_tokens,
            )?;
            let world = model.tp.config().world_size;
            let per_rank = num_tokens.div_ceil(world);
            let start = (model.tp.config().rank * per_rank).min(num_tokens);
            let owned_n = (((model.tp.config().rank + 1) * per_rank).min(num_tokens)) - start;
            let launch_n = owned_n.max(1);
            let local_workspace = mega_moe.workspace.local_base();
            crate::stage_profile::profile(ctx, "dsv4/stage/mega_moe_input", || {
                // SAFETY: sources are live contiguous step buffers; the model-owned
                // symmetric workspace spans `layout.num_bytes` through this launch.
                unsafe {
                    moe::sm90_mega_moe_stage_inputs(
                        cache_ptr(&hidden.data, ctx).offset_elems(start * hidden_dim),
                        cache_ptr(&route_indices, ctx).offset_elems(start * cfg.top_k),
                        cache_ptr(&route_weights, ctx).offset_elems(start * cfg.top_k),
                        local_workspace,
                        &mega_moe.layout,
                        owned_n,
                        cfg.top_k,
                        hidden_dim,
                        ctx.stream.cu_stream(),
                    )
                }
            })?;
            crate::stage_profile::profile(ctx, "dsv4/stage/mega_moe", || {
                // SAFETY: boot validates shape/workspace ownership; staging above
                // fully writes this step's four input regions on the same stream.
                unsafe {
                    moe::sm90_mega_moe_launch(&moe::Sm90MegaMoeLaunch {
                        shape: mega_moe.shape,
                        workspace: &mega_moe.layout,
                        num_tokens: launch_n,
                        rank_idx: model.tp.config().rank,
                        y: cache_ptr(&mega_moe.owned_out, ctx),
                        cumulative_local_expert_recv_stats: None,
                        peer_buffer_ptrs: mega_moe.workspace.peer_ptrs(),
                        local_workspace,
                        activation_clamp: model.config.swiglu_limit,
                        fast_math: mega_moe.fast_math,
                        enable_pdl: mega_moe.enable_pdl,
                        l1_weights: cache_ptr(&layer.w13_grouped.weight, ctx),
                        l1_weight_stride: hidden_dim,
                        l1_weights_sf: cache_ptr(&layer.w13_grouped.scales, ctx),
                        l2_weights: cache_ptr(&layer.w2_grouped.weight, ctx),
                        l2_weight_stride: layer.intermediate,
                        l2_weights_sf: cache_ptr(&layer.w2_grouped.scales, ctx),
                        stream: ctx.stream.cu_stream(),
                    })
                }
            })?;
            ctx.stream
                .memset_zeros(&mut out.data)
                .map_err(|error| anyhow::anyhow!("MegaMoE output zero failed: {error}"))?;
            if owned_n > 0 {
                ctx.stream
                    .memcpy_dtod(
                        &mega_moe.owned_out.slice(0..owned_n * hidden_dim),
                        &mut out
                            .data
                            .slice_mut(start * hidden_dim..(start + owned_n) * hidden_dim),
                    )
                    .map_err(|error| {
                        anyhow::anyhow!("MegaMoE owned output copy failed: {error}")
                    })?;
            }
            model.tp.all_reduce_sum(ctx, out)?;
            return Ok(false);
        }

        dsv4_moe_forward_masked_tail(
            model,
            layer,
            &route_indices,
            &route_weights,
            hidden,
            out,
            keepalive,
            tail,
        )?;
        Ok(true)
    }

    /// Routed-row ceiling for the compact FP8 decode lane: the kernels operate on
    /// real routed rows only, so the 64-aligned contiguous band's ceiling keeps
    /// B<=~16 off padded DeepGEMM materialization.
    const DSV4_DECODE_GEMV_MAX_ROUTES: usize = DSV4_DECODE_CONTIG_MAX_ROUTES;

    /// Per-expert pointer tables over the layer's f32 block-scale buffers for the
    /// decode-band grouped-GEMM MoE lane; the scale pointers index directly into
    /// the existing DeepGEMM scale buffers (no re-encoding).
    pub(crate) struct Dsv4GemvTables {
        gate_w: CudaSlice<u64>,
        gate_s: CudaSlice<u64>,
        up_w: CudaSlice<u64>,
        up_s: CudaSlice<u64>,
        w2_w: CudaSlice<u64>,
        w2_s: CudaSlice<u64>,
        /// w13-half scale columns (= H/128), the swiglu kernel's `scale_cols`.
        sc13: usize,
        /// w2 scale columns (= I/128), the down kernel's `scale_cols`.
        sc2: usize,
    }

    fn build_gemv_tables(ctx: &DeviceContext, layer: &Dsv4MoeLayer) -> Result<Dsv4GemvTables> {
        let g = layer.num_groups;
        let h = layer.hidden_dim;
        let i_dim = layer.intermediate;
        let w13 = &layer.w13_grouped;
        let w2 = &layer.w2_grouped;
        ensure!(
            w13.groups == g && w13.rows == 2 * i_dim && w13.cols == h,
            "GEMV tables: w13 cache {}x{} g={} != [2I={}, H={h}]",
            w13.rows,
            w13.cols,
            w13.groups,
            2 * i_dim
        );
        ensure!(
            w2.groups == g && w2.rows == h && w2.cols == i_dim,
            "GEMV tables: w2 cache {}x{} g={} != [H={h}, I={i_dim}]",
            w2.rows,
            w2.cols,
            w2.groups
        );
        ensure!(
            i_dim.is_multiple_of(128) && h.is_multiple_of(128),
            "GEMV tables need 128-aligned dims, got I={i_dim} H={h}"
        );
        let sr_full = (2 * i_dim) / 128;
        let sr_half = i_dim / 128;
        let sc13 = h / 128;
        let sr2 = h / 128;
        let sc2 = i_dim / 128;
        let stride13 = sr_full * sc13;
        let stride2 = sr2 * sc2;
        ensure!(
            w13.scales.len() == g * stride13 && w2.scales.len() == g * stride2,
            "GEMV tables: scale buffer {} / {} != G×stride {} / {}",
            w13.scales.len(),
            w2.scales.len(),
            g * stride13,
            g * stride2
        );

        // The MoE expert caches store f32 block scales, not UE8M0 (that encoding
        // is attention-side only). Offsets below are in BYTES (f32 ⇒ ×4).
        let (w13_base, w2_base, s13_base, s2_base) = {
            let (a, _g13) = w13.weight.device_ptr(&ctx.stream);
            let (b, _g2) = w2.weight.device_ptr(&ctx.stream);
            let (c, _gs13) = w13.scales.device_ptr(&ctx.stream);
            let (d, _gs2) = w2.scales.device_ptr(&ctx.stream);
            (a, b, c, d)
        };
        let wstride13 = (2 * i_dim * h) as u64;
        let half_off = (i_dim * h) as u64;
        let mut gate_w = Vec::with_capacity(g);
        let mut up_w = Vec::with_capacity(g);
        let mut gate_s = Vec::with_capacity(g);
        let mut up_s = Vec::with_capacity(g);
        let mut w2_w = Vec::with_capacity(g);
        let mut w2_s = Vec::with_capacity(g);
        for e in 0..g {
            let wb = w13_base + e as u64 * wstride13;
            gate_w.push(wb);
            up_w.push(wb + half_off);
            let sb = s13_base + (e * stride13 * 4) as u64;
            gate_s.push(sb);
            up_s.push(sb + (sr_half * sc13 * 4) as u64);
            w2_w.push(w2_base + (e * h * i_dim) as u64);
            w2_s.push(s2_base + (e * stride2 * 4) as u64);
        }
        let h2d = |v: &[u64]| -> Result<CudaSlice<u64>> {
            ctx.stream
                .clone_htod(v)
                .map_err(|e| anyhow::anyhow!("GEMV tables: pointer-table H2D failed: {e}"))
        };
        Ok(Dsv4GemvTables {
            gate_w: h2d(&gate_w)?,
            gate_s: h2d(&gate_s)?,
            up_w: h2d(&up_w)?,
            up_s: h2d(&up_s)?,
            w2_w: h2d(&w2_w)?,
            w2_s: h2d(&w2_s)?,
            sc13,
            sc2,
        })
    }

    /// Decode-band routed-MoE forward via grouped w8a16 GEMM (warp-per-row):
    /// compact pack, one fused gate/up pass with clamped SwiGLU, one w2 pass,
    /// then the shared scatter/combine tail. Zero pad rows and zero
    /// activation-quantize work, which removes the grouped lane's padding tax.
    #[allow(clippy::too_many_arguments)]
    fn dsv4_moe_forward_decode_fp8(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        tables: &Dsv4GemvTables,
        route_indices: &CudaSlice<i32>,
        route_weights: &CudaSlice<f32>,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        _keepalive: &mut Dsv4ForwardKeepalive,
        tail: Option<&mut Dsv4MoeTailScratch>,
    ) -> Result<()> {
        let ctx = &model.ctx;
        let cfg = &model.moe_config;
        let split = &model.split;
        let num_tokens = hidden.seq_len;
        let hidden_dim = hidden.hidden_dim;
        let i_dim = layer.intermediate;
        let topk = cfg.top_k;
        let experts_per_rank = split.experts_per_rank;
        let local_start = split.local_expert_start;
        let total_routes = num_tokens * topk;
        let rows = total_routes.max(1);

        // The model-wide scratch is pre-allocated to the band ceiling, so no
        // per-layer alloc churn; the throwaway fallback is born zero/-1 and so
        // skips `reinit`.
        let mut owned_tail;
        let scratch: &mut Dsv4MoeTailScratch = match tail {
            Some(s) => {
                s.reinit(ctx, rows)?;
                s
            }
            None => {
                owned_tail = Dsv4MoeTailScratch::new(ctx, hidden_dim, i_dim, experts_per_rank)?;
                &mut owned_tail
            }
        };
        // Kernels take `rows` as the work bound; capacity is `max_rows >= rows`.
        let counts = &scratch.counts;
        let offsets = &scratch.offsets;
        let scan_total = &scratch.scan_total;
        let cursors = &scratch.cursors;
        let packed_hidden = &scratch.packed_hidden;
        let packed_route_slot = &scratch.packed_route_slot;
        let packed_weight = &scratch.packed_weight;
        let act = &scratch.act;
        let expert_out = &scratch.expert_out;
        let route_out = &scratch.route_out;

        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_count_local_experts(
                cache_ptr(route_indices, ctx),
                cache_ptr(counts, ctx),
                num_tokens,
                topk,
                local_start,
                experts_per_rank,
                ctx.stream.cu_stream(),
            )?;
            moe::dsv4_exclusive_scan_i32(
                cache_ptr(counts, ctx),
                cache_ptr(offsets, ctx),
                cache_ptr(scan_total, ctx),
                experts_per_rank,
                ctx.stream.cu_stream(),
            )?;
        }

        // SAFETY: buffers valid on ctx.stream; shapes checked by the kernel.
        unsafe {
            moe::dsv4_pack_local_experts_with_slots(
                cache_ptr(&hidden.data, ctx),
                cache_ptr(route_indices, ctx),
                cache_ptr(route_weights, ctx),
                cache_ptr(offsets, ctx),
                cache_ptr(cursors, ctx),
                cache_ptr(&packed_hidden.data, ctx),
                cache_ptr(packed_route_slot, ctx),
                cache_ptr(packed_weight, ctx),
                num_tokens,
                hidden_dim,
                topk,
                local_start,
                experts_per_rank,
                ctx.stream.cu_stream(),
            )?;
        }

        // max_count = total_routes is the safe host upper bound on per-expert row
        // count (kernels exit early on `chunk_base >= counts[e]`).
        // SAFETY: pointer tables hold experts_per_rank entries built over the
        // layer's grouped caches; packed rows are bounded by offsets+counts.
        unsafe {
            moe::dsv4_fp8_grouped_swiglu_decode(
                cache_ptr(&tables.gate_w, ctx),
                cache_ptr(&tables.gate_s, ctx),
                cache_ptr(&tables.up_w, ctx),
                cache_ptr(&tables.up_s, ctx),
                cache_ptr(&packed_hidden.data, ctx),
                cache_ptr(&act.data, ctx),
                cache_ptr(offsets, ctx),
                cache_ptr(counts, ctx),
                experts_per_rank,
                rows,
                i_dim,
                hidden_dim,
                tables.sc13,
                model.config.swiglu_limit,
                ctx.stream.cu_stream(),
            )?;
            moe::dsv4_fp8_grouped_down_decode(
                cache_ptr(&tables.w2_w, ctx),
                cache_ptr(&tables.w2_s, ctx),
                cache_ptr(&act.data, ctx),
                cache_ptr(&expert_out.data, ctx),
                cache_ptr(offsets, ctx),
                cache_ptr(counts, ctx),
                experts_per_rank,
                rows,
                hidden_dim,
                i_dim,
                tables.sc2,
                ctx.stream.cu_stream(),
            )?;
        }

        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_scatter_all_route_slots(
                cache_ptr(&expert_out.data, ctx),
                cache_ptr(&route_out.data, ctx),
                cache_ptr(packed_route_slot, ctx),
                cache_ptr(packed_weight, ctx),
                rows,
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
        Ok(())
    }

    /// Masked-path MoE tail shared by the eager forward and the decode graph.
    /// Fully device-driven: intermediates are stream-ordered allocs (legal inside
    /// graph capture) and the -1 route-slot sentinel is a device memset.
    #[allow(clippy::too_many_arguments)]
    fn dsv4_moe_forward_masked_tail(
        model: &Dsv4Model,
        layer: &Dsv4MoeLayer,
        route_indices: &CudaSlice<i32>,
        route_weights: &CudaSlice<f32>,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        keepalive: &mut Dsv4ForwardKeepalive,
        tail: Option<&mut Dsv4MoeTailScratch>,
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
        let total_routes = num_tokens * topk;
        // Decode-band FP8 grouped GEMM lane: compact (real routed rows only, no
        // pad), 16-byte vectorized FP8 weight loads.
        if total_routes <= DSV4_DECODE_GEMV_MAX_ROUTES {
            let tables = layer.gemv_tables.get_or_init(|| {
                build_gemv_tables(ctx, layer).map(Some).unwrap_or_else(|e| {
                    log::warn!("DSv4 GEMV decode lane table build failed: {e}");
                    None
                })
            });
            if let Some(tables) = tables.as_ref() {
                return dsv4_moe_forward_decode_fp8(
                    model,
                    layer,
                    tables,
                    route_indices,
                    route_weights,
                    hidden,
                    out,
                    keepalive,
                    tail,
                );
            }
        }
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
        // Decode band: pack 64-aligned and cap the GEMM's block_m at 64 — same
        // per-tile single-group contract, half the pad rows.
        let contig_align = if total_routes <= DSV4_DECODE_CONTIG_MAX_ROUTES {
            DSV4_DECODE_CONTIG_ALIGN
        } else {
            DEEPGEMM_CONTIG_ALIGN
        };
        // SAFETY: all buffers valid on ctx.stream for the given shapes.
        unsafe {
            moe::dsv4_count_local_experts(
                cache_ptr(route_indices, ctx),
                cache_ptr(&counts, ctx),
                num_tokens,
                topk,
                local_start,
                experts_per_rank,
                ctx.stream.cu_stream(),
            )?;
            moe::moe_exclusive_scan_aligned_i32(
                cache_ptr(&counts, ctx),
                cache_ptr(&offsets, ctx),
                cache_ptr(&scan_total, ctx),
                experts_per_rank,
                contig_align,
                ctx.stream.cu_stream(),
            )?;
        }

        let packed_rows =
            deepgemm_contig_rows_cap(total_routes.max(1), experts_per_rank, contig_align);
        let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, packed_rows)?;
        keepalive.keep_hidden(&packed_hidden);
        // -1, NOT 0: the scatter treats only route_slot < 0 as invalid, and
        // zero-init made unfilled rows look like valid slot-0 rows (m=1 decode
        // overwrote route slot 0 with zero output).
        let packed_route_slot = alloc_neg1_i32(ctx, packed_rows)?;
        let packed_weight = ctx
            .stream
            .alloc_zeros::<f32>(packed_rows)
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
                cache_ptr(route_indices, ctx),
                cache_ptr(route_weights, ctx),
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

        let expert_out =
            crate::profile::profile_op(ctx, "deepgemm_grouped", None, num_tokens, || {
                deepgemm_grouped_experts(
                    ctx,
                    layer,
                    &packed_hidden,
                    &counts,
                    &offsets,
                    contig_align,
                    swiglu_limit,
                    keepalive,
                )
            })?;
        keepalive.keep_hidden(&expert_out);

        crate::profile::profile_op(ctx, "combine_scatter", None, num_tokens, || {
            let route_out = HiddenStates::zeros(ctx, hidden_dim, total_routes.max(1))?;
            keepalive.keep_hidden(&route_out);
            // SAFETY: all buffers valid on ctx.stream for the given shapes.
            unsafe {
                moe::dsv4_scatter_all_route_slots(
                    cache_ptr(&expert_out.data, ctx),
                    cache_ptr(&route_out.data, ctx),
                    cache_ptr(&packed_route_slot, ctx),
                    cache_ptr(&packed_weight, ctx),
                    expert_out.seq_len,
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
            Ok(())
        })?;

        // The shared expert is replicated: callers all-reduce the routed local
        // contribution first, then add it exactly once per rank.
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
        // Fail loud if the native DeepGEMM bridge is a build-time stub.
        moe::dsv4_deepgemm_native_preflight()?;

        // DeepEP dispatch consumes global expert ids as i64 and remaps received
        // ids to rank-local expert ids.
        let mut logits = HiddenStates::zeros(ctx, cfg.num_experts, num_tokens)?;
        gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
        let total_routes = num_tokens * topk;
        let routing = dsv4_route_device(model, layer, tokens, &logits, keepalive)?;
        let topk_idx_i64 = ctx
            .stream
            .alloc_zeros::<i64>(total_routes)
            .map_err(|e| anyhow::anyhow!("DSv4 DeepEP route-index i64 alloc failed: {e}"))?;
        keepalive.keep_i64(&topk_idx_i64);
        unsafe {
            moe::dsv4_cast_i32_to_i64(
                cache_ptr(&routing.indices, ctx),
                cache_ptr(&topk_idx_i64, ctx),
                total_routes,
                ctx.stream.cu_stream(),
            )?;
        }
        let route_weights = routing.weights;

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
            // The local count/pack kernels take i32, DeepEP returns i64.
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
            // mk_align must be 64 or 128 (C gate in deepgemm_native.cu rejects
            // any other value with CUDA_ERROR_INVALID_VALUE).
            let contig_align = if recv_slots <= DSV4_DECODE_CONTIG_MAX_ROUTES {
                DSV4_DECODE_CONTIG_ALIGN
            } else {
                DEEPGEMM_CONTIG_ALIGN
            };
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
                moe::moe_exclusive_scan_aligned_i32(
                    cache_ptr(&counts, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&scan_total, ctx),
                    experts_per_rank,
                    contig_align,
                    ctx.stream.cu_stream(),
                )?;
            }
            let packed_rows =
                deepgemm_contig_rows_cap(recv_slots.max(1), experts_per_rank, contig_align);

            let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, packed_rows)?;
            let packed_route_slot = alloc_neg1_i32(ctx, packed_rows)?;
            let packed_weight = ctx
                .stream
                .alloc_zeros::<f32>(packed_rows)
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
                contig_align,
                swiglu_limit,
                keepalive,
            )?;
            keepalive.keep_hidden(&expert_out);
            let route_out = HiddenStates::zeros(ctx, hidden_dim, recv_slots.max(1))?;
            keepalive.keep_hidden(&route_out);
            unsafe {
                // With aligned packing, valid routes span positions 0..packed_rows
                // (not just 0..recv_slots): must iterate the full packed range.
                moe::dsv4_scatter_all_route_slots(
                    cache_ptr(&expert_out.data, ctx),
                    cache_ptr(&route_out.data, ctx),
                    cache_ptr(&packed_route_slot, ctx),
                    cache_ptr(&packed_weight, ctx),
                    packed_rows,
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
        keepalive: &mut Dsv4ForwardKeepalive,
    ) -> Result<()> {
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

    pub(crate) fn dsv4_shared_expert_forward_decode_scratch(
        ctx: &DeviceContext,
        stream: &Arc<CudaStream>,
        layer: &Dsv4MoeLayer,
        hidden: &HiddenStates,
        out: &mut HiddenStates,
        swiglu_limit: f32,
        scratch: &mut Dsv4SharedDecodeScratch,
    ) -> Result<()> {
        dsv4_shared_expert_pooled(ctx, stream, layer, hidden, out, swiglu_limit, scratch)
    }

    /// FP8 DeepGEMM grouped expert pipeline over this rank's local experts;
    /// returns the padded/aligned expert output. `contig_align` must match the
    /// per-group alignment `offsets` were packed with (the bridge caps block_m at
    /// it so tiles never span groups).
    #[allow(clippy::too_many_arguments)]
    fn deepgemm_grouped_experts(
        ctx: &DeviceContext,
        layer: &Dsv4MoeLayer,
        packed_hidden: &HiddenStates,
        counts: &cudarc::driver::CudaSlice<i32>,
        offsets: &cudarc::driver::CudaSlice<i32>,
        contig_align: usize,
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

        // Contiguous grouped layout over 128-aligned route rows: the masked
        // layout materialized `num_groups * max_m` rows and overflowed the unpad
        // kernel's i32 work size at ~1.5K prompt tokens.
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
        let m_indices = alloc_neg1_i32(ctx, rows)?;
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
                contig_align,
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
                contig_align,
                stream,
            )?;
        }

        out_compact.seq_len = packed_hidden.seq_len;
        Ok(out_compact)
    }

    /// Dense shared expert: one single-group FP8 DeepGEMM pass, no routing.
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
        ensure!(
            num_tokens > 0,
            "GLM bf16 shared expert requires at least one token"
        );
        let shared_inter = layer.shared_w2.cols;
        ensure!(
            num_tokens <= scratch.max_m
                && scratch.out.hidden_dim == hidden_dim
                && scratch.w13_out.hidden_dim == 2 * shared_inter,
            "DSv4 pooled shared scratch mismatch: tokens={num_tokens} max_m={} H={hidden_dim} I={shared_inter}",
            scratch.max_m
        );
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == num_tokens,
            "DSv4 pooled shared out shape {}x{} != hidden {}x{}",
            out.hidden_dim,
            out.seq_len,
            hidden_dim,
            num_tokens
        );
        if num_tokens != 1 {
            let num_tokens_i32 = i32::try_from(num_tokens).map_err(|_| {
                anyhow::anyhow!("DSv4 shared token count {num_tokens} overflows i32")
            })?;
            stream
                .memcpy_htod(&[num_tokens_i32], &mut scratch.counts)
                .map_err(|e| anyhow::anyhow!("DSv4 shared count scratch update failed: {e}"))?;
            stream
                .memcpy_htod(&[num_tokens_i32], &mut scratch.masked_m)
                .map_err(|e| anyhow::anyhow!("DSv4 shared masked-m scratch update failed: {e}"))?;
        }
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

        // SAFETY: ptrs from live device allocations sized to the dims passed.
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
        if num_tokens != 1 {
            stream
                .memcpy_htod(&[1i32], &mut scratch.counts)
                .map_err(|e| anyhow::anyhow!("DSv4 shared count scratch reset failed: {e}"))?;
            stream
                .memcpy_htod(&[1i32], &mut scratch.masked_m)
                .map_err(|e| anyhow::anyhow!("DSv4 shared masked-m scratch reset failed: {e}"))?;
        }
        Ok(())
    }

    /// Dense shared expert: one single-group FP8 DeepGEMM pass, no routing.
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

        // Floor at 128: the masked GEMM's small-m tile path diverges below it.
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

        // SAFETY: single-group buffers valid on the selected stream; masked_m bounds
        // rows.
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

    #[derive(Clone, Copy)]
    pub(crate) enum GroupedWeightLayout {
        Normal,
        InterleavedL1,
    }

    /// Concatenate the per-expert FP8 caches into one contiguous group-major
    /// buffer (D2D), validating uniform `[rows, cols]` + 128-row alignment. Built
    /// once at load: the weights are static, and rebuilding this concat during
    /// decode would copy hundreds of MiB per layer per token.
    pub(crate) fn build_grouped_cache(
        ctx: &DeviceContext,
        caches: &[Dsv4Fp8DeepGemmWeightCache],
        rows: usize,
        cols: usize,
        layout: GroupedWeightLayout,
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
            let mut dst = weight.slice_mut(g * weight_stride..(g + 1) * weight_stride);
            match layout {
                GroupedWeightLayout::Normal => ctx
                    .stream
                    .memcpy_dtod(&cache.weight, &mut dst)
                    .map_err(|e| anyhow::anyhow!("DSv4 grouped weight D2D failed: {e}"))?,
                GroupedWeightLayout::InterleavedL1 => {
                    ensure!(
                        rows.is_multiple_of(16),
                        "MegaMoE L1 fused rows must be divisible by 16, got {rows}"
                    );
                    let half_len = weight_stride / 2;
                    let gate = cache.weight.slice(0..half_len);
                    let up = cache.weight.slice(half_len..weight_stride);
                    cuda_kernels::moe::interleave_gate_up_fp8_rows(
                        ctx,
                        &gate,
                        &up,
                        &mut dst,
                        rows / 2,
                        cols,
                    )?;
                }
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

    /// NVSHMEM low-latency DSv4 MoE forward for THIS rank's owned token slice:
    /// route → LL dispatch (FP8 pack) → masked grouped GEMM w13 → masked
    /// SwiGLU+requant → masked grouped GEMM w2 → LL combine. The caller owns the
    /// token slicing and the final all-gather; `out` is `[hidden, owned_n]`.
    #[cfg(feature = "deepep")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dsv4_moe_forward_deepep_ll(
        model: &Dsv4Model,
        transport: &crate::deepep::DeepEpTransport,
        scratch: &mut crate::deepep::DeepEpLlScratch,
        layer: &Dsv4MoeLayer,
        tokens: &[u32],
        global_tokens: usize,
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

        // owned_n may be 0 (seq_len < world): this rank still participates in the
        // dispatch/combine COLLECTIVE, so allocate 1-slot dummies for the route
        // buffers and skip the route compute.
        let total_routes = owned_n * topk;
        let topk_idx_i64 = ctx
            .stream
            .alloc_zeros::<i64>(total_routes.max(1))
            .map_err(|e| anyhow::anyhow!("deepep_ll route-index i64 alloc failed: {e}"))?;
        keepalive.keep_i64(&topk_idx_i64);
        let route_weights = if owned_n > 0 {
            let mut logits = HiddenStates::zeros(ctx, cfg.num_experts, owned_n)?;
            gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
            keepalive.keep_hidden(&logits);
            let routing = dsv4_route_device(model, layer, tokens, &logits, keepalive)?;
            keepalive.keep_f32(&routing.weights);
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
            keepalive.keep_f32(&w);
            w
        };

        // `scratch` is model-owned (outlives the forward), so it does not need the
        // forward-keepalive guard the transient buffers below get.
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
            // The grid covers only `min(global_tokens, m)` rows per expert: a
            // full-band grid measured 631 µs/layer of empty-block drain at B=1,
            // 52.9% of deepep_ll GPU time.
            moe::dsv4_deepgemm_silu_mul_masked_quant(
                p_w13_out,
                p_act_fp8,
                p_act_sc,
                p_masked,
                num_local_experts,
                m,
                global_tokens.max(1).min(m),
                2 * intermediate,
                swiglu_limit,
                stream,
            )?;
            // The w2 output layout IS the LL-combine input layout.
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

        // The shared expert is NOT added here — dsv4.rs adds it on the FULL
        // gathered `moe_out` afterward, so adding it here would double-count.
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
#[allow(unused_imports)] // consumed by the model.rs DSv4 branch
pub(crate) use dsv4_gpu::{
    Dsv4GemvTables, Dsv4MoeTailScratch, Dsv4SharedDecodeScratch, GroupedCache, GroupedWeightLayout,
    build_grouped_cache, dsv4_moe_forward, dsv4_shared_expert_forward,
    dsv4_shared_expert_forward_decode_scratch,
};
#[cfg(feature = "deepep")]
pub(crate) use dsv4_gpu::{dsv4_moe_forward_deepep, dsv4_moe_forward_deepep_ll};
#[cfg(feature = "cuda")]
pub(crate) use gpu::{
    MoeForwardScratch, moe_forward, moe_forward_into, qwen35_decode_moe_graph_capturable,
};
