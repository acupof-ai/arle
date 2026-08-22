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
    device_route_eligible(cfg) && cfg.top_k < QWEN35_DEEPGEMM_MIN_ROUTES
}

/// The block output (routed + sigmoid-gated shared expert) fully overwrites
/// `out`. Routing runs over ALL `cfg.num_experts`, but only routes landing on
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
    let use_deepgemm = has_deepgemm_grouped
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
    let (route_indices, route_weights) = if device_route_eligible(cfg) {
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
        && moe_inter.is_multiple_of(8);
    let fp8_decode_scale_cols = if weights.expert_weight_format == WeightFormat::Fp8BlockScaled
        && total_routes <= QWEN35_MOE_DECODE_MAX_ROUTES
        && hidden_dim.is_multiple_of(16)
        && moe_inter.is_multiple_of(16)
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
        anyhow::bail!("DeepGEMM MoE path requires grouped expert caches (none built at load)")
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
        let aligned_offsets = cache_ptr(scratch.dg_aligned_offsets.get(ctx, local_experts)?, ctx);
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
    let sig = sig.ok_or_else(|| anyhow::anyhow!("FP8 MoE {label} missing dispatch signature"))?;
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
                    Some(cache_ptr(&up_out.data, ctx)),
                    cache_ptr(offsets, ctx),
                    counts,
                    cache_ptr(expert_indices, ctx),
                    local_experts,
                    max_count,
                    n,
                    k,
                    first.group_size,
                    0,
                    false,
                    0.0,
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
                    0,
                    ctx,
                    ctx.stream.cu_stream(),
                )
            }
        }
        other => anyhow::bail!("unsupported Qwen3.6 MoE down format {other}"),
    }
}
