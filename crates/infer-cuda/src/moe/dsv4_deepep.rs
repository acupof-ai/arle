use anyhow::{Result, ensure};
use cuda_kernels::moe;
use cuda_kernels::prelude::HiddenStates;
use cuda_kernels::tensor::cache_ptr;

use super::dsv4::{deepgemm_grouped_experts, dsv4_route_device};
use super::{
    DEEPGEMM_CONTIG_ALIGN, DSV4_DECODE_CONTIG_ALIGN, DSV4_DECODE_CONTIG_MAX_ROUTES, alloc_neg1_i32,
    deepgemm_contig_rows_cap,
};
use crate::dsv4::{Dsv4ForwardKeepalive, Dsv4Model, Dsv4MoeLayer};
use crate::ops::gemm_batch;

pub(crate) fn dsv4_moe_forward_deepep(
    model: &Dsv4Model,
    transport: &crate::deepep::DeepEpTransport,
    layer: &Dsv4MoeLayer,
    tokens: &[u32],
    hidden: &HiddenStates,
    out: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(
        layer.w13_w4a16.is_none() && layer.w13_w4afp8.is_none(),
        "DSv4 DeepEP transport is FP8-only; unset ARLE_DSV4_MOE_TRANSPORT for W4A16/W4AFP8 checkpoints"
    );
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
    // SAFETY: `routing.indices` (i32) and `topk_idx_i64` both hold `total_routes`
    // elements on `ctx.stream`; `topk_idx_i64` is kept alive for the forward.
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
    keepalive.keep_i32(&scratch.recv_topk_idx_i32);
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
        // SAFETY: both recv_topk_idx buffers are keepalive-held scratch of at
        // least `capacity_recv * topk >= recv_slots` elements on `ctx.stream`.
        unsafe {
            moe::dsv4_cast_i64_to_i32(
                cache_ptr(&scratch.recv_topk_idx, ctx),
                cache_ptr(&scratch.recv_topk_idx_i32, ctx),
                recv_slots,
                ctx.stream.cu_stream(),
            )?;
        }
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
        // SAFETY: `recv_topk_idx_i32` holds `num_recv * topk` rank-local ids;
        // `counts`/`offsets` hold `experts_per_rank` i32 and `scan_total` one,
        // all keepalive-held and enqueued on `ctx.stream`.
        unsafe {
            moe::dsv4_count_local_experts(
                cache_ptr(&scratch.recv_topk_idx_i32, ctx),
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
        // SAFETY: `recv_x`/`recv_topk_*` hold `num_recv` rows × `topk` routes;
        // the packed_* buffers hold `packed_rows` (the aligned cap of recv_slots)
        // and `cursors`/`offsets` hold `experts_per_rank`; all keepalive-held on
        // `ctx.stream`.
        unsafe {
            moe::dsv4_pack_local_experts_with_slots(
                cache_ptr(&scratch.recv_x.data, ctx),
                cache_ptr(&scratch.recv_topk_idx_i32, ctx),
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
        // SAFETY: `expert_out` has `packed_rows` rows, `route_out` has
        // `recv_slots` rows and `local_routed` `capacity_recv >= num_recv` rows,
        // all `hidden_dim` wide, keepalive-held and ordered on `ctx.stream`;
        // `packed_route_slot` is -1 for padding rows, which the scatter skips.
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
    Ok(())
}

/// The caller owns the token slicing and the final all-gather; `out` is
/// `[hidden, owned_n]`.
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
    ensure!(
        layer.w13_w4a16.is_none() && layer.w13_w4afp8.is_none(),
        "DSv4 DeepEP-LL transport is FP8-only; unset ARLE_DSV4_MOE_TRANSPORT for W4A16/W4AFP8 checkpoints"
    );
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
        StepSlice::Owned(w)
    };

    // `scratch` is model-owned (outlives the forward), so it does not need the
    // forward-keepalive guard the transient buffers below get.
    let _expected_m = transport.ll_dispatch(ctx, scratch, hidden, &topk_idx_i64, topk)?;

    let m = scratch.m_padded;
    let sfa_aligned_m = scratch.sfa_aligned_m;
    let w13 = layer.w13_grouped.as_ref().unwrap();
    let w2 = layer.w2_grouped.as_ref().unwrap();
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
