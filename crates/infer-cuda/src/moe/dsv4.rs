use anyhow::{Result, ensure};
use cuda_kernels::moe;
use cuda_kernels::prelude::{DeviceContext, HiddenStates};
use cuda_kernels::tensor::{Dsv4Fp8DeepGemmWeightCache, cache_ptr};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::bf16;
use std::sync::Arc;

use super::{
    DEEPGEMM_CONTIG_ALIGN, DSV4_DECODE_CONTIG_ALIGN, DSV4_DECODE_CONTIG_MAX_ROUTES, alloc_neg1_i32,
    deepgemm_contig_rows_cap,
};
use crate::dsv4::GraphSlot;
use crate::dsv4::{Dsv4ForwardKeepalive, Dsv4Model, Dsv4MoeLayer};
use crate::ops::gemm_batch;

pub(super) struct DeviceRouting {
    pub(super) indices: crate::dsv4::StepSlice<i32>,
    pub(super) weights: crate::dsv4::StepSlice<f32>,
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
    pub(crate) fn new(ctx: &DeviceContext, hidden_dim: usize, shared_inter: usize) -> Result<Self> {
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

pub(super) fn dsv4_route_device(
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

    let graph_mode = model.graph_mode() && num_tokens == 1;
    let route_indices = model.step_i32(
        graph_mode,
        (layer.layer_idx, GraphSlot::RouteIndices),
        total_routes,
    )?;
    let route_weights = model.step_f32(
        graph_mode,
        (layer.layer_idx, GraphSlot::RouteWeights),
        total_routes,
    )?;
    let token_ids = if matches!(layer.routing_kind, DeepSeekV4MoeRoutingKind::Hash) {
        let token_ids = if model.graph_mode() {
            // Graph capture: the persistent pre-replay buffer, not a host-coupled memcpy node.
            model.graph_token_ids_u32()?
        } else {
            crate::dsv4::StepSlice::Owned(
                ctx.stream
                    .clone_htod(tokens)
                    .map_err(|e| anyhow::anyhow!("DSv4 device route token-id H2D failed: {e}"))?,
            )
        };
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
                let graph_mode = model.graph_mode() && num_tokens == 1;
                let mut logits = model.step_hidden(
                    graph_mode,
                    (layer.layer_idx, GraphSlot::RouterLogits),
                    cfg.num_experts,
                    num_tokens,
                )?;
                gemm_batch(ctx, &layer.gate, hidden, &mut logits)?;
                let routing = dsv4_route_device(model, layer, tokens, &logits, keepalive)?;
                Ok((routing.indices, routing.weights))
            })
        })?;

    #[cfg(all(feature = "cuda", feature = "nccl"))]
    if let Some(mega_moe) = model.mega_moe.as_ref()
        && layer.w13_w4a16.is_none()
        && layer.w13_w4afp8.is_none()
    {
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
        crate::stage_profile::profile(ctx, "dsv4/stage/mega_moe", || -> Result<_> {
            let w13 = layer
                .w13_grouped
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("DSv4 MegaMoE layer missing grouped w13 weights"))?;
            let w2 = layer
                .w2_grouped
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("DSv4 MegaMoE layer missing grouped w2 weights"))?;
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
                    l1_weights: cache_ptr(&w13.weight, ctx),
                    l1_weight_stride: hidden_dim,
                    l1_weights_sf: cache_ptr(&w13.scales, ctx),
                    l2_weights: cache_ptr(&w2.weight, ctx),
                    l2_weight_stride: layer.intermediate,
                    l2_weights_sf: cache_ptr(&w2.scales, ctx),
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
                .map_err(|error| anyhow::anyhow!("MegaMoE owned output copy failed: {error}"))?;
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
    sc13: usize,
    sc2: usize,
}

/// Per-expert device pointer tables for the W4A16 grouped-GEMV MoE lane.
/// Built lazily on first W4A16 forward; the packed INT4 weight and BF16
/// scale pointers index directly into the per-expert `DeviceMatrix`
/// allocations (no grouped-cache re-encoding).
pub(crate) struct Dsv4W4A16GemvTables {
    gate_w: CudaSlice<u64>,
    gate_s: CudaSlice<u64>,
    up_w: CudaSlice<u64>,
    up_s: CudaSlice<u64>,
    w2_w: CudaSlice<u64>,
    w2_s: CudaSlice<u64>,
    /// Identity `[0, 1, …, experts_per_rank-1]` — DSv4 offsets/counts and
    /// pointer tables share the same local-expert index space.
    expert_indices: CudaSlice<i32>,
    group_size: usize,
    /// Owned transposed scale buffers (W4AFP8 decode lane only — the W4A16
    /// lane points into per-expert `DeviceMatrix.qscales` and leaves this empty).
    scale_storage: Vec<CudaSlice<u8>>,
}

fn build_gemv_tables(ctx: &DeviceContext, layer: &Dsv4MoeLayer) -> Result<Dsv4GemvTables> {
    let g = layer.num_groups;
    let h = layer.hidden_dim;
    let i_dim = layer.intermediate;
    let w13 = layer
        .w13_grouped
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("GEMV tables: layer missing grouped w13 weights"))?;
    let w2 = layer
        .w2_grouped
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("GEMV tables: layer missing grouped w2 weights"))?;
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

/// Build per-expert device pointer tables for the W4A16 grouped-GEMV MoE
/// lane. Each fused w13 `DeviceMatrix` holds gate (rows 0..I) and up
/// (rows I..2I) back-to-back; the up pointers offset into the same buffer.
fn build_w4a16_gemv_tables(
    ctx: &DeviceContext,
    layer: &Dsv4MoeLayer,
) -> Result<Dsv4W4A16GemvTables> {
    let w13 = layer
        .w13_w4a16
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("W4A16 GEMV tables: layer has no W4A16 experts"))?;
    let w2 = layer
        .w2_w4a16
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("W4A16 GEMV tables: layer has no W4A16 down experts"))?;
    let g = w13.len();
    ensure!(
        w2.len() == g,
        "W4A16 GEMV tables: w13/w2 expert count mismatch"
    );
    let first = w13
        .first()
        .ok_or_else(|| anyhow::anyhow!("W4A16 GEMV tables: no experts"))?;
    let group_size = first.group_size;
    let i_dim = first.rows / 2;
    let h = first.cols;
    let up_weight_off = (i_dim * (h / 2)) as u64;
    let up_scale_off = (i_dim * (h / group_size) * std::mem::size_of::<bf16>()) as u64;

    let mut gate_w = Vec::with_capacity(g);
    let mut gate_s = Vec::with_capacity(g);
    let mut up_w = Vec::with_capacity(g);
    let mut up_s = Vec::with_capacity(g);
    let mut w2_w = Vec::with_capacity(g);
    let mut w2_s = Vec::with_capacity(g);
    for e in 0..g {
        let w13e = &w13[e];
        let w2e = &w2[e];
        let (w13_ptr, _g13) = w13e
            .qweight
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("W4A16 expert {e} missing qweight"))?
            .device_ptr(&ctx.stream);
        let (s13_ptr, _gs13) = w13e
            .qscales
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("W4A16 expert {e} missing qscales"))?
            .device_ptr(&ctx.stream);
        let (w2_ptr, _gw2) = w2e
            .qweight
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("W4A16 down expert {e} missing qweight"))?
            .device_ptr(&ctx.stream);
        let (s2_ptr, _gs2) = w2e
            .qscales
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("W4A16 down expert {e} missing qscales"))?
            .device_ptr(&ctx.stream);
        gate_w.push(w13_ptr);
        up_w.push(w13_ptr + up_weight_off);
        gate_s.push(s13_ptr);
        up_s.push(s13_ptr + up_scale_off);
        w2_w.push(w2_ptr);
        w2_s.push(s2_ptr);
    }
    let h2d = |v: &[u64]| -> Result<CudaSlice<u64>> {
        ctx.stream
            .clone_htod(v)
            .map_err(|e| anyhow::anyhow!("W4A16 GEMV table H2D failed: {e}"))
    };
    let expert_indices: Vec<i32> = (0..g as i32).collect();
    Ok(Dsv4W4A16GemvTables {
        gate_w: h2d(&gate_w)?,
        gate_s: h2d(&gate_s)?,
        up_w: h2d(&up_w)?,
        up_s: h2d(&up_s)?,
        w2_w: h2d(&w2_w)?,
        w2_s: h2d(&w2_s)?,
        expert_indices: ctx
            .stream
            .clone_htod(&expert_indices)
            .map_err(|e| anyhow::anyhow!("W4A16 expert_indices H2D failed: {e}"))?,
        group_size,
        scale_storage: vec![],
    })
}

/// Build W4AFP8 GEMV decode tables: transpose the CUTLASS scale layout
/// to the W4A16 GEMV kernel's row-major [N, K//128], then point into
/// the fused weight + transposed scale buffers.
///
/// w13 scales are [K//512, n13*4] with w1/w3 concatenated along N —
/// the plain CUTLASS layout, transpose directly. w2 scales are the same
/// layout with N=hidden_dim.
fn build_w4afp8_gemv_tables(
    ctx: &DeviceContext,
    layer: &Dsv4MoeLayer,
) -> Result<Dsv4W4A16GemvTables> {
    let w13 = layer
        .w13_w4afp8
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("W4AFP8 GEMV tables: layer has no W4AFP8 experts"))?;
    let w2 = layer
        .w2_w4afp8
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("W4AFP8 GEMV tables: layer has no W4AFP8 down experts"))?;
    let group_size = 128usize;
    let h = layer.hidden_dim;
    let i_dim = layer.intermediate;

    let transpose_plain = |src: &[u8], n: usize, k: usize| -> Vec<u8> {
        let num_groups = k / group_size;
        let mut dst = vec![0u8; n * num_groups * 2];
        for g in 0..num_groups {
            let chunk = g / 4;
            let sub = g % 4;
            for row in 0..n {
                let src_off = (chunk * (n * 4) + row * 4 + sub) * 2;
                let dst_off = (row * num_groups + g) * 2;
                dst[dst_off] = src[src_off];
                dst[dst_off + 1] = src[src_off + 1];
            }
        }
        dst
    };

    let w13_src = ctx
        .stream
        .clone_dtoh(&w13.scales)
        .map_err(|e| anyhow::anyhow!("W4AFP8 w13 scale download failed: {e}"))?;
    let n13 = w13.n;
    let k13 = w13.k;
    let row_bytes = (k13 / group_size) * 2;
    let per_expert_13 = n13 * row_bytes;
    let mut w13_gemv = Vec::with_capacity(w13.num_experts * per_expert_13);
    for e in 0..w13.num_experts {
        let base = e * per_expert_13;
        w13_gemv.extend_from_slice(&transpose_plain(
            &w13_src[base..base + per_expert_13],
            n13,
            k13,
        ));
    }
    let w13_gemv_scales = ctx
        .stream
        .clone_htod(&w13_gemv)
        .map_err(|e| anyhow::anyhow!("W4AFP8 w13 transposed scale upload failed: {e}"))?;

    let w2_src = ctx
        .stream
        .clone_dtoh(&w2.scales)
        .map_err(|e| anyhow::anyhow!("W4AFP8 w2 scale download failed: {e}"))?;
    let per_expert_w2 = (w2.k / 512) * (w2.n * 4) * 2;
    let mut w2_gemv = Vec::with_capacity(w2.num_experts * w2.n * (w2.k / group_size) * 2);
    for e in 0..w2.num_experts {
        let base = e * per_expert_w2;
        w2_gemv.extend_from_slice(&transpose_plain(
            &w2_src[base..base + per_expert_w2],
            w2.n,
            w2.k,
        ));
    }
    let w2_gemv_scales = ctx
        .stream
        .clone_htod(&w2_gemv)
        .map_err(|e| anyhow::anyhow!("W4AFP8 w2 transposed scale upload failed: {e}"))?;

    let w13_weight_ptr = cache_ptr(&w13.weight, ctx).as_ptr() as u64;
    let w13_scale_ptr = cache_ptr(&w13_gemv_scales, ctx).as_ptr() as u64;
    let w2_weight_ptr = cache_ptr(&w2.weight, ctx).as_ptr() as u64;
    let w2_scale_ptr = cache_ptr(&w2_gemv_scales, ctx).as_ptr() as u64;

    let w13_w_per_expert = (2 * i_dim * (h / 2)) as u64;
    let w13_s_per_expert = (2 * i_dim * (h / group_size) * 2) as u64;
    let up_weight_off = (i_dim * (h / 2)) as u64;
    let up_scale_off = (i_dim * (h / group_size) * 2) as u64;
    let w2_w_per_expert = (h * (i_dim / 2)) as u64;
    let w2_s_per_expert = (h * (i_dim / group_size) * 2) as u64;

    let e = w13.num_experts;
    let mut gate_w = Vec::with_capacity(e);
    let mut gate_s = Vec::with_capacity(e);
    let mut up_w = Vec::with_capacity(e);
    let mut up_s = Vec::with_capacity(e);
    let mut w2_w = Vec::with_capacity(e);
    let mut w2_s = Vec::with_capacity(e);
    for idx in 0..e {
        let wb = w13_weight_ptr + idx as u64 * w13_w_per_expert;
        let sb = w13_scale_ptr + idx as u64 * w13_s_per_expert;
        gate_w.push(wb);
        up_w.push(wb + up_weight_off);
        gate_s.push(sb);
        up_s.push(sb + up_scale_off);
        w2_w.push(w2_weight_ptr + idx as u64 * w2_w_per_expert);
        w2_s.push(w2_scale_ptr + idx as u64 * w2_s_per_expert);
    }
    let h2d = |v: &[u64]| -> Result<CudaSlice<u64>> {
        ctx.stream
            .clone_htod(v)
            .map_err(|e| anyhow::anyhow!("W4AFP8 GEMV table H2D failed: {e}"))
    };
    let expert_indices: Vec<i32> = (0..e as i32).collect();
    Ok(Dsv4W4A16GemvTables {
        gate_w: h2d(&gate_w)?,
        gate_s: h2d(&gate_s)?,
        up_w: h2d(&up_w)?,
        up_s: h2d(&up_s)?,
        w2_w: h2d(&w2_w)?,
        w2_s: h2d(&w2_s)?,
        expert_indices: ctx
            .stream
            .clone_htod(&expert_indices)
            .map_err(|e| anyhow::anyhow!("W4AFP8 expert_indices H2D failed: {e}"))?,
        group_size,
        scale_storage: vec![w13_gemv_scales, w2_gemv_scales],
    })
}

/// All intermediates are stream-ordered allocs (graph-capture safe).
#[allow(clippy::too_many_arguments)]
fn dsv4_moe_forward_w4a16(
    model: &Dsv4Model,
    layer: &Dsv4MoeLayer,
    tables: &Dsv4W4A16GemvTables,
    xor_mask: u32,
    use_custom_decode: bool,
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

    // The model-wide tail scratch is pre-allocated to the band ceiling (zero
    // per-step allocs, graph-replay stable); the owned fallback only serves
    // callers without one.
    let mut owned_tail;
    let scratch: &mut Dsv4MoeTailScratch = match tail {
        Some(s) if rows <= s.max_rows => {
            s.reinit(ctx, rows)?;
            s
        }
        _ => {
            owned_tail = Dsv4MoeTailScratch::new(ctx, hidden_dim, i_dim, experts_per_rank)?;
            &mut owned_tail
        }
    };
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

    // SAFETY: `route_indices` holds `num_tokens * topk` global expert ids;
    // `counts`/`offsets` hold `experts_per_rank` i32 and `scan_total` one, all
    // keepalive-held and enqueued on `ctx.stream`.
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

    // SAFETY: `hidden`/`route_indices`/`route_weights` hold `num_tokens` rows × `topk`
    // routes; packed_* hold `rows = max(total_routes, 1)`, `cursors`/`offsets`
    // `experts_per_rank`; all keepalive-held on `ctx.stream`.
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

    // Fused gate+up GEMV with clamped SwiGLU: the kernel accumulates both
    // halves and writes `act` directly, skipping the gate_out/up_out
    // round-trip and the separate SwiGLU launch.
    // SAFETY: `tables` are the layer-resident W4A16 expert tables for
    // `experts_per_rank` experts; `packed_hidden` is [rows, hidden_dim] and
    // `act` [rows, i_dim], keepalive-held on `ctx.stream`.
    unsafe {
        if use_custom_decode {
            moe::w4afp8_grouped_swiglu_decode(
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
                tables.group_size,
                xor_mask,
                model.config.swiglu_limit,
                ctx.stream.cu_stream(),
            )?;
        } else {
            moe::moe_w4a16_grouped_gemv_pair_batch(
                &tables.gate_w,
                &tables.gate_s,
                &tables.up_w,
                &tables.up_s,
                cache_ptr(&packed_hidden.data, ctx),
                cache_ptr(&act.data, ctx),
                None,
                cache_ptr(offsets, ctx),
                cache_ptr(counts, ctx),
                cache_ptr(&tables.expert_indices, ctx),
                experts_per_rank,
                rows,
                i_dim,
                hidden_dim,
                tables.group_size,
                xor_mask,
                true,
                model.config.swiglu_limit,
                ctx,
                ctx.stream.cu_stream(),
            )?;
        }
    }

    // SAFETY: `tables.w2_*` are the layer-resident W4A16 down tables; `act` is
    // [rows, i_dim] and `expert_out` [rows, hidden_dim], keepalive-held on `ctx.stream`.
    unsafe {
        moe::moe_w4a16_grouped_gemv_batch(
            &tables.w2_w,
            &tables.w2_s,
            cache_ptr(&act.data, ctx),
            cache_ptr(&expert_out.data, ctx),
            cache_ptr(offsets, ctx),
            cache_ptr(counts, ctx),
            cache_ptr(&tables.expert_indices, ctx),
            experts_per_rank,
            rows,
            hidden_dim,
            i_dim,
            tables.group_size,
            xor_mask,
            ctx,
            ctx.stream.cu_stream(),
        )?;
    }

    // SAFETY: `expert_out`/`route_out` are [rows, hidden_dim], `out` is
    // [num_tokens, hidden_dim]; `packed_route_slot` is -1 on padding rows, which
    // the scatter skips; all on `ctx.stream`.
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

/// All intermediates are stream-ordered allocs (graph-capture safe).
#[allow(clippy::too_many_arguments)]
fn dsv4_moe_forward_w4afp8(
    model: &Dsv4Model,
    layer: &Dsv4MoeLayer,
    w13: &W4Afp8ExpertWeights,
    w2: &W4Afp8ExpertWeights,
    route_indices: &CudaSlice<i32>,
    route_weights: &CudaSlice<f32>,
    hidden: &HiddenStates,
    out: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
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

    let counts = ctx
        .stream
        .alloc_zeros::<i32>(experts_per_rank)
        .map_err(|e| anyhow::anyhow!("DSv4 W4AFP8 count alloc failed: {e}"))?;
    let offsets = ctx
        .stream
        .alloc_zeros::<i32>(experts_per_rank)
        .map_err(|e| anyhow::anyhow!("DSv4 W4AFP8 offset alloc failed: {e}"))?;
    let scan_total = ctx
        .stream
        .alloc_zeros::<i32>(1)
        .map_err(|e| anyhow::anyhow!("DSv4 W4AFP8 scan-total alloc failed: {e}"))?;
    keepalive.keep_i32(&counts);
    keepalive.keep_i32(&offsets);
    keepalive.keep_i32(&scan_total);

    // SAFETY: `route_indices` holds `num_tokens * topk` global expert ids;
    // `counts`/`offsets` hold `experts_per_rank` i32 and `scan_total` one, all
    // keepalive-held and enqueued on `ctx.stream`.
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
        moe::dsv4_exclusive_scan_i32(
            cache_ptr(&counts, ctx),
            cache_ptr(&offsets, ctx),
            cache_ptr(&scan_total, ctx),
            experts_per_rank,
            ctx.stream.cu_stream(),
        )?;
    }

    let packed_hidden = HiddenStates::zeros(ctx, hidden_dim, rows)?;
    let packed_route_slot = alloc_neg1_i32(ctx, rows)?;
    let packed_weight = ctx
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|e| anyhow::anyhow!("DSv4 W4AFP8 packed_weight alloc failed: {e}"))?;
    let cursors = ctx
        .stream
        .alloc_zeros::<i32>(experts_per_rank)
        .map_err(|e| anyhow::anyhow!("DSv4 W4AFP8 cursors alloc failed: {e}"))?;
    keepalive.keep_hidden(&packed_hidden);
    keepalive.keep_i32(&packed_route_slot);
    keepalive.keep_f32(&packed_weight);
    keepalive.keep_i32(&cursors);

    // SAFETY: `hidden`/`route_indices`/`route_weights` hold `num_tokens` rows × `topk`
    // routes; packed_* hold `rows = max(total_routes, 1)`, `cursors`/`offsets`
    // `experts_per_rank`; all keepalive-held on `ctx.stream`.
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

    // CUTLASS workspace: metadata (E×136 B) + CUTLASS ws. At TP=2 E=128
    // (metadata = 17 KB, CUTLASS ws typically <16 MB); 32 MB is safe.
    // CUTLASS only writes the workspace, so no zero-fill.
    let ws_bytes = 32 * 1024 * 1024;
    // SAFETY: CUTLASS only writes the workspace before reading it, so it needs no
    // zero-fill; the allocation is keepalive-held for the forward.
    let workspace = unsafe { ctx.stream.alloc::<u8>(ws_bytes) }
        .map_err(|e| anyhow::anyhow!("DSv4 W4AFP8 workspace alloc failed: {e}"))?;
    keepalive.keep_u8(&workspace);

    let act_scale = ctx
        .stream
        .alloc_zeros::<f32>(1)
        .map_err(|e| anyhow::anyhow!("DSv4 W4AFP8 act_scale alloc failed: {e}"))?;
    keepalive.keep_f32(&act_scale);

    // problem_sizes [E, 3] (N, M, K — SGLang order) — device-side, CUTLASS reads it.
    let problem_sizes = ctx
        .stream
        .alloc_zeros::<i32>(experts_per_rank * 3)
        .map_err(|e| anyhow::anyhow!("DSv4 W4AFP8 problem_sizes alloc failed: {e}"))?;
    keepalive.keep_i32(&problem_sizes);

    let packed_fp8 = ctx
        .stream
        .alloc_zeros::<u8>(rows * hidden_dim)
        .map_err(|e| anyhow::anyhow!("DSv4 W4AFP8 packed_fp8 alloc failed: {e}"))?;
    keepalive.keep_u8(&packed_fp8);
    let gateup_out = HiddenStates::zeros(ctx, 2 * i_dim, rows)?;
    keepalive.keep_hidden(&gateup_out);

    // SAFETY: `packed_hidden`/`packed_fp8` hold `rows * hidden_dim`, `gateup_out`
    // [rows, 2*i_dim]; `w13` weights/scales are layer-resident for `experts_per_rank`
    // experts and `workspace` is `ws_bytes`; all keepalive-held on `ctx.stream`.
    unsafe {
        moe::w4a8_per_tensor_fp8_quant(
            cache_ptr(&packed_hidden.data, ctx),
            cache_ptr(&packed_fp8, ctx),
            cache_ptr(&act_scale, ctx),
            rows * hidden_dim,
            ctx.stream.cu_stream(),
        )?;
        moe::w4a8_compute_problem_sizes(
            cache_ptr(&counts, ctx),
            cache_ptr(&problem_sizes, ctx),
            experts_per_rank,
            2 * i_dim,
            hidden_dim,
            ctx.stream.cu_stream(),
        )?;
        let rc = moe::w4a8_moe_grouped_gemm(
            cache_ptr(&gateup_out.data, ctx),
            cache_ptr(&packed_fp8, ctx),
            cache_ptr(&w13.weight, ctx),
            cache_ptr(&act_scale, ctx),
            cache_ptr(&w13.scales, ctx),
            cache_ptr(&offsets, ctx),
            cache_ptr(&problem_sizes, ctx),
            experts_per_rank,
            2 * i_dim,
            hidden_dim,
            rows,
            topk,
            cache_ptr(&workspace, ctx),
            ws_bytes,
            ctx.stream.cu_stream(),
        )?;
        ensure!(rc == 0, "W4AFP8 gate+up CUTLASS GEMM failed: {rc}");
    }

    let act = HiddenStates::zeros(ctx, i_dim, rows)?;
    keepalive.keep_hidden(&act);
    // SAFETY: `gateup_out` holds `rows * 2 * i_dim` bf16 and `act` `rows * i_dim`, on `ctx.stream`.
    unsafe {
        moe::w4a8_swiglu_fused(
            cache_ptr(&gateup_out.data, ctx),
            cache_ptr(&act.data, ctx),
            rows,
            i_dim,
            model.config.swiglu_limit,
            ctx.stream.cu_stream(),
        )?;
    }

    let act_fp8 = ctx
        .stream
        .alloc_zeros::<u8>(rows * i_dim)
        .map_err(|e| anyhow::anyhow!("DSv4 W4AFP8 act_fp8 alloc failed: {e}"))?;
    keepalive.keep_u8(&act_fp8);
    let expert_out = HiddenStates::zeros(ctx, hidden_dim, rows)?;
    keepalive.keep_hidden(&expert_out);

    // SAFETY: `act`/`act_fp8` hold `rows * i_dim`, `expert_out` [rows, hidden_dim];
    // `w2` weights/scales are layer-resident for `experts_per_rank` experts and
    // `workspace` is `ws_bytes`; all keepalive-held on `ctx.stream`.
    unsafe {
        moe::w4a8_per_tensor_fp8_quant(
            cache_ptr(&act.data, ctx),
            cache_ptr(&act_fp8, ctx),
            cache_ptr(&act_scale, ctx),
            rows * i_dim,
            ctx.stream.cu_stream(),
        )?;
        moe::w4a8_compute_problem_sizes(
            cache_ptr(&counts, ctx),
            cache_ptr(&problem_sizes, ctx),
            experts_per_rank,
            hidden_dim,
            i_dim,
            ctx.stream.cu_stream(),
        )?;
        let rc = moe::w4a8_moe_grouped_gemm(
            cache_ptr(&expert_out.data, ctx),
            cache_ptr(&act_fp8, ctx),
            cache_ptr(&w2.weight, ctx),
            cache_ptr(&act_scale, ctx),
            cache_ptr(&w2.scales, ctx),
            cache_ptr(&offsets, ctx),
            cache_ptr(&problem_sizes, ctx),
            experts_per_rank,
            hidden_dim,
            i_dim,
            rows,
            topk,
            cache_ptr(&workspace, ctx),
            ws_bytes,
            ctx.stream.cu_stream(),
        )?;
        ensure!(rc == 0, "W4AFP8 down CUTLASS GEMM failed: {rc}");
    }

    let route_out = HiddenStates::zeros(ctx, hidden_dim, rows)?;
    keepalive.keep_hidden(&route_out);
    // SAFETY: `expert_out`/`route_out` are [rows, hidden_dim], `out` is
    // [num_tokens, hidden_dim]; `packed_route_slot` is -1 on padding rows, which
    // the scatter skips; all on `ctx.stream`.
    unsafe {
        moe::dsv4_scatter_all_route_slots(
            cache_ptr(&expert_out.data, ctx),
            cache_ptr(&route_out.data, ctx),
            cache_ptr(&packed_route_slot, ctx),
            cache_ptr(&packed_weight, ctx),
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

/// Decode-band routed-MoE forward via grouped w8a16 GEMM (warp-per-row):
/// zero pad rows and zero activation-quantize work, which removes the grouped
/// lane's padding tax.
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
    if layer.w13_w4a16.is_some() {
        let tables = layer.w4a16_gemv_tables.get_or_init(|| {
            build_w4a16_gemv_tables(ctx, layer)
                .map(Some)
                .unwrap_or_else(|e| {
                    log::warn!("DSv4 W4A16 GEMV table build failed: {e}");
                    None
                })
        });
        if let Some(tables) = tables.as_ref() {
            return dsv4_moe_forward_w4a16(
                model,
                layer,
                tables,
                0,
                false,
                route_indices,
                route_weights,
                hidden,
                out,
                keepalive,
                tail,
            );
        }
    }
    // W4AFP8 lane: decode-band GEMV (reuses W4A16 kernel, BF16 activations)
    // for M=1 only. A/B confirmed GEMV is 8.8% slower than CUTLASS at M=5
    // (30 routes) — the kernel is tuned for 6-route decode and does not scale.
    // M>1 (DSpark verify, batched decode) takes the SGLang CUTLASS grouped GEMM.
    if let (Some(w13), Some(w2)) = (&layer.w13_w4afp8, &layer.w2_w4afp8) {
        if num_tokens == 1 && total_routes <= DSV4_DECODE_GEMV_MAX_ROUTES {
            let tables = layer.w4afp8_gemv_tables.get_or_init(|| {
                build_w4afp8_gemv_tables(ctx, layer)
                    .expect("DSv4 W4AFP8 GEMV decode lane table build failed")
            });
            return dsv4_moe_forward_w4a16(
                model,
                layer,
                tables,
                0x08080808,
                true,
                route_indices,
                route_weights,
                hidden,
                out,
                keepalive,
                tail,
            );
        }
        return dsv4_moe_forward_w4afp8(
            model,
            layer,
            w13,
            w2,
            route_indices,
            route_weights,
            hidden,
            out,
            keepalive,
        );
    }
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

    let packed_rows = deepgemm_contig_rows_cap(total_routes.max(1), experts_per_rank, contig_align);
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
    let w13 = layer
        .w13_grouped
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 masked-tail MoE layer missing grouped w13 weights"))?;
    let w2 = layer
        .w2_grouped
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 masked-tail MoE layer missing grouped w2 weights"))?;
    ensure!(
        w13.rows == intermediate * 2 && w13.cols == hidden_dim,
        "DSv4 grouped w13 cache shape {}x{} != [2*I={}, H={}]",
        w13.rows,
        w13.cols,
        intermediate * 2,
        hidden_dim
    );
    ensure!(
        w2.rows == hidden_dim && w2.cols == intermediate,
        "DSv4 grouped w2 cache shape {}x{} != [H={}, I={}]",
        w2.rows,
        w2.cols,
        hidden_dim,
        intermediate
    );

    let expert_out = crate::profile::profile_op(ctx, "deepgemm_grouped", None, num_tokens, || {
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
pub(super) fn deepgemm_grouped_experts(
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
    let w13 = layer
        .w13_grouped
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 grouped experts layer missing grouped w13 weights"))?;
    let w2 = layer
        .w2_grouped
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 grouped experts layer missing grouped w2 weights"))?;
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
    let active_experts = ctx
        .stream
        .clone_htod(&[0i32])
        .map_err(|e| anyhow::anyhow!("DSv4 DeepGEMM contiguous active-expert H2D failed: {e}"))?;
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
        let num_tokens_i32 = i32::try_from(num_tokens)
            .map_err(|_| anyhow::anyhow!("DSv4 shared token count {num_tokens} overflows i32"))?;
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

/// W4AFP8 routed expert weights in SGLang CUTLASS layout.
/// Weight: int8 [E, N, K/2] packed signed INT4 (two's complement, low nibble = even K).
/// Scales: BF16 [E, K//512, N*4] interleaved per 512-K chunk.
pub(crate) struct W4Afp8ExpertWeights {
    pub(crate) weight: cudarc::driver::CudaSlice<u8>,
    pub(crate) scales: cudarc::driver::CudaSlice<u8>,
    pub(crate) num_experts: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
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
