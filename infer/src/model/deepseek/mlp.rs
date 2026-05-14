//! DeepSeek V4 MoE FFN scaffold.
//!
//! The local V4 1B checkpoint uses routed experts plus a shared expert on each
//! layer. Phase 0.5 records the correct tensor shape; Phase 1 supplies the
//! shared CUDA MoE primitive, and Phase 2A wires this block into forward.

#[cfg(feature = "cuda")]
use anyhow::{Result, bail, ensure};
#[cfg(feature = "cuda")]
use cuda_kernels::{
    ffi,
    prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates},
    tensor::WeightFormat,
};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
use deepseek_spec::{DeepSeekV4Config, DeepSeekV4MoeRoutingKind};
#[cfg(feature = "cuda")]
use log::info;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use super::state::{
    DeepseekDsv4GroupedBlockFormat, DeepseekExpertRuntimeScratch,
    DeepseekGroupedExpertWeightPtrCache, DeepseekMoeRuntimeCache, ensure_send_route_scratch,
};
#[cfg(feature = "cuda")]
use crate::distributed::expert_state::ExpertGroup;
#[cfg(test)]
use crate::distributed::expert_state::{ExpertRoute, ExpertRoutingWeights, LocalExpertRouting};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use crate::model::layer_communicator::LayerCommunicator;
#[cfg(feature = "cuda")]
use crate::ops;

/// One SwiGLU expert: `w2(silu(w1(x)) * w3(x))`.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4Expert {
    pub(super) w1: DeviceMatrix,
    pub(super) w2: DeviceMatrix,
    pub(super) w3: DeviceMatrix,
}

#[cfg(feature = "cuda")]
impl DeepseekV4Expert {
    /// Run one DeepSeek V4 SwiGLU expert on a packed `[tokens, hidden]` row block.
    pub(super) fn forward(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            self.w1.cols == hidden.hidden_dim && self.w3.cols == hidden.hidden_dim,
            "DeepSeek V4 expert input width mismatch: hidden_dim={} w1.cols={} w3.cols={}",
            hidden.hidden_dim,
            self.w1.cols,
            self.w3.cols
        );
        ensure!(
            self.w1.rows == self.w3.rows && self.w2.cols == self.w1.rows,
            "DeepSeek V4 expert intermediate mismatch: w1.rows={} w3.rows={} w2.cols={}",
            self.w1.rows,
            self.w3.rows,
            self.w2.cols
        );

        let phase = if hidden.seq_len > 1 {
            ops::LinearDispatchPhase::Prefill
        } else {
            ops::LinearDispatchPhase::Decode
        };
        let mut gate = HiddenStates::zeros(ctx, self.w1.rows, hidden.seq_len)?;
        ops::try_gemm_with_phase_into(ctx, &self.w1, hidden, &mut gate, phase)?;
        let mut up = HiddenStates::zeros(ctx, self.w3.rows, hidden.seq_len)?;
        ops::try_gemm_with_phase_into(ctx, &self.w3, hidden, &mut up, phase)?;
        let mut act = HiddenStates::zeros(ctx, self.w1.rows, hidden.seq_len)?;
        ops::dsv4_swiglu_clamped_batch_into(ctx, &gate, &up, &mut act, swiglu_limit)?;
        let mut out = HiddenStates::zeros(ctx, self.w2.rows, hidden.seq_len)?;
        ops::try_gemm_with_phase_into(ctx, &self.w2, &act, &mut out, phase)?;
        Ok(out)
    }

    pub(super) fn forward_scratch_input<'a>(
        &self,
        ctx: &DeviceContext,
        swiglu_limit: f32,
        scratch: &'a mut DeepseekExpertRuntimeScratch,
    ) -> Result<&'a HiddenStates> {
        let seq_len = scratch.input.seq_len;
        ensure!(
            self.w1.cols == scratch.input.hidden_dim && self.w3.cols == scratch.input.hidden_dim,
            "DeepSeek V4 expert input width mismatch: hidden_dim={} w1.cols={} w3.cols={}",
            scratch.input.hidden_dim,
            self.w1.cols,
            self.w3.cols
        );
        ensure!(
            self.w1.rows == self.w3.rows && self.w2.cols == self.w1.rows,
            "DeepSeek V4 expert intermediate mismatch: w1.rows={} w3.rows={} w2.cols={}",
            self.w1.rows,
            self.w3.rows,
            self.w2.cols
        );
        ensure!(
            scratch.capacity_tokens >= seq_len
                && scratch.hidden_dim == scratch.input.hidden_dim
                && scratch.intermediate_dim == self.w1.rows
                && scratch.output_dim == self.w2.rows,
            "DeepSeek V4 expert scratch shape mismatch"
        );

        scratch.gate.seq_len = seq_len;
        scratch.up.seq_len = seq_len;
        scratch.act.seq_len = seq_len;
        scratch.out.seq_len = seq_len;

        let phase = if seq_len > 1 {
            ops::LinearDispatchPhase::Prefill
        } else {
            ops::LinearDispatchPhase::Decode
        };
        ops::try_gemm_with_phase_into(ctx, &self.w1, &scratch.input, &mut scratch.gate, phase)?;
        ops::try_gemm_with_phase_into(ctx, &self.w3, &scratch.input, &mut scratch.up, phase)?;
        ops::dsv4_swiglu_clamped_batch_into(
            ctx,
            &scratch.gate,
            &scratch.up,
            &mut scratch.act,
            swiglu_limit,
        )?;
        ops::try_gemm_with_phase_into(ctx, &self.w2, &scratch.act, &mut scratch.out, phase)?;
        Ok(&scratch.out)
    }
}

#[cfg(feature = "cuda")]
fn dsv4_grouped_format(weight: &DeviceMatrix) -> Option<DeepseekDsv4GroupedBlockFormat> {
    match weight.weight_format {
        WeightFormat::Dsv4Fp8BlockScaled => Some(DeepseekDsv4GroupedBlockFormat::Fp8),
        WeightFormat::Dsv4Fp4BlockScaled => Some(DeepseekDsv4GroupedBlockFormat::Fp4),
        _ => None,
    }
}

#[cfg(feature = "cuda")]
fn dsv4_build_grouped_weight_ptrs<'a, F>(
    ctx: &DeviceContext,
    experts: &'a [DeepseekV4Expert],
    select: F,
) -> Result<DeepseekGroupedExpertWeightPtrCache>
where
    F: Fn(&'a DeepseekV4Expert) -> &'a DeviceMatrix,
{
    let first = select(
        experts
            .first()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 grouped expert path needs experts"))?,
    );
    let format = dsv4_grouped_format(first).ok_or_else(|| {
        anyhow::anyhow!("DeepSeek V4 grouped expert path needs raw FP8/FP4 weights")
    })?;
    ensure!(
        first.dsv4_scale_rows > 0 && first.dsv4_scale_cols > 0,
        "DeepSeek V4 grouped expert path needs block scales"
    );
    let mut weight_ptrs = Vec::with_capacity(experts.len());
    let mut scale_ptrs = Vec::with_capacity(experts.len());
    for expert in experts {
        let weight = select(expert);
        ensure!(
            dsv4_grouped_format(weight) == Some(format)
                && weight.rows == first.rows
                && weight.cols == first.cols
                && weight.dsv4_scale_rows == first.dsv4_scale_rows
                && weight.dsv4_scale_cols == first.dsv4_scale_cols,
            "DeepSeek V4 grouped expert weights must share format and shape"
        );
        let qweight = weight
            .qweight
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 grouped expert matrix missing qweight"))?;
        let scales = weight
            .dsv4_scales
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 grouped expert matrix missing scales"))?;
        let (q_ptr, _q_guard) = qweight.device_ptr(&ctx.stream);
        let (s_ptr, _s_guard) = scales.device_ptr(&ctx.stream);
        weight_ptrs.push(q_ptr as u64);
        scale_ptrs.push(s_ptr as u64);
    }
    Ok(DeepseekGroupedExpertWeightPtrCache {
        weight_ptrs: ctx.stream.clone_htod(&weight_ptrs).map_err(|err| {
            anyhow::anyhow!("DeepSeek V4 grouped expert pointer H2D failed: {err}")
        })?,
        scale_ptrs: ctx.stream.clone_htod(&scale_ptrs).map_err(|err| {
            anyhow::anyhow!("DeepSeek V4 grouped expert scale pointer H2D failed: {err}")
        })?,
        format,
        rows: first.rows,
        cols: first.cols,
        scale_rows: first.dsv4_scale_rows,
        scale_cols: first.dsv4_scale_cols,
    })
}

#[cfg(feature = "cuda")]
fn dsv4_ensure_grouped_weight_ptrs<'a, F>(
    ctx: &DeviceContext,
    experts: &'a [DeepseekV4Expert],
    slot: &'a mut Option<DeepseekGroupedExpertWeightPtrCache>,
    select: F,
) -> Result<&'a DeepseekGroupedExpertWeightPtrCache>
where
    F: Fn(&'a DeepseekV4Expert) -> &'a DeviceMatrix,
{
    if slot.is_none() {
        *slot = Some(dsv4_build_grouped_weight_ptrs(ctx, experts, select)?);
    }
    Ok(slot
        .as_ref()
        .expect("DeepSeek V4 grouped expert weight pointers cached"))
}

#[cfg(feature = "cuda")]
fn dsv4_run_grouped_block_scaled_gemv(
    ctx: &DeviceContext,
    weights: &DeepseekGroupedExpertWeightPtrCache,
    input: &HiddenStates,
    output: &mut HiddenStates,
    offsets: &CudaSlice<i32>,
    counts: &CudaSlice<i32>,
    expert_indices: Option<&CudaSlice<i32>>,
    num_active_experts: usize,
    max_count: usize,
) -> Result<()> {
    ensure!(
        input.hidden_dim == weights.cols && output.hidden_dim == weights.rows,
        "DeepSeek V4 grouped expert GEMV shape mismatch: input={} weight={}x{} output={}",
        input.hidden_dim,
        weights.rows,
        weights.cols,
        output.hidden_dim
    );
    let (w_ptr, _w_guard) = weights.weight_ptrs.device_ptr(&ctx.stream);
    let (s_ptr, _s_guard) = weights.scale_ptrs.device_ptr(&ctx.stream);
    let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
    let (y_ptr, _y_guard) = output.data.device_ptr_mut(&ctx.stream);
    let (off_ptr, _off_guard) = offsets.device_ptr(&ctx.stream);
    let (count_ptr, _count_guard) = counts.device_ptr(&ctx.stream);
    let expert_idx_guard;
    let expert_idx_ptr = if let Some(indices) = expert_indices {
        let (ptr, guard) = indices.device_ptr(&ctx.stream);
        expert_idx_guard = Some(guard);
        ptr as *const i32
    } else {
        expert_idx_guard = None;
        std::ptr::null()
    };
    let res = unsafe {
        match weights.format {
            DeepseekDsv4GroupedBlockFormat::Fp8 => ffi::dsv4_fp8_grouped_gemv_batch_cuda(
                w_ptr as *const u64,
                s_ptr as *const u64,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                off_ptr as *const i32,
                count_ptr as *const i32,
                expert_idx_ptr,
                num_active_experts as i32,
                max_count as i32,
                weights.rows as i32,
                weights.cols as i32,
                weights.scale_rows as i32,
                weights.scale_cols as i32,
                ctx.stream.cu_stream(),
            ),
            DeepseekDsv4GroupedBlockFormat::Fp4 => ffi::dsv4_fp4_grouped_gemv_batch_cuda(
                w_ptr as *const u64,
                s_ptr as *const u64,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                off_ptr as *const i32,
                count_ptr as *const i32,
                expert_idx_ptr,
                num_active_experts as i32,
                max_count as i32,
                weights.rows as i32,
                weights.cols as i32,
                weights.scale_rows as i32,
                weights.scale_cols as i32,
                ctx.stream.cu_stream(),
            ),
        }
    };
    drop(expert_idx_guard);
    res.result()
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 grouped expert GEMV failed: {err}"))
}

#[cfg(feature = "cuda")]
fn dsv4_run_grouped_block_scaled_gemv_pair(
    ctx: &DeviceContext,
    weights_a: &DeepseekGroupedExpertWeightPtrCache,
    weights_b: &DeepseekGroupedExpertWeightPtrCache,
    input: &HiddenStates,
    output_a: &mut HiddenStates,
    output_b: &mut HiddenStates,
    offsets: &CudaSlice<i32>,
    counts: &CudaSlice<i32>,
    expert_indices: Option<&CudaSlice<i32>>,
    num_active_experts: usize,
    max_count: usize,
) -> Result<bool> {
    if weights_a.format != weights_b.format
        || weights_a.rows != weights_b.rows
        || weights_a.cols != weights_b.cols
        || weights_a.scale_rows != weights_b.scale_rows
        || weights_a.scale_cols != weights_b.scale_cols
    {
        return Ok(false);
    }
    ensure!(
        input.hidden_dim == weights_a.cols
            && output_a.hidden_dim == weights_a.rows
            && output_b.hidden_dim == weights_a.rows,
        "DeepSeek V4 grouped expert GEMV pair shape mismatch: input={} weight={}x{} output_a={} output_b={}",
        input.hidden_dim,
        weights_a.rows,
        weights_a.cols,
        output_a.hidden_dim,
        output_b.hidden_dim
    );
    let (wa_ptr, _wa_guard) = weights_a.weight_ptrs.device_ptr(&ctx.stream);
    let (sa_ptr, _sa_guard) = weights_a.scale_ptrs.device_ptr(&ctx.stream);
    let (wb_ptr, _wb_guard) = weights_b.weight_ptrs.device_ptr(&ctx.stream);
    let (sb_ptr, _sb_guard) = weights_b.scale_ptrs.device_ptr(&ctx.stream);
    let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
    let (ya_ptr, _ya_guard) = output_a.data.device_ptr_mut(&ctx.stream);
    let (yb_ptr, _yb_guard) = output_b.data.device_ptr_mut(&ctx.stream);
    let (off_ptr, _off_guard) = offsets.device_ptr(&ctx.stream);
    let (count_ptr, _count_guard) = counts.device_ptr(&ctx.stream);
    let expert_idx_guard;
    let expert_idx_ptr = if let Some(indices) = expert_indices {
        let (ptr, guard) = indices.device_ptr(&ctx.stream);
        expert_idx_guard = Some(guard);
        ptr as *const i32
    } else {
        expert_idx_guard = None;
        std::ptr::null()
    };
    let res = unsafe {
        match weights_a.format {
            DeepseekDsv4GroupedBlockFormat::Fp8 => ffi::dsv4_fp8_grouped_gemv_pair_batch_cuda(
                wa_ptr as *const u64,
                sa_ptr as *const u64,
                wb_ptr as *const u64,
                sb_ptr as *const u64,
                x_ptr as *const ffi::Half,
                ya_ptr as *mut ffi::Half,
                yb_ptr as *mut ffi::Half,
                off_ptr as *const i32,
                count_ptr as *const i32,
                expert_idx_ptr,
                num_active_experts as i32,
                max_count as i32,
                weights_a.rows as i32,
                weights_a.cols as i32,
                weights_a.scale_rows as i32,
                weights_a.scale_cols as i32,
                ctx.stream.cu_stream(),
            ),
            DeepseekDsv4GroupedBlockFormat::Fp4 => ffi::dsv4_fp4_grouped_gemv_pair_batch_cuda(
                wa_ptr as *const u64,
                sa_ptr as *const u64,
                wb_ptr as *const u64,
                sb_ptr as *const u64,
                x_ptr as *const ffi::Half,
                ya_ptr as *mut ffi::Half,
                yb_ptr as *mut ffi::Half,
                off_ptr as *const i32,
                count_ptr as *const i32,
                expert_idx_ptr,
                num_active_experts as i32,
                max_count as i32,
                weights_a.rows as i32,
                weights_a.cols as i32,
                weights_a.scale_rows as i32,
                weights_a.scale_cols as i32,
                ctx.stream.cu_stream(),
            ),
        }
    };
    drop(expert_idx_guard);
    res.result()
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 grouped expert GEMV pair failed: {err}"))?;
    Ok(true)
}

#[cfg(feature = "cuda")]
fn dsv4_grouped_experts_enabled() -> Result<bool> {
    let Some(raw) = std::env::var("ARLE_DSV4_GROUPED_EXPERTS").ok() else {
        return Ok(false);
    };
    match raw.as_str() {
        "1" | "true" | "TRUE" | "on" | "ON" => Ok(true),
        "0" | "false" | "FALSE" | "off" | "OFF" => Ok(false),
        other => bail!("invalid ARLE_DSV4_GROUPED_EXPERTS value `{other}`"),
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Dsv4CountExchangeMode {
    AllGather,
    SendRecv,
}

#[cfg(feature = "cuda")]
fn dsv4_count_exchange_mode() -> Result<Dsv4CountExchangeMode> {
    let Some(raw) = std::env::var("ARLE_DSV4_COUNT_EXCHANGE").ok() else {
        return Ok(Dsv4CountExchangeMode::AllGather);
    };
    match raw.as_str() {
        "allgather" | "all_gather" | "gather" | "AllGather" | "ALLGATHER" => {
            Ok(Dsv4CountExchangeMode::AllGather)
        }
        "sendrecv" | "send_recv" | "p2p" | "SendRecv" | "SENDRECV" => {
            Ok(Dsv4CountExchangeMode::SendRecv)
        }
        other => bail!("invalid ARLE_DSV4_COUNT_EXCHANGE value `{other}`"),
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Dsv4CombineExchangeMode {
    Bf16,
    Fp8,
}

#[cfg(feature = "cuda")]
fn dsv4_combine_exchange_mode() -> Result<Dsv4CombineExchangeMode> {
    let Some(raw) = std::env::var("ARLE_DSV4_COMBINE_DTYPE").ok() else {
        return Ok(Dsv4CombineExchangeMode::Bf16);
    };
    match raw.to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(Dsv4CombineExchangeMode::Bf16),
        "fp8" | "e4m3" | "fp8_e4m3" => Ok(Dsv4CombineExchangeMode::Fp8),
        other => bail!("invalid ARLE_DSV4_COMBINE_DTYPE value `{other}`"),
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
#[allow(clippy::too_many_arguments)]
fn dsv4_run_fp8_combine_exchange(
    ctx: &DeviceContext,
    comm: &LayerCommunicator,
    route_out: &HiddenStates,
    send_u8_offsets: &[usize],
    send_u8_counts: &[usize],
    send_scale_offsets: &[usize],
    send_scale_counts: &[usize],
    recv_u8_offsets: &[usize],
    recv_u8_counts: &[usize],
    recv_scale_offsets: &[usize],
    recv_scale_counts: &[usize],
    send_route_rows: usize,
    recv_route_rows: usize,
    fp8_send: &mut CudaSlice<u8>,
    fp8_recv: &mut CudaSlice<u8>,
    scale_send: &mut CudaSlice<f32>,
    scale_recv: &mut CudaSlice<f32>,
    combine_recv: &mut HiddenStates,
) -> Result<()> {
    let hidden_dim_i32 = dsv4_usize_to_i32(route_out.hidden_dim, "combine hidden_dim")?;
    let send_route_rows_i32 = dsv4_usize_to_i32(send_route_rows, "combine send route count")?;
    let recv_route_rows_i32 = dsv4_usize_to_i32(recv_route_rows, "combine recv route count")?;
    ensure!(
        combine_recv.hidden_dim == route_out.hidden_dim,
        "DeepSeek V4 FP8 combine hidden_dim mismatch: recv={} route_out={}",
        combine_recv.hidden_dim,
        route_out.hidden_dim
    );
    combine_recv.seq_len = recv_route_rows;
    {
        let (route_ptr, _route_guard) = route_out.data.device_ptr(&ctx.stream);
        let (fp8_ptr, _fp8_guard) = fp8_send.device_ptr_mut(&ctx.stream);
        let (scale_ptr, _scale_guard) = scale_send.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::quantize_bf16_rows_to_fp8_e4m3_cuda(
                route_ptr as *const ffi::Half,
                fp8_ptr as *mut u8,
                scale_ptr as *mut f32,
                send_route_rows_i32,
                hidden_dim_i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 combine FP8 quantize failed: {err}"))?;
        }
    }
    comm.moe_grouped_send_recv_u8(
        fp8_send,
        send_u8_offsets,
        send_u8_counts,
        fp8_recv,
        recv_u8_offsets,
        recv_u8_counts,
    )?;
    comm.moe_grouped_send_recv_f32(
        scale_send,
        send_scale_offsets,
        send_scale_counts,
        scale_recv,
        recv_scale_offsets,
        recv_scale_counts,
    )?;
    {
        let (fp8_ptr, _fp8_guard) = fp8_recv.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = scale_recv.device_ptr(&ctx.stream);
        let (recv_ptr, _recv_guard) = combine_recv.data.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::dsv4_dequantize_fp8_rows_to_bf16_cuda(
                fp8_ptr as *const u8,
                scale_ptr as *const f32,
                recv_ptr as *mut ffi::Half,
                recv_route_rows_i32,
                hidden_dim_i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 combine FP8 dequantize failed: {err}"))?;
        }
    }
    Ok(())
}

/// V4 routed MoE block plus optional shared expert.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4MoeBlock {
    pub(super) gate_weight: DeviceMatrix,
    pub(super) gate_bias: Option<DeviceVec>,
    /// Hash-router table for early layers. The exact integer storage type is
    /// finalized with the Phase 2A loader; Phase 0.5 only validates the tensor
    /// name and keeps the field explicit.
    pub(super) gate_tid2eid: Option<CudaSlice<i64>>,
    pub(super) experts: Vec<DeepseekV4Expert>,
    pub(super) shared_experts: Option<DeepseekV4Expert>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)] // method called from forward.rs once MoE kernels land
impl DeepseekV4MoeBlock {
    /// Run routed V4 MoE for a packed `[tokens, hidden]` row block.
    pub(super) fn forward(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            self.experts.is_empty(),
            "DeepSeek V4 routed MoE combine is not wired yet; local experts loaded={}",
            self.experts.len()
        );
        let shared = self
            .shared_experts
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 MoE block has no shared expert"))?;
        shared.forward(ctx, hidden, swiglu_limit)
    }

    /// Run the EP-local portion of routed V4 MoE and add the shared expert.
    ///
    /// `routing` must already be localized for this EP rank. The returned
    /// tensor is this rank's partial MoE output; callers that run multiple EP
    /// ranks still need the cross-rank reduction step.
    #[cfg(test)]
    pub(super) fn forward_local_routes(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        routing: &LocalExpertRouting,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        let routed = self.forward_local_routed_only(ctx, hidden, routing, swiglu_limit)?;
        self.add_shared_expert(ctx, hidden, routed, swiglu_limit)
    }

    /// Run only this EP rank's routed expert contribution.
    ///
    /// The shared expert is intentionally excluded so callers can all-reduce
    /// routed expert outputs across EP ranks and then add the shared expert
    /// exactly once per rank.
    #[cfg(test)]
    pub(super) fn forward_local_routed_only(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        routing: &LocalExpertRouting,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            routing.experts_per_rank == self.experts.len(),
            "DeepSeek V4 routing expects {} local experts but block loaded {}",
            routing.experts_per_rank,
            self.experts.len()
        );

        let mut out = HiddenStates::zeros(ctx, hidden.hidden_dim, hidden.seq_len)?;

        for route in &routing.routes {
            ensure!(
                route.token_idx < hidden.seq_len,
                "DeepSeek V4 route token {} out of range for seq_len {}",
                route.token_idx,
                hidden.seq_len
            );
            let expert = self.experts.get(route.local_expert_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek V4 local expert {} out of range for {} local experts",
                    route.local_expert_idx,
                    self.experts.len()
                )
            })?;
            let token_hidden = hidden_token(ctx, hidden, route.token_idx)?;
            let expert_out = expert.forward(ctx, &token_hidden, swiglu_limit)?;
            ops::add_scaled_row_into(ctx, &expert_out, &mut out, route.token_idx, route.weight)?;
        }

        Ok(out)
    }

    pub(super) fn add_shared_expert(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        routed: HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        let Some(shared) = &self.shared_experts else {
            return Ok(routed);
        };
        let shared = shared.forward(ctx, hidden, swiglu_limit)?;
        ops::add_batch(ctx, &routed, &shared)
    }

    /// Route tokens with the loaded gate tensors, localize routes to this EP
    /// rank, and run the local MoE contribution.
    #[cfg(test)]
    pub(super) fn forward_routed(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
    ) -> Result<HiddenStates> {
        let routing = self.route_local(ctx, layer_idx, config, ep, hidden, token_ids)?;
        self.forward_local_routes(ctx, hidden, &routing, config.swiglu_limit)
    }

    #[cfg(test)]
    pub(super) fn route_local_for_layer(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
    ) -> Result<LocalExpertRouting> {
        self.route_local(ctx, layer_idx, config, ep, hidden, token_ids)
    }

    pub(super) fn forward_local_routed_gpu(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
    ) -> Result<HiddenStates> {
        ensure!(
            token_ids.len() == hidden.seq_len,
            "DeepSeek V4 GPU route token count {} does not match hidden seq_len {}",
            token_ids.len(),
            hidden.seq_len
        );
        ensure!(
            self.gate_weight.rows == config.n_routed_experts
                && self.gate_weight.cols == hidden.hidden_dim,
            "DeepSeek V4 GPU gate shape mismatch: gate={}x{} hidden_dim={} n_routed_experts={}",
            self.gate_weight.rows,
            self.gate_weight.cols,
            hidden.hidden_dim,
            config.n_routed_experts
        );
        ensure!(
            ep.experts_per_rank == self.experts.len(),
            "DeepSeek V4 GPU route expects {} local experts but block loaded {}",
            ep.experts_per_rank,
            self.experts.len()
        );

        let trace = dsv4_moe_trace_begin(ctx)?;
        let logits = ops::gemm(ctx, &self.gate_weight, hidden)?;
        dsv4_moe_trace_end(ctx, "ffn_route_logits", layer_idx, hidden.seq_len, trace)?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        let token_ids_gpu = ctx
            .stream
            .clone_htod(token_ids)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 token ids H2D failed: {err}"))?;
        let mut route_indices = ctx
            .stream
            .alloc_zeros::<i32>(hidden.seq_len * config.num_experts_per_tok)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 route index alloc failed: {err}"))?;
        let mut route_weights = ctx
            .stream
            .alloc_zeros::<f32>(hidden.seq_len * config.num_experts_per_tok)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 route weight alloc failed: {err}"))?;
        dsv4_moe_trace_end(ctx, "ffn_route_setup", layer_idx, hidden.seq_len, trace)?;

        let routing_kind = match config.moe_routing_kind(layer_idx) {
            DeepSeekV4MoeRoutingKind::Hash => 0,
            DeepSeekV4MoeRoutingKind::LearnedBias => 1,
        };
        let scoring_kind = match config.scoring_func.as_str() {
            "softmax" => 0,
            "sigmoid" => 1,
            "sqrtsoftplus" => 2,
            other => bail!("unsupported DSV4 GPU router scoring_func `{other}`"),
        };
        if routing_kind == 0 {
            ensure!(
                self.gate_tid2eid.is_some(),
                "hash-routed DeepSeek V4 MoE layer missing tid2eid"
            );
        } else {
            ensure!(
                self.gate_bias.is_some(),
                "bias-routed DeepSeek V4 MoE layer missing gate bias"
            );
        }

        let trace = dsv4_moe_trace_begin(ctx)?;
        {
            let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
            let bias_guard;
            let bias_ptr = if let Some(bias) = self.gate_bias.as_ref() {
                let (ptr, guard) = bias.data.device_ptr(&ctx.stream);
                bias_guard = Some(guard);
                ptr as *const ffi::Half
            } else {
                bias_guard = None;
                std::ptr::null()
            };
            let tid_guard;
            let tid_ptr = if let Some(tid2eid) = self.gate_tid2eid.as_ref() {
                let (ptr, guard) = tid2eid.device_ptr(&ctx.stream);
                tid_guard = Some(guard);
                ptr as *const i64
            } else {
                tid_guard = None;
                std::ptr::null()
            };
            let (token_ptr, _token_guard) = token_ids_gpu.device_ptr(&ctx.stream);
            let (idx_ptr, _idx_guard) = route_indices.device_ptr_mut(&ctx.stream);
            let (weight_ptr, _weight_guard) = route_weights.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_route_cuda(
                    logits_ptr as *const ffi::Half,
                    bias_ptr,
                    tid_ptr,
                    token_ptr as *const u32,
                    idx_ptr as *mut i32,
                    weight_ptr as *mut f32,
                    hidden.seq_len as i32,
                    config.n_routed_experts as i32,
                    config.num_experts_per_tok as i32,
                    routing_kind,
                    scoring_kind,
                    config.routed_scaling_factor,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU router failed: {err}"))?;
            }
            drop(bias_guard);
            drop(tid_guard);
        }
        dsv4_moe_trace_end(ctx, "ffn_route_select", layer_idx, hidden.seq_len, trace)?;

        let local_expert_start = ep.local_expert_range().start;
        let local_expert_start_i32 = i32::try_from(local_expert_start)
            .map_err(|_| anyhow::anyhow!("DeepSeek V4 local expert start overflows i32"))?;
        let experts_per_rank_i32 = i32::try_from(ep.experts_per_rank)
            .map_err(|_| anyhow::anyhow!("DeepSeek V4 experts_per_rank overflows i32"))?;
        let mut local_counts = ctx
            .stream
            .alloc_zeros::<i32>(ep.experts_per_rank)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 local route count alloc failed: {err}"))?;
        let trace = dsv4_moe_trace_begin(ctx)?;
        {
            let (idx_ptr, _idx_guard) = route_indices.device_ptr(&ctx.stream);
            let (count_ptr, _count_guard) = local_counts.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_count_local_experts_cuda(
                    idx_ptr as *const i32,
                    count_ptr as *mut i32,
                    hidden.seq_len as i32,
                    config.num_experts_per_tok as i32,
                    local_expert_start_i32,
                    experts_per_rank_i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 local route count failed: {err}"))?;
            }
        }
        dsv4_moe_trace_end(
            ctx,
            "ffn_route_count_kernel",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;
        let trace = dsv4_moe_trace_begin(ctx)?;
        let counts_host = ctx
            .stream
            .clone_dtoh(&local_counts)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 local route count D2H failed: {err}"))?;
        dsv4_moe_trace_end(ctx, "ffn_route_count_d2h", layer_idx, hidden.seq_len, trace)?;
        let mut offsets_host = Vec::with_capacity(ep.experts_per_rank);
        let mut total_local_routes = 0usize;
        for &count in &counts_host {
            ensure!(
                count >= 0,
                "DeepSeek V4 local route count kernel returned negative count {count}"
            );
            offsets_host.push(
                i32::try_from(total_local_routes).map_err(|_| {
                    anyhow::anyhow!("DeepSeek V4 packed route offset overflows i32")
                })?,
            );
            total_local_routes += usize::try_from(count)
                .map_err(|_| anyhow::anyhow!("DeepSeek V4 local route count overflows usize"))?;
        }

        let mut out = HiddenStates::zeros(ctx, hidden.hidden_dim, hidden.seq_len)?;
        if total_local_routes == 0 {
            return Ok(out);
        }

        let trace = dsv4_moe_trace_begin(ctx)?;
        let offsets_gpu = ctx
            .stream
            .clone_htod(&offsets_host)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 local route offsets H2D failed: {err}"))?;
        let mut pack_cursors = ctx
            .stream
            .alloc_zeros::<i32>(ep.experts_per_rank)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 local route cursor alloc failed: {err}"))?;
        let mut packed_hidden = HiddenStates::zeros(ctx, hidden.hidden_dim, total_local_routes)?;
        let mut packed_token = ctx
            .stream
            .alloc_zeros::<i32>(total_local_routes)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 packed token alloc failed: {err}"))?;
        let mut packed_weight = ctx
            .stream
            .alloc_zeros::<f32>(total_local_routes)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 packed weight alloc failed: {err}"))?;
        dsv4_moe_trace_end(
            ctx,
            "ffn_route_pack_setup",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;
        let trace = dsv4_moe_trace_begin(ctx)?;
        {
            let (hidden_ptr, _hidden_guard) = hidden.data.device_ptr(&ctx.stream);
            let (idx_ptr, _idx_guard) = route_indices.device_ptr(&ctx.stream);
            let (weight_ptr, _weight_guard) = route_weights.device_ptr(&ctx.stream);
            let (offset_ptr, _offset_guard) = offsets_gpu.device_ptr(&ctx.stream);
            let (cursor_ptr, _cursor_guard) = pack_cursors.device_ptr_mut(&ctx.stream);
            let (packed_hidden_ptr, _packed_hidden_guard) =
                packed_hidden.data.device_ptr_mut(&ctx.stream);
            let (packed_token_ptr, _packed_token_guard) = packed_token.device_ptr_mut(&ctx.stream);
            let (packed_weight_ptr, _packed_weight_guard) =
                packed_weight.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_pack_local_experts_cuda(
                    hidden_ptr as *const ffi::Half,
                    idx_ptr as *const i32,
                    weight_ptr as *const f32,
                    offset_ptr as *const i32,
                    cursor_ptr as *mut i32,
                    packed_hidden_ptr as *mut ffi::Half,
                    packed_token_ptr as *mut i32,
                    packed_weight_ptr as *mut f32,
                    hidden.seq_len as i32,
                    hidden.hidden_dim as i32,
                    config.num_experts_per_tok as i32,
                    local_expert_start_i32,
                    experts_per_rank_i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 local expert pack failed: {err}"))?;
            }
        }
        dsv4_moe_trace_end(
            ctx,
            "ffn_route_pack_kernel",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        for (local_expert_idx, expert) in self.experts.iter().enumerate() {
            let count = usize::try_from(counts_host[local_expert_idx])
                .map_err(|_| anyhow::anyhow!("DeepSeek V4 local route count overflows usize"))?;
            if count == 0 {
                continue;
            }
            let offset = usize::try_from(offsets_host[local_expert_idx])
                .map_err(|_| anyhow::anyhow!("DeepSeek V4 packed route offset overflows usize"))?;
            let elem_start = offset * hidden.hidden_dim;
            let elem_end = elem_start + count * hidden.hidden_dim;
            let mut expert_input = HiddenStates::zeros(ctx, hidden.hidden_dim, count)?;
            {
                let src = packed_hidden.data.slice(elem_start..elem_end);
                ctx.stream
                    .memcpy_dtod(&src, &mut expert_input.data)
                    .map_err(|err| anyhow::anyhow!("DeepSeek V4 expert input D2D failed: {err}"))?;
            }

            let expert_out = expert.forward(ctx, &expert_input, config.swiglu_limit)?;
            let (expert_ptr, _expert_guard) = expert_out.data.device_ptr(&ctx.stream);
            let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
            let (token_ptr, _token_guard) = packed_token.device_ptr(&ctx.stream);
            let (weight_ptr, _weight_guard) = packed_weight.device_ptr(&ctx.stream);
            unsafe {
                ffi::dsv4_scatter_packed_expert_cuda(
                    expert_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    token_ptr as *const i32,
                    weight_ptr as *const f32,
                    offsets_host[local_expert_idx],
                    counts_host[local_expert_idx],
                    hidden.hidden_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 packed expert scatter failed: {err}")
                })?;
            }
        }
        dsv4_moe_trace_end(ctx, "ffn_expert_loop", layer_idx, hidden.seq_len, trace)?;
        Ok(out)
    }

    fn forward_grouped_dsv4_experts_gpu(
        &self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        expert_hidden: &HiddenStates,
        expert_route_slot: &CudaSlice<i32>,
        expert_weight: &CudaSlice<f32>,
        active_experts: &[usize],
        active_offsets_host: &[i32],
        active_counts_host: &[i32],
        total_local_routes: usize,
        max_local_routes: usize,
        route_out: &mut HiddenStates,
        scratch_cache: &mut DeepseekMoeRuntimeCache,
    ) -> Result<bool> {
        if total_local_routes == 0 || max_local_routes == 0 || self.experts.is_empty() {
            return Ok(true);
        }
        if self.experts.is_empty()
            || self
                .experts
                .iter()
                .any(|expert| dsv4_grouped_format(&expert.w1).is_none())
            || self
                .experts
                .iter()
                .any(|expert| dsv4_grouped_format(&expert.w3).is_none())
            || self
                .experts
                .iter()
                .any(|expert| dsv4_grouped_format(&expert.w2).is_none())
        {
            return Ok(false);
        }

        let first = &self.experts[0];
        ensure!(
            first.w1.rows == first.w3.rows
                && first.w1.cols == expert_hidden.hidden_dim
                && first.w3.cols == expert_hidden.hidden_dim
                && first.w2.cols == first.w1.rows
                && first.w2.rows == route_out.hidden_dim,
            "DeepSeek V4 grouped expert shape mismatch"
        );
        ensure!(
            route_out.seq_len >= total_local_routes,
            "DeepSeek V4 grouped route_out rows {} smaller than routes {}",
            route_out.seq_len,
            total_local_routes
        );

        let scratch = scratch_cache.ensure_grouped_expert_scratch(
            ctx,
            route_out.hidden_dim,
            first.w1.rows,
            total_local_routes,
        )?;
        scratch.gate.seq_len = total_local_routes;
        scratch.up.seq_len = total_local_routes;
        scratch.act.seq_len = total_local_routes;
        scratch.out.seq_len = total_local_routes;
        let active_experts_i32 = active_experts
            .iter()
            .map(|&idx| {
                i32::try_from(idx)
                    .map_err(|_| anyhow::anyhow!("DeepSeek V4 active expert index overflows i32"))
            })
            .collect::<Result<Vec<_>>>()?;
        {
            let active = scratch.ensure_active_scratch(ctx, active_experts.len())?;
            ctx.stream
                .memcpy_htod(&active_experts_i32, &mut active.indices)
                .map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 active expert index H2D failed: {err}")
                })?;
            ctx.stream
                .memcpy_htod(active_offsets_host, &mut active.offsets)
                .map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 active expert offset H2D failed: {err}")
                })?;
            ctx.stream
                .memcpy_htod(active_counts_host, &mut active.counts)
                .map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 active expert count H2D failed: {err}")
                })?;
        }
        let active = scratch
            .active
            .as_ref()
            .expect("DeepSeek V4 grouped active scratch allocated");

        let paired_gate_up = {
            let w1_ptrs = dsv4_ensure_grouped_weight_ptrs(
                ctx,
                &self.experts,
                &mut scratch.w1_ptrs,
                |expert| &expert.w1,
            )?;
            let w3_ptrs = dsv4_ensure_grouped_weight_ptrs(
                ctx,
                &self.experts,
                &mut scratch.w3_ptrs,
                |expert| &expert.w3,
            )?;
            dsv4_run_grouped_block_scaled_gemv_pair(
                ctx,
                w1_ptrs,
                w3_ptrs,
                expert_hidden,
                &mut scratch.gate,
                &mut scratch.up,
                &active.offsets,
                &active.counts,
                Some(&active.indices),
                active_experts.len(),
                max_local_routes,
            )?
        };
        if !paired_gate_up {
            {
                let w1_ptrs = dsv4_ensure_grouped_weight_ptrs(
                    ctx,
                    &self.experts,
                    &mut scratch.w1_ptrs,
                    |expert| &expert.w1,
                )?;
                dsv4_run_grouped_block_scaled_gemv(
                    ctx,
                    w1_ptrs,
                    expert_hidden,
                    &mut scratch.gate,
                    &active.offsets,
                    &active.counts,
                    Some(&active.indices),
                    active_experts.len(),
                    max_local_routes,
                )?;
            }
            {
                let w3_ptrs = dsv4_ensure_grouped_weight_ptrs(
                    ctx,
                    &self.experts,
                    &mut scratch.w3_ptrs,
                    |expert| &expert.w3,
                )?;
                dsv4_run_grouped_block_scaled_gemv(
                    ctx,
                    w3_ptrs,
                    expert_hidden,
                    &mut scratch.up,
                    &active.offsets,
                    &active.counts,
                    Some(&active.indices),
                    active_experts.len(),
                    max_local_routes,
                )?;
            }
        }
        ops::dsv4_swiglu_clamped_batch_into(
            ctx,
            &scratch.gate,
            &scratch.up,
            &mut scratch.act,
            config.swiglu_limit,
        )?;
        {
            let w2_ptrs = dsv4_ensure_grouped_weight_ptrs(
                ctx,
                &self.experts,
                &mut scratch.w2_ptrs,
                |expert| &expert.w2,
            )?;
            dsv4_run_grouped_block_scaled_gemv(
                ctx,
                w2_ptrs,
                &scratch.act,
                &mut scratch.out,
                &active.offsets,
                &active.counts,
                Some(&active.indices),
                active_experts.len(),
                max_local_routes,
            )?;
        }

        let (expert_ptr, _expert_guard) = scratch.out.data.device_ptr(&ctx.stream);
        let (route_out_ptr, _route_guard) = route_out.data.device_ptr_mut(&ctx.stream);
        let (route_slot_ptr, _route_slot_guard) = expert_route_slot.device_ptr(&ctx.stream);
        let (weight_ptr, _weight_guard) = expert_weight.device_ptr(&ctx.stream);
        unsafe {
            ffi::dsv4_scatter_all_route_slots_cuda(
                expert_ptr as *const ffi::Half,
                route_out_ptr as *mut ffi::Half,
                route_slot_ptr as *const i32,
                weight_ptr as *const f32,
                total_local_routes as i32,
                route_out.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 grouped route scatter failed: {err}"))?;
        }
        Ok(true)
    }

    #[cfg(feature = "nccl")]
    pub(super) fn forward_deepep_routed_gpu(
        &self,
        ctx: &DeviceContext,
        comm: &LayerCommunicator,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
        mut moe_scratch: Option<&mut DeepseekMoeRuntimeCache>,
    ) -> Result<HiddenStates> {
        ensure!(
            ep.world_size > 1,
            "DeepSeek V4 DeepEP-style MoE path requires ep_world_size > 1"
        );
        ensure!(
            token_ids.len() == hidden.seq_len,
            "DeepSeek V4 GPU route token count {} does not match hidden seq_len {}",
            token_ids.len(),
            hidden.seq_len
        );
        ensure!(
            self.gate_weight.rows == config.n_routed_experts
                && self.gate_weight.cols == hidden.hidden_dim,
            "DeepSeek V4 GPU gate shape mismatch: gate={}x{} hidden_dim={} n_routed_experts={}",
            self.gate_weight.rows,
            self.gate_weight.cols,
            hidden.hidden_dim,
            config.n_routed_experts
        );
        ensure!(
            ep.experts_per_rank == self.experts.len(),
            "DeepSeek V4 GPU route expects {} local experts but block loaded {}",
            ep.experts_per_rank,
            self.experts.len()
        );

        let trace = dsv4_moe_trace_begin(ctx)?;
        let logits = ops::gemm(ctx, &self.gate_weight, hidden)?;
        dsv4_moe_trace_end(
            ctx,
            "ffn_deepep_route_logits",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        let has_moe_scratch = moe_scratch.is_some();
        let mut send_route_scratch_slot = if let Some(cache) = moe_scratch.as_deref_mut() {
            cache.send_route.take()
        } else {
            None
        };
        let mut dispatch_scratch = if let Some(cache) = moe_scratch.as_deref_mut() {
            Some(cache.ensure_dispatch_scratch(
                ctx,
                hidden.hidden_dim,
                hidden.seq_len,
                config.num_experts_per_tok,
                ep.world_size,
                ep.experts_per_rank,
            )?)
        } else {
            None
        };
        let token_ids_gpu_owned: CudaSlice<u32>;
        let mut route_indices_owned: CudaSlice<i32>;
        let mut route_weights_owned: CudaSlice<f32>;
        let mut send_rank_counts_owned: CudaSlice<i32>;
        let mut send_offsets_gpu_owned: CudaSlice<i32>;
        let mut rank_cursors_owned: CudaSlice<i32>;
        let mut send_hidden_owned: HiddenStates;
        let mut send_token_owned: Option<CudaSlice<i32>> = None;
        let mut send_route_slot_owned: Option<CudaSlice<i32>> = None;
        let mut send_meta_owned: CudaSlice<i32>;
        let mut all_rank_counts_owned: CudaSlice<i32>;
        let mut recv_rank_counts_owned: CudaSlice<i32>;
        let mut local_counts_owned: CudaSlice<i32>;
        let mut local_offsets_gpu_owned: CudaSlice<i32>;
        let mut local_cursors_owned: CudaSlice<i32>;
        let (
            token_ids_gpu,
            route_indices,
            route_weights,
            send_rank_counts,
            send_offsets_gpu,
            rank_cursors,
            send_hidden,
            send_meta,
            all_rank_counts,
            recv_rank_counts,
            local_counts,
            local_offsets_gpu,
            local_cursors,
        ) = if let Some(dispatch) = dispatch_scratch.as_mut() {
            let dispatch = &mut **dispatch;
            ctx.stream
                .memcpy_htod(token_ids, &mut dispatch.token_ids)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 token ids H2D failed: {err}"))?;
            dispatch.send_hidden.seq_len = hidden.seq_len * config.num_experts_per_tok;
            (
                &dispatch.token_ids,
                &mut dispatch.route_indices,
                &mut dispatch.route_weights,
                &mut dispatch.send_rank_counts,
                &mut dispatch.send_rank_offsets,
                &mut dispatch.rank_cursors,
                &mut dispatch.send_hidden,
                &mut dispatch.send_meta,
                &mut dispatch.all_rank_counts,
                &mut dispatch.recv_rank_counts,
                &mut dispatch.local_counts,
                &mut dispatch.local_offsets,
                &mut dispatch.local_cursors,
            )
        } else {
            token_ids_gpu_owned = ctx
                .stream
                .clone_htod(token_ids)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 token ids H2D failed: {err}"))?;
            route_indices_owned = ctx
                .stream
                .alloc_zeros::<i32>(hidden.seq_len * config.num_experts_per_tok)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 route index alloc failed: {err}"))?;
            route_weights_owned = ctx
                .stream
                .alloc_zeros::<f32>(hidden.seq_len * config.num_experts_per_tok)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 route weight alloc failed: {err}"))?;
            send_rank_counts_owned = ctx
                .stream
                .alloc_zeros::<i32>(ep.world_size)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 rank count alloc failed: {err}"))?;
            send_offsets_gpu_owned =
                ctx.stream
                    .alloc_zeros::<i32>(ep.world_size)
                    .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 rank route offset alloc failed: {err}")
                    })?;
            rank_cursors_owned = ctx
                .stream
                .alloc_zeros::<i32>(ep.world_size)
                .map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 rank route cursor alloc failed: {err}")
                })?;
            send_hidden_owned = HiddenStates::zeros(
                ctx,
                hidden.hidden_dim,
                hidden.seq_len * config.num_experts_per_tok,
            )?;
            send_meta_owned = ctx
                .stream
                .alloc_zeros::<i32>(hidden.seq_len * config.num_experts_per_tok * 3)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 send meta alloc failed: {err}"))?;
            all_rank_counts_owned = ctx
                .stream
                .alloc_zeros::<i32>(ep.world_size * ep.world_size)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 all rank count alloc failed: {err}"))?;
            recv_rank_counts_owned =
                ctx.stream
                    .alloc_zeros::<i32>(ep.world_size)
                    .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 recv rank count alloc failed: {err}")
                    })?;
            local_counts_owned =
                ctx.stream
                    .alloc_zeros::<i32>(ep.experts_per_rank)
                    .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 recv local count alloc failed: {err}")
                    })?;
            local_offsets_gpu_owned =
                ctx.stream
                    .alloc_zeros::<i32>(ep.experts_per_rank)
                    .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 recv local offsets alloc failed: {err}")
                    })?;
            local_cursors_owned =
                ctx.stream
                    .alloc_zeros::<i32>(ep.experts_per_rank)
                    .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 recv local cursor alloc failed: {err}")
                    })?;
            (
                &token_ids_gpu_owned,
                &mut route_indices_owned,
                &mut route_weights_owned,
                &mut send_rank_counts_owned,
                &mut send_offsets_gpu_owned,
                &mut rank_cursors_owned,
                &mut send_hidden_owned,
                &mut send_meta_owned,
                &mut all_rank_counts_owned,
                &mut recv_rank_counts_owned,
                &mut local_counts_owned,
                &mut local_offsets_gpu_owned,
                &mut local_cursors_owned,
            )
        };
        dsv4_moe_trace_end(
            ctx,
            "ffn_deepep_route_setup",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;

        let routing_kind = match config.moe_routing_kind(layer_idx) {
            DeepSeekV4MoeRoutingKind::Hash => 0,
            DeepSeekV4MoeRoutingKind::LearnedBias => 1,
        };
        let scoring_kind = match config.scoring_func.as_str() {
            "softmax" => 0,
            "sigmoid" => 1,
            "sqrtsoftplus" => 2,
            other => bail!("unsupported DSV4 GPU router scoring_func `{other}`"),
        };
        if routing_kind == 0 {
            ensure!(
                self.gate_tid2eid.is_some(),
                "hash-routed DeepSeek V4 MoE layer missing tid2eid"
            );
        } else {
            ensure!(
                self.gate_bias.is_some(),
                "bias-routed DeepSeek V4 MoE layer missing gate bias"
            );
        }

        let trace = dsv4_moe_trace_begin(ctx)?;
        {
            let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
            let bias_guard;
            let bias_ptr = if let Some(bias) = self.gate_bias.as_ref() {
                let (ptr, guard) = bias.data.device_ptr(&ctx.stream);
                bias_guard = Some(guard);
                ptr as *const ffi::Half
            } else {
                bias_guard = None;
                std::ptr::null()
            };
            let tid_guard;
            let tid_ptr = if let Some(tid2eid) = self.gate_tid2eid.as_ref() {
                let (ptr, guard) = tid2eid.device_ptr(&ctx.stream);
                tid_guard = Some(guard);
                ptr as *const i64
            } else {
                tid_guard = None;
                std::ptr::null()
            };
            let (token_ptr, _token_guard) = token_ids_gpu.device_ptr(&ctx.stream);
            let (idx_ptr, _idx_guard) = route_indices.device_ptr_mut(&ctx.stream);
            let (weight_ptr, _weight_guard) = route_weights.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_route_cuda(
                    logits_ptr as *const ffi::Half,
                    bias_ptr,
                    tid_ptr,
                    token_ptr as *const u32,
                    idx_ptr as *mut i32,
                    weight_ptr as *mut f32,
                    hidden.seq_len as i32,
                    config.n_routed_experts as i32,
                    config.num_experts_per_tok as i32,
                    routing_kind,
                    scoring_kind,
                    config.routed_scaling_factor,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU router failed: {err}"))?;
            }
            drop(bias_guard);
            drop(tid_guard);
        }
        dsv4_moe_trace_end(
            ctx,
            "ffn_deepep_route_select",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        let experts_per_rank_i32 = i32::try_from(ep.experts_per_rank)
            .map_err(|_| anyhow::anyhow!("DeepSeek V4 experts_per_rank overflows i32"))?;
        let ep_world_i32 = i32::try_from(ep.world_size)
            .map_err(|_| anyhow::anyhow!("DeepSeek V4 ep_world_size overflows i32"))?;
        dsv4_zero_i32_slice(ctx, send_rank_counts, ep.world_size)?;
        {
            let (idx_ptr, _idx_guard) = route_indices.device_ptr(&ctx.stream);
            let (count_ptr, _count_guard) = send_rank_counts.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_count_expert_ranks_cuda(
                    idx_ptr as *const i32,
                    count_ptr as *mut i32,
                    hidden.seq_len as i32,
                    config.num_experts_per_tok as i32,
                    experts_per_rank_i32,
                    ep_world_i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 rank route count failed: {err}"))?;
            }
        }
        let send_rank_counts_host = ctx
            .stream
            .clone_dtoh(send_rank_counts)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 rank route count D2H failed: {err}"))?;
        let (send_rank_offsets_i32, total_send_routes) =
            dsv4_counts_to_offsets_i32(&send_rank_counts_host, "send_rank_counts")?;
        let send_rank_offsets = dsv4_offsets_to_usize(&send_rank_offsets_i32)?;
        let send_rank_counts_usize =
            dsv4_counts_to_usize(&send_rank_counts_host, "send_rank_counts")?;
        dsv4_moe_trace_end(
            ctx,
            "ffn_deepep_count_by_rank",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        ctx.stream
            .memcpy_htod(&send_rank_offsets_i32, send_offsets_gpu)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 rank route offsets H2D failed: {err}"))?;
        dsv4_zero_i32_slice(ctx, rank_cursors, ep.world_size)?;
        send_hidden.seq_len = total_send_routes;
        let send_route_capacity = hidden.seq_len * config.num_experts_per_tok;
        let send_route_scratch = if send_route_scratch_slot.is_some() || has_moe_scratch {
            Some(ensure_send_route_scratch(
                &mut send_route_scratch_slot,
                ctx,
                send_route_capacity,
            )?)
        } else {
            None
        };
        if send_route_scratch.is_none() {
            send_token_owned = Some(
                ctx.stream
                    .alloc_zeros::<i32>(send_route_capacity)
                    .map_err(|err| anyhow::anyhow!("DeepSeek V4 send token alloc failed: {err}"))?,
            );
            send_route_slot_owned = Some(
                ctx.stream
                    .alloc_zeros::<i32>(send_route_capacity)
                    .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 send route slot alloc failed: {err}")
                    })?,
            );
        }
        let (send_token, send_route_slot) = if let Some(scratch) = send_route_scratch {
            (&mut scratch.send_token, &mut scratch.send_route_slot)
        } else {
            (
                send_token_owned
                    .as_mut()
                    .expect("DeepSeek V4 send token fallback allocated"),
                send_route_slot_owned
                    .as_mut()
                    .expect("DeepSeek V4 send route-slot fallback allocated"),
            )
        };
        {
            let (hidden_ptr, _hidden_guard) = hidden.data.device_ptr(&ctx.stream);
            let (idx_ptr, _idx_guard) = route_indices.device_ptr(&ctx.stream);
            let (weight_ptr, _weight_guard) = route_weights.device_ptr(&ctx.stream);
            let (offset_ptr, _offset_guard) = send_offsets_gpu.device_ptr(&ctx.stream);
            let (cursor_ptr, _cursor_guard) = rank_cursors.device_ptr_mut(&ctx.stream);
            let (packed_hidden_ptr, _packed_hidden_guard) =
                send_hidden.data.device_ptr_mut(&ctx.stream);
            let (token_ptr, _token_guard) = send_token.device_ptr_mut(&ctx.stream);
            let (route_slot_ptr, _route_slot_guard) = send_route_slot.device_ptr_mut(&ctx.stream);
            let (meta_ptr, _meta_guard) = send_meta.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_pack_expert_ranks_cuda(
                    hidden_ptr as *const ffi::Half,
                    idx_ptr as *const i32,
                    weight_ptr as *const f32,
                    offset_ptr as *const i32,
                    cursor_ptr as *mut i32,
                    packed_hidden_ptr as *mut ffi::Half,
                    token_ptr as *mut i32,
                    route_slot_ptr as *mut i32,
                    meta_ptr as *mut i32,
                    hidden.seq_len as i32,
                    hidden.hidden_dim as i32,
                    config.num_experts_per_tok as i32,
                    experts_per_rank_i32,
                    ep_world_i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 rank route pack failed: {err}"))?;
            }
        }
        dsv4_moe_trace_end(
            ctx,
            "ffn_deepep_pack_by_rank",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        let peer_offsets: Vec<usize> = (0..ep.world_size).collect();
        let one_per_peer = vec![1usize; ep.world_size];
        let recv_rank_counts_host = match dsv4_count_exchange_mode()? {
            Dsv4CountExchangeMode::AllGather => {
                comm.moe_all_gather_i32(send_rank_counts, ep.world_size, all_rank_counts)?;
                let all_counts_host = ctx.stream.clone_dtoh(all_rank_counts).map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 all rank count D2H failed: {err}")
                })?;
                (0..ep.world_size)
                    .map(|peer| all_counts_host[peer * ep.world_size + ep.rank])
                    .collect::<Vec<_>>()
            }
            Dsv4CountExchangeMode::SendRecv => {
                comm.moe_grouped_send_recv_i32(
                    send_rank_counts,
                    &peer_offsets,
                    &one_per_peer,
                    recv_rank_counts,
                    &peer_offsets,
                    &one_per_peer,
                )?;
                ctx.stream.clone_dtoh(recv_rank_counts).map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 recv rank count D2H failed: {err}")
                })?
            }
        };
        let (recv_rank_offsets_i32, total_recv_routes) =
            dsv4_counts_to_offsets_i32(&recv_rank_counts_host, "recv_rank_counts")?;
        let recv_rank_offsets = dsv4_offsets_to_usize(&recv_rank_offsets_i32)?;
        let recv_rank_counts_usize =
            dsv4_counts_to_usize(&recv_rank_counts_host, "recv_rank_counts")?;
        dsv4_moe_trace_end(
            ctx,
            "ffn_deepep_count_exchange",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        let send_hidden_offsets = dsv4_scale_usize(&send_rank_offsets, hidden.hidden_dim)?;
        let send_hidden_counts = dsv4_scale_usize(&send_rank_counts_usize, hidden.hidden_dim)?;
        let recv_hidden_offsets = dsv4_scale_usize(&recv_rank_offsets, hidden.hidden_dim)?;
        let recv_hidden_counts = dsv4_scale_usize(&recv_rank_counts_usize, hidden.hidden_dim)?;
        let send_meta_offsets = dsv4_scale_usize(&send_rank_offsets, 3)?;
        let send_meta_counts = dsv4_scale_usize(&send_rank_counts_usize, 3)?;
        let recv_meta_offsets = dsv4_scale_usize(&recv_rank_offsets, 3)?;
        let recv_meta_counts = dsv4_scale_usize(&recv_rank_counts_usize, 3)?;
        let mut recv_hidden = HiddenStates::zeros(ctx, hidden.hidden_dim, total_recv_routes)?;
        let mut recv_meta = ctx
            .stream
            .alloc_zeros::<i32>(total_recv_routes * 3)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 recv meta alloc failed: {err}"))?;
        comm.moe_grouped_send_recv_bf16(
            &send_hidden.data,
            &send_hidden_offsets,
            &send_hidden_counts,
            &mut recv_hidden.data,
            &recv_hidden_offsets,
            &recv_hidden_counts,
        )?;
        comm.moe_grouped_send_recv_i32(
            send_meta,
            &send_meta_offsets,
            &send_meta_counts,
            &mut recv_meta,
            &recv_meta_offsets,
            &recv_meta_counts,
        )?;
        dsv4_moe_trace_end(ctx, "ffn_deepep_dispatch", layer_idx, hidden.seq_len, trace)?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        let route_out = if total_recv_routes == 0 {
            HiddenStates::zeros(ctx, hidden.hidden_dim, 0)?
        } else {
            let local_expert_start = ep.local_expert_range().start;
            let local_expert_start_i32 = i32::try_from(local_expert_start)
                .map_err(|_| anyhow::anyhow!("DeepSeek V4 local expert start overflows i32"))?;
            dsv4_zero_i32_slice(ctx, local_counts, ep.experts_per_rank)?;
            {
                let (meta_ptr, _meta_guard) = recv_meta.device_ptr(&ctx.stream);
                let (count_ptr, _count_guard) = local_counts.device_ptr_mut(&ctx.stream);
                unsafe {
                    ffi::dsv4_count_packed_local_experts_cuda(
                        meta_ptr as *const i32,
                        count_ptr as *mut i32,
                        total_recv_routes as i32,
                        local_expert_start_i32,
                        experts_per_rank_i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 recv local expert count failed: {err}")
                    })?;
                }
            }
            let local_counts_host = ctx
                .stream
                .clone_dtoh(local_counts)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 recv local count D2H failed: {err}"))?;
            let (local_offsets_i32, total_local_routes) =
                dsv4_counts_to_offsets_i32(&local_counts_host, "recv_local_counts")?;
            let local_offsets = dsv4_offsets_to_usize(&local_offsets_i32)?;
            let local_counts_usize = dsv4_counts_to_usize(&local_counts_host, "recv_local_counts")?;
            let max_local_routes = local_counts_usize.iter().copied().max().unwrap_or(0);
            ctx.stream
                .memcpy_htod(&local_offsets_i32, local_offsets_gpu)
                .map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 recv local offsets H2D failed: {err}")
                })?;
            dsv4_zero_i32_slice(ctx, local_cursors, ep.experts_per_rank)?;
            let mut expert_hidden =
                HiddenStates::zeros(ctx, hidden.hidden_dim, total_local_routes)?;
            let mut expert_weight = ctx
                .stream
                .alloc_zeros::<f32>(total_local_routes)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 expert weight alloc failed: {err}"))?;
            let mut expert_route_slot =
                ctx.stream
                    .alloc_zeros::<i32>(total_local_routes)
                    .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 expert route-slot alloc failed: {err}")
                    })?;
            {
                let (recv_hidden_ptr, _recv_hidden_guard) =
                    recv_hidden.data.device_ptr(&ctx.stream);
                let (recv_meta_ptr, _recv_meta_guard) = recv_meta.device_ptr(&ctx.stream);
                let (offset_ptr, _offset_guard) = local_offsets_gpu.device_ptr(&ctx.stream);
                let (cursor_ptr, _cursor_guard) = local_cursors.device_ptr_mut(&ctx.stream);
                let (expert_hidden_ptr, _expert_hidden_guard) =
                    expert_hidden.data.device_ptr_mut(&ctx.stream);
                let (expert_weight_ptr, _expert_weight_guard) =
                    expert_weight.device_ptr_mut(&ctx.stream);
                let (route_slot_ptr, _route_slot_guard) =
                    expert_route_slot.device_ptr_mut(&ctx.stream);
                unsafe {
                    ffi::dsv4_pack_received_experts_cuda(
                        recv_hidden_ptr as *const ffi::Half,
                        recv_meta_ptr as *const i32,
                        offset_ptr as *const i32,
                        cursor_ptr as *mut i32,
                        expert_hidden_ptr as *mut ffi::Half,
                        expert_weight_ptr as *mut f32,
                        route_slot_ptr as *mut i32,
                        total_recv_routes as i32,
                        hidden.hidden_dim as i32,
                        local_expert_start_i32,
                        experts_per_rank_i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 recv local expert pack failed: {err}")
                    })?;
                }
            }

            let mut route_out = HiddenStates::zeros(ctx, hidden.hidden_dim, total_recv_routes)?;
            let grouped_done = if dsv4_grouped_experts_enabled()? {
                if let Some(scratch_cache) = moe_scratch.as_deref_mut() {
                    let active_experts = local_counts_usize
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &count)| (count > 0).then_some(idx))
                        .collect::<Vec<_>>();
                    if active_experts.is_empty() {
                        true
                    } else {
                        let active_offsets_i32 = active_experts
                            .iter()
                            .map(|&idx| local_offsets_i32[idx])
                            .collect::<Vec<_>>();
                        let active_counts_i32 = active_experts
                            .iter()
                            .map(|&idx| local_counts_host[idx])
                            .collect::<Vec<_>>();
                        self.forward_grouped_dsv4_experts_gpu(
                            ctx,
                            config,
                            &expert_hidden,
                            &expert_route_slot,
                            &expert_weight,
                            &active_experts,
                            &active_offsets_i32,
                            &active_counts_i32,
                            total_local_routes,
                            max_local_routes,
                            &mut route_out,
                            scratch_cache,
                        )?
                    }
                } else {
                    false
                }
            } else {
                false
            };
            if !grouped_done {
                for (local_expert_idx, expert) in self.experts.iter().enumerate() {
                    let count = local_counts_usize[local_expert_idx];
                    if count == 0 {
                        continue;
                    }
                    let offset = local_offsets[local_expert_idx];
                    let elem_start = offset * hidden.hidden_dim;
                    let elem_end = elem_start + count * hidden.hidden_dim;
                    let expert_out_ref;
                    let expert_out_owned;
                    let expert_out = if let Some(scratch_cache) = moe_scratch.as_deref_mut() {
                        let scratch = scratch_cache.ensure_expert_scratch(
                            ctx,
                            hidden.hidden_dim,
                            expert.w1.rows,
                            expert.w2.rows,
                            count,
                        )?;
                        scratch.input.seq_len = count;
                        let src = expert_hidden.data.slice(elem_start..elem_end);
                        let mut dst = scratch.input.data.slice_mut(0..count * hidden.hidden_dim);
                        ctx.stream.memcpy_dtod(&src, &mut dst).map_err(|err| {
                            anyhow::anyhow!(
                                "DeepSeek V4 recv expert input scratch D2D failed: {err}"
                            )
                        })?;
                        expert_out_ref =
                            expert.forward_scratch_input(ctx, config.swiglu_limit, scratch)?;
                        expert_out_ref
                    } else {
                        let mut expert_input = HiddenStates::zeros(ctx, hidden.hidden_dim, count)?;
                        {
                            let src = expert_hidden.data.slice(elem_start..elem_end);
                            ctx.stream
                                .memcpy_dtod(&src, &mut expert_input.data)
                                .map_err(|err| {
                                    anyhow::anyhow!(
                                        "DeepSeek V4 recv expert input D2D failed: {err}"
                                    )
                                })?;
                        }
                        expert_out_owned =
                            expert.forward(ctx, &expert_input, config.swiglu_limit)?;
                        &expert_out_owned
                    };
                    let (expert_ptr, _expert_guard) = expert_out.data.device_ptr(&ctx.stream);
                    let (route_out_ptr, _route_guard) = route_out.data.device_ptr_mut(&ctx.stream);
                    let (route_slot_ptr, _route_slot_guard) =
                        expert_route_slot.device_ptr(&ctx.stream);
                    let (weight_ptr, _weight_guard) = expert_weight.device_ptr(&ctx.stream);
                    unsafe {
                        ffi::dsv4_scatter_packed_route_slot_cuda(
                            expert_ptr as *const ffi::Half,
                            route_out_ptr as *mut ffi::Half,
                            route_slot_ptr as *const i32,
                            weight_ptr as *const f32,
                            local_offsets_i32[local_expert_idx],
                            local_counts_host[local_expert_idx],
                            hidden.hidden_dim as i32,
                            ctx.stream.cu_stream(),
                        )
                        .result()
                        .map_err(|err| {
                            anyhow::anyhow!("DeepSeek V4 recv expert route scatter failed: {err}")
                        })?;
                    }
                }
            }
            route_out
        };
        dsv4_moe_trace_end(
            ctx,
            "ffn_deepep_local_experts",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        let route_slot_capacity = hidden.seq_len * config.num_experts_per_tok;
        let route_combine_capacity = total_send_routes
            .max(total_recv_routes)
            .max(route_slot_capacity);
        let mut out = HiddenStates::zeros(ctx, hidden.hidden_dim, hidden.seq_len)?;
        let run_combine_kernel = |combine_recv: &HiddenStates,
                                  route_slot_out: Option<&mut HiddenStates>,
                                  out: &mut HiddenStates|
         -> Result<()> {
            let combine_kernel_trace = dsv4_moe_trace_begin(ctx)?;
            if hidden.seq_len > 1 {
                let mut route_slot_out_owned;
                let route_slot_out = match route_slot_out {
                    Some(scratch) => {
                        scratch.seq_len = route_slot_capacity;
                        scratch
                    }
                    None => {
                        route_slot_out_owned =
                            HiddenStates::zeros(ctx, hidden.hidden_dim, route_slot_capacity)?;
                        &mut route_slot_out_owned
                    }
                };
                {
                    let (packed_ptr, _packed_guard) = combine_recv.data.device_ptr(&ctx.stream);
                    let (slot_out_ptr, _slot_out_guard) =
                        route_slot_out.data.device_ptr_mut(&ctx.stream);
                    let (route_slot_ptr, _route_slot_guard) =
                        send_route_slot.device_ptr(&ctx.stream);
                    unsafe {
                        ffi::dsv4_zero_bf16_cuda(
                            slot_out_ptr as *mut ffi::Half,
                            (route_slot_capacity * hidden.hidden_dim) as i32,
                            ctx.stream.cu_stream(),
                        )
                        .result()
                        .map_err(|err| {
                            anyhow::anyhow!("DeepSeek V4 route slot scratch zero failed: {err}")
                        })?;
                        ffi::dsv4_scatter_route_outputs_by_slot_cuda(
                            packed_ptr as *const ffi::Half,
                            slot_out_ptr as *mut ffi::Half,
                            route_slot_ptr as *const i32,
                            total_send_routes as i32,
                            hidden.hidden_dim as i32,
                            ctx.stream.cu_stream(),
                        )
                        .result()
                        .map_err(|err| {
                            anyhow::anyhow!("DeepSeek V4 route slot scatter failed: {err}")
                        })?;
                    }
                }
                let (slot_out_ptr, _slot_out_guard) = route_slot_out.data.device_ptr(&ctx.stream);
                let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
                unsafe {
                    ffi::dsv4_combine_route_slot_outputs_cuda(
                        slot_out_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        hidden.seq_len as i32,
                        config.num_experts_per_tok as i32,
                        hidden.hidden_dim as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 route-slot combine failed: {err}")
                    })?;
                }
            } else {
                let (route_out_ptr, _route_guard) = combine_recv.data.device_ptr(&ctx.stream);
                let (token_ptr, _token_guard) = send_token.device_ptr(&ctx.stream);
                let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
                unsafe {
                    ffi::dsv4_combine_route_outputs_cuda(
                        route_out_ptr as *const ffi::Half,
                        token_ptr as *const i32,
                        out_ptr as *mut ffi::Half,
                        hidden.seq_len as i32,
                        total_send_routes as i32,
                        hidden.hidden_dim as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|err| anyhow::anyhow!("DeepSeek V4 route combine failed: {err}"))?;
                }
            }
            dsv4_moe_trace_end(
                ctx,
                "ffn_deepep_combine_kernel",
                layer_idx,
                hidden.seq_len,
                combine_kernel_trace,
            )
        };
        let combine_mode = dsv4_combine_exchange_mode()?;
        if let Some(scratch_cache) = moe_scratch.as_deref_mut() {
            let scratch = scratch_cache.ensure_route_combine_scratch(
                ctx,
                hidden.hidden_dim,
                route_combine_capacity,
            )?;
            scratch.combine_recv.seq_len = total_send_routes;
            scratch.route_slot_out.seq_len = route_slot_capacity;
            let combine_exchange_trace = dsv4_moe_trace_begin(ctx)?;
            match combine_mode {
                Dsv4CombineExchangeMode::Bf16 => {
                    comm.moe_grouped_send_recv_bf16(
                        &route_out.data,
                        &recv_hidden_offsets,
                        &recv_hidden_counts,
                        &mut scratch.combine_recv.data,
                        &send_hidden_offsets,
                        &send_hidden_counts,
                    )?;
                }
                Dsv4CombineExchangeMode::Fp8 => {
                    dsv4_run_fp8_combine_exchange(
                        ctx,
                        comm,
                        &route_out,
                        &recv_hidden_offsets,
                        &recv_hidden_counts,
                        &recv_rank_offsets,
                        &recv_rank_counts_usize,
                        &send_hidden_offsets,
                        &send_hidden_counts,
                        &send_rank_offsets,
                        &send_rank_counts_usize,
                        total_recv_routes,
                        total_send_routes,
                        &mut scratch.combine_fp8_send,
                        &mut scratch.combine_fp8_recv,
                        &mut scratch.combine_scale_send,
                        &mut scratch.combine_scale_recv,
                        &mut scratch.combine_recv,
                    )?;
                }
            }
            dsv4_moe_trace_end(
                ctx,
                "ffn_deepep_combine_exchange",
                layer_idx,
                hidden.seq_len,
                combine_exchange_trace,
            )?;
            run_combine_kernel(
                &scratch.combine_recv,
                Some(&mut scratch.route_slot_out),
                &mut out,
            )?;
        } else {
            let mut combine_recv = HiddenStates::zeros(ctx, hidden.hidden_dim, total_send_routes)?;
            let combine_exchange_trace = dsv4_moe_trace_begin(ctx)?;
            match combine_mode {
                Dsv4CombineExchangeMode::Bf16 => {
                    comm.moe_grouped_send_recv_bf16(
                        &route_out.data,
                        &recv_hidden_offsets,
                        &recv_hidden_counts,
                        &mut combine_recv.data,
                        &send_hidden_offsets,
                        &send_hidden_counts,
                    )?;
                }
                Dsv4CombineExchangeMode::Fp8 => {
                    let mut combine_fp8_send = ctx
                        .stream
                        .alloc_zeros::<u8>(total_recv_routes * hidden.hidden_dim)
                        .map_err(|err| {
                            anyhow::anyhow!("DeepSeek V4 combine FP8 send alloc failed: {err}")
                        })?;
                    let mut combine_fp8_recv = ctx
                        .stream
                        .alloc_zeros::<u8>(total_send_routes * hidden.hidden_dim)
                        .map_err(|err| {
                            anyhow::anyhow!("DeepSeek V4 combine FP8 recv alloc failed: {err}")
                        })?;
                    let mut combine_scale_send = ctx
                        .stream
                        .alloc_zeros::<f32>(total_recv_routes)
                        .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 combine FP8 send-scale alloc failed: {err}")
                    })?;
                    let mut combine_scale_recv = ctx
                        .stream
                        .alloc_zeros::<f32>(total_send_routes)
                        .map_err(|err| {
                        anyhow::anyhow!("DeepSeek V4 combine FP8 recv-scale alloc failed: {err}")
                    })?;
                    dsv4_run_fp8_combine_exchange(
                        ctx,
                        comm,
                        &route_out,
                        &recv_hidden_offsets,
                        &recv_hidden_counts,
                        &recv_rank_offsets,
                        &recv_rank_counts_usize,
                        &send_hidden_offsets,
                        &send_hidden_counts,
                        &send_rank_offsets,
                        &send_rank_counts_usize,
                        total_recv_routes,
                        total_send_routes,
                        &mut combine_fp8_send,
                        &mut combine_fp8_recv,
                        &mut combine_scale_send,
                        &mut combine_scale_recv,
                        &mut combine_recv,
                    )?;
                }
            }
            dsv4_moe_trace_end(
                ctx,
                "ffn_deepep_combine_exchange",
                layer_idx,
                hidden.seq_len,
                combine_exchange_trace,
            )?;
            run_combine_kernel(&combine_recv, None, &mut out)?;
        }
        dsv4_moe_trace_end(ctx, "ffn_deepep_combine", layer_idx, hidden.seq_len, trace)?;
        if let Some(cache) = moe_scratch.as_deref_mut() {
            cache.send_route = send_route_scratch_slot.take();
        }
        Ok(out)
    }

    #[cfg(test)]
    fn route_local(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
    ) -> Result<LocalExpertRouting> {
        ensure!(
            token_ids.len() == hidden.seq_len,
            "DeepSeek V4 route token count {} does not match hidden seq_len {}",
            token_ids.len(),
            hidden.seq_len
        );
        ensure!(
            self.gate_weight.rows == config.n_routed_experts
                && self.gate_weight.cols == hidden.hidden_dim,
            "DeepSeek V4 gate shape mismatch: gate={}x{} hidden_dim={} n_routed_experts={}",
            self.gate_weight.rows,
            self.gate_weight.cols,
            hidden.hidden_dim,
            config.n_routed_experts
        );
        if let Some(bias) = &self.gate_bias {
            ensure!(
                bias.len == config.n_routed_experts,
                "DeepSeek V4 gate bias len {} does not match n_routed_experts {}",
                bias.len,
                config.n_routed_experts
            );
        }

        let logits = ops::gemm(ctx, &self.gate_weight, hidden)?;
        let logits_host = ctx.stream.clone_dtoh(&logits.data)?;
        let bias_host = self
            .gate_bias
            .as_ref()
            .map(|bias| ctx.stream.clone_dtoh(&bias.data))
            .transpose()?
            .map(|bias| {
                bias.into_iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>()
            });
        let mut routes = Vec::with_capacity(hidden.seq_len * config.num_experts_per_tok);

        for token_idx in 0..hidden.seq_len {
            let start = token_idx * logits.hidden_dim;
            let token_logits = logits_host[start..start + logits.hidden_dim]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            let scores = config.router_scores_from_logits(&token_logits)?;
            let hash_experts = match config.moe_routing_kind(layer_idx) {
                DeepSeekV4MoeRoutingKind::Hash => {
                    Some(self.hash_experts_for_token(ctx, config, token_ids[token_idx])?)
                }
                DeepSeekV4MoeRoutingKind::LearnedBias => None,
            };
            let token_routes = config.moe_routes_from_scores(
                layer_idx,
                token_idx,
                &scores,
                bias_host.as_deref(),
                hash_experts.as_deref(),
            )?;
            routes.extend(token_routes.into_iter().map(|route| ExpertRoute {
                token_idx: route.token_idx,
                expert_idx: route.expert_idx,
                weight: route.weight,
            }));
        }

        ep.localize_routing(&ExpertRoutingWeights::new(config.n_routed_experts, routes))
    }

    #[cfg(test)]
    fn hash_experts_for_token(
        &self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        token_id: u32,
    ) -> Result<Vec<usize>> {
        let table = self
            .gate_tid2eid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("hash-routed DeepSeek V4 MoE layer missing tid2eid"))?;
        ensure!(
            (token_id as usize) < config.vocab_size,
            "DeepSeek V4 token id {token_id} exceeds vocab_size {}",
            config.vocab_size
        );
        let start = token_id as usize * config.num_experts_per_tok;
        let end = start + config.num_experts_per_tok;
        ensure!(
            end <= table.len(),
            "DeepSeek V4 tid2eid table too short: need {} entries for token {}, have {}",
            end,
            token_id,
            table.len()
        );
        let experts_i64 = ctx.stream.clone_dtoh(&table.slice(start..end))?;
        experts_i64
            .into_iter()
            .map(|expert_idx| {
                ensure!(
                    expert_idx >= 0,
                    "DeepSeek V4 tid2eid contains negative expert id"
                );
                usize::try_from(expert_idx)
                    .map_err(|_| anyhow::anyhow!("DeepSeek V4 tid2eid expert id overflow"))
            })
            .collect()
    }
}

#[cfg(feature = "cuda")]
fn dsv4_moe_trace_enabled() -> bool {
    std::env::var("ARLE_DSV4_TRACE_LAYER")
        .ok()
        .is_some_and(|raw| !matches!(raw.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"))
}

#[cfg(feature = "cuda")]
fn dsv4_zero_i32_slice(ctx: &DeviceContext, slice: &mut CudaSlice<i32>, len: usize) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    ctx.stream
        .memset_zeros(&mut slice.slice_mut(0..len))
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 i32 scratch zero failed: {err}"))
}

#[cfg(feature = "cuda")]
fn dsv4_moe_trace_begin(ctx: &DeviceContext) -> Result<Instant> {
    if dsv4_moe_trace_enabled() {
        ctx.stream
            .synchronize()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 MoE trace pre-sync failed: {err}"))?;
    }
    Ok(Instant::now())
}

#[cfg(feature = "cuda")]
fn dsv4_moe_trace_end(
    ctx: &DeviceContext,
    phase: &str,
    layer_idx: usize,
    tokens: usize,
    started: Instant,
) -> Result<()> {
    if !dsv4_moe_trace_enabled() {
        return Ok(());
    }
    ctx.stream
        .synchronize()
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 MoE trace post-sync failed: {err}"))?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    info!(
        "dsv4_trace layer={} phase={} tokens={} elapsed_ms={:.3}",
        layer_idx, phase, tokens, elapsed_ms
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn dsv4_counts_to_offsets_i32(counts: &[i32], label: &str) -> Result<(Vec<i32>, usize)> {
    let mut offsets = Vec::with_capacity(counts.len());
    let mut total = 0usize;
    for &count in counts {
        ensure!(
            count >= 0,
            "DeepSeek V4 {label} contains negative count {count}"
        );
        offsets.push(
            i32::try_from(total)
                .map_err(|_| anyhow::anyhow!("DeepSeek V4 {label} offset overflows i32"))?,
        );
        total = total
            .checked_add(
                usize::try_from(count)
                    .map_err(|_| anyhow::anyhow!("DeepSeek V4 {label} count overflows usize"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 {label} total overflows usize"))?;
    }
    Ok((offsets, total))
}

#[cfg(feature = "cuda")]
fn dsv4_counts_to_usize(counts: &[i32], label: &str) -> Result<Vec<usize>> {
    counts
        .iter()
        .map(|&count| {
            ensure!(
                count >= 0,
                "DeepSeek V4 {label} contains negative count {count}"
            );
            usize::try_from(count)
                .map_err(|_| anyhow::anyhow!("DeepSeek V4 {label} count overflows usize"))
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn dsv4_offsets_to_usize(offsets: &[i32]) -> Result<Vec<usize>> {
    offsets
        .iter()
        .map(|&offset| {
            ensure!(
                offset >= 0,
                "DeepSeek V4 offset list contains negative offset {offset}"
            );
            usize::try_from(offset)
                .map_err(|_| anyhow::anyhow!("DeepSeek V4 offset overflows usize"))
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn dsv4_scale_usize(values: &[usize], factor: usize) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|&value| {
            value
                .checked_mul(factor)
                .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 scaled route extent overflows usize"))
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn dsv4_usize_to_i32(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| anyhow::anyhow!("DeepSeek V4 {label} overflows i32"))
}

#[cfg(feature = "cuda")]
#[cfg(test)]
fn hidden_token(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    token_idx: usize,
) -> Result<HiddenStates> {
    let token = ops::extract_vec(ctx, hidden, token_idx)?;
    Ok(HiddenStates {
        data: token.data,
        hidden_dim: hidden.hidden_dim,
        seq_len: 1,
    })
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::distributed::expert_state::{ExpertGroup, LocalExpertRoute};
    use half::bf16;

    fn bf16_vec(values: &[f32]) -> Vec<bf16> {
        values.iter().map(|&value| bf16::from_f32(value)).collect()
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn tiny_config() -> DeepSeekV4Config {
        DeepSeekV4Config::from_json_str(
            r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "torch_dtype": "bfloat16",
            "vocab_size": 16,
            "hidden_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "q_lora_rank": 1,
            "o_lora_rank": 1,
            "o_groups": 1,
            "qk_rope_head_dim": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 0,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 1,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": false,
            "scoring_func": "softmax",
            "topk_method": "noaux_tc",
            "index_n_heads": 1,
            "index_head_dim": 1,
            "index_topk": 1,
            "num_hash_layers": 0,
            "sliding_window": 4,
            "compress_ratios": [0],
            "compress_rope_theta": 160000.0,
            "hc_mult": 1,
            "hc_sinkhorn_iters": 1,
            "hc_eps": 1.0e-6,
            "num_nextn_predict_layers": 0,
            "max_position_embeddings": 16,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": 1.0,
                "original_max_position_embeddings": 16,
                "beta_fast": 32.0,
                "beta_slow": 1.0
            },
            "rms_norm_eps": 1.0e-6,
            "initializer_range": 0.02,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 1
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn expert_forward_runs_clamped_swiglu_on_gpu() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, -2.0, 0.5, 3.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };
        let expert = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0,
                ]),
                3,
                2,
            )?,
            w2: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, 0.5, //
                    0.0, 1.0, -1.0,
                ]),
                2,
                3,
            )?,
            w3: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    0.5, 0.0, //
                    0.0, -1.0, //
                    1.0, -1.0,
                ]),
                3,
                2,
            )?,
        };

        let out = expert.forward(&ctx, &hidden, 2.0)?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;

        let inputs = [[1.0_f32, -2.0_f32], [0.5_f32, 3.0_f32]];
        let mut expected = Vec::new();
        for x in inputs {
            let gate = [x[0], x[1], x[0] + x[1]];
            let up = [0.5 * x[0], -x[1], x[0] - x[1]];
            let act = [
                silu(gate[0].min(2.0)) * up[0].clamp(-2.0, 2.0),
                silu(gate[1].min(2.0)) * up[1].clamp(-2.0, 2.0),
                silu(gate[2].min(2.0)) * up[2].clamp(-2.0, 2.0),
            ];
            expected.push(act[0] + 0.5 * act[2]);
            expected.push(act[1] - act[2]);
        }

        for (idx, got) in out_host.iter().enumerate() {
            assert!(
                (got.to_f32() - expected[idx]).abs() < 0.05,
                "idx={idx} expected={} got={}",
                expected[idx],
                got.to_f32()
            );
        }
        Ok(())
    }

    #[test]
    fn moe_forward_local_routes_accumulates_ep_local_experts() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, -1.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };
        let expert0 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 2.0]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
        };
        let expert1 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[-1.0, 0.5]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
        };
        let block = DeepseekV4MoeBlock {
            gate_weight: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 0.0]), 1, 2)?,
            gate_bias: None,
            gate_tid2eid: None,
            experts: vec![expert0, expert1],
            shared_experts: None,
        };
        let routing = LocalExpertRouting {
            num_global_experts: 4,
            experts_per_rank: 2,
            routes: vec![
                LocalExpertRoute {
                    token_idx: 0,
                    global_expert_idx: 0,
                    local_expert_idx: 0,
                    weight: 0.25,
                },
                LocalExpertRoute {
                    token_idx: 0,
                    global_expert_idx: 1,
                    local_expert_idx: 1,
                    weight: 0.5,
                },
                LocalExpertRoute {
                    token_idx: 1,
                    global_expert_idx: 1,
                    local_expert_idx: 1,
                    weight: 1.0,
                },
            ],
        };

        let out = block.forward_local_routes(&ctx, &hidden, &routing, 10.0)?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;

        let e0_t0 = silu(1.0) * 1.0;
        let e1_t0 = silu(2.0) * 2.0;
        let e1_t1 = silu(-1.0) * -1.0;
        let expected = [
            0.25 * e0_t0 - 0.5 * e1_t0,
            0.25 * (2.0 * e0_t0) + 0.5 * (0.5 * e1_t0),
            -e1_t1,
            0.5 * e1_t1,
        ];

        for (idx, got) in out_host.iter().enumerate() {
            assert!(
                (got.to_f32() - expected[idx]).abs() < 0.05,
                "idx={idx} expected={} got={}",
                expected[idx],
                got.to_f32()
            );
        }
        Ok(())
    }

    #[test]
    fn moe_forward_routed_computes_gate_routes_and_localizes_ep() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let config = tiny_config();
        let ep = ExpertGroup::new(0, 2, config.n_routed_experts)?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 0.0, 0.0, 2.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };
        let expert0 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 2.0]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
        };
        let expert1 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[-1.0, 0.5]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
        };
        let block = DeepseekV4MoeBlock {
            gate_weight: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, //
                    0.0, 1.0, //
                    -1.0, 0.0, //
                    0.0, -1.0,
                ]),
                4,
                2,
            )?,
            gate_bias: Some(DeviceVec::from_host(
                &ctx,
                &bf16_vec(&[0.0, 0.0, 0.0, 0.0]),
            )?),
            gate_tid2eid: None,
            experts: vec![expert0, expert1],
            shared_experts: None,
        };

        let out = block.forward_routed(&ctx, 0, &config, &ep, &hidden, &[3, 4])?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;

        let token0_scores = config.router_scores_from_logits(&[1.0, 0.0, -1.0, 0.0])?;
        let token1_scores = config.router_scores_from_logits(&[0.0, 2.0, 0.0, -2.0])?;
        let e0_t0 = silu(1.0) * 1.0;
        let e1_t1 = silu(2.0) * 2.0;
        let expected = [
            token0_scores[0] * e0_t0,
            token0_scores[0] * (2.0 * e0_t0),
            -token1_scores[1] * e1_t1,
            token1_scores[1] * (0.5 * e1_t1),
        ];

        for (idx, got) in out_host.iter().enumerate() {
            assert!(
                (got.to_f32() - expected[idx]).abs() < 0.05,
                "idx={idx} expected={} got={}",
                expected[idx],
                got.to_f32()
            );
        }
        Ok(())
    }
}
