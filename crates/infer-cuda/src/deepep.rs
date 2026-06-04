//! DSv4 native DeepEP transport glue.
//!
//! This module owns the torch-free DeepEP `Buffer` lifecycle plus the small
//! NCCL byte all-gather used to exchange CUDA IPC handles. The MoE math stays in
//! `moe.rs`; this layer only dispatches token rows to expert-owner ranks and
//! combines local expert outputs back to the original token rows.

#![cfg(all(feature = "cuda", feature = "deepep"))]

use std::sync::Mutex;

use anyhow::{Result, anyhow, bail, ensure};
use cuda_kernels::prelude::{DeviceContext, HiddenStates};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::tp::TpRuntime;

pub(crate) struct DeepEpTransport {
    world_size: u32,
    buffer: Mutex<deepep_sys::Buffer>,
}

pub(crate) struct DeepEpDispatchScratch {
    pub recv_x: HiddenStates,
    pub recv_src_idx: CudaSlice<i32>,
    pub recv_topk_idx: CudaSlice<i64>,
    pub recv_topk_weights: CudaSlice<f32>,
    pub rank_prefix: CudaSlice<i32>,
    pub recv_channel_prefix: CudaSlice<i32>,
    pub send_head: CudaSlice<i32>,
    pub num_tokens_per_rank: CudaSlice<i32>,
    pub num_tokens_per_expert: CudaSlice<i32>,
    pub is_token_in_rank: CudaSlice<u8>,
    pub channel_prefix_matrix: CudaSlice<i32>,
    pub combined_topk_weights: CudaSlice<f32>,
    pub capacity_recv: usize,
}

impl DeepEpTransport {
    pub(crate) fn should_enable_from_env() -> bool {
        matches!(
            std::env::var("ARLE_DSV4_MOE_TRANSPORT").as_deref(),
            Ok("deepep" | "native-deepep" | "native_deepep")
        ) || matches!(
            std::env::var("ARLE_DSV4_MOE_BACKEND").as_deref(),
            Ok("deepep" | "native-deepep" | "native_deepep")
        )
    }

    pub(crate) fn maybe_boot(ctx: &DeviceContext, tp: &TpRuntime) -> Result<Option<Self>> {
        if !Self::should_enable_from_env() {
            return Ok(None);
        }
        ensure!(deepep_sys::is_native(), "deepep-sys was built as a stub");
        let cfg = tp.config();
        ensure!(
            cfg.world_size >= 2,
            "native DeepEP requires world_size >= 2, got {}",
            cfg.world_size
        );
        let rank = u32::try_from(cfg.rank)?;
        let world_size = u32::try_from(cfg.world_size)?;
        let mut buffer = deepep_sys::Buffer::new(rank, world_size)
            .map_err(|e| anyhow!("DeepEP Buffer::new failed: {e}"))?;
        let (local_handle, local_device_id) = buffer
            .local_ipc_handle()
            .map_err(|e| anyhow!("DeepEP local_ipc_handle failed: {e}"))?;

        let gathered_handles = tp
            .all_gather_bytes(ctx, &local_handle, deepep_sys::IPC_HANDLE_BYTES)
            .map_err(|e| anyhow!("DeepEP IPC handle all_gather failed: {e}"))?;
        let gathered_ids = tp
            .all_gather_bytes(ctx, &local_device_id.to_ne_bytes(), 4)
            .map_err(|e| anyhow!("DeepEP device-id all_gather failed: {e}"))?;

        let mut peers = Vec::with_capacity(cfg.world_size);
        for peer in 0..cfg.world_size {
            let h0 = peer * deepep_sys::IPC_HANDLE_BYTES;
            let mut handle = [0u8; deepep_sys::IPC_HANDLE_BYTES];
            handle.copy_from_slice(&gathered_handles[h0..h0 + deepep_sys::IPC_HANDLE_BYTES]);
            let id0 = peer * 4;
            let device_id = u32::from_ne_bytes(gathered_ids[id0..id0 + 4].try_into()?);
            peers.push((handle, device_id));
        }
        buffer
            .sync(&peers)
            .map_err(|e| anyhow!("DeepEP Buffer::sync failed: {e}"))?;
        Ok(Some(Self {
            world_size,
            buffer: Mutex::new(buffer),
        }))
    }

    pub(crate) fn num_sms() -> Result<u32> {
        let value = std::env::var("ARLE_DSV4_DEEPEP_NUM_SMS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(20);
        ensure!(
            value > 0 && value.is_multiple_of(2),
            "DeepEP num_sms must be positive and even, got {value}"
        );
        Ok(value)
    }

    pub(crate) fn alloc_scratch(
        &self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        num_tokens: usize,
        topk: usize,
        num_experts: usize,
        num_sms: u32,
    ) -> Result<DeepEpDispatchScratch> {
        let num_tokens = num_tokens.max(1);
        let capacity_recv = num_tokens.saturating_mul(self.world_size as usize).max(1);
        let num_channels = (num_sms / 2) as usize;
        Ok(DeepEpDispatchScratch {
            recv_x: HiddenStates::zeros(ctx, hidden_dim, capacity_recv)?,
            recv_src_idx: ctx.stream.alloc_zeros::<i32>(capacity_recv)?,
            recv_topk_idx: ctx
                .stream
                .alloc_zeros::<i64>(capacity_recv.saturating_mul(topk))?,
            recv_topk_weights: ctx
                .stream
                .alloc_zeros::<f32>(capacity_recv.saturating_mul(topk))?,
            rank_prefix: ctx.stream.alloc_zeros::<i32>(
                (self.world_size as usize).saturating_mul(self.world_size as usize),
            )?,
            recv_channel_prefix: ctx
                .stream
                .alloc_zeros::<i32>((self.world_size as usize).saturating_mul(num_channels))?,
            send_head: ctx
                .stream
                .alloc_zeros::<i32>(num_tokens.saturating_mul(self.world_size as usize))?,
            num_tokens_per_rank: ctx.stream.alloc_zeros::<i32>(self.world_size as usize)?,
            num_tokens_per_expert: ctx.stream.alloc_zeros::<i32>(num_experts)?,
            is_token_in_rank: ctx
                .stream
                .alloc_zeros::<u8>(num_tokens.saturating_mul(self.world_size as usize))?,
            channel_prefix_matrix: ctx
                .stream
                .alloc_zeros::<i32>((self.world_size as usize).saturating_mul(num_channels))?,
            combined_topk_weights: ctx
                .stream
                .alloc_zeros::<f32>(num_tokens.saturating_mul(topk))?,
            capacity_recv,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch(
        &self,
        ctx: &DeviceContext,
        scratch: &mut DeepEpDispatchScratch,
        hidden: &HiddenStates,
        topk_idx_i64: &CudaSlice<i64>,
        topk_weights: &CudaSlice<f32>,
        num_experts: usize,
        topk: usize,
        num_sms: u32,
    ) -> Result<usize> {
        ensure!(
            hidden.seq_len > 0,
            "DeepEP dispatch needs at least one token"
        );
        ensure!(
            hidden.hidden_dim.is_multiple_of(16),
            "DeepEP hidden must be int4-aligned"
        );
        ctx.stream.synchronize()?;

        let (x_ptr, _gx) = hidden.data.device_ptr(&ctx.stream);
        let (idx_ptr, _gi) = topk_idx_i64.device_ptr(&ctx.stream);
        let (w_ptr, _gw) = topk_weights.device_ptr(&ctx.stream);
        let (recv_x_ptr, _grx) = scratch.recv_x.data.device_ptr_mut(&ctx.stream);
        let (recv_src_ptr, _grs) = scratch.recv_src_idx.device_ptr_mut(&ctx.stream);
        let (recv_idx_ptr, _gri) = scratch.recv_topk_idx.device_ptr_mut(&ctx.stream);
        let (recv_w_ptr, _grw) = scratch.recv_topk_weights.device_ptr_mut(&ctx.stream);
        let (rank_pref_ptr, _grp) = scratch.rank_prefix.device_ptr_mut(&ctx.stream);
        let (recv_chan_ptr, _grc) = scratch.recv_channel_prefix.device_ptr_mut(&ctx.stream);
        let (send_head_ptr, _gsh) = scratch.send_head.device_ptr_mut(&ctx.stream);
        let (ntpr_ptr, _gnr) = scratch.num_tokens_per_rank.device_ptr_mut(&ctx.stream);
        let (ntpe_ptr, _gne) = scratch.num_tokens_per_expert.device_ptr_mut(&ctx.stream);
        let (titr_ptr, _gtir) = scratch.is_token_in_rank.device_ptr_mut(&ctx.stream);
        let (chan_pref_ptr, _gcp) = scratch.channel_prefix_matrix.device_ptr_mut(&ctx.stream);

        let params = deepep_sys::DispatchParams {
            num_tokens: hidden.seq_len as u32,
            hidden: hidden.hidden_dim as u32,
            num_topk: topk as u32,
            num_experts: num_experts as u32,
            num_sms,
            nvl_chunked_send: 6,
            nvl_chunked_recv: 256,
            d_x: x_ptr as usize,
            d_topk_idx: idx_ptr as usize,
            d_topk_weights: w_ptr as usize,
            d_recv_x: recv_x_ptr as usize,
            d_recv_src_idx: recv_src_ptr as usize,
            d_recv_topk_idx: recv_idx_ptr as usize,
            d_recv_topk_weights: recv_w_ptr as usize,
            d_rank_prefix_matrix: rank_pref_ptr as usize,
            d_recv_channel_prefix: recv_chan_ptr as usize,
            d_send_head: send_head_ptr as usize,
            d_num_tokens_per_rank: ntpr_ptr as usize,
            d_num_tokens_per_expert: ntpe_ptr as usize,
            d_is_token_in_rank: titr_ptr as usize,
            d_channel_prefix_matrix: chan_pref_ptr as usize,
        };
        let mut guard = self
            .buffer
            .lock()
            .map_err(|_| anyhow!("DeepEP buffer mutex poisoned"))?;
        let recv = guard
            .dispatch(&params)
            .map_err(|e| anyhow!("DeepEP dispatch failed: {e}"))?;
        if recv < 0 {
            bail!("DeepEP dispatch returned negative recv count {recv}");
        }
        let recv = recv as usize;
        ensure!(
            recv <= scratch.capacity_recv,
            "DeepEP recv count {recv} exceeds capacity {}",
            scratch.capacity_recv
        );
        Ok(recv)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn combine(
        &self,
        ctx: &DeviceContext,
        scratch: &mut DeepEpDispatchScratch,
        local_expert_out: &HiddenStates,
        out: &mut HiddenStates,
        num_recv_tokens: usize,
        num_output_tokens: usize,
        topk: usize,
        num_sms: u32,
    ) -> Result<()> {
        // compute_stream=0 host-syncs DeepEP after combine, but it does not
        // wait for ARLE compute-stream kernels that produce local_expert_out.
        ctx.stream.synchronize()?;
        let (x_ptr, _gx) = local_expert_out.data.device_ptr(&ctx.stream);
        let (topk_w_ptr, _gtw) = scratch.recv_topk_weights.device_ptr(&ctx.stream);
        let (recv_src_ptr, _grs) = scratch.recv_src_idx.device_ptr(&ctx.stream);
        let (rank_pref_ptr, _grp) = scratch.rank_prefix.device_ptr(&ctx.stream);
        let (recv_chan_ptr, _grc) = scratch.recv_channel_prefix.device_ptr(&ctx.stream);
        let (send_head_ptr, _gsh) = scratch.send_head.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
        let (combined_w_ptr, _gcw) = scratch.combined_topk_weights.device_ptr_mut(&ctx.stream);
        let params = deepep_sys::CombineParams {
            num_input_tokens: num_recv_tokens as u32,
            num_output_tokens: num_output_tokens as u32,
            hidden: out.hidden_dim as u32,
            num_topk: topk as u32,
            num_sms,
            nvl_chunked_send: 6,
            nvl_chunked_recv: 256,
            d_x: x_ptr as usize,
            d_topk_weights: topk_w_ptr as usize,
            d_recv_src_idx: recv_src_ptr as usize,
            d_rank_prefix_matrix: rank_pref_ptr as usize,
            d_recv_channel_prefix: recv_chan_ptr as usize,
            d_send_head: send_head_ptr as usize,
            d_combined_x: out_ptr as usize,
            d_combined_topk_w: combined_w_ptr as usize,
            compute_stream: 0,
        };
        let mut guard = self
            .buffer
            .lock()
            .map_err(|_| anyhow!("DeepEP buffer mutex poisoned (combine)"))?;
        guard
            .combine(&params)
            .map_err(|e| anyhow!("DeepEP combine failed: {e}"))
    }
}
