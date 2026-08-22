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
use half::bf16;

use crate::runtime_flags::Dsv4MoeTransport;
use crate::tp::TpRuntime;

pub(crate) struct DeepEpTransport {
    world_size: u32,
    buffer: Mutex<deepep_sys::Buffer>,
    /// NVSHMEM low-latency buffer + its boot sizing. `Some` only when the
    /// `deepep_ll` backend is selected (token-owned LL MoE path).
    ll: Option<DeepEpLlBuffer>,
}

/// Boots-once NVSHMEM low-latency buffer state. The `Buffer` itself owns the
/// rdma allocation + NVSHMEM init; `num_max_dispatch_tokens_per_rank`/`hidden`/
/// `num_experts` are kept to size + validate per-step dispatch scratch.
pub(crate) struct DeepEpLlBuffer {
    buffer: Mutex<deepep_sys::Buffer>,
    num_max_dispatch_tokens_per_rank: u32,
    hidden: u32,
    num_experts: u32,
}

pub(crate) struct DeepEpDispatchScratch {
    pub recv_x: HiddenStates,
    pub recv_src_idx: CudaSlice<i32>,
    pub recv_topk_idx: CudaSlice<i64>,
    pub recv_topk_idx_i32: CudaSlice<i32>,
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

/// Pre-allocated NVSHMEM low-latency dispatch/combine scratch, sized ONCE per
/// model (worst-case `num_max_dispatch_tokens_per_rank`) and reused every decode
/// step (overwritten in place — no per-call alloc). Layout mirrors DeepEP's
/// `LowLatencyLayout`: the packed recv tensors are
/// `[num_local_experts, world * num_max_dispatch_tokens_per_rank, *]`.
pub(crate) struct DeepEpLlScratch {
    /// FP8 e4m3 packed recv: `[E_local, world*max_tok, hidden]` bytes.
    pub recv_x_fp8: CudaSlice<u8>,
    /// FP32 column-major scales: `[E_local, sfa_aligned(world*max_tok), hidden/128]`.
    pub recv_x_scales: CudaSlice<f32>,
    /// `int[E_local, world*max_tok]` — source token info (combine input).
    pub recv_src_info: CudaSlice<i32>,
    /// `int64[E_local, world]` — per-(expert,src-rank) layout range (combine input).
    pub recv_layout_range: CudaSlice<i64>,
    /// `int[E_local]` — per-local-expert valid row count (= masked_m).
    pub recv_count: CudaSlice<i32>,
    /// GEMM1 (w13) output: `[E_local, world*max_tok, 2*intermediate]` bf16.
    pub w13_out: CudaSlice<bf16>,
    /// SwiGLU+requant activation FP8: `[E_local, world*max_tok, intermediate]` bytes.
    pub act_fp8: CudaSlice<u8>,
    /// SwiGLU activation scales: `[E_local, sfa_aligned(world*max_tok), intermediate/128]`.
    pub act_scales: CudaSlice<f32>,
    /// GEMM2 (w2) output: `[E_local, world*max_tok, hidden]` bf16 — combine input.
    pub expert_out: CudaSlice<bf16>,
    /// Padded per-expert row capacity = `world * num_max_dispatch_tokens_per_rank`.
    pub m_padded: usize,
    /// TMA-aligned scale row stride = `tma_align(m_padded, 4)`.
    pub sfa_aligned_m: usize,
}

#[derive(Clone, Copy)]
enum DeepEpMode {
    Intranode,
    LowLatency,
}

impl DeepEpTransport {
    pub(crate) fn is_low_latency(&self) -> bool {
        self.ll.is_some()
    }

    /// Per-forward owned-token cap for the LL dispatch buffer (`Some` only on the
    /// deepep_ll path); core caps decode_rows + prefill chunk tokens to it so
    /// `dispatch` never asserts `owned_n <= num_max_dispatch_tokens_per_rank`.
    pub(crate) fn max_owned_tokens_per_forward(&self) -> Option<usize> {
        self.ll
            .as_ref()
            .map(|ll| ll.num_max_dispatch_tokens_per_rank as usize)
    }

    /// `num_max_dispatch_tokens_per_rank` knob — CLI
    /// `--deepep-max-dispatch-tokens-per-rank`, else the SGLang ecosystem env,
    /// default 256; asserted `<= 1024` per DeepEP LL layout limits.
    fn num_max_dispatch_tokens_per_rank() -> Result<u32> {
        let value = crate::runtime_flags::deepep_max_dispatch_tokens_per_rank()
            .or_else(|| {
                std::env::var("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
            })
            .unwrap_or(256);
        ensure!(
            value > 0 && value <= 1024,
            "num_max_dispatch_tokens_per_rank must be in (0, 1024], got {value}"
        );
        Ok(value)
    }

    /// Set the NVSHMEM low-latency behavior env defaults IF UNSET, so the LL
    /// transport boots correctly without depending on the launch script. These
    /// must be in place BEFORE `nvshmem_init` (inside `Buffer::new_low_latency`),
    /// which reads them at init time. Every rank writes identical values.
    ///
    /// The values mirror DeepEP's own reference low-latency setup
    /// (`deep_ep/buffers/legacy.py`, the `low_latency_mode` branch) EXACTLY — the
    /// LL `internode_ll` kernel is IBGDA-based even intra-node and hard-asserts
    /// `ibgda_get_state()->num_rc_per_pe >= num_local_experts` (kernel line 284).
    /// `NVSHMEM_IBGDA_NUM_RC_PER_PE` therefore MUST equal `num_local_experts`
    /// (DeepEP doc: "the low-latency mode requires that [num_qps_per_rank] equals
    /// to the number of local experts"). `NVSHMEM_DISABLE_P2P=0` lets same-node
    /// peers use NVLink while the cross-rank RDMA QPs stay on IBGDA — this is the
    /// `allow_nvlink_for_low_latency_mode=True` path.
    ///
    /// NOTE: `NVSHMEM_REMOTE_TRANSPORT=none` is intentionally NOT set — it would
    /// disable the IBGDA RDMA transport the LL kernel requires (the original
    /// hypothesis was falsified by a device-side QP-count assertion on first
    /// forward; the node carries 8× mlx5 HCAs, so IBGDA is available).
    fn bake_nvshmem_env(rank: u32, num_local_experts: usize) {
        let num_rc = num_local_experts.to_string();
        // Mirror DeepEP legacy.py low_latency_mode env, in order.
        let defaults: [(&str, &str); 8] = [
            ("NVSHMEM_BOOTSTRAP", "UID"),
            // Allow NVLink P2P for same-node LL peers (allow_nvlink_for_low_latency_mode).
            ("NVSHMEM_DISABLE_P2P", "0"),
            ("NVSHMEM_IB_ENABLE_IBGDA", "1"),
            // MUST equal num_local_experts (kernel asserts num_rc_per_pe >= it).
            ("NVSHMEM_IBGDA_NUM_RC_PER_PE", num_rc.as_str()),
            ("NVSHMEM_QP_DEPTH", "1024"),
            // 6 default teams + 1 extra.
            ("NVSHMEM_MAX_TEAMS", "7"),
            // Disable NVLink SHARP.
            ("NVSHMEM_DISABLE_NVLS", "1"),
            // Disable multi-node NVLink detection (single-node 8×H20).
            ("NVSHMEM_DISABLE_MNNVL", "1"),
        ];
        for (key, default) in defaults {
            let effective = match std::env::var(key) {
                Ok(existing) => existing,
                Err(_) => {
                    // SAFETY: boot is pre-steady-state (before any NVSHMEM init or
                    // worker threads touch the environment); all ranks write
                    // identical values, so there is no observable race or
                    // cross-rank divergence.
                    unsafe {
                        std::env::set_var(key, default);
                    }
                    default.to_string()
                }
            };
            log::info!("[deepep_ll] rank={rank} NVSHMEM env {key}={effective}");
        }
    }

    /// Boot the transport. `hidden`/`num_experts` are required to size the LL
    /// rdma buffer when the `deepep_ll` backend is selected (the intranode path
    /// ignores them). Both come from the model config at the call site.
    pub(crate) fn maybe_boot(
        ctx: &DeviceContext,
        tp: &TpRuntime,
        hidden: usize,
        num_experts: usize,
    ) -> Result<Option<Self>> {
        let mode = match crate::runtime_flags::dsv4_moe_transport()? {
            Dsv4MoeTransport::DeepEp => DeepEpMode::Intranode,
            Dsv4MoeTransport::DeepEpLowLatency => DeepEpMode::LowLatency,
            Dsv4MoeTransport::AllReduce | Dsv4MoeTransport::MegaMoe => return Ok(None),
        };
        ensure!(deepep_sys::is_native(), "deepep-sys was built as a stub");
        let cfg = tp.config();
        ensure!(
            cfg.world_size >= 2,
            "native DeepEP requires world_size >= 2, got {}",
            cfg.world_size
        );
        let rank = u32::try_from(cfg.rank)?;
        let world_size = u32::try_from(cfg.world_size)?;
        // Pin the REAL device ordinal this rank runs on (INFER_CUDA_DEVICES via
        // ctx.ordinal), NOT the TP rank. On a non-0-based GPU set (e.g. 4,5,6,7)
        // rank != ordinal; passing rank would create the DeepEP buffers on the
        // wrong physical GPU and the intranode barrier would fail. For 0-based
        // contiguous layouts ctx.ordinal == rank, so this is byte-identical.
        let mut buffer = deepep_sys::Buffer::new(rank, world_size, ctx.ordinal)
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

        let peers = gathered_handles
            .chunks_exact(deepep_sys::IPC_HANDLE_BYTES)
            .zip(gathered_ids.chunks_exact(4))
            .map(|(hb, ib)| {
                let handle: [u8; deepep_sys::IPC_HANDLE_BYTES] = hb.try_into()?;
                let device_id = u32::from_ne_bytes(ib.try_into()?);
                Ok((handle, device_id))
            })
            .collect::<Result<Vec<_>>>()?;
        buffer
            .sync(&peers)
            .map_err(|e| anyhow!("DeepEP Buffer::sync failed: {e}"))?;

        // LL boot: rank 0 mints the NVSHMEM uniqueid, broadcast it to all ranks
        // over the SAME byte all-gather channel the IPC-handle exchange used,
        // then every rank creates its LL buffer with the shared uid + sizing.
        let ll = if matches!(mode, DeepEpMode::LowLatency) {
            let num_max_dispatch_tokens_per_rank = Self::num_max_dispatch_tokens_per_rank()?;
            let hidden_u32 = u32::try_from(hidden)?;
            let num_experts_u32 = u32::try_from(num_experts)?;
            ensure!(
                num_experts_u32.is_multiple_of(world_size),
                "deepep_ll: num_experts {num_experts} must be divisible by world_size {world_size}"
            );
            let num_local_experts = (num_experts_u32 / world_size) as usize;
            // Bake the NVSHMEM low-latency env INTO the transport boot. The LL
            // `Buffer::new_low_latency` calls `nvshmem_init`, which reads these at
            // init time — relying on the launch script is the "env not on" failure
            // ckl flagged. Set self-contained defaults IF UNSET (user can still
            // override), and log the effective values. `NVSHMEM_IBGDA_NUM_RC_PER_PE`
            // is sized to `num_local_experts` (DeepEP LL kernel requirement).
            // `LD_LIBRARY_PATH` for the NVSHMEM lib is a dlopen-timing concern and
            // MUST stay in the launch script — it is intentionally NOT touched here.
            Self::bake_nvshmem_env(rank, num_local_experts);
            ensure!(
                hidden.is_multiple_of(512),
                "deepep_ll: hidden {hidden} must be a multiple of 512 (FP8 dispatch)"
            );
            // Every rank gathers the uid bytes; only rank 0's slot is valid.
            // Non-root ranks send zeros, then everyone reads rank-0's slot.
            let local_uid = if rank == 0 {
                deepep_sys::ll_get_uniqueid()
                    .map_err(|e| anyhow!("DeepEP ll_get_uniqueid failed: {e}"))?
            } else {
                [0u8; deepep_sys::LL_UNIQUEID_BYTES]
            };
            let gathered_uids = tp
                .all_gather_bytes(ctx, &local_uid, deepep_sys::LL_UNIQUEID_BYTES)
                .map_err(|e| anyhow!("DeepEP LL uniqueid all_gather failed: {e}"))?;
            let mut root_uid = [0u8; deepep_sys::LL_UNIQUEID_BYTES];
            root_uid.copy_from_slice(&gathered_uids[0..deepep_sys::LL_UNIQUEID_BYTES]);
            log::info!(
                "[deepep_ll] rank={rank} booting NVSHMEM LL buffer: world={world_size} \
                 hidden={hidden} experts={num_experts} max_tok={num_max_dispatch_tokens_per_rank}"
            );
            let ll_buffer = deepep_sys::Buffer::new_low_latency(
                rank,
                world_size,
                ctx.ordinal,
                num_max_dispatch_tokens_per_rank,
                hidden_u32,
                num_experts_u32,
                &root_uid,
            )
            .map_err(|e| anyhow!("DeepEP new_low_latency failed: {e}"))?;
            log::info!("[deepep_ll] rank={rank} NVSHMEM LL buffer booted");
            Some(DeepEpLlBuffer {
                buffer: Mutex::new(ll_buffer),
                num_max_dispatch_tokens_per_rank,
                hidden: hidden_u32,
                num_experts: num_experts_u32,
            })
        } else {
            None
        };

        Ok(Some(Self {
            world_size,
            buffer: Mutex::new(buffer),
            ll,
        }))
    }

    pub(crate) fn num_sms() -> Result<u32> {
        let value = crate::runtime_flags::deepep_num_sms();
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
            recv_topk_idx_i32: ctx
                .stream
                .alloc_zeros::<i32>(capacity_recv.saturating_mul(topk))?,
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
            compute_stream: ctx.stream.cu_stream() as usize,
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
            compute_stream: ctx.stream.cu_stream() as usize,
        };
        let mut guard = self
            .buffer
            .lock()
            .map_err(|_| anyhow!("DeepEP buffer mutex poisoned (combine)"))?;
        guard
            .combine(&params)
            .map_err(|e| anyhow!("DeepEP combine failed: {e}"))
    }

    fn ll(&self) -> Result<&DeepEpLlBuffer> {
        self.ll
            .as_ref()
            .ok_or_else(|| anyhow!("deepep_ll path used but LL buffer not booted"))
    }

    /// Local expert count = `num_experts / world_size`.
    pub(crate) fn ll_num_local_experts(&self) -> Result<usize> {
        let ll = self.ll()?;
        Ok((ll.num_experts / self.world_size) as usize)
    }

    /// TMA-aligned scale row stride for the masked grouped GEMM.
    fn tma_align(m: usize) -> usize {
        m.div_ceil(4) * 4
    }

    /// Allocate the LL dispatch/combine scratch ONCE per model (caller stores it
    /// per slot/model and reuses every step). `intermediate` sizes the
    /// SwiGLU+w2 stages. All buffers are zeroed; per-step kernels overwrite them.
    pub(crate) fn alloc_ll_scratch(
        &self,
        ctx: &DeviceContext,
        intermediate: usize,
    ) -> Result<DeepEpLlScratch> {
        let ll = self.ll()?;
        let world = self.world_size as usize;
        let e_local = (ll.num_experts / self.world_size) as usize;
        let hidden = ll.hidden as usize;
        let max_tok = ll.num_max_dispatch_tokens_per_rank as usize;
        let m_padded = world * max_tok;
        let sfa_aligned_m = Self::tma_align(m_padded);
        // DeepEP's LL dispatch writes the FP8 scale buffer transposed with the
        // token dim (`world*max_tok`) as the OUTER stride and NO extra TMA
        // padding. The masked GEMM's SFA TMA descriptor needs that outer stride
        // == `sfa_aligned_m`. They only agree when `world*max_tok` is already
        // TMA-aligned; otherwise the two layouts diverge silently. Fail loud.
        ensure!(
            sfa_aligned_m == m_padded,
            "deepep_ll: world*num_max_dispatch_tokens_per_rank ({m_padded}) must be TMA-aligned \
             (multiple of 4); got tma_align={sfa_aligned_m}. Pick a num_max_dispatch_tokens_per_rank \
             whose world-product is a multiple of 4."
        );
        ensure!(hidden.is_multiple_of(128), "deepep_ll hidden must be %128");
        ensure!(
            intermediate.is_multiple_of(128),
            "deepep_ll intermediate must be %128"
        );
        let recv_x_fp8 = ctx.stream.alloc_zeros::<u8>(e_local * m_padded * hidden)?;
        let recv_x_scales = ctx
            .stream
            .alloc_zeros::<f32>(e_local * sfa_aligned_m * (hidden / 128))?;
        let recv_src_info = ctx.stream.alloc_zeros::<i32>(e_local * m_padded)?;
        let recv_layout_range = ctx.stream.alloc_zeros::<i64>(e_local * world)?;
        let recv_count = ctx.stream.alloc_zeros::<i32>(e_local)?;
        let w13_out = ctx
            .stream
            .alloc_zeros::<bf16>(e_local * m_padded * 2 * intermediate)?;
        let act_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(e_local * m_padded * intermediate)?;
        let act_scales = ctx
            .stream
            .alloc_zeros::<f32>(e_local * sfa_aligned_m * (intermediate / 128))?;
        let expert_out = ctx
            .stream
            .alloc_zeros::<bf16>(e_local * m_padded * hidden)?;
        Ok(DeepEpLlScratch {
            recv_x_fp8,
            recv_x_scales,
            recv_src_info,
            recv_layout_range,
            recv_count,
            w13_out,
            act_fp8,
            act_scales,
            expert_out,
            m_padded,
            sfa_aligned_m,
        })
    }

    /// One NVSHMEM low-latency dispatch of THIS rank's owned tokens. `hidden` is
    /// the owned `[hidden_dim, owned_n]` activations (bf16), `topk_idx_i64` the
    /// owned `[owned_n, topk]` global expert ids already on device. Writes the
    /// packed FP8 recv tensors into `scratch`; returns `(expected_m, masked_m)`
    /// where `masked_m` is sized `[num_local_experts]` on device (= recv_count).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ll_dispatch(
        &self,
        ctx: &DeviceContext,
        scratch: &mut DeepEpLlScratch,
        hidden: &HiddenStates,
        topk_idx_i64: &CudaSlice<i64>,
        topk: usize,
    ) -> Result<i32> {
        let ll = self.ll()?;
        let owned_n = hidden.seq_len;
        ensure!(
            hidden.hidden_dim == ll.hidden as usize,
            "deepep_ll dispatch hidden_dim {} != boot hidden {}",
            hidden.hidden_dim,
            ll.hidden
        );
        ensure!(
            owned_n <= ll.num_max_dispatch_tokens_per_rank as usize,
            "deepep_ll owned tokens {owned_n} exceed num_max_dispatch_tokens_per_rank {}",
            ll.num_max_dispatch_tokens_per_rank
        );
        // The LL kernel runs on its own comm stream, ordered after ARLE's
        // compute stream via on-device events (compute_stream below).
        let (x_ptr, _gx) = hidden.data.device_ptr(&ctx.stream);
        let (idx_ptr, _gi) = topk_idx_i64.device_ptr(&ctx.stream);
        let (recv_x_ptr, _gr0) = scratch.recv_x_fp8.device_ptr_mut(&ctx.stream);
        let (recv_sc_ptr, _gr1) = scratch.recv_x_scales.device_ptr_mut(&ctx.stream);
        let (recv_si_ptr, _gr2) = scratch.recv_src_info.device_ptr_mut(&ctx.stream);
        let (recv_lr_ptr, _gr3) = scratch.recv_layout_range.device_ptr_mut(&ctx.stream);
        let (recv_cnt_ptr, _gr4) = scratch.recv_count.device_ptr_mut(&ctx.stream);
        let params = deepep_sys::LowLatencyDispatchParams {
            num_tokens: owned_n as u32,
            hidden: ll.hidden,
            num_topk: topk as u32,
            num_experts: ll.num_experts,
            num_max_dispatch_tokens_per_rank: ll.num_max_dispatch_tokens_per_rank,
            use_fp8: true,
            round_scale: false,
            use_ue8m0: false,
            d_x: x_ptr as usize,
            d_topk_idx: idx_ptr as usize,
            d_recv_x: recv_x_ptr as usize,
            d_recv_x_scales: recv_sc_ptr as usize,
            d_recv_src_info: recv_si_ptr as usize,
            d_recv_layout_range: recv_lr_ptr as usize,
            d_recv_count: recv_cnt_ptr as usize,
            compute_stream: ctx.stream.cu_stream() as usize,
        };
        let mut guard = ll
            .buffer
            .lock()
            .map_err(|_| anyhow!("deepep_ll buffer mutex poisoned (dispatch)"))?;
        let expected_m = guard
            .low_latency_dispatch(&params)
            .map_err(|e| anyhow!("deepep_ll dispatch failed: {e}"))?;
        Ok(expected_m)
    }

    /// One NVSHMEM low-latency combine. `expert_out` is the per-expert w2 output
    /// `[E_local, world*max_tok, hidden]` (in `scratch.expert_out`); `out` is
    /// THIS rank's owned routed output `[hidden_dim, owned_n]`. `topk_idx_i64` /
    /// `topk_weights` are the owned `[owned_n, topk]` route ids + weights.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ll_combine(
        &self,
        ctx: &DeviceContext,
        scratch: &DeepEpLlScratch,
        out: &mut HiddenStates,
        topk_idx_i64: &CudaSlice<i64>,
        topk_weights: &CudaSlice<f32>,
        owned_n: usize,
        topk: usize,
    ) -> Result<()> {
        let ll = self.ll()?;
        ensure!(
            out.seq_len == owned_n && out.hidden_dim == ll.hidden as usize,
            "deepep_ll combine out shape {}x{} != owned {}x{}",
            out.hidden_dim,
            out.seq_len,
            ll.hidden,
            owned_n
        );
        let (x_ptr, _gx) = scratch.expert_out.device_ptr(&ctx.stream);
        let (idx_ptr, _gi) = topk_idx_i64.device_ptr(&ctx.stream);
        let (w_ptr, _gw) = topk_weights.device_ptr(&ctx.stream);
        let (si_ptr, _gs) = scratch.recv_src_info.device_ptr(&ctx.stream);
        let (lr_ptr, _gl) = scratch.recv_layout_range.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
        let params = deepep_sys::LowLatencyCombineParams {
            num_combined_tokens: owned_n as u32,
            hidden: ll.hidden,
            num_topk: topk as u32,
            num_experts: ll.num_experts,
            num_max_dispatch_tokens_per_rank: ll.num_max_dispatch_tokens_per_rank,
            use_logfmt: false,
            zero_copy: false,
            d_x: x_ptr as usize,
            d_topk_idx: idx_ptr as usize,
            d_topk_weights: w_ptr as usize,
            d_src_info: si_ptr as usize,
            d_layout_range: lr_ptr as usize,
            d_combined_x: out_ptr as usize,
            compute_stream: ctx.stream.cu_stream() as usize,
        };
        let mut guard = ll
            .buffer
            .lock()
            .map_err(|_| anyhow!("deepep_ll buffer mutex poisoned (combine)"))?;
        guard
            .low_latency_combine(&params)
            .map_err(|e| anyhow!("deepep_ll combine failed: {e}"))
    }
}
