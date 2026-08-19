//! Tensor-parallel runtime config + communicator handle. Sharding math lives in
//! `infer-topo`; `world_size == 1` is the no-op path.

use cuda_kernels::tensor::CudaPipelineStreamKind;
use infer_topo::TpConfig;

fn lookup_usize(
    primary: &str,
    alias: &str,
    lookup: &mut impl FnMut(&str) -> Option<String>,
) -> Option<usize> {
    lookup(primary)
        .or_else(|| lookup(alias))
        .and_then(|value| value.trim().parse::<usize>().ok())
}

/// Ordinal count of a comma-separated `INFER_CUDA_DEVICES` list = the TP world size.
fn count_cuda_devices(value: &str) -> Option<usize> {
    let count = value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .count();
    (count > 0).then_some(count)
}

/// World size: `INFER_TP_SIZE`/`ARLE_*`, else `INFER_CUDA_DEVICES` count, else 1.
/// Rank: `INFER_TP_RANK`/`ARLE_*`, else 0.
///
/// # Errors
/// Errors if the resolved `(world_size, rank)` is invalid.
pub fn resolve_tp_config(
    mut lookup: impl FnMut(&str) -> Option<String>,
) -> infer_topo::Result<TpConfig> {
    let explicit_size = lookup_usize("INFER_TP_SIZE", "ARLE_TP_SIZE", &mut lookup);
    let device_count = lookup("INFER_CUDA_DEVICES")
        .as_deref()
        .and_then(count_cuda_devices);
    let world_size = explicit_size.or(device_count).unwrap_or(1).max(1);
    let rank = lookup_usize("INFER_TP_RANK", "ARLE_TP_RANK", &mut lookup).unwrap_or(0);
    TpConfig::new(world_size, rank)
}

/// # Errors
/// Errors if the resolved `(world_size, rank)` pair is invalid.
pub fn resolve_tp_config_from_env() -> infer_topo::Result<TpConfig> {
    resolve_tp_config(|key| std::env::var(key).ok())
}

pub enum TpComm {
    Single,
    #[cfg(feature = "nccl")]
    Nccl(Box<cuda_kernels::collective::NcclBackend>),
}

impl TpComm {
    #[must_use]
    pub fn single() -> Self {
        Self::Single
    }

    #[must_use]
    pub fn is_collective(&self) -> bool {
        match self {
            Self::Single => false,
            #[cfg(feature = "nccl")]
            Self::Nccl(_) => true,
        }
    }
}

pub struct TpRuntime {
    config: TpConfig,
    comm: TpComm,
    /// Aliases [`TpComm::Single`] when attn_dp=1 (the default TP8/EP8 route).
    attn_tp: TpComm,
    /// Aliases [`TpComm::Single`] when `ep_size == tp_size` (default TP8/EP8).
    moe_ep: TpComm,
    /// Aliases [`TpComm::Single`] when attn_cp=1 (the default).
    attn_cp: TpComm,
    /// True when the attn_cp partition IS the single global group (attn_tp=1,
    /// e.g. world=2 cp=2): `split_sub_comm` skips the redundant `ncclCommSplit`
    /// there, so the cp collectives run on the GLOBAL comm instead. Safe: the
    /// cp group contains every rank and the per-layer schedule is
    /// rank-identical, so global-comm collective ordering is unchanged.
    /// (attn_tp has no such case: attn_tp==world forces cp=1, and
    /// `attn_all_reduce_sum` already routes cp=1 to the global comm.)
    attn_cp_uses_global: bool,
    attn_cp_rank: usize,
    attn_cp_size: usize,
    /// Attention head-shard coordinates: equal to `(rank, world_size)` when
    /// attn_cp=1, so cp=1 sharding stays byte-identical to raw-TP sharding.
    attn_tp_rank: usize,
    attn_tp_size: usize,
    /// `None` = NCCL everywhere (single rank, `--comm-backend nccl`, or a failed
    /// boot probe/self-test — every degrade logs WARN).
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    oneshot: Option<oneshot::OneShotComm>,
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
pub(crate) struct SymmetricIpcBuffer {
    peer_mappings: Vec<cuda_ipc::PeerMapping>,
    local: Option<cuda_ipc::SharedRegion>,
    peer_ptrs: Vec<u64>,
    bytes: usize,
    context: Option<std::sync::Arc<cudarc::driver::CudaContext>>,
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
impl SymmetricIpcBuffer {
    pub(crate) fn local_base(&self) -> u64 {
        self.local.as_ref().map_or(0, cuda_ipc::SharedRegion::ptr)
    }

    pub(crate) fn peer_ptrs(&self) -> &[u64] {
        &self.peer_ptrs
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
impl Drop for SymmetricIpcBuffer {
    fn drop(&mut self) {
        std::mem::forget(std::mem::take(&mut self.peer_mappings));
        if let Some(local) = self.local.take() {
            std::mem::forget(local);
        }
        if let Some(context) = self.context.take() {
            std::mem::forget(context);
        }
    }
}

impl TpRuntime {
    #[must_use]
    pub fn single() -> Self {
        Self {
            config: TpConfig::single(),
            comm: TpComm::single(),
            attn_tp: TpComm::single(),
            moe_ep: TpComm::single(),
            attn_cp: TpComm::single(),
            attn_cp_uses_global: false,
            attn_cp_rank: 0,
            attn_cp_size: 1,
            attn_tp_rank: 0,
            attn_tp_size: 1,
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            oneshot: None,
        }
    }

    /// A `world_size > 1` config here performs no collectives (NCCL is wired
    /// separately).
    #[must_use]
    pub fn new(config: TpConfig) -> Self {
        Self {
            comm: TpComm::single(),
            attn_tp: TpComm::single(),
            moe_ep: TpComm::single(),
            attn_cp: TpComm::single(),
            attn_cp_uses_global: false,
            attn_cp_rank: 0,
            attn_cp_size: 1,
            attn_tp_rank: config.rank,
            attn_tp_size: config.world_size,
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            oneshot: None,
            config,
        }
    }

    /// # Errors
    /// Errors if the resolved `(world_size, rank)` pair is invalid.
    pub fn from_env() -> infer_topo::Result<Self> {
        Ok(Self::new(resolve_tp_config_from_env()?))
    }

    /// The launcher owns acquiring + broadcasting the same `unique_id` to all
    /// ranks; the CUDA device must already be bound to this rank.
    ///
    /// # Errors
    /// Errors if the resolved `(world_size, rank)` is invalid or NCCL init fails.
    #[cfg(feature = "nccl")]
    pub fn from_env_with_nccl(
        unique_id: cuda_kernels::ffi::nccl::ncclUniqueId,
    ) -> anyhow::Result<Self> {
        use infer_topo::{
            MultiAxisConfig, RankCoord, build_attn_cp_groups, build_attn_tp_groups,
            build_moe_ep_groups,
        };

        let config = resolve_tp_config_from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
        if config.is_single() {
            return Ok(Self::new(config));
        }
        let world_size = config.world_size;
        let rank = config.rank;
        let backend =
            cuda_kernels::collective::NcclBackend::init_rank(unique_id, world_size, rank)?;

        // `cfg` is identical on every rank, so the skip-if-single-group decision (and
        // thus the `split` collective sequence) matches across ranks — required for
        // `ncclCommSplit` ordering correctness.
        let cfg = MultiAxisConfig::current_route_from_env_with_defaults(world_size, world_size)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // attn_tp then moe_ep then attn_cp, unconditionally in this order on every
        // rank.
        let attn_tp = Self::split_sub_comm(&backend, &build_attn_tp_groups(cfg), rank)?;
        let moe_ep = Self::split_sub_comm(&backend, &build_moe_ep_groups(cfg), rank)?;
        // attn_cp=1 yields per-rank singleton groups (not the global group), so the
        // skip must key off `cfg` — still identical on every rank.
        let (attn_cp, attn_cp_uses_global) = if cfg.attn_cp_size == 1 {
            (TpComm::single(), false)
        } else {
            let groups = build_attn_cp_groups(cfg);
            if groups.len() == 1 {
                // The cp partition IS the global group (attn_tp=1, e.g. world=2
                // cp=2): `split_sub_comm` would collapse it to the no-op Single
                // comm. Alias the global comm instead — same membership, and
                // the rank-identical schedule keeps its collective order.
                (TpComm::single(), true)
            } else {
                (Self::split_sub_comm(&backend, &groups, rank)?, false)
            }
        };
        let coord = RankCoord::from_world_rank(cfg, rank).map_err(|e| anyhow::anyhow!("{e}"))?;
        // CP prefill staging offsets assume the cp sub-comm rank equals
        // `attn_cp_rank` (split key = position in group); log the mapping so a
        // runtime mismatch is visible without a debugger.
        if cfg.attn_cp_size > 1 {
            log::info!(
                "[tp] rank {rank}: attn_tp_rank={} attn_cp_rank={} (cp comm {})",
                coord.attn_tp_rank,
                coord.attn_cp_rank,
                if attn_cp_uses_global {
                    "GLOBAL"
                } else {
                    "split"
                },
            );
        }

        Ok(Self {
            config,
            comm: TpComm::Nccl(Box::new(backend)),
            attn_tp,
            moe_ep,
            attn_cp,
            attn_cp_uses_global,
            attn_cp_rank: coord.attn_cp_rank,
            attn_cp_size: cfg.attn_cp_size,
            attn_tp_rank: coord.attn_tp_rank,
            attn_tp_size: cfg.attn_tp_size(),
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            oneshot: None,
        })
    }

    /// `groups` is a rank-partition (each inner Vec = one group's ranks);
    /// `groups.len() == 1` means the partition is the global group, so the
    /// redundant `ncclCommSplit` is skipped. Callers MUST invoke this in the
    /// same order on every rank with identical `groups` (collective ordering).
    #[cfg(feature = "nccl")]
    fn split_sub_comm(
        backend: &cuda_kernels::collective::NcclBackend,
        groups: &[Vec<usize>],
        rank: usize,
    ) -> anyhow::Result<TpComm> {
        if groups.len() == 1 {
            return Ok(TpComm::single());
        }
        let color = groups
            .iter()
            .position(|g| g.contains(&rank))
            .ok_or_else(|| anyhow::anyhow!("rank {rank} not present in any sub-group partition"))?;
        let key = groups[color]
            .iter()
            .position(|&r| r == rank)
            .ok_or_else(|| anyhow::anyhow!("rank {rank} not present in its own sub-group"))?;
        let sub_world = groups[color].len();
        let sub = backend.split(color as i32, key as i32, sub_world, key)?;
        Ok(TpComm::Nccl(Box::new(sub)))
    }

    #[must_use]
    pub fn config(&self) -> &TpConfig {
        &self.config
    }

    #[must_use]
    pub fn comm(&self) -> &TpComm {
        &self.comm
    }

    #[must_use]
    #[allow(dead_code)] // T2.3 will consume this
    pub fn attn_tp(&self) -> &TpComm {
        &self.attn_tp
    }

    #[must_use]
    #[allow(dead_code)] // T2.3 will consume this
    pub fn moe_ep(&self) -> &TpComm {
        &self.moe_ep
    }

    /// The comm the attn_cp collectives run on: the split sub-comm, or the
    /// GLOBAL comm when the cp partition is the single global group.
    #[must_use]
    pub fn attn_cp(&self) -> &TpComm {
        if self.attn_cp_uses_global {
            &self.comm
        } else {
            &self.attn_cp
        }
    }

    #[must_use]
    pub fn attn_cp_size(&self) -> usize {
        self.attn_cp_size
    }

    #[must_use]
    pub fn attn_cp_rank(&self) -> usize {
        self.attn_cp_rank
    }

    /// This rank's attention head-shard index (== `config().rank` at attn_cp=1).
    #[must_use]
    pub fn attn_tp_rank(&self) -> usize {
        self.attn_tp_rank
    }

    /// Attention head-shard world (== `config().world_size` at attn_cp=1).
    #[must_use]
    pub fn attn_tp_size(&self) -> usize {
        self.attn_tp_size
    }

    /// 2D (attn_tp × cp) engages only when both partitions are real
    /// (world ≥ 4). The single geometry predicate — prefill ring, decode
    /// merge, pool sizing, and the spec/recall mutexes all read this.
    #[must_use]
    pub fn two_d_engaged(&self) -> bool {
        self.attn_tp_size >= 2 && self.attn_cp_size >= 2
    }

    #[must_use]
    pub fn is_single(&self) -> bool {
        self.config.is_single()
    }

    #[must_use]
    pub fn is_collective(&self) -> bool {
        self.comm.is_collective()
    }

    /// COLLECTIVE: every rank must call at the same construction point. Any
    /// probe/boot/self-test failure on ANY rank degrades EVERY rank to NCCL via
    /// the built-in ok-votes.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub fn init_oneshot_comm(&mut self, ctx: &cuda_kernels::prelude::DeviceContext) {
        if crate::runtime_flags::comm_nccl_only() {
            log::info!("[comm-oneshot] disabled via --comm-backend nccl");
            return;
        }
        let TpComm::Nccl(backend) = &self.comm else {
            return;
        };
        let world = self.config.world_size;
        let rank = self.config.rank;
        if !matches!(world, 2 | 4 | 6 | 8) {
            log::warn!(
                "[comm-oneshot] world_size={world} unsupported (need 2/4/6/8); staying on NCCL"
            );
            return;
        }
        match oneshot::OneShotComm::build(ctx, backend, world, rank) {
            Ok(Some(comm)) => {
                log::info!(
                    "[comm-oneshot] rank {rank}: one-shot AR/AG active                      (scratch {} KB, self-test passed, larger messages stay NCCL)",
                    comm.scratch_bytes() / 1024
                );
                self.oneshot = Some(comm);
            }
            Ok(None) => {
                log::warn!(
                    "[comm-oneshot] rank {rank}: degraded to NCCL (a rank failed boot/self-test;                      see its WARN above)"
                );
            }
            Err(err) => {
                log::warn!("[comm-oneshot] rank {rank}: boot collective failed: {err:#}");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn all_reduce_sum(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        buf: &mut cuda_kernels::prelude::HiddenStates,
    ) -> anyhow::Result<()> {
        // Compute stream: the AR is on the layer's critical path (the next
        // layer reads its output), so comm-stream overlap cannot hide it and
        // the fence bracket only adds per-AR event alloc + stream-wait cost.
        self.all_reduce_sum_on(ctx, buf, CudaPipelineStreamKind::Compute)
    }

    /// Like [`Self::all_reduce_sum`] but selects the stream the NCCL arm runs
    /// on. One-shot always uses the compute stream (small-message fast path).
    /// `Comm` brackets with compute↔comm fences; `Compute` is unfenced for
    /// callers that overlap the AR with comm-stream work (dsv4 shared expert).
    pub fn all_reduce_sum_on(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        buf: &mut cuda_kernels::prelude::HiddenStates,
        stream: CudaPipelineStreamKind,
    ) -> anyhow::Result<()> {
        // One-shot runs on the global comm only; sub-comms and oversized
        // (prefill) buffers fall through to NCCL.
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(os) = &self.oneshot
            && buf.data.len() * 2 <= os.scratch_bytes()
        {
            return os.all_reduce_sum_inplace(ctx, buf);
        }
        self.all_reduce_sum_over_stream(&self.comm, ctx, buf, stream)
    }

    /// Attention-side partial-sum reduce. Attention (full + linear) weights
    /// shard over the attn_tp axis, so under attn_cp>1 their o/out_proj
    /// partials span only the attn_tp group; the global comm would double-count
    /// the cp replicas. attn_cp=1 keeps the global (one-shot-capable) path.
    #[cfg(feature = "cuda")]
    pub fn attn_all_reduce_sum(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        buf: &mut cuda_kernels::prelude::HiddenStates,
    ) -> anyhow::Result<()> {
        if self.attn_cp_size > 1 {
            self.all_reduce_sum_over(&self.attn_tp, ctx, buf)
        } else {
            self.all_reduce_sum(ctx, buf)
        }
    }

    /// Splits `input`'s `[seq_len]` rows into contiguous `world_size`-way blocks,
    /// runs `compute_fn` over this rank's rows, then all-reduces so every rank
    /// holds the full `[seq_len]` result (unowned rows are zero into the reduction).
    ///
    /// `compute_fn(owned_in, owned_out, start, owned_n)` always runs, even at
    /// `owned_n == 0`: DeepEP LL dispatch/combine are collectives that every rank
    /// must enter regardless. `start` is the owned range's first row index into
    /// `input`, for callers slicing a parallel per-token array.
    ///
    /// # Errors
    /// Propagates device alloc/copy, `compute_fn`, and all-reduce errors.
    #[cfg(all(feature = "cuda", feature = "deepep"))]
    pub(crate) fn shard_rows_and_allreduce(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        input: &cuda_kernels::prelude::HiddenStates,
        out: &mut cuda_kernels::prelude::HiddenStates,
        hidden_size: usize,
        seq_len: usize,
        compute_fn: impl FnOnce(
            &cuda_kernels::prelude::HiddenStates,
            &mut cuda_kernels::prelude::HiddenStates,
            usize,
            usize,
        ) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        use cuda_kernels::prelude::HiddenStates;

        let cfg = self.config();
        let per = seq_len.div_ceil(cfg.world_size);
        let start = (cfg.rank * per).min(seq_len);
        let end = ((cfg.rank + 1) * per).min(seq_len);
        let owned_n = end - start;

        ctx.stream
            .memset_zeros(&mut out.data)
            .map_err(|e| anyhow::anyhow!("shard_rows_and_allreduce: out zero failed: {e}"))?;

        let mut owned_in = HiddenStates::zeros(ctx, hidden_size, owned_n.max(1))?;
        owned_in.seq_len = owned_n;
        if owned_n > 0 {
            ctx.stream
                .memcpy_dtod(
                    &input.data.slice(start * hidden_size..end * hidden_size),
                    &mut owned_in.data.slice_mut(0..owned_n * hidden_size),
                )
                .map_err(|e| {
                    anyhow::anyhow!("shard_rows_and_allreduce: owned-in copy failed: {e}")
                })?;
        }

        let mut owned_out = HiddenStates::zeros(ctx, hidden_size, owned_n.max(1))?;
        owned_out.seq_len = owned_n;
        compute_fn(&owned_in, &mut owned_out, start, owned_n)?;

        if owned_n > 0 {
            ctx.stream
                .memcpy_dtod(
                    &owned_out.data.slice(0..owned_n * hidden_size),
                    &mut out.data.slice_mut(start * hidden_size..end * hidden_size),
                )
                .map_err(|e| {
                    anyhow::anyhow!("shard_rows_and_allreduce: owned-out copy failed: {e}")
                })?;
        }

        self.all_reduce_sum(ctx, out)
    }

    /// # Errors
    /// Propagates the NCCL all-reduce error on multi-rank builds.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    // Without `nccl` the only arm is `Single => Ok(())`, so the args are unused.
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    pub fn all_reduce_sum_over(
        &self,
        comm: &TpComm,
        ctx: &cuda_kernels::prelude::DeviceContext,
        buf: &mut cuda_kernels::prelude::HiddenStates,
    ) -> anyhow::Result<()> {
        self.all_reduce_sum_over_stream(comm, ctx, buf, CudaPipelineStreamKind::Comm)
    }

    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    // Without `nccl` the only arm is `Single => Ok(())`, so the args are unused.
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    fn all_reduce_sum_over_stream(
        &self,
        comm: &TpComm,
        ctx: &cuda_kernels::prelude::DeviceContext,
        buf: &mut cuda_kernels::prelude::HiddenStates,
        stream: CudaPipelineStreamKind,
    ) -> anyhow::Result<()> {
        match comm {
            TpComm::Single => Ok(()),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::{CollectiveBackend, DType, ReduceOp};
                use cudarc::driver::DevicePtrMut;

                let count = buf.data.len();
                let (ptr, _guard) = buf.data.device_ptr_mut(&ctx.stream);
                let cu_stream = match stream {
                    CudaPipelineStreamKind::Comm => {
                        ctx.comm_waits_for_compute()?;
                        ctx.comm_stream.cu_stream()
                    }
                    _ => ctx.stream.cu_stream(),
                };
                // SAFETY: `ptr` is a valid device allocation of `count` BF16
                // elements on this context's device; the selected stream is on
                // the same device. The `_guard` keeps the slice borrowed (and thus
                // un-reallocated) for the duration of the FFI call.
                unsafe {
                    backend.all_reduce(
                        ptr as *mut std::ffi::c_void,
                        count,
                        DType::BF16,
                        ReduceOp::Sum,
                        cu_stream.cast::<std::ffi::c_void>(),
                    )?;
                }
                if matches!(stream, CudaPipelineStreamKind::Comm) {
                    ctx.compute_waits_for_comm()?;
                }
                Ok(())
            }
        }
    }

    /// For capacity decisions feeding the scheduler: each rank's `mem_get_info`
    /// can differ, and divergent scheduler-visible capacity diverges the
    /// deterministic planner and deadlocks NCCL. Collective: every rank MUST
    /// call this at the same construction point.
    ///
    /// # Errors
    /// Propagates device alloc/copy and NCCL errors on multi-rank builds.
    #[cfg(feature = "cuda")]
    // Without `nccl` the only arm is `Single => Ok(value)`, so `ctx` is unused.
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    pub fn all_reduce_min_scalar_i32(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        value: i32,
    ) -> anyhow::Result<i32> {
        match &self.comm {
            TpComm::Single => Ok(value),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::{CollectiveBackend, DType, ReduceOp};
                use cudarc::driver::DevicePtrMut;

                let mut buf = ctx
                    .stream
                    .alloc_zeros::<i32>(1)
                    .map_err(|e| anyhow::anyhow!("scalar all-reduce scratch alloc failed: {e}"))?;
                ctx.stream
                    .memcpy_htod(&[value], &mut buf)
                    .map_err(|e| anyhow::anyhow!("scalar all-reduce htod failed: {e}"))?;
                {
                    let (ptr, _guard) = buf.device_ptr_mut(&ctx.stream);
                    // SAFETY: `ptr` is a valid 1-element i32 device allocation on
                    // this context's device; `ctx.stream` is a stream on the same
                    // device. `_guard` keeps the buffer borrowed for the FFI call.
                    unsafe {
                        backend.all_reduce(
                            ptr as *mut std::ffi::c_void,
                            1,
                            DType::I32,
                            ReduceOp::Min,
                            ctx.stream.cu_stream().cast::<std::ffi::c_void>(),
                        )?;
                    }
                }
                let host = ctx
                    .stream
                    .clone_dtoh(&buf)
                    .map_err(|e| anyhow::anyhow!("scalar all-reduce dtoh failed: {e}"))?;
                Ok(host[0])
            }
        }
    }

    /// FlashMLA's sparse decode kernel wants global heads (`h_q` 64/128) while a
    /// TP rank holds only its local head slab. Gathers one local BF16 row per
    /// rank rank-major; the caller repacks into FlashMLA's head-major layout.
    ///
    /// # Safety
    ///
    /// `sendbuf` must point to at least `sendcount` contiguous BF16 elements on
    /// this rank's current CUDA device; `recvbuf` to writable device memory for
    /// `sendcount * world_size` BF16 elements. Both must stay valid until the
    /// collective enqueued on the comm stream completes, and every rank in the TP
    /// group must call with the same `sendcount`.
    #[cfg(feature = "cuda")]
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    pub unsafe fn all_gather_bf16_raw(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        sendbuf: *const std::ffi::c_void,
        sendcount: usize,
        recvbuf: *mut std::ffi::c_void,
    ) -> anyhow::Result<()> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(os) = &self.oneshot
            && sendcount * 2 <= os.scratch_bytes()
        {
            // SAFETY: same contract as the NCCL arm below; the one-shot
            // path stages `sendbuf` through its registered scratch.
            return unsafe { os.all_gather_bf16(ctx, sendbuf, sendcount, recvbuf) };
        }
        match &self.comm {
            TpComm::Single => anyhow::bail!("single-rank raw all_gather_bf16 is not needed"),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::{CollectiveBackend, DType};

                ctx.comm_waits_for_compute()?;
                // SAFETY: this fn's contract — `sendbuf` holds `sendcount` bf16 and
                // `recvbuf` `sendcount * world`, both live on `ctx`'s device.
                unsafe {
                    backend.all_gather(
                        sendbuf,
                        recvbuf,
                        sendcount,
                        DType::BF16,
                        ctx.comm_stream.cu_stream().cast::<std::ffi::c_void>(),
                    )?;
                }
                ctx.compute_waits_for_comm()?;
                Ok(())
            }
        }
    }

    /// All-gather over the attn_cp sub-comm (CP prefill KV/row exchange).
    /// In-place all-gather over the attn_cp sub-comm without the compute↔comm
    /// fence bracket. `sendbuf == recvbuf + attn_cp_rank * sendcount` is
    /// allowed. For callers that fence the group's real dependencies once.
    ///
    /// # Safety
    ///
    /// `sendbuf` must hold `sendcount` BF16 elements and `recvbuf`
    /// `sendcount * attn_cp_size` BF16 elements on this rank's device; both
    /// stay valid until the collective enqueued on the comm stream completes,
    /// and every cp-group rank calls with the same `sendcount`. The caller
    /// must fence `comm_waits_for_compute` before the op and
    /// `compute_waits_for_comm` before the first compute reading `recvbuf`.
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    pub unsafe fn attn_cp_all_gather_bf16_unfenced(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        sendbuf: *const std::ffi::c_void,
        sendcount: usize,
        recvbuf: *mut std::ffi::c_void,
    ) -> anyhow::Result<()> {
        match self.attn_cp() {
            TpComm::Single => anyhow::bail!("attn_cp all-gather requires the NCCL cp sub-comm"),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::{CollectiveBackend, DType};
                // SAFETY: this fn's contract.
                unsafe {
                    backend.all_gather(
                        sendbuf,
                        recvbuf,
                        sendcount,
                        DType::BF16,
                        ctx.comm_stream.cu_stream().cast::<std::ffi::c_void>(),
                    )?;
                }
                Ok(())
            }
        }
    }

    /// cp send to `peer` (cp-group rank) over the attn_cp sub-comm (GDN
    /// prefill state relay), without the compute↔comm fence bracket. For
    /// callers that issue several cp ops as one group and fence the group's
    /// real data dependencies once (GDN relay, ring rotation).
    ///
    /// # Safety
    ///
    /// `buf` must hold `count` elements of `dtype` on this rank's device and
    /// stay valid until the op enqueued on the comm stream completes; `peer`
    /// must post the matching recv. The caller must fence
    /// `comm_waits_for_compute` before the first op reading a compute-written
    /// buffer and `compute_waits_for_comm` before the first compute reading a
    /// comm-written buffer.
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    pub unsafe fn attn_cp_send_unfenced(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        buf: *const std::ffi::c_void,
        count: usize,
        dtype: cuda_kernels::collective::DType,
        peer: usize,
    ) -> anyhow::Result<()> {
        match self.attn_cp() {
            TpComm::Single => anyhow::bail!("attn_cp send requires the NCCL cp sub-comm"),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::CollectiveBackend;
                // SAFETY: this fn's contract.
                unsafe {
                    backend.send(
                        buf,
                        count,
                        dtype,
                        peer,
                        ctx.comm_stream.cu_stream().cast::<std::ffi::c_void>(),
                    )?;
                }
                Ok(())
            }
        }
    }

    /// cp recv from the attn_cp sub-comm, without the compute↔comm fence
    /// bracket. See [`Self::attn_cp_send_unfenced`] for the caller's fence
    /// obligations.
    ///
    /// # Safety
    ///
    /// `buf` must be writable for `count` elements of `dtype`; the peer must
    /// post the matching send.
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    pub unsafe fn attn_cp_recv_unfenced(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        buf: *mut std::ffi::c_void,
        count: usize,
        dtype: cuda_kernels::collective::DType,
        peer: usize,
    ) -> anyhow::Result<()> {
        match self.attn_cp() {
            TpComm::Single => anyhow::bail!("attn_cp recv requires the NCCL cp sub-comm"),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::CollectiveBackend;
                // SAFETY: this fn's contract.
                unsafe {
                    backend.recv(
                        buf,
                        count,
                        dtype,
                        peer,
                        ctx.comm_stream.cu_stream().cast::<std::ffi::c_void>(),
                    )?;
                }
                Ok(())
            }
        }
    }

    /// In-place broadcast from cp-group rank `root` over the attn_cp sub-comm
    /// (end-of-chunk GDN state agreement), without the compute↔comm fence
    /// bracket. See [`Self::attn_cp_send_unfenced`] for the caller's fence
    /// obligations.
    ///
    /// # Safety
    ///
    /// Every cp-group rank calls with the same `count`/`dtype`/`root`; `buf`
    /// must hold `count` elements and stay valid until the op completes.
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    pub unsafe fn attn_cp_broadcast_unfenced(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        buf: *mut std::ffi::c_void,
        count: usize,
        dtype: cuda_kernels::collective::DType,
        root: usize,
    ) -> anyhow::Result<()> {
        match self.attn_cp() {
            TpComm::Single => anyhow::bail!("attn_cp broadcast requires the NCCL cp sub-comm"),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::CollectiveBackend;
                // SAFETY: this fn's contract.
                unsafe {
                    backend.broadcast(
                        buf,
                        count,
                        dtype,
                        root,
                        ctx.comm_stream.cu_stream().cast::<std::ffi::c_void>(),
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Open an NCCL group on the attn_cp sub-comm. P2P sends/recvs inside a
    /// group are submitted together at `attn_cp_group_end`, so the host never
    /// blocks in `ncclSend` waiting for a peer recv that hasn't been posted.
    pub fn attn_cp_group_start(&self) -> anyhow::Result<()> {
        match self.attn_cp() {
            TpComm::Single => Ok(()),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::CollectiveBackend;
                backend.group_start()
            }
        }
    }

    /// Close an NCCL group opened by `attn_cp_group_start`. May block until
    /// all operations in the group are submitted to the device.
    pub fn attn_cp_group_end(&self) -> anyhow::Result<()> {
        match self.attn_cp() {
            TpComm::Single => Ok(()),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::CollectiveBackend;
                backend.group_end()
            }
        }
    }

    /// RAII attn_cp NCCL group: `drop` closes it after an early `?`, so the
    /// thread-local group depth can't leak and hang the next collective.
    pub fn attn_cp_group(&self) -> anyhow::Result<AttnCpGroupGuard<'_>> {
        self.attn_cp_group_start()?;
        Ok(AttnCpGroupGuard {
            tp: self,
            finished: false,
        })
    }

    /// Host-visible all-gather for small byte payloads (CUDA IPC handles, device
    /// ids); requires NCCL to be initialized.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub fn all_gather_bytes(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        input: &[u8],
        per_rank_bytes: usize,
    ) -> anyhow::Result<Vec<u8>> {
        self.all_gather_bytes_over(&self.comm, ctx, input, per_rank_bytes)
    }

    /// [`Self::all_gather_bytes`] over a sub-communicator (`attn_tp`, `attn_cp`).
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub fn all_gather_bytes_over(
        &self,
        comm: &TpComm,
        ctx: &cuda_kernels::prelude::DeviceContext,
        input: &[u8],
        per_rank_bytes: usize,
    ) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(
            input.len() == per_rank_bytes,
            "all_gather_bytes_over input len {} must equal per-rank bytes {per_rank_bytes}",
            input.len()
        );
        if per_rank_bytes == 0 {
            return Ok(Vec::new());
        }
        match comm {
            TpComm::Single => Ok(input.to_vec()),
            TpComm::Nccl(backend) => backend.all_gather_bytes(ctx, input, per_rank_bytes),
        }
    }

    /// Element-wise reduce `values` in place across `comm`; identity on
    /// single-rank.
    ///
    /// One H2D, an in-place device all-reduce, one D2H. The all-gather form this
    /// replaced moved `world_size ×` the payload back to the host and reduced it
    /// there — for KV-recall's ~2 MB envelope that was ~18 MB of copying per
    /// call, and a ring all-reduce also puts `2(W−1)/W · N` on the wire against
    /// all-gather's `(W−1)·N`.
    #[cfg(feature = "cuda")]
    #[cfg_attr(not(all(feature = "cuda", feature = "nccl")), allow(unused_variables))]
    pub fn reduce_f32_over(
        &self,
        comm: &TpComm,
        ctx: &cuda_kernels::prelude::DeviceContext,
        values: &mut [f32],
        op: cuda_kernels::collective::ReduceOp,
    ) -> anyhow::Result<()> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        {
            use cuda_kernels::collective::{CollectiveBackend, DType};
            use cudarc::driver::DevicePtrMut;

            let TpComm::Nccl(backend) = comm else {
                return Ok(());
            };
            if values.is_empty() {
                return Ok(());
            }
            let mut dev = ctx
                .stream
                .clone_htod(values)
                .map_err(|e| anyhow::anyhow!("reduce_f32_over h2d: {e}"))?;
            {
                let count = values.len();
                let (ptr, _guard) = dev.device_ptr_mut(&ctx.stream);
                // SAFETY: `ptr` is a valid device allocation of `count` f32 on
                // this context's device, and the stream belongs to it. `_guard`
                // keeps the slice borrowed across the FFI call.
                unsafe {
                    backend.all_reduce(
                        ptr as *mut std::ffi::c_void,
                        count,
                        DType::F32,
                        op,
                        ctx.stream.cu_stream().cast::<std::ffi::c_void>(),
                    )?;
                }
            }
            ctx.stream
                .memcpy_dtoh(&dev, values)
                .map_err(|e| anyhow::anyhow!("reduce_f32_over d2h: {e}"))?;
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fn symmetric_vote(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        ok: bool,
    ) -> anyhow::Result<bool> {
        let payload = [u8::from(ok), 0, 0, 0];
        Ok(self
            .all_gather_bytes(ctx, &payload, 4)?
            .chunks_exact(4)
            .all(|chunk| chunk[0] == 1))
    }

    /// Allocate one zeroed CUDA-IPC region per rank and map every peer in
    /// rank order. Collective: every rank must call with the same `bytes`.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    #[allow(dead_code)]
    pub(crate) fn alloc_symmetric_ipc(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        bytes: usize,
    ) -> anyhow::Result<SymmetricIpcBuffer> {
        use anyhow::{anyhow, ensure};
        use cuda_kernels::collective::CollectiveBackend;

        const IPC_HANDLE_BYTES: usize = 64;

        let backend = match &self.comm {
            TpComm::Nccl(backend) => backend,
            TpComm::Single => anyhow::bail!("symmetric IPC requires an NCCL communicator"),
        };
        let world = backend.world_size();
        let rank = backend.rank();
        ensure!(world >= 2, "symmetric IPC requires world_size >= 2");
        ensure!(
            (self.config.world_size, self.config.rank) == (world, rank),
            "TpConfig rank {}/{} does not match NCCL rank {rank}/{world}",
            self.config.rank,
            self.config.world_size
        );
        cuda_ipc::bind(ctx)?;

        let bytes_u64 = u64::try_from(bytes)?;
        let gathered_bytes = self.all_gather_bytes(ctx, &bytes_u64.to_ne_bytes(), 8)?;
        ensure!(
            gathered_bytes.len() == world * 8,
            "symmetric IPC byte-size exchange returned {} bytes, expected {}",
            gathered_bytes.len(),
            world * 8
        );
        let rank_bytes = gathered_bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_ne_bytes(chunk.try_into().expect("8-byte chunk")))
            .collect::<Vec<_>>();
        ensure!(
            rank_bytes.iter().all(|&value| value == bytes_u64),
            "symmetric IPC byte-size mismatch across ranks: {rank_bytes:?}"
        );
        ensure!(bytes > 0, "symmetric IPC bytes must be positive");

        let allocated = (|| -> anyhow::Result<(cuda_ipc::SharedRegion, [u8; IPC_HANDLE_BYTES])> {
            let mut handle = [0u8; IPC_HANDLE_BYTES];
            let local = cuda_ipc::SharedRegion::alloc(
                ctx,
                bytes,
                &mut handle,
                "MegaMoE symmetric region alloc",
            )?;
            Ok((local, handle))
        })();
        let all_allocated = match self.symmetric_vote(ctx, allocated.is_ok()) {
            Ok(all_allocated) => all_allocated,
            Err(err) => {
                if let Ok((local, _)) = allocated {
                    std::mem::forget(local);
                    std::mem::forget(ctx.ctx.clone());
                }
                return Err(err);
            }
        };
        if !all_allocated {
            return match allocated {
                Err(err) => Err(err),
                Ok((local, _)) => {
                    std::mem::forget(local);
                    std::mem::forget(ctx.ctx.clone());
                    Err(anyhow!("symmetric IPC allocation failed on a peer rank"))
                }
            };
        }
        let (local, handle) = allocated?;
        let mut buffer = SymmetricIpcBuffer {
            peer_mappings: Vec::with_capacity(world - 1),
            local: Some(local),
            peer_ptrs: Vec::with_capacity(world),
            bytes,
            context: Some(ctx.ctx.clone()),
        };

        let gathered_handles = self.all_gather_bytes(ctx, &handle, IPC_HANDLE_BYTES)?;
        ensure!(
            gathered_handles.len() == world * IPC_HANDLE_BYTES,
            "symmetric IPC handle exchange returned {} bytes, expected {}",
            gathered_handles.len(),
            world * IPC_HANDLE_BYTES
        );

        cuda_ipc::bind(ctx)?;
        let mut open_error = None;
        for (peer_rank, peer_handle) in gathered_handles.chunks_exact(IPC_HANDLE_BYTES).enumerate()
        {
            if peer_rank == rank {
                buffer.peer_ptrs.push(buffer.local_base());
                continue;
            }
            match cuda_ipc::PeerMapping::open(ctx, peer_handle, "MegaMoE symmetric peer open") {
                Ok(mapping) => {
                    buffer.peer_ptrs.push(mapping.ptr());
                    buffer.peer_mappings.push(mapping);
                }
                Err(err) => {
                    buffer.peer_ptrs.push(0);
                    open_error.get_or_insert(err);
                }
            }
        }
        let all_opened = self.symmetric_vote(ctx, open_error.is_none())?;
        if !all_opened {
            return Err(open_error
                .unwrap_or_else(|| anyhow!("symmetric IPC peer mapping failed on a peer rank")));
        }
        Ok(buffer)
    }

    /// Broadcast rank-0's `values` to every rank; identity on single-rank builds.
    /// DSpark spec-decode needs it: a divergent per-rank draft length mismatches
    /// the per-forward collective count and deadlocks the lockstep coordinator.
    #[cfg(feature = "cuda")]
    #[cfg_attr(not(all(feature = "cuda", feature = "nccl")), allow(unused_variables))]
    pub fn broadcast_rank0_i32(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        values: &[i32],
    ) -> anyhow::Result<Vec<i32>> {
        if self.config.world_size <= 1 {
            return Ok(values.to_vec());
        }
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let gathered = self.all_gather_bytes(ctx, &bytes, bytes.len())?;
            // Rank-major layout: rank 0's payload is the leading `bytes.len()`.
            Ok(gathered[..bytes.len()]
                .chunks_exact(4)
                .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        #[cfg(not(all(feature = "cuda", feature = "nccl")))]
        Ok(values.to_vec())
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
mod cuda_ipc {
    use anyhow::{Result, anyhow, ensure};
    use cuda_kernels::ffi::comm as car;

    pub(super) fn bind(ctx: &cuda_kernels::prelude::DeviceContext) -> Result<()> {
        ctx.ctx
            .bind_to_thread()
            .map_err(|err| anyhow!("bind symmetric IPC CUDA context failed: {err}"))
    }

    pub(super) fn check(res: cudarc::driver::sys::CUresult, what: &str) -> Result<()> {
        ensure!(
            res == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{what} failed: {res:?}"
        );
        Ok(())
    }

    pub(super) struct SharedRegion {
        ptr: u64,
        label: &'static str,
    }

    impl SharedRegion {
        pub(super) fn alloc(
            ctx: &cuda_kernels::prelude::DeviceContext,
            bytes: usize,
            handle: &mut [u8; 64],
            label: &'static str,
        ) -> Result<Self> {
            bind(ctx)?;
            let mut ptr = 0u64;
            check(
                // SAFETY: `ptr` and the 64-byte `handle` are live out-params and `bind`
                // above made `ctx` the current context.
                unsafe { car::arle_car_alloc_shared(bytes, &mut ptr, handle.as_mut_ptr()) },
                label,
            )?;
            Ok(Self { ptr, label })
        }

        pub(super) fn ptr(&self) -> u64 {
            self.ptr
        }

        pub(super) fn disarm(&mut self) {
            self.ptr = 0;
        }
    }

    impl Drop for SharedRegion {
        fn drop(&mut self) {
            if self.ptr == 0 {
                return;
            }
            // SAFETY: non-zero `ptr` is this region's own `alloc_shared` result and
            // `disarm` clears it once ownership moved, so this frees at most once.
            let res = unsafe { car::arle_car_free_shared(self.ptr) };
            if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                log::warn!(
                    "[cuda-ipc] cleanup {} ptr=0x{:x} failed: {res:?}",
                    self.label,
                    self.ptr
                );
            }
        }
    }

    pub(super) struct PeerMapping {
        ptr: u64,
        label: &'static str,
    }

    impl PeerMapping {
        pub(super) fn open(
            ctx: &cuda_kernels::prelude::DeviceContext,
            handle: &[u8],
            label: &'static str,
        ) -> Result<Self> {
            bind(ctx)?;
            let mut ptr = 0u64;
            check(
                // SAFETY: `ptr` is a live out-param and `bind` above made `ctx`
                // current;
                // the callee reads 64 bytes from `handle`, whose length is unchecked
                // here.
                unsafe { car::arle_car_open_peer(handle.as_ptr(), &mut ptr) },
                label,
            )?;
            Ok(Self { ptr, label })
        }

        pub(super) fn ptr(&self) -> u64 {
            self.ptr
        }

        pub(super) fn disarm(&mut self) {
            self.ptr = 0;
        }
    }

    impl Drop for PeerMapping {
        fn drop(&mut self) {
            if self.ptr == 0 {
                return;
            }
            // SAFETY: non-zero `ptr` is this mapping's own `open_peer` result and
            // `disarm` clears it once ownership moved, so this closes at most once.
            let res = unsafe { car::arle_car_close_peer(self.ptr) };
            if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                log::warn!(
                    "[cuda-ipc] cleanup {} ptr=0x{:x} failed: {res:?}",
                    self.label,
                    self.ptr
                );
            }
        }
    }
}

/// One-shot small-message collectives (custom allreduce + all-gather over
/// IPC-shared buffers, one persistent registered scratch per rank): decode-chain
/// messages (2×AR 14 KB + Q-AG 18 KB per layer) measured 2.6–3.6× faster than
/// NCCL on 8×H20.
///
/// Every boot step is followed by an all-ranks ok-vote, so one rank failing
/// alloc/IPC/create degrades EVERY rank instead of desyncing boot collectives.
#[cfg(all(feature = "cuda", feature = "nccl"))]
mod oneshot {
    use super::cuda_ipc::{PeerMapping, SharedRegion, check};
    use anyhow::{Result, anyhow, ensure};
    use cuda_kernels::collective::NcclBackend;
    use cuda_kernels::ffi::comm as car;
    use cuda_kernels::prelude::DeviceContext;
    use std::ffi::c_void;

    /// Covers every decode shape (B=1..32 AR ≤ 448 KB, Q-AG ≤ 18 KB/rank);
    /// prefill-sized buffers fall through to NCCL at the call sites.
    const SCRATCH_BYTES: usize = 512 * 1024;
    /// sgl `Signal` struct (≈3.5 KB) + two-shot staging tmp; the 8 KB margin
    /// absorbs upstream Signal layout drift.
    const SIGNAL_BYTES: usize = 8192 + SCRATCH_BYTES;

    pub(super) struct OneShotComm {
        handle: *mut std::os::raw::c_void,
        scratch_ptr: u64,
    }

    // SAFETY: the handle wraps host-side state plus device pointers; all calls
    // are issued from the engine thread that owns the executor (the same
    // single-threaded discipline as every other forward-path FFI handle).
    unsafe impl Send for OneShotComm {}
    // SAFETY: `&self` use is confined to the engine thread that owns the executor,
    // and the handle's device state is only touched through that thread's stream.
    unsafe impl Sync for OneShotComm {}

    impl Drop for OneShotComm {
        fn drop(&mut self) {
            // SAFETY: `handle` came from `arle_car_create` and is owned solely here —
            // `BootHandle::into_comm` nulls the boot copy, so this destroys once.
            unsafe { car::arle_car_destroy_prod(self.handle) };
        }
    }

    struct BootHandle {
        handle: *mut c_void,
        scratch_ptr: u64,
    }

    impl BootHandle {
        fn self_test(&self, ctx: &DeviceContext, backend: &NcclBackend, rank: usize) -> Result<()> {
            OneShotComm::self_test_handle(self.handle, self.scratch_ptr, ctx, backend, rank)
        }

        fn into_comm(mut self) -> OneShotComm {
            let comm = OneShotComm {
                handle: self.handle,
                scratch_ptr: self.scratch_ptr,
            };
            self.handle = std::ptr::null_mut();
            comm
        }
    }

    impl Drop for BootHandle {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                // SAFETY: non-null means boot failed before `into_comm` moved
                // ownership,
                // so this `arle_car_create` handle is still ours to destroy.
                unsafe { car::arle_car_destroy_prod(self.handle) };
            }
        }
    }

    /// All-ranks ok-vote; the 4-byte payload is required because
    /// `all_gather_bytes` stages through an i32 device buffer.
    fn vote(ctx: &DeviceContext, backend: &NcclBackend, ok: bool) -> Result<bool> {
        let payload = [u8::from(ok), 0, 0, 0];
        let gathered = backend.all_gather_bytes(ctx, &payload, 4)?;
        Ok(gathered.chunks(4).all(|c| c[0] == 1))
    }

    impl OneShotComm {
        pub(super) fn scratch_bytes(&self) -> usize {
            SCRATCH_BYTES
        }

        /// Collective: every rank must call at the same point. `Ok(None)` = a
        /// rank voted a local failure and all ranks degrade together; `Err` only
        /// on collective-infrastructure failures.
        pub(super) fn build(
            ctx: &DeviceContext,
            backend: &NcclBackend,
            world: usize,
            rank: usize,
        ) -> Result<Option<Self>> {
            let alloc_ok = (|| -> Result<(SharedRegion, SharedRegion, [u8; 64], [u8; 64])> {
                let mut sig_handle = [0u8; 64];
                let mut in_handle = [0u8; 64];
                let sig_region = SharedRegion::alloc(
                    ctx,
                    SIGNAL_BYTES,
                    &mut sig_handle,
                    "one-shot signal-region alloc",
                )?;
                let in_region = SharedRegion::alloc(
                    ctx,
                    SCRATCH_BYTES,
                    &mut in_handle,
                    "one-shot scratch alloc",
                )?;
                Ok((sig_region, in_region, sig_handle, in_handle))
            })();
            if let Err(err) = &alloc_ok {
                log::warn!("[comm-oneshot] rank {rank}: {err:#}");
            }
            if !vote(ctx, backend, alloc_ok.is_ok())? {
                return Ok(None);
            }
            let Ok((mut sig_region, mut in_region, sig_handle, in_handle)) = alloc_ok else {
                return Ok(None);
            };

            let mut payload = [0u8; 128];
            payload[..64].copy_from_slice(&sig_handle);
            payload[64..].copy_from_slice(&in_handle);
            let all = backend.all_gather_bytes(ctx, &payload, 128)?;
            ensure!(all.len() == world * 128, "one-shot handle exchange size");

            let scratch_ptr = in_region.ptr();
            let built = (|| -> Result<BootHandle> {
                let mut sigs = [0u64; 8];
                let mut ins = [0u64; 8];
                let mut peer_sigs = Vec::with_capacity(world.saturating_sub(1));
                let mut peer_ins = Vec::with_capacity(world.saturating_sub(1));
                for r in 0..world {
                    if r == rank {
                        sigs[r] = sig_region.ptr();
                        ins[r] = in_region.ptr();
                        continue;
                    }
                    let rec = &all[r * 128..(r + 1) * 128];
                    let sig = PeerMapping::open(
                        ctx,
                        &rec[..64],
                        "one-shot peer signal open (no-P2P probe)",
                    )?;
                    sigs[r] = sig.ptr();
                    peer_sigs.push(sig);
                    let input = PeerMapping::open(ctx, &rec[64..], "one-shot peer scratch open")?;
                    ins[r] = input.ptr();
                    peer_ins.push(input);
                }
                // SAFETY: `sigs`/`ins` are live `world`-length arrays of device
                // pointers —
                // this rank's own regions plus one opened peer mapping per other rank.
                let handle = unsafe {
                    car::arle_car_create(rank as i32, world as i32, sigs.as_ptr(), ins.as_ptr())
                };
                ensure!(!handle.is_null(), "one-shot CustomAllreduce create");
                sig_region.disarm();
                in_region.disarm();
                for peer in &mut peer_sigs {
                    peer.disarm();
                }
                for peer in &mut peer_ins {
                    peer.disarm();
                }
                Ok(BootHandle {
                    handle,
                    scratch_ptr,
                })
            })();
            if let Err(err) = &built {
                log::warn!("[comm-oneshot] rank {rank}: {err:#} (degrading)");
            }
            if !vote(ctx, backend, built.is_ok())? {
                return Ok(None);
            }
            let Ok(boot) = built else {
                return Ok(None);
            };

            // Self-test catches memory-ordering issues on this driver/HW combo
            // before any production traffic.
            let tested = boot.self_test(ctx, backend, rank);
            if let Err(err) = &tested {
                log::warn!("[comm-oneshot] rank {rank} self-test failed: {err:#}");
            }
            if !vote(ctx, backend, tested.is_ok())? {
                return Ok(None);
            }
            Ok(Some(boot.into_comm()))
        }

        fn self_test_handle(
            handle: *mut c_void,
            scratch_ptr: u64,
            ctx: &DeviceContext,
            backend: &NcclBackend,
            rank: usize,
        ) -> Result<()> {
            use cuda_kernels::collective::{CollectiveBackend, DType, ReduceOp};
            use cudarc::driver::DevicePtrMut;

            const ELEMS: usize = 7168;
            let stream = &ctx.stream;
            check(
                // SAFETY: `scratch_ptr` is the registered SCRATCH_BYTES region and
                // ELEMS bf16 (14 KB) fits it; `stream` is this rank's own.
                unsafe {
                    car::arle_car_fill_bf16(
                        stream.cu_stream(),
                        scratch_ptr,
                        ELEMS as i32,
                        rank as i32,
                    )
                },
                "self-test fill",
            )?;
            let mut got = stream
                .alloc_zeros::<u16>(ELEMS)
                .map_err(|e| anyhow!("self-test out alloc: {e}"))?;
            {
                let (got_ptr, _g) = got.device_ptr_mut(stream);
                check(
                    // SAFETY: `got_ptr` is live for `_g` and holds ELEMS u16; `handle`
                    // is
                    // the booted one-shot comm on this rank's `stream`.
                    unsafe {
                        car::arle_car_allreduce_bf16_into(
                            handle,
                            stream.cu_stream(),
                            got_ptr,
                            ELEMS as i32,
                            1,
                        )
                    },
                    "self-test one-shot AR",
                )?;
            }
            let mut reference = stream
                .alloc_zeros::<u16>(ELEMS)
                .map_err(|e| anyhow!("self-test ref alloc: {e}"))?;
            {
                let (ref_ptr, _g) = reference.device_ptr_mut(stream);
                check(
                    // SAFETY: `ref_ptr` is live for `_g` and holds ELEMS u16 — same
                    // seed
                    // and count as the one-shot fill above.
                    unsafe {
                        car::arle_car_fill_bf16(
                            stream.cu_stream(),
                            ref_ptr,
                            ELEMS as i32,
                            rank as i32,
                        )
                    },
                    "self-test ref fill",
                )?;
                // SAFETY: `ref_ptr` is live for `_g` and holds ELEMS bf16 on this
                // rank's
                // device; every rank runs this self-test in the same order.
                unsafe {
                    backend.all_reduce(
                        ref_ptr as *mut std::ffi::c_void,
                        ELEMS,
                        DType::BF16,
                        ReduceOp::Sum,
                        stream.cu_stream().cast::<std::ffi::c_void>(),
                    )?;
                }
            }
            let got_host = stream
                .clone_dtoh(&got)
                .map_err(|e| anyhow!("self-test dtoh: {e}"))?;
            let ref_host = stream
                .clone_dtoh(&reference)
                .map_err(|e| anyhow!("self-test ref dtoh: {e}"))?;
            let to_f32 = |bits: u16| f32::from_bits(u32::from(bits) << 16);
            let max_delta = got_host
                .iter()
                .zip(&ref_host)
                .map(|(&a, &b)| (to_f32(a) - to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            ensure!(
                max_delta <= 0.05,
                "one-shot vs NCCL max delta {max_delta} exceeds tolerance"
            );
            // Lockstep requires the one-shot output to be byte-identical on every rank.
            let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
            for v in &got_host {
                for b in v.to_le_bytes() {
                    digest ^= u64::from(b);
                    digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            let digests = backend.all_gather_bytes(ctx, &digest.to_le_bytes(), 8)?;
            ensure!(
                digests.chunks(8).all(|c| c == digest.to_le_bytes()),
                "one-shot output differs across ranks"
            );
            Ok(())
        }

        /// In-place bf16 sum allreduce through the registered scratch. Caller
        /// guarantees `len*2 <= scratch_bytes()`.
        pub(super) fn all_reduce_sum_inplace(
            &self,
            ctx: &DeviceContext,
            buf: &mut cuda_kernels::prelude::HiddenStates,
        ) -> Result<()> {
            use cudarc::driver::DevicePtrMut;

            let elems = buf.data.len();
            let (ptr, _guard) = buf.data.device_ptr_mut(&ctx.stream);
            // SAFETY: `ptr` is a live device allocation of `elems` bf16 on this
            // device; scratch is this rank's registered region of >= elems*2
            // bytes; both ops are stream-ordered on `ctx.stream`.
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    self.scratch_ptr,
                    ptr,
                    elems * 2,
                    ctx.stream.cu_stream(),
                )
                .map_err(|e| anyhow!("one-shot AR stage copy: {e}"))?;
                check(
                    car::arle_car_allreduce_bf16_into(
                        self.handle,
                        ctx.stream.cu_stream(),
                        ptr,
                        elems as i32,
                        0,
                    ),
                    "one-shot AR",
                )?;
            }
            Ok(())
        }

        /// Raw bf16 all-gather: writes `[world × sendcount]` rank-major into
        /// `recvbuf` (ncclAllGather layout). Caller guarantees
        /// `sendcount*2 <= scratch_bytes()` and `recvbuf` capacity.
        ///
        /// # Safety
        /// Both pointers are live device allocations on this rank's device,
        /// sized as documented.
        pub(super) unsafe fn all_gather_bf16(
            &self,
            ctx: &DeviceContext,
            sendbuf: *const std::ffi::c_void,
            sendcount: usize,
            recvbuf: *mut std::ffi::c_void,
        ) -> Result<()> {
            // SAFETY: this fn's contract — sendbuf holds `sendcount` bf16 and recvbuf
            // `world * sendcount`; caller checked `sendcount*2 <= scratch_bytes()`.
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    self.scratch_ptr,
                    sendbuf as u64,
                    sendcount * 2,
                    ctx.stream.cu_stream(),
                )
                .map_err(|e| anyhow!("one-shot AG stage copy: {e}"))?;
                check(
                    car::arle_car_allgather_bf16_into(
                        self.handle,
                        ctx.stream.cu_stream(),
                        recvbuf as u64,
                        sendcount as i32,
                    ),
                    "one-shot AG",
                )?;
            }
            Ok(())
        }
    }
}

/// RAII guard for an attn_cp NCCL group; see [`TpRuntime::attn_cp_group`].
pub struct AttnCpGroupGuard<'a> {
    tp: &'a TpRuntime,
    finished: bool,
}

impl<'a> AttnCpGroupGuard<'a> {
    /// Close the group, propagating errors; consumes self so drop won't double-close.
    pub fn finish(mut self) -> anyhow::Result<()> {
        self.finished = true;
        self.tp.attn_cp_group_end()
    }
}

impl Drop for AttnCpGroupGuard<'_> {
    fn drop(&mut self) {
        if !self.finished {
            if let Err(e) = self.tp.attn_cp_group_end() {
                log::error!("attn_cp group closed on drop after error: {e}");
            }
        }
    }
}
