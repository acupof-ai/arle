//! Tensor-parallel runtime config + communicator handle.
//!
//! [`resolve_tp_config`] resolves env → [`TpConfig`] (CPU-testable);
//! [`TpRuntime`] pairs it with a communicator (the NCCL handle is
//! `nccl`-gated; `world_size == 1` is the no-op path). The sharding math lives in
//! `infer-topo`.

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

/// Count the ordinals in a comma-separated `INFER_CUDA_DEVICES` list (empty
/// entries ignored). The count is the TP world size (8 ordinals ⇒ TP=8).
fn count_cuda_devices(value: &str) -> Option<usize> {
    let count = value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .count();
    (count > 0).then_some(count)
}

/// Resolve [`TpConfig`] from an env lookup. World size: `INFER_TP_SIZE`/`ARLE_*`,
/// else `INFER_CUDA_DEVICES` count, else 1. Rank: `INFER_TP_RANK`/`ARLE_*` (0).
///
/// # Errors
/// Errors if the resolved `(world_size, rank)` is invalid, via [`TpConfig::new`].
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

/// Resolve [`TpConfig`] from the process environment.
///
/// # Errors
/// Errors if the resolved `(world_size, rank)` pair is invalid.
pub fn resolve_tp_config_from_env() -> infer_topo::Result<TpConfig> {
    resolve_tp_config(|key| std::env::var(key).ok())
}

/// Tensor-parallel communicator handle. [`Self::Nccl`] (only in `nccl` builds)
/// wraps `cuda_kernels::collective::NcclBackend`; everything else uses the
/// [`Self::Single`] no-op so the code compiles GPU-free.
pub enum TpComm {
    /// Single rank: no collectives needed.
    Single,
    /// Multi-rank NCCL-backed communicator (real collectives).
    #[cfg(feature = "nccl")]
    Nccl(Box<cuda_kernels::collective::NcclBackend>),
}

impl TpComm {
    /// The single-rank no-op communicator.
    #[must_use]
    pub fn single() -> Self {
        Self::Single
    }

    /// Whether this communicator performs real cross-rank collectives.
    #[must_use]
    pub fn is_collective(&self) -> bool {
        match self {
            Self::Single => false,
            #[cfg(feature = "nccl")]
            Self::Nccl(_) => true,
        }
    }
}

/// Tensor-parallel runtime: the resolved [`TpConfig`] plus its communicator.
pub struct TpRuntime {
    config: TpConfig,
    comm: TpComm,
    /// Attention tensor-parallel sub-communicator (DSv4 DP-attention).
    /// Under the default TP8/EP8 route this aliases [`TpComm::Single`] because
    /// the attn_tp groups equal the global comm (attn_dp=1).
    // T2.3 will consume this (DP-attention token-owned EP sub-collectives).
    attn_tp: TpComm,
    /// MoE expert-parallel sub-communicator (DSv4 token-owned EP).
    /// Under the default TP8/EP8 route this aliases [`TpComm::Single`] because
    /// `ep_size == tp_size` ⇒ the moe_ep groups equal the global comm.
    // T2.3 will consume this (DP-attention token-owned EP sub-collectives).
    moe_ep: TpComm,
    /// One-shot small-message collective path (vendored sgl-kernel/vLLM custom
    /// allreduce + ARLE one-shot all-gather) for the decode critical chain.
    /// `None` = NCCL everywhere (single rank, `--comm-backend nccl`, or any
    /// boot probe/self-test failure — every degrade logs WARN, never silent).
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
    /// Single-GPU runtime (no parallelism, no-op communicator).
    #[must_use]
    pub fn single() -> Self {
        Self {
            config: TpConfig::single(),
            comm: TpComm::single(),
            attn_tp: TpComm::single(),
            moe_ep: TpComm::single(),
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            oneshot: None,
        }
    }

    /// Build a runtime from a resolved [`TpConfig`] with the no-op communicator.
    /// A `world_size > 1` config here is a valid CPU/typecheck shape that
    /// performs no collectives (NCCL is wired separately).
    #[must_use]
    pub fn new(config: TpConfig) -> Self {
        Self {
            config,
            comm: TpComm::single(),
            attn_tp: TpComm::single(),
            moe_ep: TpComm::single(),
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            oneshot: None,
        }
    }

    /// Build a runtime from a config and an explicit communicator. The sub-comms
    /// default to the no-op [`TpComm::single`]; [`Self::from_env_with_nccl`] is
    /// the path that builds real attn_tp / moe_ep sub-comms.
    #[must_use]
    pub fn with_comm(config: TpConfig, comm: TpComm) -> Self {
        Self {
            config,
            comm,
            attn_tp: TpComm::single(),
            moe_ep: TpComm::single(),
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            oneshot: None,
        }
    }

    /// Resolve the runtime from env with the no-op communicator.
    ///
    /// # Errors
    /// Errors if the resolved `(world_size, rank)` pair is invalid.
    pub fn from_env() -> infer_topo::Result<Self> {
        Ok(Self::new(resolve_tp_config_from_env()?))
    }

    /// Resolve from env and, on a multi-rank `nccl` build, bring up the real NCCL
    /// communicator via `ncclCommInitRank(unique_id, world_size, rank)`.
    /// `world_size == 1` is exactly [`Self::from_env`] (no-op), so this is safe on
    /// every path. The launcher owns acquiring + broadcasting the same `unique_id`
    /// to all ranks; the CUDA device must already be bound to this rank.
    ///
    /// # Errors
    /// Errors if the resolved `(world_size, rank)` is invalid or NCCL init fails.
    #[cfg(feature = "nccl")]
    pub fn from_env_with_nccl(
        unique_id: cuda_kernels::ffi::nccl::ncclUniqueId,
    ) -> anyhow::Result<Self> {
        use infer_topo::{MultiAxisConfig, build_attn_tp_groups, build_moe_ep_groups};

        let config = resolve_tp_config_from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
        if config.is_single() {
            return Ok(Self::new(config));
        }
        let world_size = config.world_size;
        let rank = config.rank;
        let backend =
            cuda_kernels::collective::NcclBackend::init_rank(unique_id, world_size, rank)?;

        // Env-driven axis config. Under the default TP8/EP8 this yields attn_dp=1
        // ⇒ attn_tp groups == the global group, and ep_size == tp_size ⇒ moe_ep
        // groups == the global group, so both splits below alias the global comm.
        // `cfg` is identical on every rank, so the skip-if-single-group decision
        // (and thus the sequence of `split` collectives) is identical across all
        // ranks — required for `ncclCommSplit` collective-ordering correctness.
        let cfg = MultiAxisConfig::current_route_from_env_with_defaults(world_size, world_size)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // IMPORTANT: do attn_tp then moe_ep, unconditionally-in-the-same-order on
        // every rank. `split_sub_comm` skips the split only when the partition is
        // a single global group (identical decision on all ranks).
        let attn_tp = Self::split_sub_comm(&backend, &build_attn_tp_groups(cfg), rank)?;
        let moe_ep = Self::split_sub_comm(&backend, &build_moe_ep_groups(cfg), rank)?;

        Ok(Self {
            config,
            comm: TpComm::Nccl(Box::new(backend)),
            attn_tp,
            moe_ep,
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            oneshot: None,
        })
    }

    /// Build a sub-communicator for `rank` from a rank-partition `groups`
    /// (each inner Vec = the ranks in one group). If `groups.len() == 1` the
    /// partition is a single global group (sub-size == tp_size, the builder
    /// returned the global group), so we skip the redundant `ncclCommSplit`
    /// (all ranks share one color ⇒ a wasteful global-comm clone + extra
    /// collective) and return the no-op [`TpComm::single`]. Otherwise we locate
    /// this rank's `color`/`key` and split the parent comm.
    ///
    /// Collective-ordering: callers MUST invoke this in the same order on every
    /// rank with the same (identical-per-rank) `groups`, so the skip-or-split
    /// decision and the resulting `split` collective sequence match across ranks.
    #[cfg(feature = "nccl")]
    fn split_sub_comm(
        backend: &cuda_kernels::collective::NcclBackend,
        groups: &[Vec<usize>],
        rank: usize,
    ) -> anyhow::Result<TpComm> {
        // Single global group ⇒ sub-comm aliases the global comm. Skip the split.
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

    /// The resolved tensor-parallel placement.
    #[must_use]
    pub fn config(&self) -> &TpConfig {
        &self.config
    }

    /// The communicator handle.
    #[must_use]
    pub fn comm(&self) -> &TpComm {
        &self.comm
    }

    /// The attention tensor-parallel sub-communicator (DSv4 DP-attention).
    /// Aliases [`TpComm::Single`] under the default TP8/EP8 route.
    #[must_use]
    #[allow(dead_code)] // T2.3 will consume this
    pub fn attn_tp(&self) -> &TpComm {
        &self.attn_tp
    }

    /// The MoE expert-parallel sub-communicator (DSv4 token-owned EP).
    /// Aliases [`TpComm::Single`] under the default TP8/EP8 route.
    #[must_use]
    #[allow(dead_code)] // T2.3 will consume this
    pub fn moe_ep(&self) -> &TpComm {
        &self.moe_ep
    }

    /// Whether this runtime is single-GPU (no all-reduce needed).
    #[must_use]
    pub fn is_single(&self) -> bool {
        self.config.is_single()
    }

    /// Whether the runtime will perform real cross-rank collectives.
    #[must_use]
    pub fn is_collective(&self) -> bool {
        self.comm.is_collective()
    }

    /// Whether the decode small-message path is currently using ARLE's
    /// one-shot CustomAllreduce instead of plain NCCL.
    #[must_use]
    pub fn oneshot_comm_active(&self) -> bool {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        {
            self.oneshot.is_some()
        }
        #[cfg(not(all(feature = "cuda", feature = "nccl")))]
        {
            false
        }
    }

    /// All-reduce (sum) a row-parallel GEMM output across the TP group, in place.
    ///
    /// Row-parallel linears (`o_proj`/`down_proj`) produce a partial per rank;
    /// summing reconstructs the full output. Runs in place on the compute stream
    /// — stream ordering alone sequences GEMM → all-reduce → residual-add, so no
    /// cross-stream event is needed. [`TpComm::Single`] is a no-op (single rank
    /// already holds the full output).
    ///
    /// # Errors
    /// Propagates the NCCL all-reduce error on multi-rank builds.
    /// Bring up the one-shot small-message collective path (default-on with
    /// automatic loud degrade). COLLECTIVE: every rank must call at the same
    /// construction point. `--comm-backend nccl` skips it entirely; any
    /// probe/boot/self-test failure on ANY rank degrades EVERY rank to NCCL via
    /// the built-in ok-votes (never a silent fallback, never a desynced boot).
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub fn init_oneshot_comm(&mut self, ctx: &cuda_kernels::prelude::DeviceContext) {
        if crate::runtime_flags::comm_nccl_only() {
            log::info!("[comm-oneshot] disabled via --comm-backend nccl");
            return;
        }
        let TpComm::Nccl(backend) = &self.comm else {
            return; // single rank — nothing to accelerate
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
        // One-shot path: decode-sized bf16 messages over the GLOBAL comm only
        // (sub-comms keep NCCL). Oversized (prefill) buffers fall through.
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(os) = &self.oneshot
            && buf.data.len() * 2 <= os.scratch_bytes()
        {
            return os.all_reduce_sum_inplace(ctx, buf);
        }
        self.all_reduce_sum_over(&self.comm, ctx, buf)
    }

    /// Token-owned shard + all-reduce: split `input`'s `[seq_len]` rows into
    /// contiguous `world_size`-way blocks (`div_ceil`), run `compute_fn` over
    /// just this rank's owned rows, scatter the result into `out`'s owned
    /// slice, then all-reduce so every rank ends up holding the full
    /// `[seq_len]` result (unowned rows are zero going into the reduction).
    ///
    /// Used by DSv4's DeepEP-LL token-owned MoE dispatch (both prefill and
    /// decode call sites) for its shard/zero/scatter/all-reduce bookkeeping.
    /// (The Waterfill shared-expert shard also routed through this primitive
    /// briefly, then was KILLed 2026-07-06 for no measurable perf gain — see
    /// `docs/experience/wins/2026-07-06-dsv4-deepep-waterfill-pending-remote.md`.)
    ///
    /// `compute_fn(owned_in, owned_out, start, owned_n)` always runs, even at
    /// `owned_n == 0` (`seq_len < world_size` starves some ranks): DeepEP LL
    /// dispatch/combine are collectives that every rank must enter regardless.
    /// `start` is the owned range's first row index (into `input`'s
    /// `[seq_len]` rows) — callers slicing a parallel per-token array (e.g.
    /// token ids) need it alongside `owned_n`.
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

    /// All-reduce (sum) in place over an explicit communicator (the global TP
    /// comm, or one of the [`Self::attn_tp`] / [`Self::moe_ep`] sub-comms).
    /// Same body as [`Self::all_reduce_sum`] but on the passed `comm`.
    /// `// T2.3 will consume this` — DSv4 DP-attention token-owned EP reductions.
    ///
    /// # Errors
    /// Propagates the NCCL all-reduce error on multi-rank builds.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    // T2.3 will consume this
    // Without `nccl` the only arm is `Single => Ok(())`, so the args are unused;
    // that is the intended single-GPU no-op, not a bug.
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    pub fn all_reduce_sum_over(
        &self,
        comm: &TpComm,
        ctx: &cuda_kernels::prelude::DeviceContext,
        buf: &mut cuda_kernels::prelude::HiddenStates,
    ) -> anyhow::Result<()> {
        match comm {
            TpComm::Single => Ok(()),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::{CollectiveBackend, DType, ReduceOp};
                use cudarc::driver::DevicePtrMut;

                let count = buf.data.len();
                let (ptr, _guard) = buf.data.device_ptr_mut(&ctx.stream);
                // SAFETY: `ptr` is a valid device allocation of `count` BF16
                // elements on this context's device; `ctx.stream` is a stream on
                // the same device. The `_guard` keeps the slice borrowed (and thus
                // un-reallocated) for the duration of the FFI call.
                unsafe {
                    backend.all_reduce(
                        ptr as *mut std::ffi::c_void,
                        count,
                        DType::BF16,
                        ReduceOp::Sum,
                        ctx.stream.cu_stream().cast::<std::ffi::c_void>(),
                    )?;
                }
                Ok(())
            }
        }
    }

    /// All-reduce MIN of one host `i32` across ranks (1-element device round
    /// trip). [`TpComm::Single`] is the identity.
    ///
    /// Use this for capacity decisions that feed the scheduler (e.g. the DSv4
    /// KV-budget slot clamp): each rank's `mem_get_info` can differ, and any
    /// per-rank divergence in scheduler-visible capacity diverges the
    /// deterministic planner and deadlocks NCCL — min-reducing makes the
    /// decision identical on every rank by construction. Collective: every
    /// rank MUST call this at the same construction point.
    ///
    /// # Errors
    /// Propagates device alloc/copy and NCCL errors on multi-rank builds.
    #[cfg(feature = "cuda")]
    // Without `nccl` the only arm is `Single => Ok(value)`, so `ctx` is unused;
    // that is the intended single-GPU identity, not a bug.
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

    /// Raw BF16 all-gather for TP-local attention slabs.
    ///
    /// FlashMLA's sparse decode kernel wants global heads (`h_q` 64/128), while
    /// TP=8 ranks only hold their local head slab. This helper gathers one
    /// local BF16 row from every rank into a rank-major receive buffer; the
    /// caller repacks that buffer into FlashMLA's head-major layout.
    ///
    /// # Safety
    ///
    /// `sendbuf` must point to at least `sendcount` contiguous BF16 elements on
    /// this rank's current CUDA device. `recvbuf` must point to writable device
    /// memory large enough for `sendcount * world_size` BF16 elements. Both
    /// buffers must remain valid until the collective enqueued on `ctx.stream`
    /// completes, and every rank in the TP group must call this method with the
    /// same `sendcount`.
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

                // SAFETY: this fn's contract — `sendbuf` holds `sendcount` bf16 and
                // `recvbuf` `sendcount * world`, both live on `ctx`'s device and stream.
                unsafe {
                    backend.all_gather(
                        sendbuf,
                        recvbuf,
                        sendcount,
                        DType::BF16,
                        ctx.stream.cu_stream().cast::<std::ffi::c_void>(),
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Host-visible all-gather for small byte payloads such as CUDA IPC handles.
    ///
    /// Thin TP-level wrapper over the NCCL backend helper. DeepEP boot uses this
    /// for CUDA IPC handles and device ids after NCCL is initialized.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub fn all_gather_bytes(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        input: &[u8],
        per_rank_bytes: usize,
    ) -> anyhow::Result<Vec<u8>> {
        use anyhow::ensure;

        ensure!(
            input.len() == per_rank_bytes,
            "all_gather_bytes input len {} must equal per-rank bytes {per_rank_bytes}",
            input.len()
        );
        if per_rank_bytes == 0 {
            return Ok(Vec::new());
        }
        match &self.comm {
            TpComm::Single => Ok(input.to_vec()),
            TpComm::Nccl(backend) => backend.all_gather_bytes(ctx, input, per_rank_bytes),
        }
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

    /// Broadcast rank-0's `values` to every rank (rank-0 authoritative).
    ///
    /// Single-rank / non-NCCL builds are identity. DSpark spec-decode uses this
    /// to keep the rank-local proposal (sampling + confidence truncation) and
    /// accept decision consistent: a divergent draft length feeds the
    /// variable-length verify a different token count per rank, mismatching the
    /// per-forward collective count and deadlocking the lockstep coordinator.
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
                // SAFETY: `ptr` is a live out-param and `bind` above made `ctx` current;
                // the callee reads 64 bytes from `handle`, whose length is unchecked here.
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

/// One-shot small-message collective path: vendored sgl-kernel/vLLM custom
/// allreduce + ARLE one-shot all-gather over IPC-shared buffers
/// (`csrc/comm/custom_all_reduce.cu`), staged through one persistent registered
/// scratch per rank. Decode-chain messages (2×AR 14 KB + Q-AG 18 KB per layer)
/// measured 2.6–3.6× faster than NCCL on 8×H20
/// (`wins/2026-06-10-dsv4-comm-bench-oneshot-licensed.md`).
///
/// Boot is COLLECTIVE-SAFE by construction: every local step is followed by an
/// all-ranks ok-vote (1-byte allgather), so one rank failing alloc/IPC/create
/// degrades EVERY rank to NCCL instead of desyncing the boot collectives.
#[cfg(all(feature = "cuda", feature = "nccl"))]
mod oneshot {
    use super::cuda_ipc::{PeerMapping, SharedRegion, check};
    use anyhow::{Result, anyhow, ensure};
    use cuda_kernels::collective::NcclBackend;
    use cuda_kernels::ffi::comm as car;
    use cuda_kernels::prelude::DeviceContext;
    use std::ffi::c_void;

    /// Persistent registered scratch per rank. Covers every decode shape
    /// (B=1..32 AR ≤ 448 KB, Q-AG ≤ 18 KB/rank); prefill-sized buffers fall
    /// through to NCCL at the call sites.
    const SCRATCH_BYTES: usize = 512 * 1024;
    /// Signal region = sgl `Signal` struct (≈3.5 KB) + two-shot staging tmp;
    /// 8 KB margin keeps us independent of upstream Signal layout drift.
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
            // Frees own regions + rank data, IPC-closes peer mappings.
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
                // SAFETY: non-null means boot failed before `into_comm` moved ownership,
                // so this `arle_car_create` handle is still ours to destroy.
                unsafe { car::arle_car_destroy_prod(self.handle) };
            }
        }
    }

    /// All-ranks ok-vote (4-byte payload — `all_gather_bytes` stages through
    /// an i32 device buffer and requires 4-byte alignment). Returns whether
    /// EVERY rank reported ok — the same verdict on every rank, so degrade
    /// decisions stay collective.
    fn vote(ctx: &DeviceContext, backend: &NcclBackend, ok: bool) -> Result<bool> {
        let payload = [u8::from(ok), 0, 0, 0];
        let gathered = backend.all_gather_bytes(ctx, &payload, 4)?;
        Ok(gathered.chunks(4).all(|c| c[0] == 1))
    }

    impl OneShotComm {
        pub(super) fn scratch_bytes(&self) -> usize {
            SCRATCH_BYTES
        }

        /// Collective constructor: every rank must call at the same point.
        /// Returns `Ok(None)` when any rank voted a local failure (all ranks
        /// then degrade together); `Err` only on collective-infrastructure
        /// failures (which already imply a broken NCCL comm).
        pub(super) fn build(
            ctx: &DeviceContext,
            backend: &NcclBackend,
            world: usize,
            rank: usize,
        ) -> Result<Option<Self>> {
            // Stage 1: own allocations (local) + vote.
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

            // Stage 2: handle exchange (collective; all ranks reach it).
            let mut payload = [0u8; 128];
            payload[..64].copy_from_slice(&sig_handle);
            payload[64..].copy_from_slice(&in_handle);
            let all = backend.all_gather_bytes(ctx, &payload, 128)?;
            ensure!(all.len() == world * 128, "one-shot handle exchange size");

            // Stage 3: open peers + create (local) + vote.
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
                // SAFETY: `sigs`/`ins` are live `world`-length arrays of device pointers —
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

            // Stage 4: self-test (one-shot vs NCCL on identical inputs +
            // cross-rank digest) + final vote. Catches memory-ordering issues
            // on this driver/HW combo before any production traffic.
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
            // One-shot arm: fill the registered scratch, reduce into `got`.
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
                    // SAFETY: `got_ptr` is live for `_g` and holds ELEMS u16; `handle` is
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
            // NCCL reference: identical fill (same seed), in-place AR.
            let mut reference = stream
                .alloc_zeros::<u16>(ELEMS)
                .map_err(|e| anyhow!("self-test ref alloc: {e}"))?;
            {
                let (ref_ptr, _g) = reference.device_ptr_mut(stream);
                check(
                    // SAFETY: `ref_ptr` is live for `_g` and holds ELEMS u16 — same seed
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
                // SAFETY: `ref_ptr` is live for `_g` and holds ELEMS bf16 on this rank's
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
            // Cross-rank identity (lockstep hard requirement): FNV digest of
            // the one-shot output must be byte-identical on every rank.
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

        /// In-place bf16 sum allreduce: stage `buf` into the registered
        /// scratch, one-shot/two-shot kernel writes the reduced result back
        /// into `buf`. Caller guarantees `len*2 <= scratch_bytes()`.
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

        /// Raw bf16 all-gather: stage the local chunk into the registered
        /// scratch, the one-shot kernel writes `[world × sendcount]` rank-major
        /// into `recvbuf` (ncclAllGather layout). Caller guarantees
        /// `sendcount*2 <= scratch_bytes()` and `recvbuf` capacity.
        ///
        /// # Safety
        /// Same contract as the NCCL `all_gather` arm: both pointers are live
        /// device allocations on this rank's device, sized as documented.
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
