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
}

impl TpRuntime {
    /// Single-GPU runtime (no parallelism, no-op communicator).
    #[must_use]
    pub fn single() -> Self {
        Self {
            config: TpConfig::single(),
            comm: TpComm::single(),
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
        }
    }

    /// Build a runtime from a config and an explicit communicator.
    #[must_use]
    pub fn with_comm(config: TpConfig, comm: TpComm) -> Self {
        Self { config, comm }
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
        let config = resolve_tp_config_from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
        if config.is_single() {
            return Ok(Self::new(config));
        }
        let backend = cuda_kernels::collective::NcclBackend::init_rank(
            unique_id,
            config.world_size,
            config.rank,
        )?;
        Ok(Self::with_comm(config, TpComm::Nccl(Box::new(backend))))
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
    #[cfg(feature = "cuda")]
    // Without `nccl` the only arm is `Single => Ok(())`, so the args are unused;
    // that is the intended single-GPU no-op, not a bug.
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    pub fn all_reduce_sum(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        buf: &mut cuda_kernels::prelude::HiddenStates,
    ) -> anyhow::Result<()> {
        match &self.comm {
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

    /// Raw BF16 all-gather for TP-local attention slabs.
    ///
    /// FlashMLA's sparse decode kernel wants global heads (`h_q` 64/128), while
    /// TP=8 ranks only hold their local head slab. This helper gathers one
    /// local BF16 row from every rank into a rank-major receive buffer; the
    /// caller repacks that buffer into FlashMLA's head-major layout.
    #[cfg(feature = "cuda")]
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))]
    pub unsafe fn all_gather_bf16_raw(
        &self,
        ctx: &cuda_kernels::prelude::DeviceContext,
        sendbuf: *const std::ffi::c_void,
        sendcount: usize,
        recvbuf: *mut std::ffi::c_void,
    ) -> anyhow::Result<()> {
        match &self.comm {
            TpComm::Single => anyhow::bail!("single-rank raw all_gather_bf16 is not needed"),
            #[cfg(feature = "nccl")]
            TpComm::Nccl(backend) => {
                use cuda_kernels::collective::{CollectiveBackend, DType};

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup_from(pairs: &[(&str, &str)]) -> impl FnMut(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |key| map.get(key).cloned()
    }

    #[test]
    fn resolve_defaults_to_single_gpu_when_no_env() {
        let cfg = resolve_tp_config(lookup_from(&[])).unwrap();
        assert_eq!(cfg, TpConfig::single());
        assert!(cfg.is_single());
    }

    #[test]
    fn resolve_uses_infer_cuda_devices_count_as_world_size() {
        // 8 ordinals ⇒ TP=8 trigger (project memory: INFER_CUDA_DEVICES is the TP=8 trigger).
        let cfg = resolve_tp_config(lookup_from(&[
            ("INFER_CUDA_DEVICES", "0,1,2,3,4,5,6,7"),
            ("INFER_TP_RANK", "3"),
        ]))
        .unwrap();
        assert_eq!(cfg, TpConfig::new(8, 3).unwrap());
    }

    #[test]
    fn resolve_explicit_tp_size_overrides_device_count() {
        // Explicit INFER_TP_SIZE wins over the device-list count.
        let cfg = resolve_tp_config(lookup_from(&[
            ("INFER_CUDA_DEVICES", "0,1,2,3"),
            ("INFER_TP_SIZE", "2"),
            ("INFER_TP_RANK", "1"),
        ]))
        .unwrap();
        assert_eq!(cfg, TpConfig::new(2, 1).unwrap());
    }

    #[test]
    fn resolve_accepts_arle_aliases() {
        let cfg = resolve_tp_config(lookup_from(&[("ARLE_TP_SIZE", "4"), ("ARLE_TP_RANK", "2")]))
            .unwrap();
        assert_eq!(cfg, TpConfig::new(4, 2).unwrap());
    }

    #[test]
    fn resolve_ignores_empty_device_entries() {
        // Whitespace / empty entries don't inflate the world size.
        let cfg = resolve_tp_config(lookup_from(&[("INFER_CUDA_DEVICES", "0, ,1, 2 ,")])).unwrap();
        assert_eq!(cfg, TpConfig::new(3, 0).unwrap());
    }

    #[test]
    fn resolve_rejects_rank_out_of_range() {
        let err = resolve_tp_config(lookup_from(&[
            ("INFER_TP_SIZE", "4"),
            ("INFER_TP_RANK", "4"),
        ]));
        assert!(err.is_err());
    }

    #[test]
    fn runtime_single_is_no_op() {
        let rt = TpRuntime::single();
        assert!(rt.is_single());
        assert!(!rt.is_collective());
        assert_eq!(*rt.config(), TpConfig::single());
    }

    #[test]
    fn runtime_multi_rank_config_with_noop_comm() {
        let cfg = TpConfig::new(8, 0).unwrap();
        let rt = TpRuntime::new(cfg);
        assert!(!rt.is_single());
        // No NCCL feature in this build ⇒ the no-op communicator: no collectives.
        assert!(!rt.is_collective());
        assert_eq!(rt.config().world_size, 8);
    }

    #[test]
    fn single_runtime_is_not_collective_so_graph_guard_keeps_capture() {
        // The executor disables the decode-graph capture iff `is_collective()`.
        // A single-GPU runtime must report `false` so capture stays enabled and
        // the single-GPU forward (no all-reduce) is unchanged from the pre-TP path.
        let rt = TpRuntime::single();
        assert!(
            !rt.is_collective(),
            "single GPU: graph capture stays enabled"
        );
        assert!(!rt.comm().is_collective());
    }

    // CPU mock-communicator parity: for a row-parallel `y = x @ W^T`, the sum
    // over ranks of each rank's partial GEMM equals the unsharded GEMM.

    /// CPU stand-in for the all-reduce: sums per-rank host output vectors.
    struct MockCommunicator;

    impl MockCommunicator {
        /// All-reduce (sum) over the per-rank partial output vectors.
        fn all_reduce_sum(shards: &[Vec<f32>]) -> Vec<f32> {
            let out_dim = shards.first().map_or(0, Vec::len);
            let mut acc = vec![0.0f32; out_dim];
            for shard in shards {
                assert_eq!(shard.len(), out_dim, "all output vectors must match");
                for (a, &v) in acc.iter_mut().zip(shard) {
                    *a += v;
                }
            }
            acc
        }
    }

    /// Dense reference: `y[o] = sum_i x[i] * w[o, i]`. `w` is row-major
    /// `[out_dim, in_dim]` (HF nn.Linear layout); `x` is `[in_dim]`.
    fn dense_gemv(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        assert_eq!(x.len(), in_dim);
        assert_eq!(w.len(), out_dim * in_dim);
        (0..out_dim)
            .map(|o| (0..in_dim).map(|i| x[i] * w[o * in_dim + i]).sum())
            .collect()
    }

    #[test]
    fn row_parallel_sharded_gemm_all_reduces_to_unsharded() {
        // Row-parallel `y = x @ W^T`: the input dim is split across TP ranks.
        // Each rank holds x[shard] and the W columns for that shard, computes a
        // partial y, and the all-reduce sums the partials into the full y.
        let out_dim = 5usize;
        let in_dim = 12usize;
        let world = 4usize;

        // Deterministic fake x and W.
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|n| ((n % 7) as f32) - 3.0)
            .collect();

        let full = dense_gemv(&x, &w, out_dim, in_dim);

        // Per-rank partials over the input-dim shard from infer_topo::row_shard.
        let mut partials = Vec::with_capacity(world);
        let mut covered = 0usize;
        for rank in 0..world {
            let tp = TpConfig::new(world, rank).unwrap();
            let spec = infer_topo::row_shard(in_dim, &tp);
            covered += spec.size;

            // x restricted to this rank's input columns.
            let x_shard: Vec<f32> = x[spec.range()].to_vec();
            // W restricted to the same input columns, kept row-major [out, size].
            let mut w_shard = Vec::with_capacity(out_dim * spec.size);
            for o in 0..out_dim {
                for i in spec.range() {
                    w_shard.push(w[o * in_dim + i]);
                }
            }
            partials.push(dense_gemv(&x_shard, &w_shard, out_dim, spec.size));
        }
        // The shards exactly cover the input dim (no overlap, no gap).
        assert_eq!(covered, in_dim);

        let reduced = MockCommunicator::all_reduce_sum(&partials);
        assert_eq!(reduced.len(), full.len());
        for (got, want) in reduced.iter().zip(&full) {
            assert!(
                (got - want).abs() < 1e-3,
                "sharded+reduced gemm {got} != unsharded {want}"
            );
        }
    }

    #[test]
    fn column_parallel_shard_concat_reconstructs_unsharded_output() {
        // Column-parallel `y = x @ W^T` splits the OUTPUT dim: each rank computes
        // a disjoint slice of y from its W rows; concatenation (gather) — not
        // all-reduce — reconstructs the full output. This is the dual of the
        // row-parallel all-reduce check above.
        let out_dim = 8usize;
        let in_dim = 6usize;
        let world = 4usize;

        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) - 2.0).collect();
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|n| ((n % 5) as f32) - 2.0)
            .collect();
        let full = dense_gemv(&x, &w, out_dim, in_dim);

        let mut gathered = vec![0.0f32; out_dim];
        for rank in 0..world {
            let tp = TpConfig::new(world, rank).unwrap();
            let spec = infer_topo::column_shard(out_dim, &tp);
            // W rows for this rank's output slice (contiguous), full input dim.
            let w_shard = &w[spec.offset * in_dim..spec.end() * in_dim];
            let y_shard = dense_gemv(&x, w_shard, spec.size, in_dim);
            gathered[spec.range()].copy_from_slice(&y_shard);
        }
        for (got, want) in gathered.iter().zip(&full) {
            assert!((got - want).abs() < 1e-3, "gathered {got} != full {want}");
        }
    }
}
