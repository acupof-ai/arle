//! CUDA runtime toggles: `--flag` → `EngineLoadConfig.cuda` →
//! [`apply_runtime_flags`] once per process (multiproc workers included, via
//! `ARLE_WORKER_ENGINE_CONFIG`) BEFORE executor/context construction. The
//! statics are the single truth — no env reads.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::Relaxed};

use anyhow::{Result, bail};
use infer_seam::{CommBackend, CudaRuntimeFlags};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dsv4MoeTransport {
    AllReduce,
    DeepEp,
    DeepEpLowLatency,
    MegaMoe,
}

impl Dsv4MoeTransport {
    pub(crate) fn is_deepep(self) -> bool {
        matches!(self, Self::DeepEp | Self::DeepEpLowLatency)
    }
}

static DSV4_MOE_TRANSPORT: OnceLock<Result<Dsv4MoeTransport, String>> = OnceLock::new();
static DSV4_MOE_TRANSPORT_CLI: OnceLock<Option<String>> = OnceLock::new();

pub(crate) fn dsv4_moe_transport() -> Result<Dsv4MoeTransport> {
    match DSV4_MOE_TRANSPORT.get_or_init(|| {
        let value = DSV4_MOE_TRANSPORT_CLI
            .get()
            .and_then(|v| v.clone())
            .or_else(|| std::env::var("ARLE_DSV4_MOE_TRANSPORT").ok())
            .unwrap_or_else(|| "allreduce".to_string());
        match value.as_str() {
            "allreduce" | "all_reduce" | "native" | "scalar" | "static" | "deepgemm" | "" => {
                Ok(Dsv4MoeTransport::AllReduce)
            }
            "deepep" | "native-deepep" | "native_deepep" => Ok(Dsv4MoeTransport::DeepEp),
            "deepep_ll" | "deepep-ll" | "deepep_low_latency" | "native_deepep_ll" => {
                Ok(Dsv4MoeTransport::DeepEpLowLatency)
            }
            "mega_moe" => Ok(Dsv4MoeTransport::MegaMoe),
            other => Err(format!(
                "unsupported DSv4 MoE transport `{other}` \
                 (--dsv4-moe-transport or ARLE_DSV4_MOE_TRANSPORT; \
                 expected allreduce, deepep, deepep_ll, or mega_moe)"
            )),
        }
    }) {
        Ok(mode) => Ok(*mode),
        Err(msg) => bail!("{msg}"),
    }
}

/// `usize::MAX` / `0` sentinels mean "unset → built-in default".
static SHARD_CACHE_BYTES: AtomicUsize = AtomicUsize::new(usize::MAX);
static DEEPEP_MAX_DISPATCH_TOKENS_PER_RANK: AtomicU32 = AtomicU32::new(0);

static QWEN35_DECODE_GRAPH: AtomicBool = AtomicBool::new(true);
static QWEN35_DEEPGEMM_MIN_ROUTES: AtomicUsize = AtomicUsize::new(1024);
static QWEN35_GDR_CHUNKED: AtomicBool = AtomicBool::new(true);
static NUMA_PIN: AtomicBool = AtomicBool::new(true);
static COMM_NCCL_ONLY: AtomicBool = AtomicBool::new(false);
static DSV4_DSA_INDEXER_SMS: AtomicUsize = AtomicUsize::new(78);
// Setter-only (NOT in `apply_runtime_flags`): the OPD trainer flips it right
// before the rollout student loads, and the engine's own `apply_runtime_flags`
// during load must not reset it. Off = serving default (grouped-FP8 experts).
static QWEN35_MOE_EXPERTS_BF16_RESIDENT: AtomicBool = AtomicBool::new(false);
// The M envelope of the engine being built: the most rows a decode step can
// present, and the most a prefill chunk can. `0` = undeclared.
static DENSE_GEMM_DECODE_ROWS: AtomicUsize = AtomicUsize::new(0);
static DENSE_GEMM_PREFILL_ROWS: AtomicUsize = AtomicUsize::new(0);
static SPEC_MAX_BATCH: AtomicUsize = AtomicUsize::new(16);
static DEEPEP_NUM_SMS: AtomicU32 = AtomicU32::new(20);

/// Apply the CLI-resolved flags. Must run before executor construction; the
/// mempool-retain knob additionally requires it before the FIRST
/// `DeviceContext` creation.
pub fn apply_runtime_flags(f: &CudaRuntimeFlags) {
    QWEN35_DECODE_GRAPH.store(f.qwen35_decode_graph, Relaxed);
    QWEN35_DEEPGEMM_MIN_ROUTES.store(f.qwen35_deepgemm_min_routes.max(1), Relaxed);
    QWEN35_GDR_CHUNKED.store(f.qwen35_gdr_chunked, Relaxed);
    SHARD_CACHE_BYTES.store(f.shard_cache_bytes.unwrap_or(usize::MAX), Relaxed);
    NUMA_PIN.store(f.numa_pin, Relaxed);
    COMM_NCCL_ONLY.store(f.comm_backend == CommBackend::Nccl, Relaxed);
    DSV4_DSA_INDEXER_SMS.store(f.dsv4_dsa_indexer_sms, Relaxed);
    SPEC_MAX_BATCH.store(f.spec_max_batch.max(1), Relaxed);
    DEEPEP_NUM_SMS.store(f.deepep_num_sms, Relaxed);
    DSV4_MOE_TRANSPORT_CLI.get_or_init(|| f.dsv4_moe_transport.clone());
    DEEPEP_MAX_DISPATCH_TOKENS_PER_RANK
        .store(f.deepep_max_dispatch_tokens_per_rank.unwrap_or(0), Relaxed);
    cuda_kernels::tensor::set_mempool_retain(f.mempool_retain);
}

pub(crate) fn qwen35_decode_graph() -> bool {
    QWEN35_DECODE_GRAPH.load(Relaxed)
}
pub(crate) fn qwen35_deepgemm_min_routes() -> usize {
    QWEN35_DEEPGEMM_MIN_ROUTES.load(Relaxed)
}
pub(crate) fn qwen35_gdr_chunked() -> bool {
    QWEN35_GDR_CHUNKED.load(Relaxed)
}
pub(crate) fn shard_cache_bytes() -> Option<usize> {
    match SHARD_CACHE_BYTES.load(Relaxed) {
        usize::MAX => None,
        bytes => Some(bytes),
    }
}
// Consumers are linux/nccl-gated.
#[cfg_attr(any(not(target_os = "linux"), not(feature = "nccl")), allow(dead_code))]
pub(crate) fn numa_pin() -> bool {
    NUMA_PIN.load(Relaxed)
}
#[cfg_attr(not(feature = "nccl"), allow(dead_code))]
pub(crate) fn comm_nccl_only() -> bool {
    COMM_NCCL_ONLY.load(Relaxed)
}
pub(crate) fn dsv4_dsa_indexer_sms() -> usize {
    DSV4_DSA_INDEXER_SMS.load(Relaxed)
}
/// When set, the Qwen3.6 MoE loader dequantizes routed FP8 experts to BF16
/// per-expert at load instead of building the fused grouped-FP8 cache, so
/// per-step LoRA re-merge has a mutable per-expert `DeviceMatrix` to fold into.
pub(crate) fn qwen35_moe_experts_bf16_resident() -> bool {
    QWEN35_MOE_EXPERTS_BF16_RESIDENT.load(Relaxed)
}
/// Set by `infer_cuda::set_qwen35_moe_experts_bf16_resident` before the OPD
/// rollout student loads (target set includes experts).
pub(crate) fn set_qwen35_moe_experts_bf16_resident(enabled: bool) {
    QWEN35_MOE_EXPERTS_BF16_RESIDENT.store(enabled, Relaxed);
}
/// `(max decode rows, max prefill rows)`, or `None` while undeclared. A dense
/// GEMM arm that is only correct on one side of the prefill/decode split has no
/// other way to tell them apart — `gemm_batch` sees a row count, not a phase.
pub(crate) fn dense_gemm_row_envelope() -> Option<(usize, usize)> {
    match (
        DENSE_GEMM_DECODE_ROWS.load(Relaxed),
        DENSE_GEMM_PREFILL_ROWS.load(Relaxed),
    ) {
        (0, _) | (_, 0) => None,
        envelope => Some(envelope),
    }
}
/// Set by `infer_cuda::apply_dense_gemm_row_envelope` from the one CUDA engine
/// constructor, before the weights load: the load-time DeepGEMM preparation reads
/// it, so a later call cannot move them. Widens rather than replaces — an OPD
/// process builds a teacher and a student against these same statics, and a
/// floor derived from the narrower of the two would sit inside the other's
/// decode batch.
pub(crate) fn set_dense_gemm_row_envelope(decode_rows: usize, prefill_rows: usize) {
    DENSE_GEMM_DECODE_ROWS.fetch_max(decode_rows, Relaxed);
    DENSE_GEMM_PREFILL_ROWS.fetch_max(prefill_rows, Relaxed);
}
/// Concurrency gate for speculative decode (`--spec-max-batch`): speculate only
/// when the decode batch is ≤ this, else route to the plain batched path.
pub(crate) fn spec_max_batch() -> usize {
    SPEC_MAX_BATCH.load(Relaxed)
}
#[cfg(feature = "deepep")]
pub(crate) fn deepep_num_sms() -> u32 {
    DEEPEP_NUM_SMS.load(Relaxed)
}
#[cfg(feature = "deepep")]
pub(crate) fn deepep_max_dispatch_tokens_per_rank() -> Option<u32> {
    match DEEPEP_MAX_DISPATCH_TOKENS_PER_RANK.load(Relaxed) {
        0 => None,
        v => Some(v),
    }
}
