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
static QWEN35_DEEPGEMM: AtomicBool = AtomicBool::new(true);
static QWEN35_MOE_DECODE_KERNEL: AtomicBool = AtomicBool::new(true);
static QWEN35_FA3: AtomicBool = AtomicBool::new(true);
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
static MTP_ADAPTIVE: AtomicBool = AtomicBool::new(false);
static MTP_MIN_ACCEPT_BITS: AtomicU32 = AtomicU32::new(0x3F0C_CCCD); // 0.55f32
static SPEC_MAX_BATCH: AtomicUsize = AtomicUsize::new(16);
/// NaN = unset, so an explicit `0` stays distinguishable from no flag at all.
static DSPARK_CONFIDENCE_THRESHOLD_BITS: AtomicU32 = AtomicU32::new(0x7FC0_0000);
static DEEPEP_NUM_SMS: AtomicU32 = AtomicU32::new(20);

/// Apply the CLI-resolved flags. Must run before executor construction; the
/// mempool-retain knob additionally requires it before the FIRST
/// `DeviceContext` creation.
pub fn apply_runtime_flags(f: &CudaRuntimeFlags) {
    QWEN35_DECODE_GRAPH.store(f.qwen35_decode_graph, Relaxed);
    QWEN35_DEEPGEMM.store(f.qwen35_deepgemm, Relaxed);
    QWEN35_MOE_DECODE_KERNEL.store(f.qwen35_moe_decode_kernel, Relaxed);
    QWEN35_FA3.store(f.qwen35_fa3, Relaxed);
    QWEN35_DEEPGEMM_MIN_ROUTES.store(f.qwen35_deepgemm_min_routes.max(1), Relaxed);
    QWEN35_GDR_CHUNKED.store(f.qwen35_gdr_chunked, Relaxed);
    SHARD_CACHE_BYTES.store(f.shard_cache_bytes.unwrap_or(usize::MAX), Relaxed);
    NUMA_PIN.store(f.numa_pin, Relaxed);
    COMM_NCCL_ONLY.store(f.comm_backend == CommBackend::Nccl, Relaxed);
    DSV4_DSA_INDEXER_SMS.store(f.dsv4_dsa_indexer_sms, Relaxed);
    MTP_ADAPTIVE.store(f.mtp_adaptive, Relaxed);
    MTP_MIN_ACCEPT_BITS.store(f.mtp_min_accept.to_bits(), Relaxed);
    SPEC_MAX_BATCH.store(f.spec_max_batch.max(1), Relaxed);
    DSPARK_CONFIDENCE_THRESHOLD_BITS.store(
        f.dspark_confidence_threshold.unwrap_or(f32::NAN).to_bits(),
        Relaxed,
    );
    DEEPEP_NUM_SMS.store(f.deepep_num_sms, Relaxed);
    DSV4_MOE_TRANSPORT_CLI.get_or_init(|| f.dsv4_moe_transport.clone());
    DEEPEP_MAX_DISPATCH_TOKENS_PER_RANK
        .store(f.deepep_max_dispatch_tokens_per_rank.unwrap_or(0), Relaxed);
    // `Some(true)` stays compile-gated: forcing FlashMLA on a stub build must
    // resolve to the scalar fallback, not a missing-kernel error.
    crate::attention::set_dsv4_flashmla_decode_override(
        f.dsv4_flashmla_decode
            .map(|on| on && cuda_kernels::HAS_FLASHMLA),
    );
    cuda_kernels::tensor::set_mempool_retain(f.mempool_retain);
}

pub(crate) fn qwen35_decode_graph() -> bool {
    QWEN35_DECODE_GRAPH.load(Relaxed)
}
/// `--qwen35-deepgemm` (default on): DeepGEMM SM90 BF16 m-grouped GEMMs for the
/// expert GEMMs — decode neutral, prefill needle 3k wall 9.10 -> 2.32 s. Also
/// read at LOAD time (the loader builds the contiguous grouped-B caches only
/// when enabled), so flipping it requires a process restart.
pub(crate) fn qwen35_deepgemm() -> bool {
    QWEN35_DEEPGEMM.load(Relaxed)
}
/// `--qwen35-moe-decode-kernel` (default on): the decode-band weight-read-bound
/// grouped kernels; `false` runs the hand batch kernels below the DeepGEMM floor.
/// Read per call — inside a captured decode graph the value read at capture
/// time is what replays.
pub(crate) fn qwen35_moe_decode_kernel() -> bool {
    QWEN35_MOE_DECODE_KERNEL.load(Relaxed)
}
pub(crate) fn qwen35_fa3() -> bool {
    QWEN35_FA3.load(Relaxed)
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
pub(crate) fn mtp_adaptive() -> bool {
    MTP_ADAPTIVE.load(Relaxed)
}
/// DSv4 decode reuse is unconditional (2026-07-11 license); the accessor stays
/// because executor/dsv4.rs still names it.
pub(crate) fn dsv4_decode_reuse_enabled() -> bool {
    true
}
pub(crate) fn mtp_min_accept() -> f32 {
    f32::from_bits(MTP_MIN_ACCEPT_BITS.load(Relaxed))
}
/// Concurrency gate for speculative decode (`--spec-max-batch`): speculate only
/// when the decode batch is ≤ this, else route to the plain batched path.
pub(crate) fn spec_max_batch() -> usize {
    SPEC_MAX_BATCH.load(Relaxed)
}
pub(crate) fn dspark_confidence_threshold() -> Option<f32> {
    let t = f32::from_bits(DSPARK_CONFIDENCE_THRESHOLD_BITS.load(Relaxed));
    (!t.is_nan()).then_some(t)
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
