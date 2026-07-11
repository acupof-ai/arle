//! CUDA runtime toggles: `--flag` → `EngineLoadConfig.cuda` →
//! [`apply_runtime_flags`] once per process (multiproc workers included, via
//! `ARLE_WORKER_ENGINE_CONFIG`) BEFORE executor/context construction. The
//! statics are the single truth — no env reads.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::Relaxed};

use infer_seam::{CommBackend, CudaRuntimeFlags};

/// `usize::MAX` / `0` sentinels mean "unset → built-in default".
static SHARD_CACHE_BYTES: AtomicUsize = AtomicUsize::new(usize::MAX);
static DEEPEP_MAX_DISPATCH_TOKENS_PER_RANK: AtomicU32 = AtomicU32::new(0);

static QWEN35_DECODE_GRAPH: AtomicBool = AtomicBool::new(false);
static QWEN35_BATCHED_DECODE: AtomicBool = AtomicBool::new(true);
static QWEN35_BATCHED_DECODE_ATTENTION: AtomicBool = AtomicBool::new(true);
static QWEN35_DEEPGEMM: AtomicBool = AtomicBool::new(true);
static QWEN35_MOE_DECODE_KERNEL: AtomicBool = AtomicBool::new(true);
static QWEN35_GPU_ROUTER: AtomicBool = AtomicBool::new(true);
static QWEN35_FA3: AtomicBool = AtomicBool::new(true);
static QWEN35_FA3_DECODE: AtomicBool = AtomicBool::new(false);
static QWEN35_FA3_DECODE_SPLITS: AtomicUsize = AtomicUsize::new(8);
static QWEN35_GDR_CHUNKED: AtomicBool = AtomicBool::new(false);
static NUMA_PIN: AtomicBool = AtomicBool::new(true);
static COMM_NCCL_ONLY: AtomicBool = AtomicBool::new(false);
static DSV4_DSA_INDEXER_SMS: AtomicUsize = AtomicUsize::new(78);
static DSV4_MOE_CONTIG_DECODE: AtomicBool = AtomicBool::new(false);
static DSV4_DECODE_REUSE: AtomicBool = AtomicBool::new(true); // default ON (2026-07-11 pod license)
static MTP_ADAPTIVE: AtomicBool = AtomicBool::new(false);
static MTP_MIN_ACCEPT_BITS: AtomicU32 = AtomicU32::new(0x3F0C_CCCD); // 0.55f32
static DEEPEP_NUM_SMS: AtomicU32 = AtomicU32::new(20);

/// Apply the CLI-resolved flags. Must run before executor construction; the
/// mempool-retain knob additionally requires it before the FIRST
/// `DeviceContext` creation.
pub fn apply_runtime_flags(f: &CudaRuntimeFlags) {
    QWEN35_DECODE_GRAPH.store(f.qwen35_decode_graph, Relaxed);
    QWEN35_BATCHED_DECODE.store(f.qwen35_batched_decode, Relaxed);
    QWEN35_BATCHED_DECODE_ATTENTION.store(f.qwen35_batched_decode_attention, Relaxed);
    QWEN35_DEEPGEMM.store(f.qwen35_deepgemm, Relaxed);
    QWEN35_MOE_DECODE_KERNEL.store(f.qwen35_moe_decode_kernel, Relaxed);
    QWEN35_GPU_ROUTER.store(f.qwen35_gpu_router, Relaxed);
    QWEN35_FA3.store(f.qwen35_fa3, Relaxed);
    QWEN35_FA3_DECODE.store(f.qwen35_fa3_decode, Relaxed);
    QWEN35_FA3_DECODE_SPLITS.store(f.qwen35_fa3_decode_splits.clamp(2, 256), Relaxed);
    QWEN35_GDR_CHUNKED.store(f.qwen35_gdr_chunked, Relaxed);
    SHARD_CACHE_BYTES.store(f.shard_cache_bytes.unwrap_or(usize::MAX), Relaxed);
    NUMA_PIN.store(f.numa_pin, Relaxed);
    COMM_NCCL_ONLY.store(f.comm_backend == CommBackend::Nccl, Relaxed);
    DSV4_DSA_INDEXER_SMS.store(f.dsv4_dsa_indexer_sms, Relaxed);
    DSV4_MOE_CONTIG_DECODE.store(f.dsv4_moe_contig_decode, Relaxed);
    DSV4_DECODE_REUSE.store(f.dsv4_decode_reuse, Relaxed);
    MTP_ADAPTIVE.store(f.mtp_adaptive, Relaxed);
    MTP_MIN_ACCEPT_BITS.store(f.mtp_min_accept.to_bits(), Relaxed);
    DEEPEP_NUM_SMS.store(f.deepep_num_sms, Relaxed);
    DEEPEP_MAX_DISPATCH_TOKENS_PER_RANK
        .store(f.deepep_max_dispatch_tokens_per_rank.unwrap_or(0), Relaxed);
    // `Some(true)` stays compile-gated: forcing FlashMLA on a stub build must
    // resolve to the scalar fallback, not a missing-kernel error.
    crate::attention::set_dsv4_flashmla_decode_override(
        f.dsv4_flashmla_decode
            .map(|on| on && cuda_kernels::HAS_FLASHMLA),
    );
    cuda_kernels::tilelang::set_decode_metadata_fast_page16(f.decode_metadata_fast_page16);
    cuda_kernels::tensor::set_marlin_w4_fp8_prefill(f.marlin_w4_fp8_prefill);
    cuda_kernels::tensor::set_mempool_retain(f.mempool_retain);
}

pub(crate) fn qwen35_decode_graph() -> bool {
    QWEN35_DECODE_GRAPH.load(Relaxed)
}
pub(crate) fn qwen35_batched_decode() -> bool {
    QWEN35_BATCHED_DECODE.load(Relaxed)
}
pub(crate) fn qwen35_batched_decode_attention() -> bool {
    QWEN35_BATCHED_DECODE_ATTENTION.load(Relaxed)
}
pub(crate) fn qwen35_deepgemm() -> bool {
    QWEN35_DEEPGEMM.load(Relaxed)
}
pub(crate) fn qwen35_moe_decode_kernel() -> bool {
    QWEN35_MOE_DECODE_KERNEL.load(Relaxed)
}
pub(crate) fn qwen35_gpu_router() -> bool {
    QWEN35_GPU_ROUTER.load(Relaxed)
}
pub(crate) fn qwen35_fa3() -> bool {
    QWEN35_FA3.load(Relaxed)
}
pub(crate) fn qwen35_fa3_decode() -> bool {
    QWEN35_FA3_DECODE.load(Relaxed)
}
pub(crate) fn qwen35_fa3_decode_splits() -> usize {
    QWEN35_FA3_DECODE_SPLITS.load(Relaxed)
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
pub(crate) fn dsv4_moe_contig_decode() -> bool {
    DSV4_MOE_CONTIG_DECODE.load(Relaxed)
}
/// A/B-harness lever (`infer_cuda::set_dsv4_moe_contig_decode`).
pub(crate) fn set_dsv4_moe_contig_decode(enabled: bool) {
    DSV4_MOE_CONTIG_DECODE.store(enabled, Relaxed);
}
pub(crate) fn dsv4_decode_reuse_enabled() -> bool {
    DSV4_DECODE_REUSE.load(Relaxed)
}
pub(crate) fn mtp_adaptive() -> bool {
    MTP_ADAPTIVE.load(Relaxed)
}
pub(crate) fn mtp_min_accept() -> f32 {
    f32::from_bits(MTP_MIN_ACCEPT_BITS.load(Relaxed))
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
