//! Train-time runtime toggles: `arle train … --flag` →
//! [`apply_runtime_flags`] once at CLI start. The statics are the single
//! truth — no env reads, except [`cp_ring_fa3`]: `ARLE_CP_RING_FA3` is read
//! once so the pod scalar-vs-FA3 A/B runs one binary without CLI re-plumbing.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering::Relaxed};

/// Storage dtype for forward-retained activations + transient emitted grads
/// (`--tape-precision`). CUDA-only; compute (cuBLAS accumulate), persistent grad
/// accumulators, and every fp32 island are unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapePrecision {
    Fp32 = 0,
    Bf16 = 1,
}

impl TapePrecision {
    fn from_u8(v: u8) -> Self {
        if v == 1 { Self::Bf16 } else { Self::Fp32 }
    }
}

/// Autograd knobs the OPD CLI flags control (defaults = shipped behavior).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutogradRuntimeFlags {
    /// Min tensor bytes for checkpoint host offload (`--checkpoint-offload-min-bytes`).
    pub checkpoint_offload_min_bytes: usize,
    /// Row tile for the LoRA linear backward (`--lora-linear-bwd-tile-rows`).
    pub lora_linear_bwd_tile_rows: usize,
    /// Expert tile for the MoE LoRA backward (`--moe-lora-bwd-expert-tile`).
    pub moe_lora_bwd_expert_tile: usize,
    /// FlashQLA chunkwise GDN prefill in the CUDA backend (`--gdr-chunkwise-prefill`).
    pub gdr_chunkwise_prefill: bool,
    /// Force the monolithic chunked-scan linear-attention backward (`--la-backward-mono`).
    pub la_backward_mono: bool,
    /// Force the legacy two-pass decode attention kernel (`--autograd-decode-attn-legacy`).
    pub decode_attn_legacy: bool,
    /// Retain the cuMemAllocAsync pool across syncs (`--cuda-mempool-retain`).
    pub cuda_mempool_retain: bool,
    /// Storage dtype for retained activations + emitted grads (`--tape-precision`).
    pub tape_precision: TapePrecision,
}

impl Default for AutogradRuntimeFlags {
    fn default() -> Self {
        Self {
            checkpoint_offload_min_bytes: 2 << 20,
            lora_linear_bwd_tile_rows: 1024,
            moe_lora_bwd_expert_tile: 16,
            gdr_chunkwise_prefill: false,
            la_backward_mono: false,
            decode_attn_legacy: false,
            cuda_mempool_retain: true,
            tape_precision: TapePrecision::Fp32,
        }
    }
}

static CHECKPOINT_OFFLOAD_MIN_BYTES: AtomicUsize = AtomicUsize::new(2 << 20);
static LORA_LINEAR_BWD_TILE_ROWS: AtomicUsize = AtomicUsize::new(1024);
static MOE_LORA_BWD_EXPERT_TILE: AtomicUsize = AtomicUsize::new(16);
static GDR_CHUNKWISE_PREFILL: AtomicBool = AtomicBool::new(false);
static LA_BACKWARD_MONO: AtomicBool = AtomicBool::new(false);
static DECODE_ATTN_LEGACY: AtomicBool = AtomicBool::new(false);
static TAPE_PRECISION: AtomicU8 = AtomicU8::new(0);

pub fn apply_runtime_flags(f: &AutogradRuntimeFlags) {
    CHECKPOINT_OFFLOAD_MIN_BYTES.store(f.checkpoint_offload_min_bytes, Relaxed);
    LORA_LINEAR_BWD_TILE_ROWS.store(f.lora_linear_bwd_tile_rows.max(1), Relaxed);
    MOE_LORA_BWD_EXPERT_TILE.store(f.moe_lora_bwd_expert_tile.max(1), Relaxed);
    GDR_CHUNKWISE_PREFILL.store(f.gdr_chunkwise_prefill, Relaxed);
    LA_BACKWARD_MONO.store(f.la_backward_mono, Relaxed);
    DECODE_ATTN_LEGACY.store(f.decode_attn_legacy, Relaxed);
    #[cfg(feature = "cuda")]
    cuda_kernels::tensor::set_mempool_retain(f.cuda_mempool_retain);
    TAPE_PRECISION.store(f.tape_precision as u8, Relaxed);
}

pub(crate) fn checkpoint_offload_min_bytes() -> usize {
    CHECKPOINT_OFFLOAD_MIN_BYTES.load(Relaxed)
}
pub(crate) fn lora_linear_bwd_tile_rows() -> usize {
    LORA_LINEAR_BWD_TILE_ROWS.load(Relaxed)
}
pub(crate) fn moe_lora_bwd_expert_tile() -> usize {
    MOE_LORA_BWD_EXPERT_TILE.load(Relaxed)
}
#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn gdr_chunkwise_prefill() -> bool {
    GDR_CHUNKWISE_PREFILL.load(Relaxed)
}
/// Also a test A/B lever (`set_la_backward_mono`).
#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn la_backward_mono() -> bool {
    LA_BACKWARD_MONO.load(Relaxed)
}
#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn decode_attn_legacy() -> bool {
    DECODE_ATTN_LEGACY.load(Relaxed)
}
#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn tape_precision() -> TapePrecision {
    TapePrecision::from_u8(TAPE_PRECISION.load(Relaxed))
}
/// FA3 route for the CP ring per-block kernels. Default OFF until the hd256
/// CP grad-parity gate is green — measured 2.17x/step, but its first correctness
/// A/B ran at hd128 where FA3 never engages. `ARLE_CP_RING_FA3=1` opts in;
/// hd256 / sm90 / real-kernel-marker gating still applies at the callsite.
#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn cp_ring_fa3() -> bool {
    static CP_RING_FA3: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CP_RING_FA3.get_or_init(|| {
        std::env::var("ARLE_CP_RING_FA3")
            .ok()
            .is_some_and(|v| matches!(v.trim(), "1" | "true" | "on"))
    })
}
/// `true` when retained activations + emitted grads store bf16 (CUDA-only path).
#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn tape_bf16() -> bool {
    matches!(tape_precision(), TapePrecision::Bf16)
}
