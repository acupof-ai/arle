//! Train-time runtime toggles: `arle train … --flag` →
//! [`apply_runtime_flags`] once at CLI start. The statics are the single
//! truth — no env reads.

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
    /// Reload a host-offloaded checkpoint to device before its backward replay
    /// (`--checkpoint-reload-device`).
    pub checkpoint_reload_device: bool,
    /// Pinned-host byte budget for parked checkpoints; 0 = pageable path
    /// (`--checkpoint-pinned-offload-bytes`).
    pub checkpoint_pinned_offload_bytes: usize,
    /// Row tile for the LoRA linear backward (`--lora-linear-bwd-tile-rows`).
    pub lora_linear_bwd_tile_rows: usize,
    /// Expert tile for the MoE LoRA backward (`--moe-lora-bwd-expert-tile`).
    pub moe_lora_bwd_expert_tile: usize,
    /// FlashQLA chunkwise GDN prefill in the CUDA backend (`--gdr-chunkwise-prefill`).
    pub gdr_chunkwise_prefill: bool,
    /// Native FP8 DeepGEMM for frozen-weight forward projections (`--fp8-native-gemm`).
    pub fp8_native_gemm: bool,
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
            checkpoint_reload_device: true,
            checkpoint_pinned_offload_bytes: 0,
            lora_linear_bwd_tile_rows: 1024,
            moe_lora_bwd_expert_tile: 16,
            gdr_chunkwise_prefill: true,
            fp8_native_gemm: false,
            la_backward_mono: false,
            decode_attn_legacy: false,
            cuda_mempool_retain: true,
            tape_precision: TapePrecision::Fp32,
        }
    }
}

static CHECKPOINT_OFFLOAD_MIN_BYTES: AtomicUsize = AtomicUsize::new(2 << 20);
static CHECKPOINT_RELOAD_DEVICE: AtomicBool = AtomicBool::new(true);
static CHECKPOINT_PINNED_OFFLOAD_BYTES: AtomicUsize = AtomicUsize::new(0);
static LORA_LINEAR_BWD_TILE_ROWS: AtomicUsize = AtomicUsize::new(1024);
static MOE_LORA_BWD_EXPERT_TILE: AtomicUsize = AtomicUsize::new(16);
static GDR_CHUNKWISE_PREFILL: AtomicBool = AtomicBool::new(true);
static FP8_NATIVE_GEMM: AtomicBool = AtomicBool::new(false);
static LA_BACKWARD_MONO: AtomicBool = AtomicBool::new(false);
static DECODE_ATTN_LEGACY: AtomicBool = AtomicBool::new(false);
static TAPE_PRECISION: AtomicU8 = AtomicU8::new(0);

pub fn apply_runtime_flags(f: &AutogradRuntimeFlags) {
    CHECKPOINT_OFFLOAD_MIN_BYTES.store(f.checkpoint_offload_min_bytes, Relaxed);
    CHECKPOINT_RELOAD_DEVICE.store(f.checkpoint_reload_device, Relaxed);
    CHECKPOINT_PINNED_OFFLOAD_BYTES.store(f.checkpoint_pinned_offload_bytes, Relaxed);
    LORA_LINEAR_BWD_TILE_ROWS.store(f.lora_linear_bwd_tile_rows.max(1), Relaxed);
    MOE_LORA_BWD_EXPERT_TILE.store(f.moe_lora_bwd_expert_tile.max(1), Relaxed);
    GDR_CHUNKWISE_PREFILL.store(f.gdr_chunkwise_prefill, Relaxed);
    FP8_NATIVE_GEMM.store(f.fp8_native_gemm, Relaxed);
    LA_BACKWARD_MONO.store(f.la_backward_mono, Relaxed);
    DECODE_ATTN_LEGACY.store(f.decode_attn_legacy, Relaxed);
    #[cfg(feature = "cuda")]
    cuda_kernels::tensor::set_mempool_retain(f.cuda_mempool_retain);
    TAPE_PRECISION.store(f.tape_precision as u8, Relaxed);
}

pub(crate) fn checkpoint_offload_min_bytes() -> usize {
    CHECKPOINT_OFFLOAD_MIN_BYTES.load(Relaxed)
}
pub(crate) fn checkpoint_reload_device() -> bool {
    CHECKPOINT_RELOAD_DEVICE.load(Relaxed)
}
#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn checkpoint_pinned_offload_bytes() -> usize {
    CHECKPOINT_PINNED_OFFLOAD_BYTES.load(Relaxed)
}
/// Test-only A/B lever for the reload arm (the CLI flag is the production path).
#[cfg(test)]
pub(crate) fn set_checkpoint_reload_device(on: bool) {
    CHECKPOINT_RELOAD_DEVICE.store(on, Relaxed);
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
#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn fp8_native_gemm() -> bool {
    FP8_NATIVE_GEMM.load(Relaxed)
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
/// `true` when retained activations + emitted grads store bf16 (CUDA-only path).
#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn tape_bf16() -> bool {
    matches!(tape_precision(), TapePrecision::Bf16)
}
