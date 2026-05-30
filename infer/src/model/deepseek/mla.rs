//! DeepSeek V4 hybrid attention scaffold.
//!
//! The file name is historical from the earlier V3 MLA scaffold. The serving
//! target is now the actual `DeepseekV4ForCausalLM` checkpoint, whose attention
//! shape uses Q-LoRA, a single KV head, O-LoRA grouping, optional compressor /
//! indexer blocks, sliding-window local attention, and HCA/CSA-style sparse
//! streams. This file is now just the typed weight container for one V4
//! attention block (`DeepseekV4Attention` + its `Compressor`/`Indexer`
//! sub-blocks), populated by the safetensors loader. The forward logic —
//! FlashMLA prefill/decode, the incremental SW/compressed path, and the legacy
//! hybrid fallback — lives in `weights.rs` (`finish_attention_gpu` et al.), NOT
//! here. The earlier Phase-0.5 `forward_prefill`/`forward_decode` scaffold
//! methods were never wired and have been removed.

#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceMatrix, DeviceVec};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// Optional compressor sub-block for CSA/HCA layers.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // fields populated by the safetensors loader once kernels land
pub(super) struct DeepseekV4Compressor {
    pub(super) wkv: DeviceMatrix,
    pub(super) wgate: DeviceMatrix,
    pub(super) ape: DeviceMatrix,
    pub(super) norm: DeviceVec,
}

/// Optional sparse indexer sub-block used by compressed sparse-attention layers.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // fields populated by the safetensors loader once kernels land
pub(super) struct DeepseekV4Indexer {
    pub(super) wq_b: DeviceMatrix,
    pub(super) weights_proj: DeviceMatrix,
    pub(super) compressor: DeepseekV4Compressor,
}

/// Weights for one DeepSeek V4 attention block.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // fields populated by the safetensors loader once kernels land
pub(super) struct DeepseekV4Attention {
    pub(super) wq_a: DeviceMatrix,
    pub(super) q_norm: DeviceVec,
    pub(super) wq_b: DeviceMatrix,
    pub(super) wkv: DeviceMatrix,
    pub(super) kv_norm: DeviceVec,
    pub(super) wo_a: DeviceMatrix,
    pub(super) wo_b: DeviceMatrix,
    pub(super) attn_sink: DeviceVec,
    /// f32 mirror of `attn_sink` for FlashMLA SM90 sparse prefill, which
    /// requires `float[h_q]` (vendor/flashmla/csrc/params.h:153). Same
    /// length as `attn_sink` (full TP-replicated `[num_attention_heads]`);
    /// FlashMLA dispatch offsets by `tp.rank * local_heads` into this
    /// buffer. Populated once at model load via `arle_bf16_to_f32_cuda`.
    pub(super) attn_sink_f32: CudaSlice<f32>,
    pub(super) compressor: Option<DeepseekV4Compressor>,
    pub(super) indexer: Option<DeepseekV4Indexer>,
}
