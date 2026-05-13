//! DeepSeek V4 hybrid attention scaffold.
//!
//! The file name is historical from the earlier V3 MLA scaffold. The serving
//! target is now the actual `DeepseekV4ForCausalLM` checkpoint, whose attention
//! shape uses Q-LoRA, a single KV head, O-LoRA grouping, optional compressor /
//! indexer blocks, sliding-window local attention, and HCA/CSA-style sparse
//! streams. Phase 0.5 keeps the typed weight container and explicit TODOs; CUDA
//! kernels land in Phase 2A.

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceMatrix, DeviceVec, HiddenStates};

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
    pub(super) compressor: Option<DeepseekV4Compressor>,
    pub(super) indexer: Option<DeepseekV4Indexer>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)] // methods called from forward.rs once V4 kernels land
impl DeepseekV4Attention {
    /// Run V4 prefill for a packed `[seq, hidden]` row block.
    pub(super) fn forward_prefill(
        &self,
        _hidden: &HiddenStates,
        _start_pos: usize,
    ) -> Result<HiddenStates> {
        todo!("DeepSeek V4 attention kernel — Phase 2A")
    }

    /// Run V4 decode for a single token using the V4 cache/state layout.
    pub(super) fn forward_decode(&self, _token_pos: usize) -> Result<()> {
        todo!("DeepSeek V4 attention kernel — Phase 2A")
    }
}
