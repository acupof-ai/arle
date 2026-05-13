//! DeepSeek V4 MoE FFN scaffold.
//!
//! The local V4 1B checkpoint uses routed experts plus a shared expert on each
//! layer. Phase 0.5 records the correct tensor shape; Phase 1 supplies the
//! shared CUDA MoE primitive, and Phase 2A wires this block into forward.

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceMatrix, DeviceVec, HiddenStates};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// One SwiGLU expert: `w2(silu(w1(x)) * w3(x))`.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4Expert {
    pub(super) w1: DeviceMatrix,
    pub(super) w2: DeviceMatrix,
    pub(super) w3: DeviceMatrix,
}

/// V4 routed MoE block plus optional shared expert.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4MoeBlock {
    pub(super) gate_weight: DeviceMatrix,
    pub(super) gate_bias: Option<DeviceVec>,
    /// Hash-router table for early layers. The exact integer storage type is
    /// finalized with the Phase 2A loader; Phase 0.5 only validates the tensor
    /// name and keeps the field explicit.
    pub(super) gate_tid2eid: Option<CudaSlice<i64>>,
    pub(super) experts: Vec<DeepseekV4Expert>,
    pub(super) shared_experts: Option<DeepseekV4Expert>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)] // method called from forward.rs once MoE kernels land
impl DeepseekV4MoeBlock {
    /// Run routed V4 MoE for a packed `[tokens, hidden]` row block.
    pub(super) fn forward(&self, _hidden: &HiddenStates) -> Result<HiddenStates> {
        todo!("DeepSeek V4 MoE primitive — Phase 1/2A")
    }
}
