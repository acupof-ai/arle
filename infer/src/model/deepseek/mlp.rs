//! Dense SwiGLU MLP for non-MoE DeepSeek layers.
//!
//! The nano fixture and SKU-A use this on every layer; SKU-B uses it on the
//! `first_k_dense_replace` prefix layers and the per-layer `shared_experts`
//! sub-block (see substrate plan §6.2). MoE expert wiring is intentionally
//! deferred until SKU-B kernels land.

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceMatrix, HiddenStates};

/// Standard SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // fields populated by the safetensors loader once MLA kernel lands
pub(super) struct DenseMlp {
    pub(super) gate_proj: DeviceMatrix,
    pub(super) up_proj: DeviceMatrix,
    pub(super) down_proj: DeviceMatrix,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)] // method called from forward.rs once MLA kernel lands
impl DenseMlp {
    /// Run the MLP for a packed `[tokens, hidden]` row block.
    ///
    /// Stub: actual kernel reuses `ops::silu_into` + grouped GEMM. Wired in
    /// when MLA forward lands; until then this returns `todo!()` so any
    /// caller hits a clearly-named site.
    pub(super) fn forward(&self, _hidden: &HiddenStates) -> Result<HiddenStates> {
        todo!("DeepSeek dense MLP — wires alongside MLA forward")
    }
}
