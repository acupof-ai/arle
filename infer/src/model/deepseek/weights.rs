//! DeepSeek model weights.
//!
//! Mirrors the `Qwen3Model` shape: pool-shared immutable weights with per-
//! request mutable state living in [`super::state::DeepseekState`]. The
//! tensor name layout follows
//! [`deepseek_spec::DeepSeekConfig::layer_tensor_names`] and
//! [`deepseek_spec::DeepSeekConfig::shard_for_global_tensor`].

use anyhow::Result;

use super::config::DeepseekRuntimeConfig;
#[cfg(feature = "cuda")]
use super::mla::MlaAttention;
#[cfg(feature = "cuda")]
use super::mlp::DenseMlp;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec};
use deepseek_spec::DeepSeekConfig;

/// One DeepSeek transformer layer: input norm → MLA → post norm → dense MLP.
///
/// MoE wiring (the `DeepSeekMoeTensorNames` branch in
/// `DeepSeekConfig::layer_tensor_names`) is deferred until SKU-B; the nano
/// fixture stays on the dense MLP path on every layer.
#[cfg(feature = "cuda")]
pub(super) struct DeepseekLayer {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attention: MlaAttention,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: DenseMlp,
}

/// DeepSeek model — weights and config only. Mutable per-slot state lives in
/// [`super::state::DeepseekState`].
pub struct DeepseekModel {
    pub(super) config: DeepseekRuntimeConfig,
    #[cfg(feature = "cuda")]
    pub(super) ctx: DeviceContext,
    #[cfg(feature = "cuda")]
    pub(super) embed_tokens: DeviceMatrix,
    /// `None` when `tie_word_embeddings == true` (nano / SKU-A / SKU-B all
    /// tie); the sampler reads through `embed_tokens` in that case.
    #[cfg(feature = "cuda")]
    pub(super) lm_head: Option<DeviceMatrix>,
    #[cfg(feature = "cuda")]
    pub(super) norm: DeviceVec,
    #[cfg(feature = "cuda")]
    pub(super) layers: Vec<DeepseekLayer>,
}

impl DeepseekModel {
    /// Read-only view of the runtime config (architecture + serving knobs).
    pub fn config(&self) -> &DeepseekRuntimeConfig {
        &self.config
    }

    /// Read-only view of the underlying spec config.
    pub fn spec(&self) -> &DeepSeekConfig {
        &self.config.spec
    }
}

#[cfg(feature = "cuda")]
impl DeepseekModel {
    /// Allocate a model from a spec config without loading any weights.
    ///
    /// Stub: returns `todo!()` until safetensors loading and the MLA + dense
    /// MLP weight allocation paths land. The intent is to mirror
    /// `Qwen3Model::from_safetensors_with_runtime` once kernels exist.
    pub fn from_config(_config: DeepseekRuntimeConfig) -> Result<Self> {
        todo!("DeepSeek weight allocation — wires alongside MLA kernel")
    }

    /// Load a checkpoint by safetensors path, validating tensor names against
    /// `DeepSeekConfig::layer_tensor_names` / `mtp_tensor_names` /
    /// `shard_for_global_tensor`.
    pub fn from_safetensors(_path: &str, _config: DeepseekRuntimeConfig) -> Result<Self> {
        todo!("DeepSeek safetensors loader — wires alongside MLA kernel")
    }
}
