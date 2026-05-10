//! DeepSeek V4 model weights.
//!
//! The runtime target is the local `DeepseekV4ForCausalLM` checkpoint at
//! `infer/models/dsv4-mini-1B-init/`. V3-era `DeepSeekConfig::nano()` remains a
//! train-side fixture, but infer-side DeepSeek wiring uses
//! [`deepseek_spec::DeepSeekV4Config`] and its HF tensor-name contract.

use std::path::Path;

use anyhow::{Result, bail};

use super::config::DeepseekRuntimeConfig;
#[cfg(feature = "cuda")]
use super::mla::DeepseekV4Attention;
#[cfg(feature = "cuda")]
use super::mlp::DeepseekV4MoeBlock;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec};
use deepseek_spec::DeepSeekV4Config;

use crate::deepseek_v4_manifest::{
    DeepseekV4CheckpointManifest, validate_deepseek_v4_checkpoint_manifest,
};

/// Hyper-connection tensors used by the V4 layer/head mixers.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4HyperConnection {
    pub(super) base: DeviceVec,
    pub(super) mix_fn: DeviceVec,
    pub(super) scale: DeviceVec,
}

/// One DeepSeek V4 transformer layer.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // fields populated by the safetensors loader once kernels land
pub(super) struct DeepseekLayer {
    pub(super) attn_norm: DeviceVec,
    pub(super) hc_attn: DeepseekV4HyperConnection,
    pub(super) attention: DeepseekV4Attention,
    pub(super) ffn_norm: DeviceVec,
    pub(super) hc_ffn: DeepseekV4HyperConnection,
    pub(super) ffn: DeepseekV4MoeBlock,
}

/// DeepSeek V4 model: immutable weights plus runtime config. Mutable per-slot
/// state lives in [`super::state::DeepseekState`].
#[allow(dead_code)] // fields populated by the safetensors loader once kernels land
pub struct DeepseekModel {
    pub(super) config: DeepseekRuntimeConfig,
    #[cfg(feature = "cuda")]
    pub(super) ctx: DeviceContext,
    #[cfg(feature = "cuda")]
    pub(super) embed_tokens: DeviceMatrix,
    #[cfg(feature = "cuda")]
    pub(super) lm_head: DeviceMatrix,
    #[cfg(feature = "cuda")]
    pub(super) norm: DeviceVec,
    #[cfg(feature = "cuda")]
    pub(super) head_hc: DeepseekV4HyperConnection,
    #[cfg(feature = "cuda")]
    pub(super) layers: Vec<DeepseekLayer>,
}

impl DeepseekModel {
    /// Read-only view of the runtime config.
    pub fn config(&self) -> &DeepseekRuntimeConfig {
        &self.config
    }

    /// Read-only view of the underlying DeepSeek V4 spec config.
    pub fn spec(&self) -> &DeepSeekV4Config {
        &self.config.spec
    }

    /// Every layer in the local V4 1B checkpoint has a routed MoE FFN plus
    /// shared expert. The old dense/nano runtime path is no longer the serving
    /// target.
    pub fn is_dense_layer(&self, _idx: usize) -> bool {
        false
    }

    /// Parse the safetensors manifest and verify every tensor required by the
    /// DeepSeek V4 spec is present. This is a cold-path truth gate and performs
    /// no GPU allocation.
    pub fn validate_checkpoint_manifest(
        model_path: impl AsRef<Path>,
        config: &DeepSeekV4Config,
    ) -> Result<DeepseekV4CheckpointManifest> {
        validate_deepseek_v4_checkpoint_manifest(model_path, config)
    }
}

#[cfg(feature = "cuda")]
impl DeepseekModel {
    /// Allocate a model from a spec config without loading weights.
    ///
    /// Phase 0.5 intentionally stops before GPU allocation; return an error
    /// instead of panicking so loader tests can distinguish "parsed V4 config"
    /// from "kernels not implemented yet".
    pub fn from_config(_config: DeepseekRuntimeConfig) -> Result<Self> {
        bail!("DeepSeek V4 weight allocation is pending Phase 2A kernels")
    }

    /// Load a V4 checkpoint by safetensors path.
    ///
    /// Phase 0.5 validates config + tensor-name truth, then returns a typed
    /// not-implemented error before GPU allocation.
    pub fn from_safetensors(path: &str, config: DeepseekRuntimeConfig) -> Result<Self> {
        let _manifest = Self::validate_checkpoint_manifest(path, &config.spec)?;
        bail!("DeepSeek V4 CUDA forward kernels are pending Phase 2A")
    }
}
