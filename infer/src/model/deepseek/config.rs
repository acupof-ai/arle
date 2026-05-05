//! Runtime configuration for the DeepSeek model scaffold.
//!
//! Wraps the canonical [`deepseek_spec::DeepSeekConfig`] with the
//! infer-runtime-specific knobs (CUDA graph capture toggle, future tensor-
//! parallel placement, etc.). Mirrors `qwen3::config::Config` /
//! `qwen3::weights::ModelRuntimeConfig`.

use std::ops::Deref;

use deepseek_spec::DeepSeekConfig;

use crate::tensor_parallel::TpConfig;

/// Composite runtime config: the spec-level architecture parameters plus the
/// infer-side serving knobs.
#[derive(Debug, Clone)]
pub struct DeepseekRuntimeConfig {
    pub spec: DeepSeekConfig,
    /// Capture decode-path CUDA graphs once per `(slot_count, batch_size)` and
    /// replay thereafter. Default `true` matches `Qwen3Model`.
    pub enable_cuda_graph: bool,
    /// Tensor-parallel placement. Single-rank by default; multi-rank wiring
    /// follows the `LayerCommunicator` rollout (see `infer/src/model/AGENTS.md`).
    pub tp: TpConfig,
}

impl DeepseekRuntimeConfig {
    /// Build a runtime config with default serving knobs.
    pub fn from_spec(spec: DeepSeekConfig) -> Self {
        Self {
            spec,
            enable_cuda_graph: true,
            tp: TpConfig::single(),
        }
    }
}

impl Deref for DeepseekRuntimeConfig {
    type Target = DeepSeekConfig;

    fn deref(&self) -> &Self::Target {
        &self.spec
    }
}
