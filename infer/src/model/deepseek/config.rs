//! Runtime configuration for the DeepSeek V4 model scaffold.
//!
//! Wraps the canonical [`deepseek_spec::DeepSeekV4Config`] with the infer-side
//! serving knobs. The older V3-era [`deepseek_spec::DeepSeekConfig`] remains in
//! the spec crate for train fixtures, but the runtime target for
//! `infer/models/dsv4-mini-1B-init/` is the V4 HF checkpoint shape.

use std::ops::Deref;
use std::path::Path;

use anyhow::{Context, Result};
use deepseek_spec::DeepSeekV4Config;

use crate::tensor_parallel::TpConfig;

/// Composite runtime config: the spec-level architecture parameters plus the
/// infer-side serving knobs.
#[derive(Debug, Clone)]
pub struct DeepseekRuntimeConfig {
    pub spec: DeepSeekV4Config,
    /// Capture decode-path CUDA graphs once per `(slot_count, batch_size)` and
    /// replay thereafter. Default `true` matches `Qwen3Model`.
    pub enable_cuda_graph: bool,
    /// Tensor-parallel placement. Single-rank by default; multi-rank wiring
    /// follows the `LayerCommunicator` rollout (see `infer/src/model/AGENTS.md`).
    pub tp: TpConfig,
}

impl DeepseekRuntimeConfig {
    /// Build a runtime config with default serving knobs.
    pub fn from_spec(spec: DeepSeekV4Config) -> Self {
        Self {
            spec,
            enable_cuda_graph: true,
            tp: TpConfig::single(),
        }
    }

    /// Parse `<model_dir>/config.json` as the DeepSeek V4 runtime target.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let config_path = model_dir.as_ref().join("config.json");
        let spec = DeepSeekV4Config::from_json_file(&config_path)
            .with_context(|| format!("loading DeepSeek V4 config {}", config_path.display()))?;
        Ok(Self::from_spec(spec))
    }
}

impl Deref for DeepseekRuntimeConfig {
    type Target = DeepSeekV4Config;

    fn deref(&self) -> &Self::Target {
        &self.spec
    }
}
