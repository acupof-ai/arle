//! Runtime configuration for the DeepSeek V4 model scaffold.
//!
//! Wraps the canonical [`deepseek_spec::DeepSeekV4Config`] with the infer-side
//! serving knobs. The runtime target for `infer/models/dsv4-mini-1B-init/` is
//! the V4 HF checkpoint shape; older DeepSeek V3/nano configs are intentionally
//! unsupported.

use std::ops::Deref;
use std::path::Path;

use anyhow::{Context, Result};
use deepseek_spec::DeepSeekV4Config;

use crate::distributed::expert_state::ExpertGroup;
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
    /// Expert-parallel placement for routed MoE experts.
    pub ep: ExpertGroup,
}

impl DeepseekRuntimeConfig {
    /// Build a runtime config with default serving knobs.
    pub fn from_spec(spec: DeepSeekV4Config) -> Self {
        let ep = ExpertGroup::new(0, 1, spec.n_routed_experts)
            .expect("DeepSeekV4Config validation guarantees routed experts");
        Self {
            spec,
            enable_cuda_graph: true,
            tp: TpConfig::single(),
            ep,
        }
    }

    /// Parse `<model_dir>/config.json` as the DeepSeek V4 runtime target.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let config_path = model_dir.as_ref().join("config.json");
        let spec = DeepSeekV4Config::from_json_file(&config_path)
            .with_context(|| format!("loading DeepSeek V4 config {}", config_path.display()))?;
        let mut runtime = Self::from_spec(spec);
        runtime.ep = ExpertGroup::from_env(runtime.spec.n_routed_experts)
            .context("loading DeepSeek V4 expert-parallel env")?;
        Ok(runtime)
    }
}

impl Deref for DeepseekRuntimeConfig {
    type Target = DeepSeekV4Config;

    fn deref(&self) -> &Self::Target {
        &self.spec
    }
}
