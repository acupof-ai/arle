//! Runtime configuration for the DeepSeek V4 model scaffold.
//!
//! Wraps the canonical [`deepseek_spec::DeepSeekV4Config`] with the infer-side
//! serving knobs. The runtime target for `infer/models/dsv4-mini-1B-init/` is
//! the V4 HF checkpoint shape; older DeepSeek V3/nano configs are intentionally
//! unsupported.

use std::ops::Deref;
use std::path::Path;

use anyhow::{Context, Result, bail};
use deepseek_spec::DeepSeekV4Config;

use crate::distributed::expert_state::ExpertGroup;
use crate::tensor_parallel::{MultiAxisConfig, RankCoord, TpConfig};

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
    /// SGLang-style multi-axis rank layout requested for this worker. Today
    /// DSv4 still executes global TP/EP only; this field is the explicit
    /// topology contract used by startup guards and trace diagnostics.
    pub axes: MultiAxisConfig,
    /// This worker's coordinate inside `axes`.
    pub rank_coord: RankCoord,
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
            axes: MultiAxisConfig::single(),
            rank_coord: RankCoord::from_world_rank(MultiAxisConfig::single(), 0)
                .expect("single-rank coordinate is valid"),
        }
    }

    /// Parse `<model_dir>/config.json` as the DeepSeek V4 runtime target.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let config_path = model_dir.as_ref().join("config.json");
        let spec = DeepSeekV4Config::from_json_file(&config_path)
            .with_context(|| format!("loading DeepSeek V4 config {}", config_path.display()))?;
        let mut runtime = Self::from_spec(spec);
        runtime.tp = TpConfig::from_env().context("loading DeepSeek V4 tensor-parallel env")?;
        runtime.ep = ExpertGroup::from_env(runtime.spec.n_routed_experts)
            .context("loading DeepSeek V4 expert-parallel env")?;
        runtime.axes = MultiAxisConfig::current_route_from_env_with_defaults(
            runtime.tp.world_size,
            runtime.ep.world_size,
        )
        .context("loading DeepSeek V4 multi-axis env")?;
        let coord_rank = if runtime.tp.world_size > 1 {
            runtime.tp.rank
        } else {
            runtime.ep.rank
        };
        runtime.rank_coord = RankCoord::from_world_rank(runtime.axes, coord_rank)
            .context("loading DeepSeek V4 rank coordinate")?;
        runtime
            .validate_current_axis_support(None)
            .context("validating DeepSeek V4 multi-axis support")?;
        runtime
            .validate_sglang_path_claim(None)
            .context("validating DeepSeek V4 SGLang path claim")?;
        Ok(runtime)
    }

    /// Current executable DSv4 code supports global TP/EP only. Parsing a
    /// richer SGLang-style layout is useful for diagnostics, but accepting it
    /// silently would make benchmark output non-comparable.
    pub fn validate_current_axis_support(&self, worker_count: Option<usize>) -> Result<()> {
        let axis_tp_size = self.tp.world_size.max(self.ep.world_size);
        if !self
            .axes
            .is_global_tp_ep_only(axis_tp_size, self.ep.world_size)
        {
            bail!(
                "DeepSeek V4 advanced multi-axis layout is parsed but not wired into execution yet: axes={} tp_world={} ep_world={}. Current executable path supports only global TP/EP; see docs/plans/2026-06-01-dsv4-sglang-path-alignment.md.",
                self.axes.summary(),
                self.tp.world_size,
                self.ep.world_size,
            );
        }
        if let Some(worker_count) = worker_count
            && self.axes.world_size() != worker_count
        {
            bail!(
                "DeepSeek V4 axes.world_size ({}) must match CUDA worker count ({worker_count}); axes={}",
                self.axes.world_size(),
                self.axes.summary(),
            );
        }
        Ok(())
    }

    /// Fail closed for explicit SGLang-path claims. This is intentionally
    /// stricter than the default path because ARLE does not yet have
    /// token-owned DP/EP request shards or batched FlashMLA attention wired.
    pub fn validate_sglang_path_claim(&self, worker_count: Option<usize>) -> Result<()> {
        if !dsv4_env_flag_enabled("ARLE_DSV4_SGLANG_PATH")? {
            return Ok(());
        }
        let mut missing = Vec::new();
        if !dsv4_env_flag_enabled("ARLE_MULTIPROC_SERVE")? {
            missing.push("ARLE_MULTIPROC_SERVE=1 (one CUDA process per rank)".to_string());
        }
        if let Some(worker_count) = worker_count
            && self.axes.world_size() != worker_count
        {
            missing.push(format!(
                "multi-axis world_size must match CUDA workers: axes.world={} workers={worker_count}",
                self.axes.world_size()
            ));
        }
        if !dsv4_env_value_is_one_of("ARLE_DSV4_MOE_BACKEND", &["native-deepep", "native_deepep"]) {
            missing.push("ARLE_DSV4_MOE_BACKEND=native-deepep".to_string());
        }
        if !dsv4_env_flag_enabled("ARLE_DSV4_SHARED_KV_POOL")? {
            missing.push("ARLE_DSV4_SHARED_KV_POOL=1 (persistent paged FP8 KV pool)".to_string());
        }
        missing.push(
            "distributed request fanout still submits the full logical request to every rank; token-owned DP/EP request shards are not implemented".to_string(),
        );
        missing.push(
            "DeepSeek decode attention still loops per row in forward_decode_batch; batched FlashMLA with SGLang sparse/recent indices is not wired".to_string(),
        );
        missing.push(
            "LayerCommunicator only attaches global TP/EP groups; attention-DP/CP and MoE-DP subgroup communicators are not wired".to_string(),
        );

        bail!(
            "ARLE_DSV4_SGLANG_PATH=1 requested, but the DSv4 SGLang path contract is incomplete:\n - {}\nUnset ARLE_DSV4_SGLANG_PATH to run the current replicated-token default path.",
            missing.join("\n - ")
        );
    }
}

fn dsv4_env_flag_enabled(key: &str) -> Result<bool> {
    let Some(raw) = std::env::var(key).ok() else {
        return Ok(false);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("invalid {key} value `{raw}`: expected 0/1, true/false, yes/no, or on/off"),
    }
}

fn dsv4_env_value_is_one_of(key: &str, expected: &[&str]) -> bool {
    std::env::var(key)
        .ok()
        .map(|raw| {
            let normalized = raw.trim().to_ascii_lowercase();
            expected.iter().any(|value| normalized == *value)
        })
        .unwrap_or(false)
}

impl Deref for DeepseekRuntimeConfig {
    type Target = DeepSeekV4Config;

    fn deref(&self) -> &Self::Target {
        &self.spec
    }
}
