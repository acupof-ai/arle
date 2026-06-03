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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DeepseekPerformanceProfile {
    DebugFallback,
    SglangBestPractice,
}

impl DeepseekPerformanceProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DebugFallback => "debug-fallback",
            Self::SglangBestPractice => "sglang-best-practice",
        }
    }

    pub const fn requires_best_practice(self) -> bool {
        matches!(self, Self::SglangBestPractice)
    }
}

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

    /// Current debug/default DSv4 code supports global TP/EP only.
    ///
    /// The explicit SGLang profile may carry richer axes as contract input so
    /// request-owner groups and startup diagnostics can be validated before the
    /// model fails closed on missing communicators/row ownership. That profile
    /// must still reject before serving opens until the execution path consumes
    /// the richer layout.
    pub fn validate_current_axis_support(&self, worker_count: Option<usize>) -> Result<()> {
        self.validate_current_axis_support_for_profile(worker_count, self.performance_profile()?)
    }

    fn validate_current_axis_support_for_profile(
        &self,
        worker_count: Option<usize>,
        profile: DeepseekPerformanceProfile,
    ) -> Result<()> {
        let axis_tp_size = self.tp.world_size.max(self.ep.world_size);
        if !self
            .axes
            .is_global_tp_ep_only(axis_tp_size, self.ep.world_size)
            && !profile.requires_best_practice()
        {
            bail!(
                "DeepSeek V4 advanced multi-axis layout is parsed but not wired into debug execution yet: axes={} tp_world={} ep_world={}. Use ARLE_DSV4_PERFORMANCE_PROFILE=sglang only for fail-closed SGLang contract validation; see docs/plans/2026-06-01-dsv4-sglang-path-alignment.md.",
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

    pub fn performance_profile(&self) -> Result<DeepseekPerformanceProfile> {
        dsv4_performance_profile_from_env()
    }

    /// Fail closed for explicit SGLang-path claims at config-load time.
    ///
    /// This checks only static env/topology prerequisites. Runtime contracts
    /// that require loaded model state, scheduler knobs, or graph capability are
    /// validated later by the serving startup path.
    pub fn validate_sglang_path_claim(&self, worker_count: Option<usize>) -> Result<()> {
        let profile = self.performance_profile()?;
        if !profile.requires_best_practice() {
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
        if !dsv4_env_value_is_one_of(
            "ARLE_DSV4_EXPERT_BACKEND",
            &["deepgemm", "required-deepgemm", "required_deepgemm"],
        ) {
            missing.push("ARLE_DSV4_EXPERT_BACKEND=deepgemm (required, not auto)".to_string());
        }
        if !dsv4_env_flag_enabled("ARLE_DSV4_SHARED_KV_POOL")? {
            missing.push("ARLE_DSV4_SHARED_KV_POOL=1 (persistent paged FP8 KV pool)".to_string());
        }
        if !missing.is_empty() {
            bail!(
                "DeepSeek V4 profile `{}` requested, but the DSv4 SGLang best-practice static contract is incomplete:\n - {}\nSet ARLE_DSV4_PERFORMANCE_PROFILE=debug-fallback to run the current replicated-token debug lane.",
                profile.as_str(),
                missing.join("\n - ")
            );
        }
        Ok(())
    }
}

pub fn dsv4_performance_profile_from_env() -> Result<DeepseekPerformanceProfile> {
    if let Some(raw) = std::env::var("ARLE_DSV4_PERFORMANCE_PROFILE").ok() {
        return match raw.trim().to_ascii_lowercase().as_str() {
            "" | "debug" | "debug-fallback" | "fallback" | "replicated-token"
            | "replicated_token" => Ok(DeepseekPerformanceProfile::DebugFallback),
            "sglang"
            | "sglang-best-practice"
            | "best-practice"
            | "best_practice"
            | "high-perf"
            | "high_perf"
            | "performance" => Ok(DeepseekPerformanceProfile::SglangBestPractice),
            other => bail!(
                "invalid ARLE_DSV4_PERFORMANCE_PROFILE value `{other}`: expected debug-fallback or sglang"
            ),
        };
    }
    if dsv4_env_flag_enabled("ARLE_DSV4_HIGH_PERF")?
        || dsv4_env_flag_enabled("ARLE_DSV4_SGLANG_PATH")?
    {
        return Ok(DeepseekPerformanceProfile::SglangBestPractice);
    }
    Ok(DeepseekPerformanceProfile::DebugFallback)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::expert_state::ExpertGroup;
    use crate::tensor_parallel::{MultiAxisConfig, RankCoord, TpConfig};

    fn runtime_with_axes(axes: MultiAxisConfig) -> DeepseekRuntimeConfig {
        let spec = DeepSeekV4Config::from_json_str(
            r#"{
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "torch_dtype": "bfloat16",
                "vocab_size": 128,
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "hidden_act": "silu",
                "swiglu_limit": 10.0,
                "q_lora_rank": 1,
                "o_lora_rank": 1,
                "o_groups": 1,
                "qk_rope_head_dim": 1,
                "n_routed_experts": 8,
                "n_shared_experts": 0,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 4,
                "routed_scaling_factor": 1.0,
                "norm_topk_prob": false,
                "scoring_func": "softmax",
                "topk_method": "noaux_tc",
                "index_n_heads": 1,
                "index_head_dim": 1,
                "index_topk": 1,
                "num_hash_layers": 0,
                "sliding_window": 16,
                "compress_ratios": [0],
                "compress_rope_theta": 160000.0,
                "hc_mult": 1,
                "hc_sinkhorn_iters": 1,
                "hc_eps": 1.0e-6,
                "num_nextn_predict_layers": 0,
                "max_position_embeddings": 128,
                "rope_theta": 10000.0,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 1.0,
                    "original_max_position_embeddings": 128,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0
                },
                "rms_norm_eps": 1.0e-6,
                "initializer_range": 0.02,
                "tie_word_embeddings": false,
                "attention_bias": false,
                "attention_dropout": 0.0,
                "bos_token_id": 0,
                "eos_token_id": 1
            }"#,
        )
        .expect("tiny DSv4 config");
        DeepseekRuntimeConfig {
            spec,
            enable_cuda_graph: true,
            tp: TpConfig::new(8, 0).expect("tp"),
            ep: ExpertGroup::new(0, 4, 8).expect("ep"),
            axes,
            rank_coord: RankCoord::from_world_rank(axes, 0).expect("rank coord"),
        }
    }

    #[test]
    fn advanced_axes_remain_blocked_for_debug_profile() {
        let axes = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 4,
            attn_dp_size: 4,
            attn_cp_size: 1,
            moe_dp_size: 2,
        };
        let runtime = runtime_with_axes(axes);
        let err = runtime
            .validate_current_axis_support_for_profile(
                Some(8),
                DeepseekPerformanceProfile::DebugFallback,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("advanced multi-axis layout"), "got: {err}");
    }

    #[test]
    fn advanced_axes_are_allowed_only_as_sglang_contract_input() {
        let axes = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 4,
            attn_dp_size: 4,
            attn_cp_size: 1,
            moe_dp_size: 2,
        };
        let runtime = runtime_with_axes(axes);
        runtime
            .validate_current_axis_support_for_profile(
                Some(8),
                DeepseekPerformanceProfile::SglangBestPractice,
            )
            .expect("SGLang profile may carry advanced axes to fail-closed startup contract");
    }
}
