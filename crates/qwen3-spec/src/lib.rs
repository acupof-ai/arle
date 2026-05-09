use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Qwen3ConfigError {
    #[error("invalid qwen3 config: {0}")]
    InvalidConfig(&'static str),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Qwen3ConfigError>;

/// How a tensor's weight should be split across a tensor-parallel group.
/// Mirrors SGLang `python/sglang/srt/layers/linear.py` parallel-linear classes.
///
/// `dim` follows the HF safetensors layout for nn.Linear: row 0 is the output
/// (out_features) axis and row 1 is the input (in_features) axis. So
/// `Column { dim: 0 }` matches SGLang's `ColumnParallelLinear` (split output)
/// and `Row { dim: 1 }` matches `RowParallelLinear` (split input).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard {
    /// Replicated on every rank (norms, biases that aren't per-head, etc).
    Replicated,
    /// Column-parallel: split along output dim. SGLang `linear.py:289`.
    Column { dim: usize },
    /// Row-parallel: split along input dim. SGLang `linear.py:1335`.
    Row { dim: usize },
    /// Merged column-parallel: SGLang `linear.py:485`. Used by `gate_up_proj`
    /// and other fused projections; per-projection sizes come from config at
    /// runtime (not encoded here, since they're model-dependent).
    MergedColumn { dim: usize },
    /// Fused QKV: SGLang `linear.py:889 QKVParallelLinear`. The KV-head
    /// replication rule (SGLang `models/qwen3.py:84-95`) is applied at
    /// runtime, not encoded here.
    QkvFused { dim: usize },
    /// Vocab-parallel: SGLang `vocab_parallel_embedding.py:161`.
    /// Used for `embed_tokens` and (untied) `lm_head`.
    VocabParallel { dim: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3LayerTensorNames {
    pub layer_prefix: String,
    pub attention_prefix: String,
    pub mlp_prefix: String,
    pub input_layernorm: String,
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub q_norm: String,
    pub k_norm: String,
    pub post_attention_layernorm: String,
    pub mlp_gate_proj: String,
    pub mlp_up_proj: String,
    pub mlp_down_proj: String,
}

impl Qwen3LayerTensorNames {
    /// Returns `Some(Shard)` for tensors this spec knows about; `None` for
    /// any name not part of a transformer layer (caller falls back to
    /// `Shard::Replicated`). Per-layer tensors only — global tensors live
    /// on `Qwen3Config::shard_for_global_tensor`.
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.q_proj || name == self.k_proj || name == self.v_proj {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.o_proj {
            return Some(Shard::Row { dim: 1 });
        }
        if name == self.mlp_gate_proj || name == self.mlp_up_proj {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.mlp_down_proj {
            return Some(Shard::Row { dim: 1 });
        }
        if name == self.input_layernorm
            || name == self.post_attention_layernorm
            || name == self.q_norm
            || name == self.k_norm
        {
            return Some(Shard::Replicated);
        }
        None
    }
}

/// Long-context RoPE scaling config (HF transformers / SGLang
/// `rope_scaling` schema). `None` ⇒ vanilla RoPE with `rope_theta` base.
/// Applied during `precompute_rope` to extend native context window.
///
/// Per `docs/plans/M_rope-yarn-scaling.md` Phase 1a:config 接인 only;
/// inv_freq compute integration (Phase 1b-2) is the codex-pickup work.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RopeScalingConfig {
    /// YARN scaling (Peng et al. 2023). Used by Qwen3.6 long-ctx,
    /// DeepSeek V3, Llama 3.1.
    Yarn {
        factor: f32,
        original_max_position_embeddings: usize,
        #[serde(default = "default_yarn_beta_fast")]
        beta_fast: f32,
        #[serde(default = "default_yarn_beta_slow")]
        beta_slow: f32,
        #[serde(default)]
        attention_factor: Option<f32>,
        #[serde(default = "default_yarn_mscale")]
        mscale: f32,
    },
    /// Linear position interpolation (Chen et al. 2023).
    Linear { factor: f32 },
    /// NTK-aware scaling (kaiokendev 2023).
    NtkAware { factor: f32 },
}

fn default_yarn_beta_fast() -> f32 {
    32.0
}
fn default_yarn_beta_slow() -> f32 {
    1.0
}
fn default_yarn_mscale() -> f32 {
    1.0
}

/// Compute the per-dimension RoPE inverse frequencies, optionally scaled
/// by a long-context `RopeScalingConfig` variant.
///
/// Returns a vector of `head_dim / 2` floats, one per RoPE pair. Vanilla
/// (scaling = None) reproduces the standard
/// `inv_freq[i] = 1.0 / base.powf(2*i / head_dim)` formula used by all
/// Qwen3-family models pre-2026-05.
///
/// Phase 1b of M_rope-yarn-scaling: pure-function inv_freq compute is
/// independent of CUDA / Metal precompute_rope path. Phase 2 (codex
/// pickup) wires this into `weight_loader.rs::precompute_rope`.
pub fn compute_scaled_inv_freq(
    head_dim: usize,
    base: f32,
    scaling: Option<&RopeScalingConfig>,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let vanilla: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(i as f32 * 2.0 / head_dim as f32))
        .collect();
    match scaling {
        None => vanilla,
        Some(RopeScalingConfig::Linear { factor }) => {
            // Position interpolation: divide freq by factor so that position p maps to
            // effective position p / factor in the original train range.
            vanilla.into_iter().map(|f| f / factor).collect()
        }
        Some(RopeScalingConfig::NtkAware { factor }) => {
            // NTK-aware: scale base by factor^(dim/(dim-2)). Recompute inv_freq with the
            // larger base so high frequencies extrapolate while low frequencies shift gently.
            let exponent = (head_dim as f32) / (head_dim as f32 - 2.0);
            let scaled_base = base * factor.powf(exponent);
            (0..half_dim)
                .map(|i| 1.0 / scaled_base.powf(i as f32 * 2.0 / head_dim as f32))
                .collect()
        }
        Some(RopeScalingConfig::Yarn {
            factor,
            original_max_position_embeddings,
            beta_fast,
            beta_slow,
            ..
        }) => {
            // Per Peng et al. 2023 §3.2: blend NTK extrapolation (high freq) and
            // linear interpolation (low freq) using a smooth ramp keyed off
            // wavelength vs (original_max_pos / beta_*) thresholds.
            let max_pos = *original_max_position_embeddings as f32;
            let low_freq_wavelen = max_pos / beta_fast;
            let high_freq_wavelen = max_pos / beta_slow;
            vanilla
                .into_iter()
                .map(|freq| {
                    let wavelen = std::f32::consts::TAU / freq;
                    if wavelen < high_freq_wavelen {
                        // High frequency: pure extrapolation (preserve original freq).
                        freq
                    } else if wavelen > low_freq_wavelen {
                        // Low frequency: pure interpolation (divide by factor).
                        freq / factor
                    } else {
                        // Mid-range: smooth blend of extrap and interp.
                        let smooth = (max_pos / wavelen - beta_slow) / (beta_fast - beta_slow);
                        smooth * freq + (1.0 - smooth) * (freq / factor)
                    }
                })
                .collect()
        }
    }
}

/// Compute the YARN attention-score scaling factor (applied to logits
/// before softmax to compensate for the broader effective key range when
/// extending context). Per Peng et al. 2023 §3.4: `1 + 0.1 * mscale * ln(factor)`.
/// Returns `1.0` for None / Linear / NtkAware (no attention scaling).
pub fn compute_attention_factor(scaling: Option<&RopeScalingConfig>) -> f32 {
    match scaling {
        Some(RopeScalingConfig::Yarn {
            factor,
            attention_factor,
            mscale,
            ..
        }) => attention_factor.unwrap_or_else(|| 1.0 + 0.1 * mscale * factor.ln()),
        _ => 1.0,
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(alias = "num_kv_heads")]
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
}

impl Qwen3Config {
    pub fn embed_tokens_tensor_name(&self) -> &'static str {
        "model.embed_tokens.weight"
    }

    pub fn norm_tensor_name(&self) -> &'static str {
        "model.norm.weight"
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json_str(&content)
    }

    pub fn from_json_str(content: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(content)?;
        Self::from_json_value(&value)
    }

    pub fn from_json_value(value: &serde_json::Value) -> Result<Self> {
        let config: Self = serde_json::from_value(value.clone())?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 || self.head_dim == 0 {
            return Err(Qwen3ConfigError::InvalidConfig(
                "attention heads and head_dim must be non-zero",
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Qwen3ConfigError::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(Qwen3ConfigError::InvalidConfig(
                "head_dim must be even for RoPE",
            ));
        }
        if self.max_position_embeddings == 0 {
            return Err(Qwen3ConfigError::InvalidConfig(
                "max_position_embeddings must be non-zero",
            ));
        }
        Ok(())
    }

    pub fn lm_head_tensor_name(&self) -> &'static str {
        if self.tie_word_embeddings {
            self.embed_tokens_tensor_name()
        } else {
            "lm_head.weight"
        }
    }

    /// Sharding for non-layer ("global") tensors. Returns `None` for any name
    /// not recognised; callers fall back to `Shard::Replicated`.
    pub fn shard_for_global_tensor(&self, name: &str) -> Option<Shard> {
        match name {
            "model.embed_tokens.weight" => Some(Shard::VocabParallel { dim: 0 }),
            "lm_head.weight" => Some(Shard::VocabParallel { dim: 0 }),
            "model.norm.weight" => Some(Shard::Replicated),
            _ => None,
        }
    }

    pub fn rope_cache_len_hint(&self) -> Option<usize> {
        Some(self.max_position_embeddings)
    }

    pub fn layer_tensor_names(&self, layer_idx: usize) -> Qwen3LayerTensorNames {
        let layer_prefix = format!("model.layers.{layer_idx}");
        let attention_prefix = format!("{layer_prefix}.self_attn");
        let mlp_prefix = format!("{layer_prefix}.mlp");

        Qwen3LayerTensorNames {
            layer_prefix: layer_prefix.clone(),
            attention_prefix: attention_prefix.clone(),
            mlp_prefix: mlp_prefix.clone(),
            input_layernorm: format!("{layer_prefix}.input_layernorm.weight"),
            q_proj: format!("{attention_prefix}.q_proj.weight"),
            k_proj: format!("{attention_prefix}.k_proj.weight"),
            v_proj: format!("{attention_prefix}.v_proj.weight"),
            o_proj: format!("{attention_prefix}.o_proj.weight"),
            q_norm: format!("{attention_prefix}.q_norm.weight"),
            k_norm: format!("{attention_prefix}.k_norm.weight"),
            post_attention_layernorm: format!("{layer_prefix}.post_attention_layernorm.weight"),
            mlp_gate_proj: format!("{mlp_prefix}.gate_proj.weight"),
            mlp_up_proj: format!("{mlp_prefix}.up_proj.weight"),
            mlp_down_proj: format!("{mlp_prefix}.down_proj.weight"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates_core_fields() {
        let cfg = Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 2048,
                "intermediate_size": 5632,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": true,
                "max_position_embeddings": 32768
            }"#,
        )
        .unwrap();

        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.embed_tokens_tensor_name(), "model.embed_tokens.weight");
        assert_eq!(cfg.norm_tensor_name(), "model.norm.weight");
        assert_eq!(cfg.lm_head_tensor_name(), "model.embed_tokens.weight");
        assert_eq!(cfg.rope_cache_len_hint(), Some(32768));
    }

    #[test]
    fn accepts_num_kv_heads_alias() {
        let cfg = Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": false,
                "max_position_embeddings": 32768
            }"#,
        )
        .unwrap();

        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.lm_head_tensor_name(), "lm_head.weight");
    }

    #[test]
    fn exposes_canonical_layer_tensor_names() {
        let cfg = Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": false,
                "max_position_embeddings": 32768
            }"#,
        )
        .unwrap();

        let names = cfg.layer_tensor_names(7);
        assert_eq!(names.layer_prefix, "model.layers.7");
        assert_eq!(names.attention_prefix, "model.layers.7.self_attn");
        assert_eq!(names.mlp_prefix, "model.layers.7.mlp");
        assert_eq!(
            names.input_layernorm,
            "model.layers.7.input_layernorm.weight"
        );
        assert_eq!(names.q_proj, "model.layers.7.self_attn.q_proj.weight");
        assert_eq!(names.k_proj, "model.layers.7.self_attn.k_proj.weight");
        assert_eq!(names.v_proj, "model.layers.7.self_attn.v_proj.weight");
        assert_eq!(names.o_proj, "model.layers.7.self_attn.o_proj.weight");
        assert_eq!(names.q_norm, "model.layers.7.self_attn.q_norm.weight");
        assert_eq!(names.k_norm, "model.layers.7.self_attn.k_norm.weight");
        assert_eq!(
            names.post_attention_layernorm,
            "model.layers.7.post_attention_layernorm.weight"
        );
        assert_eq!(names.mlp_gate_proj, "model.layers.7.mlp.gate_proj.weight");
        assert_eq!(names.mlp_up_proj, "model.layers.7.mlp.up_proj.weight");
        assert_eq!(names.mlp_down_proj, "model.layers.7.mlp.down_proj.weight");
    }

    fn shard_test_config() -> Qwen3Config {
        Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": false,
                "max_position_embeddings": 32768
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn every_layer_tensor_name_has_a_shard() {
        let cfg = shard_test_config();
        let names = cfg.layer_tensor_names(0);
        for name in [
            &names.input_layernorm,
            &names.q_proj,
            &names.k_proj,
            &names.v_proj,
            &names.o_proj,
            &names.q_norm,
            &names.k_norm,
            &names.post_attention_layernorm,
            &names.mlp_gate_proj,
            &names.mlp_up_proj,
            &names.mlp_down_proj,
        ] {
            assert!(names.shard_for(name).is_some(), "missing shard for {name}");
        }
    }

    #[test]
    fn qkv_proj_is_column_dim0_and_o_proj_is_row_dim1() {
        let cfg = shard_test_config();
        let names = cfg.layer_tensor_names(3);
        assert_eq!(
            names.shard_for(&names.q_proj),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&names.k_proj),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&names.v_proj),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(names.shard_for(&names.o_proj), Some(Shard::Row { dim: 1 }));
    }

    #[test]
    fn mlp_gate_and_up_proj_are_column_dim0() {
        let cfg = shard_test_config();
        let names = cfg.layer_tensor_names(0);
        assert_eq!(
            names.shard_for(&names.mlp_gate_proj),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&names.mlp_up_proj),
            Some(Shard::Column { dim: 0 })
        );
    }

    #[test]
    fn mlp_down_proj_is_row_dim1() {
        let cfg = shard_test_config();
        let names = cfg.layer_tensor_names(0);
        assert_eq!(
            names.shard_for(&names.mlp_down_proj),
            Some(Shard::Row { dim: 1 })
        );
    }

    #[test]
    fn embed_tokens_and_lm_head_are_vocab_parallel_dim0() {
        let cfg = shard_test_config();
        assert_eq!(
            cfg.shard_for_global_tensor("model.embed_tokens.weight"),
            Some(Shard::VocabParallel { dim: 0 })
        );
        assert_eq!(
            cfg.shard_for_global_tensor("lm_head.weight"),
            Some(Shard::VocabParallel { dim: 0 })
        );
    }

    #[test]
    fn norm_tensors_are_replicated() {
        let cfg = shard_test_config();
        let names = cfg.layer_tensor_names(0);
        assert_eq!(
            names.shard_for(&names.input_layernorm),
            Some(Shard::Replicated)
        );
        assert_eq!(
            names.shard_for(&names.post_attention_layernorm),
            Some(Shard::Replicated)
        );
        assert_eq!(names.shard_for(&names.q_norm), Some(Shard::Replicated));
        assert_eq!(names.shard_for(&names.k_norm), Some(Shard::Replicated));
        assert_eq!(
            cfg.shard_for_global_tensor("model.norm.weight"),
            Some(Shard::Replicated)
        );
    }

    #[test]
    fn unknown_tensor_returns_none() {
        let cfg = shard_test_config();
        let names = cfg.layer_tensor_names(0);
        assert_eq!(names.shard_for("model.layers.0.unknown.weight"), None);
        assert_eq!(cfg.shard_for_global_tensor("not.a.tensor"), None);
    }

    #[test]
    fn parses_yarn_rope_scaling() {
        let cfg = Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768
                },
                "tie_word_embeddings": false,
                "max_position_embeddings": 131072
            }"#,
        )
        .unwrap();

        match cfg.rope_scaling {
            Some(RopeScalingConfig::Yarn {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                attention_factor,
                mscale,
            }) => {
                assert_eq!(factor, 4.0);
                assert_eq!(original_max_position_embeddings, 32768);
                assert_eq!(beta_fast, 32.0);
                assert_eq!(beta_slow, 1.0);
                assert_eq!(attention_factor, None);
                assert_eq!(mscale, 1.0);
            }
            _ => panic!("expected Yarn rope_scaling"),
        }
    }

    #[test]
    fn parses_linear_and_ntk_rope_scaling() {
        let lin = Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 24,
                "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                "rope_scaling": { "type": "linear", "factor": 2.0 },
                "tie_word_embeddings": true, "max_position_embeddings": 65536
            }"#,
        )
        .unwrap();
        assert_eq!(
            lin.rope_scaling,
            Some(RopeScalingConfig::Linear { factor: 2.0 })
        );

        let ntk = Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 24,
                "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                "rope_scaling": { "type": "ntk_aware", "factor": 4.0 },
                "tie_word_embeddings": true, "max_position_embeddings": 131072
            }"#,
        )
        .unwrap();
        assert_eq!(
            ntk.rope_scaling,
            Some(RopeScalingConfig::NtkAware { factor: 4.0 })
        );
    }

    #[test]
    fn missing_rope_scaling_defaults_to_none() {
        let cfg = Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 24,
                "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                "tie_word_embeddings": true, "max_position_embeddings": 32768
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.rope_scaling, None);
    }

    #[test]
    fn vanilla_inv_freq_matches_legacy_formula() {
        // Phase 1b sanity: scaling=None must reproduce the existing
        // weight_loader.rs::precompute_rope inv_freq exactly.
        let head_dim = 128;
        let base = 1_000_000.0_f32;
        let computed = compute_scaled_inv_freq(head_dim, base, None);
        let legacy: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / base.powf(i as f32 * 2.0 / head_dim as f32))
            .collect();
        assert_eq!(computed.len(), legacy.len());
        for (c, l) in computed.iter().zip(legacy.iter()) {
            assert!((c - l).abs() < 1e-6, "computed {c} != legacy {l}");
        }
    }

    #[test]
    fn linear_scaling_divides_freq_by_factor() {
        let head_dim = 64;
        let base = 1_000_000.0_f32;
        let scaling = RopeScalingConfig::Linear { factor: 4.0 };
        let scaled = compute_scaled_inv_freq(head_dim, base, Some(&scaling));
        let vanilla = compute_scaled_inv_freq(head_dim, base, None);
        for (s, v) in scaled.iter().zip(vanilla.iter()) {
            assert!((s - v / 4.0).abs() < 1e-6, "linear: {s} != {v}/4");
        }
    }

    #[test]
    fn ntk_aware_scaling_increases_base() {
        // NTK-aware: scaled_base = base * factor^(dim/(dim-2)). New inv_freq
        // for i=0 stays 1.0 (theta^0), but later dimensions decay slower.
        let head_dim = 128;
        let base = 1_000_000.0_f32;
        let scaling = RopeScalingConfig::NtkAware { factor: 4.0 };
        let scaled = compute_scaled_inv_freq(head_dim, base, Some(&scaling));
        let vanilla = compute_scaled_inv_freq(head_dim, base, None);
        // i=0 always equals 1.0 regardless of base
        assert!((scaled[0] - 1.0).abs() < 1e-6);
        assert!((vanilla[0] - 1.0).abs() < 1e-6);
        // Later dimensions must be smaller than vanilla (scaled_base > base
        // ⇒ 1/scaled_base^x < 1/base^x for x > 0)
        for i in 1..scaled.len() {
            assert!(
                scaled[i] < vanilla[i],
                "ntk dim {i}: scaled {} should be < vanilla {}",
                scaled[i],
                vanilla[i]
            );
        }
    }

    #[test]
    fn yarn_scaling_blends_extrap_and_interp() {
        // YARN: high-freq dimensions extrapolate (≈ vanilla freq), low-freq
        // dimensions interpolate (≈ vanilla freq / factor), mid blends.
        let head_dim = 128;
        let base = 1_000_000.0_f32;
        let factor = 4.0_f32;
        let scaling = RopeScalingConfig::Yarn {
            factor,
            original_max_position_embeddings: 32_768,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
            mscale: 1.0,
        };
        let scaled = compute_scaled_inv_freq(head_dim, base, Some(&scaling));
        let vanilla = compute_scaled_inv_freq(head_dim, base, None);
        // First (high-freq) dim should be near vanilla (pure extrapolation)
        assert!(
            (scaled[0] - vanilla[0]).abs() / vanilla[0] < 0.01,
            "yarn high-freq dim 0: {} vs vanilla {}",
            scaled[0],
            vanilla[0]
        );
        // Last (low-freq) dim should be near vanilla/factor (pure interpolation)
        let last = scaled.len() - 1;
        let expected_interp = vanilla[last] / factor;
        assert!(
            (scaled[last] - expected_interp).abs() / expected_interp < 0.01,
            "yarn low-freq dim {last}: {} vs vanilla/factor {expected_interp}",
            scaled[last]
        );
    }

    #[test]
    fn yarn_factor_one_is_near_noop() {
        // Sanity: factor=1.0 means no interpolation; result should be ~vanilla.
        let head_dim = 64;
        let base = 1_000_000.0_f32;
        let scaling = RopeScalingConfig::Yarn {
            factor: 1.0,
            original_max_position_embeddings: 32_768,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
            mscale: 1.0,
        };
        let scaled = compute_scaled_inv_freq(head_dim, base, Some(&scaling));
        let vanilla = compute_scaled_inv_freq(head_dim, base, None);
        for (s, v) in scaled.iter().zip(vanilla.iter()) {
            assert!(
                (s - v).abs() / v.abs() < 1e-3,
                "yarn factor=1: scaled {s} != vanilla {v}"
            );
        }
    }

    #[test]
    fn attention_factor_for_yarn_uses_mscale_log_factor() {
        let scaling = RopeScalingConfig::Yarn {
            factor: 4.0,
            original_max_position_embeddings: 32_768,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
            mscale: 1.0,
        };
        let af = compute_attention_factor(Some(&scaling));
        // 1 + 0.1 * 1.0 * ln(4) ≈ 1 + 0.1386 = 1.1386
        let expected = 1.0 + 0.1 * (4.0_f32).ln();
        assert!((af - expected).abs() < 1e-5, "{af} != {expected}");
    }

    #[test]
    fn attention_factor_for_explicit_override_used_directly() {
        let scaling = RopeScalingConfig::Yarn {
            factor: 4.0,
            original_max_position_embeddings: 32_768,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: Some(0.95),
            mscale: 1.0,
        };
        assert_eq!(compute_attention_factor(Some(&scaling)), 0.95);
    }

    #[test]
    fn attention_factor_is_one_for_non_yarn() {
        assert_eq!(compute_attention_factor(None), 1.0);
        assert_eq!(
            compute_attention_factor(Some(&RopeScalingConfig::Linear { factor: 2.0 })),
            1.0
        );
        assert_eq!(
            compute_attention_factor(Some(&RopeScalingConfig::NtkAware { factor: 2.0 })),
            1.0
        );
    }

    #[test]
    fn end_to_end_yarn_parsed_config_drives_scaled_inv_freq() {
        // Phase 1c integration: prove that JSON config with rope_scaling
        // round-trips through Qwen3Config and drives compute_scaled_inv_freq
        // correctly (i.e., M_rope-yarn-scaling Phase 1+2 chain works end-to-end
        // without separate manual RopeScalingConfig construction).
        let cfg = Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 2560,
                "intermediate_size": 8960,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 2.0,
                    "original_max_position_embeddings": 40960
                },
                "tie_word_embeddings": false,
                "max_position_embeddings": 81920
            }"#,
        )
        .unwrap();

        // Verify field round-tripped
        let scaling = cfg.rope_scaling.as_ref().expect("rope_scaling parsed");
        assert!(matches!(
            scaling,
            RopeScalingConfig::Yarn { factor: 2.0, .. }
        ));

        // Drive compute_scaled_inv_freq through the parsed config
        let scaled =
            compute_scaled_inv_freq(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling.as_ref());
        let vanilla = compute_scaled_inv_freq(cfg.head_dim, cfg.rope_theta, None);
        assert_eq!(scaled.len(), cfg.head_dim / 2);

        // Sanity: high-freq dim 0 ≈ vanilla, low-freq last dim ≈ vanilla/2
        assert!(
            (scaled[0] - vanilla[0]).abs() / vanilla[0] < 0.01,
            "yarn high-freq: {} vs {}",
            scaled[0],
            vanilla[0]
        );
        let last = scaled.len() - 1;
        let expected_interp = vanilla[last] / 2.0;
        assert!(
            (scaled[last] - expected_interp).abs() / expected_interp < 0.01,
            "yarn low-freq: {} vs {}",
            scaled[last],
            expected_interp
        );

        // attention_factor for factor=2.0 = 1 + 0.1 * 1.0 * ln(2)
        let af = compute_attention_factor(cfg.rope_scaling.as_ref());
        let expected_af = 1.0 + 0.1 * (2.0_f32).ln();
        assert!(
            (af - expected_af).abs() < 1e-5,
            "attention_factor: {af} vs {expected_af}"
        );
    }

    #[test]
    fn end_to_end_no_rope_scaling_drives_vanilla() {
        // Sanity: Qwen3-4B native config (no rope_scaling) must produce
        // bit-equivalent vanilla inv_freq via the compute helper.
        let cfg = Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 2560, "intermediate_size": 8960, "num_hidden_layers": 36,
                "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                "tie_word_embeddings": false, "max_position_embeddings": 40960
            }"#,
        )
        .unwrap();
        assert!(cfg.rope_scaling.is_none());

        let scaled =
            compute_scaled_inv_freq(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling.as_ref());
        let legacy: Vec<f32> = (0..cfg.head_dim / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(i as f32 * 2.0 / cfg.head_dim as f32))
            .collect();
        assert_eq!(scaled.len(), legacy.len());
        for (s, l) in scaled.iter().zip(legacy.iter()) {
            assert!((s - l).abs() < 1e-6, "{s} != {l}");
        }
    }

    #[test]
    fn rejects_invalid_head_layout() {
        let err = Qwen3Config::from_json_str(
            r#"{
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_hidden_layers": 2,
                "num_attention_heads": 6,
                "num_key_value_heads": 4,
                "head_dim": 127,
                "vocab_size": 32000,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": true,
                "max_position_embeddings": 2048
            }"#,
        )
        .unwrap_err();

        assert!(matches!(err, Qwen3ConfigError::InvalidConfig(_)));
    }
}
