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

/// Default RoPE base for the Qwen3 family when neither a top-level
/// `rope_theta` nor a nested `rope_parameters.rope_theta` is present.
/// Every shipped Qwen3 config uses 1e6.
pub const DEFAULT_ROPE_THETA: f32 = 1_000_000.0;

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
    /// Model-default stop tokens from `config.json`'s `eos_token_id`
    /// (scalar or list in HF exports), normalized to a flat list. Empty when
    /// the config omits it. Serving uses these as the fallback stop set for
    /// requests that supply none.
    pub eos_token_ids: Vec<u32>,
}

/// Nested RoPE block emitted by newer HF transformers exports, e.g.
/// `"rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}`.
/// Older exports keep `rope_theta` at the top level instead. We capture
/// the nested block here so deserialization succeeds under either layout.
#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default)]
    rope_theta: Option<f32>,
}

/// Wire-format mirror of [`Qwen3Config`] used only during deserialization.
/// `rope_theta` is optional and a sibling `rope_parameters` block is
/// captured so configs that nest the base under `rope_parameters` (newer
/// HF exports) parse without a "missing field rope_theta" error. The
/// normalize step in [`Qwen3Config::from_json_value`] resolves the final
/// `rope_theta`.
#[derive(Debug, Deserialize)]
struct Qwen3ConfigRaw {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(alias = "num_kv_heads")]
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    rope_scaling: Option<RopeScalingConfig>,
    tie_word_embeddings: bool,
    max_position_embeddings: usize,
    /// HF exports write `eos_token_id` as a scalar OR a list; tolerate both.
    #[serde(default)]
    eos_token_id: Option<EosTokenIds>,
}

/// `eos_token_id` wire forms: `1`, `[1, 2]`, or absent.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl EosTokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            EosTokenIds::One(id) => vec![id],
            EosTokenIds::Many(ids) => ids,
        }
    }
}

impl From<Qwen3ConfigRaw> for Qwen3Config {
    fn from(raw: Qwen3ConfigRaw) -> Self {
        // Resolution order, HF-consistent: an explicit top-level `rope_theta`
        // wins (preserves the existing Qwen3-0.6B path exactly), else fall
        // back to `rope_parameters.rope_theta`, else the family default 1e6.
        let rope_theta = raw
            .rope_theta
            .or_else(|| raw.rope_parameters.and_then(|p| p.rope_theta))
            .unwrap_or(DEFAULT_ROPE_THETA);
        Qwen3Config {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta,
            rope_scaling: raw.rope_scaling,
            tie_word_embeddings: raw.tie_word_embeddings,
            max_position_embeddings: raw.max_position_embeddings,
            eos_token_ids: raw
                .eos_token_id
                .map(EosTokenIds::into_vec)
                .unwrap_or_default(),
        }
    }
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
        let raw: Qwen3ConfigRaw = serde_json::from_value(value.clone())?;
        let config: Self = raw.into();
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
