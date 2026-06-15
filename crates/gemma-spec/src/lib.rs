use std::fs;
use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Gemma4ConfigError {
    #[error("invalid gemma4 config: {0}")]
    InvalidConfig(&'static str),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Gemma4ConfigError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Gemma4LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Gemma4RopeParameters {
    #[serde(default, alias = "type")]
    pub rope_type: String,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Gemma4RopeConfig {
    Single(Gemma4RopeParameters),
    PerLayer {
        sliding_attention: Option<Gemma4RopeParameters>,
        full_attention: Option<Gemma4RopeParameters>,
    },
}

impl<'de> Deserialize<'de> for Gemma4RopeConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let per_layer =
            value.get("sliding_attention").is_some() || value.get("full_attention").is_some();
        if per_layer {
            #[derive(Deserialize)]
            struct RawPerLayer {
                #[serde(default)]
                sliding_attention: Option<Gemma4RopeParameters>,
                #[serde(default)]
                full_attention: Option<Gemma4RopeParameters>,
            }
            let raw: RawPerLayer =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            Ok(Self::PerLayer {
                sliding_attention: raw.sliding_attention,
                full_attention: raw.full_attention,
            })
        } else {
            let raw: Gemma4RopeParameters =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            Ok(Self::Single(raw))
        }
    }
}

impl Gemma4RopeConfig {
    #[must_use]
    pub fn for_layer_type(&self, layer_type: Gemma4LayerType) -> Option<&Gemma4RopeParameters> {
        match self {
            Self::Single(params) => Some(params),
            Self::PerLayer {
                sliding_attention,
                full_attention,
            } => match layer_type {
                Gemma4LayerType::SlidingAttention => sliding_attention.as_ref(),
                Gemma4LayerType::FullAttention => full_attention.as_ref(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Gemma4TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_activation: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_parameters: Option<Gemma4RopeConfig>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    pub sliding_window: usize,
    pub layer_types: Vec<Gemma4LayerType>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    #[serde(default)]
    pub swa_num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub swa_head_dim: Option<usize>,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub use_second_mlp_block: bool,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default, alias = "num_experts_per_tok")]
    pub top_k_experts: Option<usize>,
    #[serde(default, alias = "expert_intermediate_size")]
    pub moe_intermediate_size: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct DiffusionGemmaSamplerConfig {
    #[serde(default)]
    pub entropy_bound: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct DiffusionGemmaGenerationConfig {
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub max_denoising_steps: Option<usize>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub confidence_threshold: Option<f32>,
    #[serde(default)]
    pub stability_threshold: Option<usize>,
    #[serde(default)]
    pub t_min: Option<f32>,
    #[serde(default)]
    pub t_max: Option<f32>,
    #[serde(default)]
    pub sampler_config: Option<DiffusionGemmaSamplerConfig>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct DiffusionGemmaConfig {
    pub canvas_length: usize,
    #[serde(default)]
    pub generation_config: DiffusionGemmaGenerationConfig,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    pub text_config: Gemma4TextConfig,
}

fn default_use_cache() -> bool {
    true
}

fn default_tie_word_embeddings() -> bool {
    true
}

impl Gemma4TextConfig {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json_str(&content)
    }

    pub fn from_json_str(content: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(content)?;
        Self::from_json_value(&value)
    }

    pub fn from_json_value(value: &serde_json::Value) -> Result<Self> {
        let text = value.get("text_config").unwrap_or(value);
        let config: Self = serde_json::from_value(text.clone())?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_hidden_layers == 0 || self.layer_types.len() != self.num_hidden_layers {
            return Err(Gemma4ConfigError::InvalidConfig(
                "layer_types length must equal num_hidden_layers",
            ));
        }
        if self.layer_types.last() != Some(&Gemma4LayerType::FullAttention) {
            return Err(Gemma4ConfigError::InvalidConfig(
                "Gemma4 final layer must be full/global attention",
            ));
        }
        if self.head_dim == 0 || self.num_attention_heads == 0 || self.num_key_value_heads == 0 {
            return Err(Gemma4ConfigError::InvalidConfig(
                "attention head counts and head_dim must be non-zero",
            ));
        }
        Ok(())
    }

    pub fn local_kv_heads(&self) -> usize {
        self.swa_num_key_value_heads
            .unwrap_or(self.num_key_value_heads)
    }

    pub fn local_attention_head_dim(&self) -> usize {
        self.swa_head_dim.unwrap_or(self.head_dim)
    }

    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .is_some_and(|kind| *kind == Gemma4LayerType::FullAttention)
    }

    pub fn global_kv_heads(&self) -> usize {
        self.num_global_key_value_heads
            .unwrap_or(self.num_key_value_heads)
    }

    pub fn global_attention_head_dim(&self) -> usize {
        self.global_head_dim.unwrap_or(self.head_dim)
    }

    pub fn attention_head_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_global_layer(layer_idx) {
            self.global_attention_head_dim()
        } else {
            self.local_attention_head_dim()
        }
    }

    pub fn kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_global_layer(layer_idx) {
            self.global_kv_heads()
        } else {
            self.local_kv_heads()
        }
    }

    pub fn has_per_layer_embeddings(&self) -> bool {
        self.vocab_size_per_layer_input.unwrap_or(0) > 0
            && self.hidden_size_per_layer_input.unwrap_or(0) > 0
    }

    pub fn first_kv_shared_layer_idx(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers)
    }

    pub fn is_kv_shared_layer(&self, layer_idx: usize) -> bool {
        self.num_kv_shared_layers > 0 && layer_idx >= self.first_kv_shared_layer_idx()
    }

    pub fn kv_shared_source_layer(&self, layer_idx: usize) -> Option<usize> {
        if !self.is_kv_shared_layer(layer_idx) {
            return None;
        }
        let first_shared = self.first_kv_shared_layer_idx();
        let target = *self.layer_types.get(layer_idx)?;
        self.layer_types[..first_shared]
            .iter()
            .rposition(|kind| *kind == target)
    }

    pub fn layer_intermediate_size(&self, layer_idx: usize) -> usize {
        if self.use_double_wide_mlp && self.is_kv_shared_layer(layer_idx) {
            self.intermediate_size.saturating_mul(2)
        } else {
            self.intermediate_size
        }
    }

    pub fn uses_moe_block(&self) -> bool {
        self.enable_moe_block
            || (self.num_experts.unwrap_or(0) > 0
                && self.top_k_experts.unwrap_or(0) > 0
                && self.moe_intermediate_size.unwrap_or(0) > 0)
    }

    pub fn moe_top_k(&self) -> Option<usize> {
        self.top_k_experts
            .filter(|top_k| self.uses_moe_block() && *top_k > 0)
    }
}

impl DiffusionGemmaConfig {
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
        if self.canvas_length == 0 {
            return Err(Gemma4ConfigError::InvalidConfig(
                "canvas_length must be greater than zero",
            ));
        }
        if self.generation_config.max_denoising_steps == Some(0) {
            return Err(Gemma4ConfigError::InvalidConfig(
                "max_denoising_steps must be greater than zero",
            ));
        }
        self.text_config.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_text_config_and_enforces_final_global() {
        let cfg = Gemma4TextConfig::from_json_str(
            r#"{
                "model_type": "gemma4",
                "text_config": {
                    "vocab_size": 262144,
                    "hidden_size": 2304,
                    "intermediate_size": 9216,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "max_position_embeddings": 131072,
                    "initializer_range": 0.02,
                    "rms_norm_eps": 1e-6,
                    "sliding_window": 512,
                    "layer_types": [
                        "sliding_attention",
                        "sliding_attention",
                        "sliding_attention",
                        "full_attention"
                    ],
                    "vocab_size_per_layer_input": 262144,
                    "hidden_size_per_layer_input": 256,
                    "num_global_key_value_heads": 1,
                    "global_head_dim": 512,
                    "attention_k_eq_v": true,
                    "num_kv_shared_layers": 3
                }
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.num_hidden_layers, 4);
        assert!(cfg.is_global_layer(3));
        assert!(cfg.has_per_layer_embeddings());
        assert_eq!(cfg.global_kv_heads(), 1);
        assert_eq!(cfg.global_attention_head_dim(), 512);
    }

    #[test]
    fn derives_gemma4_ple_and_kv_share_contract() {
        let cfg = Gemma4TextConfig::from_json_str(
            r#"{
                "vocab_size": 262144,
                "hidden_size": 1536,
                "intermediate_size": 4096,
                "num_hidden_layers": 6,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "hidden_activation": "gelu_pytorch_tanh",
                "max_position_embeddings": 32768,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "sliding_window": 1024,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention"
                ],
                "global_head_dim": 512,
                "hidden_size_per_layer_input": 256,
                "vocab_size_per_layer_input": 262144,
                "num_kv_shared_layers": 3,
                "use_double_wide_mlp": true
            }"#,
        )
        .unwrap();
        assert!(cfg.has_per_layer_embeddings());
        assert_eq!(cfg.first_kv_shared_layer_idx(), 3);
        assert_eq!(cfg.kv_shared_source_layer(3), Some(1));
        assert_eq!(cfg.kv_shared_source_layer(4), Some(1));
        assert_eq!(cfg.kv_shared_source_layer(5), Some(2));
        assert_eq!(cfg.attention_head_dim_for_layer(0), 256);
        assert_eq!(cfg.attention_head_dim_for_layer(5), 512);
        assert_eq!(cfg.layer_intermediate_size(2), 4096);
        assert_eq!(cfg.layer_intermediate_size(3), 8192);
    }

    #[test]
    fn parses_diffusion_gemma_text_config_shape() {
        let cfg = DiffusionGemmaConfig::from_json_str(
            r#"{
                "architectures": ["DiffusionGemmaForBlockDiffusion"],
                "model_type": "diffusion_gemma",
                "canvas_length": 256,
                "eos_token_id": [1, 106, 50],
                "generation_config": {
                    "max_new_tokens": 256,
                    "max_denoising_steps": 48,
                    "confidence_threshold": 0.005,
                    "stability_threshold": 1,
                    "t_min": 0.4,
                    "t_max": 0.8,
                    "sampler_config": {
                        "entropy_bound": 0.1
                    }
                },
                "text_config": {
                    "vocab_size": 262144,
                    "hidden_size": 2816,
                    "intermediate_size": 9216,
                    "num_hidden_layers": 6,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "max_position_embeddings": 131072,
                    "initializer_range": 0.02,
                    "rms_norm_eps": 1e-6,
                    "sliding_window": 512,
                    "layer_types": [
                        "sliding_attention",
                        "sliding_attention",
                        "sliding_attention",
                        "sliding_attention",
                        "sliding_attention",
                        "full_attention"
                    ],
                    "rope_parameters": {
                        "sliding_attention": {
                            "rope_theta": 10000.0
                        },
                        "full_attention": {
                            "rope_theta": 1000000.0,
                            "partial_rotary_factor": 0.25
                        }
                    },
                    "vocab_size_per_layer_input": 262144,
                    "hidden_size_per_layer_input": 512,
                    "num_global_key_value_heads": 1,
                    "global_head_dim": 512,
                    "attention_k_eq_v": true,
                    "num_kv_shared_layers": 3,
                    "num_experts": 128,
                    "top_k_experts": 8,
                    "moe_intermediate_size": 512
                }
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.canvas_length, 256);
        assert_eq!(cfg.generation_config.max_denoising_steps, Some(48));
        assert_eq!(cfg.text_config.hidden_size, 2816);
        assert_eq!(cfg.text_config.global_attention_head_dim(), 512);
        assert!(cfg.text_config.uses_moe_block());
        assert_eq!(cfg.text_config.moe_top_k(), Some(8));
        let rope = cfg.text_config.rope_parameters.as_ref().unwrap();
        let full = rope
            .for_layer_type(Gemma4LayerType::FullAttention)
            .expect("full attention rope");
        assert_eq!(full.rope_theta, Some(1_000_000.0));
        assert_eq!(full.partial_rotary_factor, Some(0.25));
        let sliding = rope
            .for_layer_type(Gemma4LayerType::SlidingAttention)
            .expect("sliding attention rope");
        assert_eq!(sliding.rope_theta, Some(10_000.0));
    }

    #[test]
    fn rejects_non_global_final_layer() {
        let err = Gemma4TextConfig::from_json_str(
            r#"{
                "vocab_size": 10,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "hidden_activation": "gelu_pytorch_tanh",
                "max_position_embeddings": 128,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "sliding_window": 512,
                "layer_types": ["sliding_attention"]
            }"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("final layer"));
    }
}
