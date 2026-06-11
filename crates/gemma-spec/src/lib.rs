use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
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
    pub rope_parameters: Option<Gemma4RopeParameters>,
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
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
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

    pub fn has_per_layer_embeddings(&self) -> bool {
        self.vocab_size_per_layer_input.unwrap_or(0) > 0
            && self.hidden_size_per_layer_input.unwrap_or(0) > 0
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
