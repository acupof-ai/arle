use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Qwen3ConfigError {
    #[error("missing or invalid config field `{field}`")]
    InvalidConfigField { field: &'static str },
    #[error("invalid qwen3 config: {0}")]
    InvalidConfig(&'static str),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Qwen3ConfigError>;

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
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
}

impl Qwen3Config {
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
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(Qwen3ConfigError::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(Qwen3ConfigError::InvalidConfig("head_dim must be even for RoPE"));
        }
        if self.max_position_embeddings.is_none_or(|value| value == 0) {
            return Err(Qwen3ConfigError::InvalidConfig(
                "max_position_embeddings must be non-zero",
            ));
        }
        Ok(())
    }

    pub fn lm_head_tensor_name(&self) -> &'static str {
        if self.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        }
    }

    pub fn rope_cache_len_hint(&self) -> Option<usize> {
        self.max_position_embeddings
    }

    pub fn resolved_max_position_embeddings(&self) -> Result<usize> {
        self.max_position_embeddings.ok_or(Qwen3ConfigError::InvalidConfig(
            "max_position_embeddings must be non-zero",
        ))
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
