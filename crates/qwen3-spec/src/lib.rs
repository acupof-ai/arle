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
