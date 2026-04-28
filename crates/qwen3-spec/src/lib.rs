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
