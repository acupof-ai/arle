use std::{path::Path, str::FromStr};

use thiserror::Error;

use crate::{
    qwen3::Qwen3Config,
    qwen35::{LayerType, Qwen35Config},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Auto,
    Qwen35,
    Qwen3,
}

impl FromStr for ModelFamily {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "qwen35" | "qwen3.5" => Ok(Self::Qwen35),
            "qwen3" => Ok(Self::Qwen3),
            _ => Err(format!("unknown model family: {value}")),
        }
    }
}

#[derive(Debug, Error)]
pub enum ModelFamilyError {
    #[error("{0}")]
    Custom(String),
}

pub fn resolve_model_family(
    config_path: &Path,
    requested: ModelFamily,
) -> Result<ModelFamily, ModelFamilyError> {
    match requested {
        ModelFamily::Auto => {
            if Qwen35Config::from_json_file(config_path).is_ok() {
                Ok(ModelFamily::Qwen35)
            } else if Qwen3Config::from_json_file(config_path).is_ok() {
                Ok(ModelFamily::Qwen3)
            } else {
                Err(ModelFamilyError::Custom(format!(
                    "unable to infer model family from {}; neither qwen3 nor qwen3.5 config parsers accepted it",
                    config_path.display()
                )))
            }
        }
        family => Ok(family),
    }
}

pub fn synthetic_qwen3_config(seq: usize) -> Qwen3Config {
    Qwen3Config {
        vocab_size: 256,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        max_position_embeddings: seq,
        rms_norm_eps: 1.0e-6,
        rope_theta: 1_000_000.0,
        tie_word_embeddings: false,
    }
}

pub fn synthetic_qwen35_dense_config(seq: usize) -> Qwen35Config {
    Qwen35Config {
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        vocab_size: 256,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![255],
        bos_token_id: Some(1),
        eos_token_id: 255,
        tie_word_embeddings: false,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        head_dim: 16,
        linear_num_key_heads: 4,
        linear_key_head_dim: 16,
        linear_num_value_heads: 4,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 1.0,
        rotary_dim: 16,
        rope_cache_len_hint: Some(seq),
        layer_types: vec![LayerType::FullAttention; 2],
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
    }
}
