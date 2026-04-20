use std::fs;
use std::ops::Deref;
use std::path::Path;

use anyhow::Result;
use qwen3_spec::Qwen3Config as Qwen3Spec;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct Config {
    pub spec: Qwen3Spec,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub stop_token_ids: Vec<u32>,
}

impl Deref for Config {
    type Target = Qwen3Spec;

    fn deref(&self) -> &Self::Target {
        &self.spec
    }
}

#[derive(Debug, serde::Deserialize)]
struct GenerationConfig {
    eos_token_id: EosTokenIds,
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum EosTokenIds {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Single(token_id) => vec![token_id],
            Self::Multiple(token_ids) => token_ids,
        }
    }
}

impl Config {
    pub fn from_parts(
        spec: Qwen3Spec,
        bos_token_id: u32,
        eos_token_id: u32,
        stop_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            spec,
            bos_token_id,
            eos_token_id,
            stop_token_ids,
        }
    }

    pub fn from_file(model_path: &str) -> Result<Self> {
        let config_path = Path::new(model_path).join("config.json");
        let mut value: Value = serde_json::from_str(&fs::read_to_string(&config_path)?)?;
        if value.get("max_position_embeddings").is_none() {
            if let Some(context_length) = value.get("context_length").and_then(Value::as_u64) {
                value["max_position_embeddings"] = json!(context_length as usize);
            }
        }

        let spec = Qwen3Spec::from_json_value(&value)?;
        let bos_token_id = read_u32(&value, "bos_token_id")?;
        let eos_token_id = read_u32(&value, "eos_token_id")?;
        let stop_token_ids = Self::load_stop_token_ids(model_path, eos_token_id)?;

        Ok(Self {
            spec,
            bos_token_id,
            eos_token_id,
            stop_token_ids,
        })
    }

    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    fn load_stop_token_ids(model_path: &str, fallback_eos_token_id: u32) -> Result<Vec<u32>> {
        let generation_config_path = Path::new(model_path).join("generation_config.json");
        match fs::read_to_string(&generation_config_path) {
            Ok(content) => {
                let generation_config: GenerationConfig = serde_json::from_str(&content)?;
                let mut stop_token_ids = generation_config.eos_token_id.into_vec();
                stop_token_ids.dedup();
                Ok(stop_token_ids)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Ok(vec![fallback_eos_token_id])
            }
            Err(err) => Err(err.into()),
        }
    }
}

fn read_u32(value: &Value, field: &'static str) -> Result<u32> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .map(|raw| raw as u32)
        .ok_or_else(|| anyhow::anyhow!("missing or invalid config field `{field}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
    const MODEL_8B_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-8B");

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_load_config() {
        let config = Config::from_file(MODEL_PATH).unwrap();

        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 151_936);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert_eq!(config.bos_token_id, 151_643);
        assert_eq!(config.eos_token_id, 151_645);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.stop_token_ids, vec![151_645, 151_643]);
        assert!(config.is_stop_token(151_645));
        assert!(config.is_stop_token(151_643));
        assert_eq!(config.lm_head_tensor_name(), "model.embed_tokens.weight");
    }

    #[test]
    fn test_config_gqa_ratio() {
        let config = Config::from_file(MODEL_PATH).unwrap();

        // GQA: Q heads / KV heads = 4, meaning 4 Q heads share 1 KV head
        let gqa_ratio = config.num_attention_heads / config.num_key_value_heads;
        assert_eq!(gqa_ratio, 4);

        assert_eq!(config.head_dim, 128);
    }

    #[test]
    #[ignore = "requires Qwen3-8B model"]
    #[allow(clippy::float_cmp)]
    fn test_load_8b_config() {
        let config = Config::from_file(MODEL_8B_PATH).unwrap();

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 12_288);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 151_936);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert_eq!(config.bos_token_id, 151_643);
        assert_eq!(config.eos_token_id, 151_645);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.stop_token_ids, vec![151_645, 151_643]);
        assert!(config.is_stop_token(151_645));
        assert!(config.is_stop_token(151_643));
        assert_eq!(config.lm_head_tensor_name(), "lm_head.weight");
    }
}
