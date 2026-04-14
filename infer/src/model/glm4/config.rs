use anyhow::Result;
use serde::Deserialize;
use std::fs;

/// GLM-4 config.json fields.
///
/// Key differences from Qwen3:
/// - `ffn_hidden_size` instead of `intermediate_size`
/// - `num_layers` instead of `num_hidden_layers`
/// - `multi_query_group_num` instead of `num_key_value_heads`
/// - `kv_channels` instead of `head_dim`
/// - `padded_vocab_size` instead of `vocab_size`
/// - `layernorm_epsilon` instead of `rms_norm_eps`
/// - `rope_ratio` for RoPE scaling (GLM-4 uses theta * rope_ratio)
/// - `seq_length` for max sequence length
#[derive(Debug, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub multi_query_group_num: usize,
    pub kv_channels: usize,
    pub padded_vocab_size: usize,
    #[serde(default = "default_layernorm_eps")]
    pub layernorm_epsilon: f32,
    #[serde(default = "default_rope_ratio")]
    pub rope_ratio: f32,
    #[serde(default = "default_seq_length")]
    pub seq_length: usize,
    #[serde(skip)]
    pub stop_token_ids: Vec<u32>,
}

fn default_layernorm_eps() -> f32 {
    1e-5
}

fn default_rope_ratio() -> f32 {
    1.0
}

fn default_seq_length() -> usize {
    8192
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    eos_token_id: EosTokenIds,
}

#[derive(Debug, Deserialize)]
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
    pub fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let mut config: Config = serde_json::from_str(&content)?;
        config.stop_token_ids = Self::load_stop_token_ids(model_path)?;
        Ok(config)
    }

    /// RoPE base theta, adjusted by rope_ratio.
    /// GLM-4 uses base theta 10000 scaled by rope_ratio.
    pub fn rope_theta(&self) -> f32 {
        10000.0 * self.rope_ratio
    }

    pub fn rope_cache_len_hint(&self) -> Option<usize> {
        Some(self.seq_length)
    }

    /// Convenience accessors that map GLM-4 field names to the common naming
    /// convention used by the rest of the codebase.
    pub fn num_hidden_layers(&self) -> usize {
        self.num_layers
    }

    pub fn num_key_value_heads(&self) -> usize {
        self.multi_query_group_num
    }

    pub fn head_dim(&self) -> usize {
        self.kv_channels
    }

    pub fn vocab_size(&self) -> usize {
        self.padded_vocab_size
    }

    pub fn intermediate_size(&self) -> usize {
        self.ffn_hidden_size
    }

    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    fn load_stop_token_ids(model_path: &str) -> Result<Vec<u32>> {
        let generation_config_path = format!("{}/generation_config.json", model_path);
        match fs::read_to_string(&generation_config_path) {
            Ok(content) => {
                let generation_config: GenerationConfig = serde_json::from_str(&content)?;
                let mut stop_token_ids = generation_config.eos_token_id.into_vec();
                stop_token_ids.dedup();
                Ok(stop_token_ids)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                // GLM-4 commonly uses token 151329 (<|endoftext|>) and
                // 151336 (<|user|>) / 151338 (<|assistant|>) as stop tokens.
                // Without generation_config.json, default to the standard EOS.
                Ok(vec![151329, 151336, 151338])
            }
            Err(err) => Err(err.into()),
        }
    }
}
