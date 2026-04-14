use anyhow::Result;
use serde::Deserialize;
use std::fs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    partial_rotary_factor: f64,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    rope_parameters: RopeParameters,
    eos_token_id: u32,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawConfig {
    /// Multi-modal config with nested text_config (Qwen3.5-4B, Qwen3.5-MoE)
    Nested { text_config: TextConfig },
    /// Flat text-only config (Carnice-27b uses Qwen3_5ForCausalLM directly)
    Flat(TextConfig),
}

impl RawConfig {
    fn into_text(self) -> TextConfig {
        match self {
            Self::Nested { text_config } => text_config,
            Self::Flat(t) => t,
        }
    }
}

/// Qwen3.5 model configuration (text-only).
#[derive(Debug, Clone)]
pub(crate) struct Config35 {
    // Common
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) eos_token_id: u32,
    /// All stop token IDs (from generation_config.json or fallback to eos_token_id).
    pub(crate) stop_token_ids: Vec<u32>,

    // Full attention params
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,

    // Linear attention params
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_conv_kernel_dim: usize,

    // RoPE
    pub(crate) rope_theta: f32,
    pub(crate) rotary_dim: usize,

    // Layer layout
    pub(crate) layer_types: Vec<LayerType>,
}

impl Config35 {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let raw: RawConfig = serde_json::from_str(&content)?;
        let t = raw.into_text();

        let layer_types: Vec<LayerType> = t
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" => Ok(LayerType::FullAttention),
                "linear_attention" => Ok(LayerType::LinearAttention),
                other => Err(anyhow::anyhow!("Unknown layer type: {}", other)),
            })
            .collect::<Result<_>>()?;

        anyhow::ensure!(
            layer_types.len() == t.num_hidden_layers,
            "layer_types length {} != num_hidden_layers {}",
            layer_types.len(),
            t.num_hidden_layers
        );

        let rotary_dim = (t.head_dim as f64 * t.rope_parameters.partial_rotary_factor) as usize;
        let stop_token_ids = Self::load_stop_token_ids(model_path, t.eos_token_id)?;

        Ok(Self {
            hidden_size: t.hidden_size,
            intermediate_size: t.intermediate_size,
            num_hidden_layers: t.num_hidden_layers,
            vocab_size: t.vocab_size,
            rms_norm_eps: t.rms_norm_eps as f32,
            eos_token_id: t.eos_token_id,
            stop_token_ids,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            linear_num_key_heads: t.linear_num_key_heads,
            linear_key_head_dim: t.linear_key_head_dim,
            linear_num_value_heads: t.linear_num_value_heads,
            linear_value_head_dim: t.linear_value_head_dim,
            linear_conv_kernel_dim: t.linear_conv_kernel_dim,
            rope_theta: t.rope_parameters.rope_theta as f32,
            rotary_dim,
            layer_types,
        })
    }

    pub(crate) fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    fn load_stop_token_ids(model_path: &str, fallback_eos_token_id: u32) -> Result<Vec<u32>> {
        let generation_config_path = format!("{}/generation_config.json", model_path);
        match fs::read_to_string(&generation_config_path) {
            Ok(content) => {
                let v: serde_json::Value = serde_json::from_str(&content)?;
                let mut ids = Vec::new();
                if let Some(eos) = v.get("eos_token_id") {
                    match eos {
                        serde_json::Value::Number(n) => {
                            if let Some(id) = n.as_u64() {
                                ids.push(id as u32);
                            }
                        }
                        serde_json::Value::Array(arr) => {
                            for item in arr {
                                if let Some(id) = item.as_u64() {
                                    ids.push(id as u32);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if ids.is_empty() {
                    ids.push(fallback_eos_token_id);
                }
                ids.dedup();
                Ok(ids)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Ok(vec![fallback_eos_token_id])
            }
            Err(err) => Err(err.into()),
        }
    }

    /// Number of full attention layers in the model.
    pub(crate) fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == LayerType::FullAttention)
            .count()
    }

    /// Total Q dimension for full attention (includes gate).
    pub(crate) fn full_attn_q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim * 2
    }

    /// Q dimension for full attention (without gate).
    pub(crate) fn full_attn_q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// KV dimension for full attention.
    pub(crate) fn full_attn_kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// QKV projection output dimension for linear attention.
    pub(crate) fn linear_attn_qkv_dim(&self) -> usize {
        let q_dim = self.linear_num_key_heads * self.linear_key_head_dim;
        let k_dim = q_dim;
        let v_dim = self.linear_num_value_heads * self.linear_value_head_dim;
        q_dim + k_dim + v_dim
    }

    /// Z projection output dimension for linear attention.
    pub(crate) fn linear_attn_z_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_load_config35() {
        let config = Config35::from_file(MODEL_PATH).unwrap();

        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.intermediate_size, 9216);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.vocab_size, 248_320);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.eos_token_id, 248_044);

        // Full attention
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 256);

        // Linear attention
        assert_eq!(config.linear_num_key_heads, 16);
        assert_eq!(config.linear_key_head_dim, 128);
        assert_eq!(config.linear_num_value_heads, 32);
        assert_eq!(config.linear_value_head_dim, 128);
        assert_eq!(config.linear_conv_kernel_dim, 4);

        // RoPE
        assert_eq!(config.rope_theta, 1e7);
        assert_eq!(config.rotary_dim, 64);
    }

    #[test]
    fn test_layer_types() {
        let config = Config35::from_file(MODEL_PATH).unwrap();

        assert_eq!(config.layer_types.len(), 32);
        assert_eq!(config.num_full_attention_layers(), 8);

        // Full attention at indices 3, 7, 11, 15, 19, 23, 27, 31 (every 4th, starting at 3)
        for (i, lt) in config.layer_types.iter().enumerate() {
            let expected = if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            };
            assert_eq!(*lt, expected, "layer {} mismatch", i);
        }
    }

    #[test]
    fn test_derived_dimensions() {
        let config = Config35::from_file(MODEL_PATH).unwrap();

        // Full attention: q_proj = [8192, 2560] (q + gate)
        assert_eq!(config.full_attn_q_proj_dim(), 8192);
        assert_eq!(config.full_attn_q_dim(), 4096);
        assert_eq!(config.full_attn_kv_dim(), 1024);

        // Linear attention: in_proj_qkv = [8192, 2560]
        // q=2048 + k=2048 + v=4096 = 8192
        assert_eq!(config.linear_attn_qkv_dim(), 8192);
        assert_eq!(config.linear_attn_z_dim(), 4096);
    }
}
