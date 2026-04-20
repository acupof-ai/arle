pub(crate) use qwen35_spec::{LayerType, Qwen35Config as Config35};

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");

    /// Candidate locations for the Qwen3.6 MoE config.json under the HF cache.
    /// Test scans them; if none exists, the test is `ignore`-style skipped.
    fn qwen36_config_path() -> Option<std::path::PathBuf> {
        let hf_home = std::env::var_os("HF_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| {
                    std::path::PathBuf::from(h)
                        .join(".cache")
                        .join("huggingface")
                })
            })?;
        let snapshots_root = hf_home
            .join("hub")
            .join("models--mlx-community--Qwen3.6-35B-A3B-4bit")
            .join("snapshots");
        let entries = std::fs::read_dir(&snapshots_root).ok()?;
        for entry in entries.flatten() {
            let candidate = entry.path().join("config.json");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        None
    }

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

        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 256);

        assert_eq!(config.linear_num_key_heads, 16);
        assert_eq!(config.linear_key_head_dim, 128);
        assert_eq!(config.linear_num_value_heads, 32);
        assert_eq!(config.linear_value_head_dim, 128);
        assert_eq!(config.linear_conv_kernel_dim, 4);

        assert_eq!(config.rope_theta, 1_000_000.0);
        assert_eq!(config.rotary_dim, 128);
        assert_eq!(config.rope_cache_len_hint(), Some(32_768));
    }

    #[test]
    fn test_layer_types() {
        let config = Config35::from_file(MODEL_PATH).unwrap();
        assert_eq!(config.layer_types.len(), config.num_hidden_layers);

        for (index, layer_type) in config.layer_types.iter().enumerate() {
            let expected = if index < 8 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            };
            assert_eq!(*layer_type, expected);
        }
        assert_eq!(config.num_full_attention_layers(), 8);
    }

    #[test]
    fn test_projection_dims() {
        let config = Config35::from_file(MODEL_PATH).unwrap();
        assert_eq!(config.full_attn_q_proj_dim(), 8192);
        assert_eq!(config.full_attn_q_dim(), 4096);
        assert_eq!(config.full_attn_kv_dim(), 1024);
        assert_eq!(config.linear_attn_qkv_dim(), 8192);
        assert_eq!(config.linear_attn_z_dim(), 4096);
    }

    /// Qwen3.6 MoE config smoke test. Skipped unless the mlx-community
    /// Qwen3.6-35B-A3B-4bit weights have been downloaded under the HF cache.
    #[test]
    fn test_load_config36_moe_when_weights_present() {
        let Some(path) = qwen36_config_path() else {
            eprintln!("skipping Qwen3.6 MoE test: config.json not found under HF cache");
            return;
        };
        let content = std::fs::read_to_string(&path).expect("read Qwen3.6 config.json");
        let config = Config35::from_json_str(&content).expect("parse Qwen3.6 config.json");
        assert!(
            config.is_moe(),
            "Qwen3.6 checkpoint must report is_moe() == true"
        );
        assert_eq!(config.num_experts, 256);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.moe_intermediate_size, 512);
        assert_eq!(config.shared_expert_intermediate_size, 512);
        assert_eq!(config.decoder_sparse_step, 1);
        assert_eq!(config.num_hidden_layers, 40);
        // All 40 layers should be MoE (mlp_only_layers is empty).
        for idx in 0..config.num_hidden_layers {
            assert!(config.is_moe_layer(idx));
        }
    }
}
