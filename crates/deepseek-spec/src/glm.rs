//! GLM-5.2 (`model_type=glm_moe_dsa` / `GlmMoeDsaForCausalLM`) config dialect.
//!
//! GLM-5.2 is the same architecture family as DSv4-Flash — MLA + DeepSeek
//! Sparse Attention (DSA) Lightning Indexer + DeepSeek-MoE (sigmoid/noaux_tc) +
//! MTP. SGLang models it as `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`. The
//! runtime (FlashMLA sparse decode, interleaved RoPE `freqs_cis`, DeepGEMM FP8
//! MoE, EP=8 DeepEP) is shared with [`crate::v4`]; this module only parses the
//! GLM config.json dialect.
//!
//! Two structural differences from DSv4-Flash drive the still-pending runtime
//! adapter (NOT in this module — it touches the shared [`crate::v4`] structs):
//!   1. **Indexer without compressor.** GLM is DeepSeek-V3.2-style: the DSA
//!      indexer runs over the full (uncompressed) MLA latent KV. DSv4's
//!      `DeepSeekV4IndexerShape` currently embeds a compressor shape; the GLM
//!      path needs a 4th attention mode (indexer ✓, compressor ✗).
//!   2. **Plain `o_proj`.** GLM has no output low-rank (`o_lora_rank`/`o_groups`
//!      absent); DSv4's MLA forward always projects through `wo_a → wo_b`.
//!
//! Per-layer schedule is explicit here (`mlp_layer_types`, `indexer_types`)
//! rather than DSv4's `compress_ratios`-derived modes.

use serde::Deserialize;

use crate::{DeepSeekConfigError, Result};

/// `rope_parameters` sub-object: `{"rope_theta": 8e6, "rope_type": "default"}`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmRopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default, alias = "type")]
    pub rope_type: String,
}

fn default_rope_theta() -> f64 {
    10_000.0
}

/// Faithful parse of a `glm_moe_dsa` checkpoint config.json. Unknown keys are
/// ignored (serde default), so the full HF config parses without modelling
/// every field. Mapping into the shared runtime config is a separate adapter.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmMoeDsaConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    // MLA low-rank. GLM has NO o_lora_rank/o_groups (plain o_proj).
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,

    // DSA Lightning Indexer.
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    #[serde(default = "default_one")]
    pub index_topk_freq: usize,
    #[serde(default)]
    pub index_skip_topk_offset: usize,
    #[serde(default)]
    pub index_share_for_mtp_iteration: bool,
    #[serde(default)]
    pub indexer_rope_interleave: bool,

    // RoPE. `rope_interleave=true` ⇒ interleaved (GPT-J adjacent-pair), which
    // matches the official DSA `freqs_cis` layout the shared runtime already
    // builds (`dsv4_dsa_freqs_cis_real`).
    #[serde(default)]
    pub rope_interleave: bool,
    #[serde(default)]
    pub rope_parameters: Option<GlmRopeParameters>,
    #[serde(default)]
    pub max_position_embeddings: usize,

    // MoE (DeepSeek-style routing).
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub moe_intermediate_size: usize,
    #[serde(default = "default_one")]
    pub n_group: usize,
    #[serde(default = "default_one")]
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub scoring_func: String,
    #[serde(default)]
    pub topk_method: String,
    #[serde(default)]
    pub first_k_dense_replace: usize,

    // Explicit per-layer schedule (replaces DSv4 compress_ratios).
    #[serde(default)]
    pub mlp_layer_types: Vec<String>,
    #[serde(default)]
    pub indexer_types: Vec<String>,

    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    #[serde(default)]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_one() -> usize {
    1
}

impl GlmMoeDsaConfig {
    pub fn from_json_str(content: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "glm_moe_dsa"
            && !self
                .architectures
                .iter()
                .any(|a| a == "GlmMoeDsaForCausalLM")
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "GLM config must be model_type=glm_moe_dsa / GlmMoeDsaForCausalLM",
            ));
        }
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.q_lora_rank == 0
            || self.kv_lora_rank == 0
            || self.qk_rope_head_dim == 0
            || self.index_n_heads == 0
            || self.index_head_dim == 0
            || self.index_topk == 0
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "GLM MLA/indexer dimensions must be non-zero",
            ));
        }
        if self.num_experts_per_tok > self.n_routed_experts {
            return Err(DeepSeekConfigError::InvalidConfig(
                "num_experts_per_tok must not exceed n_routed_experts",
            ));
        }
        let n = self.num_hidden_layers;
        if !self.mlp_layer_types.is_empty() && self.mlp_layer_types.len() != n {
            return Err(DeepSeekConfigError::InvalidConfig(
                "mlp_layer_types length must match num_hidden_layers",
            ));
        }
        if !self.indexer_types.is_empty() && self.indexer_types.len() != n {
            return Err(DeepSeekConfigError::InvalidConfig(
                "indexer_types length must match num_hidden_layers",
            ));
        }
        Ok(())
    }

    /// MoE (sparse) vs dense MLP for `layer_idx`. Falls back to the
    /// `first_k_dense_replace` rule when `mlp_layer_types` is absent.
    pub fn is_sparse_layer(&self, layer_idx: usize) -> bool {
        match self.mlp_layer_types.get(layer_idx) {
            Some(kind) => kind == "sparse",
            None => layer_idx >= self.first_k_dense_replace,
        }
    }

    /// `true` ⇒ this layer recomputes the indexer top-k set; `false` ("shared")
    /// reuses the cached top-k from the previous "full" layer. Falls back to the
    /// `index_topk_freq` schedule when `indexer_types` is absent.
    pub fn is_full_indexer_layer(&self, layer_idx: usize) -> bool {
        match self.indexer_types.get(layer_idx) {
            Some(kind) => kind == "full",
            None => self.index_topk_freq <= 1 || layer_idx.is_multiple_of(self.index_topk_freq),
        }
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.rope_theta)
            .unwrap_or_else(default_rope_theta)
    }

    /// GLM uses plain `o_proj` (no output low-rank).
    pub fn has_output_low_rank(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Representative slice of a real GLM-5.2-FP8 config.json (short layer arrays).
    const GLM52: &str = r#"{
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "vocab_size": 154880,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "num_hidden_layers": 6,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 192,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 256,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "index_share_for_mtp_iteration": true,
        "indexer_rope_interleave": true,
        "rope_interleave": true,
        "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
        "max_position_embeddings": 1048576,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "first_k_dense_replace": 3,
        "mlp_layer_types": ["dense","dense","dense","sparse","sparse","sparse"],
        "indexer_types": ["full","full","full","shared","shared","full"],
        "num_nextn_predict_layers": 1,
        "rms_norm_eps": 1e-05,
        "hidden_act": "silu",
        "tie_word_embeddings": false,
        "attention_bias": false,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    }"#;

    #[test]
    fn parses_glm52_config() {
        let c = GlmMoeDsaConfig::from_json_str(GLM52).unwrap();
        assert_eq!(c.model_type, "glm_moe_dsa");
        assert_eq!(c.q_lora_rank, 2048);
        assert_eq!(c.kv_lora_rank, 512);
        assert_eq!(c.qk_nope_head_dim, 192);
        assert_eq!(c.qk_rope_head_dim, 64);
        assert_eq!(c.v_head_dim, 256);
        assert_eq!(c.index_topk, 2048);
        assert_eq!(c.n_routed_experts, 256);
        assert_eq!(c.scoring_func, "sigmoid");
        assert_eq!(c.topk_method, "noaux_tc");
        assert!(c.rope_interleave);
        assert_eq!(c.rope_theta(), 8_000_000.0);
        assert!(!c.has_output_low_rank());
    }

    #[test]
    fn layer_schedule_from_explicit_arrays() {
        let c = GlmMoeDsaConfig::from_json_str(GLM52).unwrap();
        // first_k_dense_replace=3 ⇒ layers 0..3 dense, 3..6 sparse.
        assert!(!c.is_sparse_layer(0));
        assert!(!c.is_sparse_layer(2));
        assert!(c.is_sparse_layer(3));
        assert!(c.is_sparse_layer(5));
        // indexer_types: full,full,full,shared,shared,full
        assert!(c.is_full_indexer_layer(0));
        assert!(c.is_full_indexer_layer(2));
        assert!(!c.is_full_indexer_layer(3));
        assert!(!c.is_full_indexer_layer(4));
        assert!(c.is_full_indexer_layer(5));
    }

    #[test]
    fn rejects_mismatched_layer_arrays() {
        let bad = GLM52.replace(
            "\"mlp_layer_types\": [\"dense\",\"dense\",\"dense\",\"sparse\",\"sparse\",\"sparse\"]",
            "\"mlp_layer_types\": [\"dense\",\"sparse\"]",
        );
        assert!(GlmMoeDsaConfig::from_json_str(&bad).is_err());
    }

    #[test]
    fn falls_back_to_freq_schedule_without_arrays() {
        let no_arrays = GLM52
            .replace("\"indexer_types\": [\"full\",\"full\",\"full\",\"shared\",\"shared\",\"full\"],", "")
            .replace("\"mlp_layer_types\": [\"dense\",\"dense\",\"dense\",\"sparse\",\"sparse\",\"sparse\"],", "");
        let c = GlmMoeDsaConfig::from_json_str(&no_arrays).unwrap();
        // index_topk_freq=4 ⇒ full at 0,4; shared elsewhere.
        assert!(c.is_full_indexer_layer(0));
        assert!(!c.is_full_indexer_layer(1));
        assert!(c.is_full_indexer_layer(4));
        // first_k_dense_replace=3 fallback.
        assert!(!c.is_sparse_layer(2));
        assert!(c.is_sparse_layer(3));
    }
}
