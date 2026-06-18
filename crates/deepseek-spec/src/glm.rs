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

    /// Map this `glm_moe_dsa` config onto the shared [`crate::v4::DeepSeekV4Config`]
    /// runtime authority at the DeepSeek-V3.2 (V32) MLA shape (`head_dim =
    /// kv_lora_rank + qk_rope_head_dim = 576`). The result drives the shared
    /// FlashMLA V32 path with `SparseIndexed` mode on every layer, a plain
    /// `o_proj`, hyper-connections bypassed (`hc_mult == 1`), and no MTP
    /// (`num_nextn_predict_layers == 0` — the checkpoint ships no MTP tensors).
    ///
    /// Does NOT call `DeepSeekV4Config::validate` (that gate is DSv4-strict and
    /// rejects the GLM dialect's `o_lora_rank == 0` / `model_type`).
    pub fn into_deepseek_v4(&self) -> Result<crate::v4::DeepSeekV4Config> {
        use crate::v4::{
            DeepSeekV4AttentionMode, DeepSeekV4Config, DeepSeekV4RopeParameters, TensorDialect,
        };

        let n = self.num_hidden_layers;
        let rope_theta = self.rope_theta() as f32;

        Ok(DeepSeekV4Config {
            architectures: self.architectures.clone(),
            model_type: self.model_type.clone(),
            dtype: "bfloat16".to_string(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: n,
            num_attention_heads: self.num_attention_heads,
            // MLA latent KV: one logical kv head; GLM's nominal 64 is not the MLA head count.
            num_key_value_heads: 1,
            head_dim: self.kv_lora_rank + self.qk_rope_head_dim,
            hidden_act: "silu".to_string(),
            // GLM has an UNCLAMPED SwiGLU (no gpt-oss-style ±limit clamp that DSv4
            // uses, e.g. 10.0). The DSv4 FP8 MoE swiglu kernel clamps
            // `gate=min(gate,limit)` / `up=clamp(up,±limit)` and rejects `limit<=0`,
            // so express "no clamp" as a large finite sentinel: `fminf`/`fmaxf`
            // against f32::MAX is a no-op for any real activation, and it satisfies
            // the kernel's `limit > 0` precondition.
            swiglu_limit: f32::MAX,
            q_lora_rank: self.q_lora_rank,
            // GLM has plain o_proj; the wo_a→wo_b low-rank path is unused.
            o_lora_rank: 0,
            o_groups: 1,
            qk_rope_head_dim: self.qk_rope_head_dim,
            n_routed_experts: self.n_routed_experts,
            n_shared_experts: self.n_shared_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate_size: self.moe_intermediate_size,
            routed_scaling_factor: self.routed_scaling_factor,
            norm_topk_prob: self.norm_topk_prob,
            scoring_func: self.scoring_func.clone(),
            topk_method: self.topk_method.clone(),
            index_n_heads: self.index_n_heads,
            index_head_dim: self.index_head_dim,
            index_topk: self.index_topk,
            num_hash_layers: 0,
            sliding_window: 0,
            // Mode comes from the explicit per-layer override; this is a placeholder.
            compress_ratios: vec![0; n],
            compress_rope_theta: rope_theta,
            // GLM has no hyper-connections: identity mixers (stream_dim == hidden).
            hc_mult: 1,
            hc_sinkhorn_iters: 0,
            hc_eps: 0.0,
            // Checkpoint ships no MTP tensors.
            num_nextn_predict_layers: 0,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta,
            rope_parameters: DeepSeekV4RopeParameters {
                rope_type: "default".to_string(),
                factor: 1.0,
                original_max_position_embeddings: 0,
                beta_fast: 0.0,
                beta_slow: 0.0,
                rope_theta: Some(rope_theta),
            },
            rms_norm_eps: self.rms_norm_eps,
            initializer_range: 0.0,
            tie_word_embeddings: self.tie_word_embeddings,
            attention_bias: self.attention_bias,
            attention_dropout: 0.0,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            // GLM-5.2 dialect extensions.
            kv_lora_rank: self.kv_lora_rank,
            qk_nope_head_dim: self.qk_nope_head_dim,
            v_head_dim: self.v_head_dim,
            plain_o_proj: true,
            // DSA indexer on every layer, no compressor.
            per_layer_attention_mode: Some(vec![DeepSeekV4AttentionMode::SparseIndexed; n]),
            per_layer_dense_mlp: Some((0..n).map(|l| !self.is_sparse_layer(l)).collect()),
            per_layer_full_indexer: Some((0..n).map(|l| self.is_full_indexer_layer(l)).collect()),
            // GLM-5.2 HF tensor names.
            tensor_dialect: TensorDialect::Glm,
        })
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
    fn into_deepseek_v4_maps_glm52() {
        use crate::v4::DeepSeekV4AttentionMode;

        let c = GlmMoeDsaConfig::from_json_str(GLM52).unwrap();
        let v4 = c.into_deepseek_v4().unwrap();

        // V32 MLA shape: head_dim = kv_lora_rank (512) + qk_rope_head_dim (64).
        assert_eq!(v4.head_dim, 576);
        assert!(v4.plain_o_proj);

        let modes = v4.per_layer_attention_mode.as_ref().unwrap();
        assert_eq!(modes.len(), 6);
        assert!(
            modes
                .iter()
                .all(|m| *m == DeepSeekV4AttentionMode::SparseIndexed)
        );

        // first_k_dense_replace=3 ⇒ layers 0..3 dense, 3..6 MoE.
        let dense = v4.per_layer_dense_mlp.as_ref().unwrap();
        assert_eq!(dense, &vec![true, true, true, false, false, false]);

        assert_eq!(v4.num_key_value_heads, 1);
        assert_eq!(v4.hc_mult, 1);
        assert_eq!(v4.tensor_dialect, crate::v4::TensorDialect::Glm);
    }

    /// GLM HF tensor names emitted by the `Glm` dialect, cross-checked against the
    /// real `/data02/models/GLM-5.2-FP8/model.safetensors.index.json`.
    #[test]
    fn glm_dialect_emits_hf_tensor_names() {
        let v4 = GlmMoeDsaConfig::from_json_str(GLM52)
            .unwrap()
            .into_deepseek_v4()
            .unwrap();

        let top = v4.tensor_names();
        assert_eq!(top.embed_tokens(), "model.embed_tokens.weight");
        assert_eq!(top.lm_head(), "lm_head.weight");
        assert_eq!(top.norm(), "model.norm.weight");

        // Layer 0 is dense (first_k_dense_replace=3) and a "full" indexer layer.
        let l0 = v4.layer_tensor_names(0);
        assert_eq!(l0.attn_norm, "model.layers.0.input_layernorm.weight");
        assert_eq!(
            l0.ffn_norm,
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(l0.attn.wq_a, "model.layers.0.self_attn.q_a_proj.weight");
        assert_eq!(
            l0.attn.q_norm,
            "model.layers.0.self_attn.q_a_layernorm.weight"
        );
        assert_eq!(l0.attn.wq_b, "model.layers.0.self_attn.q_b_proj.weight");
        assert_eq!(
            l0.attn.wkv,
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        );
        assert_eq!(
            l0.attn.kv_norm,
            "model.layers.0.self_attn.kv_a_layernorm.weight"
        );
        assert_eq!(
            l0.attn.kv_b_proj.as_deref(),
            Some("model.layers.0.self_attn.kv_b_proj.weight")
        );
        assert_eq!(
            l0.attn.o_proj.as_deref(),
            Some("model.layers.0.self_attn.o_proj.weight")
        );
        assert!(l0.attn.compressor.is_none());
        // Full-indexer layer carries the DSA Lightning Indexer weights.
        let idx = l0.attn.indexer.as_ref().unwrap();
        assert_eq!(idx.wq_b, "model.layers.0.self_attn.indexer.wq_b.weight");
        assert_eq!(
            idx.weights_proj,
            "model.layers.0.self_attn.indexer.weights_proj.weight"
        );
        assert_eq!(
            idx.wk.as_deref(),
            Some("model.layers.0.self_attn.indexer.wk.weight")
        );
        assert_eq!(
            idx.k_norm.as_deref(),
            Some("model.layers.0.self_attn.indexer.k_norm.weight")
        );
        assert_eq!(
            idx.k_norm_bias.as_deref(),
            Some("model.layers.0.self_attn.indexer.k_norm.bias")
        );
        assert!(idx.compressor.is_none());
        // Dense layer: a plain MLP, no router gate / experts.
        let dense = l0.ffn.dense_mlp.as_ref().unwrap();
        assert_eq!(dense.w1, "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(dense.w3, "model.layers.0.mlp.up_proj.weight");
        assert_eq!(dense.w2, "model.layers.0.mlp.down_proj.weight");

        // Layer 3 is sparse (MoE) and a "shared" indexer layer (no indexer tensors).
        let l3 = v4.layer_tensor_names(3);
        assert!(l3.attn.indexer.is_none());
        assert!(l3.ffn.dense_mlp.is_none());
        assert_eq!(l3.ffn.gate_weight, "model.layers.3.mlp.gate.weight");
        assert_eq!(
            l3.ffn.gate_bias.as_deref(),
            Some("model.layers.3.mlp.gate.e_score_correction_bias")
        );
        assert_eq!(
            l3.ffn.expert(0).w1,
            "model.layers.3.mlp.experts.0.gate_proj.weight"
        );
        assert_eq!(
            l3.ffn.expert(0).w3,
            "model.layers.3.mlp.experts.0.up_proj.weight"
        );
        assert_eq!(
            l3.ffn.expert(0).w2,
            "model.layers.3.mlp.experts.0.down_proj.weight"
        );
        let shared = l3.ffn.shared_experts.as_ref().unwrap();
        assert_eq!(
            shared.w1,
            "model.layers.3.mlp.shared_experts.gate_proj.weight"
        );
        assert_eq!(
            shared.w3,
            "model.layers.3.mlp.shared_experts.up_proj.weight"
        );
        assert_eq!(
            shared.w2,
            "model.layers.3.mlp.shared_experts.down_proj.weight"
        );
    }

    /// The `Dsv4` dialect (default) must keep the abbreviated DSv4 names
    /// byte-unchanged — the GLM dialect work must not perturb the DSv4 path.
    #[test]
    fn dsv4_dialect_names_unchanged() {
        use crate::v4::DeepSeekV4Config;
        let cfg = DeepSeekV4Config::from_json_str(
            r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "torch_dtype": "bfloat16",
            "vocab_size": 129280, "hidden_size": 4096, "num_hidden_layers": 3,
            "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
            "hidden_act": "silu", "swiglu_limit": 10.0,
            "q_lora_rank": 1024, "o_lora_rank": 1024, "o_groups": 8,
            "qk_rope_head_dim": 64, "n_routed_experts": 256, "n_shared_experts": 1,
            "num_experts_per_tok": 6, "moe_intermediate_size": 2048,
            "routed_scaling_factor": 1.5, "norm_topk_prob": true,
            "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
            "index_n_heads": 64, "index_head_dim": 128, "index_topk": 512,
            "num_hash_layers": 1, "sliding_window": 128,
            "compress_ratios": [0, 4, 32], "compress_rope_theta": 160000.0,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1.0e-6,
            "num_nextn_predict_layers": 0, "max_position_embeddings": 1048576,
            "rope_theta": 10000.0,
            "rope_scaling": {"type": "yarn", "factor": 16.0,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32.0, "beta_slow": 1.0},
            "rms_norm_eps": 1.0e-6, "initializer_range": 0.02,
            "tie_word_embeddings": false, "attention_bias": false,
            "attention_dropout": 0.0, "bos_token_id": 0, "eos_token_id": 1
        }"#,
        )
        .unwrap();
        assert_eq!(cfg.tensor_dialect, crate::v4::TensorDialect::Dsv4);

        let top = cfg.tensor_names();
        assert_eq!(top.embed_tokens(), "embed.weight");
        assert_eq!(top.lm_head(), "head.weight");
        assert_eq!(top.norm(), "norm.weight");

        // CSA layer (compress_ratio=4): abbreviated names + compressor + indexer.
        let l1 = cfg.layer_tensor_names(1);
        assert_eq!(l1.attn_norm, "layers.1.attn_norm.weight");
        assert_eq!(l1.ffn_norm, "layers.1.ffn_norm.weight");
        assert_eq!(l1.attn.wq_a, "layers.1.attn.wq_a.weight");
        assert_eq!(l1.attn.q_norm, "layers.1.attn.q_norm.weight");
        assert_eq!(l1.attn.wkv, "layers.1.attn.wkv.weight");
        assert_eq!(l1.attn.wo_a, "layers.1.attn.wo_a.weight");
        assert_eq!(l1.attn.wo_b, "layers.1.attn.wo_b.weight");
        assert_eq!(l1.attn.attn_sink, "layers.1.attn.attn_sink");
        assert!(l1.attn.kv_b_proj.is_none());
        assert!(l1.attn.o_proj.is_none());
        assert!(l1.attn.compressor.is_some());
        let idx = l1.attn.indexer.as_ref().unwrap();
        assert_eq!(idx.wq_b, "layers.1.attn.indexer.wq_b.weight");
        assert!(idx.wk.is_none() && idx.k_norm.is_none());
        assert!(idx.compressor.is_some());
        assert_eq!(l1.ffn.gate_bias.as_deref(), Some("layers.1.ffn.gate.bias"));
        assert_eq!(l1.ffn.expert(7).w2, "layers.1.ffn.experts.7.w2.weight");
        assert!(l1.ffn.dense_mlp.is_none());
        assert_eq!(
            l1.ffn.shared_experts.as_ref().unwrap().w3,
            "layers.1.ffn.shared_experts.w3.weight"
        );
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
