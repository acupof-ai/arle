use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{DeepSeekConfigError, Result, Shard};

/// Tensor-name dialect for the safetensors checkpoint this config drives.
///
/// `Dsv4` (default) emits the DSv4-Flash abbreviated names (`layers.N.attn.wq_a`
/// …) byte-unchanged. `Glm` emits the GLM-5.2 (`glm_moe_dsa`) HF names
/// (`model.layers.N.self_attn.q_a_proj` …); the GLM adapter
/// [`crate::glm::GlmMoeDsaConfig::into_deepseek_v4`] sets it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDialect {
    #[default]
    Dsv4,
    Glm,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4RopeParameters {
    #[serde(default, alias = "type")]
    pub rope_type: String,
    pub factor: f32,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    #[serde(default)]
    pub rope_theta: Option<f32>,
}

impl DeepSeekV4RopeParameters {
    pub fn original_seq_len_i32(&self) -> Result<i32> {
        i32::try_from(self.original_max_position_embeddings).map_err(|_| {
            DeepSeekConfigError::InvalidConfig("original_max_position_embeddings overflows i32")
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    #[serde(alias = "torch_dtype")]
    pub dtype: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub swiglu_limit: f32,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub qk_rope_head_dim: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub scoring_func: String,
    pub topk_method: String,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub num_hash_layers: usize,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f32,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub num_nextn_predict_layers: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    #[serde(alias = "rope_scaling")]
    pub rope_parameters: DeepSeekV4RopeParameters,
    pub rms_norm_eps: f32,
    pub initializer_range: f32,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    // GLM-5.2 (glm_moe_dsa) dialect extensions. Default/None ⇒ DSv4-Flash
    // absorbed-MODEL1 semantics, byte-unchanged. Populated by
    // `crate::glm::GlmMoeDsaConfig::into_deepseek_v4`.
    #[serde(default)]
    pub kv_lora_rank: usize,
    #[serde(default)]
    pub qk_nope_head_dim: usize,
    #[serde(default)]
    pub v_head_dim: usize,
    /// `true` ⇒ standard plain `o_proj` (GLM); `false` ⇒ DSv4 wo_a→wo_b low-rank.
    #[serde(default)]
    pub plain_o_proj: bool,
    /// Explicit per-layer attention mode (GLM). `None` ⇒ derive from `compress_ratios`.
    #[serde(skip)]
    pub per_layer_attention_mode: Option<Vec<DeepSeekV4AttentionMode>>,
    /// Per-layer dense-MLP (vs MoE) flag (GLM `mlp_layer_types`). `None` ⇒ all-MoE.
    #[serde(skip)]
    pub per_layer_dense_mlp: Option<Vec<bool>>,
    /// Per-layer "full indexer" (recompute topk) flag. `None` ⇒ DSv4.
    #[serde(skip)]
    pub per_layer_full_indexer: Option<Vec<bool>>,
    /// Safetensors tensor-name dialect. `Dsv4` (default) ⇒ abbreviated DSv4
    /// names byte-unchanged; `Glm` ⇒ GLM-5.2 HF names.
    #[serde(skip)]
    pub tensor_dialect: TensorDialect,
    // DSpark spec-decode config. `dspark_block_size == 0` ⇒ not a DSpark
    // checkpoint (native MTP / non-spec), byte-unchanged. The DSpark *draft*
    // config sets 5; the base serve `config.json` stays 0 unless `--spec-type
    // dspark` merges the draft, so `is_dspark()` is false on a plain FP8 serve.
    #[serde(default)]
    pub dspark_block_size: usize,
    #[serde(default)]
    pub dspark_target_layer_ids: Vec<usize>,
    #[serde(default)]
    pub dspark_markov_rank: usize,
    #[serde(default)]
    pub dspark_noise_token_id: u32,
    /// Number of stacked DSpark draft blocks (`mtp.0`…`mtp.{n-1}`). `0` ⇒ not a
    /// DSpark checkpoint; the real checkpoint = 3.
    #[serde(default)]
    pub dspark_num_stages: usize,
}

impl DeepSeekV4Config {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json_str(&content)
    }

    pub fn from_json_str(content: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(content)?;
        Self::from_json_value(&value)
    }

    pub fn from_json_value(value: &serde_json::Value) -> Result<Self> {
        let mut value = value.clone();
        normalize_rope_parameters_aliases(&mut value);
        let mut config: Self = serde_json::from_value(value)?;
        // 0731 checkpoint ships 46 compress_ratios (43 hidden + MTP + 2 trailing);
        // truncate to hidden layers, validated below.
        if config.compress_ratios.len() > config.num_hidden_layers + config.num_nextn_predict_layers
        {
            config
                .compress_ratios
                .truncate(config.num_hidden_layers + config.num_nextn_predict_layers);
        }
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "deepseek_v4" {
            return Err(DeepSeekConfigError::InvalidConfig(
                "model_type must be deepseek_v4",
            ));
        }
        if !self
            .architectures
            .iter()
            .any(|arch| arch == "DeepseekV4ForCausalLM")
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "architectures must contain DeepseekV4ForCausalLM",
            ));
        }
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "hidden size, layers, and attention heads must be non-zero",
            ));
        }
        if self.num_key_value_heads != 1 {
            return Err(DeepSeekConfigError::InvalidConfig(
                "DSV4 replica expects num_key_value_heads=1",
            ));
        }
        if self.q_lora_rank == 0
            || self.o_lora_rank == 0
            || self.o_groups == 0
            || self.qk_rope_head_dim == 0
            || self.index_n_heads == 0
            || self.index_head_dim == 0
            || self.index_topk == 0
            || self.hc_mult == 0
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "DSV4 low-rank, indexer, mHC, and routing dimensions must be non-zero",
            ));
        }
        if self.n_routed_experts == 0
            || self.num_experts_per_tok == 0
            || self.moe_intermediate_size == 0
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "DSV4 routed MoE dimensions must be non-zero",
            ));
        }
        if self.num_experts_per_tok > self.n_routed_experts {
            return Err(DeepSeekConfigError::InvalidConfig(
                "num_experts_per_tok must not exceed n_routed_experts",
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.o_groups) {
            return Err(DeepSeekConfigError::InvalidConfig(
                "num_attention_heads must be divisible by o_groups",
            ));
        }
        let compress_ratio_count = self.compress_ratios.len();
        let hidden_plus_mtp = self.num_hidden_layers + self.num_nextn_predict_layers;
        if compress_ratio_count != self.num_hidden_layers && compress_ratio_count != hidden_plus_mtp
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "compress_ratios length must match num_hidden_layers or include MTP layers",
            ));
        }
        if self.rope_parameters.rope_type.is_empty() {
            return Err(DeepSeekConfigError::InvalidConfig(
                "rope_parameters rope_type/type must be set",
            ));
        }
        Ok(())
    }

    pub fn tensor_names(&self) -> DeepSeekV4TensorNames {
        DeepSeekV4TensorNames {
            dialect: self.tensor_dialect,
        }
    }

    pub fn layer_tensor_names(&self, layer_idx: usize) -> DeepSeekV4LayerTensorNames {
        let compress_ratio = self.compress_ratios[layer_idx];
        // GLM: per-layer dense-MLP / full-indexer come from the explicit schedule,
        // not `compress_ratios`. `indexer present` ⇒ this layer carries indexer
        // tensors (a "full" layer); "shared" layers reuse the prior topk and ship
        // no indexer weights. `dense MLP` ⇒ no expert stack, a plain FFN.
        let is_dense_mlp = self
            .per_layer_dense_mlp
            .as_ref()
            .and_then(|flags| flags.get(layer_idx).copied())
            .unwrap_or(false);
        let has_indexer = match &self.per_layer_full_indexer {
            // GLM: indexer tensors only on "full" layers.
            Some(flags) => flags.get(layer_idx).copied().unwrap_or(false),
            // DSv4: derive from compress_ratio (CSA layers carry the indexer).
            None => self
                .attention_mode_for_compress_ratio(compress_ratio)
                .has_indexer(),
        };
        self.tensor_names().layer(
            layer_idx,
            compress_ratio,
            layer_idx < self.num_hash_layers,
            self.n_shared_experts > 0,
            is_dense_mlp,
            has_indexer,
        )
    }

    pub fn mtp_tensor_names(&self, mtp_idx: usize) -> DeepSeekV4MtpTensorNames {
        DeepSeekV4MtpTensorNames::new(format!("mtp.{mtp_idx}"), self.n_shared_experts > 0)
    }

    pub fn is_dspark(&self) -> bool {
        self.dspark_block_size > 0
    }

    /// GLM-5.2 has a single hyper-connection (`hc_mult == 1`, identity mixers),
    /// unlike DSv4-Flash's multi-HC folds.
    pub fn is_glm(&self) -> bool {
        self.hc_mult == 1
    }

    pub fn dspark_num_stages(&self) -> usize {
        self.dspark_num_stages
    }

    pub fn dspark_tensor_names(&self, mtp_idx: usize) -> DeepSeekV4DsparkTensorNames {
        DeepSeekV4DsparkTensorNames::new(
            format!("mtp.{mtp_idx}"),
            mtp_idx,
            self.dspark_num_stages(),
            self.n_shared_experts > 0,
        )
    }

    pub fn shard_for_global_tensor(&self, name: &str) -> Option<Shard> {
        match name {
            "embed.weight" | "head.weight" => Some(Shard::VocabParallel { dim: 0 }),
            "norm.weight" | "hc_head_base" | "hc_head_fn" | "hc_head_scale" => {
                Some(Shard::Replicated)
            }
            _ => None,
        }
    }

    pub fn attention_mode_for_compress_ratio(
        &self,
        compress_ratio: usize,
    ) -> DeepSeekV4AttentionMode {
        DeepSeekV4AttentionMode::from_compress_ratio(compress_ratio)
    }

    pub fn attention_layer_plan(&self, layer_idx: usize) -> Option<DeepSeekV4AttentionLayerPlan> {
        let compress_ratio = *self.compress_ratios.get(layer_idx)?;
        let mode = match &self.per_layer_attention_mode {
            Some(modes) => *modes.get(layer_idx)?,
            None => self.attention_mode_for_compress_ratio(compress_ratio),
        };
        Some(DeepSeekV4AttentionLayerPlan {
            layer_idx,
            compress_ratio,
            mode,
            hash_routing: self.moe_routing_kind(layer_idx) == DeepSeekV4MoeRoutingKind::Hash,
            has_compressor: mode.has_compressor(),
            has_indexer: mode.has_indexer(),
            sliding_window: self.sliding_window,
            index_topk: mode.has_indexer().then_some(self.index_topk),
        })
    }

    pub fn compressor_shape(&self, compress_ratio: usize) -> Option<DeepSeekV4CompressorShape> {
        (compress_ratio > 0).then(|| {
            let overlap = compress_ratio < 16;
            let coeff = if overlap { 2 } else { 1 };
            DeepSeekV4CompressorShape {
                compress_ratio,
                overlap,
                wkv_rows: coeff * self.head_dim,
                wkv_cols: self.hidden_size,
                wgate_rows: coeff * self.head_dim,
                wgate_cols: self.hidden_size,
                ape_rows: compress_ratio,
                ape_cols: coeff * self.head_dim,
                norm_len: self.head_dim,
            }
        })
    }

    pub fn indexer_shape(&self, compress_ratio: usize) -> Option<DeepSeekV4IndexerShape> {
        let mode = self.attention_mode_for_compress_ratio(compress_ratio);
        mode.has_indexer().then(|| DeepSeekV4IndexerShape {
            compress_ratio,
            wq_b_rows: self.index_n_heads * self.index_head_dim,
            wq_b_cols: self.q_lora_rank,
            weights_proj_rows: self.index_n_heads,
            weights_proj_cols: self.hidden_size,
            key_head_dim: self.index_head_dim,
            key_heads: self.index_n_heads,
            topk: self.index_topk,
            compressor: (mode == DeepSeekV4AttentionMode::CompressedSparse).then(|| {
                self.compressor_shape(compress_ratio)
                    .expect("CSA compress_ratio must have compressor shape")
            }),
        })
    }

    pub fn moe_routing_kind(&self, layer_idx: usize) -> DeepSeekV4MoeRoutingKind {
        if layer_idx < self.num_hash_layers {
            DeepSeekV4MoeRoutingKind::Hash
        } else {
            DeepSeekV4MoeRoutingKind::LearnedBias
        }
    }

    pub fn router_scores_from_logits(&self, logits: &[f32]) -> Result<Vec<f32>> {
        if logits.len() != self.n_routed_experts {
            return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                "router logits length {} does not match n_routed_experts {}",
                logits.len(),
                self.n_routed_experts
            )));
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(DeepSeekConfigError::InvalidForwardBatch(
                "router logits must be finite".to_string(),
            ));
        }
        match self.scoring_func.as_str() {
            "softmax" => Ok(stable_softmax(logits)),
            "sigmoid" => Ok(logits.iter().map(|&value| sigmoid(value)).collect()),
            "sqrtsoftplus" => Ok(logits
                .iter()
                .map(|&value| stable_softplus(value).sqrt())
                .collect()),
            _ => Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                "unsupported DSV4 router scoring_func `{}`",
                self.scoring_func
            ))),
        }
    }

    pub fn moe_routes_from_scores(
        &self,
        layer_idx: usize,
        token_idx: usize,
        scores: &[f32],
        bias: Option<&[f32]>,
        hash_experts: Option<&[usize]>,
    ) -> Result<Vec<DeepSeekV4MoeRoute>> {
        if scores.len() != self.n_routed_experts {
            return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                "router scores length {} does not match n_routed_experts {}",
                scores.len(),
                self.n_routed_experts
            )));
        }
        if scores
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(DeepSeekConfigError::InvalidForwardBatch(
                "router scores must be finite and non-negative".to_string(),
            ));
        }

        let selected = match self.moe_routing_kind(layer_idx) {
            DeepSeekV4MoeRoutingKind::Hash => {
                let hash_experts = hash_experts.ok_or_else(|| {
                    DeepSeekConfigError::InvalidForwardBatch(format!(
                        "hash-routed layer {layer_idx} requires tid2eid experts"
                    ))
                })?;
                validate_expert_indices_in_range(hash_experts, self.n_routed_experts)?;
                if hash_experts.len() != self.num_experts_per_tok {
                    return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                        "hash expert count {} does not match num_experts_per_tok {}",
                        hash_experts.len(),
                        self.num_experts_per_tok
                    )));
                }
                hash_experts.to_vec()
            }
            DeepSeekV4MoeRoutingKind::LearnedBias => {
                let bias = bias.ok_or_else(|| {
                    DeepSeekConfigError::InvalidForwardBatch(format!(
                        "bias-routed layer {layer_idx} requires gate bias"
                    ))
                })?;
                if bias.len() != self.n_routed_experts {
                    return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                        "gate bias length {} does not match n_routed_experts {}",
                        bias.len(),
                        self.n_routed_experts
                    )));
                }
                if bias.iter().any(|value| !value.is_finite()) {
                    return Err(DeepSeekConfigError::InvalidForwardBatch(
                        "gate bias must be finite".to_string(),
                    ));
                }
                topk_indices_by_score(scores, bias, self.num_experts_per_tok)
            }
        };

        let selected_sum = selected
            .iter()
            .map(|&expert_idx| scores[expert_idx])
            .sum::<f32>();
        let normalize = self.scoring_func != "softmax";
        let denom = if normalize {
            selected_sum + 1.0e-9
        } else {
            1.0
        };
        Ok(selected
            .into_iter()
            .map(|expert_idx| DeepSeekV4MoeRoute {
                token_idx,
                expert_idx,
                weight: scores[expert_idx] / denom * self.routed_scaling_factor,
            })
            .collect())
    }
}

fn normalize_rope_parameters_aliases(value: &mut serde_json::Value) {
    for key in ["rope_parameters", "rope_scaling"] {
        let Some(rope) = value
            .get_mut(key)
            .and_then(serde_json::Value::as_object_mut)
        else {
            continue;
        };
        if rope.contains_key("rope_type") {
            rope.remove("type");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeepSeekV4AttentionMode {
    SlidingWindow,
    CompressedSparse,
    HybridCompressed,
    /// GLM DSA: Lightning Indexer over the full latent KV, no compressor.
    SparseIndexed,
}

impl DeepSeekV4AttentionMode {
    pub fn from_compress_ratio(compress_ratio: usize) -> Self {
        match compress_ratio {
            0 => Self::SlidingWindow,
            1..=15 => Self::CompressedSparse,
            _ => Self::HybridCompressed,
        }
    }

    pub fn has_compressor(self) -> bool {
        matches!(self, Self::CompressedSparse | Self::HybridCompressed)
    }

    pub fn has_indexer(self) -> bool {
        matches!(self, Self::CompressedSparse | Self::SparseIndexed)
    }

    pub fn flashmla_mode_int(self) -> i32 {
        match self {
            Self::CompressedSparse | Self::SparseIndexed => 1,
            Self::SlidingWindow | Self::HybridCompressed => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepSeekV4AttentionLayerPlan {
    pub layer_idx: usize,
    pub compress_ratio: usize,
    pub mode: DeepSeekV4AttentionMode,
    pub hash_routing: bool,
    pub has_compressor: bool,
    pub has_indexer: bool,
    pub sliding_window: usize,
    pub index_topk: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepSeekV4CompressorShape {
    pub compress_ratio: usize,
    pub overlap: bool,
    pub wkv_rows: usize,
    pub wkv_cols: usize,
    pub wgate_rows: usize,
    pub wgate_cols: usize,
    pub ape_rows: usize,
    pub ape_cols: usize,
    pub norm_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepSeekV4IndexerShape {
    pub compress_ratio: usize,
    pub wq_b_rows: usize,
    pub wq_b_cols: usize,
    pub weights_proj_rows: usize,
    pub weights_proj_cols: usize,
    pub key_heads: usize,
    pub key_head_dim: usize,
    pub topk: usize,
    pub compressor: Option<DeepSeekV4CompressorShape>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeepSeekV4MoeRoutingKind {
    Hash,
    LearnedBias,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4MoeRoute {
    pub token_idx: usize,
    pub expert_idx: usize,
    pub weight: f32,
}

fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut denom = 0.0_f32;
    let exp = logits
        .iter()
        .map(|&value| {
            let value = (value - max).exp();
            denom += value;
            value
        })
        .collect::<Vec<_>>();
    exp.into_iter().map(|value| value / denom).collect()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn stable_softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn validate_expert_indices_in_range(indices: &[usize], n_routed_experts: usize) -> Result<()> {
    for &expert_idx in indices {
        if expert_idx >= n_routed_experts {
            return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                "expert {expert_idx} out of range for n_routed_experts {n_routed_experts}"
            )));
        }
    }
    Ok(())
}

fn topk_indices_by_score(scores: &[f32], bias: &[f32], k: usize) -> Vec<usize> {
    let mut indices = (0..scores.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| {
        let score_b = scores[b] + bias[b];
        let score_a = scores[a] + bias[a];
        score_b.total_cmp(&score_a).then_with(|| a.cmp(&b))
    });
    indices.truncate(k);
    indices
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4TensorNames {
    pub dialect: TensorDialect,
}

impl DeepSeekV4TensorNames {
    pub fn embed_tokens(&self) -> &'static str {
        match self.dialect {
            TensorDialect::Dsv4 => "embed.weight",
            TensorDialect::Glm => "model.embed_tokens.weight",
        }
    }

    pub fn norm(&self) -> &'static str {
        match self.dialect {
            TensorDialect::Dsv4 => "norm.weight",
            TensorDialect::Glm => "model.norm.weight",
        }
    }

    pub fn lm_head(&self) -> &'static str {
        match self.dialect {
            TensorDialect::Dsv4 => "head.weight",
            TensorDialect::Glm => "lm_head.weight",
        }
    }

    pub fn head_hc(&self) -> DeepSeekV4HyperConnectionTensorNames {
        DeepSeekV4HyperConnectionTensorNames::new("hc_head")
    }

    pub fn layer(
        &self,
        layer_idx: usize,
        compress_ratio: usize,
        hash_routing: bool,
        include_shared_experts: bool,
        dense_mlp: bool,
        has_indexer: bool,
    ) -> DeepSeekV4LayerTensorNames {
        let prefix = match self.dialect {
            TensorDialect::Dsv4 => format!("layers.{layer_idx}"),
            TensorDialect::Glm => format!("model.layers.{layer_idx}"),
        };
        DeepSeekV4LayerTensorNames::new(
            prefix,
            compress_ratio,
            hash_routing,
            include_shared_experts,
            self.dialect,
            dense_mlp,
            has_indexer,
        )
    }

    pub fn mtp(&self, mtp_idx: usize) -> DeepSeekV4MtpTensorNames {
        DeepSeekV4MtpTensorNames::new(format!("mtp.{mtp_idx}"), true)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4HyperConnectionTensorNames {
    pub base: String,
    pub mix_fn: String,
    pub scale: String,
}

impl DeepSeekV4HyperConnectionTensorNames {
    fn new(prefix: &str) -> Self {
        Self {
            base: format!("{prefix}_base"),
            mix_fn: format!("{prefix}_fn"),
            scale: format!("{prefix}_scale"),
        }
    }

    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        (name == self.base || name == self.mix_fn || name == self.scale)
            .then_some(Shard::Replicated)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4CompressorTensorNames {
    pub prefix: String,
    pub wkv: String,
    pub wgate: String,
    pub ape: String,
    pub norm: String,
}

impl DeepSeekV4CompressorTensorNames {
    fn new(prefix: String) -> Self {
        Self {
            wkv: format!("{prefix}.wkv.weight"),
            wgate: format!("{prefix}.wgate.weight"),
            ape: format!("{prefix}.ape"),
            norm: format!("{prefix}.norm.weight"),
            prefix,
        }
    }

    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        match name {
            n if n == self.wkv || n == self.wgate => Some(Shard::Column { dim: 0 }),
            n if n == self.ape || n == self.norm => Some(Shard::Replicated),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4IndexerTensorNames {
    pub prefix: String,
    pub wq_b: String,
    pub weights_proj: String,
    /// `None` ⇒ GLM SparseIndexed (indexer over full latent, no key compressor).
    pub compressor: Option<DeepSeekV4CompressorTensorNames>,
    /// GLM only: indexer key projection `wk` `[index_n_heads*index_head_dim, hidden]`
    /// (DSv4 derives indexer keys through the compressor instead). `None` ⇒ DSv4.
    pub wk: Option<String>,
    /// GLM only: indexer key RMSNorm (`weight` + `bias`). `None` ⇒ DSv4.
    pub k_norm: Option<String>,
    pub k_norm_bias: Option<String>,
}

impl DeepSeekV4IndexerTensorNames {
    fn new(prefix: String, has_compressor: bool, dialect: TensorDialect) -> Self {
        let (wk, k_norm, k_norm_bias) = match dialect {
            TensorDialect::Dsv4 => (None, None, None),
            TensorDialect::Glm => (
                Some(format!("{prefix}.wk.weight")),
                Some(format!("{prefix}.k_norm.weight")),
                Some(format!("{prefix}.k_norm.bias")),
            ),
        };
        Self {
            wq_b: format!("{prefix}.wq_b.weight"),
            weights_proj: format!("{prefix}.weights_proj.weight"),
            compressor: has_compressor
                .then(|| DeepSeekV4CompressorTensorNames::new(format!("{prefix}.compressor"))),
            wk,
            k_norm,
            k_norm_bias,
            prefix,
        }
    }

    pub fn shard_for(
        &self,
        config: &DeepSeekV4Config,
        name: &str,
        tensor_parallel_size: usize,
    ) -> Option<Shard> {
        if name == self.wq_b || name == self.weights_proj {
            return Some(
                if config.index_n_heads.is_multiple_of(tensor_parallel_size) {
                    Shard::Column { dim: 0 }
                } else {
                    Shard::Replicated
                },
            );
        }
        if self.wk.as_deref() == Some(name) {
            return Some(
                if config.index_n_heads.is_multiple_of(tensor_parallel_size) {
                    Shard::Column { dim: 0 }
                } else {
                    Shard::Replicated
                },
            );
        }
        if self.k_norm.as_deref() == Some(name) || self.k_norm_bias.as_deref() == Some(name) {
            return Some(Shard::Replicated);
        }
        self.compressor
            .as_ref()
            .and_then(|compressor| compressor.shard_for(name))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4AttentionTensorNames {
    pub prefix: String,
    pub wq_a: String,
    pub q_norm: String,
    pub wq_b: String,
    /// KV down-projection latent. DSv4 `wkv`; GLM `kv_a_proj_with_mqa`.
    pub wkv: String,
    pub kv_norm: String,
    pub wo_a: String,
    pub wo_b: String,
    pub attn_sink: String,
    /// GLM only: non-absorbed `kv_b_proj`
    /// `[num_heads*(qk_nope+v_head_dim), kv_lora_rank]`, split at load into
    /// `w_kc`/`w_vc`. `None` ⇒ DSv4 (pre-absorbed checkpoint, no `kv_b`).
    pub kv_b_proj: Option<String>,
    /// GLM only: plain output projection `[hidden, num_heads*v_head_dim]`
    /// (replaces the DSv4 `wo_a`/`wo_b` low-rank). `None` ⇒ DSv4.
    pub o_proj: Option<String>,
    pub compressor: Option<DeepSeekV4CompressorTensorNames>,
    pub indexer: Option<DeepSeekV4IndexerTensorNames>,
}

impl DeepSeekV4AttentionTensorNames {
    fn new(
        prefix: String,
        compress_ratio: usize,
        dialect: TensorDialect,
        has_indexer: bool,
    ) -> Self {
        match dialect {
            TensorDialect::Dsv4 => {
                let compressor = (compress_ratio > 0)
                    .then(|| DeepSeekV4CompressorTensorNames::new(format!("{prefix}.compressor")));
                let indexer = (compress_ratio > 0 && compress_ratio < 16)
                    // DSv4 CSA indexer always rides a key compressor.
                    .then(|| {
                        DeepSeekV4IndexerTensorNames::new(
                            format!("{prefix}.indexer"),
                            true,
                            TensorDialect::Dsv4,
                        )
                    });
                Self {
                    wq_a: format!("{prefix}.wq_a.weight"),
                    q_norm: format!("{prefix}.q_norm.weight"),
                    wq_b: format!("{prefix}.wq_b.weight"),
                    wkv: format!("{prefix}.wkv.weight"),
                    kv_norm: format!("{prefix}.kv_norm.weight"),
                    wo_a: format!("{prefix}.wo_a.weight"),
                    wo_b: format!("{prefix}.wo_b.weight"),
                    attn_sink: format!("{prefix}.attn_sink"),
                    kv_b_proj: None,
                    o_proj: None,
                    compressor,
                    indexer,
                    prefix,
                }
            }
            TensorDialect::Glm => {
                // GLM ships a standard non-absorbed MLA: kv_a_proj_with_mqa (down)
                // + kv_b_proj (up, split into w_kc/w_vc at load) and a plain o_proj
                // — no wo_a/wo_b low-rank, no attn_sink. The indexer (DSA Lightning
                // Indexer) is present only on "full" layers; it has its own wk key
                // projection (no key compressor).
                let indexer = has_indexer.then(|| {
                    DeepSeekV4IndexerTensorNames::new(
                        format!("{prefix}.indexer"),
                        false,
                        TensorDialect::Glm,
                    )
                });
                Self {
                    wq_a: format!("{prefix}.q_a_proj.weight"),
                    q_norm: format!("{prefix}.q_a_layernorm.weight"),
                    wq_b: format!("{prefix}.q_b_proj.weight"),
                    wkv: format!("{prefix}.kv_a_proj_with_mqa.weight"),
                    kv_norm: format!("{prefix}.kv_a_layernorm.weight"),
                    // Plain-o GLM has no wo_a/wo_b; keep DSv4-shaped placeholders
                    // unused (loader gates on `o_proj`/`config.plain_o_proj`).
                    wo_a: format!("{prefix}.wo_a.weight"),
                    wo_b: format!("{prefix}.wo_b.weight"),
                    attn_sink: format!("{prefix}.attn_sink"),
                    kv_b_proj: Some(format!("{prefix}.kv_b_proj.weight")),
                    o_proj: Some(format!("{prefix}.o_proj.weight")),
                    compressor: None,
                    indexer,
                    prefix,
                }
            }
        }
    }

    pub fn shard_for(
        &self,
        config: &DeepSeekV4Config,
        name: &str,
        tensor_parallel_size: usize,
    ) -> Option<Shard> {
        if name == self.wq_a || name == self.q_norm || name == self.wkv || name == self.kv_norm {
            return Some(Shard::Replicated);
        }
        if name == self.wq_b {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.wo_a {
            return Some(if config.o_groups.is_multiple_of(tensor_parallel_size) {
                Shard::Column { dim: 0 }
            } else {
                Shard::Replicated
            });
        }
        if name == self.wo_b {
            return Some(Shard::Row { dim: 1 });
        }
        if name == self.attn_sink {
            return Some(Shard::Replicated);
        }
        // GLM non-absorbed MLA: kv_b_proj (up, per-head) is column-parallel over
        // heads; plain o_proj (over num_heads*v_head_dim) is row-parallel.
        if self.kv_b_proj.as_deref() == Some(name) {
            return Some(Shard::Column { dim: 0 });
        }
        if self.o_proj.as_deref() == Some(name) {
            return Some(Shard::Row { dim: 1 });
        }
        self.compressor
            .as_ref()
            .and_then(|compressor| compressor.shard_for(name))
            .or_else(|| {
                self.indexer
                    .as_ref()
                    .and_then(|indexer| indexer.shard_for(config, name, tensor_parallel_size))
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4ExpertTensorNames {
    pub prefix: String,
    pub w1: String,
    pub w2: String,
    pub w3: String,
}

impl DeepSeekV4ExpertTensorNames {
    fn new(prefix: String, dialect: TensorDialect) -> Self {
        // w1 = gate, w3 = up, w2 = down. DSv4 abbreviates; GLM ships HF names.
        let (w1, w2, w3) = match dialect {
            TensorDialect::Dsv4 => (
                format!("{prefix}.w1.weight"),
                format!("{prefix}.w2.weight"),
                format!("{prefix}.w3.weight"),
            ),
            TensorDialect::Glm => (
                format!("{prefix}.gate_proj.weight"),
                format!("{prefix}.down_proj.weight"),
                format!("{prefix}.up_proj.weight"),
            ),
        };
        Self { w1, w2, w3, prefix }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4MoeTensorNames {
    pub prefix: String,
    pub gate_weight: String,
    pub gate_bias: Option<String>,
    pub gate_tid2eid: Option<String>,
    pub experts_prefix: String,
    pub shared_experts: Option<DeepSeekV4ExpertTensorNames>,
    /// GLM `first_k_dense_replace` layers: a plain FFN (`gate_proj`/`up_proj`/
    /// `down_proj`) replaces the expert stack. `None` ⇒ MoE layer. DSv4 always
    /// `None` (all layers MoE).
    pub dense_mlp: Option<DeepSeekV4ExpertTensorNames>,
    dialect: TensorDialect,
}

impl DeepSeekV4MoeTensorNames {
    fn new(
        prefix: String,
        hash_routing: bool,
        include_shared_experts: bool,
        dialect: TensorDialect,
        dense_mlp: bool,
    ) -> Self {
        // GLM uses `gate.e_score_correction_bias` for the noaux_tc correction;
        // DSv4 uses `gate.bias`.
        let gate_bias = (!hash_routing).then(|| match dialect {
            TensorDialect::Dsv4 => format!("{prefix}.gate.bias"),
            TensorDialect::Glm => format!("{prefix}.gate.e_score_correction_bias"),
        });
        // A GLM dense-MLP layer has no router gate / experts; its FFN matrices
        // sit directly under `mlp.{gate,up,down}_proj`.
        let dense_mlp = (dense_mlp && dialect == TensorDialect::Glm)
            .then(|| DeepSeekV4ExpertTensorNames::new(prefix.clone(), TensorDialect::Glm));
        Self {
            gate_weight: format!("{prefix}.gate.weight"),
            gate_bias,
            gate_tid2eid: hash_routing.then(|| format!("{prefix}.gate.tid2eid")),
            experts_prefix: format!("{prefix}.experts"),
            shared_experts: include_shared_experts.then(|| {
                DeepSeekV4ExpertTensorNames::new(format!("{prefix}.shared_experts"), dialect)
            }),
            dense_mlp,
            dialect,
            prefix,
        }
    }

    pub fn expert(&self, expert_idx: usize) -> DeepSeekV4ExpertTensorNames {
        DeepSeekV4ExpertTensorNames::new(
            format!("{}.{}", self.experts_prefix, expert_idx),
            self.dialect,
        )
    }

    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.gate_weight
            || self.gate_bias.as_ref().is_some_and(|bias| name == bias)
            || self
                .gate_tid2eid
                .as_ref()
                .is_some_and(|table| name == table)
        {
            return Some(Shard::Replicated);
        }
        if name.starts_with(&self.experts_prefix) {
            return Some(Shard::ExpertParallel { dim: 0 });
        }
        if let Some(shared) = &self.shared_experts {
            if name == shared.w1 || name == shared.w3 {
                return Some(Shard::Column { dim: 0 });
            }
            if name == shared.w2 {
                return Some(Shard::Row { dim: 1 });
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4LayerTensorNames {
    pub prefix: String,
    pub attn_norm: String,
    pub ffn_norm: String,
    pub hc_attn: DeepSeekV4HyperConnectionTensorNames,
    pub hc_ffn: DeepSeekV4HyperConnectionTensorNames,
    pub attn: DeepSeekV4AttentionTensorNames,
    pub ffn: DeepSeekV4MoeTensorNames,
}

impl DeepSeekV4LayerTensorNames {
    #[allow(clippy::too_many_arguments)]
    fn new(
        prefix: String,
        compress_ratio: usize,
        hash_routing: bool,
        include_shared_experts: bool,
        dialect: TensorDialect,
        dense_mlp: bool,
        has_indexer: bool,
    ) -> Self {
        let (attn_norm, ffn_norm, attn_prefix, ffn_prefix) = match dialect {
            TensorDialect::Dsv4 => (
                format!("{prefix}.attn_norm.weight"),
                format!("{prefix}.ffn_norm.weight"),
                format!("{prefix}.attn"),
                format!("{prefix}.ffn"),
            ),
            TensorDialect::Glm => (
                format!("{prefix}.input_layernorm.weight"),
                format!("{prefix}.post_attention_layernorm.weight"),
                format!("{prefix}.self_attn"),
                format!("{prefix}.mlp"),
            ),
        };
        Self {
            attn_norm,
            ffn_norm,
            hc_attn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_attn")),
            hc_ffn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_ffn")),
            attn: DeepSeekV4AttentionTensorNames::new(
                attn_prefix,
                compress_ratio,
                dialect,
                has_indexer,
            ),
            ffn: DeepSeekV4MoeTensorNames::new(
                ffn_prefix,
                hash_routing,
                include_shared_experts,
                dialect,
                dense_mlp,
            ),
            prefix,
        }
    }

    pub fn shard_for(
        &self,
        config: &DeepSeekV4Config,
        name: &str,
        tensor_parallel_size: usize,
    ) -> Option<Shard> {
        if name == self.attn_norm || name == self.ffn_norm {
            return Some(Shard::Replicated);
        }
        self.hc_attn
            .shard_for(name)
            .or_else(|| self.hc_ffn.shard_for(name))
            .or_else(|| self.attn.shard_for(config, name, tensor_parallel_size))
            .or_else(|| self.ffn.shard_for(name))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4MtpTensorNames {
    pub prefix: String,
    pub enorm: String,
    pub hnorm: String,
    pub e_proj: String,
    pub h_proj: String,
    pub attn_norm: String,
    pub ffn_norm: String,
    pub norm: String,
    pub hc_attn: DeepSeekV4HyperConnectionTensorNames,
    pub hc_ffn: DeepSeekV4HyperConnectionTensorNames,
    pub hc_head: DeepSeekV4HyperConnectionTensorNames,
    pub attn: DeepSeekV4AttentionTensorNames,
    pub ffn: DeepSeekV4MoeTensorNames,
}

impl DeepSeekV4MtpTensorNames {
    fn new(prefix: String, include_shared_experts: bool) -> Self {
        Self {
            enorm: format!("{prefix}.enorm.weight"),
            hnorm: format!("{prefix}.hnorm.weight"),
            e_proj: format!("{prefix}.e_proj.weight"),
            h_proj: format!("{prefix}.h_proj.weight"),
            attn_norm: format!("{prefix}.attn_norm.weight"),
            ffn_norm: format!("{prefix}.ffn_norm.weight"),
            norm: format!("{prefix}.norm.weight"),
            hc_attn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_attn")),
            hc_ffn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_ffn")),
            hc_head: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_head")),
            // MTP is DSv4-only (GLM ships num_nextn_predict_layers=0): DSv4 dialect,
            // no indexer (compress_ratio=0), no dense MLP.
            attn: DeepSeekV4AttentionTensorNames::new(
                format!("{prefix}.attn"),
                0,
                TensorDialect::Dsv4,
                false,
            ),
            ffn: DeepSeekV4MoeTensorNames::new(
                format!("{prefix}.ffn"),
                false,
                include_shared_experts,
                TensorDialect::Dsv4,
                false,
            ),
            prefix,
        }
    }

    pub fn shard_for(
        &self,
        config: &DeepSeekV4Config,
        name: &str,
        tensor_parallel_size: usize,
    ) -> Option<Shard> {
        if name == self.enorm
            || name == self.hnorm
            || name == self.attn_norm
            || name == self.ffn_norm
            || name == self.norm
        {
            return Some(Shard::Replicated);
        }
        if name == self.e_proj || name == self.h_proj {
            return Some(Shard::Replicated);
        }
        self.hc_attn
            .shard_for(name)
            .or_else(|| self.hc_ffn.shard_for(name))
            .or_else(|| self.hc_head.shard_for(name))
            .or_else(|| self.attn.shard_for(config, name, tensor_parallel_size))
            .or_else(|| self.ffn.shard_for(name))
    }
}

/// One stage of the DSpark spec-decode draft — a full DSv4 transformer block
/// (`attn` MLA + `ffn` 256-expert MoE + `attn_norm`/`ffn_norm` + `hc_attn`/
/// `hc_ffn`), same body as native MTP. The draft is 3 such blocks stacked
/// (`mtp.0` → `mtp.1` → `mtp.2`); stage-only extras carry the entry/exit heads:
///
/// - `mtp.0` (entry): `main_proj` (fp8-block 3-tap fusion) + `main_norm`.
/// - `mtp.{n-1}` (exit): `hc_head`, final `norm`, the low-rank Markov
///   token-transition head (`markov_w1`/`markov_w2`), and the scalar
///   `confidence_proj`.
/// - middle stages: bare block, all extras `None`.
///
/// The output head is tied to `embed.weight` (no `lm_head` in the checkpoint).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4DsparkTensorNames {
    pub prefix: String,
    pub attn_norm: String,
    pub ffn_norm: String,
    pub hc_attn: DeepSeekV4HyperConnectionTensorNames,
    pub hc_ffn: DeepSeekV4HyperConnectionTensorNames,
    pub attn: DeepSeekV4AttentionTensorNames,
    pub ffn: DeepSeekV4MoeTensorNames,
    /// `Some` iff entry stage (`stage_idx == 0`).
    pub main_proj: Option<String>,
    pub main_norm: Option<String>,
    /// `Some` iff exit stage (`stage_idx == num_stages - 1`).
    pub hc_head: Option<DeepSeekV4HyperConnectionTensorNames>,
    pub norm: Option<String>,
    pub markov_w1: Option<String>,
    pub markov_w2: Option<String>,
    pub confidence_proj: Option<String>,
}

impl DeepSeekV4DsparkTensorNames {
    fn new(
        prefix: String,
        stage_idx: usize,
        num_stages: usize,
        include_shared_experts: bool,
    ) -> Self {
        let is_entry = stage_idx == 0;
        let is_exit = stage_idx + 1 == num_stages;
        Self {
            attn_norm: format!("{prefix}.attn_norm.weight"),
            ffn_norm: format!("{prefix}.ffn_norm.weight"),
            hc_attn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_attn")),
            hc_ffn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_ffn")),
            // DSpark is DSv4-only: DSv4 dialect, no indexer (compress_ratio=0),
            // no dense MLP — mirrors the native MTP body exactly.
            attn: DeepSeekV4AttentionTensorNames::new(
                format!("{prefix}.attn"),
                0,
                TensorDialect::Dsv4,
                false,
            ),
            ffn: DeepSeekV4MoeTensorNames::new(
                format!("{prefix}.ffn"),
                false,
                include_shared_experts,
                TensorDialect::Dsv4,
                false,
            ),
            main_proj: is_entry.then(|| format!("{prefix}.main_proj.weight")),
            main_norm: is_entry.then(|| format!("{prefix}.main_norm.weight")),
            hc_head: is_exit
                .then(|| DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_head"))),
            norm: is_exit.then(|| format!("{prefix}.norm.weight")),
            markov_w1: is_exit.then(|| format!("{prefix}.markov_head.markov_w1.weight")),
            markov_w2: is_exit.then(|| format!("{prefix}.markov_head.markov_w2.weight")),
            confidence_proj: is_exit.then(|| format!("{prefix}.confidence_head.proj.weight")),
            prefix,
        }
    }

    pub fn shard_for(
        &self,
        config: &DeepSeekV4Config,
        name: &str,
        tensor_parallel_size: usize,
    ) -> Option<Shard> {
        if name == self.attn_norm || name == self.ffn_norm {
            return Some(Shard::Replicated);
        }
        // Scaffolding: the fp8-block 3-tap fusion, the final norm, and the small
        // Markov/confidence heads are replicated (no TP shard); the forward
        // tranche revisits markov/lm_head vocab-sharding. Guard the position-
        // dependent `Option` fields before comparing.
        for extra in [
            self.main_proj.as_deref(),
            self.main_norm.as_deref(),
            self.norm.as_deref(),
            self.markov_w1.as_deref(),
            self.markov_w2.as_deref(),
            self.confidence_proj.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if name == extra {
                return Some(Shard::Replicated);
            }
        }
        self.hc_attn
            .shard_for(name)
            .or_else(|| self.hc_ffn.shard_for(name))
            .or_else(|| self.hc_head.as_ref().and_then(|hc| hc.shard_for(name)))
            .or_else(|| self.attn.shard_for(config, name, tensor_parallel_size))
            .or_else(|| self.ffn.shard_for(name))
    }
}
