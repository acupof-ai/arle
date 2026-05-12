use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{DeepSeekConfigError, Result, Shard};

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
        let config: Self = serde_json::from_value(value.clone())?;
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
        if self.compress_ratios.len() != self.num_hidden_layers {
            return Err(DeepSeekConfigError::InvalidConfig(
                "compress_ratios length must match num_hidden_layers",
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
        DeepSeekV4TensorNames
    }

    pub fn layer_tensor_names(&self, layer_idx: usize) -> DeepSeekV4LayerTensorNames {
        let compress_ratio = self.compress_ratios[layer_idx];
        self.tensor_names().layer(
            layer_idx,
            compress_ratio,
            layer_idx < self.num_hash_layers,
            self.n_shared_experts > 0,
        )
    }

    pub fn mtp_tensor_names(&self, mtp_idx: usize) -> DeepSeekV4MtpTensorNames {
        DeepSeekV4MtpTensorNames::new(format!("mtp.{mtp_idx}"), self.n_shared_experts > 0)
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
        let mode = self.attention_mode_for_compress_ratio(compress_ratio);
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

    pub fn attention_operator_summary(&self) -> DeepSeekV4AttentionOperatorSummary {
        let mut summary = DeepSeekV4AttentionOperatorSummary::default();
        for layer_idx in 0..self.num_hidden_layers {
            let plan = self
                .attention_layer_plan(layer_idx)
                .expect("compress_ratios length validated");
            match plan.mode {
                DeepSeekV4AttentionMode::SlidingWindow => summary.sliding_window_layers += 1,
                DeepSeekV4AttentionMode::CompressedSparse => summary.csa_layers += 1,
                DeepSeekV4AttentionMode::HybridCompressed => summary.hca_layers += 1,
            }
            if plan.hash_routing {
                summary.hash_routed_moe_layers += 1;
            } else {
                summary.bias_routed_moe_layers += 1;
            }
        }
        summary
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
        (self.attention_mode_for_compress_ratio(compress_ratio)
            == DeepSeekV4AttentionMode::CompressedSparse)
            .then(|| DeepSeekV4IndexerShape {
                compress_ratio,
                wq_b_rows: self.index_n_heads * self.index_head_dim,
                wq_b_cols: self.q_lora_rank,
                weights_proj_rows: self.index_n_heads,
                weights_proj_cols: self.hidden_size,
                key_head_dim: self.index_head_dim,
                key_heads: self.index_n_heads,
                topk: self.index_topk,
                compressor: self
                    .compressor_shape(compress_ratio)
                    .expect("CSA compress_ratio must have compressor shape"),
            })
    }

    pub fn output_projection_shape(&self) -> DeepSeekV4OutputProjectionShape {
        let heads_per_group = self.num_attention_heads / self.o_groups;
        DeepSeekV4OutputProjectionShape {
            heads_per_group,
            wo_a_rows: self.o_groups * self.o_lora_rank,
            wo_a_cols: heads_per_group * self.head_dim,
            wo_b_rows: self.hidden_size,
            wo_b_cols: self.o_groups * self.o_lora_rank,
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeepSeekV4AttentionMode {
    SlidingWindow,
    CompressedSparse,
    HybridCompressed,
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
        self == Self::CompressedSparse
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepSeekV4AttentionOperatorSummary {
    pub sliding_window_layers: usize,
    pub csa_layers: usize,
    pub hca_layers: usize,
    pub hash_routed_moe_layers: usize,
    pub bias_routed_moe_layers: usize,
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
    pub compressor: DeepSeekV4CompressorShape,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepSeekV4OutputProjectionShape {
    pub heads_per_group: usize,
    pub wo_a_rows: usize,
    pub wo_a_cols: usize,
    pub wo_b_rows: usize,
    pub wo_b_cols: usize,
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
pub struct DeepSeekV4TensorNames;

impl DeepSeekV4TensorNames {
    pub fn embed_tokens(&self) -> &'static str {
        "embed.weight"
    }

    pub fn norm(&self) -> &'static str {
        "norm.weight"
    }

    pub fn lm_head(&self) -> &'static str {
        "head.weight"
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
    ) -> DeepSeekV4LayerTensorNames {
        DeepSeekV4LayerTensorNames::new(
            format!("layers.{layer_idx}"),
            compress_ratio,
            hash_routing,
            include_shared_experts,
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
    pub compressor: DeepSeekV4CompressorTensorNames,
}

impl DeepSeekV4IndexerTensorNames {
    fn new(prefix: String) -> Self {
        Self {
            wq_b: format!("{prefix}.wq_b.weight"),
            weights_proj: format!("{prefix}.weights_proj.weight"),
            compressor: DeepSeekV4CompressorTensorNames::new(format!("{prefix}.compressor")),
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
        self.compressor.shard_for(name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4AttentionTensorNames {
    pub prefix: String,
    pub wq_a: String,
    pub q_norm: String,
    pub wq_b: String,
    pub wkv: String,
    pub kv_norm: String,
    pub wo_a: String,
    pub wo_b: String,
    pub attn_sink: String,
    pub compressor: Option<DeepSeekV4CompressorTensorNames>,
    pub indexer: Option<DeepSeekV4IndexerTensorNames>,
}

impl DeepSeekV4AttentionTensorNames {
    fn new(prefix: String, compress_ratio: usize) -> Self {
        let compressor = (compress_ratio > 0)
            .then(|| DeepSeekV4CompressorTensorNames::new(format!("{prefix}.compressor")));
        let indexer = (compress_ratio > 0 && compress_ratio < 16)
            .then(|| DeepSeekV4IndexerTensorNames::new(format!("{prefix}.indexer")));
        Self {
            wq_a: format!("{prefix}.wq_a.weight"),
            q_norm: format!("{prefix}.q_norm.weight"),
            wq_b: format!("{prefix}.wq_b.weight"),
            wkv: format!("{prefix}.wkv.weight"),
            kv_norm: format!("{prefix}.kv_norm.weight"),
            wo_a: format!("{prefix}.wo_a.weight"),
            wo_b: format!("{prefix}.wo_b.weight"),
            attn_sink: format!("{prefix}.attn_sink"),
            compressor,
            indexer,
            prefix,
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
    fn new(prefix: String) -> Self {
        Self {
            w1: format!("{prefix}.w1.weight"),
            w2: format!("{prefix}.w2.weight"),
            w3: format!("{prefix}.w3.weight"),
            prefix,
        }
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
}

impl DeepSeekV4MoeTensorNames {
    fn new(prefix: String, hash_routing: bool, include_shared_experts: bool) -> Self {
        Self {
            gate_weight: format!("{prefix}.gate.weight"),
            gate_bias: (!hash_routing).then(|| format!("{prefix}.gate.bias")),
            gate_tid2eid: hash_routing.then(|| format!("{prefix}.gate.tid2eid")),
            experts_prefix: format!("{prefix}.experts"),
            shared_experts: include_shared_experts
                .then(|| DeepSeekV4ExpertTensorNames::new(format!("{prefix}.shared_experts"))),
            prefix,
        }
    }

    pub fn expert(&self, expert_idx: usize) -> DeepSeekV4ExpertTensorNames {
        DeepSeekV4ExpertTensorNames::new(format!("{}.{}", self.experts_prefix, expert_idx))
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
    fn new(
        prefix: String,
        compress_ratio: usize,
        hash_routing: bool,
        include_shared_experts: bool,
    ) -> Self {
        Self {
            attn_norm: format!("{prefix}.attn_norm.weight"),
            ffn_norm: format!("{prefix}.ffn_norm.weight"),
            hc_attn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_attn")),
            hc_ffn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_ffn")),
            attn: DeepSeekV4AttentionTensorNames::new(format!("{prefix}.attn"), compress_ratio),
            ffn: DeepSeekV4MoeTensorNames::new(
                format!("{prefix}.ffn"),
                hash_routing,
                include_shared_experts,
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
            attn: DeepSeekV4AttentionTensorNames::new(format!("{prefix}.attn"), 0),
            ffn: DeepSeekV4MoeTensorNames::new(
                format!("{prefix}.ffn"),
                false,
                include_shared_experts,
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
            return Some(Shard::Column { dim: 0 });
        }
        self.hc_attn
            .shard_for(name)
            .or_else(|| self.hc_ffn.shard_for(name))
            .or_else(|| self.hc_head.shard_for(name))
            .or_else(|| self.attn.shard_for(config, name, tensor_parallel_size))
            .or_else(|| self.ffn.shard_for(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replica_config_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../infer/models/dsv4-mini-1B-init/config.json")
    }

    fn replica_config() -> DeepSeekV4Config {
        DeepSeekV4Config::from_json_file(replica_config_path()).unwrap()
    }

    #[test]
    fn parses_hf_flash_alias_config_fields() {
        let cfg = DeepSeekV4Config::from_json_str(
            r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "torch_dtype": "bfloat16",
            "vocab_size": 129280,
            "hidden_size": 4096,
            "num_hidden_layers": 2,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "qk_rope_head_dim": 64,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 2048,
            "routed_scaling_factor": 1.5,
            "norm_topk_prob": true,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
            "num_hash_layers": 1,
            "sliding_window": 128,
            "compress_ratios": [0, 4],
            "compress_rope_theta": 160000.0,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1.0e-6,
            "num_nextn_predict_layers": 1,
            "max_position_embeddings": 1048576,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": 16.0,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32.0,
                "beta_slow": 1.0
            },
            "rms_norm_eps": 1.0e-6,
            "initializer_range": 0.02,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 1
        }"#,
        )
        .unwrap();

        assert_eq!(cfg.dtype, "bfloat16");
        assert_eq!(cfg.rope_parameters.rope_type, "yarn");
        assert_eq!(cfg.rope_parameters.factor, 16.0);
        assert_eq!(cfg.pad_token_id, None);
    }

    #[test]
    fn parses_hf_replica_config() {
        let cfg = replica_config();
        assert_eq!(cfg.model_type, "deepseek_v4");
        assert_eq!(cfg.dtype, "bfloat16");
        assert_eq!(cfg.vocab_size, 129_280);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.swiglu_limit, 10.0);
        assert_eq!(cfg.q_lora_rank, 384);
        assert_eq!(cfg.o_lora_rank, 384);
        assert_eq!(cfg.o_groups, 4);
        assert_eq!(cfg.scoring_func, "sqrtsoftplus");
        assert_eq!(cfg.topk_method, "noaux_tc");
        assert_eq!(cfg.index_n_heads, 8);
        assert_eq!(cfg.index_head_dim, 64);
        assert_eq!(cfg.index_topk, 128);
        assert_eq!(cfg.num_hash_layers, 2);
        assert_eq!(cfg.sliding_window, 64);
        assert_eq!(cfg.compress_ratios.len(), 24);
        assert_eq!(cfg.compress_ratios[2], 4);
        assert_eq!(cfg.compress_rope_theta, 160_000.0);
        assert_eq!(cfg.hc_mult, 4);
        assert_eq!(cfg.hc_sinkhorn_iters, 20);
        assert_eq!(cfg.hc_eps, 1.0e-6);
        assert_eq!(cfg.num_nextn_predict_layers, 1);
        assert_eq!(cfg.rope_parameters.rope_type, "yarn");
        assert_eq!(cfg.rope_parameters.factor, 16.0);
        assert!(!cfg.attention_bias);
        assert_eq!(cfg.attention_dropout, 0.0);
        assert_eq!(cfg.pad_token_id, None);
    }

    #[test]
    fn tensor_names_match_hf_replica_layout() {
        let cfg = replica_config();
        let top = cfg.tensor_names();
        assert_eq!(top.embed_tokens(), "embed.weight");
        assert_eq!(top.lm_head(), "head.weight");
        assert_eq!(top.head_hc().mix_fn, "hc_head_fn");

        let sw_hash = cfg.layer_tensor_names(0);
        assert_eq!(sw_hash.attn.wq_a, "layers.0.attn.wq_a.weight");
        assert_eq!(sw_hash.attn.wkv, "layers.0.attn.wkv.weight");
        assert_eq!(sw_hash.hc_attn.mix_fn, "layers.0.hc_attn_fn");
        assert!(sw_hash.attn.compressor.is_none());
        assert_eq!(
            sw_hash.ffn.gate_tid2eid.as_deref(),
            Some("layers.0.ffn.gate.tid2eid")
        );
        assert!(sw_hash.ffn.gate_bias.is_none());

        let csa = cfg.layer_tensor_names(2);
        assert_eq!(
            csa.attn.compressor.as_ref().unwrap().wgate,
            "layers.2.attn.compressor.wgate.weight"
        );
        assert_eq!(
            csa.attn.indexer.as_ref().unwrap().compressor.ape,
            "layers.2.attn.indexer.compressor.ape"
        );
        assert_eq!(csa.ffn.gate_bias.as_deref(), Some("layers.2.ffn.gate.bias"));
        assert_eq!(csa.ffn.expert(7).w2, "layers.2.ffn.experts.7.w2.weight");

        let hca = cfg.layer_tensor_names(3);
        assert!(hca.attn.compressor.is_some());
        assert!(hca.attn.indexer.is_none());

        let mtp = cfg.mtp_tensor_names(0);
        assert_eq!(mtp.enorm, "mtp.0.enorm.weight");
        assert_eq!(mtp.e_proj, "mtp.0.e_proj.weight");
        assert_eq!(mtp.attn.attn_sink, "mtp.0.attn.attn_sink");
        assert_eq!(mtp.hc_head.scale, "mtp.0.hc_head_scale");
        assert_eq!(
            mtp.ffn.shared_experts.as_ref().unwrap().w3,
            "mtp.0.ffn.shared_experts.w3.weight"
        );
    }

    #[test]
    fn shard_policy_handles_dsv4_shapes() {
        let cfg = replica_config();
        let csa = cfg.layer_tensor_names(2);
        assert_eq!(
            csa.shard_for(&cfg, &csa.attn.wkv, 4),
            Some(Shard::Replicated)
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.attn.wq_b, 4),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.attn.wo_a, 4),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.attn.wo_b, 4),
            Some(Shard::Row { dim: 1 })
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.hc_ffn.mix_fn, 4),
            Some(Shard::Replicated)
        );
        assert_eq!(
            csa.shard_for(&cfg, csa.attn.indexer.as_ref().unwrap().wq_b.as_str(), 4),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            csa.shard_for(&cfg, csa.attn.indexer.as_ref().unwrap().wq_b.as_str(), 3),
            Some(Shard::Replicated)
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.ffn.gate_weight, 4),
            Some(Shard::Replicated)
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.ffn.expert(0).w1, 4),
            Some(Shard::ExpertParallel { dim: 0 })
        );
        assert_eq!(
            cfg.shard_for_global_tensor("head.weight"),
            Some(Shard::VocabParallel { dim: 0 })
        );
    }
}
