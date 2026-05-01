use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DeepSeekConfigError {
    #[error("invalid deepseek config: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid deepseek MoE forward batch: {0}")]
    InvalidForwardBatch(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, DeepSeekConfigError>;

/// How a DeepSeek tensor should be partitioned across distributed ranks.
///
/// `dim` follows the HF safetensors `nn.Linear` layout: dim 0 is output
/// features, dim 1 is input features. `ExpertParallel` means the expert axis is
/// owned by EP/MoE-EP placement rather than tensor-parallel slicing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard {
    Replicated,
    Column { dim: usize },
    Row { dim: usize },
    MergedColumn { dim: usize },
    VocabParallel { dim: usize },
    ExpertParallel { dim: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekMlaTensorNames {
    pub attention_prefix: String,
    pub q_proj: String,
    pub q_a_proj: String,
    pub q_a_layernorm: String,
    pub q_b_proj: String,
    pub kv_a_proj_with_mqa: String,
    pub kv_a_layernorm: String,
    pub kv_b_proj: String,
    pub o_proj: String,
}

impl DeepSeekMlaTensorNames {
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.q_proj {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.q_a_proj || name == self.kv_a_proj_with_mqa {
            return Some(Shard::Replicated);
        }
        if name == self.q_b_proj || name == self.kv_b_proj {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.o_proj {
            return Some(Shard::Row { dim: 1 });
        }
        if name == self.q_a_layernorm || name == self.kv_a_layernorm {
            return Some(Shard::Replicated);
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekDenseMlpTensorNames {
    pub mlp_prefix: String,
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

impl DeepSeekDenseMlpTensorNames {
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.gate_proj || name == self.up_proj {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.down_proj {
            return Some(Shard::Row { dim: 1 });
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekExpertTensorNames {
    pub expert_prefix: String,
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

impl DeepSeekExpertTensorNames {
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.gate_proj || name == self.up_proj {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.down_proj {
            return Some(Shard::Row { dim: 1 });
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekMoeTensorNames {
    pub mlp_prefix: String,
    pub router: String,
    pub experts_prefix: String,
    pub shared_experts: Option<DeepSeekDenseMlpTensorNames>,
}

impl DeepSeekMoeTensorNames {
    pub fn expert(&self, expert_idx: usize) -> DeepSeekExpertTensorNames {
        let expert_prefix = format!("{}.experts.{expert_idx}", self.mlp_prefix);
        DeepSeekExpertTensorNames {
            expert_prefix: expert_prefix.clone(),
            gate_proj: format!("{expert_prefix}.gate_proj.weight"),
            up_proj: format!("{expert_prefix}.up_proj.weight"),
            down_proj: format!("{expert_prefix}.down_proj.weight"),
        }
    }

    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.router {
            return Some(Shard::Replicated);
        }
        if self
            .shared_experts
            .as_ref()
            .and_then(|shared| shared.shard_for(name))
            .is_some()
        {
            return self
                .shared_experts
                .as_ref()
                .and_then(|shared| shared.shard_for(name));
        }
        if name.starts_with(&self.experts_prefix) {
            if name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight") {
                return Some(Shard::Column { dim: 0 });
            }
            if name.ends_with(".down_proj.weight") {
                return Some(Shard::Row { dim: 1 });
            }
            return Some(Shard::ExpertParallel { dim: 0 });
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeepSeekMlpTensorNames {
    Dense(DeepSeekDenseMlpTensorNames),
    Moe(DeepSeekMoeTensorNames),
}

impl DeepSeekMlpTensorNames {
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        match self {
            Self::Dense(mlp) => mlp.shard_for(name),
            Self::Moe(moe) => moe.shard_for(name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekLayerTensorNames {
    pub layer_prefix: String,
    pub input_layernorm: String,
    pub post_attention_layernorm: String,
    pub attention: DeepSeekMlaTensorNames,
    pub mlp: DeepSeekMlpTensorNames,
}

impl DeepSeekLayerTensorNames {
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.input_layernorm || name == self.post_attention_layernorm {
            return Some(Shard::Replicated);
        }
        self.attention
            .shard_for(name)
            .or_else(|| self.mlp.shard_for(name))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekMtpTensorNames {
    pub mtp_prefix: String,
    pub embed_tokens: String,
    pub enorm: String,
    pub hnorm: String,
    pub eh_proj: String,
    pub layer: DeepSeekLayerTensorNames,
    pub shared_head_norm: String,
    pub lm_head: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExpertParallelConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub moe_intermediate_size: usize,
    pub n_shared_experts: usize,
    pub routed_scaling_factor: f32,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
}

impl ExpertParallelConfig {
    pub fn from_deepseek_config(config: &DeepSeekConfig) -> Self {
        Self {
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            first_k_dense_replace: config.first_k_dense_replace,
            moe_intermediate_size: config.moe_intermediate_size,
            n_shared_experts: config.n_shared_experts,
            routed_scaling_factor: config.routed_scaling_factor,
            n_group: config.n_group,
            topk_group: config.topk_group,
            norm_topk_prob: config.norm_topk_prob,
        }
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.is_moe() && layer_idx >= self.first_k_dense_replace
    }

    pub fn experts_per_rank(&self, world_size: usize) -> Option<usize> {
        (world_size > 0 && self.num_experts.is_multiple_of(world_size))
            .then_some(self.num_experts / world_size)
    }

    pub fn local_expert_range(
        &self,
        rank: usize,
        world_size: usize,
    ) -> Option<std::ops::Range<usize>> {
        let experts_per_rank = self.experts_per_rank(world_size)?;
        (rank < world_size).then_some(rank * experts_per_rank..(rank + 1) * experts_per_rank)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekMoeRoute {
    pub token_idx: usize,
    pub expert_idx: usize,
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekMoeForwardBatch {
    pub layer_idx: usize,
    pub token_count: usize,
    pub hidden_size: usize,
    pub routes: Vec<DeepSeekMoeRoute>,
    pub include_shared_experts: bool,
}

impl DeepSeekMoeForwardBatch {
    pub fn from_config(
        config: &DeepSeekConfig,
        layer_idx: usize,
        token_count: usize,
        routes: impl Into<Vec<DeepSeekMoeRoute>>,
    ) -> Self {
        Self {
            layer_idx,
            token_count,
            hidden_size: config.hidden_size,
            routes: routes.into(),
            include_shared_experts: config.n_shared_experts > 0,
        }
    }

    pub fn validate_against(&self, config: &DeepSeekConfig) -> Result<()> {
        let total_forward_layers = config.num_hidden_layers + config.num_nextn_predict_layers;
        if self.layer_idx >= total_forward_layers {
            return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                "layer {} out of range for total forward layers {}",
                self.layer_idx, total_forward_layers
            )));
        }
        if !config.is_moe_layer(self.layer_idx) {
            return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                "layer {} is not a MoE layer",
                self.layer_idx
            )));
        }
        if self.hidden_size != config.hidden_size {
            return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                "hidden_size {} does not match config hidden_size {}",
                self.hidden_size, config.hidden_size
            )));
        }
        let expected_shared_experts = config.n_shared_experts > 0;
        if self.include_shared_experts != expected_shared_experts {
            return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                "include_shared_experts {} does not match config shared expert requirement {}",
                self.include_shared_experts, expected_shared_experts
            )));
        }
        let mut routes_per_token = vec![0_usize; self.token_count];
        let mut experts_per_token = vec![Vec::<usize>::new(); self.token_count];
        for route in &self.routes {
            if route.token_idx >= self.token_count {
                return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                    "route token {} out of range for token_count {}",
                    route.token_idx, self.token_count
                )));
            }
            if route.expert_idx >= config.num_experts {
                return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                    "route expert {} out of range for num_experts {}",
                    route.expert_idx, config.num_experts
                )));
            }
            if !route.weight.is_finite() || route.weight < 0.0 {
                return Err(DeepSeekConfigError::InvalidForwardBatch(
                    "route weight must be finite and non-negative".to_string(),
                ));
            }
            if experts_per_token[route.token_idx].contains(&route.expert_idx) {
                return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                    "token {} routes to expert {} more than once",
                    route.token_idx, route.expert_idx
                )));
            }
            routes_per_token[route.token_idx] += 1;
            experts_per_token[route.token_idx].push(route.expert_idx);
        }
        for (token_idx, route_count) in routes_per_token.into_iter().enumerate() {
            if route_count != config.num_experts_per_tok {
                return Err(DeepSeekConfigError::InvalidForwardBatch(format!(
                    "token {token_idx} route count {route_count} does not match num_experts_per_tok {}",
                    config.num_experts_per_tok
                )));
            }
        }
        Ok(())
    }

    pub fn expert_inputs(&self) -> Vec<DeepSeekExpertForwardInput> {
        let mut inputs = Vec::new();
        for route in &self.routes {
            if let Some(input_idx) = inputs
                .iter()
                .position(|input: &DeepSeekExpertForwardInput| input.expert_idx == route.expert_idx)
            {
                inputs[input_idx].token_indices.push(route.token_idx);
                inputs[input_idx].route_weights.push(route.weight);
            } else {
                inputs.push(DeepSeekExpertForwardInput {
                    expert_idx: route.expert_idx,
                    token_indices: vec![route.token_idx],
                    route_weights: vec![route.weight],
                });
            }
        }
        inputs.sort_by_key(|input| input.expert_idx);
        inputs
    }

    pub fn ep_forward_plan(
        &self,
        config: &DeepSeekConfig,
        rank: usize,
        world_size: usize,
    ) -> Result<DeepSeekMoeForwardPlan> {
        self.validate_against(config)?;
        let ep = config.expert_parallel_config();
        let local_experts = ep.local_expert_range(rank, world_size).ok_or_else(|| {
            DeepSeekConfigError::InvalidForwardBatch(format!(
                "cannot place {} experts across world_size {} rank {}",
                ep.num_experts, world_size, rank
            ))
        })?;
        let local_expert_indices: Vec<_> = local_experts.collect();
        let local_inputs = self
            .expert_inputs()
            .into_iter()
            .filter(|input| local_expert_indices.contains(&input.expert_idx))
            .collect();
        Ok(DeepSeekMoeForwardPlan {
            layer_idx: self.layer_idx,
            rank,
            world_size,
            local_experts: local_expert_indices,
            local_inputs,
            include_shared_experts: self.include_shared_experts,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekExpertForwardInput {
    pub expert_idx: usize,
    pub token_indices: Vec<usize>,
    pub route_weights: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepSeekExpertForwardOutput {
    pub expert_idx: usize,
    pub token_indices: Vec<usize>,
    pub hidden_size: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekMoeForwardPlan {
    pub layer_idx: usize,
    pub rank: usize,
    pub world_size: usize,
    pub local_experts: Vec<usize>,
    pub local_inputs: Vec<DeepSeekExpertForwardInput>,
    pub include_shared_experts: bool,
}

#[derive(Debug, Deserialize)]
struct RawDeepSeekConfig {
    vocab_size: usize,
    hidden_size: usize,
    #[serde(default)]
    intermediate_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(alias = "num_kv_heads")]
    num_key_value_heads: usize,
    #[serde(default)]
    n_shared_experts: usize,
    #[serde(alias = "n_routed_experts")]
    num_experts: usize,
    #[serde(default)]
    routed_scaling_factor: f32,
    kv_lora_rank: usize,
    #[serde(default)]
    q_lora_rank: Option<usize>,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    qk_nope_head_dim: usize,
    #[serde(default)]
    n_group: usize,
    #[serde(default)]
    topk_group: usize,
    num_experts_per_tok: usize,
    first_k_dense_replace: usize,
    #[serde(default = "default_norm_topk_prob")]
    norm_topk_prob: bool,
    #[serde(default)]
    hidden_act: Option<String>,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default)]
    rope_interleave: bool,
    #[serde(default)]
    num_nextn_predict_layers: usize,
}

fn default_norm_topk_prob() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    10_000.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub moe_intermediate_size: usize,
    pub n_shared_experts: usize,
    pub routed_scaling_factor: f32,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub hidden_act: Option<String>,
    pub kv_lora_rank: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope_interleave: bool,
    pub num_nextn_predict_layers: usize,
}

impl DeepSeekConfig {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json_str(&content)
    }

    pub fn from_json_str(content: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(content)?;
        Self::from_json_value(&value)
    }

    pub fn from_json_value(value: &serde_json::Value) -> Result<Self> {
        let raw: RawDeepSeekConfig = serde_json::from_value(value.clone())?;
        let config = Self::from_raw(raw);
        config.validate()?;
        Ok(config)
    }

    fn from_raw(raw: RawDeepSeekConfig) -> Self {
        Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            max_position_embeddings: raw.max_position_embeddings,
            rope_theta: raw.rope_theta,
            rms_norm_eps: raw.rms_norm_eps,
            bos_token_id: raw.bos_token_id,
            eos_token_id: raw.eos_token_id,
            tie_word_embeddings: raw.tie_word_embeddings,
            num_experts: raw.num_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            first_k_dense_replace: raw.first_k_dense_replace,
            moe_intermediate_size: raw.moe_intermediate_size,
            n_shared_experts: raw.n_shared_experts,
            routed_scaling_factor: raw.routed_scaling_factor,
            n_group: raw.n_group,
            topk_group: raw.topk_group,
            norm_topk_prob: raw.norm_topk_prob,
            hidden_act: raw.hidden_act,
            kv_lora_rank: raw.kv_lora_rank,
            q_lora_rank: raw.q_lora_rank,
            qk_rope_head_dim: raw.qk_rope_head_dim,
            qk_nope_head_dim: raw.qk_nope_head_dim,
            v_head_dim: raw.v_head_dim,
            rope_interleave: raw.rope_interleave,
            num_nextn_predict_layers: raw.num_nextn_predict_layers,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "hidden size, layers, and attention heads must be non-zero",
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }
        if self.kv_lora_rank == 0
            || self.qk_rope_head_dim == 0
            || self.qk_nope_head_dim == 0
            || self.v_head_dim == 0
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "MLA rank and head dimensions must be non-zero",
            ));
        }
        if self.num_experts > 0 && self.num_experts_per_tok == 0 {
            return Err(DeepSeekConfigError::InvalidConfig(
                "num_experts_per_tok must be non-zero for MoE configs",
            ));
        }
        Ok(())
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.is_moe() && layer_idx >= self.first_k_dense_replace
    }

    pub fn expert_parallel_config(&self) -> ExpertParallelConfig {
        ExpertParallelConfig::from_deepseek_config(self)
    }

    pub fn has_mtp(&self) -> bool {
        self.num_nextn_predict_layers > 0
    }

    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.qk_head_dim()
    }

    pub fn kv_b_proj_dim(&self) -> usize {
        self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim)
    }

    pub fn embed_tokens_tensor_name(&self) -> &'static str {
        "model.embed_tokens.weight"
    }

    pub fn norm_tensor_name(&self) -> &'static str {
        "model.norm.weight"
    }

    pub fn lm_head_tensor_name(&self) -> &'static str {
        if self.tie_word_embeddings {
            self.embed_tokens_tensor_name()
        } else {
            "lm_head.weight"
        }
    }

    pub fn shard_for_global_tensor(&self, name: &str) -> Option<Shard> {
        match name {
            "model.embed_tokens.weight" => Some(Shard::VocabParallel { dim: 0 }),
            "lm_head.weight" => Some(Shard::VocabParallel { dim: 0 }),
            "model.norm.weight" => Some(Shard::Replicated),
            _ => None,
        }
    }

    pub fn layer_tensor_names(&self, layer_idx: usize) -> DeepSeekLayerTensorNames {
        let layer_prefix = format!("model.layers.{layer_idx}");
        let attention_prefix = format!("{layer_prefix}.self_attn");
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let attention = DeepSeekMlaTensorNames {
            attention_prefix: attention_prefix.clone(),
            q_proj: format!("{attention_prefix}.q_proj.weight"),
            q_a_proj: format!("{attention_prefix}.q_a_proj.weight"),
            q_a_layernorm: format!("{attention_prefix}.q_a_layernorm.weight"),
            q_b_proj: format!("{attention_prefix}.q_b_proj.weight"),
            kv_a_proj_with_mqa: format!("{attention_prefix}.kv_a_proj_with_mqa.weight"),
            kv_a_layernorm: format!("{attention_prefix}.kv_a_layernorm.weight"),
            kv_b_proj: format!("{attention_prefix}.kv_b_proj.weight"),
            o_proj: format!("{attention_prefix}.o_proj.weight"),
        };
        let mlp = if self.is_moe_layer(layer_idx) {
            DeepSeekMlpTensorNames::Moe(DeepSeekMoeTensorNames {
                mlp_prefix: mlp_prefix.clone(),
                router: format!("{mlp_prefix}.gate.weight"),
                experts_prefix: format!("{mlp_prefix}.experts"),
                shared_experts: (self.n_shared_experts > 0).then(|| DeepSeekDenseMlpTensorNames {
                    mlp_prefix: format!("{mlp_prefix}.shared_experts"),
                    gate_proj: format!("{mlp_prefix}.shared_experts.gate_proj.weight"),
                    up_proj: format!("{mlp_prefix}.shared_experts.up_proj.weight"),
                    down_proj: format!("{mlp_prefix}.shared_experts.down_proj.weight"),
                }),
            })
        } else {
            DeepSeekMlpTensorNames::Dense(DeepSeekDenseMlpTensorNames {
                mlp_prefix: mlp_prefix.clone(),
                gate_proj: format!("{mlp_prefix}.gate_proj.weight"),
                up_proj: format!("{mlp_prefix}.up_proj.weight"),
                down_proj: format!("{mlp_prefix}.down_proj.weight"),
            })
        };
        DeepSeekLayerTensorNames {
            layer_prefix: layer_prefix.clone(),
            input_layernorm: format!("{layer_prefix}.input_layernorm.weight"),
            post_attention_layernorm: format!("{layer_prefix}.post_attention_layernorm.weight"),
            attention,
            mlp,
        }
    }

    pub fn mtp_tensor_names(&self, mtp_idx: usize) -> DeepSeekMtpTensorNames {
        let mtp_prefix = format!("model.layers.{}", self.num_hidden_layers + mtp_idx);
        DeepSeekMtpTensorNames {
            mtp_prefix: mtp_prefix.clone(),
            embed_tokens: format!("{mtp_prefix}.embed_tokens.weight"),
            enorm: format!("{mtp_prefix}.enorm.weight"),
            hnorm: format!("{mtp_prefix}.hnorm.weight"),
            eh_proj: format!("{mtp_prefix}.eh_proj.weight"),
            layer: self.layer_tensor_names(self.num_hidden_layers + mtp_idx),
            shared_head_norm: "model.norm.weight".to_string(),
            lm_head: self.lm_head_tensor_name().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEEPSEEK_V3_CONFIG: &str = r#"{
        "architectures": ["DeepseekV3ForCausalLM"],
        "vocab_size": 129280,
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "n_shared_experts": 1,
        "n_routed_experts": 256,
        "routed_scaling_factor": 2.5,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "qk_nope_head_dim": 128,
        "n_group": 8,
        "topk_group": 4,
        "num_experts_per_tok": 8,
        "first_k_dense_replace": 3,
        "norm_topk_prob": true,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": false,
        "rope_theta": 10000.0,
        "rope_interleave": true,
        "num_nextn_predict_layers": 1
    }"#;

    fn topk_routes(
        token_idx: usize,
        expert_indices: impl IntoIterator<Item = usize>,
    ) -> Vec<DeepSeekMoeRoute> {
        expert_indices
            .into_iter()
            .map(|expert_idx| DeepSeekMoeRoute {
                token_idx,
                expert_idx,
                weight: 1.0,
            })
            .collect()
    }

    #[test]
    fn parses_deepseek_v3_reference_config() {
        let cfg = DeepSeekConfig::from_json_str(DEEPSEEK_V3_CONFIG).unwrap();

        assert_eq!(cfg.vocab_size, 129_280);
        assert_eq!(cfg.hidden_size, 7168);
        assert_eq!(cfg.num_hidden_layers, 61);
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.q_lora_rank, Some(1536));
        assert_eq!(cfg.qk_head_dim(), 192);
        assert_eq!(cfg.kv_b_proj_dim(), 32_768);
        assert!(cfg.is_moe());
        assert!(cfg.has_mtp());
        assert!(!cfg.is_moe_layer(2));
        assert!(cfg.is_moe_layer(3));
    }

    #[test]
    fn tensor_names_cover_dense_and_moe_layers() {
        let cfg = DeepSeekConfig::from_json_str(DEEPSEEK_V3_CONFIG).unwrap();
        let dense = cfg.layer_tensor_names(0);
        assert_eq!(
            dense.attention.kv_a_proj_with_mqa,
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        );
        assert_eq!(
            dense.shard_for("model.layers.0.self_attn.q_b_proj.weight"),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            dense.shard_for("model.layers.0.mlp.down_proj.weight"),
            Some(Shard::Row { dim: 1 })
        );

        let moe = cfg.layer_tensor_names(3);
        let DeepSeekMlpTensorNames::Moe(moe_names) = &moe.mlp else {
            panic!("layer 3 should be MoE");
        };
        assert_eq!(moe_names.router, "model.layers.3.mlp.gate.weight");
        let expert = moe_names.expert(17);
        assert_eq!(
            expert.gate_proj,
            "model.layers.3.mlp.experts.17.gate_proj.weight"
        );
        assert_eq!(
            expert.shard_for("model.layers.3.mlp.experts.17.down_proj.weight"),
            Some(Shard::Row { dim: 1 })
        );
        assert_eq!(
            moe.shard_for("model.layers.3.mlp.shared_experts.up_proj.weight"),
            Some(Shard::Column { dim: 0 })
        );
    }

    #[test]
    fn global_and_mtp_names_are_stable() {
        let cfg = DeepSeekConfig::from_json_str(DEEPSEEK_V3_CONFIG).unwrap();
        assert_eq!(
            cfg.shard_for_global_tensor("model.embed_tokens.weight"),
            Some(Shard::VocabParallel { dim: 0 })
        );
        assert_eq!(
            cfg.shard_for_global_tensor("model.norm.weight"),
            Some(Shard::Replicated)
        );
        let mtp = cfg.mtp_tensor_names(0);
        assert_eq!(mtp.embed_tokens, "model.layers.61.embed_tokens.weight");
        assert_eq!(mtp.eh_proj, "model.layers.61.eh_proj.weight");
        assert_eq!(mtp.lm_head, "lm_head.weight");
    }

    #[test]
    fn expert_parallel_config_projects_moe_fields() {
        let cfg = DeepSeekConfig::from_json_str(DEEPSEEK_V3_CONFIG).unwrap();
        let ep = cfg.expert_parallel_config();

        assert_eq!(ep.num_experts, 256);
        assert_eq!(ep.num_experts_per_tok, 8);
        assert_eq!(ep.first_k_dense_replace, 3);
        assert_eq!(ep.moe_intermediate_size, 2048);
        assert_eq!(ep.experts_per_rank(8), Some(32));
        assert_eq!(ep.experts_per_rank(7), None);
        assert!(!ep.is_moe_layer(2));
        assert!(ep.is_moe_layer(3));
    }

    #[test]
    fn moe_forward_batch_groups_routes_by_expert() {
        let cfg = DeepSeekConfig::from_json_str(DEEPSEEK_V3_CONFIG).unwrap();
        let batch = DeepSeekMoeForwardBatch::from_config(
            &cfg,
            3,
            2,
            topk_routes(0, 17..25)
                .into_iter()
                .chain(topk_routes(1, [17, 32, 33, 34, 35, 36, 37, 38]))
                .collect::<Vec<_>>(),
        );

        batch.validate_against(&cfg).unwrap();
        assert_eq!(batch.hidden_size, 7168);
        assert!(batch.include_shared_experts);

        let inputs = batch.expert_inputs();
        let expert17 = inputs.iter().find(|input| input.expert_idx == 17).unwrap();
        let expert32 = inputs.iter().find(|input| input.expert_idx == 32).unwrap();
        assert_eq!(expert17.token_indices, vec![0, 1]);
        assert_eq!(expert17.route_weights, vec![1.0, 1.0]);
        assert_eq!(expert32.token_indices, vec![1]);
    }

    #[test]
    fn moe_forward_plan_filters_local_ep_experts() {
        let cfg = DeepSeekConfig::from_json_str(DEEPSEEK_V3_CONFIG).unwrap();
        let batch = DeepSeekMoeForwardBatch::from_config(
            &cfg,
            3,
            2,
            topk_routes(0, 10..18)
                .into_iter()
                .chain(topk_routes(1, 34..42))
                .collect::<Vec<_>>(),
        );

        let rank0 = batch.ep_forward_plan(&cfg, 0, 8).unwrap();
        let rank1 = batch.ep_forward_plan(&cfg, 1, 8).unwrap();

        assert_eq!(rank0.local_experts, (0..32).collect::<Vec<_>>());
        assert_eq!(rank0.local_inputs.len(), 8);
        assert!(
            rank0
                .local_inputs
                .iter()
                .any(|input| input.expert_idx == 17)
        );
        assert_eq!(rank1.local_experts, (32..64).collect::<Vec<_>>());
        assert_eq!(rank1.local_inputs.len(), 8);
        assert!(
            rank1
                .local_inputs
                .iter()
                .any(|input| input.expert_idx == 41)
        );

        let mtp_batch = DeepSeekMoeForwardBatch::from_config(
            &cfg,
            cfg.num_hidden_layers,
            1,
            topk_routes(0, 0..8),
        );
        assert!(mtp_batch.validate_against(&cfg).is_ok());
    }

    #[test]
    fn moe_forward_batch_rejects_invalid_routes() {
        let cfg = DeepSeekConfig::from_json_str(DEEPSEEK_V3_CONFIG).unwrap();
        let dense_layer = DeepSeekMoeForwardBatch::from_config(
            &cfg,
            2,
            1,
            vec![DeepSeekMoeRoute {
                token_idx: 0,
                expert_idx: 0,
                weight: 1.0,
            }],
        );
        assert!(dense_layer.validate_against(&cfg).is_err());

        let invalid_expert = DeepSeekMoeForwardBatch::from_config(
            &cfg,
            3,
            1,
            vec![DeepSeekMoeRoute {
                token_idx: 0,
                expert_idx: 256,
                weight: 1.0,
            }],
        );
        assert!(invalid_expert.validate_against(&cfg).is_err());

        let invalid_token = DeepSeekMoeForwardBatch::from_config(
            &cfg,
            3,
            1,
            vec![DeepSeekMoeRoute {
                token_idx: 1,
                expert_idx: 0,
                weight: 1.0,
            }],
        );
        assert!(invalid_token.validate_against(&cfg).is_err());

        let invalid_layer =
            DeepSeekMoeForwardBatch::from_config(&cfg, 999, 1, topk_routes(0, 0..8));
        assert!(invalid_layer.validate_against(&cfg).is_err());

        let mut negative_weight =
            DeepSeekMoeForwardBatch::from_config(&cfg, 3, 1, topk_routes(0, 0..8));
        negative_weight.routes[0].weight = -1.0;
        assert!(negative_weight.validate_against(&cfg).is_err());

        let overloaded_token = DeepSeekMoeForwardBatch::from_config(
            &cfg,
            3,
            2,
            (0..9)
                .map(|expert_idx| DeepSeekMoeRoute {
                    token_idx: 0,
                    expert_idx,
                    weight: 1.0,
                })
                .collect::<Vec<_>>(),
        );
        assert!(overloaded_token.validate_against(&cfg).is_err());

        let duplicate_expert = DeepSeekMoeForwardBatch::from_config(
            &cfg,
            3,
            1,
            vec![
                DeepSeekMoeRoute {
                    token_idx: 0,
                    expert_idx: 0,
                    weight: 0.5,
                },
                DeepSeekMoeRoute {
                    token_idx: 0,
                    expert_idx: 0,
                    weight: 0.5,
                },
            ],
        );
        assert!(duplicate_expert.validate_against(&cfg).is_err());

        let mut missing_shared =
            DeepSeekMoeForwardBatch::from_config(&cfg, 3, 1, topk_routes(0, 0..8));
        missing_shared.include_shared_experts = false;
        assert!(missing_shared.validate_against(&cfg).is_err());
    }

    #[test]
    fn rejects_invalid_mla_dimensions() {
        let err = DeepSeekConfig::from_json_str(
            r#"{
                "vocab_size": 100,
                "hidden_size": 16,
                "intermediate_size": 32,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "n_routed_experts": 0,
                "kv_lora_rank": 0,
                "qk_rope_head_dim": 8,
                "v_head_dim": 8,
                "qk_nope_head_dim": 8,
                "num_experts_per_tok": 0,
                "first_k_dense_replace": 0,
                "max_position_embeddings": 1024,
                "rms_norm_eps": 1e-6
            }"#,
        )
        .unwrap_err();
        assert!(matches!(err, DeepSeekConfigError::InvalidConfig(_)));
    }
}
