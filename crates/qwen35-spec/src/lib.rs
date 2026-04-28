use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Qwen35ConfigError {
    #[error("invalid qwen3.5 config: {0}")]
    InvalidConfig(&'static str),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Qwen35ConfigError>;

/// How a tensor's weight should be split across a tensor-parallel group.
/// Mirrors SGLang `python/sglang/srt/layers/linear.py` parallel-linear classes.
///
/// `dim` follows the HF safetensors layout for nn.Linear: row 0 is the output
/// (out_features) axis and row 1 is the input (in_features) axis. So
/// `Column { dim: 0 }` matches SGLang's `ColumnParallelLinear` (split output)
/// and `Row { dim: 1 }` matches `RowParallelLinear` (split input).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard {
    /// Replicated on every rank (norms, biases that aren't per-head, etc).
    Replicated,
    /// Column-parallel: split along output dim. SGLang `linear.py:289`.
    Column { dim: usize },
    /// Row-parallel: split along input dim. SGLang `linear.py:1335`.
    Row { dim: usize },
    /// Merged column-parallel: SGLang `linear.py:485`. Used by `gate_up_proj`
    /// and other fused projections; per-projection sizes come from config at
    /// runtime (not encoded here, since they're model-dependent).
    MergedColumn { dim: usize },
    /// Fused QKV: SGLang `linear.py:889 QKVParallelLinear`. The KV-head
    /// replication rule (SGLang `models/qwen3.py:84-95`) is applied at
    /// runtime, not encoded here.
    QkvFused { dim: usize },
    /// Vocab-parallel: SGLang `vocab_parallel_embedding.py:161`.
    /// Used for `embed_tokens` and (untied) `lm_head`.
    VocabParallel { dim: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35CommonLayerTensorNames {
    pub layer_prefix: String,
    pub mlp_prefix: String,
    pub input_layernorm: String,
    pub post_attention_layernorm: String,
    pub mlp_gate_proj: String,
    pub mlp_up_proj: String,
    pub mlp_down_proj: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35FullAttentionTensorNames {
    pub attention_prefix: String,
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub q_norm: String,
    pub k_norm: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35LinearAttentionTensorNames {
    pub attention_prefix: String,
    pub in_proj_qkv: String,
    pub in_proj_z: String,
    pub in_proj_b: String,
    pub in_proj_a: String,
    pub conv1d_weight: String,
    pub dt_bias: String,
    pub a_log: String,
    pub norm: String,
    pub out_proj: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen35AttentionTensorNames {
    Full(Qwen35FullAttentionTensorNames),
    Linear(Qwen35LinearAttentionTensorNames),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35LayerTensorNames {
    pub common: Qwen35CommonLayerTensorNames,
    pub attention: Qwen35AttentionTensorNames,
}

impl Qwen35CommonLayerTensorNames {
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.mlp_gate_proj || name == self.mlp_up_proj {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.mlp_down_proj {
            return Some(Shard::Row { dim: 1 });
        }
        if name == self.input_layernorm || name == self.post_attention_layernorm {
            return Some(Shard::Replicated);
        }
        None
    }
}

impl Qwen35FullAttentionTensorNames {
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.q_proj || name == self.k_proj || name == self.v_proj {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.o_proj {
            return Some(Shard::Row { dim: 1 });
        }
        if name == self.q_norm || name == self.k_norm {
            return Some(Shard::Replicated);
        }
        None
    }
}

impl Qwen35LinearAttentionTensorNames {
    /// Linear-attention (Gated DeltaNet) shard mapping. Mirrors SGLang
    /// `models/qwen3_5.py`: in-projections are column-parallel (the
    /// checkpoint stores `in_proj_qkv` already fused along dim 0, so
    /// `MergedColumn` is the right shape contract); `conv1d`, per-head
    /// `dt_bias`, and `A_log` are split along dim 0 with
    /// `sharded_weight_loader(0)`; `out_proj` is row-parallel; the gated
    /// RMSNorm scale is replicated.
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.in_proj_qkv {
            return Some(Shard::MergedColumn { dim: 0 });
        }
        if name == self.in_proj_z || name == self.in_proj_b || name == self.in_proj_a {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.conv1d_weight || name == self.dt_bias || name == self.a_log {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.out_proj {
            return Some(Shard::Row { dim: 1 });
        }
        if name == self.norm {
            return Some(Shard::Replicated);
        }
        None
    }
}

impl Qwen35AttentionTensorNames {
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        match self {
            Self::Full(attn) => attn.shard_for(name),
            Self::Linear(attn) => attn.shard_for(name),
        }
    }
}

impl Qwen35LayerTensorNames {
    /// Returns `Some(Shard)` for tensors this spec knows about; `None` for
    /// any name not part of a transformer layer (caller falls back to
    /// `Shard::Replicated`). Per-layer tensors only — global tensors live
    /// on `Qwen35Config::shard_for_global_tensor`.
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        self.common
            .shard_for(name)
            .or_else(|| self.attention.shard_for(name))
    }
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f32,
    partial_rotary_factor: f32,
}

#[derive(Debug, Deserialize, Default)]
struct MoeConfigRaw {
    #[serde(default)]
    num_experts: usize,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default = "default_decoder_sparse_step")]
    decoder_sparse_step: usize,
    #[serde(default)]
    moe_intermediate_size: usize,
    #[serde(default)]
    shared_expert_intermediate_size: usize,
    #[serde(default = "default_norm_topk_prob")]
    norm_topk_prob: bool,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
}

fn default_decoder_sparse_step() -> usize {
    1
}

fn default_norm_topk_prob() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    hidden_size: usize,
    #[serde(default)]
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(alias = "num_kv_heads")]
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    layer_types: Vec<LayerType>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    rope_parameters: RopeParameters,
    eos_token_id: u32,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default = "default_tie_word_embeddings")]
    tie_word_embeddings: bool,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    context_length: Option<usize>,
    #[serde(default)]
    seq_length: Option<usize>,

    // ── Mixture-of-Experts fields (Qwen3.6 / Qwen3_5_Moe). ─────────────────
    // Accepted both flat inside `text_config` (Qwen3.6 HF layout) and nested
    // under a `moe_config` sub-block. When both are present the nested values
    // are merged on top of the flat ones (any non-default nested field wins).
    #[serde(default)]
    num_experts: usize,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default = "default_decoder_sparse_step")]
    decoder_sparse_step: usize,
    #[serde(default)]
    moe_intermediate_size: usize,
    #[serde(default)]
    shared_expert_intermediate_size: usize,
    #[serde(default = "default_norm_topk_prob")]
    norm_topk_prob: bool,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
    #[serde(default)]
    moe_config: Option<MoeConfigRaw>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawConfig {
    Nested { text_config: TextConfig },
    Flat(TextConfig),
}

impl RawConfig {
    fn into_text(self) -> TextConfig {
        match self {
            Self::Nested { text_config } => text_config,
            Self::Flat(text_config) => text_config,
        }
    }
}

fn default_tie_word_embeddings() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub stop_token_ids: Vec<u32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub rotary_dim: usize,
    pub rope_cache_len_hint: Option<usize>,
    pub layer_types: Vec<LayerType>,

    // ── Mixture-of-Experts (Qwen3.6 / Qwen3_5_Moe). ────────────────────────
    // `num_experts == 0` means the model is dense (classic Qwen3.5). When
    // populated, these fields describe the `SparseMoeBlock` shape per the
    // mlx-lm `qwen3_5_moe.py` reference. See [`Qwen35Config::is_moe`] and
    // [`Qwen35Config::is_moe_layer`].
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}

impl Qwen35Config {
    /// Train-side current truth: the active train stack only supports dense
    /// MLP layers with full-attention blocks. Infer-side parsing remains
    /// broader, but train entrypoints use this helper to fail early when a
    /// hybrid or MoE config is passed in.
    pub fn is_train_dense_full_attention_only(&self) -> bool {
        self.num_hidden_layers > 0
            && self.layer_types.len() == self.num_hidden_layers
            && !self.is_moe()
            && self
                .layer_types
                .iter()
                .all(|&layer| layer == LayerType::FullAttention)
    }

    /// Shared train-side contract for scratch pretrain. Dense full-attn and
    /// dense hybrid linear-attn are both allowed; MoE remains rejected.
    pub fn validate_train_scratch_contract(&self) -> Result<()> {
        self.validate()?;
        if self.is_moe() {
            return Err(Qwen35ConfigError::InvalidConfig(
                "train-side qwen3.5 currently supports dense MLP layers only",
            ));
        }
        if self.rope_cache_len_hint.is_none() {
            return Err(Qwen35ConfigError::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ));
        }
        Ok(())
    }

    /// Shared train-side dense/full-attn contract for places that still
    /// intentionally pin the older scratch acceptance surface.
    pub fn validate_train_dense_full_attention_contract(&self) -> Result<()> {
        self.validate_train_scratch_contract()?;
        if self
            .layer_types
            .iter()
            .any(|layer_type| *layer_type != LayerType::FullAttention)
        {
            return Err(Qwen35ConfigError::InvalidConfig(
                "train-side qwen3.5 currently supports full-attention layers only",
            ));
        }
        if self.rotary_dim != self.head_dim {
            return Err(Qwen35ConfigError::InvalidConfig(
                "train-side qwen3.5 requires rotary_dim == head_dim",
            ));
        }
        Ok(())
    }

    /// Shared train-side contract for LoRA / frozen-eval Qwen3.5. This is
    /// intentionally broader than the scratch-pretrain path: dense full-attn
    /// and hybrid linear-attn configs are allowed, but MoE remains rejected.
    pub fn validate_train_lora_or_frozen_contract(&self) -> Result<()> {
        self.validate()?;
        if self.is_moe() {
            return Err(Qwen35ConfigError::InvalidConfig(
                "train-side qwen3.5 currently supports dense MLP layers only",
            ));
        }
        if self.rope_cache_len_hint.is_none() {
            return Err(Qwen35ConfigError::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ));
        }
        Ok(())
    }

    pub fn model_prefix(&self) -> &'static str {
        "model.language_model"
    }

    pub fn embed_tokens_tensor_name(&self) -> &'static str {
        "model.language_model.embed_tokens.weight"
    }

    pub fn norm_tensor_name(&self) -> &'static str {
        "model.language_model.norm.weight"
    }

    pub fn lm_head_tensor_name(&self) -> &'static str {
        self.embed_tokens_tensor_name()
    }

    /// Sharding for non-layer ("global") tensors. Returns `None` for any
    /// name not recognised; callers fall back to `Shard::Replicated`.
    pub fn shard_for_global_tensor(&self, name: &str) -> Option<Shard> {
        if name == self.embed_tokens_tensor_name() {
            return Some(Shard::VocabParallel { dim: 0 });
        }
        if name == "lm_head.weight" {
            return Some(Shard::VocabParallel { dim: 0 });
        }
        if name == self.norm_tensor_name() {
            return Some(Shard::Replicated);
        }
        None
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json_str(&content)
    }

    pub fn from_json_str(content: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(content)?;
        Self::from_json_value(&value)
    }

    pub fn from_json_value(value: &serde_json::Value) -> Result<Self> {
        let raw: RawConfig = serde_json::from_value(value.clone())?;
        let text = raw.into_text();
        let stop_token_ids = vec![text.eos_token_id];
        Self::from_text_config(text, stop_token_ids)
    }

    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let content = fs::read_to_string(&config_path)?;
        let value: serde_json::Value = serde_json::from_str(&content)?;
        let raw: RawConfig = serde_json::from_value(value)?;
        let text = raw.into_text();
        let stop_token_ids = Self::load_stop_token_ids(model_dir, text.eos_token_id)?;
        Self::from_text_config(text, stop_token_ids)
    }

    pub fn from_file(model_path: &str) -> Result<Self> {
        Self::from_model_dir(model_path)
    }

    fn from_text_config(text: TextConfig, stop_token_ids: Vec<u32>) -> Result<Self> {
        let rotary_dim =
            (text.head_dim as f32 * text.rope_parameters.partial_rotary_factor) as usize;

        // Merge nested `moe_config` sub-block (if present) on top of the flat
        // text_config MoE fields. Nested fields override flat ones only when
        // non-default (non-zero / non-empty); this lets either layout succeed.
        let TextConfig {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            layer_types,
            linear_conv_kernel_dim,
            linear_key_head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_value_head_dim,
            rope_parameters,
            eos_token_id: _eos_token_id,
            bos_token_id,
            tie_word_embeddings,
            max_position_embeddings,
            context_length,
            seq_length,
            num_experts: mut moe_num_experts,
            num_experts_per_tok: mut moe_num_experts_per_tok,
            decoder_sparse_step: mut moe_decoder_sparse_step,
            moe_intermediate_size: mut moe_intermediate_size_val,
            shared_expert_intermediate_size: mut moe_shared_expert_intermediate_size,
            norm_topk_prob: mut moe_norm_topk_prob,
            mlp_only_layers: mut moe_mlp_only_layers,
            moe_config,
        } = text;

        if let Some(nested) = moe_config {
            if nested.num_experts != 0 {
                moe_num_experts = nested.num_experts;
            }
            if nested.num_experts_per_tok != 0 {
                moe_num_experts_per_tok = nested.num_experts_per_tok;
            }
            if nested.decoder_sparse_step != default_decoder_sparse_step() {
                moe_decoder_sparse_step = nested.decoder_sparse_step;
            }
            if nested.moe_intermediate_size != 0 {
                moe_intermediate_size_val = nested.moe_intermediate_size;
            }
            if nested.shared_expert_intermediate_size != 0 {
                moe_shared_expert_intermediate_size = nested.shared_expert_intermediate_size;
            }
            if !nested.norm_topk_prob {
                moe_norm_topk_prob = nested.norm_topk_prob;
            }
            if !nested.mlp_only_layers.is_empty() {
                moe_mlp_only_layers = nested.mlp_only_layers;
            }
        }

        let config = Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            vocab_size,
            rms_norm_eps,
            stop_token_ids,
            bos_token_id,
            eos_token_id: _eos_token_id,
            tie_word_embeddings,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            linear_num_key_heads,
            linear_key_head_dim,
            linear_num_value_heads,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            rope_theta: rope_parameters.rope_theta,
            partial_rotary_factor: rope_parameters.partial_rotary_factor,
            rotary_dim,
            rope_cache_len_hint: max_position_embeddings.or(context_length).or(seq_length),
            layer_types,
            num_experts: moe_num_experts,
            num_experts_per_tok: moe_num_experts_per_tok,
            decoder_sparse_step: moe_decoder_sparse_step,
            moe_intermediate_size: moe_intermediate_size_val,
            shared_expert_intermediate_size: moe_shared_expert_intermediate_size,
            norm_topk_prob: moe_norm_topk_prob,
            mlp_only_layers: moe_mlp_only_layers,
        };
        config.validate()?;
        Ok(config)
    }

    /// Whether this checkpoint uses Mixture-of-Experts MLP blocks.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Whether the given layer index uses a MoE block (sparse) vs a dense MLP.
    ///
    /// Mirrors the mlx-lm `qwen3_5_moe.py` selection rule: a layer is MoE iff
    /// the model has experts, the layer is not in `mlp_only_layers`, and the
    /// (1-indexed) layer id is a multiple of `decoder_sparse_step`.
    pub fn is_moe_layer(&self, idx: usize) -> bool {
        self.is_moe()
            && !self.mlp_only_layers.contains(&idx)
            && (idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_hidden_layers == 0 || self.layer_types.is_empty() {
            return Err(Qwen35ConfigError::InvalidConfig(
                "num_hidden_layers and layer_types must be non-zero",
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(Qwen35ConfigError::InvalidConfig(
                "layer_types length must equal num_hidden_layers",
            ));
        }
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 || self.head_dim == 0 {
            return Err(Qwen35ConfigError::InvalidConfig(
                "full-attention heads and head_dim must be non-zero",
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Qwen35ConfigError::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }
        if self.linear_num_key_heads == 0
            || self.linear_num_value_heads == 0
            || self.linear_key_head_dim == 0
            || self.linear_value_head_dim == 0
        {
            return Err(Qwen35ConfigError::InvalidConfig(
                "linear-attention heads and dims must be non-zero",
            ));
        }
        if !self
            .linear_num_value_heads
            .is_multiple_of(self.linear_num_key_heads)
        {
            return Err(Qwen35ConfigError::InvalidConfig(
                "linear_num_value_heads must be divisible by linear_num_key_heads",
            ));
        }
        if self.linear_conv_kernel_dim < 2 {
            return Err(Qwen35ConfigError::InvalidConfig(
                "linear_conv_kernel_dim must be at least 2",
            ));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(Qwen35ConfigError::InvalidConfig(
                "head_dim must be even for RoPE",
            ));
        }
        if self.rotary_dim == 0 || !self.rotary_dim.is_multiple_of(2) {
            return Err(Qwen35ConfigError::InvalidConfig(
                "rotary_dim must be even and non-zero",
            ));
        }
        Ok(())
    }

    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    pub fn load_stop_token_ids(model_dir: impl AsRef<Path>, fallback_eos: u32) -> Result<Vec<u32>> {
        let generation_config_path = model_dir.as_ref().join("generation_config.json");
        let ids = match fs::read_to_string(&generation_config_path) {
            Ok(content) => {
                let value: serde_json::Value = serde_json::from_str(&content)?;
                let mut ids = Vec::new();
                if let Some(eos) = value.get("eos_token_id") {
                    match eos {
                        serde_json::Value::Number(number) => {
                            if let Some(id) = number.as_u64() {
                                ids.push(id as u32);
                            }
                        }
                        serde_json::Value::Array(array) => {
                            for item in array {
                                if let Some(id) = item.as_u64() {
                                    ids.push(id as u32);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if ids.is_empty() {
                    vec![fallback_eos]
                } else {
                    ids
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => vec![fallback_eos],
            Err(err) => return Err(err.into()),
        };

        let mut deduped = ids;
        deduped.sort_unstable();
        deduped.dedup();
        Ok(deduped)
    }

    pub fn rope_cache_len_hint(&self) -> Option<usize> {
        self.rope_cache_len_hint
    }

    pub fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&layer| layer == LayerType::FullAttention)
            .count()
    }

    pub fn full_attn_q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim * 2
    }

    pub fn full_attn_q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn full_attn_kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn linear_attn_qkv_dim(&self) -> usize {
        let q_dim = self.linear_num_key_heads * self.linear_key_head_dim;
        let k_dim = q_dim;
        let v_dim = self.linear_num_value_heads * self.linear_value_head_dim;
        q_dim + k_dim + v_dim
    }

    pub fn linear_attn_z_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn layer_tensor_names(&self, layer_idx: usize) -> Qwen35LayerTensorNames {
        let layer_prefix = format!("{}.layers.{layer_idx}", self.model_prefix());
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let common = Qwen35CommonLayerTensorNames {
            layer_prefix: layer_prefix.clone(),
            mlp_prefix: mlp_prefix.clone(),
            input_layernorm: format!("{layer_prefix}.input_layernorm.weight"),
            post_attention_layernorm: format!("{layer_prefix}.post_attention_layernorm.weight"),
            mlp_gate_proj: format!("{mlp_prefix}.gate_proj.weight"),
            mlp_up_proj: format!("{mlp_prefix}.up_proj.weight"),
            mlp_down_proj: format!("{mlp_prefix}.down_proj.weight"),
        };

        let attention = match self.layer_types[layer_idx] {
            LayerType::FullAttention => {
                let attention_prefix = format!("{layer_prefix}.self_attn");
                Qwen35AttentionTensorNames::Full(Qwen35FullAttentionTensorNames {
                    attention_prefix: attention_prefix.clone(),
                    q_proj: format!("{attention_prefix}.q_proj.weight"),
                    k_proj: format!("{attention_prefix}.k_proj.weight"),
                    v_proj: format!("{attention_prefix}.v_proj.weight"),
                    o_proj: format!("{attention_prefix}.o_proj.weight"),
                    q_norm: format!("{attention_prefix}.q_norm.weight"),
                    k_norm: format!("{attention_prefix}.k_norm.weight"),
                })
            }
            LayerType::LinearAttention => {
                let attention_prefix = format!("{layer_prefix}.linear_attn");
                Qwen35AttentionTensorNames::Linear(Qwen35LinearAttentionTensorNames {
                    attention_prefix: attention_prefix.clone(),
                    in_proj_qkv: format!("{attention_prefix}.in_proj_qkv.weight"),
                    in_proj_z: format!("{attention_prefix}.in_proj_z.weight"),
                    in_proj_b: format!("{attention_prefix}.in_proj_b.weight"),
                    in_proj_a: format!("{attention_prefix}.in_proj_a.weight"),
                    conv1d_weight: format!("{attention_prefix}.conv1d.weight"),
                    dt_bias: format!("{attention_prefix}.dt_bias"),
                    a_log: format!("{attention_prefix}.A_log"),
                    norm: format!("{attention_prefix}.norm.weight"),
                    out_proj: format!("{attention_prefix}.out_proj.weight"),
                })
            }
        };

        Qwen35LayerTensorNames { common, attention }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    const NESTED_CONFIG_JSON: &str = r#"{
        "text_config": {
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "layer_types": [
                "full_attention", "full_attention", "linear_attention", "linear_attention",
                "linear_attention", "linear_attention", "linear_attention", "linear_attention",
                "linear_attention", "linear_attention", "linear_attention", "linear_attention",
                "linear_attention", "linear_attention", "linear_attention", "linear_attention",
                "linear_attention", "linear_attention", "linear_attention", "linear_attention",
                "linear_attention", "linear_attention", "linear_attention", "linear_attention",
                "linear_attention", "linear_attention", "linear_attention", "linear_attention",
                "linear_attention", "linear_attention", "full_attention", "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5
            },
            "eos_token_id": 248044,
            "bos_token_id": 248000,
            "tie_word_embeddings": true,
            "max_position_embeddings": 32768
        }
    }"#;

    const DENSE_FULL_ATTENTION_CONFIG_JSON: &str = r#"{
        "text_config": {
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "layer_types": [
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 1.0
            },
            "eos_token_id": 248044,
            "bos_token_id": 248000,
            "tie_word_embeddings": true,
            "max_position_embeddings": 32768
        }
    }"#;

    #[test]
    fn parses_nested_qwen35_config() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.linear_num_key_heads, 16);
        assert_eq!(config.linear_num_value_heads, 32);
        assert_eq!(config.linear_conv_kernel_dim, 4);
        assert_eq!(config.eos_token_id, 248_044);
        assert_eq!(config.bos_token_id, Some(248_000));
        assert_eq!(config.stop_token_ids, vec![248_044]);
        assert_eq!(config.rotary_dim, 128);
        assert_eq!(config.rope_cache_len_hint, Some(32_768));
        assert_eq!(config.num_full_attention_layers(), 4);
        assert_eq!(config.full_attn_q_proj_dim(), 8192);
        assert_eq!(config.full_attn_q_dim(), 4096);
        assert_eq!(config.full_attn_kv_dim(), 1024);
        assert_eq!(config.linear_attn_qkv_dim(), 8192);
        assert_eq!(config.linear_attn_z_dim(), 4096);
    }

    #[test]
    fn validates_dense_full_attention_train_contract() {
        let config = Qwen35Config::from_json_str(DENSE_FULL_ATTENTION_CONFIG_JSON).unwrap();
        assert!(config.is_train_dense_full_attention_only());
        config.validate_train_scratch_contract().unwrap();
        config
            .validate_train_dense_full_attention_contract()
            .unwrap();
    }

    #[test]
    fn accepts_hybrid_configs_for_scratch_contract() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        config.validate_train_scratch_contract().unwrap();
    }

    #[test]
    fn rejects_hybrid_configs_for_dense_train_contract() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        assert!(!config.is_train_dense_full_attention_only());
        let err = config
            .validate_train_dense_full_attention_contract()
            .unwrap_err();
        assert!(matches!(
            err,
            Qwen35ConfigError::InvalidConfig(
                "train-side qwen3.5 currently supports full-attention layers only"
            )
        ));
    }

    #[test]
    fn rejects_moe_configs_for_train_contract() {
        let mut config = Qwen35Config::from_json_str(DENSE_FULL_ATTENTION_CONFIG_JSON).unwrap();
        config.num_experts = 8;
        config.num_experts_per_tok = 2;
        assert!(!config.is_train_dense_full_attention_only());
        let err = config
            .validate_train_dense_full_attention_contract()
            .unwrap_err();
        assert!(matches!(
            err,
            Qwen35ConfigError::InvalidConfig(
                "train-side qwen3.5 currently supports dense MLP layers only"
            )
        ));
    }

    #[test]
    fn parses_flat_qwen35_config() {
        let config = Qwen35Config::from_json_str(
            r#"{
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_hidden_layers": 64,
                "num_attention_heads": 40,
                "num_kv_heads": 8,
                "head_dim": 128,
                "vocab_size": 152064,
                "rms_norm_eps": 1e-6,
                "layer_types": ["full_attention", "linear_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 20,
                "linear_num_value_heads": 40,
                "linear_value_head_dim": 128,
                "rope_parameters": {
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 1.0
                },
                "eos_token_id": 151645,
                "seq_length": 65536
            }"#,
        )
        .unwrap_err();

        assert!(matches!(
            config,
            Qwen35ConfigError::InvalidConfig("layer_types length must equal num_hidden_layers")
        ));
    }

    #[test]
    fn exposes_full_attention_tensor_names() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        let names = config.layer_tensor_names(0);
        assert_eq!(
            names.common.input_layernorm,
            "model.language_model.layers.0.input_layernorm.weight"
        );
        match names.attention {
            Qwen35AttentionTensorNames::Full(attn) => {
                assert_eq!(
                    attn.q_proj,
                    "model.language_model.layers.0.self_attn.q_proj.weight"
                );
                assert_eq!(
                    attn.k_norm,
                    "model.language_model.layers.0.self_attn.k_norm.weight"
                );
            }
            Qwen35AttentionTensorNames::Linear(_) => panic!("expected full-attention names"),
        }
    }

    #[test]
    fn exposes_linear_attention_tensor_names() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        let names = config.layer_tensor_names(2);
        assert_eq!(
            names.common.mlp_gate_proj,
            "model.language_model.layers.2.mlp.gate_proj.weight"
        );
        match names.attention {
            Qwen35AttentionTensorNames::Linear(attn) => {
                assert_eq!(
                    attn.in_proj_qkv,
                    "model.language_model.layers.2.linear_attn.in_proj_qkv.weight"
                );
                assert_eq!(
                    attn.norm,
                    "model.language_model.layers.2.linear_attn.norm.weight"
                );
                assert_eq!(
                    attn.a_log,
                    "model.language_model.layers.2.linear_attn.A_log"
                );
            }
            Qwen35AttentionTensorNames::Full(_) => panic!("expected linear-attention names"),
        }
    }

    const QWEN36_MOE_FLAT_JSON: &str = r#"{
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "layer_types": [
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            },
            "eos_token_id": 248044,
            "tie_word_embeddings": false,
            "max_position_embeddings": 262144,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        }
    }"#;

    #[test]
    fn parses_qwen36_moe_flat_config() {
        let config = Qwen35Config::from_json_str(QWEN36_MOE_FLAT_JSON).unwrap();
        assert!(config.is_moe());
        assert_eq!(config.num_experts, 256);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.moe_intermediate_size, 512);
        assert_eq!(config.shared_expert_intermediate_size, 512);
        assert_eq!(config.decoder_sparse_step, 1);
        assert!(config.norm_topk_prob);
        assert!(config.mlp_only_layers.is_empty());
        // Qwen3.6 omits `intermediate_size` at the text_config root — must
        // default to 0 rather than fail deserialization.
        assert_eq!(config.intermediate_size, 0);
        assert_eq!(config.num_hidden_layers, 40);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 2);
        // With decoder_sparse_step=1 and empty mlp_only_layers, every layer is MoE.
        for idx in 0..config.num_hidden_layers {
            assert!(config.is_moe_layer(idx), "layer {idx} should be MoE");
        }
    }

    #[test]
    fn parses_qwen36_moe_with_nested_moe_config() {
        // Same shape but MoE fields pushed into a nested `moe_config` block.
        let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 248320,
                "rms_norm_eps": 1e-6,
                "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "rope_parameters": {
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25
                },
                "eos_token_id": 248044,
                "moe_config": {
                    "num_experts": 128,
                    "num_experts_per_tok": 4,
                    "decoder_sparse_step": 2,
                    "moe_intermediate_size": 1024,
                    "shared_expert_intermediate_size": 1024,
                    "norm_topk_prob": false,
                    "mlp_only_layers": [0]
                }
            }
        }"#;
        let config = Qwen35Config::from_json_str(json).unwrap();
        assert!(config.is_moe());
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.num_experts_per_tok, 4);
        assert_eq!(config.decoder_sparse_step, 2);
        assert_eq!(config.moe_intermediate_size, 1024);
        assert!(!config.norm_topk_prob);
        assert_eq!(config.mlp_only_layers, vec![0]);
        // With decoder_sparse_step=2 and mlp_only_layers=[0]:
        //   idx 0 -> in mlp_only_layers -> false
        //   idx 1 -> (1+1)%2 == 0 -> true
        //   idx 2 -> (2+1)%2 != 0 -> false
        //   idx 3 -> (3+1)%2 == 0 -> true
        assert!(!config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
        assert!(!config.is_moe_layer(2));
        assert!(config.is_moe_layer(3));
    }

    #[test]
    fn dense_qwen35_config_is_not_moe() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        assert!(!config.is_moe());
        assert_eq!(config.num_experts, 0);
        assert!(!config.is_moe_layer(0));
        assert!(!config.is_moe_layer(5));
    }

    #[test]
    fn every_full_attention_tensor_name_has_a_shard() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        let names = config.layer_tensor_names(0);
        let attn = match &names.attention {
            Qwen35AttentionTensorNames::Full(a) => a.clone(),
            _ => panic!("expected full attention layer at idx 0"),
        };
        for name in [
            &names.common.input_layernorm,
            &names.common.post_attention_layernorm,
            &names.common.mlp_gate_proj,
            &names.common.mlp_up_proj,
            &names.common.mlp_down_proj,
            &attn.q_proj,
            &attn.k_proj,
            &attn.v_proj,
            &attn.o_proj,
            &attn.q_norm,
            &attn.k_norm,
        ] {
            assert!(names.shard_for(name).is_some(), "missing shard for {name}");
        }
    }

    #[test]
    fn every_linear_attention_tensor_name_has_a_shard() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        let names = config.layer_tensor_names(2);
        let attn = match &names.attention {
            Qwen35AttentionTensorNames::Linear(a) => a.clone(),
            _ => panic!("expected linear attention layer at idx 2"),
        };
        for name in [
            &names.common.input_layernorm,
            &names.common.post_attention_layernorm,
            &names.common.mlp_gate_proj,
            &names.common.mlp_up_proj,
            &names.common.mlp_down_proj,
            &attn.in_proj_qkv,
            &attn.in_proj_z,
            &attn.in_proj_b,
            &attn.in_proj_a,
            &attn.conv1d_weight,
            &attn.dt_bias,
            &attn.a_log,
            &attn.norm,
            &attn.out_proj,
        ] {
            assert!(names.shard_for(name).is_some(), "missing shard for {name}");
        }
    }

    #[test]
    fn full_attn_qkv_proj_is_column_dim0_and_o_proj_is_row_dim1() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        let names = config.layer_tensor_names(0);
        let Qwen35AttentionTensorNames::Full(attn) = names.attention.clone() else {
            panic!("expected full-attention layer");
        };
        assert_eq!(
            names.shard_for(&attn.q_proj),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&attn.k_proj),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&attn.v_proj),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(names.shard_for(&attn.o_proj), Some(Shard::Row { dim: 1 }));
    }

    #[test]
    fn mlp_gate_and_up_proj_are_column_dim0() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        let names = config.layer_tensor_names(0);
        assert_eq!(
            names.shard_for(&names.common.mlp_gate_proj),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&names.common.mlp_up_proj),
            Some(Shard::Column { dim: 0 })
        );
    }

    #[test]
    fn mlp_down_proj_is_row_dim1() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        let names = config.layer_tensor_names(0);
        assert_eq!(
            names.shard_for(&names.common.mlp_down_proj),
            Some(Shard::Row { dim: 1 })
        );
    }

    #[test]
    fn embed_tokens_and_lm_head_are_vocab_parallel_dim0() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        assert_eq!(
            config.shard_for_global_tensor(config.embed_tokens_tensor_name()),
            Some(Shard::VocabParallel { dim: 0 })
        );
        assert_eq!(
            config.shard_for_global_tensor("lm_head.weight"),
            Some(Shard::VocabParallel { dim: 0 })
        );
    }

    #[test]
    fn norm_tensors_are_replicated() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        let full = config.layer_tensor_names(0);
        assert_eq!(
            full.shard_for(&full.common.input_layernorm),
            Some(Shard::Replicated)
        );
        assert_eq!(
            full.shard_for(&full.common.post_attention_layernorm),
            Some(Shard::Replicated)
        );
        let Qwen35AttentionTensorNames::Full(attn) = full.attention.clone() else {
            panic!();
        };
        assert_eq!(full.shard_for(&attn.q_norm), Some(Shard::Replicated));
        assert_eq!(full.shard_for(&attn.k_norm), Some(Shard::Replicated));

        let lin = config.layer_tensor_names(2);
        let Qwen35AttentionTensorNames::Linear(attn) = lin.attention.clone() else {
            panic!();
        };
        assert_eq!(lin.shard_for(&attn.norm), Some(Shard::Replicated));

        assert_eq!(
            config.shard_for_global_tensor(config.norm_tensor_name()),
            Some(Shard::Replicated)
        );
    }

    #[test]
    fn linear_attn_in_proj_qkv_is_merged_column_and_out_proj_is_row() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        let names = config.layer_tensor_names(2);
        let Qwen35AttentionTensorNames::Linear(attn) = names.attention.clone() else {
            panic!("expected linear-attention layer at idx 2");
        };
        assert_eq!(
            names.shard_for(&attn.in_proj_qkv),
            Some(Shard::MergedColumn { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&attn.in_proj_z),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&attn.in_proj_b),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&attn.in_proj_a),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&attn.conv1d_weight),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            names.shard_for(&attn.dt_bias),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(names.shard_for(&attn.a_log), Some(Shard::Column { dim: 0 }));
        assert_eq!(names.shard_for(&attn.out_proj), Some(Shard::Row { dim: 1 }));
    }

    #[test]
    fn unknown_tensor_returns_none() {
        let config = Qwen35Config::from_json_str(NESTED_CONFIG_JSON).unwrap();
        let names = config.layer_tensor_names(0);
        assert_eq!(
            names.shard_for("model.language_model.layers.0.unknown.weight"),
            None
        );
        assert_eq!(config.shard_for_global_tensor("not.a.tensor"), None);
    }

    #[test]
    fn loads_stop_tokens_from_generation_config() {
        let temp = tempdir().unwrap();
        fs::write(temp.path().join("config.json"), NESTED_CONFIG_JSON).unwrap();
        fs::write(
            temp.path().join("generation_config.json"),
            r#"{"eos_token_id":[248044,248167,248044]}"#,
        )
        .unwrap();

        let config = Qwen35Config::from_model_dir(temp.path()).unwrap();
        assert_eq!(config.stop_token_ids, vec![248_044, 248_167]);
        assert!(config.is_stop_token(248_167));
    }
}
