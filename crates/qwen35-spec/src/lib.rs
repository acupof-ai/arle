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

/// Mirrors SGLang `python/sglang/srt/layers/linear.py` parallel-linear classes.
///
/// `dim` follows the HF safetensors layout for nn.Linear: row 0 is the output
/// (out_features) axis and row 1 is the input (in_features) axis. So
/// `Column { dim: 0 }` matches SGLang's `ColumnParallelLinear` (split output)
/// and `Row { dim: 1 }` matches `RowParallelLinear` (split input).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard {
    Replicated,
    Column {
        dim: usize,
    },
    Row {
        dim: usize,
    },
    /// Merged column-parallel: used by `gate_up_proj` and other fused
    /// projections; per-projection sizes come from config at runtime (not
    /// encoded here, since they're model-dependent).
    MergedColumn {
        dim: usize,
    },
    /// Fused QKV. The KV-head replication rule (SGLang `models/qwen3.py:84-95`)
    /// is applied at runtime, not encoded here.
    QkvFused {
        dim: usize,
    },
    /// Used for `embed_tokens` and (untied) `lm_head`.
    VocabParallel {
        dim: usize,
    },
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

/// NextN-MTP draft-head tensor names for Qwen3.6 speculative decode. The head is
/// a single FULL-attention transformer layer (`mtp.layers.0.*`) plus the `fc`
/// concat-projection (`[hidden, 2*hidden]`), two pre-`fc` RMSNorms over the
/// candidate embedding + previous hidden, and a final RMSNorm before the SHARED
/// lm_head. All names are top-level `mtp.*` (verified against Qwen3.6-27B-FP8;
/// 15 tensors). `lm_head` + `embed_tokens` are shared with the base model and
/// are NOT part of this set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35MtpTensorNames {
    pub fc: String,
    pub pre_fc_norm_embedding: String,
    pub pre_fc_norm_hidden: String,
    pub norm: String,
    pub layer: Qwen35LayerTensorNames,
}

/// MoE block tensor names for one sparse layer (Qwen3.6 / Qwen3_5_Moe).
///
/// Router + shared expert are common to both routed-expert layouts:
///   router : `{mlp}.gate.weight`                               `[E, hidden]`
///   shared : `{mlp}.shared_expert.{gate,up,down}_proj.weight`
///   s-gate : `{mlp}.shared_expert_gate.weight`                 `[1, hidden]`
///
/// Routed experts ship in one of two layouts:
///   • per-expert: `{mlp}.experts.{i}.{gate,up,down}_proj.weight` — one 2D
///     matrix per projection per expert (tiny-random smokes / some HF
///     exports). Names via [`Self::expert_gate_proj`] and siblings.
///   • stacked+fused: `{mlp}.experts.gate_up_proj`
///     `[E, 2*moe_inter, hidden]` (gate ‖ up fused on the output axis: gate =
///     rows `[0, moe_inter)`, up = rows `[moe_inter, 2*moe_inter)`) +
///     `{mlp}.experts.down_proj` `[E, hidden, moe_inter]`. Both are HF
///     `nn.Parameter`s, stored WITHOUT a `.weight` suffix — verified against
///     the production Qwen3.6-35B-A3B safetensors index (BF16, E=256,
///     moe_inter=512, hidden=2048; gate-first row order proven e2e).
///
/// The legacy mlx-lm `switch_mlp.*` stacked convention is NOT part of this
/// contract (loaders reject it loudly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35MoeTensorNames {
    pub mlp_prefix: String,
    pub router_gate: String,
    pub shared_expert_gate_proj: String,
    pub shared_expert_up_proj: String,
    pub shared_expert_down_proj: String,
    pub shared_expert_gate: String,
    pub experts_stacked_gate_up_proj: String,
    pub experts_stacked_down_proj: String,
}

impl Qwen35MoeTensorNames {
    pub fn expert_gate_proj(&self, expert_idx: usize) -> String {
        format!("{}.experts.{expert_idx}.gate_proj.weight", self.mlp_prefix)
    }

    pub fn expert_up_proj(&self, expert_idx: usize) -> String {
        format!("{}.experts.{expert_idx}.up_proj.weight", self.mlp_prefix)
    }

    pub fn expert_down_proj(&self, expert_idx: usize) -> String {
        format!("{}.experts.{expert_idx}.down_proj.weight", self.mlp_prefix)
    }

    /// TP shard contract for the MoE block. The router gate and the shared
    /// expert's sigmoid gate stay replicated (routing must be computed
    /// identically on every rank); the shared expert is column/row-sharded
    /// like a dense MLP. Routed experts return `None`: EP assigns whole
    /// experts by index (the executor's `ExpertSplit`), it never TP-slices
    /// an expert matrix.
    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.router_gate || name == self.shared_expert_gate {
            return Some(Shard::Replicated);
        }
        if name == self.shared_expert_gate_proj || name == self.shared_expert_up_proj {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.shared_expert_down_proj {
            return Some(Shard::Row { dim: 1 });
        }
        None
    }
}

impl Qwen35CommonLayerTensorNames {
    pub fn moe_tensor_names(&self) -> Qwen35MoeTensorNames {
        let mlp = &self.mlp_prefix;
        Qwen35MoeTensorNames {
            mlp_prefix: mlp.clone(),
            router_gate: format!("{mlp}.gate.weight"),
            shared_expert_gate_proj: format!("{mlp}.shared_expert.gate_proj.weight"),
            shared_expert_up_proj: format!("{mlp}.shared_expert.up_proj.weight"),
            shared_expert_down_proj: format!("{mlp}.shared_expert.down_proj.weight"),
            shared_expert_gate: format!("{mlp}.shared_expert_gate.weight"),
            experts_stacked_gate_up_proj: format!("{mlp}.experts.gate_up_proj"),
            experts_stacked_down_proj: format!("{mlp}.experts.down_proj"),
        }
    }

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
    /// Returns `None` for any name not part of a transformer layer; callers
    /// fall back to `Shard::Replicated`. Global tensors live on
    /// `Qwen35Config::shard_for_global_tensor`.
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
    #[serde(default)]
    mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    mrope_interleaved: Option<bool>,
    /// HF `rope_parameters.rope_type` (newer exports; older ones omit it).
    /// `None` / `"default"` ⇒ vanilla RoPE. Scaled types populate
    /// [`Qwen35Config::rope_scaling`] so downstream guards
    /// (e.g. infer-cuda's `ensure!(rope_scaling.is_none())`) stay live.
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(default)]
    factor: Option<f32>,
    #[serde(default)]
    original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    beta_fast: Option<f32>,
    #[serde(default)]
    beta_slow: Option<f32>,
    #[serde(default)]
    attention_factor: Option<f32>,
    #[serde(default)]
    mscale: Option<f32>,
}

impl RopeParameters {
    fn rotary_dim(&self, head_dim: usize) -> Result<usize> {
        if !self.partial_rotary_factor.is_finite()
            || self.partial_rotary_factor <= 0.0
            || self.partial_rotary_factor > 1.0
        {
            return Err(Qwen35ConfigError::InvalidConfig(
                "partial_rotary_factor must be finite and in (0, 1]",
            ));
        }
        let scaled = head_dim as f64 * self.partial_rotary_factor as f64;
        let rounded = scaled.round();
        let tolerance = 4.0 * f32::EPSILON as f64 * head_dim as f64;
        if (scaled - rounded).abs() > tolerance || rounded > usize::MAX as f64 {
            return Err(Qwen35ConfigError::InvalidConfig(
                "head_dim * partial_rotary_factor must be an integer",
            ));
        }
        let rotary_dim = rounded as usize;
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
            return Err(Qwen35ConfigError::InvalidConfig(
                "rotary_dim must be even and non-zero",
            ));
        }
        match (&self.mrope_section, self.mrope_interleaved) {
            (None, None) => {}
            (Some(section), Some(true)) => {
                let sum = section.iter().try_fold(0usize, |sum, &width| {
                    (width > 0)
                        .then_some(width)
                        .and_then(|width| sum.checked_add(width))
                });
                if section.is_empty() || sum != Some(rotary_dim / 2) {
                    return Err(Qwen35ConfigError::InvalidConfig(
                        "mrope_section must contain positive widths summing to rotary_dim / 2",
                    ));
                }
            }
            _ => {
                return Err(Qwen35ConfigError::InvalidConfig(
                    "mrope_section requires mrope_interleaved=true",
                ));
            }
        }
        Ok(rotary_dim)
    }

    /// Resolve the long-context scaling config from the flat HF
    /// `rope_parameters` fields. `rope_type` absent or `"default"` ⇒ `None`
    /// (behavior unchanged for vanilla checkpoints); unsupported types are a
    /// loud error rather than a silent drop.
    fn rope_scaling(&self) -> Result<Option<RopeScalingConfig>> {
        match self.rope_type.as_deref() {
            None | Some("default") => Ok(None),
            Some("yarn") => {
                let factor = self.factor.ok_or(Qwen35ConfigError::InvalidConfig(
                    "rope_parameters.rope_type=yarn requires `factor`",
                ))?;
                let original_max_position_embeddings = self
                    .original_max_position_embeddings
                    .ok_or(Qwen35ConfigError::InvalidConfig(
                        "rope_parameters.rope_type=yarn requires \
                         `original_max_position_embeddings`",
                    ))?;
                Ok(Some(RopeScalingConfig::Yarn {
                    factor,
                    original_max_position_embeddings,
                    beta_fast: self.beta_fast.unwrap_or_else(default_yarn_beta_fast),
                    beta_slow: self.beta_slow.unwrap_or_else(default_yarn_beta_slow),
                    attention_factor: self.attention_factor,
                    mscale: self.mscale.unwrap_or_else(default_yarn_mscale),
                }))
            }
            Some("linear") => {
                let factor = self.factor.ok_or(Qwen35ConfigError::InvalidConfig(
                    "rope_parameters.rope_type=linear requires `factor`",
                ))?;
                Ok(Some(RopeScalingConfig::Linear { factor }))
            }
            Some(_) => Err(Qwen35ConfigError::InvalidConfig(
                "unsupported rope_parameters.rope_type (expected \
                 default / yarn / linear)",
            )),
        }
    }
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

    // Mixture-of-Experts fields (Qwen3.6 / Qwen3_5_Moe).
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
    Nested {
        text_config: TextConfig,
        #[serde(default)]
        tie_word_embeddings: Option<bool>,
    },
    Flat(TextConfig),
}

impl RawConfig {
    fn into_text(self) -> TextConfig {
        match self {
            Self::Nested {
                mut text_config,
                tie_word_embeddings,
            } => {
                if let Some(tie_word_embeddings) = tie_word_embeddings {
                    text_config.tie_word_embeddings = tie_word_embeddings;
                }
                text_config
            }
            Self::Flat(text_config) => text_config,
        }
    }
}

fn default_tie_word_embeddings() -> bool {
    true
}

/// Long-context RoPE scaling config (HF `rope_scaling` schema). `None` ⇒
/// vanilla RoPE with `rope_theta` base.
///
/// Mirror of `qwen3_spec::RopeScalingConfig`; duplicated per-crate to avoid a
/// new shared rope-spec crate.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RopeScalingConfig {
    /// YARN scaling (Peng et al. 2023).
    Yarn {
        factor: f32,
        original_max_position_embeddings: usize,
        #[serde(default = "default_yarn_beta_fast")]
        beta_fast: f32,
        #[serde(default = "default_yarn_beta_slow")]
        beta_slow: f32,
        #[serde(default)]
        attention_factor: Option<f32>,
        #[serde(default = "default_yarn_mscale")]
        mscale: f32,
    },
    /// Linear position interpolation (Chen et al. 2023).
    Linear { factor: f32 },
    /// NTK-aware scaling (kaiokendev 2023).
    NtkAware { factor: f32 },
}

fn default_yarn_beta_fast() -> f32 {
    32.0
}
fn default_yarn_beta_slow() -> f32 {
    1.0
}
fn default_yarn_mscale() -> f32 {
    1.0
}

pub fn compute_scaled_inv_freq(
    head_dim: usize,
    base: f32,
    scaling: Option<&RopeScalingConfig>,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let vanilla: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(i as f32 * 2.0 / head_dim as f32))
        .collect();
    match scaling {
        None => vanilla,
        Some(RopeScalingConfig::Linear { factor }) => {
            vanilla.into_iter().map(|f| f / factor).collect()
        }
        Some(RopeScalingConfig::NtkAware { factor }) => {
            let exponent = (head_dim as f32) / (head_dim as f32 - 2.0);
            let scaled_base = base * factor.powf(exponent);
            (0..half_dim)
                .map(|i| 1.0 / scaled_base.powf(i as f32 * 2.0 / head_dim as f32))
                .collect()
        }
        Some(RopeScalingConfig::Yarn {
            factor,
            original_max_position_embeddings,
            beta_fast,
            beta_slow,
            ..
        }) => {
            let max_pos = *original_max_position_embeddings as f32;
            let low_freq_wavelen = max_pos / beta_fast;
            let high_freq_wavelen = max_pos / beta_slow;
            vanilla
                .into_iter()
                .map(|freq| {
                    let wavelen = std::f32::consts::TAU / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / factor
                    } else {
                        let smooth = (max_pos / wavelen - beta_slow) / (beta_fast - beta_slow);
                        smooth * freq + (1.0 - smooth) * (freq / factor)
                    }
                })
                .collect()
        }
    }
}

/// YARN attention-score scaling factor (Peng et al. 2023 §3.4).
pub fn compute_attention_factor(scaling: Option<&RopeScalingConfig>) -> f32 {
    match scaling {
        Some(RopeScalingConfig::Yarn {
            factor,
            attention_factor,
            mscale,
            ..
        }) => attention_factor.unwrap_or_else(|| 1.0 + 0.1 * mscale * factor.ln()),
        _ => 1.0,
    }
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
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    pub partial_rotary_factor: f32,
    pub rotary_dim: usize,
    pub rope_cache_len_hint: Option<usize>,
    pub layer_types: Vec<LayerType>,

    // Mixture-of-Experts (Qwen3.6 / Qwen3_5_Moe).
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

    /// Whether full-attention Q projection carries a per-head sigmoid gate
    /// fused into the projection output. Qwen3.5 / Qwen3.6 ship with the
    /// gate (so `q_proj` rows = `num_heads * head_dim * 2`); vanilla Qwen3
    /// (0.6B / 1.7B / 4B / 8B) does not (`q_proj` rows = `num_heads * head_dim`).
    /// Default is `true` to preserve back-compat with existing Qwen3.5/3.6
    /// callers; `qwen35_loader` flips it to `false` when it detects the
    /// flat HF Qwen3 config schema.
    #[serde(default = "default_full_attn_gated")]
    pub full_attn_gated: bool,
}

fn default_full_attn_gated() -> bool {
    true
}

impl Qwen35Config {
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

    /// Train-side contract for LoRA / frozen-eval Qwen3.5/Qwen3.6: dense full-attn,
    /// hybrid linear-attn, and MoE configs are allowed because the base is
    /// frozen and only LoRA adapters train.
    pub fn validate_train_lora_or_frozen_contract(&self) -> Result<()> {
        self.validate()?;
        if self.rope_cache_len_hint.is_none() {
            return Err(Qwen35ConfigError::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ));
        }
        if self.is_moe()
            && (self.num_experts_per_tok == 0
                || self.moe_intermediate_size == 0
                || self.shared_expert_intermediate_size == 0)
        {
            return Err(Qwen35ConfigError::InvalidConfig(
                "train-side qwen3.6 MoE LoRA requires non-zero expert dimensions",
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
        if self.tie_word_embeddings {
            self.embed_tokens_tensor_name()
        } else {
            // Untied checkpoints (Qwen3.6-35B-A3B et al.) store the head as a
            // TOP-LEVEL `lm_head.weight` — no `model.language_model` prefix
            // (`shard_for_global_tensor` already matches this literal).
            "lm_head.weight"
        }
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

    fn from_text_config(text: TextConfig, stop_token_ids: Vec<u32>) -> Result<Self> {
        let rotary_dim = text.rope_parameters.rotary_dim(text.head_dim)?;
        let rope_scaling = text.rope_parameters.rope_scaling()?;

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
            rope_scaling,
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
            // qwen35-spec's HF parser is the canonical Qwen3.5 / Qwen3.6 path,
            // where the full-attention block carries the per-head gate. The
            // `qwen35_loader` train-side path flips this to `false` when it
            // detects vanilla Qwen3 (flat-config schema, no `text_config`).
            full_attn_gated: true,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Mirrors the mlx-lm `qwen3_5_moe.py` selection rule; layer ids are
    /// 1-indexed (`idx + 1`).
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

    pub fn load_stop_token_ids(model_dir: impl AsRef<Path>, fallback_eos: u32) -> Result<Vec<u32>> {
        let generation_config_path = model_dir.as_ref().join("generation_config.json");
        let ids = match fs::read_to_string(&generation_config_path) {
            Ok(content) => {
                let value: serde_json::Value = serde_json::from_str(&content)?;
                let ids: Vec<u32> = match value.get("eos_token_id") {
                    Some(serde_json::Value::Number(n)) => {
                        n.as_u64().into_iter().map(|id| id as u32).collect()
                    }
                    Some(serde_json::Value::Array(arr)) => arr
                        .iter()
                        .filter_map(|v| v.as_u64().map(|id| id as u32))
                        .collect(),
                    _ => vec![],
                };
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
        if self.full_attn_gated {
            self.num_attention_heads * self.head_dim * 2
        } else {
            self.num_attention_heads * self.head_dim
        }
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

    /// Tensor names for the NextN-MTP draft head (Qwen3.6 speculative decode).
    /// The head's single transformer layer is ALWAYS full attention, regardless
    /// of the base `layer_types` mix. Top-level `mtp.*` per the checkpoint.
    pub fn mtp_tensor_names(&self) -> Qwen35MtpTensorNames {
        let lp = "mtp.layers.0".to_string();
        let mlp_prefix = format!("{lp}.mlp");
        let ap = format!("{lp}.self_attn");
        Qwen35MtpTensorNames {
            fc: "mtp.fc.weight".to_string(),
            pre_fc_norm_embedding: "mtp.pre_fc_norm_embedding.weight".to_string(),
            pre_fc_norm_hidden: "mtp.pre_fc_norm_hidden.weight".to_string(),
            norm: "mtp.norm.weight".to_string(),
            layer: Qwen35LayerTensorNames {
                common: Qwen35CommonLayerTensorNames {
                    layer_prefix: lp.clone(),
                    mlp_prefix: mlp_prefix.clone(),
                    input_layernorm: format!("{lp}.input_layernorm.weight"),
                    post_attention_layernorm: format!("{lp}.post_attention_layernorm.weight"),
                    mlp_gate_proj: format!("{mlp_prefix}.gate_proj.weight"),
                    mlp_up_proj: format!("{mlp_prefix}.up_proj.weight"),
                    mlp_down_proj: format!("{mlp_prefix}.down_proj.weight"),
                },
                attention: Qwen35AttentionTensorNames::Full(Qwen35FullAttentionTensorNames {
                    attention_prefix: ap.clone(),
                    q_proj: format!("{ap}.q_proj.weight"),
                    k_proj: format!("{ap}.k_proj.weight"),
                    v_proj: format!("{ap}.v_proj.weight"),
                    o_proj: format!("{ap}.o_proj.weight"),
                    q_norm: format!("{ap}.q_norm.weight"),
                    k_norm: format!("{ap}.k_norm.weight"),
                }),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsparkLayerType {
    Full,
    /// Sliding-window attention; the window comes from
    /// `DsparkConfig::sliding_window`, which the config must then declare.
    Sliding,
}

/// DSpark block-draft config, parsed from the DRAFT checkpoint dir's
/// `config.json`. Covers both checkpoint flavors:
/// - **DFlash** (z-lab, `architectures: [..DFlash..]`): same-position
///   denoising — block rows `1..block_size` fill their OWN positions, so a
///   block proposes `block_size - 1` draft tokens.
/// - **DSpark** (DeepSpec, `architectures: [..DSpark..]`): next-token labels —
///   the anchor row is the first prediction position, so a block proposes up
///   to `block_size` draft tokens. Markov/confidence heads are detected from
///   the safetensors, not the config. Either flavor may nest
///   `mask_token_id`/`target_layer_ids` under `dflash_config` or keep them
///   top-level.
#[derive(Debug, Clone, PartialEq)]
pub struct DsparkConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Long-context scaling the drafter was trained with. Dropping it makes the
    /// draft's positions disagree with the target's from layer one, and the
    /// verify step then rejects nearly everything -- a `yarn` drafter read as
    /// vanilla measured 13% acceptance falling to 0% at c=8.
    pub rope_scaling: Option<RopeScalingConfig>,
    /// `None` = no window. DeepSpec draft configs declare no `sliding_window`
    /// at all, so substituting one gives a converted checkpoint a reach it was
    /// never trained with.
    pub sliding_window: Option<usize>,
    pub layer_types: Vec<DsparkLayerType>,
    pub block_size: usize,
    pub mask_token_id: u32,
    /// Trunk layers whose residual-stream OUTPUT is tapped (`-1` = the
    /// embedding output, i.e. the residual stream before layer 0).
    pub target_layer_ids: Vec<i64>,
    /// `true` = DSpark next-token proposal; `false` = DFlash same-position.
    pub next_token_heads: bool,
}

impl DsparkConfig {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(dir.as_ref().join("config.json"))?;
        let v: serde_json::Value = serde_json::from_str(&content)?;
        let usize_of = |name: &'static str| -> Result<usize> {
            v.get(name)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .ok_or(Qwen35ConfigError::InvalidConfig(name))
        };
        let f32_of = |name: &'static str, default: f32| -> f32 {
            v.get(name)
                .and_then(serde_json::Value::as_f64)
                .map_or(default, |f| f as f32)
        };
        // The proposal convention travels as `speculative_tokens`
        // (`block_size - 1` = same-position DFlash rows). Checkpoints converted
        // before that field existed encode the same bit as the "dflash_config"
        // nesting — see scripts/convert_dspark_speculators.py. `architectures`
        // is NOT a discriminator here: the converter hardcodes "DSparkDraftModel".
        let dflash = v.get("dflash_config").filter(|d| d.is_object());
        let field = |name: &str| dflash.and_then(|d| d.get(name)).or_else(|| v.get(name));
        let mask_token_id = field("mask_token_id")
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as u32)
            .ok_or(Qwen35ConfigError::InvalidConfig("mask_token_id"))?;
        let target_layer_ids: Vec<i64> = field("target_layer_ids")
            .and_then(serde_json::Value::as_array)
            .map(|a| a.iter().filter_map(serde_json::Value::as_i64).collect())
            .ok_or(Qwen35ConfigError::InvalidConfig("target_layer_ids"))?;
        if target_layer_ids.is_empty() {
            return Err(Qwen35ConfigError::InvalidConfig("empty target_layer_ids"));
        }
        let num_hidden_layers = usize_of("num_hidden_layers")?;
        let layer_types = match v.get("layer_types").and_then(serde_json::Value::as_array) {
            Some(a) => a
                .iter()
                .map(|t| match t.as_str() {
                    Some("sliding_attention") => Ok(DsparkLayerType::Sliding),
                    Some("full_attention") => Ok(DsparkLayerType::Full),
                    _ => Err(Qwen35ConfigError::InvalidConfig("layer_types")),
                })
                .collect::<Result<Vec<_>>>()?,
            None => vec![DsparkLayerType::Full; num_hidden_layers],
        };
        if layer_types.len() != num_hidden_layers {
            return Err(Qwen35ConfigError::InvalidConfig(
                "layer_types length != num_hidden_layers",
            ));
        }
        let sliding_window = v
            .get("sliding_window")
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize);
        if sliding_window.is_none() && layer_types.contains(&DsparkLayerType::Sliding) {
            return Err(Qwen35ConfigError::InvalidConfig(
                "sliding_attention layers without sliding_window",
            ));
        }
        let block_size = field("block_size")
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize)
            .ok_or(Qwen35ConfigError::InvalidConfig("block_size"))?;
        // transformers >= 5.12 nests the RoPE base and its scaling under
        // `rope_parameters`; older exports keep `rope_theta` at the top level
        // and carry no scaling. Reading only the top level silently defaults
        // the base AND discards the scaling.
        let rope = v.get("rope_parameters").filter(|r| r.is_object());
        let rope_num = |name: &str| -> Option<f64> {
            rope.and_then(|r| r.get(name))
                .and_then(serde_json::Value::as_f64)
        };
        let rope_theta = v
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .or_else(|| rope_num("rope_theta"))
            .map_or(1e7, |f| f as f32);
        let rope_scaling = match rope
            .and_then(|r| r.get("rope_type"))
            .and_then(serde_json::Value::as_str)
        {
            None | Some("default") => None,
            Some("yarn") => Some(RopeScalingConfig::Yarn {
                factor: rope_num("factor").ok_or(Qwen35ConfigError::InvalidConfig(
                    "dspark rope_parameters.rope_type=yarn requires `factor`",
                ))? as f32,
                original_max_position_embeddings: rope_num("original_max_position_embeddings")
                    .ok_or(Qwen35ConfigError::InvalidConfig(
                        "dspark rope_parameters.rope_type=yarn requires \
                         `original_max_position_embeddings`",
                    ))? as usize,
                beta_fast: rope_num("beta_fast").map_or_else(default_yarn_beta_fast, |f| f as f32),
                beta_slow: rope_num("beta_slow").map_or_else(default_yarn_beta_slow, |f| f as f32),
                attention_factor: rope_num("attention_factor").map(|f| f as f32),
                mscale: rope_num("mscale").map_or_else(default_yarn_mscale, |f| f as f32),
            }),
            // Loud, like the trunk's resolver: a scaling we cannot reproduce
            // makes every draft wrong, and silence looks like a weak drafter.
            Some(_) => {
                return Err(Qwen35ConfigError::InvalidConfig(
                    "dspark rope_parameters.rope_type is not one this drafter path supports",
                ));
            }
        };
        Ok(Self {
            hidden_size: usize_of("hidden_size")?,
            intermediate_size: usize_of("intermediate_size")?,
            num_hidden_layers,
            num_attention_heads: usize_of("num_attention_heads")?,
            num_key_value_heads: usize_of("num_key_value_heads")?,
            head_dim: usize_of("head_dim")?,
            rms_norm_eps: f32_of("rms_norm_eps", 1e-6),
            rope_theta,
            rope_scaling,
            sliding_window,
            layer_types,
            block_size,
            mask_token_id,
            target_layer_ids,
            next_token_heads: v
                .get("speculative_tokens")
                .and_then(serde_json::Value::as_u64)
                .map_or(dflash.is_none(), |n| n as usize != block_size - 1),
        })
    }

    /// Draft tokens one block proposes: next-token heads draft from every block
    /// row; same-position (DFlash) row 0 carries the already-known anchor.
    #[must_use]
    pub fn max_draft_tokens(&self) -> usize {
        if self.next_token_heads {
            self.block_size
        } else {
            self.block_size - 1
        }
    }
}

/// Draft-keep policy for [`dspark_verify_lens`]: the additive verify-step cost
/// model `step_ms = bias + row · verify_rows` (the same shape sglang profiles
/// for its DSpark planner). Cost defaults are H20
/// ThinkingCap-27B c=16 measurements (trunk 116 + draft 95 fixed, 0.53/row).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DsparkSps {
    pub bias_ms: f32,
    pub row_ms: f32,
}

impl Default for DsparkSps {
    fn default() -> Self {
        Self {
            bias_ms: 211.0,
            row_ms: 0.53,
        }
    }
}

/// DSpark §3.2.2 / sglang `compute_verify_token_budget` + `verify_lens_topk`:
/// per-request draft-keep lengths maximizing goodput
/// `Θ(B) = (R + Σ top-B survival) / (bias + row·(R + B))`.
/// `survivals[r][j]` = calibrated `P(first j+1 drafts of request r all accept)`
/// (cumprod of per-position confidence, monotone decreasing). B=0 is an arm,
/// so the result never predicts worse than not speculating. Survival is
/// monotone per request, so the global admission cut yields prefix lengths.
pub fn dspark_verify_lens(survivals: &[&[f32]], sps: DsparkSps) -> Vec<usize> {
    const EPS: f32 = 1e-6;
    let r = survivals.len() as f32;
    let mut all: Vec<f32> = survivals
        .iter()
        .flat_map(|s| s.iter().copied())
        .filter(|p| *p >= EPS)
        .collect();
    all.sort_unstable_by(|a, b| b.total_cmp(a));
    let mut best = r / (sps.bias_ms + sps.row_ms * r);
    let mut cut = f32::INFINITY;
    let mut sum = 0.0f32;
    for (i, &p) in all.iter().enumerate() {
        sum += p;
        let theta = (r + sum) / (sps.bias_ms + sps.row_ms * (r + (i + 1) as f32));
        if theta > best {
            (best, cut) = (theta, p);
        }
    }
    survivals
        .iter()
        .map(|s| s.iter().take_while(|p| **p >= cut).count())
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkLayerTensorNames {
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub q_norm: String,
    pub k_norm: String,
    pub input_layernorm: String,
    pub post_attention_layernorm: String,
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

/// DSpark draft checkpoint tensor names. `fc`/`hidden_norm`/`norm` + per-layer
/// tensors are the always-present backbone; the markov + confidence entries
/// name OPTIONAL heads (absent in z-lab DFlash checkpoints — probe with
/// `has_tensor` before loading). Embeddings + lm_head are SHARED with the
/// trunk and not listed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkTensorNames {
    pub fc: String,
    pub hidden_norm: String,
    pub norm: String,
    pub markov_w1: String,
    pub markov_w2: String,
    /// Present only in gated/RNN markov variants — probed to fail closed
    /// (vanilla w1/w2 is the only wired kind).
    pub markov_gate_proj: String,
    pub markov_joint_proj: String,
    pub confidence_weight: String,
    pub confidence_bias: String,
    pub layers: Vec<DsparkLayerTensorNames>,
}

#[must_use]
pub fn dspark_tensor_names(num_layers: usize) -> DsparkTensorNames {
    let layers = (0..num_layers)
        .map(|i| {
            let attn = format!("layers.{i}.self_attn");
            let mlp = format!("layers.{i}.mlp");
            DsparkLayerTensorNames {
                q_proj: format!("{attn}.q_proj.weight"),
                k_proj: format!("{attn}.k_proj.weight"),
                v_proj: format!("{attn}.v_proj.weight"),
                o_proj: format!("{attn}.o_proj.weight"),
                q_norm: format!("{attn}.q_norm.weight"),
                k_norm: format!("{attn}.k_norm.weight"),
                input_layernorm: format!("layers.{i}.input_layernorm.weight"),
                post_attention_layernorm: format!("layers.{i}.post_attention_layernorm.weight"),
                gate_proj: format!("{mlp}.gate_proj.weight"),
                up_proj: format!("{mlp}.up_proj.weight"),
                down_proj: format!("{mlp}.down_proj.weight"),
            }
        })
        .collect();
    DsparkTensorNames {
        fc: "fc.weight".to_string(),
        hidden_norm: "hidden_norm.weight".to_string(),
        norm: "norm.weight".to_string(),
        markov_w1: "markov_head.markov_w1.weight".to_string(),
        markov_w2: "markov_head.markov_w2.weight".to_string(),
        markov_gate_proj: "markov_head.gate_proj.weight".to_string(),
        markov_joint_proj: "markov_head.joint_proj.weight".to_string(),
        confidence_weight: "confidence_head.proj.weight".to_string(),
        confidence_bias: "confidence_head.proj.bias".to_string(),
        layers,
    }
}
