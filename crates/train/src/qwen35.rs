use std::{
    collections::{HashMap, HashSet},
    f32::consts::TAU,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use autograd::{
    AutogradError, Device, Tape, Tensor, TensorId, TensorStore,
    ops::{
        LinearAttentionParams, MoeGroupedLinearExpert, MoeGroupedLinearInput, MoeGroupedRoute,
        MoeTopK, add, add_broadcast, cat_seq, causal_sdpa_recompute,
        causal_sdpa_recompute_with_q_start, causal_sdpa_with_q_start, checkpoint_sequential,
        embedding, linear_attention_boundary, linear_attention_core,
        linear_attention_core_with_carry, linear_attention_core_with_carry_taped,
        linear_attention_ctx_bytes, linear_attention_row_transient_bytes, matmul_bt_with_site,
        moe_grouped_linear, moe_grouped_weighted_scatter, moe_topk_softmax,
        moe_topk_softmax_with_indices, mul, repeat_kv, reshape, rmsnorm, rope, sigmoid, silu,
        slice, transpose,
    },
    tape::checkpoint_replay_mem_stage,
};
pub use qwen35_spec::{LayerType, Qwen35Config, Qwen35ConfigError};
use qwen35_spec::{Qwen35AttentionTensorNames, Qwen35MoeTensorNames};
use thiserror::Error;

use crate::lora::{
    LinearWithLora, LoraConfig, LoraTargetSet, leak_name, next_uniform, seed_from_name,
};
use crate::tensor_parallel::{TpContext, maybe_all_reduce};

#[path = "qwen35/attention_full.rs"]
mod attention_full;
#[path = "qwen35/attention_linear.rs"]
mod attention_linear;
#[path = "qwen35/checkpoint_policy.rs"]
mod checkpoint_policy;
#[path = "qwen35/construct.rs"]
mod construct;
#[path = "qwen35/diagnostics.rs"]
mod diagnostics;
#[path = "qwen35/forward.rs"]
mod forward;
#[path = "qwen35/frozen_prefix.rs"]
mod frozen_prefix;
#[path = "qwen35/kv_cache.rs"]
mod kv_cache;
#[path = "qwen35/layer.rs"]
mod layer;
#[path = "qwen35/mlp.rs"]
mod mlp;
#[path = "qwen35/params.rs"]
mod params;
#[path = "qwen35/profile.rs"]
mod profile;
#[path = "qwen35/rollout.rs"]
mod rollout;
#[path = "qwen35/tensor_ops.rs"]
mod tensor_ops;
#[cfg(test)]
#[path = "qwen35/tests.rs"]
mod tests;
#[path = "qwen35/tp_dims.rs"]
mod tp_dims;

use checkpoint_policy::*;
use kv_cache::*;
use params::*;
use profile::*;
use tensor_ops::*;
use tp_dims::*;

pub use profile::{
    Qwen35AttentionForwardProfile, Qwen35LayerForwardProfile, Qwen35RolloutForwardProfile,
};
pub use rollout::{
    forward_rollout_cached, forward_rollout_cached_device_token,
    forward_rollout_cached_device_token_profiled, forward_rollout_cached_profiled,
};
pub(crate) use tensor_ops::qwen35_to_autograd;

#[derive(Debug, Error)]
pub enum Qwen35Error {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Config(#[from] Qwen35ConfigError),
    #[error("invalid qwen3.5 config: {0}")]
    InvalidConfig(&'static str),
    #[error("input_ids len {input_len} does not match expected {expected_len}")]
    InputLenMismatch {
        input_len: usize,
        expected_len: usize,
    },
    #[error("position id {position} is out of bounds for rope cache size {upper}")]
    PositionOutOfBounds { position: usize, upper: usize },
}

pub type Result<T> = std::result::Result<T, Qwen35Error>;

#[derive(Debug, Clone)]
struct Qwen35FullAttention {
    q_proj: LinearWithLora,
    k_proj: LinearWithLora,
    v_proj: LinearWithLora,
    o_proj: LinearWithLora,
    q_norm: TensorId,
    k_norm: TensorId,
}

#[derive(Debug, Clone)]
struct Qwen35LinearAttention {
    in_proj_qkv: LinearWithLora,
    in_proj_z: LinearWithLora,
    in_proj_b: LinearWithLora,
    in_proj_a: LinearWithLora,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    norm: TensorId,
    out_proj: LinearWithLora,
}

#[derive(Debug, Clone)]
enum Qwen35Attention {
    Full(Qwen35FullAttention),
    Linear(Qwen35LinearAttention),
}

/// OPD frozen-prompt-KV: a full-attention layer's captured prompt-prefix K/V
/// (repeat_kv'd, at absolute positions `0..gen_start`, `requires_grad=false`).
/// Only K/V are captured — the prompt's Q is never queried by the gen segment.
#[derive(Debug, Clone, Copy)]
struct PrefixKv {
    k: TensorId,
    v: TensorId,
}

/// OPD frozen-prompt-KV: a linear-attention layer's captured boundary recurrent
/// state + causal-conv window after the prompt prefix (`requires_grad=false`).
#[derive(Debug, Clone, Copy)]
struct PrefixState {
    state: TensorId,
    conv_window: TensorId,
}

#[derive(Debug, Clone, Copy)]
enum LayerPrefix {
    Full(PrefixKv),
    Linear(PrefixState),
}

#[derive(Debug, Clone)]
struct WritebackPrefixCache {
    layers: Vec<LayerPrefix>,
}

#[derive(Debug, Clone)]
struct Qwen35DenseMlp {
    gate_proj: LinearWithLora,
    up_proj: LinearWithLora,
    down_proj: LinearWithLora,
}

#[derive(Debug, Clone)]
struct Qwen35SparseExpert {
    gate_proj: LinearWithLora,
    up_proj: LinearWithLora,
    down_proj: LinearWithLora,
}

#[derive(Debug, Clone)]
struct Qwen35SparseMlp {
    router_gate: LinearWithLora,
    shared_gate_proj: LinearWithLora,
    shared_up_proj: LinearWithLora,
    shared_down_proj: LinearWithLora,
    shared_expert_gate: LinearWithLora,
    experts: Vec<Qwen35SparseExpert>,
    top_k: usize,
}

#[derive(Debug, Clone)]
enum Qwen35Mlp {
    Dense(Box<Qwen35DenseMlp>),
    Sparse(Box<Qwen35SparseMlp>),
}

#[derive(Debug, Clone)]
struct Qwen35Layer {
    index: usize,
    input_layernorm: TensorId,
    self_attn: Qwen35Attention,
    post_attention_layernorm: TensorId,
    mlp: Qwen35Mlp,
}

#[derive(Debug, Clone)]
struct Qwen35LayerKvCache {
    k: Option<TensorId>,
    v: Option<TensorId>,
    max_seq_len: usize,
    seq_cursor: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen35KvCache {
    layers: Vec<Qwen35LayerKvCache>,
    seq_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceWindow {
    pub start: usize,
    pub end: usize,
}

impl SequenceWindow {
    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(self) -> bool {
        self.end <= self.start
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35MoeRouteSignature {
    pub layer: usize,
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
    pub indices: Vec<usize>,
}

/// How a sparse MLP obtains its top-k route: compute it, compute and record it,
/// or replay a recorded one.
enum MoeRouteMode<'a> {
    Free,
    Collect(&'a mut Vec<Qwen35MoeRouteSignature>),
    Frozen {
        signatures: &'a [Qwen35MoeRouteSignature],
        next: &'a mut usize,
    },
}

#[derive(Debug, Clone)]
pub struct Qwen35Model {
    config: Qwen35Config,
    tp: TpContext,
    lora: Option<LoraConfig>,
    lora_target_set: LoraTargetSet,
    lora_layer_start: Option<usize>,
    lora_skip_experts: bool,
    layers: Vec<Qwen35Layer>,
    embed_tokens: TensorId,
    final_norm: TensorId,
    lm_head: TensorId,
    cos_cache: TensorId,
    sin_cache: TensorId,
    param_names: HashMap<&'static str, TensorId>,
    adapter_names: HashMap<&'static str, TensorId>,
    param_ids: Vec<TensorId>,
    gradient_checkpointing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35InitMode {
    ScratchTrain,
    LoraOrFrozen { materialize_frozen_base: bool },
}
