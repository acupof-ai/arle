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

/// Per-layer captured prompt prefix for the frozen-prompt-KV writeback split.
#[derive(Debug, Clone, Copy)]
enum LayerPrefix {
    Full(PrefixKv),
    Linear(PrefixState),
}

/// One captured prefix per model layer (phase-1 prompt pass output), consumed by
/// the phase-2 taped gen-segment forward.
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

/// Qwen3.5/3.6-specific TP dimension math: apply the mesh-level `TpContext`
/// (from `crate::tensor_parallel`) to this model's head/intermediate dimensions.
/// The coordinate view + `divide`/`maybe_all_reduce` are model-agnostic and live
/// in `tensor_parallel`; only these `Qwen35Config`-reading shard sizes are here.
trait Qwen35TpDims {
    fn validate(self, cfg: &Qwen35Config) -> Result<()>;
    fn local_attention_heads(self, cfg: &Qwen35Config) -> Result<usize>;
    fn local_key_value_heads(self, cfg: &Qwen35Config) -> Result<usize>;
    fn local_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize>;
    fn local_moe_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize>;
    fn local_shared_expert_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize>;
    fn full_attn_q_proj_dim(self, cfg: &Qwen35Config) -> Result<usize>;
    fn full_attn_q_dim(self, cfg: &Qwen35Config) -> Result<usize>;
    fn full_attn_kv_dim(self, cfg: &Qwen35Config) -> Result<usize>;
}

impl Qwen35TpDims for TpContext {
    fn validate(self, cfg: &Qwen35Config) -> Result<()> {
        if self.world_size == 0 {
            return Err(Qwen35Error::InvalidConfig(
                "tensor-parallel world size must be non-zero",
            ));
        }
        if self.rank >= self.world_size {
            return Err(Qwen35Error::InvalidConfig(
                "tensor-parallel rank must be smaller than world size",
            ));
        }
        if !self.is_enabled() {
            return Ok(());
        }
        if !cfg
            .layer_types
            .iter()
            .all(|&layer| layer == LayerType::FullAttention)
        {
            return Err(Qwen35Error::InvalidConfig(
                "tensor-parallel train path currently supports full-attention layers only",
            ));
        }
        let _ = self.local_attention_heads(cfg)?;
        let _ = self.local_key_value_heads(cfg)?;
        let _ = self.local_intermediate_size(cfg)?;
        Ok(())
    }

    fn local_attention_heads(self, cfg: &Qwen35Config) -> Result<usize> {
        self.divide(cfg.num_attention_heads)
            .ok_or(Qwen35Error::InvalidConfig(
                "num_attention_heads must divide tensor-parallel world size",
            ))
    }

    fn local_key_value_heads(self, cfg: &Qwen35Config) -> Result<usize> {
        self.divide(cfg.num_key_value_heads)
            .ok_or(Qwen35Error::InvalidConfig(
                "num_key_value_heads must divide tensor-parallel world size",
            ))
    }

    fn local_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize> {
        self.divide(cfg.intermediate_size)
            .ok_or(Qwen35Error::InvalidConfig(
                "intermediate_size must divide tensor-parallel world size",
            ))
    }

    fn local_moe_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize> {
        self.divide(cfg.moe_intermediate_size)
            .ok_or(Qwen35Error::InvalidConfig(
                "moe_intermediate_size must divide tensor-parallel world size",
            ))
    }

    fn local_shared_expert_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize> {
        self.divide(cfg.shared_expert_intermediate_size)
            .ok_or(Qwen35Error::InvalidConfig(
                "shared_expert_intermediate_size must divide tensor-parallel world size",
            ))
    }

    fn full_attn_q_proj_dim(self, cfg: &Qwen35Config) -> Result<usize> {
        let local_heads = self.local_attention_heads(cfg)?;
        Ok(if cfg.full_attn_gated {
            local_heads * cfg.head_dim * 2
        } else {
            local_heads * cfg.head_dim
        })
    }

    fn full_attn_q_dim(self, cfg: &Qwen35Config) -> Result<usize> {
        Ok(self.local_attention_heads(cfg)? * cfg.head_dim)
    }

    fn full_attn_kv_dim(self, cfg: &Qwen35Config) -> Result<usize> {
        Ok(self.local_key_value_heads(cfg)? * cfg.head_dim)
    }
}

fn trace_model_component(trace: bool, component: &'static str, duration: Duration) {
    if trace {
        println!(
            "qwen35_full_forward_trace scope=model component={} seconds={:.6}",
            component,
            duration.as_secs_f64()
        );
    }
}

fn trace_forward_component(
    trace: bool,
    layer_index: usize,
    component: &'static str,
    duration: Duration,
) {
    if trace {
        println!(
            "qwen35_full_forward_trace scope=layer layer={} component={} seconds={:.6}",
            layer_index,
            component,
            duration.as_secs_f64()
        );
    }
}

fn trace_attention_component(
    trace: bool,
    layer_index: usize,
    component: &'static str,
    duration: Duration,
) {
    if trace {
        println!(
            "qwen35_full_forward_trace scope=attention layer={} component={} seconds={:.6}",
            layer_index,
            component,
            duration.as_secs_f64()
        );
    }
}

impl Qwen35KvCache {
    pub fn new(model: &Qwen35Model, max_seq_len: usize) -> Self {
        Self {
            layers: vec![
                Qwen35LayerKvCache {
                    k: None,
                    v: None,
                    max_seq_len,
                    seq_cursor: 0,
                };
                model.layers.len()
            ],
            seq_len: 0,
        }
    }

    pub fn extend_tensor_ids(&self, keep: &mut HashSet<TensorId>) {
        for layer in &self.layers {
            if let Some(k) = layer.k {
                keep.insert(k);
            }
            if let Some(v) = layer.v {
                keep.insert(v);
            }
        }
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct Qwen35AttentionForwardProfile {
    /// Host-side enqueue/API wall-clock attribution for profile harnesses.
    /// CUDA kernel elapsed time still requires NVTX/nsys cross-checking.
    pub q_proj: Duration,
    pub q_layout: Duration,
    pub k_proj: Duration,
    pub v_proj: Duration,
    pub kv_split: Duration,
    pub qk_norm: Duration,
    pub rope: Duration,
    pub repeat_kv: Duration,
    pub append_kv: Duration,
    pub sdpa: Duration,
    pub gate: Duration,
    pub merge: Duration,
    pub o_proj: Duration,
    pub linear_qkv_proj: Duration,
    pub linear_z_proj: Duration,
    pub linear_b_proj: Duration,
    pub linear_a_proj: Duration,
    pub linear_core: Duration,
    pub linear_out_proj: Duration,
}

#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct Qwen35LayerForwardProfile {
    /// Host-side enqueue/API wall-clock attribution for profile harnesses.
    /// CUDA kernel elapsed time still requires NVTX/nsys cross-checking.
    pub input_rmsnorm: Duration,
    pub attention: Duration,
    pub attention_detail: Qwen35AttentionForwardProfile,
    pub attention_residual: Duration,
    pub post_attention_rmsnorm: Duration,
    pub mlp: Duration,
    pub mlp_residual: Duration,
}

#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct Qwen35RolloutForwardProfile {
    pub total: Duration,
    pub cache_select: Duration,
    pub embedding: Duration,
    pub final_norm: Duration,
    pub lm_head: Duration,
    pub layers: Vec<Qwen35LayerForwardProfile>,
}

#[doc(hidden)]
impl Qwen35RolloutForwardProfile {
    pub fn input_rmsnorm_total(&self) -> Duration {
        self.layers.iter().map(|layer| layer.input_rmsnorm).sum()
    }

    pub fn attention_total(&self) -> Duration {
        self.layers.iter().map(|layer| layer.attention).sum()
    }

    pub fn attention_residual_total(&self) -> Duration {
        self.layers
            .iter()
            .map(|layer| layer.attention_residual)
            .sum()
    }

    pub fn post_attention_rmsnorm_total(&self) -> Duration {
        self.layers
            .iter()
            .map(|layer| layer.post_attention_rmsnorm)
            .sum()
    }

    pub fn mlp_total(&self) -> Duration {
        self.layers.iter().map(|layer| layer.mlp).sum()
    }

    pub fn mlp_residual_total(&self) -> Duration {
        self.layers.iter().map(|layer| layer.mlp_residual).sum()
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

#[doc(hidden)]
pub fn forward_rollout_cached(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    input_ids: &[u32],
    position_ids: &[u32],
    cache: &mut Qwen35KvCache,
) -> Result<TensorId> {
    model.forward_rollout_cached(store, tape, input_ids, position_ids, cache)
}

#[doc(hidden)]
pub fn forward_rollout_cached_profiled(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    input_ids: &[u32],
    position_ids: &[u32],
    cache: &mut Qwen35KvCache,
) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
    model.forward_rollout_cached_profiled(store, tape, input_ids, position_ids, cache)
}

#[doc(hidden)]
pub fn forward_rollout_cached_device_token(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    token_id: TensorId,
    position_id: u32,
    cache: &mut Qwen35KvCache,
) -> Result<TensorId> {
    model.forward_rollout_cached_device_token(store, tape, token_id, position_id, cache)
}

#[doc(hidden)]
pub fn forward_rollout_cached_device_token_profiled(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    token_id: TensorId,
    position_id: u32,
    cache: &mut Qwen35KvCache,
) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
    model.forward_rollout_cached_device_token_profiled(store, tape, token_id, position_id, cache)
}

fn qwen35_rmsnorm(
    x: TensorId,
    weight: TensorId,
    eps: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let weight_shape = store
        .get(weight)
        .ok_or(AutogradError::InvalidTensorId(weight))?
        .shape
        .clone();
    if weight_shape.len() != 1 {
        return Err(AutogradError::InvalidRank {
            expected: "rank-1 RMSNorm weight",
            got: weight_shape.len(),
        }
        .into());
    }
    let ones = store.alloc(Tensor::new(
        vec![1.0; weight_shape[0]],
        weight_shape,
        false,
    )?);
    let offset_weight = add(weight, ones, store, tape)?;
    Ok(rmsnorm(x, offset_weight, eps, store, tape)?)
}

impl Qwen35Layer {
    /// This layer's trainable param ids — fed to `checkpoint_sequential`'s
    /// `layer_params` so each group's saved inputs carry every layer's params
    /// (that's how `requires_grad` is true and param grads come back).
    fn checkpoint_param_ids(&self, skip_experts: bool, store: &TensorStore) -> Vec<TensorId> {
        let mut params = Vec::new();
        params.push(self.input_layernorm);
        params.push(self.post_attention_layernorm);
        collect_mlp_ids(&self.mlp, skip_experts, &mut params);
        match &self.self_attn {
            Qwen35Attention::Full(attn) => {
                collect_linear_ids(&attn.q_proj, &mut params);
                collect_linear_ids(&attn.k_proj, &mut params);
                collect_linear_ids(&attn.v_proj, &mut params);
                collect_linear_ids(&attn.o_proj, &mut params);
                params.push(attn.q_norm);
                params.push(attn.k_norm);
            }
            Qwen35Attention::Linear(attn) => {
                collect_linear_ids(&attn.in_proj_qkv, &mut params);
                collect_linear_ids(&attn.in_proj_z, &mut params);
                collect_linear_ids(&attn.in_proj_b, &mut params);
                collect_linear_ids(&attn.in_proj_a, &mut params);
                collect_linear_ids(&attn.out_proj, &mut params);
                params.push(attn.conv1d_weight);
                params.push(attn.dt_bias);
                params.push(attn.a_log);
                params.push(attn.norm);
            }
        }
        params.sort_unstable();
        params.dedup();
        params.retain(|&id| store.get(id).is_some_and(|tensor| tensor.requires_grad));
        params
    }

    fn forward(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cp: crate::context_parallel::CpContext,
        cos: TensorId,
        sin: TensorId,
        cp_positions: Option<&[usize]>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];
        checkpoint_replay_mem_stage(tape, store, "layer_enter");

        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        checkpoint_replay_mem_stage(tape, store, "post_input_norm");
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => self.forward_full_attention(
                h,
                attn,
                cfg,
                tp,
                cp,
                cos,
                sin,
                batch,
                seq_len,
                cp_positions,
                store,
                tape,
            )?,
            Qwen35Attention::Linear(attn) => {
                self.forward_linear_attention(h, attn, cfg, cp, batch, seq_len, store, tape)?
            }
        };
        checkpoint_replay_mem_stage(tape, store, "post_attention");
        let x = add(x, attn_out, store, tape)?;
        checkpoint_replay_mem_stage(tape, store, "post_attention_residual");

        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        checkpoint_replay_mem_stage(tape, store, "post_mlp_norm");
        // MLP is position-wise — chunking is exact.
        let mlp_out = {
            let mut param_ids = Vec::new();
            collect_mlp_ids(&self.mlp, false, &mut param_ids);
            param_ids.retain(|&id| store.get(id).is_some_and(|t| t.requires_grad));
            let layer = self.clone();
            let cfg_owned = cfg.clone();
            autograd::ops::checkpoint_seq_chunked(
                h,
                param_ids,
                crate::runtime_flags::OPD_SEQ_CHUNK,
                store,
                tape,
                move |st, tp_tape, _start, inp| {
                    let shape = st
                        .get(inp[0])
                        .ok_or(AutogradError::InvalidTensorId(inp[0]))?
                        .shape
                        .clone();
                    layer
                        .forward_mlp(inp[0], &cfg_owned, tp, shape[0], shape[1], st, tp_tape)
                        .map_err(qwen35_to_autograd)
                },
            )?
        };
        checkpoint_replay_mem_stage(tape, store, "post_mlp");
        let out = add(x, mlp_out, store, tape)?;
        checkpoint_replay_mem_stage(tape, store, "layer_exit");
        Ok(out)
    }

    /// OPD frozen-prompt-KV phase 1 (off-tape): capture this layer's prompt
    /// prefix K/V (full attention) or boundary state (linear attention) AND
    /// produce the layer's attention+MLP output so the next layer's capture
    /// sees the correct residual stream.
    ///
    /// `prefix_kv` = accumulated K/V from earlier prompt chunks (None for the
    /// first chunk). `chunk_start` = absolute row index of this chunk's first
    /// token (0 for the first chunk). The returned `LayerPrefix` holds the FULL
    /// K/V (prefix + this chunk) for the gen-segment pass. The caller passes a
    /// DISABLED tape.
    #[allow(clippy::too_many_arguments)]
    fn forward_capture_prefix(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        chunk_start: usize,
        prefix_kv: Option<&PrefixKv>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(TensorId, LayerPrefix)> {
        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let (attn_out, prefix) = match &self.self_attn {
            Qwen35Attention::Full(attn) => {
                // K/V for this chunk (RMSNorm + RoPE + repeat_kv, requires_grad=false).
                let chunk_kv = self.forward_full_attention_capture_prefix_kv(
                    h, attn, cfg, tp, cos, sin, batch, seq_len, store, tape,
                )?;
                // Full K/V = accumulated prefix + this chunk.
                let full_kv = if let Some(prefix) = prefix_kv {
                    PrefixKv {
                        k: cat_seq(prefix.k, chunk_kv.k, store, tape)?,
                        v: cat_seq(prefix.v, chunk_kv.v, store, tape)?,
                    }
                } else {
                    chunk_kv
                };
                // Attention: Q from this chunk, K/V = full (prefix + chunk).
                // q_start = chunk_start so causal masking is by absolute position.
                let attn_out = self.forward_full_attention_with_kv(
                    h,
                    attn,
                    cfg,
                    tp,
                    cos,
                    sin,
                    &full_kv,
                    batch,
                    seq_len,
                    chunk_start,
                    store,
                    tape,
                )?;
                (attn_out, LayerPrefix::Full(full_kv))
            }
            Qwen35Attention::Linear(attn) => {
                let prefix = self.forward_linear_attention_capture_prefix_state(
                    h, attn, cfg, batch, seq_len, store, tape,
                )?;
                let attn_out = self.forward_linear_attention(
                    h,
                    attn,
                    cfg,
                    crate::context_parallel::CpContext::single(),
                    batch,
                    seq_len,
                    store,
                    tape,
                )?;
                (attn_out, LayerPrefix::Linear(prefix))
            }
        };
        let x = add(x, attn_out, store, tape)?;
        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let mlp_out = self.forward_mlp(h, cfg, tp, batch, seq_len, store, tape)?;
        let next = add(x, mlp_out, store, tape)?;
        Ok((next, prefix))
    }

    /// Full-attention forward with pre-computed K/V (used by the chunked
    /// prefix pass: K/V spans prefix + chunk, Q spans the chunk only).
    /// `q_start` = absolute row of the chunk's first token, so the causal
    /// mask lets Q[i] attend to K/V[0 .. q_start+i+1].
    #[allow(clippy::too_many_arguments)]
    fn forward_full_attention_with_kv(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        kv: &PrefixKv,
        batch: usize,
        seq_len: usize,
        q_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let local_attention_heads = tp.local_attention_heads(cfg)?;
        let q_full = attn.q_proj.forward(h, store, tape)?;
        let (q, gate) = if cfg.full_attn_gated {
            let q_full = reshape(
                q_full,
                &[batch, seq_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            let q = slice(
                q_full,
                &[0, 0, 0, 0],
                &[batch, seq_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            let gate = slice(
                q_full,
                &[0, 0, 0, cfg.head_dim],
                &[batch, seq_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            (
                transpose(q, 1, 2, store, tape)?,
                Some(transpose(gate, 1, 2, store, tape)?),
            )
        } else {
            let q = reshape(
                q_full,
                &[batch, seq_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            (transpose(q, 1, 2, store, tape)?, None)
        };
        let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
        let q = rope(q, cos, sin, store, tape)?;
        // kv.k / kv.v are already RMSNorm'd, RoPE'd, and repeat_kv'd (full heads).
        let attn_hidden = causal_sdpa_recompute_with_q_start(q, kv.k, kv.v, q_start, store, tape)?;
        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            seq_len,
            local_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let out = attn.o_proj.forward(attn_hidden, store, tape)?;
        Ok(maybe_all_reduce(out, tp, store, tape)?)
    }

    /// OPD frozen-prompt-KV phase 2 (per layer, TAPED): residual block over the
    /// gen rows only (`x` = `[batch, gen_len, hidden]`), where attention consumes
    /// this layer's captured prefix. RMSNorm + MLP are position-local, so the gen
    /// residual stream is exact given the seeded attention.
    #[allow(clippy::too_many_arguments)]
    fn forward_gen_segment(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos_gen: TensorId,
        sin_gen: TensorId,
        layer_prefix: &LayerPrefix,
        batch: usize,
        gen_start: usize,
        gen_len: usize,
        cp: crate::context_parallel::CpContext,
        cp_positions: Option<&[usize]>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let attn_out = match (&self.self_attn, layer_prefix) {
            (Qwen35Attention::Full(attn), LayerPrefix::Full(prefix_kv)) => self
                .forward_full_attention_gen_segment(
                    h,
                    attn,
                    cfg,
                    tp,
                    cos_gen,
                    sin_gen,
                    prefix_kv,
                    batch,
                    gen_start,
                    gen_len,
                    cp,
                    cp_positions,
                    store,
                    tape,
                )?,
            (Qwen35Attention::Linear(attn), LayerPrefix::Linear(prefix_state)) => {
                if cp.is_enabled() {
                    return Err(Qwen35Error::InvalidConfig(
                        "frozen-prompt-KV + CP is not yet implemented for linear attention layers",
                    ));
                }
                self.forward_linear_attention_gen_segment(
                    h,
                    attn,
                    cfg,
                    prefix_state,
                    batch,
                    gen_len,
                    store,
                    tape,
                )?
            }
            _ => {
                return Err(Qwen35Error::InvalidConfig(
                    "frozen-prompt-KV layer prefix kind does not match the layer attention kind",
                ));
            }
        };
        let x = add(x, attn_out, store, tape)?;
        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let mlp_out = self.forward_mlp(h, cfg, tp, batch, gen_len, store, tape)?;
        Ok(add(x, mlp_out, store, tape)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_profiled(
        &self,
        layer_index: usize,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        trace: bool,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(TensorId, Qwen35LayerForwardProfile)> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];
        let mut profile = Qwen35LayerForwardProfile::default();

        let started = Instant::now();
        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        profile.input_rmsnorm += started.elapsed();
        trace_forward_component(trace, layer_index, "input_rmsnorm", profile.input_rmsnorm);

        let started = Instant::now();
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => {
                let mut attention_profile = Qwen35AttentionForwardProfile::default();
                let out = self.forward_full_attention_profiled(
                    layer_index,
                    h,
                    attn,
                    cfg,
                    tp,
                    cos,
                    sin,
                    batch,
                    seq_len,
                    trace,
                    store,
                    tape,
                    &mut attention_profile,
                )?;
                profile.attention_detail = attention_profile;
                out
            }
            Qwen35Attention::Linear(attn) => {
                let mut attention_profile = Qwen35AttentionForwardProfile::default();
                let out = self.forward_linear_attention_profiled(
                    layer_index,
                    h,
                    attn,
                    cfg,
                    batch,
                    seq_len,
                    trace,
                    store,
                    tape,
                    &mut attention_profile,
                )?;
                profile.attention_detail = attention_profile;
                out
            }
        };
        profile.attention += started.elapsed();
        trace_forward_component(trace, layer_index, "attention_total", profile.attention);

        let started = Instant::now();
        let x = add(x, attn_out, store, tape)?;
        profile.attention_residual += started.elapsed();
        trace_forward_component(
            trace,
            layer_index,
            "attention_residual",
            profile.attention_residual,
        );

        let started = Instant::now();
        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        profile.post_attention_rmsnorm += started.elapsed();
        trace_forward_component(
            trace,
            layer_index,
            "post_attention_rmsnorm",
            profile.post_attention_rmsnorm,
        );

        let started = Instant::now();
        let mlp_out = self.forward_mlp(h, cfg, tp, batch, seq_len, store, tape)?;
        profile.mlp += started.elapsed();
        trace_forward_component(trace, layer_index, "mlp", profile.mlp);

        let started = Instant::now();
        let out = add(x, mlp_out, store, tape)?;
        profile.mlp_residual += started.elapsed();
        trace_forward_component(trace, layer_index, "mlp_residual", profile.mlp_residual);

        Ok((out, profile))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_collecting_moe_routes(
        &self,
        layer_index: usize,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        route_signatures: &mut Vec<Qwen35MoeRouteSignature>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];

        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => self.forward_full_attention(
                h,
                attn,
                cfg,
                tp,
                crate::context_parallel::CpContext::single(),
                cos,
                sin,
                batch,
                seq_len,
                None,
                store,
                tape,
            )?,
            Qwen35Attention::Linear(attn) => self.forward_linear_attention(
                h,
                attn,
                cfg,
                crate::context_parallel::CpContext::single(),
                batch,
                seq_len,
                store,
                tape,
            )?,
        };
        let x = add(x, attn_out, store, tape)?;

        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let mlp_out = self.forward_mlp_collecting_routes(
            layer_index,
            h,
            cfg,
            tp,
            batch,
            seq_len,
            route_signatures,
            store,
            tape,
        )?;
        Ok(add(x, mlp_out, store, tape)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_frozen_moe_routes(
        &self,
        layer_index: usize,
        x: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        frozen_routes: &[Qwen35MoeRouteSignature],
        next_route: &mut usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];

        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => self.forward_full_attention(
                h,
                attn,
                cfg,
                tp,
                crate::context_parallel::CpContext::single(),
                cos,
                sin,
                batch,
                seq_len,
                None,
                store,
                tape,
            )?,
            Qwen35Attention::Linear(attn) => self.forward_linear_attention(
                h,
                attn,
                cfg,
                crate::context_parallel::CpContext::single(),
                batch,
                seq_len,
                store,
                tape,
            )?,
        };
        let x = add(x, attn_out, store, tape)?;

        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let mlp_out = self.forward_mlp_with_frozen_routes(
            layer_index,
            h,
            cfg,
            tp,
            batch,
            seq_len,
            frozen_routes,
            next_route,
            store,
            tape,
        )?;
        Ok(add(x, mlp_out, store, tape)?)
    }

    fn forward_with_kv_cache(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        cos: TensorId,
        sin: TensorId,
        layer_cache: &mut Qwen35LayerKvCache,
        q_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];

        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => self.forward_full_attention_with_kv_cache(
                h,
                attn,
                cfg,
                cos,
                sin,
                batch,
                seq_len,
                layer_cache,
                q_start,
                store,
                tape,
            )?,
            Qwen35Attention::Linear(_) => {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache requires full-attention layers",
                ));
            }
        };
        let x = add(x, attn_out, store, tape)?;

        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let mlp_out = self.forward_mlp(h, cfg, TpContext::single(), batch, seq_len, store, tape)?;
        Ok(add(x, mlp_out, store, tape)?)
    }

    fn forward_with_kv_cache_profiled(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        cos: TensorId,
        sin: TensorId,
        layer_cache: &mut Qwen35LayerKvCache,
        q_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(TensorId, Qwen35LayerForwardProfile)> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        if x_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "rank-3 hidden states [batch, seq, hidden]",
                got: x_shape.len(),
            }
            .into());
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];
        let mut profile = Qwen35LayerForwardProfile::default();

        let started = Instant::now();
        let h = qwen35_rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        profile.input_rmsnorm += started.elapsed();

        let started = Instant::now();
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => {
                let mut attention_profile = Qwen35AttentionForwardProfile::default();
                let out = self.forward_full_attention_with_kv_cache_profiled(
                    h,
                    attn,
                    cfg,
                    cos,
                    sin,
                    batch,
                    seq_len,
                    layer_cache,
                    q_start,
                    store,
                    tape,
                    &mut attention_profile,
                )?;
                profile.attention_detail = attention_profile;
                out
            }
            Qwen35Attention::Linear(_) => {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache requires full-attention layers",
                ));
            }
        };
        profile.attention += started.elapsed();

        let started = Instant::now();
        let x = add(x, attn_out, store, tape)?;
        profile.attention_residual += started.elapsed();

        let started = Instant::now();
        let h = qwen35_rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        profile.post_attention_rmsnorm += started.elapsed();

        let started = Instant::now();
        let mlp_out = self.forward_mlp(h, cfg, TpContext::single(), batch, seq_len, store, tape)?;
        profile.mlp += started.elapsed();

        let started = Instant::now();
        let out = add(x, mlp_out, store, tape)?;
        profile.mlp_residual += started.elapsed();

        Ok((out, profile))
    }

    fn forward_mlp(
        &self,
        h: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        match &self.mlp {
            Qwen35Mlp::Dense(mlp) => {
                let gate_raw = mlp.gate_proj.forward(h, store, tape)?;
                let up = mlp.up_proj.forward(h, store, tape)?;
                let gate = silu(gate_raw, store, tape)?;
                let act = mul(gate, up, store, tape)?;
                let mlp_out = mlp.down_proj.forward(act, store, tape)?;
                // tape-disabled checkpoint forward: free dead transients now
                // instead of at closure exit, cutting the single-layer peak.
                if !tape.enabled {
                    for id in [gate_raw, gate, up, act] {
                        store.free(id)?;
                    }
                }
                Ok(maybe_all_reduce(mlp_out, tp, store, tape)?)
            }
            Qwen35Mlp::Sparse(mlp) => {
                self.forward_sparse_mlp(mlp, h, cfg, tp, batch, seq_len, store, tape)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_mlp_collecting_routes(
        &self,
        layer_index: usize,
        h: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        batch: usize,
        seq_len: usize,
        route_signatures: &mut Vec<Qwen35MoeRouteSignature>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        match &self.mlp {
            Qwen35Mlp::Dense(_) => self.forward_mlp(h, cfg, tp, batch, seq_len, store, tape),
            Qwen35Mlp::Sparse(mlp) => self.forward_sparse_mlp_collecting_routes(
                layer_index,
                mlp,
                h,
                cfg,
                tp,
                batch,
                seq_len,
                route_signatures,
                store,
                tape,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_mlp_with_frozen_routes(
        &self,
        layer_index: usize,
        h: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        batch: usize,
        seq_len: usize,
        frozen_routes: &[Qwen35MoeRouteSignature],
        next_route: &mut usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        match &self.mlp {
            Qwen35Mlp::Dense(_) => self.forward_mlp(h, cfg, tp, batch, seq_len, store, tape),
            Qwen35Mlp::Sparse(mlp) => {
                let route = frozen_routes
                    .get(*next_route)
                    .ok_or(Qwen35Error::InvalidConfig(
                        "frozen MoE routes missing sparse layer signature",
                    ))?;
                *next_route += 1;
                self.forward_sparse_mlp_with_frozen_routes(
                    layer_index,
                    mlp,
                    h,
                    cfg,
                    tp,
                    batch,
                    seq_len,
                    route,
                    store,
                    tape,
                )
            }
        }
    }

    fn forward_sparse_mlp(
        &self,
        mlp: &Qwen35SparseMlp,
        h: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let tokens = batch * seq_len;
        let flat_h = reshape(h, &[tokens, cfg.hidden_size], store, tape)?;
        let router_logits = mlp.router_gate.forward(flat_h, store, tape)?;
        let routes = moe_topk_softmax(router_logits, mlp.top_k, store, tape)?;
        let grouped_routes = build_qwen35_grouped_routes(&routes)?;

        let gate_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.gate_proj);
        let up_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.up_proj);
        let down_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.down_proj);

        let routed_gate = moe_grouped_linear(
            flat_h,
            &gate_experts,
            &grouped_routes,
            MoeGroupedLinearInput::TokenRows,
            store,
            tape,
        )?;
        let routed_up = moe_grouped_linear(
            flat_h,
            &up_experts,
            &grouped_routes,
            MoeGroupedLinearInput::TokenRows,
            store,
            tape,
        )?;
        let routed_gate = silu(routed_gate, store, tape)?;
        let routed_hidden = mul(routed_gate, routed_up, store, tape)?;
        let routed_down = moe_grouped_linear(
            routed_hidden,
            &down_experts,
            &grouped_routes,
            MoeGroupedLinearInput::Packed,
            store,
            tape,
        )?;
        let routed = moe_grouped_weighted_scatter(
            routed_down,
            routes.weights,
            &grouped_routes,
            tokens,
            store,
            tape,
        )?;

        let shared_gate = mlp.shared_gate_proj.forward(flat_h, store, tape)?;
        let shared_up = mlp.shared_up_proj.forward(flat_h, store, tape)?;
        let shared_gate = silu(shared_gate, store, tape)?;
        let shared_hidden = mul(shared_gate, shared_up, store, tape)?;
        let shared = mlp.shared_down_proj.forward(shared_hidden, store, tape)?;
        let shared_expert_gate = mlp.shared_expert_gate.forward(flat_h, store, tape)?;
        let shared_expert_gate = sigmoid(shared_expert_gate, store, tape)?;
        let shared_expert_gate =
            broadcast_to_shape(shared_expert_gate, &[tokens, cfg.hidden_size], store, tape)?;
        let shared = mul(shared, shared_expert_gate, store, tape)?;

        // Row-parallel down_proj (routed + shared) yields per-rank partial sums;
        // one all-reduce on their sum completes both across the TP group.
        let out = add(routed, shared, store, tape)?;
        let out = maybe_all_reduce(out, tp, store, tape)?;
        Ok(reshape(
            out,
            &[batch, seq_len, cfg.hidden_size],
            store,
            tape,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_sparse_mlp_collecting_routes(
        &self,
        layer_index: usize,
        mlp: &Qwen35SparseMlp,
        h: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        batch: usize,
        seq_len: usize,
        route_signatures: &mut Vec<Qwen35MoeRouteSignature>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let tokens = batch * seq_len;
        let flat_h = reshape(h, &[tokens, cfg.hidden_size], store, tape)?;
        let router_logits = mlp.router_gate.forward(flat_h, store, tape)?;
        let routes = moe_topk_softmax(router_logits, mlp.top_k, store, tape)?;
        route_signatures.push(Qwen35MoeRouteSignature {
            layer: layer_index,
            tokens: routes.tokens,
            experts: routes.experts,
            top_k: routes.top_k,
            indices: routes.indices.clone(),
        });
        let grouped_routes = build_qwen35_grouped_routes(&routes)?;

        let gate_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.gate_proj);
        let up_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.up_proj);
        let down_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.down_proj);

        let routed_gate = moe_grouped_linear(
            flat_h,
            &gate_experts,
            &grouped_routes,
            MoeGroupedLinearInput::TokenRows,
            store,
            tape,
        )?;
        let routed_up = moe_grouped_linear(
            flat_h,
            &up_experts,
            &grouped_routes,
            MoeGroupedLinearInput::TokenRows,
            store,
            tape,
        )?;
        let routed_gate = silu(routed_gate, store, tape)?;
        let routed_hidden = mul(routed_gate, routed_up, store, tape)?;
        let routed_down = moe_grouped_linear(
            routed_hidden,
            &down_experts,
            &grouped_routes,
            MoeGroupedLinearInput::Packed,
            store,
            tape,
        )?;
        let routed = moe_grouped_weighted_scatter(
            routed_down,
            routes.weights,
            &grouped_routes,
            tokens,
            store,
            tape,
        )?;

        let shared_gate = mlp.shared_gate_proj.forward(flat_h, store, tape)?;
        let shared_up = mlp.shared_up_proj.forward(flat_h, store, tape)?;
        let shared_gate = silu(shared_gate, store, tape)?;
        let shared_hidden = mul(shared_gate, shared_up, store, tape)?;
        let shared = mlp.shared_down_proj.forward(shared_hidden, store, tape)?;
        let shared_expert_gate = mlp.shared_expert_gate.forward(flat_h, store, tape)?;
        let shared_expert_gate = sigmoid(shared_expert_gate, store, tape)?;
        let shared_expert_gate =
            broadcast_to_shape(shared_expert_gate, &[tokens, cfg.hidden_size], store, tape)?;
        let shared = mul(shared, shared_expert_gate, store, tape)?;

        let out = add(routed, shared, store, tape)?;
        let out = maybe_all_reduce(out, tp, store, tape)?;
        Ok(reshape(
            out,
            &[batch, seq_len, cfg.hidden_size],
            store,
            tape,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_sparse_mlp_with_frozen_routes(
        &self,
        layer_index: usize,
        mlp: &Qwen35SparseMlp,
        h: TensorId,
        cfg: &Qwen35Config,
        tp: TpContext,
        batch: usize,
        seq_len: usize,
        route: &Qwen35MoeRouteSignature,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let tokens = batch * seq_len;
        validate_qwen35_moe_route_signature(
            route,
            layer_index,
            tokens,
            mlp.experts.len(),
            mlp.top_k,
        )?;
        let flat_h = reshape(h, &[tokens, cfg.hidden_size], store, tape)?;
        let router_logits = mlp.router_gate.forward(flat_h, store, tape)?;
        let routes =
            moe_topk_softmax_with_indices(router_logits, mlp.top_k, &route.indices, store, tape)?;
        let grouped_routes = build_qwen35_grouped_routes(&routes)?;

        let gate_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.gate_proj);
        let up_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.up_proj);
        let down_experts = qwen35_grouped_linear_experts(&mlp.experts, |expert| &expert.down_proj);

        let routed_gate = moe_grouped_linear(
            flat_h,
            &gate_experts,
            &grouped_routes,
            MoeGroupedLinearInput::TokenRows,
            store,
            tape,
        )?;
        let routed_up = moe_grouped_linear(
            flat_h,
            &up_experts,
            &grouped_routes,
            MoeGroupedLinearInput::TokenRows,
            store,
            tape,
        )?;
        let routed_gate = silu(routed_gate, store, tape)?;
        let routed_hidden = mul(routed_gate, routed_up, store, tape)?;
        let routed_down = moe_grouped_linear(
            routed_hidden,
            &down_experts,
            &grouped_routes,
            MoeGroupedLinearInput::Packed,
            store,
            tape,
        )?;
        let routed = moe_grouped_weighted_scatter(
            routed_down,
            routes.weights,
            &grouped_routes,
            tokens,
            store,
            tape,
        )?;

        let shared_gate = mlp.shared_gate_proj.forward(flat_h, store, tape)?;
        let shared_up = mlp.shared_up_proj.forward(flat_h, store, tape)?;
        let shared_gate = silu(shared_gate, store, tape)?;
        let shared_hidden = mul(shared_gate, shared_up, store, tape)?;
        let shared = mlp.shared_down_proj.forward(shared_hidden, store, tape)?;
        let shared_expert_gate = mlp.shared_expert_gate.forward(flat_h, store, tape)?;
        let shared_expert_gate = sigmoid(shared_expert_gate, store, tape)?;
        let shared_expert_gate =
            broadcast_to_shape(shared_expert_gate, &[tokens, cfg.hidden_size], store, tape)?;
        let shared = mul(shared, shared_expert_gate, store, tape)?;

        let out = add(routed, shared, store, tape)?;
        let out = maybe_all_reduce(out, tp, store, tape)?;
        Ok(reshape(
            out,
            &[batch, seq_len, cfg.hidden_size],
            store,
            tape,
        )?)
    }

    fn forward_full_attention(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cp: crate::context_parallel::CpContext,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        cp_positions: Option<&[usize]>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let local_attention_heads = tp.local_attention_heads(cfg)?;
        let local_key_value_heads = tp.local_key_value_heads(cfg)?;
        let kv_repeat = local_attention_heads / local_key_value_heads;

        // Non-CP: k/v full-seq (small), chunk q_proj+SDPA+o_proj to bound the
        // full-seq f32 GEMM output. CP uses its own q-shard + all-gather-kv.
        if !cp.is_enabled() {
            return self.forward_full_attention_chunked(
                h,
                attn,
                cfg,
                tp,
                cos,
                sin,
                batch,
                seq_len,
                local_attention_heads,
                local_key_value_heads,
                kv_repeat,
                crate::runtime_flags::OPD_SEQ_CHUNK,
                store,
                tape,
            );
        }

        // Qwen3.5 / Qwen3.6 ship a gated Q projection: q_proj rows =
        // `num_heads * head_dim * 2`, with the second half acting as a
        // per-head sigmoid gate applied to the attention output. Vanilla
        // Qwen3 (0.6B / 1.7B / 4B / 8B) is un-gated: q_proj rows =
        // `num_heads * head_dim`. The arch flag `cfg.full_attn_gated`
        // selects between the two paths so `qwen35_loader` can load both
        // checkpoint families without an arch fork.
        let q_full = attn.q_proj.forward(h, store, tape)?;
        let (q, gate) = if cfg.full_attn_gated {
            let q_full = reshape(
                q_full,
                &[batch, seq_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            let q = slice(
                q_full,
                &[0, 0, 0, 0],
                &[batch, seq_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            let gate = slice(
                q_full,
                &[0, 0, 0, cfg.head_dim],
                &[batch, seq_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            (
                transpose(q, 1, 2, store, tape)?,
                Some(transpose(gate, 1, 2, store, tape)?),
            )
        } else {
            let q = reshape(
                q_full,
                &[batch, seq_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            (transpose(q, 1, 2, store, tape)?, None)
        };

        let k = attn.k_proj.forward(h, store, tape)?;
        let v = attn.v_proj.forward(h, store, tape)?;
        let k = split_heads(
            k,
            batch,
            seq_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            v,
            batch,
            seq_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;

        let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
        let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        let q = rope(q, cos, sin, store, tape)?;
        let k = rope(k, cos, sin, store, tape)?;

        let kv_repeat = local_attention_heads / local_key_value_heads;
        let attn_hidden = if cp.is_enabled() {
            // Context-parallel ring attention: q/k/v are this rank's LOCAL shard
            // ([b, heads, seq_len, hd] / [b, kv_heads, seq_len, hd]) at absolute
            // rows [cp.rank*seq_len, ..). The ring rotates K/V cp.size times through
            // the flash-2 device kernel, attending the causal prefix on-device —
            // NEVER materializing the full sequence (peak O(seq_len·hd), not
            // O(full_seq·hd)), which is the fix for the option-B slice_bwd OOM at
            // local seq > 65535. GQA repeat happens per-block inside the kernel, so
            // k/v ship at kv_heads width (kv_repeat× less comm). The launcher pads
            // global seq to a multiple of cp.size and shards RoPE positions so the
            // absolute q_abs = cp.rank*seq_len agrees with the baked-in positions.
            let _ = kv_repeat; // GQA resolved inside the ring kernel, not here
            debug_assert_eq!(
                store.get(cos).map(|t| t.shape[t.shape.len() - 2]),
                Some(seq_len),
                "CP: cos rows must equal local seq_len (launcher must shard positions)"
            );
            // Absolute position of each local row — passed in as data, the SAME slice
            // that built cos/sin, so the ring masks causally by true position and a
            // zigzag shard's two chunks (front+back) attend the right prefix. Threaded
            // (not re-derived) so any sharding scheme lives only in the caller (opd.rs),
            // not here — no `global = local*size` equal-shard assumption baked in.
            let positions = cp_positions.ok_or(AutogradError::TapeInvariant(
                "CP full-attention requires cp_positions (the shard's absolute rows)",
            ))?;
            debug_assert_eq!(
                positions.len(),
                seq_len,
                "CP: cp_positions must give one absolute position per local row"
            );
            autograd::ops::ring_attention::cp_causal_sdpa(
                q,
                k,
                v,
                cp.size,
                cp.rank,
                Some(positions),
                store,
                tape,
            )?
        } else {
            let k = repeat_kv(k, kv_repeat, store, tape)?;
            let v = repeat_kv(v, kv_repeat, store, tape)?;
            causal_sdpa_recompute(q, k, v, store, tape)?
        };
        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            seq_len,
            local_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let out = attn.o_proj.forward(attn_hidden, store, tape)?;
        Ok(maybe_all_reduce(out, tp, store, tape)?)
    }

    /// Long-sequence single-GPU full attention: k/v are computed full-sequence
    /// (kv_dim is small), then q_proj + causal SDPA + o_proj are run per
    /// sequence chunk via `checkpoint_seq_chunked`. This bounds the q/o
    /// projection f32 outputs (and their LoRA deltas) to the chunk row count
    /// instead of the full sequence — the 131K forward-OOM site.
    #[allow(clippy::too_many_arguments)]
    fn forward_full_attention_chunked(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        local_attention_heads: usize,
        local_key_value_heads: usize,
        kv_repeat: usize,
        chunk: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let head_dim = cfg.head_dim;
        let rms_norm_eps = cfg.rms_norm_eps;
        let full_attn_gated = cfg.full_attn_gated;

        // k/v: small enough to keep full-sequence.
        let k = attn.k_proj.forward(h, store, tape)?;
        let v = attn.v_proj.forward(h, store, tape)?;
        let k = split_heads(
            k,
            batch,
            seq_len,
            local_key_value_heads,
            head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            v,
            batch,
            seq_len,
            local_key_value_heads,
            head_dim,
            store,
            tape,
        )?;
        let k = qwen35_rmsnorm(k, attn.k_norm, rms_norm_eps, store, tape)?;
        let k = rope(k, cos, sin, store, tape)?;
        let k = repeat_kv(k, kv_repeat, store, tape)?;
        let v = repeat_kv(v, kv_repeat, store, tape)?;

        // Capture the linears and norm for the replay closure.
        let q_proj = attn.q_proj.clone();
        let o_proj = attn.o_proj.clone();
        let q_norm = attn.q_norm;

        // Saved inputs: k, v (full-seq intermediates), cos/sin, and the
        // trainable q_proj/o_proj/q_norm weights. `checkpoint_seq_chunked`
        // accumulates their gradients across chunks during backward.
        let mut param_ids = vec![k, v, cos, sin];
        collect_linear_ids(&q_proj, &mut param_ids);
        collect_linear_ids(&o_proj, &mut param_ids);
        param_ids.push(q_norm);

        let out = autograd::ops::checkpoint_seq_chunked(
            h,
            param_ids,
            chunk,
            store,
            tape,
            move |st, tp_tape, start, inp| {
                let h_chunk = inp[0];
                let k = inp[1];
                let v = inp[2];
                let cos_full = inp[3];
                let sin_full = inp[4];

                let chunk_shape = st
                    .get(h_chunk)
                    .ok_or(AutogradError::InvalidTensorId(h_chunk))?
                    .shape
                    .clone();
                let chunk_len = chunk_shape[1];

                // q projection (with LoRA) for this chunk.
                let q_full = q_proj.forward(h_chunk, st, tp_tape)?;

                let (q, gate) = if full_attn_gated {
                    let q_full = reshape(
                        q_full,
                        &[batch, chunk_len, local_attention_heads, head_dim * 2],
                        st,
                        tp_tape,
                    )?;
                    let q = slice(
                        q_full,
                        &[0, 0, 0, 0],
                        &[batch, chunk_len, local_attention_heads, head_dim],
                        st,
                        tp_tape,
                    )?;
                    let gate = slice(
                        q_full,
                        &[0, 0, 0, head_dim],
                        &[batch, chunk_len, local_attention_heads, head_dim * 2],
                        st,
                        tp_tape,
                    )?;
                    (
                        transpose(q, 1, 2, st, tp_tape)?,
                        Some(transpose(gate, 1, 2, st, tp_tape)?),
                    )
                } else {
                    let q = reshape(
                        q_full,
                        &[batch, chunk_len, local_attention_heads, head_dim],
                        st,
                        tp_tape,
                    )?;
                    (transpose(q, 1, 2, st, tp_tape)?, None)
                };

                // q norm + rope (cos/sin sliced to this chunk's positions).
                let q = qwen35_rmsnorm(q, q_norm, rms_norm_eps, st, tp_tape)
                    .map_err(qwen35_to_autograd)?;
                let cos_shape = st
                    .get(cos_full)
                    .ok_or(AutogradError::InvalidTensorId(cos_full))?
                    .shape
                    .clone();
                let rotary_dim = cos_shape[1];
                let cos_chunk = slice(
                    cos_full,
                    &[start, 0],
                    &[start + chunk_len, rotary_dim],
                    st,
                    tp_tape,
                )?;
                let sin_chunk = slice(
                    sin_full,
                    &[start, 0],
                    &[start + chunk_len, rotary_dim],
                    st,
                    tp_tape,
                )?;
                let q = rope(q, cos_chunk, sin_chunk, st, tp_tape)?;

                // Causal SDPA: q at [start, start+chunk) attends k/v at [0, start+chunk).
                let attn_hidden = causal_sdpa_recompute_with_q_start(q, k, v, start, st, tp_tape)?;

                let attn_hidden = if let Some(gate) = gate {
                    let gate = sigmoid(gate, st, tp_tape)?;
                    mul(attn_hidden, gate, st, tp_tape)?
                } else {
                    attn_hidden
                };

                let attn_hidden = merge_heads(
                    attn_hidden,
                    batch,
                    chunk_len,
                    local_attention_heads,
                    head_dim,
                    st,
                    tp_tape,
                )
                .map_err(qwen35_to_autograd)?;
                o_proj.forward(attn_hidden, st, tp_tape)
            },
        )?;

        Ok(maybe_all_reduce(out, tp, store, tape)?)
    }

    /// OPD frozen-prompt-KV phase 1: compute the prompt prefix's repeat_kv'd K/V
    /// at absolute positions `0..gen_start` (k_proj/v_proj → split_heads →
    /// k_norm → rope(cos/sin sliced to the prefix) → repeat_kv), returning them
    /// as `requires_grad=false` constants. Off-tape (caller passes a disabled
    /// tape). Only K/V — the prompt's Q is never queried by the gen segment.
    #[allow(clippy::too_many_arguments)]
    fn forward_full_attention_capture_prefix_kv(
        &self,
        h_prefix: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos_prefix: TensorId,
        sin_prefix: TensorId,
        batch: usize,
        gen_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<PrefixKv> {
        let local_attention_heads = tp.local_attention_heads(cfg)?;
        let local_key_value_heads = tp.local_key_value_heads(cfg)?;
        let k = attn.k_proj.forward(h_prefix, store, tape)?;
        let v = attn.v_proj.forward(h_prefix, store, tape)?;
        let k = split_heads(
            k,
            batch,
            gen_start,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            v,
            batch,
            gen_start,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        let k = rope(k, cos_prefix, sin_prefix, store, tape)?;
        let kv_repeat = local_attention_heads / local_key_value_heads;
        let k = repeat_kv(k, kv_repeat, store, tape)?;
        let v = repeat_kv(v, kv_repeat, store, tape)?;
        // Pin as constants so the gen-segment concat treats them as frozen.
        store.set_requires_grad(k, false)?;
        store.set_requires_grad(v, false)?;
        // Largest retained buffer, no grad chain — the bf16 tape's first target.
        store.quantize_frozen_bf16(k)?;
        store.quantize_frozen_bf16(v)?;
        Ok(PrefixKv { k, v })
    }

    /// OPD frozen-prompt-KV phase 2: TAPED full attention over the gen rows only
    /// (`h_gen` is `[batch, gen_len, hidden]`). Projects Q/K/V for the gen rows,
    /// q_norm/k_norm, rope at gen positions (cos_gen/sin_gen already sliced to
    /// `gen_start..seq_len`), repeat_kv, then concatenates the frozen prefix K/V
    /// along the seq axis (grad flows into k_gen/v_gen only) and runs cached SDPA
    /// with `q_start = gen_start`. The gate/merge/o_proj/all_reduce tail is
    /// identical to `forward_full_attention`.
    #[allow(clippy::too_many_arguments)]
    fn forward_full_attention_gen_segment(
        &self,
        h_gen: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos_gen: TensorId,
        sin_gen: TensorId,
        prefix_kv: &PrefixKv,
        batch: usize,
        gen_start: usize,
        gen_len: usize,
        cp: crate::context_parallel::CpContext,
        cp_positions: Option<&[usize]>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let q_full = attn.q_proj.forward(h_gen, store, tape)?;
        let local_attention_heads = tp.local_attention_heads(cfg)?;
        let local_key_value_heads = tp.local_key_value_heads(cfg)?;
        let (q, gate) = if cfg.full_attn_gated {
            let q_full = reshape(
                q_full,
                &[batch, gen_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            let q = slice(
                q_full,
                &[0, 0, 0, 0],
                &[batch, gen_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            let gate = slice(
                q_full,
                &[0, 0, 0, cfg.head_dim],
                &[batch, gen_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            (
                transpose(q, 1, 2, store, tape)?,
                Some(transpose(gate, 1, 2, store, tape)?),
            )
        } else {
            let q = reshape(
                q_full,
                &[batch, gen_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            (transpose(q, 1, 2, store, tape)?, None)
        };

        let k = attn.k_proj.forward(h_gen, store, tape)?;
        let v = attn.v_proj.forward(h_gen, store, tape)?;
        let k = split_heads(
            k,
            batch,
            gen_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            v,
            batch,
            gen_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;

        let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
        let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        let q = rope(q, cos_gen, sin_gen, store, tape)?;
        let k = rope(k, cos_gen, sin_gen, store, tape)?;

        // CP + off-CP share one path: cp_causal_sdpa_with_prefix handles cp_size==1
        // by falling through to its prefix CPU block (repeat_kv + cat + q_start
        // causal SDPA), identical to the old inline else branch. Gen k/v stay at
        // kv_heads width under CP (GQA resolved in the ring kernel); off-CP the
        // prefix block repeat_kv's them to match the full-heads prefix.
        let attn_hidden = autograd::ops::ring_attention::cp_causal_sdpa_with_prefix(
            q,
            k,
            v,
            cp.size,
            cp.rank,
            cp_positions,
            Some((prefix_kv.k, prefix_kv.v)),
            gen_start,
            store,
            tape,
        )?;

        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            gen_len,
            local_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let out = attn.o_proj.forward(attn_hidden, store, tape)?;
        Ok(maybe_all_reduce(out, tp, store, tape)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_full_attention_profiled(
        &self,
        layer_index: usize,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        tp: TpContext,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        trace: bool,
        store: &mut TensorStore,
        tape: &mut Tape,
        profile: &mut Qwen35AttentionForwardProfile,
    ) -> Result<TensorId> {
        let started = Instant::now();
        let q_full = attn.q_proj.forward(h, store, tape)?;
        profile.q_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "q_proj", profile.q_proj);

        let local_attention_heads = tp.local_attention_heads(cfg)?;
        let local_key_value_heads = tp.local_key_value_heads(cfg)?;
        let started = Instant::now();
        let (q, gate) = if cfg.full_attn_gated {
            let q_full = reshape(
                q_full,
                &[batch, seq_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            let q = slice(
                q_full,
                &[0, 0, 0, 0],
                &[batch, seq_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            let gate = slice(
                q_full,
                &[0, 0, 0, cfg.head_dim],
                &[batch, seq_len, local_attention_heads, cfg.head_dim * 2],
                store,
                tape,
            )?;
            (
                transpose(q, 1, 2, store, tape)?,
                Some(transpose(gate, 1, 2, store, tape)?),
            )
        } else {
            let q = reshape(
                q_full,
                &[batch, seq_len, local_attention_heads, cfg.head_dim],
                store,
                tape,
            )?;
            (transpose(q, 1, 2, store, tape)?, None)
        };
        profile.q_layout += started.elapsed();
        trace_attention_component(trace, layer_index, "q_layout", profile.q_layout);

        let started = Instant::now();
        let k = attn.k_proj.forward(h, store, tape)?;
        profile.k_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "k_proj", profile.k_proj);

        let started = Instant::now();
        let v = attn.v_proj.forward(h, store, tape)?;
        profile.v_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "v_proj", profile.v_proj);

        let started = Instant::now();
        let k = split_heads(
            k,
            batch,
            seq_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            v,
            batch,
            seq_len,
            local_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        profile.kv_split += started.elapsed();
        trace_attention_component(trace, layer_index, "kv_split", profile.kv_split);

        let started = Instant::now();
        let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
        let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        profile.qk_norm += started.elapsed();
        trace_attention_component(trace, layer_index, "qk_norm", profile.qk_norm);

        let started = Instant::now();
        let q = rope(q, cos, sin, store, tape)?;
        let k = rope(k, cos, sin, store, tape)?;
        profile.rope += started.elapsed();
        trace_attention_component(trace, layer_index, "rope", profile.rope);

        let kv_repeat = local_attention_heads / local_key_value_heads;
        let started = Instant::now();
        let k = repeat_kv(k, kv_repeat, store, tape)?;
        let v = repeat_kv(v, kv_repeat, store, tape)?;
        profile.repeat_kv += started.elapsed();
        trace_attention_component(trace, layer_index, "repeat_kv", profile.repeat_kv);

        let started = Instant::now();
        let attn_hidden = causal_sdpa_recompute(q, k, v, store, tape)?;
        profile.sdpa += started.elapsed();
        trace_attention_component(trace, layer_index, "sdpa", profile.sdpa);

        let started = Instant::now();
        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        profile.gate += started.elapsed();
        trace_attention_component(trace, layer_index, "gate", profile.gate);

        let started = Instant::now();
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            seq_len,
            local_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        profile.merge += started.elapsed();
        trace_attention_component(trace, layer_index, "merge", profile.merge);

        let started = Instant::now();
        let out = attn.o_proj.forward(attn_hidden, store, tape)?;
        let out = maybe_all_reduce(out, tp, store, tape)?;
        profile.o_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "o_proj", profile.o_proj);
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_full_attention_with_kv_cache(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        layer_cache: &mut Qwen35LayerKvCache,
        q_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let q_full = attn.q_proj.forward(h, store, tape)?;
        let decode_prepare_fast = !tape.enabled
            && seq_len == 1
            && cfg.rotary_dim == cfg.head_dim
            && store.backend().device() == Device::Cuda;
        let (q, gate, k, v) = if decode_prepare_fast {
            let (q, gate) =
                qwen_decode_prepare_q(q_full, attn.q_norm, cos, sin, cfg, batch, store)?;
            let k = attn.k_proj.forward(h, store, tape)?;
            let v = attn.v_proj.forward(h, store, tape)?;
            let (k, v) = qwen_decode_prepare_kv(k, v, attn.k_norm, cos, sin, cfg, batch, store)?;
            (q, gate, k, v)
        } else {
            let (q, gate) = if cfg.full_attn_gated {
                let q_full = reshape(
                    q_full,
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim * 2],
                    store,
                    tape,
                )?;
                let q = slice(
                    q_full,
                    &[0, 0, 0, 0],
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim],
                    store,
                    tape,
                )?;
                let gate = slice(
                    q_full,
                    &[0, 0, 0, cfg.head_dim],
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim * 2],
                    store,
                    tape,
                )?;
                (
                    transpose(q, 1, 2, store, tape)?,
                    Some(transpose(gate, 1, 2, store, tape)?),
                )
            } else {
                let q = reshape(
                    q_full,
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim],
                    store,
                    tape,
                )?;
                (transpose(q, 1, 2, store, tape)?, None)
            };

            let k = attn.k_proj.forward(h, store, tape)?;
            let v = attn.v_proj.forward(h, store, tape)?;
            let k = split_heads(
                k,
                batch,
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                store,
                tape,
            )?;
            let v = split_heads(
                v,
                batch,
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                store,
                tape,
            )?;

            let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
            let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
            let q = rope(q, cos, sin, store, tape)?;
            let k = rope(k, cos, sin, store, tape)?;
            (q, gate, k, v)
        };

        if layer_cache.seq_cursor != q_start {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer cursor diverged from global cache length",
            ));
        }
        let prev_kv_len = layer_cache.seq_cursor;
        let k_cache = append_cached_kv(
            layer_cache.k,
            k,
            layer_cache.max_seq_len,
            prev_kv_len,
            store,
        )?;
        let v_cache = append_cached_kv(
            layer_cache.v,
            v,
            layer_cache.max_seq_len,
            prev_kv_len,
            store,
        )?;
        let kv_len = prev_kv_len + seq_len;
        layer_cache.k = Some(k_cache);
        layer_cache.v = Some(v_cache);
        layer_cache.seq_cursor = kv_len;

        let kv_repeat = cfg.num_attention_heads / cfg.num_key_value_heads;
        let attn_hidden = if !tape.enabled && seq_len == 1 && q_start + 1 == kv_len {
            causal_sdpa_decode_gqa_cached(q, k_cache, v_cache, kv_len, q_start, store, tape)?
        } else {
            if prev_kv_len != 0 {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache only supports one-token decode after the initial prompt",
                ));
            }
            let k_all = repeat_kv(k, kv_repeat, store, tape)?;
            let v_all = repeat_kv(v, kv_repeat, store, tape)?;
            causal_sdpa_with_q_start(q, k_all, v_all, q_start, store, tape)?
        };
        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            seq_len,
            cfg.num_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        Ok(attn.o_proj.forward(attn_hidden, store, tape)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_full_attention_with_kv_cache_profiled(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        layer_cache: &mut Qwen35LayerKvCache,
        q_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
        profile: &mut Qwen35AttentionForwardProfile,
    ) -> Result<TensorId> {
        let started = Instant::now();
        let q_full = attn.q_proj.forward(h, store, tape)?;
        profile.q_proj += started.elapsed();

        let decode_prepare_fast = !tape.enabled
            && seq_len == 1
            && cfg.rotary_dim == cfg.head_dim
            && store.backend().device() == Device::Cuda;
        let (q, gate, k, v) = if decode_prepare_fast {
            let started = Instant::now();
            let (q, gate) =
                qwen_decode_prepare_q(q_full, attn.q_norm, cos, sin, cfg, batch, store)?;
            profile.q_layout += started.elapsed();

            let started = Instant::now();
            let k = attn.k_proj.forward(h, store, tape)?;
            profile.k_proj += started.elapsed();

            let started = Instant::now();
            let v = attn.v_proj.forward(h, store, tape)?;
            profile.v_proj += started.elapsed();

            let started = Instant::now();
            let (k, v) = qwen_decode_prepare_kv(k, v, attn.k_norm, cos, sin, cfg, batch, store)?;
            profile.kv_split += started.elapsed();
            (q, gate, k, v)
        } else {
            let started = Instant::now();
            let (q, gate) = if cfg.full_attn_gated {
                let q_full = reshape(
                    q_full,
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim * 2],
                    store,
                    tape,
                )?;
                let q = slice(
                    q_full,
                    &[0, 0, 0, 0],
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim],
                    store,
                    tape,
                )?;
                let gate = slice(
                    q_full,
                    &[0, 0, 0, cfg.head_dim],
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim * 2],
                    store,
                    tape,
                )?;
                (
                    transpose(q, 1, 2, store, tape)?,
                    Some(transpose(gate, 1, 2, store, tape)?),
                )
            } else {
                let q = reshape(
                    q_full,
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim],
                    store,
                    tape,
                )?;
                (transpose(q, 1, 2, store, tape)?, None)
            };
            profile.q_layout += started.elapsed();

            let started = Instant::now();
            let k = attn.k_proj.forward(h, store, tape)?;
            profile.k_proj += started.elapsed();

            let started = Instant::now();
            let v = attn.v_proj.forward(h, store, tape)?;
            profile.v_proj += started.elapsed();

            let started = Instant::now();
            let k = split_heads(
                k,
                batch,
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                store,
                tape,
            )?;
            let v = split_heads(
                v,
                batch,
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                store,
                tape,
            )?;
            profile.kv_split += started.elapsed();

            let started = Instant::now();
            let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
            let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
            profile.qk_norm += started.elapsed();

            let started = Instant::now();
            let q = rope(q, cos, sin, store, tape)?;
            let k = rope(k, cos, sin, store, tape)?;
            profile.rope += started.elapsed();
            (q, gate, k, v)
        };

        let started = Instant::now();
        if layer_cache.seq_cursor != q_start {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer cursor diverged from global cache length",
            ));
        }
        let prev_kv_len = layer_cache.seq_cursor;
        let k_cache = append_cached_kv(
            layer_cache.k,
            k,
            layer_cache.max_seq_len,
            prev_kv_len,
            store,
        )?;
        let v_cache = append_cached_kv(
            layer_cache.v,
            v,
            layer_cache.max_seq_len,
            prev_kv_len,
            store,
        )?;
        let kv_len = prev_kv_len + seq_len;
        layer_cache.k = Some(k_cache);
        layer_cache.v = Some(v_cache);
        layer_cache.seq_cursor = kv_len;
        profile.append_kv += started.elapsed();

        let kv_repeat = cfg.num_attention_heads / cfg.num_key_value_heads;
        let attn_hidden = if !tape.enabled && seq_len == 1 && q_start + 1 == kv_len {
            let started = Instant::now();
            let out =
                causal_sdpa_decode_gqa_cached(q, k_cache, v_cache, kv_len, q_start, store, tape)?;
            profile.sdpa += started.elapsed();
            out
        } else {
            if prev_kv_len != 0 {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache only supports one-token decode after the initial prompt",
                ));
            }
            let repeat_started = Instant::now();
            let k_all = repeat_kv(k, kv_repeat, store, tape)?;
            let v_all = repeat_kv(v, kv_repeat, store, tape)?;
            profile.repeat_kv += repeat_started.elapsed();
            let started = Instant::now();
            let out = causal_sdpa_with_q_start(q, k_all, v_all, q_start, store, tape)?;
            profile.sdpa += started.elapsed();
            out
        };

        let started = Instant::now();
        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        profile.gate += started.elapsed();

        let started = Instant::now();
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            seq_len,
            cfg.num_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        profile.merge += started.elapsed();

        let started = Instant::now();
        let out = attn.o_proj.forward(attn_hidden, store, tape)?;
        profile.o_proj += started.elapsed();
        Ok(out)
    }

    fn forward_linear_attention(
        &self,
        h: TensorId,
        attn: &Qwen35LinearAttention,
        cfg: &Qwen35Config,
        cp: crate::context_parallel::CpContext,
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let qkv = attn.in_proj_qkv.forward(h, store, tape)?;
        let z = attn.in_proj_z.forward(h, store, tape)?;
        let b_proj = attn.in_proj_b.forward(h, store, tape)?;
        let a_proj = attn.in_proj_a.forward(h, store, tape)?;
        // CP shards sequence; linear_attention_core_cp all-to-alls it into the head
        // axis and runs the full-seq recurrence on this rank's head slice. cp.size==1
        // is the single-card core verbatim.
        let linear = autograd::ops::linear_attention_core_cp(
            qkv,
            z,
            b_proj,
            a_proj,
            attn.conv1d_weight,
            attn.dt_bias,
            attn.a_log,
            attn.norm,
            la_params(cfg, batch, seq_len),
            cp.size,
            cp.rank,
            store,
            tape,
        )?;
        Ok(attn.out_proj.forward(linear, store, tape)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_linear_attention_capture_prefix_state(
        &self,
        h_prefix: TensorId,
        attn: &Qwen35LinearAttention,
        cfg: &Qwen35Config,
        batch: usize,
        gen_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<PrefixState> {
        let qkv = attn.in_proj_qkv.forward(h_prefix, store, tape)?;
        let b_proj = attn.in_proj_b.forward(h_prefix, store, tape)?;
        let a_proj = attn.in_proj_a.forward(h_prefix, store, tape)?;
        let params = la_params(cfg, batch, gen_start);
        let (state, conv_window) = if store.backend().device() == Device::Cuda
            && params.seq_len >= params.conv_kernel - 1
        {
            linear_attention_boundary(
                qkv,
                b_proj,
                a_proj,
                attn.conv1d_weight,
                attn.dt_bias,
                attn.a_log,
                params,
                None,
                None,
                store,
            )?
        } else {
            let z = attn.in_proj_z.forward(h_prefix, store, tape)?;
            let (_, state, conv) = linear_attention_core_with_carry(
                qkv,
                z,
                b_proj,
                a_proj,
                attn.conv1d_weight,
                attn.dt_bias,
                attn.a_log,
                attn.norm,
                params,
                None,
                None,
                true,
                store,
            )?;
            (
                state.ok_or(Qwen35Error::InvalidConfig("missing linear-attention state"))?,
                conv.ok_or(Qwen35Error::InvalidConfig(
                    "missing linear-attention conv tail",
                ))?,
            )
        };
        Ok(PrefixState { state, conv_window })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_linear_attention_gen_segment(
        &self,
        h_gen: TensorId,
        attn: &Qwen35LinearAttention,
        cfg: &Qwen35Config,
        prefix_state: &PrefixState,
        batch: usize,
        gen_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let qkv = attn.in_proj_qkv.forward(h_gen, store, tape)?;
        let z = attn.in_proj_z.forward(h_gen, store, tape)?;
        let b_proj = attn.in_proj_b.forward(h_gen, store, tape)?;
        let a_proj = attn.in_proj_a.forward(h_gen, store, tape)?;
        let linear = linear_attention_core_with_carry_taped(
            qkv,
            z,
            b_proj,
            a_proj,
            attn.conv1d_weight,
            attn.dt_bias,
            attn.a_log,
            attn.norm,
            la_params(cfg, batch, gen_len),
            Some(prefix_state.state),
            Some(prefix_state.conv_window),
            store,
            tape,
        )?;
        Ok(attn.out_proj.forward(linear, store, tape)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_linear_attention_profiled(
        &self,
        layer_index: usize,
        h: TensorId,
        attn: &Qwen35LinearAttention,
        cfg: &Qwen35Config,
        batch: usize,
        seq_len: usize,
        trace: bool,
        store: &mut TensorStore,
        tape: &mut Tape,
        profile: &mut Qwen35AttentionForwardProfile,
    ) -> Result<TensorId> {
        let started = Instant::now();
        let qkv = attn.in_proj_qkv.forward(h, store, tape)?;
        profile.linear_qkv_proj += started.elapsed();
        trace_attention_component(
            trace,
            layer_index,
            "linear_qkv_proj",
            profile.linear_qkv_proj,
        );

        let started = Instant::now();
        let z = attn.in_proj_z.forward(h, store, tape)?;
        profile.linear_z_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "linear_z_proj", profile.linear_z_proj);

        let started = Instant::now();
        let b_proj = attn.in_proj_b.forward(h, store, tape)?;
        profile.linear_b_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "linear_b_proj", profile.linear_b_proj);

        let started = Instant::now();
        let a_proj = attn.in_proj_a.forward(h, store, tape)?;
        profile.linear_a_proj += started.elapsed();
        trace_attention_component(trace, layer_index, "linear_a_proj", profile.linear_a_proj);

        let started = Instant::now();
        let linear = linear_attention_core(
            qkv,
            z,
            b_proj,
            a_proj,
            attn.conv1d_weight,
            attn.dt_bias,
            attn.a_log,
            attn.norm,
            la_params(cfg, batch, seq_len),
            store,
            tape,
        )?;
        profile.linear_core += started.elapsed();
        trace_attention_component(trace, layer_index, "linear_core", profile.linear_core);

        let started = Instant::now();
        let out = attn.out_proj.forward(linear, store, tape)?;
        profile.linear_out_proj += started.elapsed();
        trace_attention_component(
            trace,
            layer_index,
            "linear_out_proj",
            profile.linear_out_proj,
        );
        Ok(out)
    }
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

/// The one config→params mapping — every linear-attention call site and the
/// checkpoint gate's ctx-bytes model must agree field-for-field.
fn la_params(cfg: &Qwen35Config, batch: usize, seq_len: usize) -> LinearAttentionParams {
    LinearAttentionParams {
        batch,
        seq_len,
        num_key_heads: cfg.linear_num_key_heads,
        num_value_heads: cfg.linear_num_value_heads,
        key_dim: cfg.linear_key_head_dim,
        value_dim: cfg.linear_value_head_dim,
        conv_kernel: cfg.linear_conv_kernel_dim,
        eps: cfg.rms_norm_eps,
    }
}

/// Peak device bytes one LINEAR-attention layer holds — the whole O(seq) story of a
/// checkpointed forward, since `linear_attention_core` runs the full sequence in one
/// call while MLP and full attention chunk. UNDER-models the measured 27B slope
/// (211712 vs 231424 B/token, −9%); `log_ckpt_peak` exists to name that drift.
fn la_layer_peak_bytes(cfg: &Qwen35Config, batch: usize, seq_len: usize) -> usize {
    let p = la_params(cfg, batch, seq_len);
    // x and h: the residual stream is the layer's, not the kernel's.
    let residual = 2 * cfg.hidden_size;
    linear_attention_row_transient_bytes(p)
        .saturating_add(linear_attention_ctx_bytes(p))
        .saturating_add(batch.saturating_mul(seq_len).saturating_mul(4 * residual))
}

/// Peak device bytes a single FULL-attention layer holds. `repeat_kv`
/// materializes k/v at the full sequence × local attention heads; everything
/// downstream is chunk-bounded.
fn full_attn_layer_peak_bytes(
    cfg: &Qwen35Config,
    tp: TpContext,
    batch: usize,
    seq_len: usize,
) -> Result<usize> {
    let kv = 2 * tp.local_attention_heads(cfg)? * cfg.head_dim;
    let residual = 2 * cfg.hidden_size;
    Ok(batch
        .saturating_mul(seq_len)
        .saturating_mul(4 * (kv + residual)))
}

impl Qwen35Model {
    pub fn new(cfg: &Qwen35Config, store: &mut TensorStore) -> Result<Self> {
        Self::new_internal(
            cfg,
            None,
            LoraTargetSet::AllLinear,
            None,
            false,
            TpContext::single(),
            Qwen35InitMode::ScratchTrain,
            store,
        )
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    pub fn tensor_parallel(&self) -> TpContext {
        self.tp
    }

    pub fn gradient_checkpointing(&self) -> bool {
        self.gradient_checkpointing
    }

    pub fn lora_layer_start(&self) -> Option<usize> {
        self.lora_layer_start
    }

    pub fn set_gradient_checkpointing(&mut self, enabled: bool) {
        self.gradient_checkpointing = enabled;
    }

    /// The one checkpointing decision. Checkpointing trades a full forward
    /// re-run in backward (plus boundary host offload) for tape memory — a
    /// good trade only when the full tape would NOT fit: modeled footprint
    /// (per layer the hidden-stream + dominant MLP term, plus the exact LA
    /// ctx bytes for hybrid layers via `linear_attention_ctx_bytes`) ×4
    /// against free VRAM. ×4 is calibrated, not a guess: the measured
    /// full-tape footprint is 2.9–3.3× the modeled floor (0.8B B=4 seq=1040
    /// [vram-ramp]: 39.1 GB forward vs 13.6 modeled, 45 GB with CE+backward,
    /// wins/2026-07-24; 27B hybrid measured 2.54×), ×~1.2 headroom for pool
    /// retention and shape variance. The 2026-07-24 ×4 revert was the broken
    /// batched LA backward crashing under checkpoint — fixed by ecc058b20 —
    /// not the ratio. Unknown memory info keeps the safe default
    /// (checkpoint). `ARLE_FORCE_CHECKPOINT` engages unconditionally — the
    /// CUDA finite-diff gate needs the checkpoint path at tiny shapes the
    /// estimate would decline.
    fn should_checkpoint(
        &self,
        batch: usize,
        seq_len: usize,
        store: &TensorStore,
        tape: &Tape,
    ) -> bool {
        self.gradient_checkpointing
            && tape.enabled
            && (std::env::var_os("ARLE_FORCE_CHECKPOINT").is_some()
                || store.backend().device_mem_info().is_none_or(|(free, _)| {
                    let cfg = &self.config;
                    let per_layer =
                        batch * seq_len * (8 * cfg.hidden_size + 3 * cfg.intermediate_size) * 4;
                    let la_layers = self
                        .layers
                        .iter()
                        .filter(|layer| matches!(layer.self_attn, Qwen35Attention::Linear(_)))
                        .count();
                    let la_ctx = linear_attention_ctx_bytes(la_params(cfg, batch, seq_len));
                    let total = per_layer
                        .saturating_mul(self.layers.len())
                        .saturating_add(la_ctx.saturating_mul(la_layers));
                    let engage = total.saturating_mul(4) > free;
                    // Log first decision and every flip: the gate's failure class is
                    // a silent false-non-engage (97 GB OOM, errors/2026-07-24), so
                    // the breadcrumb must exist even when it never engages.
                    static LAST: AtomicU8 = AtomicU8::new(u8::MAX);
                    if LAST.swap(engage as u8, Ordering::Relaxed) != engage as u8 {
                        eprintln!(
                            "[ckpt-gate] engage={engage} batch={batch} seq={seq_len} \
                         modeled={total}B (la_layers={la_layers}) free={free}B"
                        );
                    }
                    engage
                }))
    }

    /// Run the layer stack checkpointed one layer per group. Groups of K do one
    /// host offload per K layers, but a group's forward holds all K layers' live
    /// activations until it saves — K× the per-layer peak, which OOMs at long
    /// seq. The frozen/LoRA boundary still forces a split.
    fn checkpoint_layers<PF, FF>(
        &self,
        hidden: TensorId,
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
        layer_params: PF,
        layer_fn: FF,
    ) -> Result<TensorId>
    where
        FF: Fn(usize, TensorId, &mut TensorStore, &mut Tape) -> autograd::Result<TensorId>
            + Clone
            + Send
            + Sync
            + 'static,
        PF: Fn(usize) -> Vec<TensorId>,
    {
        // The rebase sets the watermark to current used, so it reads back as the floor.
        let floor = store
            .backend()
            .reset_mem_pool_used_high()
            .ok()
            .and_then(|()| store.backend().mem_pool_used_high());
        let out = checkpoint_sequential(
            hidden,
            self.layers.len(),
            1,
            self.lora_layer_start,
            store,
            tape,
            layer_params,
            layer_fn,
        )?;
        self.log_ckpt_peak(batch, seq_len, floor, store, tape)?;
        Ok(out)
    }

    /// Print the modeled post-checkpoint peak against the measured one, once per
    /// new high-water. `should_checkpoint` only answers "would the FULL tape
    /// fit" — nothing modeled what checkpointing actually leaves resident, so
    /// every memory wall cost a ~3000 s run to find. The model: one layer is
    /// live at a time, so peak = the floor measured just before the stack + the
    /// widest single layer + the saved checkpoints that stayed on device.
    ///
    /// Treat the number as a FIRST MEASUREMENT, not a validated model. The two
    /// H20 points on hand (seq 65536 → 66987 MiB, 131072 → 81451 MiB) are two
    /// unknowns in two equations and cannot validate anything, and no constant
    /// here was fitted to close a gap. `drift` is the output that matters.
    fn log_ckpt_peak(
        &self,
        batch: usize,
        seq_len: usize,
        floor: Option<u64>,
        store: &TensorStore,
        tape: &Tape,
    ) -> Result<()> {
        // Without both the floor and the watermark there is no drift to report.
        let (Some(floor), Some(actual)) = (floor, store.backend().mem_pool_used_high()) else {
            return Ok(());
        };
        static HIGH: AtomicU64 = AtomicU64::new(0);
        if HIGH.fetch_max(actual, Ordering::Relaxed) >= actual {
            return Ok(());
        }
        let cfg = &self.config;
        // Group size is 1, so the peak is the largest single layer.
        let mut layer = 0;
        for l in &self.layers {
            layer = layer.max(match &l.self_attn {
                Qwen35Attention::Linear(_) => la_layer_peak_bytes(cfg, batch, seq_len),
                Qwen35Attention::Full(_) => {
                    full_attn_layer_peak_bytes(cfg, self.tp, batch, seq_len)?
                }
            });
        }
        // Offloaded groups leave nothing behind; resident ones keep one hidden each.
        let saved = if tape.offload_checkpoints() {
            0
        } else {
            batch
                .saturating_mul(seq_len)
                .saturating_mul(cfg.hidden_size * 4)
                .saturating_mul(self.layers.len())
        };
        let modeled = floor.saturating_add(layer.saturating_add(saved) as u64);
        let mib = |bytes: u64| bytes >> 20;
        eprintln!(
            "[ckpt-peak] unvalidated batch={batch} seq={seq_len} floor={}MiB layer={}MiB \
             saved={}MiB modeled={}MiB actual={}MiB drift={:+}MiB",
            mib(floor),
            mib(layer as u64),
            mib(saved as u64),
            mib(modeled),
            mib(actual),
            mib(actual) as i64 - mib(modeled) as i64,
        );
        Ok(())
    }

    pub fn supports_rollout_kv_cache(&self) -> bool {
        !self.tp.is_enabled()
            && self
                .layers
                .iter()
                .all(|layer| matches!(layer.self_attn, Qwen35Attention::Full(_)))
    }

    fn ensure_rollout_cache_supported(&self) -> Result<()> {
        if self.tp.is_enabled() {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache requires a non-tensor-parallel train model",
            ));
        }
        Ok(())
    }

    pub fn new_for_eval(cfg: &Qwen35Config, store: &mut TensorStore) -> Result<Self> {
        Self::new_internal(
            cfg,
            None,
            LoraTargetSet::AllLinear,
            None,
            false,
            TpContext::single(),
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: true,
            },
            store,
        )
    }

    pub(crate) fn new_for_checkpoint_load(
        cfg: &Qwen35Config,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            None,
            LoraTargetSet::AllLinear,
            None,
            false,
            TpContext::single(),
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: false,
            },
            store,
        )
    }

    pub fn new_with_lora(
        cfg: &Qwen35Config,
        lora: Option<LoraConfig>,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            lora,
            LoraTargetSet::AllLinear,
            None,
            false,
            TpContext::single(),
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: true,
            },
            store,
        )
    }

    pub fn new_with_lora_targets(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_with_lora_targets_layer_start(cfg, lora, target_set, None, store)
    }

    pub fn new_with_lora_targets_layer_start(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            Some(lora),
            target_set,
            lora_layer_start,
            false,
            TpContext::single(),
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: true,
            },
            store,
        )
    }

    pub(crate) fn new_with_lora_targets_for_checkpoint_load(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_skip_experts: bool,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_with_lora_targets_for_checkpoint_load_layer_start(
            cfg,
            lora,
            target_set,
            None,
            lora_skip_experts,
            store,
        )
    }

    pub(crate) fn new_with_lora_targets_for_checkpoint_load_layer_start(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        lora_skip_experts: bool,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            Some(lora),
            target_set,
            lora_layer_start,
            lora_skip_experts,
            TpContext::single(),
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: false,
            },
            store,
        )
    }

    #[doc(hidden)]
    pub fn new_with_lora_targets_and_tp(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        tp: TpContext,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_with_lora_targets_and_tp_layer_start(cfg, lora, target_set, None, tp, store)
    }

    #[doc(hidden)]
    pub fn new_with_lora_targets_and_tp_layer_start(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        tp: TpContext,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            Some(lora),
            target_set,
            lora_layer_start,
            false,
            tp,
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: true,
            },
            store,
        )
    }

    pub fn new_lora_from_base(
        base: &Qwen35Model,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_lora_from_base_layer_start(base, lora, target_set, None, store)
    }

    pub fn new_lora_from_base_layer_start(
        base: &Qwen35Model,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        store: &mut TensorStore,
    ) -> Result<Self> {
        let mut model = Self::new_with_lora_targets_and_tp_layer_start(
            &base.config,
            lora,
            target_set,
            lora_layer_start,
            base.tp,
            store,
        )?;
        model.gradient_checkpointing = base.gradient_checkpointing;
        model.share_base_parameters_from(base)?;

        let keep = base
            .all_parameter_ids()
            .into_iter()
            .chain(model.all_parameter_ids())
            .collect::<HashSet<_>>();
        store.retain_ids(&keep);
        Ok(model)
    }

    fn new_internal(
        cfg: &Qwen35Config,
        lora: Option<LoraConfig>,
        lora_target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        lora_skip_experts: bool,
        tp: TpContext,
        mode: Qwen35InitMode,
        store: &mut TensorStore,
    ) -> Result<Self> {
        match mode {
            Qwen35InitMode::ScratchTrain => cfg.validate_train_scratch_contract()?,
            Qwen35InitMode::LoraOrFrozen { .. } => cfg.validate_train_lora_or_frozen_contract()?,
        }
        tp.validate(cfg)?;
        if let Some(start) = lora_layer_start
            && start >= cfg.num_hidden_layers
        {
            return Err(Qwen35Error::InvalidConfig(
                "lora_layer_start must be less than num_hidden_layers",
            ));
        }
        let mut param_names = HashMap::new();
        let mut adapter_names = HashMap::new();
        let mut param_ids = Vec::new();
        let mut seen = HashSet::new();
        let mut register_named =
            |target: &mut HashMap<&'static str, TensorId>, name: &'static str, id: TensorId| {
                target.insert(name, id);
                if seen.insert(id) {
                    param_ids.push(id);
                }
            };
        let base_requires_grad = matches!(mode, Qwen35InitMode::ScratchTrain) && lora.is_none();
        let materialize_frozen_base = match mode {
            Qwen35InitMode::ScratchTrain => true,
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base,
            } => materialize_frozen_base,
        };

        let embed_tokens_name = cfg.embed_tokens_tensor_name();
        let embed_tokens = normal_or_unmaterialized_parameter(
            embed_tokens_name,
            &[cfg.vocab_size, cfg.hidden_size],
            0.02,
            base_requires_grad,
            materialize_frozen_base,
            store,
        )?;
        register_named(&mut param_names, embed_tokens_name, embed_tokens);

        let lm_head_name = cfg.lm_head_tensor_name();
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens
        } else {
            normal_or_unmaterialized_parameter(
                lm_head_name,
                &[cfg.vocab_size, cfg.hidden_size],
                0.02,
                base_requires_grad,
                materialize_frozen_base,
                store,
            )?
        };
        register_named(&mut param_names, lm_head_name, lm_head);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_lora = lora_for_layer(lora, lora_layer_start, layer_idx);
            let names = cfg.layer_tensor_names(layer_idx);
            let input_layernorm_name = leak_name(names.common.input_layernorm.clone());
            let post_attention_layernorm_name =
                leak_name(names.common.post_attention_layernorm.clone());

            let input_layernorm = ones_or_unmaterialized_parameter(
                input_layernorm_name,
                &[cfg.hidden_size],
                base_requires_grad,
                materialize_frozen_base,
                store,
            )?;
            let mlp = if cfg.is_moe_layer(layer_idx) {
                if !cfg.norm_topk_prob {
                    return Err(Qwen35Error::InvalidConfig(
                        "train-side Qwen3.6 MoE currently requires norm_topk_prob=true",
                    ));
                }
                let moe_names = names.common.moe_tensor_names();
                Qwen35Mlp::Sparse(Box::new(new_sparse_mlp(
                    &moe_names,
                    cfg,
                    tp,
                    base_requires_grad,
                    materialize_frozen_base,
                    layer_lora,
                    lora_target_set,
                    lora_skip_experts,
                    store,
                )?))
            } else {
                let gate_proj_name = leak_name(names.common.mlp_gate_proj);
                let up_proj_name = leak_name(names.common.mlp_up_proj);
                let down_proj_name = leak_name(names.common.mlp_down_proj);
                Qwen35Mlp::Dense(Box::new(Qwen35DenseMlp {
                    gate_proj: linear_with_base_init(
                        gate_proj_name,
                        cfg.hidden_size,
                        tp.local_intermediate_size(cfg)?,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, gate_proj_name),
                        materialize_frozen_base,
                        store,
                    )?,
                    up_proj: linear_with_base_init(
                        up_proj_name,
                        cfg.hidden_size,
                        tp.local_intermediate_size(cfg)?,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, up_proj_name),
                        materialize_frozen_base,
                        store,
                    )?,
                    down_proj: linear_with_base_init(
                        down_proj_name,
                        tp.local_intermediate_size(cfg)?,
                        cfg.hidden_size,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, down_proj_name),
                        materialize_frozen_base,
                        store,
                    )?,
                }))
            };
            let post_attention_layernorm = ones_or_unmaterialized_parameter(
                post_attention_layernorm_name,
                &[cfg.hidden_size],
                base_requires_grad,
                materialize_frozen_base,
                store,
            )?;

            register_named(&mut param_names, input_layernorm_name, input_layernorm);
            register_mlp(
                &mut param_names,
                &mut adapter_names,
                &mut register_named,
                &mlp,
            );
            register_named(
                &mut param_names,
                post_attention_layernorm_name,
                post_attention_layernorm,
            );

            let self_attn = match names.attention {
                Qwen35AttentionTensorNames::Full(attn_names) => {
                    let q_proj_name = leak_name(attn_names.q_proj);
                    let k_proj_name = leak_name(attn_names.k_proj);
                    let v_proj_name = leak_name(attn_names.v_proj);
                    let o_proj_name = leak_name(attn_names.o_proj);
                    let q_norm_name = leak_name(attn_names.q_norm);
                    let k_norm_name = leak_name(attn_names.k_norm);

                    let q_proj = linear_with_base_init(
                        q_proj_name,
                        cfg.hidden_size,
                        tp.full_attn_q_proj_dim(cfg)?,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, q_proj_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let k_proj = linear_with_base_init(
                        k_proj_name,
                        cfg.hidden_size,
                        tp.full_attn_kv_dim(cfg)?,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, k_proj_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let v_proj = linear_with_base_init(
                        v_proj_name,
                        cfg.hidden_size,
                        tp.full_attn_kv_dim(cfg)?,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, v_proj_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let o_proj = linear_with_base_init(
                        o_proj_name,
                        tp.full_attn_q_dim(cfg)?,
                        cfg.hidden_size,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, o_proj_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let q_norm = ones_or_unmaterialized_parameter(
                        q_norm_name,
                        &[cfg.head_dim],
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;
                    let k_norm = ones_or_unmaterialized_parameter(
                        k_norm_name,
                        &[cfg.head_dim],
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;

                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &q_proj,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &k_proj,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &v_proj,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &o_proj,
                    );
                    register_named(&mut param_names, q_norm_name, q_norm);
                    register_named(&mut param_names, k_norm_name, k_norm);

                    Qwen35Attention::Full(Qwen35FullAttention {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                    })
                }
                Qwen35AttentionTensorNames::Linear(attn_names) => {
                    let in_proj_qkv_name = leak_name(attn_names.in_proj_qkv);
                    let in_proj_z_name = leak_name(attn_names.in_proj_z);
                    let in_proj_b_name = leak_name(attn_names.in_proj_b);
                    let in_proj_a_name = leak_name(attn_names.in_proj_a);
                    let conv1d_weight_name = leak_name(attn_names.conv1d_weight);
                    let dt_bias_name = leak_name(attn_names.dt_bias);
                    let a_log_name = leak_name(attn_names.a_log);
                    let norm_name = leak_name(attn_names.norm);
                    let out_proj_name = leak_name(attn_names.out_proj);

                    let in_proj_qkv = linear_with_base_init(
                        in_proj_qkv_name,
                        cfg.hidden_size,
                        cfg.linear_attn_qkv_dim(),
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, in_proj_qkv_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let in_proj_z = linear_with_base_init(
                        in_proj_z_name,
                        cfg.hidden_size,
                        cfg.linear_attn_z_dim(),
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, in_proj_z_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let in_proj_b = linear_with_base_init(
                        in_proj_b_name,
                        cfg.hidden_size,
                        cfg.linear_num_value_heads,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, in_proj_b_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let in_proj_a = linear_with_base_init(
                        in_proj_a_name,
                        cfg.hidden_size,
                        cfg.linear_num_value_heads,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, in_proj_a_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let conv1d_weight = normal_or_unmaterialized_parameter(
                        conv1d_weight_name,
                        &[cfg.linear_attn_qkv_dim(), cfg.linear_conv_kernel_dim],
                        0.02,
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;
                    let dt_bias = normal_or_unmaterialized_parameter(
                        dt_bias_name,
                        &[cfg.linear_num_value_heads],
                        0.02,
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;
                    let a_log = normal_or_unmaterialized_parameter(
                        a_log_name,
                        &[cfg.linear_num_value_heads],
                        0.02,
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;
                    let norm = ones_or_unmaterialized_parameter(
                        norm_name,
                        &[cfg.linear_value_head_dim],
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;
                    let out_proj = linear_with_base_init(
                        out_proj_name,
                        cfg.linear_attn_z_dim(),
                        cfg.hidden_size,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, out_proj_name),
                        materialize_frozen_base,
                        store,
                    )?;

                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &in_proj_qkv,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &in_proj_z,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &in_proj_b,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &in_proj_a,
                    );
                    register_named(&mut param_names, conv1d_weight_name, conv1d_weight);
                    register_named(&mut param_names, dt_bias_name, dt_bias);
                    register_named(&mut param_names, a_log_name, a_log);
                    register_named(&mut param_names, norm_name, norm);
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &out_proj,
                    );

                    Qwen35Attention::Linear(Qwen35LinearAttention {
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_b,
                        in_proj_a,
                        conv1d_weight,
                        dt_bias,
                        a_log,
                        norm,
                        out_proj,
                    })
                }
            };

            layers.push(Qwen35Layer {
                input_layernorm,
                self_attn,
                post_attention_layernorm,
                mlp,
            });
        }

        let final_norm_name = cfg.norm_tensor_name();
        let final_norm = ones_or_unmaterialized_parameter(
            final_norm_name,
            &[cfg.hidden_size],
            base_requires_grad,
            materialize_frozen_base,
            store,
        )?;
        register_named(&mut param_names, final_norm_name, final_norm);

        let (cos_cache, sin_cache) = build_rope_cache(cfg, store)?;
        if seen.insert(cos_cache) {
            param_ids.push(cos_cache);
        }
        if seen.insert(sin_cache) {
            param_ids.push(sin_cache);
        }

        Ok(Self {
            config: cfg.clone(),
            tp,
            lora,
            lora_target_set,
            lora_layer_start,
            lora_skip_experts,
            layers,
            embed_tokens,
            final_norm,
            lm_head,
            cos_cache,
            sin_cache,
            param_names,
            adapter_names,
            param_ids,
            gradient_checkpointing: false,
        })
    }

    pub fn all_parameter_ids(&self) -> Vec<TensorId> {
        self.param_ids.clone()
    }

    pub fn lm_head_weight_id(&self) -> TensorId {
        self.lm_head
    }

    fn share_base_parameters_from(&mut self, base: &Qwen35Model) -> Result<()> {
        if self.layers.len() != base.layers.len() {
            return Err(Qwen35Error::InvalidConfig(
                "cannot share Qwen3.5 base weights across mismatched layer counts",
            ));
        }

        self.embed_tokens = base.embed_tokens;
        self.final_norm = base.final_norm;
        self.lm_head = base.lm_head;
        self.cos_cache = base.cos_cache;
        self.sin_cache = base.sin_cache;

        for (layer, base_layer) in self.layers.iter_mut().zip(&base.layers) {
            layer.input_layernorm = base_layer.input_layernorm;
            layer.post_attention_layernorm = base_layer.post_attention_layernorm;
            share_base_attention(&mut layer.self_attn, &base_layer.self_attn)?;
            share_base_mlp(&mut layer.mlp, &base_layer.mlp)?;
        }

        self.param_names = base.param_names.clone();
        let adapter_ids = self.adapter_names.values().copied().collect::<HashSet<_>>();
        let mut seen = HashSet::new();
        let param_ids: Vec<_> = base
            .param_ids
            .iter()
            .copied()
            .chain(
                self.param_ids
                    .iter()
                    .copied()
                    .filter(|id| adapter_ids.contains(id)),
            )
            .filter(|id| seen.insert(*id))
            .collect();
        self.param_ids = param_ids;
        Ok(())
    }

    fn detach_before_lora_layer(
        &self,
        hidden: TensorId,
        layer_index: usize,
        store: &mut TensorStore,
        tape: &Tape,
    ) -> Result<TensorId> {
        if tape.enabled && self.lora_layer_start == Some(layer_index) {
            Ok(store.detach(hidden)?)
        } else {
            Ok(hidden)
        }
    }

    pub fn clone_frozen(&self, store: &mut TensorStore) -> Self {
        let cloned = Self::new_internal(
            &self.config,
            self.lora,
            self.lora_target_set,
            self.lora_layer_start,
            self.lora_skip_experts,
            self.tp,
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: true,
            },
            store,
        )
        .expect("clone_frozen should preserve config");
        copy_frozen_tensor_map(&self.param_names, &cloned.param_names, store);
        copy_frozen_tensor_map(&self.adapter_names, &cloned.adapter_names, store);
        copy_frozen_tensor(self.cos_cache, cloned.cos_cache, store);
        copy_frozen_tensor(self.sin_cache, cloned.sin_cache, store);

        cloned
    }

    pub fn forward_tokens(
        &self,
        input_ids: &[usize],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> autograd::Result<TensorId> {
        self.forward_batch_tokens(input_ids, 1, input_ids.len(), store, tape)
    }

    pub fn forward_batch_tokens(
        &self,
        input_ids: &[usize],
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> autograd::Result<TensorId> {
        let position_ids = (0..seq_len).collect::<Vec<_>>();
        self.forward_batch_tokens_with_positions(input_ids, &position_ids, batch, store, tape)
    }

    pub fn forward_batch_tokens_with_positions(
        &self,
        input_ids: &[usize],
        position_ids: &[usize],
        batch: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> autograd::Result<TensorId> {
        self.forward_batch_indices(store, tape, input_ids, position_ids, batch)
            .map_err(qwen35_to_autograd)
    }

    /// Batched sibling of [`Self::forward_hidden_states`]: post-final-norm hidden
    /// `[batch, seq_len, hidden]` for `input_ids` (row-major, len `batch*seq_len`),
    /// positions `0..seq_len` per row. The logits path (`forward_batch_tokens`) is
    /// this plus `lm_head`; callers that project only a few masked positions (fused
    /// chunked CE) take the hidden and skip the full `[batch, seq_len, vocab]` tile.
    pub fn forward_batch_hidden(
        &self,
        input_ids: &[usize],
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> autograd::Result<TensorId> {
        let position_ids = (0..seq_len).collect::<Vec<_>>();
        self.forward_batch_hidden_indices(
            store,
            tape,
            input_ids,
            &position_ids,
            batch,
            crate::context_parallel::CpContext::single(),
        )
        .map_err(qwen35_to_autograd)
    }

    pub fn forward_batch(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        batch: usize,
        seq_len: usize,
    ) -> Result<TensorId> {
        if input_ids.len() != batch * seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: batch * seq_len,
            });
        }
        if position_ids.len() != seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: position_ids.len(),
                expected_len: seq_len,
            });
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if seq_len > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "sequence length exceeds configured rope cache length",
            ));
        }

        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_indices(store, tape, &token_indices, &positions, batch)
    }

    fn forward_rollout_cached(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        cache: &mut Qwen35KvCache,
    ) -> Result<TensorId> {
        self.ensure_rollout_cache_supported()?;
        if input_ids.len() != position_ids.len() {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: position_ids.len(),
            });
        }
        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_indices_with_kv_cache(store, tape, &token_indices, &positions, cache)
    }

    fn forward_rollout_cached_profiled(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        cache: &mut Qwen35KvCache,
    ) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
        self.ensure_rollout_cache_supported()?;
        if input_ids.len() != position_ids.len() {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: position_ids.len(),
            });
        }
        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_indices_with_kv_cache_profiled(
            store,
            tape,
            &token_indices,
            &positions,
            cache,
        )
    }

    fn forward_rollout_cached_device_token(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_id: TensorId,
        position_id: u32,
        cache: &mut Qwen35KvCache,
    ) -> Result<TensorId> {
        self.ensure_rollout_cache_supported()?;
        if tape.enabled {
            return Err(Qwen35Error::InvalidConfig(
                "device-token rollout requires tape disabled",
            ));
        }
        if cache.layers.len() != self.layers.len() {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer count does not match model",
            ));
        }
        let position = position_id as usize;
        if position != cache.seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache device token requires position equal to cache length",
            ));
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if cache.seq_len + 1 > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache sequence length exceeds configured rope cache length",
            ));
        }

        let q_start = cache.seq_len;
        let cos = select_cache_rows(self.cos_cache, &[position], store)?;
        let sin = select_cache_rows(self.sin_cache, &[position], store)?;

        let mut hidden = embedding_device_f32_ids(self.embed_tokens, token_id, 1, store)?;
        hidden = reshape(hidden, &[1, 1, self.config.hidden_size], store, tape)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut cache.layers[layer_index];
            hidden = layer.forward_with_kv_cache(
                hidden,
                &self.config,
                cos,
                sin,
                layer_cache,
                q_start,
                store,
                tape,
            )?;
        }
        cache.seq_len += 1;
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        linear_forward(hidden, self.lm_head, store, tape)
    }

    fn forward_rollout_cached_device_token_profiled(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_id: TensorId,
        position_id: u32,
        cache: &mut Qwen35KvCache,
    ) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
        self.ensure_rollout_cache_supported()?;
        let total_started = Instant::now();
        let mut profile = Qwen35RolloutForwardProfile::default();

        if tape.enabled {
            return Err(Qwen35Error::InvalidConfig(
                "device-token rollout requires tape disabled",
            ));
        }
        if cache.layers.len() != self.layers.len() {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer count does not match model",
            ));
        }
        let position = position_id as usize;
        if position != cache.seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache device token requires position equal to cache length",
            ));
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if cache.seq_len + 1 > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache sequence length exceeds configured rope cache length",
            ));
        }

        let q_start = cache.seq_len;
        let started = Instant::now();
        let cos = select_cache_rows(self.cos_cache, &[position], store)?;
        let sin = select_cache_rows(self.sin_cache, &[position], store)?;
        profile.cache_select += started.elapsed();

        let started = Instant::now();
        let mut hidden = embedding_device_f32_ids(self.embed_tokens, token_id, 1, store)?;
        hidden = reshape(hidden, &[1, 1, self.config.hidden_size], store, tape)?;
        profile.embedding += started.elapsed();

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut cache.layers[layer_index];
            let (next_hidden, layer_profile) = layer.forward_with_kv_cache_profiled(
                hidden,
                &self.config,
                cos,
                sin,
                layer_cache,
                q_start,
                store,
                tape,
            )?;
            hidden = next_hidden;
            profile.layers.push(layer_profile);
        }
        cache.seq_len += 1;

        let started = Instant::now();
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        profile.final_norm += started.elapsed();

        let started = Instant::now();
        let logits = linear_forward(hidden, self.lm_head, store, tape)?;
        profile.lm_head += started.elapsed();
        profile.total = total_started.elapsed();
        Ok((logits, profile))
    }

    fn forward_batch_indices(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        batch: usize,
    ) -> Result<TensorId> {
        let hidden = self.forward_batch_hidden_indices(
            store,
            tape,
            token_indices,
            positions,
            batch,
            crate::context_parallel::CpContext::single(),
        )?;
        linear_forward(hidden, self.lm_head, store, tape)
    }

    fn forward_batch_indices_profiled(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        batch: usize,
        trace: bool,
    ) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
        let total_started = Instant::now();
        let mut profile = Qwen35RolloutForwardProfile::default();
        let seq_len = positions.len();
        if token_indices.len() != batch * seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: token_indices.len(),
                expected_len: batch * seq_len,
            });
        }

        let started = Instant::now();
        let cos = select_cache_rows(self.cos_cache, positions, store)?;
        let sin = select_cache_rows(self.sin_cache, positions, store)?;
        profile.cache_select += started.elapsed();
        trace_model_component(trace, "cache_select", profile.cache_select);

        let started = Instant::now();
        let mut hidden = embedding(self.embed_tokens, token_indices, store, tape)?;
        hidden = reshape(
            hidden,
            &[batch, seq_len, self.config.hidden_size],
            store,
            tape,
        )?;
        profile.embedding += started.elapsed();
        trace_model_component(trace, "embedding", profile.embedding);

        if self.should_checkpoint(batch, seq_len, store, tape) {
            // Checkpoint layers in adaptive groups (one host offload per
            // group). Numerically exact vs per-layer: same layer order, same
            // detach point (the LoRA boundary forces a group split), same params
            // in each checkpoint's saved inputs.
            let layers = Arc::new(self.layers.clone());
            let cfg = self.config.clone();
            let tp = self.tp;
            let (cos_id, sin_id) = (cos, sin);
            let layer_fn = {
                let layers = Arc::clone(&layers);
                move |idx: usize, h, s: &mut TensorStore, t: &mut Tape| {
                    layers[idx]
                        .forward(
                            h,
                            &cfg,
                            tp,
                            crate::context_parallel::CpContext::single(),
                            cos_id,
                            sin_id,
                            None,
                            s,
                            t,
                        )
                        .map_err(qwen35_to_autograd)
                }
            };
            // Param ids only read `requires_grad`; precompute so `layer_params`
            // doesn't alias the mutable `store` borrow during the call.
            let param_ids: Vec<Vec<TensorId>> = self
                .layers
                .iter()
                .map(|l| l.checkpoint_param_ids(self.lora_skip_experts, store))
                .collect();
            hidden = self.checkpoint_layers(
                hidden,
                batch,
                seq_len,
                store,
                tape,
                |idx| param_ids[idx].clone(),
                layer_fn,
            )?;
            // Grouped path records no per-layer profile; keep profile.layers len.
            profile
                .layers
                .resize_with(self.layers.len(), Qwen35LayerForwardProfile::default);
        } else {
            for (layer_index, layer) in self.layers.iter().enumerate() {
                hidden = self.detach_before_lora_layer(hidden, layer_index, store, tape)?;
                let started = Instant::now();
                let (next_hidden, layer_profile) = layer.forward_profiled(
                    layer_index,
                    hidden,
                    &self.config,
                    self.tp,
                    cos,
                    sin,
                    trace,
                    store,
                    tape,
                )?;
                hidden = next_hidden;
                trace_model_component(trace, "layer_total", started.elapsed());
                profile.layers.push(layer_profile);
            }
        }

        let started = Instant::now();
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        profile.final_norm += started.elapsed();
        trace_model_component(trace, "final_norm", profile.final_norm);

        let started = Instant::now();
        let logits = linear_forward(hidden, self.lm_head, store, tape)?;
        profile.lm_head += started.elapsed();
        trace_model_component(trace, "lm_head", profile.lm_head);
        profile.total = total_started.elapsed();
        trace_model_component(trace, "total", profile.total);
        Ok((logits, profile))
    }

    fn forward_batch_hidden_indices(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        batch: usize,
        cp: crate::context_parallel::CpContext,
    ) -> Result<TensorId> {
        let seq_len = positions.len();
        if token_indices.len() != batch * seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: token_indices.len(),
                expected_len: batch * seq_len,
            });
        }
        let cos = select_cache_rows(self.cos_cache, positions, store)?;
        let sin = select_cache_rows(self.sin_cache, positions, store)?;

        // Under CP the ring masks by the shard's absolute row positions — the SAME
        // slice that built cos/sin above. Shared as `Arc<[usize]>` so the `'static`
        // checkpoint closure can capture it; `None` off-CP (contiguous, unused).
        let cp_positions: Option<Arc<[usize]>> =
            cp.is_enabled().then(|| Arc::from(positions.to_vec()));

        let mut hidden = embedding(self.embed_tokens, token_indices, store, tape)?;
        hidden = reshape(
            hidden,
            &[batch, seq_len, self.config.hidden_size],
            store,
            tape,
        )?;
        if self.should_checkpoint(batch, seq_len, store, tape) {
            // Group-checkpoint (see forward_batch_indices_profiled) — exact vs
            // per-layer.
            let layers = Arc::new(self.layers.clone());
            let cfg = self.config.clone();
            let tp = self.tp;
            let (cos_id, sin_id) = (cos, sin);

            // Per-layer wall aggregation for ARLE_OPD_PROFILE=1 (this is the
            // masked-writeback forward path). Each layer_fn call is timed and —
            // when ARLE_OPD_PROFILE_SYNC=1 — stream-synced so the measurement
            // captures kernel wall not enqueue latency. Backward recompute
            // re-invokes layer_fn, so counts[idx] surfaces the recompute
            // multiplier. Both env vars default OFF: shipping path unchanged.
            let profile_enabled = std::env::var("ARLE_OPD_PROFILE").is_ok();
            let profile_sync = std::env::var("ARLE_OPD_PROFILE_SYNC").is_ok();
            let num_layers = self.layers.len();
            let layer_times: Arc<Mutex<Vec<Duration>>> =
                Arc::new(Mutex::new(vec![Duration::default(); num_layers]));
            let layer_counts: Arc<Mutex<Vec<usize>>> =
                Arc::new(Mutex::new(vec![0usize; num_layers]));

            let layer_fn = {
                let layers = Arc::clone(&layers);
                let layer_times = Arc::clone(&layer_times);
                let layer_counts = Arc::clone(&layer_counts);
                let cp_positions = cp_positions.clone();
                move |idx: usize, h, s: &mut TensorStore, t: &mut Tape| {
                    let pos = cp_positions.as_deref();
                    if !profile_enabled {
                        return layers[idx]
                            .forward(h, &cfg, tp, cp, cos_id, sin_id, pos, s, t)
                            .map_err(qwen35_to_autograd);
                    }
                    if profile_sync {
                        let _ = s.backend().stream_synchronize();
                    }
                    let t0 = Instant::now();
                    let result = layers[idx]
                        .forward(h, &cfg, tp, cp, cos_id, sin_id, pos, s, t)
                        .map_err(qwen35_to_autograd);
                    if profile_sync {
                        let _ = s.backend().stream_synchronize();
                    }
                    let dt = t0.elapsed();
                    if let Ok(mut times) = layer_times.lock() {
                        times[idx] += dt;
                    }
                    if let Ok(mut counts) = layer_counts.lock() {
                        counts[idx] += 1;
                    }
                    result
                }
            };
            // See forward_batch_indices_profiled: precompute param ids so the
            // closure doesn't alias the mutable `store` borrow.
            let param_ids: Vec<Vec<TensorId>> = self
                .layers
                .iter()
                .map(|l| l.checkpoint_param_ids(self.lora_skip_experts, store))
                .collect();
            hidden = self.checkpoint_layers(
                hidden,
                batch,
                seq_len,
                store,
                tape,
                |idx| param_ids[idx].clone(),
                layer_fn,
            )?;

            if profile_enabled
                && let (Ok(times), Ok(counts)) = (layer_times.lock(), layer_counts.lock())
            {
                let total: Duration = times.iter().sum();
                eprintln!(
                    "[opd-profile] masked-writeback forward layer wall \
                         (checkpointed, sync={profile_sync}); counts include backward recompute:"
                );
                for (idx, (t, c)) in times.iter().zip(counts.iter()).enumerate() {
                    eprintln!(
                        "[opd-profile]   layer[{idx:>2}] wall={:>10.3}ms calls={}",
                        t.as_secs_f64() * 1000.0,
                        c
                    );
                }
                eprintln!(
                    "[opd-profile] forward layers sum: {:.3}s across {num_layers} layers",
                    total.as_secs_f64()
                );
            }
        } else {
            // #170 attribution probe: full-tape per-layer VRAM ramp. free is
            // pool-reserved-aware (cuMemGetInfo), so slope = tape + transients.
            let vram_ramp = std::env::var("ARLE_OPD_VRAM_TRACE").is_ok();
            for (layer_index, layer) in self.layers.iter().enumerate() {
                hidden = self.detach_before_lora_layer(hidden, layer_index, store, tape)?;
                hidden = layer.forward(
                    hidden,
                    &self.config,
                    self.tp,
                    cp,
                    cos,
                    sin,
                    cp_positions.as_deref(),
                    store,
                    tape,
                )?;
                if vram_ramp && let Some((free, _)) = store.backend().device_mem_info() {
                    eprintln!("[vram-ramp] layer={layer_index} free={free}");
                }
            }
        }
        qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )
    }

    fn forward_batch_indices_with_kv_cache(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        cache: &mut Qwen35KvCache,
    ) -> Result<TensorId> {
        self.ensure_rollout_cache_supported()?;
        let seq_len = positions.len();
        if token_indices.len() != seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: token_indices.len(),
                expected_len: seq_len,
            });
        }
        if cache.layers.len() != self.layers.len() {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer count does not match model",
            ));
        }
        if seq_len == 0 {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache requires at least one token",
            ));
        }
        for (offset, &position) in positions.iter().enumerate() {
            if position != cache.seq_len + offset {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache requires contiguous positions starting at cache length",
                ));
            }
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if cache.seq_len + seq_len > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache sequence length exceeds configured rope cache length",
            ));
        }

        let q_start = cache.seq_len;
        let cos = select_cache_rows(self.cos_cache, positions, store)?;
        let sin = select_cache_rows(self.sin_cache, positions, store)?;

        let mut hidden = embedding(self.embed_tokens, token_indices, store, tape)?;
        hidden = reshape(hidden, &[1, seq_len, self.config.hidden_size], store, tape)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut cache.layers[layer_index];
            hidden = layer.forward_with_kv_cache(
                hidden,
                &self.config,
                cos,
                sin,
                layer_cache,
                q_start,
                store,
                tape,
            )?;
        }
        cache.seq_len += seq_len;
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        let hidden = if seq_len == 1 {
            hidden
        } else {
            slice(
                hidden,
                &[0, seq_len - 1, 0],
                &[1, seq_len, self.config.hidden_size],
                store,
                tape,
            )?
        };
        linear_forward(hidden, self.lm_head, store, tape)
    }

    fn forward_batch_indices_with_kv_cache_profiled(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        cache: &mut Qwen35KvCache,
    ) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
        self.ensure_rollout_cache_supported()?;
        let total_started = Instant::now();
        let mut profile = Qwen35RolloutForwardProfile::default();
        let seq_len = positions.len();
        if token_indices.len() != seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: token_indices.len(),
                expected_len: seq_len,
            });
        }
        if cache.layers.len() != self.layers.len() {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer count does not match model",
            ));
        }
        if seq_len == 0 {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache requires at least one token",
            ));
        }
        for (offset, &position) in positions.iter().enumerate() {
            if position != cache.seq_len + offset {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache requires contiguous positions starting at cache length",
                ));
            }
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if cache.seq_len + seq_len > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache sequence length exceeds configured rope cache length",
            ));
        }

        let q_start = cache.seq_len;
        let started = Instant::now();
        let cos = select_cache_rows(self.cos_cache, positions, store)?;
        let sin = select_cache_rows(self.sin_cache, positions, store)?;
        profile.cache_select += started.elapsed();

        let started = Instant::now();
        let mut hidden = embedding(self.embed_tokens, token_indices, store, tape)?;
        hidden = reshape(hidden, &[1, seq_len, self.config.hidden_size], store, tape)?;
        profile.embedding += started.elapsed();

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut cache.layers[layer_index];
            let (next_hidden, layer_profile) = layer.forward_with_kv_cache_profiled(
                hidden,
                &self.config,
                cos,
                sin,
                layer_cache,
                q_start,
                store,
                tape,
            )?;
            hidden = next_hidden;
            profile.layers.push(layer_profile);
        }
        cache.seq_len += seq_len;

        let started = Instant::now();
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        profile.final_norm += started.elapsed();

        let hidden = if seq_len == 1 {
            hidden
        } else {
            slice(
                hidden,
                &[0, seq_len - 1, 0],
                &[1, seq_len, self.config.hidden_size],
                store,
                tape,
            )?
        };

        let started = Instant::now();
        let logits = linear_forward(hidden, self.lm_head, store, tape)?;
        profile.lm_head += started.elapsed();
        profile.total = total_started.elapsed();
        Ok((logits, profile))
    }

    pub fn forward(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> Result<TensorId> {
        self.forward_batch(store, tape, input_ids, position_ids, 1, position_ids.len())
    }

    #[doc(hidden)]
    pub fn forward_profiled_for_diagnostics(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        trace: bool,
    ) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_indices_profiled(store, tape, &token_indices, &positions, 1, trace)
    }

    pub fn forward_hidden_states(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        cp: crate::context_parallel::CpContext,
    ) -> Result<TensorId> {
        if input_ids.len() != position_ids.len() {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: position_ids.len(),
            });
        }
        if input_ids.is_empty() {
            return Err(Qwen35Error::InvalidConfig(
                "hidden-state forward requires at least one token",
            ));
        }
        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_hidden_indices(store, tape, &token_indices, &positions, 1, cp)
    }

    /// OPD frozen-prompt-KV writeback forward: forward+backward ONLY the
    /// generated segment (`gen_ids` = rows `gen_start..seq_len`), seeding each
    /// layer's attention from the prompt prefix (`prompt_ids` = rows
    /// `0..gen_start`) captured off-tape. Returns `[1, gen_len, hidden]`.
    ///
    /// Two phases:
    ///  1. OFF-TAPE prompt pass (no checkpoint, no offload): embed the prompt,
    ///     run the layers with the tape disabled; at each layer capture the
    ///     prefix K/V (full) or boundary state+conv (linear) AND propagate the
    ///     prompt hidden so the next layer's capture sees the correct input. The
    ///     prompt hidden is discarded after.
    ///  2. TAPED gen pass: embed ONLY `gen_ids` fresh → `[1, gen_len, hidden]`
    ///     (RMSNorm + MLP are position-local, so feeding `embed(gen_ids)` is
    ///     exact — only attention reads the prefix). Run the layers via
    ///     `checkpoint_sequential` (or a per-layer loop when checkpointing is
    ///     off), each consuming its captured prefix, then final_norm.
    pub fn forward_hidden_states_gen_segment(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        prompt_ids: &[u32],
        gen_ids: &[u32],
        prompt_positions: &[u32],
        gen_positions: &[u32],
        cp: crate::context_parallel::CpContext,
    ) -> Result<TensorId> {
        if prompt_ids.len() != prompt_positions.len() {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: prompt_ids.len(),
                expected_len: prompt_positions.len(),
            });
        }
        if gen_ids.len() != gen_positions.len() {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: gen_ids.len(),
                expected_len: gen_positions.len(),
            });
        }
        if gen_ids.is_empty() {
            return Err(Qwen35Error::InvalidConfig(
                "frozen-prompt-KV forward requires at least one generated token",
            ));
        }
        let gen_start = prompt_ids.len();
        let gen_len = gen_ids.len();
        let batch = 1usize;

        // ---- PHASE 1: off-tape prompt prefix capture ----
        // Chunked: the prompt is processed in OPD_SEQ_CHUNK-row pieces so the
        // per-layer hidden/MLP transients stay O(chunk × hidden) instead of
        // O(prompt × hidden). Each layer's K/V is accumulated across chunks
        // (prefix + chunk) and used as the attention K/V for the next chunk.
        let prefix_cache = if gen_start > 0 {
            let prompt_token_indices = prompt_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
            let prompt_pos = prompt_positions
                .iter()
                .map(|&id| id as usize)
                .collect::<Vec<_>>();

            let mut prefix_tape = Tape::new();
            prefix_tape.set_enabled(false);

            let chunk = crate::runtime_flags::OPD_SEQ_CHUNK;
            let num_chunks = gen_start.div_ceil(chunk);
            // Accumulated K/V per layer (None before the first chunk touches it).
            let mut layer_prefix: Vec<Option<PrefixKv>> = vec![None; self.layers.len()];

            for c in 0..num_chunks {
                let start = c * chunk;
                let end = (start + chunk).min(gen_start);
                let chunk_len = end - start;
                let chunk_ids = &prompt_token_indices[start..end];
                let chunk_pos = &prompt_pos[start..end];
                let cos_chunk = select_cache_rows(self.cos_cache, chunk_pos, store)?;
                let sin_chunk = select_cache_rows(self.sin_cache, chunk_pos, store)?;

                let mut h = embedding(self.embed_tokens, chunk_ids, store, &mut prefix_tape)?;
                h = reshape(
                    h,
                    &[batch, chunk_len, self.config.hidden_size],
                    store,
                    &mut prefix_tape,
                )?;

                for (li, layer) in self.layers.iter().enumerate() {
                    let (next, prefix) = layer.forward_capture_prefix(
                        h,
                        &self.config,
                        self.tp,
                        cos_chunk,
                        sin_chunk,
                        batch,
                        chunk_len,
                        start,
                        layer_prefix[li].as_ref(),
                        store,
                        &mut prefix_tape,
                    )?;
                    h = next;
                    if let LayerPrefix::Full(kv) = prefix {
                        layer_prefix[li] = Some(kv);
                    }
                }
            }

            let layers = layer_prefix
                .into_iter()
                .map(|opt| {
                    opt.map(LayerPrefix::Full)
                        .expect("every full-attn layer must have captured prefix K/V")
                })
                .collect();
            WritebackPrefixCache { layers }
        } else {
            // gen_start == 0: no prefix; an empty cache forces the gen pass to seed
            // from zeros (equivalent to the full sequence with no prompt).
            return Err(Qwen35Error::InvalidConfig(
                "frozen-prompt-KV forward requires a non-empty prompt prefix",
            ));
        };

        // ---- PHASE 2: taped gen-segment forward ----
        let gen_token_indices = gen_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let gen_pos = gen_positions
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let cos_gen = select_cache_rows(self.cos_cache, &gen_pos, store)?;
        let sin_gen = select_cache_rows(self.sin_cache, &gen_pos, store)?;
        // CP: gen_positions are the absolute rows of this rank's gen shard; pass them
        // through so the ring kernel masks causally by true position. Off-CP: None.
        // Arc so checkpoint_sequential's per-group layer_fn.clone() is a refcount
        // bump, not a deep copy of the position vector.
        let cp_positions: Option<Arc<[usize]>> = cp.is_enabled().then(|| Arc::from(gen_pos));

        let mut hidden = embedding(self.embed_tokens, &gen_token_indices, store, tape)?;
        hidden = reshape(
            hidden,
            &[batch, gen_len, self.config.hidden_size],
            store,
            tape,
        )?;

        if self.should_checkpoint(batch, gen_len, store, tape) {
            let cache = Arc::new(prefix_cache);
            let layers = Arc::new(self.layers.clone());
            let cfg = self.config.clone();
            let tp = self.tp;
            let (cos_id, sin_id) = (cos_gen, sin_gen);
            let cp_positions = cp_positions.clone();
            let layer_fn = {
                let layers = Arc::clone(&layers);
                let cache = Arc::clone(&cache);
                move |idx: usize, h, s: &mut TensorStore, t: &mut Tape| {
                    layers[idx]
                        .forward_gen_segment(
                            h,
                            &cfg,
                            tp,
                            cos_id,
                            sin_id,
                            &cache.layers[idx],
                            batch,
                            gen_start,
                            gen_len,
                            cp,
                            cp_positions.as_deref(),
                            s,
                            t,
                        )
                        .map_err(qwen35_to_autograd)
                }
            };
            let param_ids: Vec<Vec<TensorId>> = self
                .layers
                .iter()
                .map(|l| l.checkpoint_param_ids(self.lora_skip_experts, store))
                .collect();
            hidden = self.checkpoint_layers(
                hidden,
                batch,
                gen_len,
                store,
                tape,
                |idx| param_ids[idx].clone(),
                layer_fn,
            )?;
        } else {
            for (layer_index, layer) in self.layers.iter().enumerate() {
                hidden = self.detach_before_lora_layer(hidden, layer_index, store, tape)?;
                hidden = layer.forward_gen_segment(
                    hidden,
                    &self.config,
                    self.tp,
                    cos_gen,
                    sin_gen,
                    &prefix_cache.layers[layer_index],
                    batch,
                    gen_start,
                    gen_len,
                    cp,
                    cp_positions.as_deref(),
                    store,
                    tape,
                )?;
            }
        }
        qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )
    }

    pub fn logits_from_hidden_window(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        hidden: TensorId,
        window: SequenceWindow,
    ) -> Result<TensorId> {
        if window.start >= window.end {
            return Err(Qwen35Error::InvalidConfig(
                "hidden logits window must be non-empty",
            ));
        }
        let hidden_shape = store
            .get(hidden)
            .ok_or(AutogradError::InvalidTensorId(hidden))?
            .shape
            .clone();
        if hidden_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "3",
                got: hidden_shape.len(),
            }
            .into());
        }
        if hidden_shape[0] != 1 || hidden_shape[2] != self.config.hidden_size {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![1, hidden_shape[1], self.config.hidden_size],
                got: hidden_shape.clone(),
            }
            .into());
        }
        if window.end > hidden_shape[1] {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: window.end,
                expected_len: hidden_shape[1],
            });
        }
        let hidden_window = if window.start == 0 && window.end == hidden_shape[1] {
            hidden
        } else {
            slice(
                hidden,
                &[0, window.start, 0],
                &[1, window.end, self.config.hidden_size],
                store,
                tape,
            )?
        };
        linear_forward(hidden_window, self.lm_head, store, tape)
    }

    #[doc(hidden)]
    pub fn forward_with_moe_routes_for_diagnostics(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> Result<(TensorId, Vec<Qwen35MoeRouteSignature>)> {
        let seq_len = position_ids.len();
        if input_ids.len() != seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: seq_len,
            });
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if seq_len > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "sequence length exceeds configured rope cache length",
            ));
        }

        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let cos = select_cache_rows(self.cos_cache, &positions, store)?;
        let sin = select_cache_rows(self.sin_cache, &positions, store)?;

        let mut hidden = embedding(self.embed_tokens, &token_indices, store, tape)?;
        hidden = reshape(hidden, &[1, seq_len, self.config.hidden_size], store, tape)?;
        let mut route_signatures = Vec::new();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_collecting_moe_routes(
                layer_index,
                hidden,
                &self.config,
                self.tp,
                cos,
                sin,
                &mut route_signatures,
                store,
                tape,
            )?;
        }
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        let logits = linear_forward(hidden, self.lm_head, store, tape)?;
        Ok((logits, route_signatures))
    }

    #[doc(hidden)]
    pub fn forward_with_frozen_moe_routes_for_diagnostics(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        frozen_routes: &[Qwen35MoeRouteSignature],
    ) -> Result<TensorId> {
        let seq_len = position_ids.len();
        if input_ids.len() != seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: seq_len,
            });
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if seq_len > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "sequence length exceeds configured rope cache length",
            ));
        }

        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let cos = select_cache_rows(self.cos_cache, &positions, store)?;
        let sin = select_cache_rows(self.sin_cache, &positions, store)?;

        let mut hidden = embedding(self.embed_tokens, &token_indices, store, tape)?;
        hidden = reshape(hidden, &[1, seq_len, self.config.hidden_size], store, tape)?;
        let mut next_route = 0usize;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_with_frozen_moe_routes(
                layer_index,
                hidden,
                &self.config,
                self.tp,
                cos,
                sin,
                frozen_routes,
                &mut next_route,
                store,
                tape,
            )?;
        }
        if next_route != frozen_routes.len() {
            return Err(Qwen35Error::InvalidConfig(
                "frozen MoE routes contain unused signatures",
            ));
        }
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        linear_forward(hidden, self.lm_head, store, tape)
    }

    pub fn forward_logits_window(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        window: SequenceWindow,
    ) -> Result<TensorId> {
        validate_sequence_window(input_ids, position_ids, window)?;
        let prefix_len = window.end;
        let token_indices = input_ids[..prefix_len]
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let positions = position_ids[..prefix_len]
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let hidden = self.forward_batch_hidden_indices(
            store,
            tape,
            &token_indices,
            &positions,
            1,
            crate::context_parallel::CpContext::single(),
        )?;
        self.logits_from_hidden_window(store, tape, hidden, window)
    }

    #[doc(hidden)]
    pub fn forward_mlp_for_diagnostics(
        &self,
        layer_idx: usize,
        hidden: TensorId,
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or(Qwen35Error::InvalidConfig(
                "diagnostic MLP layer index out of range",
            ))?;
        layer.forward_mlp(hidden, &self.config, self.tp, batch, seq_len, store, tape)
    }

    #[doc(hidden)]
    pub fn forward_lm_head_tail_for_diagnostics(
        &self,
        hidden: TensorId,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        linear_forward(hidden, self.lm_head, store, tape)
    }

    pub fn param_name_map(&self) -> HashMap<&'static str, TensorId> {
        self.param_names.clone()
    }

    pub fn adapter_name_map(&self) -> HashMap<&'static str, TensorId> {
        self.adapter_names.clone()
    }

    pub fn materialized_param_name_map(
        &self,
        store: &mut TensorStore,
    ) -> Result<HashMap<&'static str, TensorId>> {
        if self.lora.is_none() {
            return Ok(self.param_names.clone());
        }
        let mut map = self.param_names.clone();
        for layer in &self.layers {
            match &layer.self_attn {
                Qwen35Attention::Full(attn) => {
                    let merged_q = {
                        let tensor = attn.q_proj.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_k = {
                        let tensor = attn.k_proj.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_v = {
                        let tensor = attn.v_proj.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_o = {
                        let tensor = attn.o_proj.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    for (name, _) in attn.q_proj.parameter_name_map() {
                        map.insert(name, merged_q);
                    }
                    for (name, _) in attn.k_proj.parameter_name_map() {
                        map.insert(name, merged_k);
                    }
                    for (name, _) in attn.v_proj.parameter_name_map() {
                        map.insert(name, merged_v);
                    }
                    for (name, _) in attn.o_proj.parameter_name_map() {
                        map.insert(name, merged_o);
                    }
                }
                Qwen35Attention::Linear(attn) => {
                    let merged_qkv = {
                        let tensor = attn.in_proj_qkv.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_z = {
                        let tensor = attn.in_proj_z.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_b = {
                        let tensor = attn.in_proj_b.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_a = {
                        let tensor = attn.in_proj_a.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_out = {
                        let tensor = attn.out_proj.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    for (name, _) in attn.in_proj_qkv.parameter_name_map() {
                        map.insert(name, merged_qkv);
                    }
                    for (name, _) in attn.in_proj_z.parameter_name_map() {
                        map.insert(name, merged_z);
                    }
                    for (name, _) in attn.in_proj_b.parameter_name_map() {
                        map.insert(name, merged_b);
                    }
                    for (name, _) in attn.in_proj_a.parameter_name_map() {
                        map.insert(name, merged_a);
                    }
                    for (name, _) in attn.out_proj.parameter_name_map() {
                        map.insert(name, merged_out);
                    }
                }
            }
            insert_materialized_mlp_params(&mut map, &layer.mlp, store)?;
        }
        Ok(map)
    }
}

fn validate_sequence_window(
    input_ids: &[u32],
    position_ids: &[u32],
    window: SequenceWindow,
) -> Result<()> {
    if input_ids.len() != position_ids.len() {
        return Err(Qwen35Error::InputLenMismatch {
            input_len: input_ids.len(),
            expected_len: position_ids.len(),
        });
    }
    if window.start >= window.end {
        return Err(Qwen35Error::InvalidConfig(
            "sequence logits window must be non-empty",
        ));
    }
    if window.end > input_ids.len() {
        return Err(Qwen35Error::InputLenMismatch {
            input_len: window.end,
            expected_len: input_ids.len(),
        });
    }
    Ok(())
}

fn linear_forward(
    x: TensorId,
    weight: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x_shape = store
        .get(x)
        .ok_or(AutogradError::InvalidTensorId(x))?
        .shape
        .clone();
    let weight_shape = store
        .get(weight)
        .ok_or(AutogradError::InvalidTensorId(weight))?
        .shape
        .clone();
    if weight_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: weight_shape.len(),
        }
        .into());
    }

    let input_dim = *x_shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if input_dim != weight_shape[1] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![weight_shape[1]],
            got: vec![input_dim],
        }
        .into());
    }

    let prefix_elems = x_shape.iter().product::<usize>() / input_dim;
    let flat_x = reshape(x, &[prefix_elems, input_dim], store, tape)?;
    let projected = matmul_bt_with_site(flat_x, weight, store, tape, "lm_head")?;
    let mut output_shape = x_shape[..x_shape.len() - 1].to_vec();
    output_shape.push(weight_shape[0]);
    Ok(reshape(projected, &output_shape, store, tape)?)
}

fn register_linear(
    param_names: &mut HashMap<&'static str, TensorId>,
    adapter_names: &mut HashMap<&'static str, TensorId>,
    register_named: &mut impl FnMut(&mut HashMap<&'static str, TensorId>, &'static str, TensorId),
    linear: &LinearWithLora,
) {
    for (name, id) in linear.parameter_name_map() {
        register_named(param_names, name, id);
    }
    for (name, id) in linear.adapter_ordered() {
        register_named(adapter_names, name, id);
    }
}

fn linear_with_base_init(
    base_name: &'static str,
    in_features: usize,
    out_features: usize,
    base_requires_grad: bool,
    lora: Option<LoraConfig>,
    materialize_frozen_base: bool,
    store: &mut TensorStore,
) -> Result<LinearWithLora> {
    if materialize_frozen_base || base_requires_grad {
        Ok(LinearWithLora::new(
            base_name,
            in_features,
            out_features,
            base_requires_grad,
            lora,
            store,
        )?)
    } else {
        Ok(LinearWithLora::new_with_unmaterialized_base(
            base_name,
            in_features,
            out_features,
            base_requires_grad,
            lora,
            store,
        )?)
    }
}

fn new_sparse_mlp(
    names: &Qwen35MoeTensorNames,
    cfg: &Qwen35Config,
    tp: TpContext,
    base_requires_grad: bool,
    materialize_frozen_base: bool,
    lora: Option<LoraConfig>,
    lora_target_set: LoraTargetSet,
    lora_skip_experts: bool,
    store: &mut TensorStore,
) -> Result<Qwen35SparseMlp> {
    let router_gate_name = leak_name(names.router_gate.clone());
    let shared_gate_proj_name = leak_name(names.shared_expert_gate_proj.clone());
    let shared_up_proj_name = leak_name(names.shared_expert_up_proj.clone());
    let shared_down_proj_name = leak_name(names.shared_expert_down_proj.clone());
    let shared_expert_gate_name = leak_name(names.shared_expert_gate.clone());

    // When --lora-skip-experts is set, routed expert projections are frozen
    // (no LoRA adapters). Only attention + shared expert carry LoRA.
    let expert_lora = if lora_skip_experts { None } else { lora };

    // TP: column-parallel gate/up shard the intermediate (out) dim; row-parallel
    // down shards its input dim. Router stays replicated (full num_experts). The
    // forward all-reduces the summed expert+shared output over the TP group.
    let local_moe_intermediate = tp.local_moe_intermediate_size(cfg)?;
    let local_shared_intermediate = tp.local_shared_expert_intermediate_size(cfg)?;

    let experts = (0..cfg.num_experts)
        .map(|expert_idx| {
            let gate_proj_name = leak_name(names.expert_gate_proj(expert_idx));
            let up_proj_name = leak_name(names.expert_up_proj(expert_idx));
            let down_proj_name = leak_name(names.expert_down_proj(expert_idx));
            Ok(Qwen35SparseExpert {
                gate_proj: linear_with_base_init(
                    gate_proj_name,
                    cfg.hidden_size,
                    local_moe_intermediate,
                    base_requires_grad,
                    lora_for_name(expert_lora, lora_target_set, gate_proj_name),
                    materialize_frozen_base,
                    store,
                )?,
                up_proj: linear_with_base_init(
                    up_proj_name,
                    cfg.hidden_size,
                    local_moe_intermediate,
                    base_requires_grad,
                    lora_for_name(expert_lora, lora_target_set, up_proj_name),
                    materialize_frozen_base,
                    store,
                )?,
                down_proj: linear_with_base_init(
                    down_proj_name,
                    local_moe_intermediate,
                    cfg.hidden_size,
                    base_requires_grad,
                    lora_for_name(expert_lora, lora_target_set, down_proj_name),
                    materialize_frozen_base,
                    store,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Qwen35SparseMlp {
        router_gate: linear_with_base_init(
            router_gate_name,
            cfg.hidden_size,
            cfg.num_experts,
            base_requires_grad,
            lora_for_name(lora, lora_target_set, router_gate_name),
            materialize_frozen_base,
            store,
        )?,
        shared_gate_proj: linear_with_base_init(
            shared_gate_proj_name,
            cfg.hidden_size,
            local_shared_intermediate,
            base_requires_grad,
            lora_for_name(lora, lora_target_set, shared_gate_proj_name),
            materialize_frozen_base,
            store,
        )?,
        shared_up_proj: linear_with_base_init(
            shared_up_proj_name,
            cfg.hidden_size,
            local_shared_intermediate,
            base_requires_grad,
            lora_for_name(lora, lora_target_set, shared_up_proj_name),
            materialize_frozen_base,
            store,
        )?,
        shared_down_proj: linear_with_base_init(
            shared_down_proj_name,
            local_shared_intermediate,
            cfg.hidden_size,
            base_requires_grad,
            lora_for_name(lora, lora_target_set, shared_down_proj_name),
            materialize_frozen_base,
            store,
        )?,
        shared_expert_gate: linear_with_base_init(
            shared_expert_gate_name,
            cfg.hidden_size,
            1,
            base_requires_grad,
            lora_for_name(lora, lora_target_set, shared_expert_gate_name),
            materialize_frozen_base,
            store,
        )?,
        experts,
        top_k: cfg.num_experts_per_tok,
    })
}

fn share_base_attention(
    attention: &mut Qwen35Attention,
    base_attention: &Qwen35Attention,
) -> Result<()> {
    match (attention, base_attention) {
        (Qwen35Attention::Full(attn), Qwen35Attention::Full(base_attn)) => {
            attn.q_proj.set_base_weight(base_attn.q_proj.base_weight());
            attn.k_proj.set_base_weight(base_attn.k_proj.base_weight());
            attn.v_proj.set_base_weight(base_attn.v_proj.base_weight());
            attn.o_proj.set_base_weight(base_attn.o_proj.base_weight());
            attn.q_norm = base_attn.q_norm;
            attn.k_norm = base_attn.k_norm;
            Ok(())
        }
        (Qwen35Attention::Linear(attn), Qwen35Attention::Linear(base_attn)) => {
            attn.in_proj_qkv
                .set_base_weight(base_attn.in_proj_qkv.base_weight());
            attn.in_proj_z
                .set_base_weight(base_attn.in_proj_z.base_weight());
            attn.in_proj_b
                .set_base_weight(base_attn.in_proj_b.base_weight());
            attn.in_proj_a
                .set_base_weight(base_attn.in_proj_a.base_weight());
            attn.conv1d_weight = base_attn.conv1d_weight;
            attn.dt_bias = base_attn.dt_bias;
            attn.a_log = base_attn.a_log;
            attn.norm = base_attn.norm;
            attn.out_proj
                .set_base_weight(base_attn.out_proj.base_weight());
            Ok(())
        }
        _ => Err(Qwen35Error::InvalidConfig(
            "cannot share Qwen3.5 base weights across mismatched attention layer types",
        )),
    }
}

fn share_base_mlp(mlp: &mut Qwen35Mlp, base_mlp: &Qwen35Mlp) -> Result<()> {
    match (mlp, base_mlp) {
        (Qwen35Mlp::Dense(mlp), Qwen35Mlp::Dense(base_mlp)) => {
            mlp.gate_proj
                .set_base_weight(base_mlp.gate_proj.base_weight());
            mlp.up_proj.set_base_weight(base_mlp.up_proj.base_weight());
            mlp.down_proj
                .set_base_weight(base_mlp.down_proj.base_weight());
            Ok(())
        }
        (Qwen35Mlp::Sparse(mlp), Qwen35Mlp::Sparse(base_mlp)) => {
            mlp.router_gate
                .set_base_weight(base_mlp.router_gate.base_weight());
            mlp.shared_gate_proj
                .set_base_weight(base_mlp.shared_gate_proj.base_weight());
            mlp.shared_up_proj
                .set_base_weight(base_mlp.shared_up_proj.base_weight());
            mlp.shared_down_proj
                .set_base_weight(base_mlp.shared_down_proj.base_weight());
            mlp.shared_expert_gate
                .set_base_weight(base_mlp.shared_expert_gate.base_weight());
            if mlp.experts.len() != base_mlp.experts.len() {
                return Err(Qwen35Error::InvalidConfig(
                    "cannot share Qwen3.6 MoE base weights across mismatched expert counts",
                ));
            }
            for (expert, base_expert) in mlp.experts.iter_mut().zip(&base_mlp.experts) {
                expert
                    .gate_proj
                    .set_base_weight(base_expert.gate_proj.base_weight());
                expert
                    .up_proj
                    .set_base_weight(base_expert.up_proj.base_weight());
                expert
                    .down_proj
                    .set_base_weight(base_expert.down_proj.base_weight());
            }
            Ok(())
        }
        _ => Err(Qwen35Error::InvalidConfig(
            "cannot share Qwen3.5 base weights across mismatched MLP layer types",
        )),
    }
}

fn lora_for_layer(
    lora: Option<LoraConfig>,
    lora_layer_start: Option<usize>,
    layer_idx: usize,
) -> Option<LoraConfig> {
    if lora_layer_start.is_some_and(|start| layer_idx < start) {
        None
    } else {
        lora
    }
}

fn lora_for_name(
    lora: Option<LoraConfig>,
    target_set: LoraTargetSet,
    base_name: &str,
) -> Option<LoraConfig> {
    lora.filter(|_| target_set.includes(base_name))
}

fn collect_linear_ids(linear: &LinearWithLora, ids: &mut Vec<TensorId>) {
    ids.extend(linear.parameter_name_map().values().copied());
    ids.extend(linear.adapter_ordered().into_iter().map(|(_, id)| id));
}

fn collect_mlp_ids(mlp: &Qwen35Mlp, skip_experts: bool, ids: &mut Vec<TensorId>) {
    match mlp {
        Qwen35Mlp::Dense(dense) => {
            collect_linear_ids(&dense.gate_proj, ids);
            collect_linear_ids(&dense.up_proj, ids);
            collect_linear_ids(&dense.down_proj, ids);
        }
        Qwen35Mlp::Sparse(sparse) => {
            collect_linear_ids(&sparse.router_gate, ids);
            collect_linear_ids(&sparse.shared_gate_proj, ids);
            collect_linear_ids(&sparse.shared_up_proj, ids);
            collect_linear_ids(&sparse.shared_down_proj, ids);
            collect_linear_ids(&sparse.shared_expert_gate, ids);
            if !skip_experts {
                for expert in &sparse.experts {
                    collect_linear_ids(&expert.gate_proj, ids);
                    collect_linear_ids(&expert.up_proj, ids);
                    collect_linear_ids(&expert.down_proj, ids);
                }
            }
        }
    }
}

fn register_mlp(
    param_names: &mut HashMap<&'static str, TensorId>,
    adapter_names: &mut HashMap<&'static str, TensorId>,
    register_named: &mut impl FnMut(&mut HashMap<&'static str, TensorId>, &'static str, TensorId),
    mlp: &Qwen35Mlp,
) {
    match mlp {
        Qwen35Mlp::Dense(dense) => {
            register_linear(param_names, adapter_names, register_named, &dense.gate_proj);
            register_linear(param_names, adapter_names, register_named, &dense.up_proj);
            register_linear(param_names, adapter_names, register_named, &dense.down_proj);
        }
        Qwen35Mlp::Sparse(sparse) => {
            register_linear(
                param_names,
                adapter_names,
                register_named,
                &sparse.router_gate,
            );
            register_linear(
                param_names,
                adapter_names,
                register_named,
                &sparse.shared_gate_proj,
            );
            register_linear(
                param_names,
                adapter_names,
                register_named,
                &sparse.shared_up_proj,
            );
            register_linear(
                param_names,
                adapter_names,
                register_named,
                &sparse.shared_down_proj,
            );
            register_linear(
                param_names,
                adapter_names,
                register_named,
                &sparse.shared_expert_gate,
            );
            for expert in &sparse.experts {
                register_linear(
                    param_names,
                    adapter_names,
                    register_named,
                    &expert.gate_proj,
                );
                register_linear(param_names, adapter_names, register_named, &expert.up_proj);
                register_linear(
                    param_names,
                    adapter_names,
                    register_named,
                    &expert.down_proj,
                );
            }
        }
    }
}

fn insert_materialized_linear(
    map: &mut HashMap<&'static str, TensorId>,
    linear: &LinearWithLora,
    store: &mut TensorStore,
) -> Result<()> {
    let tensor = linear.merged_tensor(store)?;
    let merged = store.alloc(tensor);
    for (name, _) in linear.parameter_name_map() {
        map.insert(name, merged);
    }
    Ok(())
}

fn insert_materialized_mlp_params(
    map: &mut HashMap<&'static str, TensorId>,
    mlp: &Qwen35Mlp,
    store: &mut TensorStore,
) -> Result<()> {
    match mlp {
        Qwen35Mlp::Dense(dense) => {
            insert_materialized_linear(map, &dense.gate_proj, store)?;
            insert_materialized_linear(map, &dense.up_proj, store)?;
            insert_materialized_linear(map, &dense.down_proj, store)?;
        }
        Qwen35Mlp::Sparse(sparse) => {
            insert_materialized_linear(map, &sparse.router_gate, store)?;
            insert_materialized_linear(map, &sparse.shared_gate_proj, store)?;
            insert_materialized_linear(map, &sparse.shared_up_proj, store)?;
            insert_materialized_linear(map, &sparse.shared_down_proj, store)?;
            insert_materialized_linear(map, &sparse.shared_expert_gate, store)?;
            for expert in &sparse.experts {
                insert_materialized_linear(map, &expert.gate_proj, store)?;
                insert_materialized_linear(map, &expert.up_proj, store)?;
                insert_materialized_linear(map, &expert.down_proj, store)?;
            }
        }
    }
    Ok(())
}

fn build_qwen35_grouped_routes(routes: &MoeTopK) -> Result<Vec<MoeGroupedRoute>> {
    if routes.indices.len() != routes.tokens * routes.top_k {
        return Err(AutogradError::InvalidIndicesLen {
            expected: routes.tokens * routes.top_k,
            got: routes.indices.len(),
        }
        .into());
    }
    let mut counts = vec![0usize; routes.experts];
    let mut grouped = Vec::with_capacity(routes.indices.len());
    for token in 0..routes.tokens {
        for slot in 0..routes.top_k {
            let expert = routes.indices[token * routes.top_k + slot];
            if expert >= routes.experts {
                return Err(AutogradError::IndexOutOfBounds {
                    index: expert,
                    upper: routes.experts,
                }
                .into());
            }
            let row = counts[expert];
            counts[expert] += 1;
            grouped.push(MoeGroupedRoute {
                expert,
                row,
                token,
                slot,
            });
        }
    }
    grouped.sort_by_key(|route| (route.expert, route.row));
    Ok(grouped)
}

fn validate_qwen35_moe_route_signature(
    route: &Qwen35MoeRouteSignature,
    layer: usize,
    tokens: usize,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    if route.layer != layer {
        return Err(Qwen35Error::InvalidConfig(
            "frozen MoE route layer index mismatch",
        ));
    }
    if route.tokens != tokens || route.experts != experts || route.top_k != top_k {
        return Err(Qwen35Error::InvalidConfig(
            "frozen MoE route shape mismatch",
        ));
    }
    if route.indices.len() != tokens * top_k {
        return Err(AutogradError::InvalidIndicesLen {
            expected: tokens * top_k,
            got: route.indices.len(),
        }
        .into());
    }
    for &expert in &route.indices {
        if expert >= experts {
            return Err(AutogradError::IndexOutOfBounds {
                index: expert,
                upper: experts,
            }
            .into());
        }
    }
    Ok(())
}

fn qwen35_grouped_linear_experts(
    experts: &[Qwen35SparseExpert],
    select: impl Fn(&Qwen35SparseExpert) -> &LinearWithLora,
) -> Vec<MoeGroupedLinearExpert> {
    experts
        .iter()
        .map(|expert| {
            let parts = select(expert).parts();
            MoeGroupedLinearExpert {
                weight: parts.weight,
                lora_a: parts.lora_a,
                lora_b: parts.lora_b,
                lora_scale: parts.lora_scale,
            }
        })
        .collect()
}

fn broadcast_to_shape(
    x: TensorId,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let zeros = store.alloc(Tensor::new(
        vec![0.0; shape.iter().product()],
        shape.to_vec(),
        false,
    )?);
    Ok(add_broadcast(zeros, x, store, tape)?)
}

pub(crate) fn qwen35_to_autograd(err: Qwen35Error) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(err.to_string().into_boxed_str()))
}

fn copy_frozen_tensor_map(
    source: &HashMap<&'static str, TensorId>,
    target: &HashMap<&'static str, TensorId>,
    store: &mut TensorStore,
) {
    let mut names = source.keys().copied().collect::<Vec<_>>();
    names.sort_unstable();
    for name in names {
        copy_frozen_tensor(source[&name], target[&name], store);
    }
}

fn copy_frozen_tensor(source_id: TensorId, target_id: TensorId, store: &mut TensorStore) {
    let mut replacement = store
        .get(source_id)
        .cloned()
        .expect("source parameter should remain readable");
    replacement.requires_grad = false;
    replacement.grad = None;
    store.tensors[target_id] = Some(replacement);
}

fn append_cached_kv(
    cached: Option<TensorId>,
    next: TensorId,
    max_seq_len: usize,
    seq_cursor: usize,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let next_shape = store
        .get(next)
        .ok_or(AutogradError::InvalidTensorId(next))?
        .shape
        .clone();
    if next_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "rank-4 KV tensor [batch, heads, seq, dim]",
            got: next_shape.len(),
        }
        .into());
    }
    let next_seq_len = next_shape[2];
    if seq_cursor + next_seq_len > max_seq_len {
        return Err(Qwen35Error::InvalidConfig(
            "rollout KV cache append exceeds preallocated max_seq_len",
        ));
    }

    let cached = if let Some(cached) = cached {
        cached
    } else {
        let cache_shape = vec![next_shape[0], next_shape[1], max_seq_len, next_shape[3]];
        let handle = store.backend().zeros(&cache_shape)?;
        store.alloc_device_tensor(cache_shape, handle)?
    };

    let cached_shape = store
        .get(cached)
        .ok_or(AutogradError::InvalidTensorId(cached))?
        .shape
        .clone();

    store.ensure_device(cached)?;
    store.ensure_device(next)?;
    let cached_handle = store
        .get(cached)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "append_cached_kv: cached tensor missing device handle",
        ))?;
    let next_handle = store
        .get(next)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "append_cached_kv: next tensor missing device handle",
        ))?;
    let out_handle = store.backend().kv_cache_write_axis2(
        &cached_handle,
        &cached_shape,
        &next_handle,
        &next_shape,
        seq_cursor,
    )?;
    store.replace_device_handle(cached, out_handle)?;
    Ok(cached)
}

fn causal_sdpa_decode_gqa_cached(
    q: TensorId,
    k_cache: TensorId,
    v_cache: TensorId,
    kv_len: usize,
    q_start: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if tape.enabled {
        return Err(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa_cached is rollout-only and requires tape disabled",
        )
        .into());
    }
    let q_shape = store
        .get(q)
        .ok_or(AutogradError::InvalidTensorId(q))?
        .shape
        .clone();
    let k_shape = store
        .get(k_cache)
        .ok_or(AutogradError::InvalidTensorId(k_cache))?
        .shape
        .clone();
    let v_shape = store
        .get(v_cache)
        .ok_or(AutogradError::InvalidTensorId(v_cache))?
        .shape
        .clone();
    store.ensure_device(q)?;
    store.ensure_device(k_cache)?;
    store.ensure_device(v_cache)?;
    let q_handle = store
        .get(q)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa_cached: q missing device handle",
        ))?;
    let k_handle = store
        .get(k_cache)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa_cached: k missing device handle",
        ))?;
    let v_handle = store
        .get(v_cache)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa_cached: v missing device handle",
        ))?;
    let (out_handle, out_shape) = store.backend().causal_sdpa_decode_gqa_cache(
        &q_handle, &q_shape, &k_handle, &k_shape, &v_handle, &v_shape, kv_len, q_start,
    )?;
    Ok(store.alloc_device_tensor(out_shape, out_handle)?)
}

fn qwen_decode_prepare_q(
    q_full: TensorId,
    q_norm: TensorId,
    cos: TensorId,
    sin: TensorId,
    cfg: &Qwen35Config,
    batch: usize,
    store: &mut TensorStore,
) -> Result<(TensorId, Option<TensorId>)> {
    store.ensure_device(q_full)?;
    store.ensure_device(q_norm)?;
    store.ensure_device(cos)?;
    store.ensure_device(sin)?;

    let q_full_shape = store
        .get(q_full)
        .ok_or(AutogradError::InvalidTensorId(q_full))?
        .shape
        .clone();
    if q_full_shape.len() != 3 || q_full_shape[0] != batch || q_full_shape[1] != 1 {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![batch, 1, cfg.full_attn_q_proj_dim()],
            got: q_full_shape,
        }
        .into());
    }
    let q_norm_shape = store
        .get(q_norm)
        .ok_or(AutogradError::InvalidTensorId(q_norm))?
        .shape
        .clone();
    let cos_shape = store
        .get(cos)
        .ok_or(AutogradError::InvalidTensorId(cos))?
        .shape
        .clone();
    let sin_shape = store
        .get(sin)
        .ok_or(AutogradError::InvalidTensorId(sin))?
        .shape
        .clone();
    let q_full_handle = store
        .get(q_full)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_q: q_full missing device handle",
        ))?;
    let q_norm_handle = store
        .get(q_norm)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_q: q_norm missing device handle",
        ))?;
    let cos_handle = store
        .get(cos)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_q: cos missing device handle",
        ))?;
    let sin_handle = store
        .get(sin)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_q: sin missing device handle",
        ))?;

    let (q_handle, gate_handle, out_shape) = store.backend().qwen_decode_prepare_q(
        &q_full_handle,
        &q_full_shape,
        &q_norm_handle,
        &q_norm_shape,
        &cos_handle,
        &cos_shape,
        &sin_handle,
        &sin_shape,
        cfg.num_attention_heads,
        cfg.head_dim,
        cfg.full_attn_gated,
        cfg.rms_norm_eps,
    )?;
    let q = store.alloc_device_tensor(out_shape.clone(), q_handle)?;
    let gate = gate_handle
        .map(|handle| store.alloc_device_tensor(out_shape, handle))
        .transpose()?;
    Ok((q, gate))
}

fn qwen_decode_prepare_kv(
    k_full: TensorId,
    v_full: TensorId,
    k_norm: TensorId,
    cos: TensorId,
    sin: TensorId,
    cfg: &Qwen35Config,
    batch: usize,
    store: &mut TensorStore,
) -> Result<(TensorId, TensorId)> {
    store.ensure_device(k_full)?;
    store.ensure_device(v_full)?;
    store.ensure_device(k_norm)?;
    store.ensure_device(cos)?;
    store.ensure_device(sin)?;

    let k_full_shape = store
        .get(k_full)
        .ok_or(AutogradError::InvalidTensorId(k_full))?
        .shape
        .clone();
    if k_full_shape.len() != 3
        || k_full_shape[0] != batch
        || k_full_shape[1] != 1
        || k_full_shape[2] != cfg.num_key_value_heads * cfg.head_dim
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![batch, 1, cfg.num_key_value_heads * cfg.head_dim],
            got: k_full_shape,
        }
        .into());
    }
    let v_full_shape = store
        .get(v_full)
        .ok_or(AutogradError::InvalidTensorId(v_full))?
        .shape
        .clone();
    let k_norm_shape = store
        .get(k_norm)
        .ok_or(AutogradError::InvalidTensorId(k_norm))?
        .shape
        .clone();
    let cos_shape = store
        .get(cos)
        .ok_or(AutogradError::InvalidTensorId(cos))?
        .shape
        .clone();
    let sin_shape = store
        .get(sin)
        .ok_or(AutogradError::InvalidTensorId(sin))?
        .shape
        .clone();
    let k_full_handle = store
        .get(k_full)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_kv: k_full missing device handle",
        ))?;
    let v_full_handle = store
        .get(v_full)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_kv: v_full missing device handle",
        ))?;
    let k_norm_handle = store
        .get(k_norm)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_kv: k_norm missing device handle",
        ))?;
    let cos_handle = store
        .get(cos)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_kv: cos missing device handle",
        ))?;
    let sin_handle = store
        .get(sin)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_kv: sin missing device handle",
        ))?;

    let (k_handle, v_handle, out_shape) = store.backend().qwen_decode_prepare_kv(
        &k_full_handle,
        &k_full_shape,
        &v_full_handle,
        &v_full_shape,
        &k_norm_handle,
        &k_norm_shape,
        &cos_handle,
        &cos_shape,
        &sin_handle,
        &sin_shape,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.rms_norm_eps,
    )?;
    let k = store.alloc_device_tensor(out_shape.clone(), k_handle)?;
    let v = store.alloc_device_tensor(out_shape, v_handle)?;
    Ok((k, v))
}

fn embedding_device_f32_ids(
    table: TensorId,
    ids: TensorId,
    n_ids: usize,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let table_shape = store
        .get(table)
        .ok_or(AutogradError::InvalidTensorId(table))?
        .shape
        .clone();
    if table_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: table_shape.len(),
        }
        .into());
    }
    store.ensure_device(table)?;
    store.ensure_device(ids)?;
    let table_handle = store
        .get(table)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "embedding_device_f32_ids: table missing device handle",
        ))?;
    let ids_handle = store
        .get(ids)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "embedding_device_f32_ids: ids missing device handle",
        ))?;
    let out_handle =
        store
            .backend()
            .embedding_from_f32_ids(&table_handle, &table_shape, &ids_handle, n_ids)?;
    Ok(store.alloc_device_tensor(vec![1, n_ids, table_shape[1]], out_handle)?)
}

fn split_heads(
    x: TensorId,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x = reshape(x, &[batch, seq_len, heads, head_dim], store, tape)?;
    Ok(transpose(x, 1, 2, store, tape)?)
}

fn merge_heads(
    x: TensorId,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x = transpose(x, 1, 2, store, tape)?;
    Ok(reshape(
        x,
        &[batch, seq_len, heads * head_dim],
        store,
        tape,
    )?)
}

fn select_cache_rows(
    cache: TensorId,
    position_ids: &[usize],
    store: &mut TensorStore,
) -> Result<TensorId> {
    let cache_tensor = store
        .get(cache)
        .ok_or(AutogradError::InvalidTensorId(cache))?;
    if cache_tensor.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: cache_tensor.shape.len(),
        }
        .into());
    }

    let rows = cache_tensor.shape[0];
    let cols = cache_tensor.shape[1];
    let mut data = Vec::with_capacity(position_ids.len() * cols);
    for &position in position_ids {
        if position >= rows {
            return Err(Qwen35Error::PositionOutOfBounds {
                position,
                upper: rows,
            });
        }
        let base = position * cols;
        data.extend_from_slice(&cache_tensor.data[base..base + cols]);
    }
    let output_shape = vec![position_ids.len(), cols];
    Ok(store.alloc(Tensor::new(data, output_shape, false)?))
}

fn build_rope_cache(cfg: &Qwen35Config, store: &mut TensorStore) -> Result<(TensorId, TensorId)> {
    let max_positions = cfg.rope_cache_len_hint.ok_or(Qwen35Error::InvalidConfig(
        "train-side qwen3.5 requires rope_cache_len_hint",
    ))?;
    let half_dim = cfg.rotary_dim / 2;
    let inv_freq = (0..half_dim)
        .map(|index| {
            1.0 / cfg
                .rope_theta
                .powf((2.0 * index as f32) / cfg.rotary_dim as f32)
        })
        .collect::<Vec<_>>();
    let mut cos = vec![0.0; max_positions * half_dim];
    let mut sin = vec![0.0; max_positions * half_dim];

    for position in 0..max_positions {
        let base = position * half_dim;
        for (freq_index, &freq) in inv_freq.iter().enumerate() {
            let angle = position as f32 * freq;
            cos[base + freq_index] = angle.cos();
            sin[base + freq_index] = angle.sin();
        }
    }

    let cos_cache = store.alloc(Tensor::new(cos, vec![max_positions, half_dim], false)?);
    let sin_cache = store.alloc(Tensor::new(sin, vec![max_positions, half_dim], false)?);
    Ok((cos_cache, sin_cache))
}

fn normal_parameter(
    name: &'static str,
    shape: &[usize],
    std: f32,
    requires_grad: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let mut state = seed_from_name(name);
    let size = shape.iter().product();
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let u1 = next_uniform(&mut state).max(f32::MIN_POSITIVE);
        let u2 = next_uniform(&mut state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        data.push(std * radius * theta.cos());
        if data.len() < size {
            data.push(std * radius * theta.sin());
        }
    }
    Ok(store.alloc(Tensor::new(data, shape.to_vec(), requires_grad)?))
}

fn normal_or_unmaterialized_parameter(
    name: &'static str,
    shape: &[usize],
    std: f32,
    requires_grad: bool,
    materialize_frozen_base: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    if requires_grad || materialize_frozen_base {
        normal_parameter(name, shape, std, requires_grad, store)
    } else {
        let _ = name;
        Ok(store.alloc(Tensor::unmaterialized(shape.to_vec(), false)?))
    }
}

fn ones_parameter(
    name: &'static str,
    shape: &[usize],
    requires_grad: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let _ = name;
    Ok(store.alloc(Tensor::new(
        vec![1.0; shape.iter().product()],
        shape.to_vec(),
        requires_grad,
    )?))
}

fn ones_or_unmaterialized_parameter(
    name: &'static str,
    shape: &[usize],
    requires_grad: bool,
    materialize_frozen_base: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    if requires_grad || materialize_frozen_base {
        ones_parameter(name, shape, requires_grad, store)
    } else {
        let _ = name;
        Ok(store.alloc(Tensor::unmaterialized(shape.to_vec(), false)?))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::error::Error;
    #[cfg(feature = "cuda")]
    use std::{env, sync::Arc};

    #[cfg(feature = "cuda")]
    use autograd::backend_cuda::CudaBackend;
    use autograd::{
        BackwardOp, Tape, TensorId, TensorStore,
        ops::{mul, sum},
    };

    use super::{LayerType, Qwen35Config, Qwen35KvCache, Qwen35Model};
    use crate::lora::{LoraConfig, LoraTargetSet};

    type TestResult<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

    fn tiny_qwen35_config(max_seq_len: usize) -> Qwen35Config {
        Qwen35Config {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            vocab_size: 16,
            rms_norm_eps: 1.0e-6,
            stop_token_ids: vec![15],
            bos_token_id: Some(1),
            eos_token_id: 15,
            tie_word_embeddings: false,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_num_value_heads: 2,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000.0,
            rope_scaling: None,
            partial_rotary_factor: 1.0,
            rotary_dim: 8,
            rope_cache_len_hint: Some(max_seq_len),
            layer_types: vec![LayerType::FullAttention; 2],
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
            full_attn_gated: true,
        }
    }

    fn tiny_qwen36_moe_config(max_seq_len: usize) -> Qwen35Config {
        Qwen35Config {
            hidden_size: 8,
            intermediate_size: 0,
            num_hidden_layers: 1,
            vocab_size: 12,
            rms_norm_eps: 1.0e-6,
            stop_token_ids: vec![11],
            bos_token_id: Some(1),
            eos_token_id: 11,
            tie_word_embeddings: false,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_num_value_heads: 2,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000.0,
            rope_scaling: None,
            partial_rotary_factor: 1.0,
            rotary_dim: 4,
            rope_cache_len_hint: Some(max_seq_len),
            layer_types: vec![LayerType::FullAttention],
            num_experts: 3,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            moe_intermediate_size: 6,
            shared_expert_intermediate_size: 6,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
            full_attn_gated: true,
        }
    }

    fn logits_host(store: &mut TensorStore, logits: TensorId) -> TestResult<Vec<f32>> {
        Ok(store.to_host(logits)?)
    }

    fn squared_logits_loss(
        model: &Qwen35Model,
        store: &mut TensorStore,
        tape: &mut Tape,
        tokens: &[u32],
        positions: &[u32],
    ) -> TestResult<TensorId> {
        let logits = model.forward(store, tape, tokens, positions)?;
        let sq = mul(logits, logits, store, tape)?;
        Ok(sum(sq, store, tape)?)
    }

    fn squared_logits_loss_value(
        model: &Qwen35Model,
        store: &mut TensorStore,
        tokens: &[u32],
        positions: &[u32],
    ) -> TestResult<f32> {
        let mut tape = Tape::new();
        tape.set_enabled(false);
        let logits = model.forward(store, &mut tape, tokens, positions)?;
        let host = store.to_host(logits)?;
        Ok(host.iter().map(|value| value * value).sum())
    }

    fn set_tensor_element(
        store: &mut TensorStore,
        tensor_id: TensorId,
        index: usize,
        value: f32,
    ) -> TestResult {
        let tensor = store
            .get_mut(tensor_id)
            .ok_or_else(|| format!("missing tensor {tensor_id}"))?;
        let slot = tensor
            .data
            .get_mut(index)
            .ok_or_else(|| format!("tensor {tensor_id} missing index {index}"))?;
        *slot = value;
        Ok(())
    }

    fn fill_tensor(store: &mut TensorStore, tensor_id: TensorId, value: f32) -> TestResult {
        let tensor = store
            .get_mut(tensor_id)
            .ok_or_else(|| format!("missing tensor {tensor_id}"))?;
        for slot in &mut tensor.data {
            *slot = value;
        }
        tensor.device_handle = None;
        Ok(())
    }

    fn set_moe_router_tie(
        store: &mut TensorStore,
        model: &Qwen35Model,
        layer_idx: usize,
    ) -> TestResult {
        let name = format!("model.language_model.layers.{layer_idx}.mlp.gate.weight");
        let params = model.param_name_map();
        let router = *params
            .get(name.as_str())
            .ok_or_else(|| format!("missing router tensor {name}"))?;
        fill_tensor(store, router, 0.0)
    }

    fn first_nonzero_grad_index(store: &mut TensorStore, grad_id: TensorId) -> TestResult<usize> {
        let grad = store.to_host(grad_id)?;
        grad.iter()
            .enumerate()
            .filter(|(_, value)| value.abs() > 1.0e-7)
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .map(|(idx, _)| idx)
            .ok_or_else(|| "gradient has no stable non-zero element".into())
    }

    fn greedy_next(host: &[f32], seq_len: usize, vocab: usize) -> u32 {
        let row = &host[(seq_len - 1) * vocab..seq_len * vocab];
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .expect("non-empty vocab")
    }

    #[test]
    fn qwen35_lora_layer_start_limits_adapters_and_tape_prefix() -> TestResult {
        let mut store = TensorStore::default();
        let cfg = tiny_qwen35_config(8);
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
        };
        let model = Qwen35Model::new_with_lora_targets_layer_start(
            &cfg,
            lora,
            LoraTargetSet::AllLinear,
            Some(1),
            &mut store,
        )?;
        assert_eq!(model.lora_layer_start(), Some(1));

        let adapters = model.adapter_name_map();
        assert!(!adapters.is_empty(), "suffix LoRA must allocate adapters");
        for (name, &id) in &adapters {
            assert!(
                !name.contains(".layers.0."),
                "prefix layer adapter should not exist: {name}"
            );
            assert!(
                name.contains(".layers.1."),
                "suffix adapter must live in layer 1 for the tiny 2-layer config: {name}"
            );
            let tensor = store
                .get(id)
                .ok_or_else(|| format!("missing adapter {name}"))?;
            assert!(tensor.requires_grad, "adapter {name} must be trainable");
        }

        let prefix_params = model
            .param_name_map()
            .into_iter()
            .filter(|(name, _)| name.contains(".layers.0."))
            .map(|(name, id)| (id, name))
            .collect::<HashMap<_, _>>();
        let prefix_param_ids = prefix_params.keys().copied().collect::<HashSet<_>>();
        let suffix_adapter_ids = adapters.values().copied().collect::<HashSet<_>>();

        let tokens = [1_u32, 3];
        let positions = [0_u32, 1];
        let mut tape = Tape::new();
        let loss = squared_logits_loss(&model, &mut store, &mut tape, &tokens, &positions)?;

        let mut prefix_refs = Vec::new();
        let mut saw_suffix_adapter_site = false;
        for entry in &tape.entries {
            if let Some(site) = entry.profile_site() {
                assert!(
                    !site.contains(".layers.0."),
                    "tape must not record prefix layer matmul site {site}"
                );
                saw_suffix_adapter_site |= site.contains(".layers.1.") && site.ends_with(".lora_b");
            }
            for input_id in &entry.input_ids {
                if prefix_param_ids.contains(input_id) {
                    let name = prefix_params.get(input_id).copied().unwrap_or("<unknown>");
                    prefix_refs.push((entry.op.name(), name));
                }
            }
        }
        assert!(
            prefix_refs.is_empty(),
            "tape must not reference prefix layer params: {prefix_refs:?}"
        );
        assert!(
            saw_suffix_adapter_site,
            "tape must include trainable suffix LoRA adapter work"
        );

        let grads = tape.backward(loss, &mut store)?;
        for (name, &id) in &adapters {
            assert!(grads.contains_key(&id), "missing adapter grad for {name}");
        }
        assert!(
            grads.keys().all(|id| !prefix_param_ids.contains(id)),
            "prefix layer params must not receive gradients"
        );
        assert!(
            grads.keys().any(|id| suffix_adapter_ids.contains(id)),
            "at least one suffix adapter must receive a gradient"
        );
        Ok(())
    }

    #[test]
    fn qwen35_rollout_kv_cache_matches_full_forward_tokens() -> TestResult {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        tape.set_enabled(false);

        let cfg = tiny_qwen35_config(16);
        let model = Qwen35Model::new_for_eval(&cfg, &mut store)?;
        let vocab = cfg.vocab_size;
        let mut rollout = vec![1_u32, 3, 8];
        let mut cache = Qwen35KvCache::new(&model, rollout.len() + 5);

        for step in 0..5 {
            let full_positions = (0..rollout.len() as u32).collect::<Vec<_>>();
            let full_logits = model.forward(&mut store, &mut tape, &rollout, &full_positions)?;
            let full_host = logits_host(&mut store, full_logits)?;
            let full_next = greedy_next(&full_host, rollout.len(), vocab);

            let (cached_input, cached_positions, cached_seq_len) = if step == 0 {
                (rollout.clone(), full_positions, 1)
            } else {
                let last = *rollout.last().expect("rollout stays non-empty");
                (vec![last], vec![(rollout.len() - 1) as u32], 1)
            };
            let cached_logits = model.forward_rollout_cached(
                &mut store,
                &mut tape,
                &cached_input,
                &cached_positions,
                &mut cache,
            )?;
            let cached_host = logits_host(&mut store, cached_logits)?;
            let cached_next = greedy_next(&cached_host, cached_seq_len, vocab);

            let full_row = &full_host[(rollout.len() - 1) * vocab..rollout.len() * vocab];
            let cached_row = &cached_host[(cached_seq_len - 1) * vocab..cached_seq_len * vocab];
            let max_abs = full_row
                .iter()
                .zip(cached_row.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_abs <= 1.0e-5,
                "cached rollout logits must match full-forward last row at step {step}; max_abs={max_abs}"
            );
            assert_eq!(
                cached_next, full_next,
                "cached rollout token diverged at step {step}"
            );

            rollout.push(full_next);
        }

        assert_eq!(cache.seq_len, rollout.len() - 1);
        Ok(())
    }

    #[test]
    fn qwen35_logits_window_matches_full_forward_slice() -> TestResult {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        tape.set_enabled(false);

        let cfg = tiny_qwen35_config(16);
        let model = Qwen35Model::new_for_eval(&cfg, &mut store)?;
        let tokens = vec![1_u32, 3, 8, 4, 2];
        let positions = (0..tokens.len() as u32).collect::<Vec<_>>();
        let full_logits = model.forward(&mut store, &mut tape, &tokens, &positions)?;
        let full_host = logits_host(&mut store, full_logits)?;

        let window = super::SequenceWindow { start: 1, end: 4 };
        let window_logits =
            model.forward_logits_window(&mut store, &mut tape, &tokens, &positions, window)?;
        let window_host = logits_host(&mut store, window_logits)?;

        let expected_len = window.len() * cfg.vocab_size;
        assert_eq!(window_host.len(), expected_len);
        for local_pos in 0..window.len() {
            let full_start = (window.start + local_pos) * cfg.vocab_size;
            let window_start = local_pos * cfg.vocab_size;
            let full_row = &full_host[full_start..full_start + cfg.vocab_size];
            let window_row = &window_host[window_start..window_start + cfg.vocab_size];
            let max_abs = full_row
                .iter()
                .zip(window_row.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_abs <= 1.0e-5,
                "window row {local_pos} must match full logits slice; max_abs={max_abs}"
            );
        }
        Ok(())
    }

    #[test]
    fn qwen36_moe_lora_param_names_match_checkpoint_contract() -> TestResult {
        let mut store = TensorStore::default();
        let cfg = tiny_qwen36_moe_config(8);
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
        };
        let model =
            Qwen35Model::new_with_lora_targets(&cfg, lora, LoraTargetSet::AllLinear, &mut store)?;
        let params = model.param_name_map();
        let adapters = model.adapter_name_map();
        for name in [
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
            "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
            "lm_head.weight",
        ] {
            assert!(params.contains_key(name), "missing param {name}");
        }
        assert!(
            !params.contains_key("model.language_model.layers.0.mlp.gate_proj.weight"),
            "MoE layers must not expose dense MLP gate_proj names"
        );
        assert!(
            adapters
                .contains_key("model.language_model.layers.0.mlp.experts.0.up_proj.weight.lora_b"),
            "routed expert LoRA-B must be trainable"
        );
        assert!(
            adapters.contains_key(
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight.lora_b"
            ),
            "shared expert LoRA-B must be trainable"
        );
        Ok(())
    }

    #[test]
    fn qwen36_checkpoint_load_constructor_keeps_frozen_base_unmaterialized() -> TestResult {
        let mut store = TensorStore::default();
        let cfg = tiny_qwen36_moe_config(8);
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
        };
        let model = Qwen35Model::new_with_lora_targets_for_checkpoint_load(
            &cfg,
            lora,
            LoraTargetSet::AllLinear,
            false,
            &mut store,
        )?;
        let params = model.param_name_map();
        let adapters = model.adapter_name_map();
        let base_name = "model.language_model.layers.0.mlp.experts.0.up_proj.weight";
        let adapter_name = "model.language_model.layers.0.mlp.experts.0.up_proj.weight.lora_b";
        let base = *params
            .get(base_name)
            .ok_or_else(|| format!("missing base tensor {base_name}"))?;
        let adapter = *adapters
            .get(adapter_name)
            .ok_or_else(|| format!("missing adapter tensor {adapter_name}"))?;

        let base_tensor = store
            .get(base)
            .ok_or_else(|| format!("missing tensor {base}"))?;
        assert_eq!(
            base_tensor.shape,
            vec![cfg.moe_intermediate_size, cfg.hidden_size]
        );
        assert_eq!(base_tensor.dirty, autograd::tensor::Dirty::Device);
        assert!(base_tensor.data.is_empty());
        assert!(base_tensor.device_handle.is_none());
        assert!(!base_tensor.requires_grad);

        let adapter_tensor = store
            .get(adapter)
            .ok_or_else(|| format!("missing tensor {adapter}"))?;
        assert!(adapter_tensor.requires_grad);
        assert_eq!(
            adapter_tensor.data.len(),
            cfg.moe_intermediate_size * lora.rank
        );
        assert_eq!(adapter_tensor.dirty, autograd::tensor::Dirty::Host);
        Ok(())
    }

    #[test]
    fn qwen36_moe_lora_gradient_matches_finite_difference() -> TestResult {
        let mut store = TensorStore::default();
        let cfg = tiny_qwen36_moe_config(8);
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
        };
        let model =
            Qwen35Model::new_with_lora_targets(&cfg, lora, LoraTargetSet::AllLinear, &mut store)?;
        set_moe_router_tie(&mut store, &model, 0)?;
        let target_name = "model.language_model.layers.0.mlp.experts.0.up_proj.weight.lora_b";
        let target = *model
            .adapter_name_map()
            .get(target_name)
            .ok_or_else(|| format!("missing adapter {target_name}"))?;
        let tokens = [1_u32, 3];
        let positions = [0_u32, 1];

        let mut tape = Tape::new();
        let loss = squared_logits_loss(&model, &mut store, &mut tape, &tokens, &positions)?;
        let grads = tape.backward(loss, &mut store)?;
        let grad_id = *grads
            .get(&target)
            .ok_or_else(|| format!("missing grad for {target_name}"))?;
        let probe_index = first_nonzero_grad_index(&mut store, grad_id)?;
        let analytic = store.to_host(grad_id)?[probe_index];

        let base = store
            .get(target)
            .and_then(|tensor| tensor.data.get(probe_index).copied())
            .ok_or_else(|| format!("missing target element {probe_index}"))?;
        let eps = 1.0e-3_f32;
        set_tensor_element(&mut store, target, probe_index, base + eps)?;
        let plus = squared_logits_loss_value(&model, &mut store, &tokens, &positions)?;
        set_tensor_element(&mut store, target, probe_index, base - eps)?;
        let minus = squared_logits_loss_value(&model, &mut store, &tokens, &positions)?;
        set_tensor_element(&mut store, target, probe_index, base)?;
        let numeric = (plus - minus) / (2.0 * eps);
        let denom = analytic.abs().max(numeric.abs()).max(1.0e-6);
        let rel_err = (analytic - numeric).abs() / denom;
        eprintln!(
            "qwen36_moe_lora_fd target={target_name}[{probe_index}] analytic={analytic:.9e} numeric={numeric:.9e} rel_err={rel_err:.3e}"
        );
        assert!(
            rel_err <= 1.0e-2,
            "Qwen35 integrated MoE LoRA finite diff failed: analytic={analytic:.9e} numeric={numeric:.9e} rel_err={rel_err:.3e}"
        );
        Ok(())
    }

    fn run_qwen35_checkpoint_lora_fd_gate(
        store: &mut TensorStore,
    ) -> TestResult<(&'static str, usize, f32, f32, f32)> {
        let cfg = tiny_qwen35_config(8);
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
        };
        let mut model =
            Qwen35Model::new_with_lora_targets(&cfg, lora, LoraTargetSet::AllLinear, store)?;
        model.set_gradient_checkpointing(true);
        assert!(model.gradient_checkpointing());
        // The auto-gate declines the tiny shape on a real GPU (modeled ≪ free);
        // this test exists to exercise the checkpoint path, so force-engage.
        // SAFETY: forced checkpointing only changes which (equivalent) path
        // concurrent tests take.
        unsafe {
            std::env::set_var("ARLE_FORCE_CHECKPOINT", "1");
        }

        let tokens = [1_u32, 3, 8];
        let positions = [0_u32, 1, 2];
        let mut tape = Tape::new();
        let loss = squared_logits_loss(&model, store, &mut tape, &tokens, &positions)?;
        let checkpoint_entries = tape
            .entries
            .iter()
            .filter(|entry| entry.op == BackwardOp::Checkpoint)
            .count();
        assert_eq!(
            checkpoint_entries, cfg.num_hidden_layers,
            "one checkpoint entry per layer is required"
        );
        let adapter_ids = model
            .adapter_name_map()
            .values()
            .copied()
            .collect::<HashSet<_>>();
        let frozen_param_ids = model
            .param_name_map()
            .values()
            .copied()
            .collect::<HashSet<_>>();
        let mut saw_checkpoint_adapter = false;
        for entry in tape
            .entries
            .iter()
            .filter(|entry| entry.op == BackwardOp::Checkpoint)
        {
            for &input_id in entry.input_ids.iter().skip(1) {
                let tensor = store
                    .get(input_id)
                    .ok_or_else(|| format!("checkpoint input tensor {input_id} missing"))?;
                assert!(
                    tensor.requires_grad,
                    "checkpoint input {input_id} must be trainable; frozen base params would force useless large gradient computation"
                );
                assert!(
                    !frozen_param_ids.contains(&input_id),
                    "checkpoint input {input_id} leaked a frozen base parameter"
                );
                saw_checkpoint_adapter |= adapter_ids.contains(&input_id);
            }
        }
        assert!(
            saw_checkpoint_adapter,
            "checkpoint inputs must include trainable LoRA adapters"
        );
        let grads = tape.backward(loss, store)?;

        let mut best: Option<(&'static str, TensorId, usize, f32)> = None;
        let mut adapters = model.adapter_name_map().into_iter().collect::<Vec<_>>();
        adapters.sort_by_key(|(name, _)| *name);
        for (name, tensor_id) in adapters {
            let Some(&grad_id) = grads.get(&tensor_id) else {
                continue;
            };
            let grad = store.to_host(grad_id)?;
            for (index, &value) in grad.iter().enumerate() {
                if best
                    .as_ref()
                    .is_none_or(|(_, _, _, current)| value.abs() > current.abs())
                {
                    best = Some((name, tensor_id, index, value));
                }
            }
        }

        let (name, tensor_id, index, analytic) =
            best.ok_or("checkpointed LoRA backward produced no adapter gradients")?;
        assert!(
            analytic.abs() > 1.0e-8,
            "finite-diff probe gradient is too small for a relative gate: {name}[{index}]={analytic}"
        );
        let base_value = store
            .to_host(tensor_id)?
            .get(index)
            .copied()
            .ok_or("probe tensor index missing")?;
        let eps = 1.0e-3_f32;
        set_tensor_element(store, tensor_id, index, base_value + eps)?;
        let loss_plus = squared_logits_loss_value(&model, store, &tokens, &positions)?;
        set_tensor_element(store, tensor_id, index, base_value - eps)?;
        let loss_minus = squared_logits_loss_value(&model, store, &tokens, &positions)?;
        set_tensor_element(store, tensor_id, index, base_value)?;

        let numeric = (loss_plus - loss_minus) / (2.0 * eps);
        let rel_err = (analytic - numeric).abs() / numeric.abs().max(analytic.abs()).max(1.0e-12);
        eprintln!(
            "qwen35_checkpoint_fd name={name} index={index} eps={eps:.1e} analytic={analytic:.9e} numeric={numeric:.9e} rel_err={rel_err:.3e}"
        );
        assert!(
            rel_err <= 1.0e-2,
            "checkpointed LoRA finite diff failed for {name}[{index}]: analytic={analytic:.9e} numeric={numeric:.9e} rel_err={rel_err:.3e}"
        );
        Ok((name, index, analytic, numeric, rel_err))
    }

    #[test]
    fn qwen35_gradient_checkpointing_lora_finite_diff_gate() -> TestResult {
        let mut store = TensorStore::default();
        run_qwen35_checkpoint_lora_fd_gate(&mut store)?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn qwen35_gradient_checkpointing_lora_cuda_finite_diff_gate() -> TestResult {
        let ordinal = env::var("ARLE_CUDA_TEST_DEVICE")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(0);
        let backend = Arc::new(CudaBackend::new(ordinal)?);
        let mut store = TensorStore::with_backend(backend);
        run_qwen35_checkpoint_lora_fd_gate(&mut store)?;
        Ok(())
    }
}
