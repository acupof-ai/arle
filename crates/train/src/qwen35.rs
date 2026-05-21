use std::{
    collections::{HashMap, HashSet},
    f32::consts::TAU,
    time::{Duration, Instant},
};

use autograd::{
    AutogradError, Device, Tape, Tensor, TensorId, TensorStore,
    ops::{
        LinearAttentionParams, add, causal_sdpa, causal_sdpa_decode_gqa, causal_sdpa_with_q_start,
        embedding, linear_attention_core, matmul_bt, mul, repeat_kv, reshape, rmsnorm, rope,
        sigmoid, silu, slice, transpose,
    },
};
use qwen35_spec::Qwen35AttentionTensorNames;
pub use qwen35_spec::{LayerType, Qwen35Config, Qwen35ConfigError};
use thiserror::Error;

use crate::{
    causal_lm::CausalLm,
    lora::{LinearWithLora, LoraConfig, LoraTargetSet},
};

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

#[derive(Debug, Clone)]
struct Qwen35Mlp {
    gate_proj: LinearWithLora,
    up_proj: LinearWithLora,
    down_proj: LinearWithLora,
}

#[derive(Debug, Clone)]
struct Qwen35Layer {
    input_layernorm: TensorId,
    self_attn: Qwen35Attention,
    post_attention_layernorm: TensorId,
    mlp: Qwen35Mlp,
}

#[derive(Debug, Clone, Default)]
struct Qwen35LayerKvCache {
    k: Option<TensorId>,
    v: Option<TensorId>,
}

#[derive(Debug, Clone)]
pub struct Qwen35KvCache {
    layers: Vec<Qwen35LayerKvCache>,
    seq_len: usize,
}

impl Qwen35KvCache {
    pub fn new(model: &Qwen35Model) -> Self {
        Self {
            layers: vec![Qwen35LayerKvCache::default(); model.layers.len()],
            seq_len: 0,
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

impl Qwen35Layer {
    fn forward(
        &self,
        x: TensorId,
        cfg: &Qwen35Config,
        cos: TensorId,
        sin: TensorId,
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

        let h = rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let attn_out = match &self.self_attn {
            Qwen35Attention::Full(attn) => {
                self.forward_full_attention(h, attn, cfg, cos, sin, batch, seq_len, store, tape)?
            }
            Qwen35Attention::Linear(attn) => {
                self.forward_linear_attention(h, attn, cfg, batch, seq_len, store, tape)?
            }
        };
        let x = add(x, attn_out, store, tape)?;

        let h = rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let gate = self.mlp.gate_proj.forward(h, store, tape)?;
        let up = self.mlp.up_proj.forward(h, store, tape)?;
        let gate = silu(gate, store, tape)?;
        let act = mul(gate, up, store, tape)?;
        let mlp_out = self.mlp.down_proj.forward(act, store, tape)?;
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

        let h = rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
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

        let h = rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let gate = self.mlp.gate_proj.forward(h, store, tape)?;
        let up = self.mlp.up_proj.forward(h, store, tape)?;
        let gate = silu(gate, store, tape)?;
        let act = mul(gate, up, store, tape)?;
        let mlp_out = self.mlp.down_proj.forward(act, store, tape)?;
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
        let h = rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
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
        let h = rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        profile.post_attention_rmsnorm += started.elapsed();

        let started = Instant::now();
        let gate = self.mlp.gate_proj.forward(h, store, tape)?;
        let up = self.mlp.up_proj.forward(h, store, tape)?;
        let gate = silu(gate, store, tape)?;
        let act = mul(gate, up, store, tape)?;
        let mlp_out = self.mlp.down_proj.forward(act, store, tape)?;
        profile.mlp += started.elapsed();

        let started = Instant::now();
        let out = add(x, mlp_out, store, tape)?;
        profile.mlp_residual += started.elapsed();

        Ok((out, profile))
    }

    fn forward_full_attention(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
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

        let q = rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
        let k = rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        let q = rope(q, cos, sin, store, tape)?;
        let k = rope(k, cos, sin, store, tape)?;

        let kv_repeat = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k = repeat_kv(k, kv_repeat, store, tape)?;
        let v = repeat_kv(v, kv_repeat, store, tape)?;

        let attn_hidden = causal_sdpa(q, k, v, store, tape)?;
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

            let q = rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
            let k = rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
            let q = rope(q, cos, sin, store, tape)?;
            let k = rope(k, cos, sin, store, tape)?;
            (q, gate, k, v)
        };

        let k_all = append_cached_kv(layer_cache.k, k, store)?;
        let v_all = append_cached_kv(layer_cache.v, v, store)?;
        layer_cache.k = Some(k_all);
        layer_cache.v = Some(v_all);

        let kv_repeat = cfg.num_attention_heads / cfg.num_key_value_heads;
        let kv_len = store
            .get(k_all)
            .ok_or(AutogradError::InvalidTensorId(k_all))?
            .shape
            .get(2)
            .copied()
            .ok_or(AutogradError::InvalidRank {
                expected: "4",
                got: 0,
            })?;
        let attn_hidden = if !tape.enabled && seq_len == 1 && q_start + 1 == kv_len && kv_len <= 32
        {
            causal_sdpa_decode_gqa(q, k_all, v_all, q_start, store, tape)?
        } else {
            let k_all = repeat_kv(k_all, kv_repeat, store, tape)?;
            let v_all = repeat_kv(v_all, kv_repeat, store, tape)?;
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
            let q = rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
            let k = rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
            profile.qk_norm += started.elapsed();

            let started = Instant::now();
            let q = rope(q, cos, sin, store, tape)?;
            let k = rope(k, cos, sin, store, tape)?;
            profile.rope += started.elapsed();
            (q, gate, k, v)
        };

        let started = Instant::now();
        let k_all = append_cached_kv(layer_cache.k, k, store)?;
        let v_all = append_cached_kv(layer_cache.v, v, store)?;
        layer_cache.k = Some(k_all);
        layer_cache.v = Some(v_all);
        profile.append_kv += started.elapsed();

        let kv_repeat = cfg.num_attention_heads / cfg.num_key_value_heads;
        let kv_len = store
            .get(k_all)
            .ok_or(AutogradError::InvalidTensorId(k_all))?
            .shape
            .get(2)
            .copied()
            .ok_or(AutogradError::InvalidRank {
                expected: "4",
                got: 0,
            })?;
        let attn_hidden = if !tape.enabled && seq_len == 1 && q_start + 1 == kv_len && kv_len <= 32
        {
            let started = Instant::now();
            let out = causal_sdpa_decode_gqa(q, k_all, v_all, q_start, store, tape)?;
            profile.sdpa += started.elapsed();
            out
        } else {
            let repeat_started = Instant::now();
            let k_all = repeat_kv(k_all, kv_repeat, store, tape)?;
            let v_all = repeat_kv(v_all, kv_repeat, store, tape)?;
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
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let qkv = attn.in_proj_qkv.forward(h, store, tape)?;
        let z = attn.in_proj_z.forward(h, store, tape)?;
        let b_proj = attn.in_proj_b.forward(h, store, tape)?;
        let a_proj = attn.in_proj_a.forward(h, store, tape)?;
        let linear = linear_attention_core(
            qkv,
            z,
            b_proj,
            a_proj,
            attn.conv1d_weight,
            attn.dt_bias,
            attn.a_log,
            attn.norm,
            LinearAttentionParams {
                batch,
                seq_len,
                num_key_heads: cfg.linear_num_key_heads,
                num_value_heads: cfg.linear_num_value_heads,
                key_dim: cfg.linear_key_head_dim,
                value_dim: cfg.linear_value_head_dim,
                conv_kernel: cfg.linear_conv_kernel_dim,
                eps: cfg.rms_norm_eps,
            },
            store,
            tape,
        )?;
        Ok(attn.out_proj.forward(linear, store, tape)?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen35Model {
    config: Qwen35Config,
    lora: Option<LoraConfig>,
    lora_target_set: LoraTargetSet,
    layers: Vec<Qwen35Layer>,
    embed_tokens: TensorId,
    final_norm: TensorId,
    lm_head: TensorId,
    cos_cache: TensorId,
    sin_cache: TensorId,
    param_names: HashMap<&'static str, TensorId>,
    adapter_names: HashMap<&'static str, TensorId>,
    param_ids: Vec<TensorId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35InitMode {
    ScratchTrain,
    LoraOrFrozen,
}

impl Qwen35Model {
    pub fn new(cfg: &Qwen35Config, store: &mut TensorStore) -> Result<Self> {
        Self::new_internal(
            cfg,
            None,
            LoraTargetSet::AllLinear,
            Qwen35InitMode::ScratchTrain,
            store,
        )
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    pub fn supports_rollout_kv_cache(&self) -> bool {
        self.layers
            .iter()
            .all(|layer| matches!(layer.self_attn, Qwen35Attention::Full(_)))
    }

    pub fn new_for_eval(cfg: &Qwen35Config, store: &mut TensorStore) -> Result<Self> {
        Self::new_internal(
            cfg,
            None,
            LoraTargetSet::AllLinear,
            Qwen35InitMode::LoraOrFrozen,
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
            Qwen35InitMode::LoraOrFrozen,
            store,
        )
    }

    pub fn new_with_lora_targets(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            Some(lora),
            target_set,
            Qwen35InitMode::LoraOrFrozen,
            store,
        )
    }

    pub fn new_lora_from_base(
        base: &Qwen35Model,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        store: &mut TensorStore,
    ) -> Result<Self> {
        let mut model = Self::new_with_lora_targets(&base.config, lora, target_set, store)?;
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
        mode: Qwen35InitMode,
        store: &mut TensorStore,
    ) -> Result<Self> {
        match mode {
            Qwen35InitMode::ScratchTrain => cfg.validate_train_scratch_contract()?,
            Qwen35InitMode::LoraOrFrozen => cfg.validate_train_lora_or_frozen_contract()?,
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

        let embed_tokens_name = cfg.embed_tokens_tensor_name();
        let embed_tokens = normal_parameter(
            embed_tokens_name,
            &[cfg.vocab_size, cfg.hidden_size],
            0.02,
            base_requires_grad,
            store,
        )?;
        register_named(&mut param_names, embed_tokens_name, embed_tokens);

        let lm_head_name = if cfg.tie_word_embeddings {
            cfg.lm_head_tensor_name()
        } else {
            leak_name(format!("{}.lm_head.weight", cfg.model_prefix()))
        };
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens
        } else {
            normal_parameter(
                lm_head_name,
                &[cfg.vocab_size, cfg.hidden_size],
                0.02,
                base_requires_grad,
                store,
            )?
        };
        register_named(&mut param_names, lm_head_name, lm_head);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let names = cfg.layer_tensor_names(layer_idx);
            let input_layernorm_name = leak_name(names.common.input_layernorm);
            let post_attention_layernorm_name = leak_name(names.common.post_attention_layernorm);
            let gate_proj_name = leak_name(names.common.mlp_gate_proj);
            let up_proj_name = leak_name(names.common.mlp_up_proj);
            let down_proj_name = leak_name(names.common.mlp_down_proj);

            let input_layernorm = ones_parameter(
                input_layernorm_name,
                &[cfg.hidden_size],
                base_requires_grad,
                store,
            )?;
            let gate_proj = LinearWithLora::new(
                gate_proj_name,
                cfg.hidden_size,
                cfg.intermediate_size,
                base_requires_grad,
                lora_for_name(lora, lora_target_set, gate_proj_name),
                store,
            )?;
            let up_proj = LinearWithLora::new(
                up_proj_name,
                cfg.hidden_size,
                cfg.intermediate_size,
                base_requires_grad,
                lora_for_name(lora, lora_target_set, up_proj_name),
                store,
            )?;
            let down_proj = LinearWithLora::new(
                down_proj_name,
                cfg.intermediate_size,
                cfg.hidden_size,
                base_requires_grad,
                lora_for_name(lora, lora_target_set, down_proj_name),
                store,
            )?;
            let post_attention_layernorm = ones_parameter(
                post_attention_layernorm_name,
                &[cfg.hidden_size],
                base_requires_grad,
                store,
            )?;

            register_named(&mut param_names, input_layernorm_name, input_layernorm);
            for (name, id) in gate_proj.parameter_name_map() {
                register_named(&mut param_names, name, id);
            }
            for (name, id) in gate_proj.adapter_name_map() {
                register_named(&mut adapter_names, name, id);
            }
            for (name, id) in up_proj.parameter_name_map() {
                register_named(&mut param_names, name, id);
            }
            for (name, id) in up_proj.adapter_name_map() {
                register_named(&mut adapter_names, name, id);
            }
            for (name, id) in down_proj.parameter_name_map() {
                register_named(&mut param_names, name, id);
            }
            for (name, id) in down_proj.adapter_name_map() {
                register_named(&mut adapter_names, name, id);
            }
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

                    let q_proj = LinearWithLora::new(
                        q_proj_name,
                        cfg.hidden_size,
                        cfg.full_attn_q_proj_dim(),
                        base_requires_grad,
                        lora_for_name(lora, lora_target_set, q_proj_name),
                        store,
                    )?;
                    let k_proj = LinearWithLora::new(
                        k_proj_name,
                        cfg.hidden_size,
                        cfg.full_attn_kv_dim(),
                        base_requires_grad,
                        lora_for_name(lora, lora_target_set, k_proj_name),
                        store,
                    )?;
                    let v_proj = LinearWithLora::new(
                        v_proj_name,
                        cfg.hidden_size,
                        cfg.full_attn_kv_dim(),
                        base_requires_grad,
                        lora_for_name(lora, lora_target_set, v_proj_name),
                        store,
                    )?;
                    let o_proj = LinearWithLora::new(
                        o_proj_name,
                        cfg.full_attn_q_dim(),
                        cfg.hidden_size,
                        base_requires_grad,
                        lora_for_name(lora, lora_target_set, o_proj_name),
                        store,
                    )?;
                    let q_norm =
                        ones_parameter(q_norm_name, &[cfg.head_dim], base_requires_grad, store)?;
                    let k_norm =
                        ones_parameter(k_norm_name, &[cfg.head_dim], base_requires_grad, store)?;

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

                    let in_proj_qkv = LinearWithLora::new(
                        in_proj_qkv_name,
                        cfg.hidden_size,
                        cfg.linear_attn_qkv_dim(),
                        base_requires_grad,
                        lora_for_name(lora, lora_target_set, in_proj_qkv_name),
                        store,
                    )?;
                    let in_proj_z = LinearWithLora::new(
                        in_proj_z_name,
                        cfg.hidden_size,
                        cfg.linear_attn_z_dim(),
                        base_requires_grad,
                        lora_for_name(lora, lora_target_set, in_proj_z_name),
                        store,
                    )?;
                    let in_proj_b = LinearWithLora::new(
                        in_proj_b_name,
                        cfg.hidden_size,
                        cfg.linear_num_value_heads,
                        base_requires_grad,
                        lora_for_name(lora, lora_target_set, in_proj_b_name),
                        store,
                    )?;
                    let in_proj_a = LinearWithLora::new(
                        in_proj_a_name,
                        cfg.hidden_size,
                        cfg.linear_num_value_heads,
                        base_requires_grad,
                        lora_for_name(lora, lora_target_set, in_proj_a_name),
                        store,
                    )?;
                    let conv1d_weight = normal_parameter(
                        conv1d_weight_name,
                        &[cfg.linear_attn_qkv_dim(), cfg.linear_conv_kernel_dim],
                        0.02,
                        base_requires_grad,
                        store,
                    )?;
                    let dt_bias = normal_parameter(
                        dt_bias_name,
                        &[cfg.linear_num_value_heads],
                        0.02,
                        base_requires_grad,
                        store,
                    )?;
                    let a_log = normal_parameter(
                        a_log_name,
                        &[cfg.linear_num_value_heads],
                        0.02,
                        base_requires_grad,
                        store,
                    )?;
                    let norm = ones_parameter(
                        norm_name,
                        &[cfg.linear_value_head_dim],
                        base_requires_grad,
                        store,
                    )?;
                    let out_proj = LinearWithLora::new(
                        out_proj_name,
                        cfg.linear_attn_z_dim(),
                        cfg.hidden_size,
                        base_requires_grad,
                        lora_for_name(lora, lora_target_set, out_proj_name),
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
                mlp: Qwen35Mlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
            });
        }

        let final_norm_name = cfg.norm_tensor_name();
        let final_norm = ones_parameter(
            final_norm_name,
            &[cfg.hidden_size],
            base_requires_grad,
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
            lora,
            lora_target_set,
            layers,
            embed_tokens,
            final_norm,
            lm_head,
            cos_cache,
            sin_cache,
            param_names,
            adapter_names,
            param_ids,
        })
    }

    pub fn all_parameter_ids(&self) -> Vec<TensorId> {
        self.param_ids.clone()
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
            layer
                .mlp
                .gate_proj
                .set_base_weight(base_layer.mlp.gate_proj.base_weight());
            layer
                .mlp
                .up_proj
                .set_base_weight(base_layer.mlp.up_proj.base_weight());
            layer
                .mlp
                .down_proj
                .set_base_weight(base_layer.mlp.down_proj.base_weight());
        }

        self.param_names = base.param_names.clone();
        let adapter_ids = self.adapter_names.values().copied().collect::<HashSet<_>>();
        let mut seen = HashSet::new();
        let mut param_ids = Vec::with_capacity(base.param_ids.len() + adapter_ids.len());
        for &id in &base.param_ids {
            if seen.insert(id) {
                param_ids.push(id);
            }
        }
        for &id in &self.param_ids {
            if adapter_ids.contains(&id) && seen.insert(id) {
                param_ids.push(id);
            }
        }
        self.param_ids = param_ids;
        Ok(())
    }

    pub fn clone_frozen(&self, store: &mut TensorStore) -> Self {
        let cloned = match self.lora {
            Some(lora) => {
                Self::new_with_lora_targets(&self.config, lora, self.lora_target_set, store)
            }
            None => Self::new_for_eval(&self.config, store),
        }
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
        let hidden = rmsnorm(
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
        let hidden = rmsnorm(
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
        let seq_len = positions.len();
        if token_indices.len() != batch * seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: token_indices.len(),
                expected_len: batch * seq_len,
            });
        }
        let cos = select_cache_rows(self.cos_cache, positions, store)?;
        let sin = select_cache_rows(self.sin_cache, positions, store)?;

        let mut hidden = embedding(self.embed_tokens, token_indices, store, tape)?;
        hidden = reshape(
            hidden,
            &[batch, seq_len, self.config.hidden_size],
            store,
            tape,
        )?;
        for layer in &self.layers {
            hidden = layer.forward(hidden, &self.config, cos, sin, store, tape)?;
        }
        let hidden = rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        linear_forward(hidden, self.lm_head, store, tape)
    }

    fn forward_batch_indices_with_kv_cache(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        cache: &mut Qwen35KvCache,
    ) -> Result<TensorId> {
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
        let hidden = rmsnorm(
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
        let hidden = rmsnorm(
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
            let merged_gate = {
                let tensor = layer.mlp.gate_proj.merged_tensor(store)?;
                store.alloc(tensor)
            };
            let merged_up = {
                let tensor = layer.mlp.up_proj.merged_tensor(store)?;
                store.alloc(tensor)
            };
            let merged_down = {
                let tensor = layer.mlp.down_proj.merged_tensor(store)?;
                store.alloc(tensor)
            };

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
            for (name, _) in layer.mlp.gate_proj.parameter_name_map() {
                map.insert(name, merged_gate);
            }
            for (name, _) in layer.mlp.up_proj.parameter_name_map() {
                map.insert(name, merged_up);
            }
            for (name, _) in layer.mlp.down_proj.parameter_name_map() {
                map.insert(name, merged_down);
            }
        }
        Ok(map)
    }
}

impl CausalLm for Qwen35Model {
    fn forward_with_positions(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> autograd::Result<TensorId> {
        Qwen35Model::forward(self, store, tape, input_ids, position_ids).map_err(qwen35_to_autograd)
    }

    fn param_name_map(&self) -> HashMap<&'static str, TensorId> {
        Qwen35Model::param_name_map(self)
    }

    fn adapter_name_map(&self) -> HashMap<&'static str, TensorId> {
        Qwen35Model::adapter_name_map(self)
    }

    fn materialized_param_name_map(
        &self,
        store: &mut TensorStore,
        _tape: &mut Tape,
    ) -> autograd::Result<HashMap<&'static str, TensorId>> {
        Qwen35Model::materialized_param_name_map(self, store).map_err(qwen35_to_autograd)
    }

    fn all_parameter_ids(&self) -> Vec<TensorId> {
        Qwen35Model::all_parameter_ids(self)
    }
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
    let projected = matmul_bt(flat_x, weight, store, tape)?;
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
    for (name, id) in linear.adapter_name_map() {
        register_named(adapter_names, name, id);
    }
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

fn lora_for_name(
    lora: Option<LoraConfig>,
    target_set: LoraTargetSet,
    base_name: &str,
) -> Option<LoraConfig> {
    lora.filter(|_| target_set.includes(base_name))
}

fn qwen35_to_autograd(err: Qwen35Error) -> AutogradError {
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
    store: &mut TensorStore,
) -> Result<TensorId> {
    let Some(cached) = cached else {
        return Ok(next);
    };

    let cached_shape = store
        .get(cached)
        .ok_or(AutogradError::InvalidTensorId(cached))?
        .shape
        .clone();
    let next_shape = store
        .get(next)
        .ok_or(AutogradError::InvalidTensorId(next))?
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
    let (out_handle, out_shape) =
        store
            .backend()
            .concat_axis2(&cached_handle, &cached_shape, &next_handle, &next_shape)?;
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

fn seed_from_name(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn next_uniform(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    bits as f32 / (u32::MAX >> 8) as f32
}

fn leak_name(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use autograd::{Tape, TensorId, TensorStore};

    use super::{LayerType, Qwen35Config, Qwen35KvCache, Qwen35Model};

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

    fn logits_host(store: &mut TensorStore, logits: TensorId) -> TestResult<Vec<f32>> {
        Ok(store.to_host(logits)?)
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
    fn qwen35_rollout_kv_cache_matches_full_forward_tokens() -> TestResult {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        tape.set_enabled(false);

        let cfg = tiny_qwen35_config(16);
        let model = Qwen35Model::new_for_eval(&cfg, &mut store)?;
        let vocab = cfg.vocab_size;
        let mut cache = Qwen35KvCache::new(&model);
        let mut rollout = vec![1_u32, 3, 8];

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
}
