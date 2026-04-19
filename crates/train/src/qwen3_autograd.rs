use std::{
    collections::HashMap,
    f32::consts::TAU,
    fs,
    path::Path,
};

use autograd::{
    AutogradError, GpuTensor, Tape, TensorId, TensorStore,
    ops::{add, causal_sdpa, embedding, matmul, mul, repeat_kv, reshape, rmsnorm, rope, silu, transpose},
};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Qwen3AutogradError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("missing or invalid config field `{field}`")]
    InvalidConfigField { field: &'static str },
    #[error("invalid qwen3 config: {0}")]
    InvalidConfig(&'static str),
    #[error("input_ids len {input_len} does not match position_ids len {position_len}")]
    InputLenMismatch {
        input_len: usize,
        position_len: usize,
    },
    #[error("position id {position} is out of bounds for rope cache size {upper}")]
    PositionOutOfBounds { position: usize, upper: usize },
}

pub type Result<T> = std::result::Result<T, Qwen3AutogradError>;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

impl Qwen3Config {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let value: Value = serde_json::from_str(&json)?;
        let cfg = Self {
            vocab_size: read_usize(&value, "vocab_size")?,
            hidden_size: read_usize(&value, "hidden_size")?,
            num_hidden_layers: read_usize(&value, "num_hidden_layers")?,
            num_attention_heads: read_usize(&value, "num_attention_heads")?,
            num_kv_heads: read_usize_alias(&value, "num_kv_heads", "num_key_value_heads")?,
            head_dim: read_usize(&value, "head_dim")?,
            intermediate_size: read_usize(&value, "intermediate_size")?,
            max_position_embeddings: read_usize(&value, "max_position_embeddings")?,
            rms_norm_eps: read_f32(&value, "rms_norm_eps")?,
            rope_theta: read_f32(&value, "rope_theta")?,
            tie_word_embeddings: read_bool(&value, "tie_word_embeddings")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.hidden_size != self.num_attention_heads * self.head_dim {
            return Err(Qwen3AutogradError::InvalidConfig(
                "hidden_size must equal num_attention_heads * head_dim",
            ));
        }
        if self.num_attention_heads == 0 || self.num_kv_heads == 0 || self.head_dim == 0 {
            return Err(Qwen3AutogradError::InvalidConfig(
                "attention heads and head_dim must be non-zero",
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_kv_heads) {
            return Err(Qwen3AutogradError::InvalidConfig(
                "num_attention_heads must be divisible by num_kv_heads",
            ));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(Qwen3AutogradError::InvalidConfig(
                "head_dim must be even for RoPE",
            ));
        }
        if self.max_position_embeddings == 0 {
            return Err(Qwen3AutogradError::InvalidConfig(
                "max_position_embeddings must be non-zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Qwen3Attention {
    q_proj: TensorId,
    k_proj: TensorId,
    v_proj: TensorId,
    o_proj: TensorId,
    q_norm: TensorId,
    k_norm: TensorId,
}

#[derive(Debug, Clone)]
struct Qwen3Mlp {
    gate_proj: TensorId,
    up_proj: TensorId,
    down_proj: TensorId,
}

#[derive(Debug, Clone)]
struct Qwen3Layer {
    input_layernorm: TensorId,
    self_attn: Qwen3Attention,
    post_attention_layernorm: TensorId,
    mlp: Qwen3Mlp,
}

impl Qwen3Layer {
    fn forward(
        &self,
        x: TensorId,
        cfg: &Qwen3Config,
        cos: TensorId,
        sin: TensorId,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let seq_len = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .get(1)
            .copied()
            .ok_or(AutogradError::InvalidRank {
                expected: "rank-3 hidden states",
                got: 0,
            })?;

        let h = rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let q = linear_forward(h, self.self_attn.q_proj, store, tape)?;
        let k = linear_forward(h, self.self_attn.k_proj, store, tape)?;
        let v = linear_forward(h, self.self_attn.v_proj, store, tape)?;

        let q = split_heads(q, 1, seq_len, cfg.num_attention_heads, cfg.head_dim, store, tape)?;
        let k = split_heads(q_or_kv_heads_tensor(k), 1, seq_len, cfg.num_kv_heads, cfg.head_dim, store, tape)?;
        let v = split_heads(q_or_kv_heads_tensor(v), 1, seq_len, cfg.num_kv_heads, cfg.head_dim, store, tape)?;

        let q = rmsnorm(q, self.self_attn.q_norm, cfg.rms_norm_eps, store, tape)?;
        let k = rmsnorm(k, self.self_attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        let q = rope(q, cos, sin, store, tape)?;
        let k = rope(k, cos, sin, store, tape)?;
        let kv_repeat = cfg.num_attention_heads / cfg.num_kv_heads;
        let k = repeat_kv(k, kv_repeat, store, tape)?;
        let v = repeat_kv(v, kv_repeat, store, tape)?;
        let attn = causal_sdpa(q, k, v, store, tape)?;
        let attn = merge_heads(attn, 1, seq_len, cfg.num_attention_heads, cfg.head_dim, store, tape)?;
        let attn_out = linear_forward(attn, self.self_attn.o_proj, store, tape)?;
        let x = add(x, attn_out, store, tape)?;

        let h = rmsnorm(x, self.post_attention_layernorm, cfg.rms_norm_eps, store, tape)?;
        let gate = linear_forward(h, self.mlp.gate_proj, store, tape)?;
        let up = linear_forward(h, self.mlp.up_proj, store, tape)?;
        let gate = silu(gate, store, tape)?;
        let act = mul(gate, up, store, tape)?;
        let mlp_out = linear_forward(act, self.mlp.down_proj, store, tape)?;
        Ok(add(x, mlp_out, store, tape)?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3Model {
    config: Qwen3Config,
    layers: Vec<Qwen3Layer>,
    embed_tokens: TensorId,
    final_norm: TensorId,
    lm_head: TensorId,
    cos_cache: TensorId,
    sin_cache: TensorId,
    param_names: HashMap<&'static str, TensorId>,
}

impl Qwen3Model {
    pub fn new(cfg: &Qwen3Config, store: &mut TensorStore) -> Result<Self> {
        cfg.validate()?;

        let mut param_names = HashMap::new();
        let embed_tokens = normal_parameter("model.embed_tokens.weight", &[cfg.vocab_size, cfg.hidden_size], 0.02, store)?;
        param_names.insert("model.embed_tokens.weight", embed_tokens);

        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens
        } else {
            normal_parameter("lm_head.weight", &[cfg.vocab_size, cfg.hidden_size], 0.02, store)?
        };
        param_names.insert("lm_head.weight", lm_head);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            let input_layernorm = ones_parameter(leak_name(format!("{prefix}.input_layernorm.weight")), &[cfg.hidden_size], store)?;
            let q_proj = normal_parameter(
                leak_name(format!("{prefix}.self_attn.q_proj.weight")),
                &[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size],
                0.02,
                store,
            )?;
            let k_proj = normal_parameter(
                leak_name(format!("{prefix}.self_attn.k_proj.weight")),
                &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
                0.02,
                store,
            )?;
            let v_proj = normal_parameter(
                leak_name(format!("{prefix}.self_attn.v_proj.weight")),
                &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
                0.02,
                store,
            )?;
            let o_proj = normal_parameter(
                leak_name(format!("{prefix}.self_attn.o_proj.weight")),
                &[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim],
                0.02,
                store,
            )?;
            let q_norm = ones_parameter(leak_name(format!("{prefix}.self_attn.q_norm.weight")), &[cfg.head_dim], store)?;
            let k_norm = ones_parameter(leak_name(format!("{prefix}.self_attn.k_norm.weight")), &[cfg.head_dim], store)?;
            let gate_proj = normal_parameter(
                leak_name(format!("{prefix}.mlp.gate_proj.weight")),
                &[cfg.intermediate_size, cfg.hidden_size],
                0.02,
                store,
            )?;
            let up_proj = normal_parameter(
                leak_name(format!("{prefix}.mlp.up_proj.weight")),
                &[cfg.intermediate_size, cfg.hidden_size],
                0.02,
                store,
            )?;
            let down_proj = normal_parameter(
                leak_name(format!("{prefix}.mlp.down_proj.weight")),
                &[cfg.hidden_size, cfg.intermediate_size],
                0.02,
                store,
            )?;
            let post_attention_layernorm =
                ones_parameter(leak_name(format!("{prefix}.post_attention_layernorm.weight")), &[cfg.hidden_size], store)?;

            param_names.insert(leak_name(format!("{prefix}.input_layernorm.weight")), input_layernorm);
            param_names.insert(leak_name(format!("{prefix}.self_attn.q_proj.weight")), q_proj);
            param_names.insert(leak_name(format!("{prefix}.self_attn.k_proj.weight")), k_proj);
            param_names.insert(leak_name(format!("{prefix}.self_attn.v_proj.weight")), v_proj);
            param_names.insert(leak_name(format!("{prefix}.self_attn.o_proj.weight")), o_proj);
            param_names.insert(leak_name(format!("{prefix}.self_attn.q_norm.weight")), q_norm);
            param_names.insert(leak_name(format!("{prefix}.self_attn.k_norm.weight")), k_norm);
            param_names.insert(leak_name(format!("{prefix}.mlp.gate_proj.weight")), gate_proj);
            param_names.insert(leak_name(format!("{prefix}.mlp.up_proj.weight")), up_proj);
            param_names.insert(leak_name(format!("{prefix}.mlp.down_proj.weight")), down_proj);
            param_names.insert(
                leak_name(format!("{prefix}.post_attention_layernorm.weight")),
                post_attention_layernorm,
            );

            layers.push(Qwen3Layer {
                input_layernorm,
                self_attn: Qwen3Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                },
                post_attention_layernorm,
                mlp: Qwen3Mlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
            });
        }

        let final_norm = ones_parameter("model.norm.weight", &[cfg.hidden_size], store)?;
        param_names.insert("model.norm.weight", final_norm);

        let (cos_cache, sin_cache) = build_rope_cache(cfg, store)?;

        Ok(Self {
            config: cfg.clone(),
            layers,
            embed_tokens,
            final_norm,
            lm_head,
            cos_cache,
            sin_cache,
            param_names,
        })
    }

    pub fn forward(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> Result<TensorId> {
        if input_ids.len() != position_ids.len() {
            return Err(Qwen3AutogradError::InputLenMismatch {
                input_len: input_ids.len(),
                position_len: position_ids.len(),
            });
        }
        if input_ids.is_empty() {
            return Err(Qwen3AutogradError::InvalidConfig(
                "forward requires at least one token",
            ));
        }

        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let cos = select_cache_rows(self.cos_cache, &positions, store)?;
        let sin = select_cache_rows(self.sin_cache, &positions, store)?;

        let mut hidden = embedding(self.embed_tokens, &token_indices, store, tape)?;
        for layer in &self.layers {
            hidden = layer.forward(hidden, &self.config, cos, sin, store, tape)?;
        }
        let hidden = rmsnorm(hidden, self.final_norm, self.config.rms_norm_eps, store, tape)?;
        linear_forward(hidden, self.lm_head, store, tape)
    }

    pub fn param_name_map(&self) -> HashMap<&'static str, TensorId> {
        self.param_names.clone()
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
    let weight_t = transpose(weight, 0, 1, store, tape)?;
    let projected = matmul(flat_x, weight_t, store, tape)?;
    let mut output_shape = x_shape[..x_shape.len() - 1].to_vec();
    output_shape.push(weight_shape[0]);
    Ok(reshape(projected, &output_shape, store, tape)?)
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
    Ok(reshape(x, &[batch, seq_len, heads * head_dim], store, tape)?)
}

fn select_cache_rows(cache: TensorId, position_ids: &[usize], store: &mut TensorStore) -> Result<TensorId> {
    let cache_tensor = store
        .get(cache)
        .ok_or(AutogradError::InvalidTensorId(cache))?
        .clone();
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
            return Err(Qwen3AutogradError::PositionOutOfBounds {
                position,
                upper: rows,
            });
        }
        let base = position * cols;
        data.extend_from_slice(&cache_tensor.data[base..base + cols]);
    }
    Ok(store.alloc(GpuTensor::new(data, vec![position_ids.len(), cols], false)?))
}

fn build_rope_cache(cfg: &Qwen3Config, store: &mut TensorStore) -> Result<(TensorId, TensorId)> {
    let half_dim = cfg.head_dim / 2;
    let inv_freq = (0..half_dim)
        .map(|index| 1.0 / cfg.rope_theta.powf((2.0 * index as f32) / cfg.head_dim as f32))
        .collect::<Vec<_>>();
    let mut cos = vec![0.0; cfg.max_position_embeddings * half_dim];
    let mut sin = vec![0.0; cfg.max_position_embeddings * half_dim];

    for position in 0..cfg.max_position_embeddings {
        let base = position * half_dim;
        for (freq_index, &freq) in inv_freq.iter().enumerate() {
            let angle = position as f32 * freq;
            cos[base + freq_index] = angle.cos();
            sin[base + freq_index] = angle.sin();
        }
    }

    let cos_cache = store.alloc(GpuTensor::new(
        cos,
        vec![cfg.max_position_embeddings, half_dim],
        false,
    )?);
    let sin_cache = store.alloc(GpuTensor::new(
        sin,
        vec![cfg.max_position_embeddings, half_dim],
        false,
    )?);
    Ok((cos_cache, sin_cache))
}

fn normal_parameter(
    name: &'static str,
    shape: &[usize],
    std: f32,
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
    Ok(store.alloc(GpuTensor::new(data, shape.to_vec(), true)?))
}

fn ones_parameter(name: &'static str, shape: &[usize], store: &mut TensorStore) -> Result<TensorId> {
    let _ = name;
    Ok(store.alloc(GpuTensor::new(vec![1.0; shape.iter().product()], shape.to_vec(), true)?))
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

fn read_usize(value: &Value, field: &'static str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .map(|raw| raw as usize)
        .ok_or(Qwen3AutogradError::InvalidConfigField { field })
}

fn read_usize_alias(value: &Value, field: &'static str, alias: &'static str) -> Result<usize> {
    value
        .get(field)
        .or_else(|| value.get(alias))
        .and_then(Value::as_u64)
        .map(|raw| raw as usize)
        .ok_or(Qwen3AutogradError::InvalidConfigField { field })
}

fn read_f32(value: &Value, field: &'static str) -> Result<f32> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .map(|raw| raw as f32)
        .ok_or(Qwen3AutogradError::InvalidConfigField { field })
}

fn read_bool(value: &Value, field: &'static str) -> Result<bool> {
    value
        .get(field)
        .and_then(Value::as_bool)
        .ok_or(Qwen3AutogradError::InvalidConfigField { field })
}

fn q_or_kv_heads_tensor(x: TensorId) -> TensorId {
    x
}
