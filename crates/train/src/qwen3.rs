use std::{collections::HashMap, f32::consts::TAU};

use autograd::{
    AutogradError, Tape, Tensor, TensorId, TensorStore,
    ops::{
        add, causal_sdpa, embedding, matmul, mul, repeat_kv, reshape, rmsnorm, rope, silu,
        transpose,
    },
};
pub use qwen3_spec::{Qwen3Config, Qwen3ConfigError};
use thiserror::Error;

use crate::policy::{GrpoPolicy, GrpoPolicyConfig};

#[derive(Debug, Error)]
pub enum Qwen3Error {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Config(#[from] Qwen3ConfigError),
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

pub type Result<T> = std::result::Result<T, Qwen3Error>;

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
        let q = linear_forward(h, self.self_attn.q_proj, store, tape)?;
        let k = linear_forward(h, self.self_attn.k_proj, store, tape)?;
        let v = linear_forward(h, self.self_attn.v_proj, store, tape)?;

        let q = split_heads(
            q,
            batch,
            seq_len,
            cfg.num_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let k = split_heads(
            q_or_kv_heads_tensor(k),
            batch,
            seq_len,
            cfg.num_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let v = split_heads(
            q_or_kv_heads_tensor(v),
            batch,
            seq_len,
            cfg.num_key_value_heads,
            cfg.head_dim,
            store,
            tape,
        )?;

        let q = rmsnorm(q, self.self_attn.q_norm, cfg.rms_norm_eps, store, tape)?;
        let k = rmsnorm(k, self.self_attn.k_norm, cfg.rms_norm_eps, store, tape)?;
        let q = rope(q, cos, sin, store, tape)?;
        let k = rope(k, cos, sin, store, tape)?;
        let kv_repeat = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k = repeat_kv(k, kv_repeat, store, tape)?;
        let v = repeat_kv(v, kv_repeat, store, tape)?;
        let attn = causal_sdpa(q, k, v, store, tape)?;
        let attn = merge_heads(
            attn,
            batch,
            seq_len,
            cfg.num_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        let attn_out = linear_forward(attn, self.self_attn.o_proj, store, tape)?;
        let x = add(x, attn_out, store, tape)?;

        let h = rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
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
    param_ids: Vec<TensorId>,
}

impl Qwen3Model {
    pub fn new(cfg: &Qwen3Config, store: &mut TensorStore) -> Result<Self> {
        cfg.validate()?;

        let mut param_names = HashMap::new();
        let mut param_ids = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut register_param = |name: &'static str, id: TensorId| {
            param_names.insert(name, id);
            if seen.insert(id) {
                param_ids.push(id);
            }
        };
        let embed_tokens = normal_parameter(
            "model.embed_tokens.weight",
            &[cfg.vocab_size, cfg.hidden_size],
            0.02,
            store,
        )?;
        register_param("model.embed_tokens.weight", embed_tokens);

        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens
        } else {
            normal_parameter(
                "lm_head.weight",
                &[cfg.vocab_size, cfg.hidden_size],
                0.02,
                store,
            )?
        };
        register_param("lm_head.weight", lm_head);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            let input_layernorm = ones_parameter(
                leak_name(format!("{prefix}.input_layernorm.weight")),
                &[cfg.hidden_size],
                store,
            )?;
            let q_proj = normal_parameter(
                leak_name(format!("{prefix}.self_attn.q_proj.weight")),
                &[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size],
                0.02,
                store,
            )?;
            let k_proj = normal_parameter(
                leak_name(format!("{prefix}.self_attn.k_proj.weight")),
                &[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size],
                0.02,
                store,
            )?;
            let v_proj = normal_parameter(
                leak_name(format!("{prefix}.self_attn.v_proj.weight")),
                &[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size],
                0.02,
                store,
            )?;
            let o_proj = normal_parameter(
                leak_name(format!("{prefix}.self_attn.o_proj.weight")),
                &[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim],
                0.02,
                store,
            )?;
            let q_norm = ones_parameter(
                leak_name(format!("{prefix}.self_attn.q_norm.weight")),
                &[cfg.head_dim],
                store,
            )?;
            let k_norm = ones_parameter(
                leak_name(format!("{prefix}.self_attn.k_norm.weight")),
                &[cfg.head_dim],
                store,
            )?;
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
            let post_attention_layernorm = ones_parameter(
                leak_name(format!("{prefix}.post_attention_layernorm.weight")),
                &[cfg.hidden_size],
                store,
            )?;

            register_param(
                leak_name(format!("{prefix}.input_layernorm.weight")),
                input_layernorm,
            );
            register_param(
                leak_name(format!("{prefix}.self_attn.q_proj.weight")),
                q_proj,
            );
            register_param(
                leak_name(format!("{prefix}.self_attn.k_proj.weight")),
                k_proj,
            );
            register_param(
                leak_name(format!("{prefix}.self_attn.v_proj.weight")),
                v_proj,
            );
            register_param(
                leak_name(format!("{prefix}.self_attn.o_proj.weight")),
                o_proj,
            );
            register_param(
                leak_name(format!("{prefix}.self_attn.q_norm.weight")),
                q_norm,
            );
            register_param(
                leak_name(format!("{prefix}.self_attn.k_norm.weight")),
                k_norm,
            );
            register_param(
                leak_name(format!("{prefix}.mlp.gate_proj.weight")),
                gate_proj,
            );
            register_param(leak_name(format!("{prefix}.mlp.up_proj.weight")), up_proj);
            register_param(
                leak_name(format!("{prefix}.mlp.down_proj.weight")),
                down_proj,
            );
            register_param(
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
        register_param("model.norm.weight", final_norm);

        let (cos_cache, sin_cache) = build_rope_cache(cfg, store)?;
        if seen.insert(cos_cache) {
            param_ids.push(cos_cache);
        }
        if seen.insert(sin_cache) {
            param_ids.push(sin_cache);
        }

        Ok(Self {
            config: cfg.clone(),
            layers,
            embed_tokens,
            final_norm,
            lm_head,
            cos_cache,
            sin_cache,
            param_names,
            param_ids,
        })
    }

    pub fn all_parameter_ids(&self) -> Vec<TensorId> {
        self.param_ids.clone()
    }

    pub fn clone_frozen(&self, store: &mut TensorStore) -> Self {
        let cloned = Self::new(&self.config, store).expect("clone_frozen should preserve config");
        let source_ids = self.all_parameter_ids();
        let target_ids = cloned.all_parameter_ids();
        assert_eq!(
            source_ids.len(),
            target_ids.len(),
            "clone_frozen parameter topology drifted",
        );

        for (source_id, target_id) in source_ids.into_iter().zip(target_ids) {
            let mut replacement = store
                .get(source_id)
                .cloned()
                .expect("source parameter should remain readable");
            replacement.requires_grad = false;
            replacement.grad = None;
            store.tensors[target_id] = Some(replacement);
        }

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
        let position_ids = (0..seq_len).map(|index| index as u32).collect::<Vec<_>>();
        let input_ids = input_ids.iter().map(|&id| id as u32).collect::<Vec<_>>();
        self.forward_batch(store, tape, &input_ids, &position_ids, batch, seq_len)
            .map_err(qwen3_to_autograd)
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
            return Err(Qwen3Error::InputLenMismatch {
                input_len: input_ids.len(),
                position_len: batch * seq_len,
            });
        }
        if position_ids.len() != seq_len {
            return Err(Qwen3Error::InputLenMismatch {
                input_len: input_ids.len(),
                position_len: position_ids.len(),
            });
        }
        if seq_len > self.config.max_position_embeddings {
            return Err(Qwen3Error::InvalidConfig(
                "sequence length exceeds context window",
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
}

impl GrpoPolicyConfig for Qwen3Config {
    fn max_seq_len(&self) -> usize {
        self.max_position_embeddings
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl GrpoPolicy for Qwen3Model {
    type Config = Qwen3Config;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn forward_single(
        &self,
        input_ids: &[usize],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> autograd::Result<TensorId> {
        Qwen3Model::forward_tokens(self, input_ids, store, tape)
    }

    fn forward_batch_tokens(
        &self,
        input_ids: &[usize],
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> autograd::Result<TensorId> {
        Qwen3Model::forward_batch_tokens(self, input_ids, batch, seq_len, store, tape)
    }

    fn all_parameter_ids(&self) -> Vec<TensorId> {
        Qwen3Model::all_parameter_ids(self)
    }

    fn clone_frozen(&self, store: &mut TensorStore) -> Self {
        Qwen3Model::clone_frozen(self, store)
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

fn qwen3_to_autograd(err: Qwen3Error) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(err.to_string().into_boxed_str()))
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
    // Borrow the cache tensor — the full RoPE cache is ~max_position_embeddings *
    // head_dim/2 floats (8–10 MiB for Qwen3 configs); cloning it twice per
    // forward would be O(max_pos) instead of O(seq_len).
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
            return Err(Qwen3Error::PositionOutOfBounds {
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

fn build_rope_cache(cfg: &Qwen3Config, store: &mut TensorStore) -> Result<(TensorId, TensorId)> {
    let half_dim = cfg.head_dim / 2;
    let inv_freq = (0..half_dim)
        .map(|index| {
            1.0 / cfg
                .rope_theta
                .powf((2.0 * index as f32) / cfg.head_dim as f32)
        })
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

    let cos_cache = store.alloc(Tensor::new(
        cos,
        vec![cfg.max_position_embeddings, half_dim],
        false,
    )?);
    let sin_cache = store.alloc(Tensor::new(
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
    Ok(store.alloc(Tensor::new(data, shape.to_vec(), true)?))
}

fn ones_parameter(
    name: &'static str,
    shape: &[usize],
    store: &mut TensorStore,
) -> Result<TensorId> {
    let _ = name;
    Ok(store.alloc(Tensor::new(
        vec![1.0; shape.iter().product()],
        shape.to_vec(),
        true,
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

fn q_or_kv_heads_tensor(x: TensorId) -> TensorId {
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::{Tape, TensorStore};

    #[test]
    fn batch_forward_matches_repeated_single_forward() {
        let cfg = Qwen3Config {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            max_position_embeddings: 8,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: false,
        };

        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let model = Qwen3Model::new(&cfg, &mut store).expect("model");

        let single_ids = vec![1, 2, 3, 4];
        let single = model
            .forward_tokens(&single_ids, &mut store, &mut tape)
            .expect("single forward");
        let single_logits = store.to_host(single).expect("single logits");

        tape.entries.clear();
        let batch_ids = [single_ids.clone(), single_ids.clone()].concat();
        let batched = model
            .forward_batch_tokens(&batch_ids, 2, single_ids.len(), &mut store, &mut tape)
            .expect("batch forward");
        let batched_logits = store.to_host(batched).expect("batch logits");

        assert_eq!(batched_logits.len(), 2 * single_logits.len());
        assert_eq!(&batched_logits[..single_logits.len()], &single_logits[..]);
        assert_eq!(&batched_logits[single_logits.len()..], &single_logits[..]);
    }
}
