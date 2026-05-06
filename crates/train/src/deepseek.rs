use std::{collections::HashMap, f32::consts::TAU};

use autograd::{
    AutogradError, Result, Tape, Tensor, TensorId, TensorStore,
    ops::{add, causal_sdpa, embedding, matmul, mul, reshape, rmsnorm, silu, slice, transpose},
};
use deepseek_spec::{DeepSeekConfig, DeepSeekConfigError, DeepSeekMlpTensorNames};
use thiserror::Error;

use crate::{
    causal_lm::CausalLm,
    policy::{GrpoPolicy, GrpoPolicyConfig},
};

#[derive(Debug, Error)]
pub enum DeepseekTrainError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Config(#[from] DeepSeekConfigError),
    #[error("invalid DeepSeek nano train config: {0}")]
    InvalidConfig(&'static str),
    #[error("input_ids len {input_len} does not match expected len {position_len}")]
    InputLenMismatch {
        input_len: usize,
        position_len: usize,
    },
    #[error("position {position} is outside rope cache with {upper} rows")]
    PositionOutOfBounds { position: usize, upper: usize },
}

type DeepseekResult<T> = std::result::Result<T, DeepseekTrainError>;

#[derive(Debug, Clone)]
struct Linear {
    name: &'static str,
    weight: TensorId,
}

impl Linear {
    fn new(
        name: &'static str,
        in_features: usize,
        out_features: usize,
        store: &mut TensorStore,
    ) -> Result<Self> {
        let weight = normal_parameter(name, &[out_features, in_features], 0.02, true, store)?;
        Ok(Self { name, weight })
    }

    fn forward(&self, x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
        linear_forward(x, self.weight, store, tape)
    }
}

#[derive(Debug, Clone)]
struct DeepseekMla {
    q_proj: Linear,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: TensorId,
    kv_b_proj: Linear,
    o_proj: Linear,
}

impl DeepseekMla {
    fn forward(
        &self,
        x: TensorId,
        cfg: &DeepSeekConfig,
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
                expected: "3",
                got: x_shape.len(),
            });
        }
        let batch = x_shape[0];
        let seq_len = x_shape[1];
        let heads = cfg.num_attention_heads;

        let q = self.q_proj.forward(x, store, tape)?;
        let q = reshape(q, &[batch, seq_len, heads, cfg.qk_head_dim()], store, tape)?;
        let q_nope = slice(
            q,
            &[0, 0, 0, 0],
            &[batch, seq_len, heads, cfg.qk_nope_head_dim],
            store,
            tape,
        )?;
        let q_nope = transpose(q_nope, 1, 2, store, tape)?;

        let kv = self.kv_a_proj_with_mqa.forward(x, store, tape)?;
        let kv_latent = slice(
            kv,
            &[0, 0, 0],
            &[batch, seq_len, cfg.kv_lora_rank],
            store,
            tape,
        )?;
        let kv_latent = rmsnorm(
            kv_latent,
            self.kv_a_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let kv = self.kv_b_proj.forward(kv_latent, store, tape)?;
        let kv = reshape(
            kv,
            &[batch, seq_len, heads, cfg.qk_nope_head_dim + cfg.v_head_dim],
            store,
            tape,
        )?;
        let k_nope = slice(
            kv,
            &[0, 0, 0, 0],
            &[batch, seq_len, heads, cfg.qk_nope_head_dim],
            store,
            tape,
        )?;
        let v = slice(
            kv,
            &[0, 0, 0, cfg.qk_nope_head_dim],
            &[batch, seq_len, heads, cfg.qk_nope_head_dim + cfg.v_head_dim],
            store,
            tape,
        )?;
        let k_nope = transpose(k_nope, 1, 2, store, tape)?;
        let v = transpose(v, 1, 2, store, tape)?;
        let context = causal_sdpa(q_nope, k_nope, v, store, tape)?;
        let context = merge_heads(context, batch, seq_len, heads, cfg.v_head_dim, store, tape)?;
        self.o_proj.forward(context, store, tape)
    }
}

#[derive(Debug, Clone)]
struct DeepseekMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DeepseekMlp {
    fn forward(&self, x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
        let gate = self.gate_proj.forward(x, store, tape)?;
        let up = self.up_proj.forward(x, store, tape)?;
        let gate = silu(gate, store, tape)?;
        let hidden = mul(gate, up, store, tape)?;
        self.down_proj.forward(hidden, store, tape)
    }
}

#[derive(Debug, Clone)]
struct DeepseekLayer {
    input_layernorm: TensorId,
    self_attn: DeepseekMla,
    post_attention_layernorm: TensorId,
    mlp: DeepseekMlp,
}

impl DeepseekLayer {
    fn forward(
        &self,
        x: TensorId,
        cfg: &DeepSeekConfig,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let h = rmsnorm(x, self.input_layernorm, cfg.rms_norm_eps, store, tape)?;
        let attn = self.self_attn.forward(h, cfg, store, tape)?;
        let x = add(x, attn, store, tape)?;
        let h = rmsnorm(
            x,
            self.post_attention_layernorm,
            cfg.rms_norm_eps,
            store,
            tape,
        )?;
        let mlp = self.mlp.forward(h, store, tape)?;
        add(x, mlp, store, tape)
    }
}

#[derive(Debug, Clone)]
pub struct DeepseekNanoModel {
    config: DeepSeekConfig,
    layers: Vec<DeepseekLayer>,
    embed_tokens: TensorId,
    final_norm: TensorId,
    lm_head: TensorId,
    cos_cache: TensorId,
    sin_cache: TensorId,
    param_names: HashMap<&'static str, TensorId>,
    param_ids: Vec<TensorId>,
}

impl DeepseekNanoModel {
    pub fn new(cfg: &DeepSeekConfig, store: &mut TensorStore) -> DeepseekResult<Self> {
        validate_nano_train_config(cfg)?;

        let mut param_names = HashMap::new();
        let mut param_ids = Vec::new();
        let mut register = |map: &mut HashMap<&'static str, TensorId>, name: &'static str, id| {
            map.insert(name, id);
            if !param_ids.contains(&id) {
                param_ids.push(id);
            }
        };

        let embed_tokens = normal_parameter(
            cfg.embed_tokens_tensor_name(),
            &[cfg.vocab_size, cfg.hidden_size],
            0.02,
            true,
            store,
        )?;
        register(
            &mut param_names,
            cfg.embed_tokens_tensor_name(),
            embed_tokens,
        );
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens
        } else {
            normal_parameter(
                "lm_head.weight",
                &[cfg.vocab_size, cfg.hidden_size],
                0.02,
                true,
                store,
            )?
        };
        register(&mut param_names, "lm_head.weight", lm_head);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let names = cfg.layer_tensor_names(layer_idx);
            let input_layernorm_name = leak_name(names.input_layernorm);
            let post_attention_layernorm_name = leak_name(names.post_attention_layernorm);
            let post_attention_layernorm =
                ones_parameter(post_attention_layernorm_name, &[cfg.hidden_size], store)?;
            let q_proj_name = leak_name(names.attention.q_proj);
            let kv_a_proj_name = leak_name(names.attention.kv_a_proj_with_mqa);
            let kv_a_layernorm_name = leak_name(names.attention.kv_a_layernorm);
            let kv_b_proj_name = leak_name(names.attention.kv_b_proj);
            let o_proj_name = leak_name(names.attention.o_proj);

            let input_layernorm = ones_parameter(input_layernorm_name, &[cfg.hidden_size], store)?;
            let q_proj = Linear::new(q_proj_name, cfg.hidden_size, cfg.q_proj_dim(), store)?;
            let kv_a_proj_with_mqa = Linear::new(
                kv_a_proj_name,
                cfg.hidden_size,
                cfg.kv_lora_rank + cfg.qk_rope_head_dim,
                store,
            )?;
            let kv_a_layernorm = ones_parameter(kv_a_layernorm_name, &[cfg.kv_lora_rank], store)?;
            let kv_b_proj =
                Linear::new(kv_b_proj_name, cfg.kv_lora_rank, cfg.kv_b_proj_dim(), store)?;
            let o_proj = Linear::new(
                o_proj_name,
                cfg.num_attention_heads * cfg.v_head_dim,
                cfg.hidden_size,
                store,
            )?;
            register(&mut param_names, input_layernorm_name, input_layernorm);
            register(&mut param_names, q_proj.name, q_proj.weight);
            register(
                &mut param_names,
                kv_a_proj_with_mqa.name,
                kv_a_proj_with_mqa.weight,
            );
            register(&mut param_names, kv_a_layernorm_name, kv_a_layernorm);
            register(&mut param_names, kv_b_proj.name, kv_b_proj.weight);
            register(&mut param_names, o_proj.name, o_proj.weight);

            let DeepSeekMlpTensorNames::Dense(mlp_names) = names.mlp else {
                return Err(DeepseekTrainError::InvalidConfig(
                    "nano train model only supports dense MLP layers",
                ));
            };
            let gate_proj_name = leak_name(mlp_names.gate_proj);
            let up_proj_name = leak_name(mlp_names.up_proj);
            let down_proj_name = leak_name(mlp_names.down_proj);
            let gate_proj = Linear::new(
                gate_proj_name,
                cfg.hidden_size,
                cfg.intermediate_size,
                store,
            )?;
            let up_proj = Linear::new(up_proj_name, cfg.hidden_size, cfg.intermediate_size, store)?;
            let down_proj = Linear::new(
                down_proj_name,
                cfg.intermediate_size,
                cfg.hidden_size,
                store,
            )?;
            register(&mut param_names, gate_proj.name, gate_proj.weight);
            register(&mut param_names, up_proj.name, up_proj.weight);
            register(&mut param_names, down_proj.name, down_proj.weight);
            register(
                &mut param_names,
                post_attention_layernorm_name,
                post_attention_layernorm,
            );

            layers.push(DeepseekLayer {
                input_layernorm,
                self_attn: DeepseekMla {
                    q_proj,
                    kv_a_proj_with_mqa,
                    kv_a_layernorm,
                    kv_b_proj,
                    o_proj,
                },
                post_attention_layernorm,
                mlp: DeepseekMlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
            });
        }

        let final_norm = ones_parameter(cfg.norm_tensor_name(), &[cfg.hidden_size], store)?;
        register(&mut param_names, cfg.norm_tensor_name(), final_norm);
        let (cos_cache, sin_cache) = build_rope_cache(cfg, store)?;
        param_ids.push(cos_cache);
        param_ids.push(sin_cache);

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

    pub fn forward_batch_tokens_with_positions(
        &self,
        input_ids: &[usize],
        position_ids: &[usize],
        batch: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> DeepseekResult<TensorId> {
        let seq_len = position_ids.len();
        if input_ids.len() != batch * seq_len {
            return Err(DeepseekTrainError::InputLenMismatch {
                input_len: input_ids.len(),
                position_len: batch * seq_len,
            });
        }
        let _cos = select_cache_rows(self.cos_cache, position_ids, store)?;
        let _sin = select_cache_rows(self.sin_cache, position_ids, store)?;

        let mut hidden = embedding(self.embed_tokens, input_ids, store, tape)?;
        hidden = reshape(
            hidden,
            &[batch, seq_len, self.config.hidden_size],
            store,
            tape,
        )?;
        for layer in &self.layers {
            hidden = layer.forward(hidden, &self.config, store, tape)?;
        }
        let hidden = rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        Ok(linear_forward(hidden, self.lm_head, store, tape)?)
    }

    pub fn param_name_map(&self) -> HashMap<&'static str, TensorId> {
        self.param_names.clone()
    }

    pub fn all_parameter_ids(&self) -> Vec<TensorId> {
        self.param_ids.clone()
    }
}

impl GrpoPolicyConfig for DeepSeekConfig {
    fn max_seq_len(&self) -> usize {
        self.max_position_embeddings
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl GrpoPolicy for DeepseekNanoModel {
    type Config = DeepSeekConfig;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn forward_single(
        &self,
        input_ids: &[usize],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        self.forward_batch_tokens_with_positions(
            input_ids,
            &(0..input_ids.len()).collect::<Vec<_>>(),
            1,
            store,
            tape,
        )
        .map_err(deepseek_to_autograd)
    }

    fn forward_batch_tokens(
        &self,
        input_ids: &[usize],
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        self.forward_batch_tokens_with_positions(
            input_ids,
            &(0..seq_len).collect::<Vec<_>>(),
            batch,
            store,
            tape,
        )
        .map_err(deepseek_to_autograd)
    }

    fn forward_batch_tokens_with_positions(
        &self,
        input_ids: &[usize],
        position_ids: &[usize],
        batch: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        self.forward_batch_tokens_with_positions(input_ids, position_ids, batch, store, tape)
            .map_err(deepseek_to_autograd)
    }

    fn all_parameter_ids(&self) -> Vec<TensorId> {
        self.all_parameter_ids()
    }

    fn clone_frozen(&self, store: &mut TensorStore) -> Self {
        let cloned = Self::new(&self.config, store).expect("clone_frozen preserves config");
        copy_tensor_map(&self.param_names, &cloned.param_names, store);
        copy_tensor(self.cos_cache, cloned.cos_cache, store);
        copy_tensor(self.sin_cache, cloned.sin_cache, store);
        cloned
    }
}

impl CausalLm for DeepseekNanoModel {
    fn forward_with_positions(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> Result<TensorId> {
        let input_ids = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let position_ids = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_tokens_with_positions(&input_ids, &position_ids, 1, store, tape)
            .map_err(deepseek_to_autograd)
    }

    fn param_name_map(&self) -> HashMap<&'static str, TensorId> {
        self.param_name_map()
    }
}

fn validate_nano_train_config(cfg: &DeepSeekConfig) -> DeepseekResult<()> {
    cfg.validate()?;
    if cfg.is_moe() || cfg.has_mtp() || cfg.q_lora_rank.is_some() {
        return Err(DeepseekTrainError::InvalidConfig(
            "nano train model requires dense layers, no MTP, and direct q_proj",
        ));
    }
    if cfg.qk_nope_head_dim != cfg.v_head_dim {
        return Err(DeepseekTrainError::InvalidConfig(
            "nano train attention currently requires qk_nope_head_dim == v_head_dim",
        ));
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
    let input_dim = *x_shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if weight_shape.len() != 2 || input_dim != weight_shape[1] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![weight_shape.get(1).copied().unwrap_or(0)],
            got: vec![input_dim],
        });
    }
    let rows = x_shape.iter().product::<usize>() / input_dim;
    let flat = reshape(x, &[rows, input_dim], store, tape)?;
    let weight_t = transpose(weight, 0, 1, store, tape)?;
    let out = matmul(flat, weight_t, store, tape)?;
    let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
    out_shape.push(weight_shape[0]);
    reshape(out, &out_shape, store, tape)
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
    reshape(x, &[batch, seq_len, heads * head_dim], store, tape)
}

fn select_cache_rows(
    cache: TensorId,
    position_ids: &[usize],
    store: &mut TensorStore,
) -> DeepseekResult<TensorId> {
    let cache_tensor = store
        .get(cache)
        .ok_or(AutogradError::InvalidTensorId(cache))?;
    let rows = cache_tensor.shape[0];
    let cols = cache_tensor.shape[1];
    let mut data = Vec::with_capacity(position_ids.len() * cols);
    for &position in position_ids {
        if position >= rows {
            return Err(DeepseekTrainError::PositionOutOfBounds {
                position,
                upper: rows,
            });
        }
        let base = position * cols;
        data.extend_from_slice(&cache_tensor.data[base..base + cols]);
    }
    Ok(store.alloc(Tensor::new(data, vec![position_ids.len(), cols], false)?))
}

fn build_rope_cache(cfg: &DeepSeekConfig, store: &mut TensorStore) -> Result<(TensorId, TensorId)> {
    let half_dim = cfg.qk_rope_head_dim / 2;
    let inv_freq = (0..half_dim)
        .map(|index| {
            1.0 / cfg
                .rope_theta
                .powf((2.0 * index as f32) / cfg.qk_rope_head_dim as f32)
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
    Ok((
        store.alloc(Tensor::new(
            cos,
            vec![cfg.max_position_embeddings, half_dim],
            false,
        )?),
        store.alloc(Tensor::new(
            sin,
            vec![cfg.max_position_embeddings, half_dim],
            false,
        )?),
    ))
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
    store: &mut TensorStore,
) -> Result<TensorId> {
    let _ = name;
    Ok(store.alloc(Tensor::new(
        vec![1.0; shape.iter().product()],
        shape.to_vec(),
        true,
    )?))
}

fn copy_tensor(src: TensorId, dst: TensorId, store: &mut TensorStore) {
    let src_tensor = store.get(src).expect("source tensor exists").clone();
    let dst_tensor = store.get_mut(dst).expect("destination tensor exists");
    dst_tensor.data.clone_from(&src_tensor.data);
    dst_tensor.requires_grad = false;
}

fn copy_tensor_map(
    src: &HashMap<&'static str, TensorId>,
    dst: &HashMap<&'static str, TensorId>,
    store: &mut TensorStore,
) {
    for (name, &src_id) in src {
        if let Some(&dst_id) = dst.get(name) {
            copy_tensor(src_id, dst_id, store);
        }
    }
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

fn deepseek_to_autograd(err: DeepseekTrainError) -> AutogradError {
    match err {
        DeepseekTrainError::Autograd(err) => err,
        other => AutogradError::TapeInvariant(Box::leak(other.to_string().into_boxed_str())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::{Tape, TensorStore};

    #[test]
    fn nano_forward_shape_matches_vocab() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let cfg = DeepSeekConfig::nano();
        let model = DeepseekNanoModel::new(&cfg, &mut store).unwrap();
        let logits = model
            .forward_batch_tokens_with_positions(
                &[0, 1, 2, 3, 4, 5],
                &[0, 1, 2],
                2,
                &mut store,
                &mut tape,
            )
            .unwrap();
        assert_eq!(store.get(logits).unwrap().shape, [2, 3, cfg.vocab_size]);
    }
}
