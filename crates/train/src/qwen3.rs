use std::{
    collections::{HashMap, HashSet},
    f32::consts::TAU,
};

use autograd::{
    AutogradError, Tape, Tensor, TensorId, TensorStore,
    ops::{
        add, causal_sdpa, embedding, matmul, mul, repeat_kv, reshape, rmsnorm, rope, silu,
        transpose,
    },
};
pub use qwen3_spec::{Qwen3Config, Qwen3ConfigError};
use thiserror::Error;

use crate::{
    causal_lm::CausalLm,
    lora::{LinearWithLora, LoraConfig},
    policy::{GrpoPolicy, GrpoPolicyConfig},
};

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
    q_proj: LinearWithLora,
    k_proj: LinearWithLora,
    v_proj: LinearWithLora,
    o_proj: LinearWithLora,
    q_norm: TensorId,
    k_norm: TensorId,
}

#[derive(Debug, Clone)]
struct Qwen3Mlp {
    gate_proj: LinearWithLora,
    up_proj: LinearWithLora,
    down_proj: LinearWithLora,
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
        let q = self.self_attn.q_proj.forward(h, store, tape)?;
        let k = self.self_attn.k_proj.forward(h, store, tape)?;
        let v = self.self_attn.v_proj.forward(h, store, tape)?;

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
        let attn_out = self.self_attn.o_proj.forward(attn, store, tape)?;
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
}

#[derive(Debug, Clone)]
pub struct Qwen3Model {
    config: Qwen3Config,
    lora: Option<LoraConfig>,
    layers: Vec<Qwen3Layer>,
    embed_tokens: TensorId,
    final_norm: TensorId,
    lm_head: TensorId,
    cos_cache: TensorId,
    sin_cache: TensorId,
    param_names: HashMap<&'static str, TensorId>,
    adapter_names: HashMap<&'static str, TensorId>,
    param_ids: Vec<TensorId>,
}

impl Qwen3Model {
    pub fn new(cfg: &Qwen3Config, store: &mut TensorStore) -> Result<Self> {
        Self::new_with_lora(cfg, None, store)
    }

    pub fn new_with_lora(
        cfg: &Qwen3Config,
        lora: Option<LoraConfig>,
        store: &mut TensorStore,
    ) -> Result<Self> {
        cfg.validate()?;

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
        let base_requires_grad = lora.is_none();
        let embed_tokens_name = cfg.embed_tokens_tensor_name();
        let embed_tokens = normal_parameter(
            embed_tokens_name,
            &[cfg.vocab_size, cfg.hidden_size],
            0.02,
            base_requires_grad,
            store,
        )?;
        register_named(&mut param_names, embed_tokens_name, embed_tokens);

        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens
        } else {
            normal_parameter(
                "lm_head.weight",
                &[cfg.vocab_size, cfg.hidden_size],
                0.02,
                base_requires_grad,
                store,
            )?
        };
        register_named(&mut param_names, "lm_head.weight", lm_head);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let names = cfg.layer_tensor_names(layer_idx);
            let input_layernorm_name = leak_name(names.input_layernorm);
            let q_proj_name = leak_name(names.q_proj);
            let k_proj_name = leak_name(names.k_proj);
            let v_proj_name = leak_name(names.v_proj);
            let o_proj_name = leak_name(names.o_proj);
            let q_norm_name = leak_name(names.q_norm);
            let k_norm_name = leak_name(names.k_norm);
            let gate_proj_name = leak_name(names.mlp_gate_proj);
            let up_proj_name = leak_name(names.mlp_up_proj);
            let down_proj_name = leak_name(names.mlp_down_proj);
            let post_attention_layernorm_name = leak_name(names.post_attention_layernorm);

            let input_layernorm = ones_parameter(
                input_layernorm_name,
                &[cfg.hidden_size],
                base_requires_grad,
                store,
            )?;
            let q_proj = LinearWithLora::new(
                q_proj_name,
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                base_requires_grad,
                lora,
                store,
            )?;
            let k_proj = LinearWithLora::new(
                k_proj_name,
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                base_requires_grad,
                lora,
                store,
            )?;
            let v_proj = LinearWithLora::new(
                v_proj_name,
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                base_requires_grad,
                lora,
                store,
            )?;
            let o_proj = LinearWithLora::new(
                o_proj_name,
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                base_requires_grad,
                lora,
                store,
            )?;
            let q_norm = ones_parameter(q_norm_name, &[cfg.head_dim], base_requires_grad, store)?;
            let k_norm = ones_parameter(k_norm_name, &[cfg.head_dim], base_requires_grad, store)?;
            let gate_proj = LinearWithLora::new(
                gate_proj_name,
                cfg.hidden_size,
                cfg.intermediate_size,
                base_requires_grad,
                lora,
                store,
            )?;
            let up_proj = LinearWithLora::new(
                up_proj_name,
                cfg.hidden_size,
                cfg.intermediate_size,
                base_requires_grad,
                lora,
                store,
            )?;
            let down_proj = LinearWithLora::new(
                down_proj_name,
                cfg.intermediate_size,
                cfg.hidden_size,
                base_requires_grad,
                lora,
                store,
            )?;
            let post_attention_layernorm = ones_parameter(
                post_attention_layernorm_name,
                &[cfg.hidden_size],
                base_requires_grad,
                store,
            )?;

            register_named(&mut param_names, input_layernorm_name, input_layernorm);
            for (name, id) in q_proj.parameter_name_map() {
                register_named(&mut param_names, name, id);
            }
            for (name, id) in q_proj.adapter_name_map() {
                register_named(&mut adapter_names, name, id);
            }
            for (name, id) in k_proj.parameter_name_map() {
                register_named(&mut param_names, name, id);
            }
            for (name, id) in k_proj.adapter_name_map() {
                register_named(&mut adapter_names, name, id);
            }
            for (name, id) in v_proj.parameter_name_map() {
                register_named(&mut param_names, name, id);
            }
            for (name, id) in v_proj.adapter_name_map() {
                register_named(&mut adapter_names, name, id);
            }
            for (name, id) in o_proj.parameter_name_map() {
                register_named(&mut param_names, name, id);
            }
            for (name, id) in o_proj.adapter_name_map() {
                register_named(&mut adapter_names, name, id);
            }
            register_named(&mut param_names, q_norm_name, q_norm);
            register_named(&mut param_names, k_norm_name, k_norm);
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

    pub fn clone_frozen(&self, store: &mut TensorStore) -> Self {
        let cloned = Self::new_with_lora(&self.config, self.lora, store)
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
            let merged_q = {
                let tensor = layer.self_attn.q_proj.merged_tensor(store)?;
                store.alloc(tensor)
            };
            let merged_k = {
                let tensor = layer.self_attn.k_proj.merged_tensor(store)?;
                store.alloc(tensor)
            };
            let merged_v = {
                let tensor = layer.self_attn.v_proj.merged_tensor(store)?;
                store.alloc(tensor)
            };
            let merged_o = {
                let tensor = layer.self_attn.o_proj.merged_tensor(store)?;
                store.alloc(tensor)
            };
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

            for (name, _) in layer.self_attn.q_proj.parameter_name_map() {
                map.insert(name, merged_q);
            }
            for (name, _) in layer.self_attn.k_proj.parameter_name_map() {
                map.insert(name, merged_k);
            }
            for (name, _) in layer.self_attn.v_proj.parameter_name_map() {
                map.insert(name, merged_v);
            }
            for (name, _) in layer.self_attn.o_proj.parameter_name_map() {
                map.insert(name, merged_o);
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

impl CausalLm for Qwen3Model {
    fn forward_with_positions(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> autograd::Result<TensorId> {
        Qwen3Model::forward(self, store, tape, input_ids, position_ids).map_err(qwen3_to_autograd)
    }

    fn param_name_map(&self) -> HashMap<&'static str, TensorId> {
        Qwen3Model::param_name_map(self)
    }

    fn adapter_name_map(&self) -> HashMap<&'static str, TensorId> {
        Qwen3Model::adapter_name_map(self)
    }

    fn materialized_param_name_map(
        &self,
        store: &mut TensorStore,
        _tape: &mut Tape,
    ) -> autograd::Result<HashMap<&'static str, TensorId>> {
        Qwen3Model::materialized_param_name_map(self, store).map_err(qwen3_to_autograd)
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
