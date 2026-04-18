use std::collections::HashSet;

use autograd::{
    AutogradError, GpuTensor, Result, Tape, TensorId, TensorStore,
    module::{Linear, Module},
    ops::{
        add, add_broadcast, embedding, gelu, matmul, mul_scalar, reshape, rmsnorm, softmax,
        transpose,
    },
};

use crate::lora::{LoraConfig, LoraLinear};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TinyLMConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub lora: Option<LoraConfig>,
}

impl Default for TinyLMConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            d_model: 384,
            n_layers: 6,
            n_heads: 6,
            d_head: 64,
            d_ff: 1024,
            max_seq_len: 128,
            lora: None,
        }
    }
}

#[derive(Debug, Clone)]
enum MaybeLora {
    Plain(Linear),
    Lora(LoraLinear),
}

impl MaybeLora {
    fn new(
        in_features: usize,
        out_features: usize,
        with_bias: bool,
        lora: Option<LoraConfig>,
        store: &mut TensorStore,
    ) -> Self {
        match lora {
            Some(cfg) => Self::Lora(LoraLinear::new(
                in_features,
                out_features,
                with_bias,
                &cfg,
                store,
            )),
            None => Self::Plain(Linear::new(in_features, out_features, with_bias, store)),
        }
    }

    fn forward(&self, x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
        match self {
            Self::Plain(linear) => linear.forward(x, store, tape),
            Self::Lora(linear) => linear.forward(x, store, tape),
        }
    }

    fn parameters(&self) -> Vec<TensorId> {
        match self {
            Self::Plain(linear) => linear.parameters(),
            Self::Lora(linear) => linear.trainable_parameters(),
        }
    }

    fn base_parameters(&self) -> Vec<TensorId> {
        match self {
            Self::Plain(linear) => linear.parameters(),
            Self::Lora(linear) => linear.base_parameters(),
        }
    }

    fn freeze_base(&self, store: &mut TensorStore) {
        match self {
            Self::Plain(linear) => linear.freeze(store),
            Self::Lora(linear) => linear.freeze_base(store),
        }
    }

    fn is_lora(&self) -> bool {
        matches!(self, Self::Lora(_))
    }
}

#[derive(Debug, Clone)]
struct Block {
    attn_norm_weight: TensorId,
    wq: MaybeLora,
    wk: MaybeLora,
    wv: MaybeLora,
    wo: MaybeLora,
    ffn_norm_weight: TensorId,
    w1: MaybeLora,
    w2: MaybeLora,
}

impl Block {
    fn new(config: TinyLMConfig, store: &mut TensorStore) -> Result<Self> {
        Ok(Self {
            attn_norm_weight: ones_parameter(&[config.d_model], store)?,
            wq: MaybeLora::new(config.d_model, config.d_model, false, config.lora, store),
            wk: MaybeLora::new(config.d_model, config.d_model, false, config.lora, store),
            wv: MaybeLora::new(config.d_model, config.d_model, false, config.lora, store),
            wo: MaybeLora::new(config.d_model, config.d_model, false, config.lora, store),
            ffn_norm_weight: ones_parameter(&[config.d_model], store)?,
            w1: MaybeLora::new(config.d_model, config.d_ff, false, config.lora, store),
            w2: MaybeLora::new(config.d_ff, config.d_model, false, config.lora, store),
        })
    }

    fn forward(
        &self,
        x: TensorId,
        batch: usize,
        seq_len: usize,
        config: TinyLMConfig,
        causal_mask: TensorId,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let h = rmsnorm(x, self.attn_norm_weight, 1e-6, store, tape)?;
        let q = self.wq.forward(h, store, tape)?;
        let k = self.wk.forward(h, store, tape)?;
        let v = self.wv.forward(h, store, tape)?;

        let q = split_heads(q, batch, seq_len, config, store, tape)?;
        let k = split_heads(k, batch, seq_len, config, store, tape)?;
        let v = split_heads(v, batch, seq_len, config, store, tape)?;
        let scores = attention_scores(q, k, batch, seq_len, config, store, tape)?;
        let scores = add_broadcast(scores, causal_mask, store, tape)?;
        let attn = softmax(scores, store, tape)?;
        let ctx = attention_context(attn, v, batch, seq_len, config, store, tape)?;
        let ctx = merge_heads(ctx, batch, seq_len, config, store, tape)?;
        let out = self.wo.forward(ctx, store, tape)?;
        let x = add(x, out, store, tape)?;

        let h = rmsnorm(x, self.ffn_norm_weight, 1e-6, store, tape)?;
        let ff = self.w1.forward(h, store, tape)?;
        let ff = gelu(ff, store, tape)?;
        let ff = self.w2.forward(ff, store, tape)?;
        add(x, ff, store, tape)
    }

    fn parameters(&self) -> Vec<TensorId> {
        let mut params = Vec::new();
        if !self.wq.is_lora() {
            params.push(self.attn_norm_weight);
            params.push(self.ffn_norm_weight);
        }
        params.extend(self.wq.parameters());
        params.extend(self.wk.parameters());
        params.extend(self.wv.parameters());
        params.extend(self.wo.parameters());
        params.extend(self.w1.parameters());
        params.extend(self.w2.parameters());
        params
    }

    fn base_parameters(&self) -> Vec<TensorId> {
        let mut params = vec![self.attn_norm_weight, self.ffn_norm_weight];
        params.extend(self.wq.base_parameters());
        params.extend(self.wk.base_parameters());
        params.extend(self.wv.base_parameters());
        params.extend(self.wo.base_parameters());
        params.extend(self.w1.base_parameters());
        params.extend(self.w2.base_parameters());
        params
    }

    fn freeze_base(&self, store: &mut TensorStore) {
        freeze_parameter(self.attn_norm_weight, store);
        freeze_parameter(self.ffn_norm_weight, store);
        self.wq.freeze_base(store);
        self.wk.freeze_base(store);
        self.wv.freeze_base(store);
        self.wo.freeze_base(store);
        self.w1.freeze_base(store);
        self.w2.freeze_base(store);
    }
}

#[derive(Debug, Clone)]
pub struct TinyLM {
    config: TinyLMConfig,
    token_embed: TensorId,
    pos_embed: TensorId,
    blocks: Vec<Block>,
    final_norm_weight: TensorId,
}

impl TinyLM {
    pub fn new(config: TinyLMConfig, store: &mut TensorStore) -> Result<Self> {
        validate_config(config)?;

        let token_embed = uniform_parameter(
            &[config.vocab_size, config.d_model],
            0.02,
            0x544F_4B45,
            store,
        )?;
        let pos_embed = uniform_parameter(
            &[config.max_seq_len, config.d_model],
            0.02,
            0x504F_5301,
            store,
        )?;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(Block::new(config, store)?);
        }

        let final_norm_weight = ones_parameter(&[config.d_model], store)?;

        if config.lora.is_some() {
            freeze_parameter(token_embed, store);
            freeze_parameter(pos_embed, store);
            freeze_parameter(final_norm_weight, store);
            for block in &blocks {
                block.freeze_base(store);
            }
        }

        Ok(Self {
            config,
            token_embed,
            pos_embed,
            blocks,
            final_norm_weight,
        })
    }

    pub fn config(&self) -> TinyLMConfig {
        self.config
    }

    pub fn parameter_count(&self, store: &TensorStore) -> usize {
        self.parameters()
            .iter()
            .map(|&id| store.get(id).map_or(0, |tensor| tensor.size))
            .sum()
    }

    pub fn base_parameter_ids(&self) -> Vec<TensorId> {
        let mut params = vec![self.token_embed, self.pos_embed, self.final_norm_weight];
        for block in &self.blocks {
            params.extend(block.base_parameters());
        }
        params
    }

    pub fn all_parameter_ids(&self) -> Vec<TensorId> {
        let mut params = Vec::new();
        let mut seen = HashSet::new();
        for id in self
            .base_parameter_ids()
            .into_iter()
            .chain(self.parameters())
        {
            if seen.insert(id) {
                params.push(id);
            }
        }
        params
    }

    pub fn clone_frozen(&self, store: &mut TensorStore) -> Self {
        let cloned = Self::new(self.config, store).expect("clone_frozen should preserve config");
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

    pub fn forward(
        &self,
        indices: &[usize],
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        if indices.len() != batch * seq_len {
            return Err(AutogradError::InvalidIndicesLen {
                expected: batch * seq_len,
                got: indices.len(),
            });
        }
        if seq_len > self.config.max_seq_len {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![self.config.max_seq_len],
                got: vec![seq_len],
            });
        }

        let tok = embedding_reshape(self.token_embed, indices, batch, seq_len, store, tape)?;
        let pos_indices = position_indices(batch, seq_len);
        let pos = embedding_reshape(self.pos_embed, &pos_indices, batch, seq_len, store, tape)?;
        let mut x = add(tok, pos, store, tape)?;
        let causal_mask = causal_mask(seq_len, store)?;

        for block in &self.blocks {
            x = block.forward(x, batch, seq_len, self.config, causal_mask, store, tape)?;
        }

        let x = rmsnorm(x, self.final_norm_weight, 1e-6, store, tape)?;
        let weight_t = transpose(self.token_embed, 0, 1, store, tape)?;
        let flat_x = reshape(x, &[batch * seq_len, self.config.d_model], store, tape)?;
        let logits = matmul(flat_x, weight_t, store, tape)?;
        reshape(
            logits,
            &[batch, seq_len, self.config.vocab_size],
            store,
            tape,
        )
    }
}

impl Module for TinyLM {
    fn parameters(&self) -> Vec<TensorId> {
        let mut params = Vec::new();
        if self.config.lora.is_none() {
            params.push(self.token_embed);
            params.push(self.pos_embed);
            params.push(self.final_norm_weight);
        }
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params
    }
}

fn validate_config(config: TinyLMConfig) -> Result<()> {
    let packed_heads = config.n_heads * config.d_head;
    if config.d_model != packed_heads {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![config.d_model],
            got: vec![packed_heads],
        });
    }
    if config.vocab_size == 0 || config.max_seq_len == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "positive vocab and sequence lengths",
            got: 0,
        });
    }
    if let Some(lora) = config.lora
        && lora.rank == 0
    {
        return Err(AutogradError::InvalidRank {
            expected: "positive LoRA rank",
            got: 0,
        });
    }
    Ok(())
}

fn embedding_reshape(
    table: TensorId,
    indices: &[usize],
    batch: usize,
    seq_len: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let hidden = store
        .get(table)
        .ok_or(AutogradError::InvalidTensorId(table))?
        .shape[1];
    let embedded = embedding(table, indices, store, tape)?;
    reshape(embedded, &[batch, seq_len, hidden], store, tape)
}

fn split_heads(
    x: TensorId,
    batch: usize,
    seq_len: usize,
    config: TinyLMConfig,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x = reshape(
        x,
        &[batch, seq_len, config.n_heads, config.d_head],
        store,
        tape,
    )?;
    transpose(x, 1, 2, store, tape)
}

fn merge_heads(
    x: TensorId,
    batch: usize,
    seq_len: usize,
    config: TinyLMConfig,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x = transpose(x, 1, 2, store, tape)?;
    reshape(x, &[batch, seq_len, config.d_model], store, tape)
}

fn attention_scores(
    q: TensorId,
    k: TensorId,
    batch: usize,
    seq_len: usize,
    config: TinyLMConfig,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let q = reshape(
        q,
        &[batch * config.n_heads, seq_len, config.d_head],
        store,
        tape,
    )?;
    let k = transpose(k, 2, 3, store, tape)?;
    let k = reshape(
        k,
        &[batch * config.n_heads, config.d_head, seq_len],
        store,
        tape,
    )?;
    let scores = matmul(q, k, store, tape)?;
    let scores = mul_scalar(scores, 1.0 / (config.d_head as f32).sqrt(), store, tape)?;
    reshape(
        scores,
        &[batch, config.n_heads, seq_len, seq_len],
        store,
        tape,
    )
}

fn attention_context(
    attn: TensorId,
    v: TensorId,
    batch: usize,
    seq_len: usize,
    config: TinyLMConfig,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let attn = reshape(
        attn,
        &[batch * config.n_heads, seq_len, seq_len],
        store,
        tape,
    )?;
    let v = reshape(
        v,
        &[batch * config.n_heads, seq_len, config.d_head],
        store,
        tape,
    )?;
    let ctx = matmul(attn, v, store, tape)?;
    reshape(
        ctx,
        &[batch, config.n_heads, seq_len, config.d_head],
        store,
        tape,
    )
}

fn causal_mask(seq_len: usize, store: &mut TensorStore) -> Result<TensorId> {
    let mut data = vec![0.0; seq_len * seq_len];
    for row in 0..seq_len {
        for col in row + 1..seq_len {
            data[(row * seq_len) + col] = -1.0e9;
        }
    }
    Ok(store.alloc(GpuTensor::new(data, vec![seq_len, seq_len], false)?))
}

fn position_indices(batch: usize, seq_len: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(batch * seq_len);
    for _ in 0..batch {
        indices.extend(0..seq_len);
    }
    indices
}

fn ones_parameter(shape: &[usize], store: &mut TensorStore) -> Result<TensorId> {
    let size = shape.iter().product();
    Ok(store.alloc(GpuTensor::new(vec![1.0; size], shape.to_vec(), true)?))
}

fn uniform_parameter(
    shape: &[usize],
    bound: f32,
    seed: u32,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let size = shape.iter().product();
    let mut state = seed;
    let data = (0..size)
        .map(|_| sample_uniform(&mut state, bound))
        .collect::<Vec<_>>();
    Ok(store.alloc(GpuTensor::new(data, shape.to_vec(), true)?))
}

fn sample_uniform(state: &mut u32, bound: f32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let unit = (*state >> 8) as f32 / (u32::MAX >> 8) as f32;
    ((unit * 2.0) - 1.0) * bound
}

fn freeze_parameter(id: TensorId, store: &mut TensorStore) {
    store
        .get_mut(id)
        .expect("parameter must exist while freezing")
        .requires_grad = false;
}
