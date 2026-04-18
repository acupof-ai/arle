use autograd::{
    AutogradError, GpuTensor, Result, Tape, TensorId, TensorStore,
    module::Module,
    ops::{
        add, add_broadcast, embedding, gelu, matmul, mul_scalar, reshape, rmsnorm, softmax,
        transpose,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TinyLMConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
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
        }
    }
}

#[derive(Debug, Clone)]
struct LinearNoBias {
    weight: TensorId,
}

impl LinearNoBias {
    fn new(
        in_features: usize,
        out_features: usize,
        salt: u32,
        store: &mut TensorStore,
    ) -> Result<Self> {
        let bound = 1.0 / (in_features as f32).sqrt();
        let mut state = 0x9E37_79B9_u32 ^ ((in_features as u32) << 16) ^ out_features as u32 ^ salt;
        let weight_data = (0..out_features * in_features)
            .map(|_| sample_uniform(&mut state, bound))
            .collect::<Vec<_>>();
        let weight = store.alloc(GpuTensor::new(
            weight_data,
            vec![out_features, in_features],
            true,
        )?);
        Ok(Self { weight })
    }

    fn forward(&self, x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
        linear_last_dim(x, self.weight, store, tape)
    }

    fn parameters(&self) -> Vec<TensorId> {
        vec![self.weight]
    }
}

#[derive(Debug, Clone)]
struct Block {
    attn_norm_weight: TensorId,
    wq: LinearNoBias,
    wk: LinearNoBias,
    wv: LinearNoBias,
    wo: LinearNoBias,
    ffn_norm_weight: TensorId,
    w1: LinearNoBias,
    w2: LinearNoBias,
}

impl Block {
    fn new(index: usize, config: TinyLMConfig, store: &mut TensorStore) -> Result<Self> {
        Ok(Self {
            attn_norm_weight: ones_parameter(&[config.d_model], store)?,
            wq: LinearNoBias::new(config.d_model, config.d_model, block_salt(index, 1), store)?,
            wk: LinearNoBias::new(config.d_model, config.d_model, block_salt(index, 2), store)?,
            wv: LinearNoBias::new(config.d_model, config.d_model, block_salt(index, 3), store)?,
            wo: LinearNoBias::new(config.d_model, config.d_model, block_salt(index, 4), store)?,
            ffn_norm_weight: ones_parameter(&[config.d_model], store)?,
            w1: LinearNoBias::new(config.d_model, config.d_ff, block_salt(index, 5), store)?,
            w2: LinearNoBias::new(config.d_ff, config.d_model, block_salt(index, 6), store)?,
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
        let mut params = vec![self.attn_norm_weight, self.ffn_norm_weight];
        params.extend(self.wq.parameters());
        params.extend(self.wk.parameters());
        params.extend(self.wv.parameters());
        params.extend(self.wo.parameters());
        params.extend(self.w1.parameters());
        params.extend(self.w2.parameters());
        params
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
        for index in 0..config.n_layers {
            blocks.push(Block::new(index, config, store)?);
        }

        Ok(Self {
            config,
            token_embed,
            pos_embed,
            blocks,
            final_norm_weight: ones_parameter(&[config.d_model], store)?,
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
        let mut params = vec![self.token_embed, self.pos_embed, self.final_norm_weight];
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

fn linear_last_dim(
    x: TensorId,
    weight: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let input_shape = store
        .get(x)
        .ok_or(AutogradError::InvalidTensorId(x))?
        .shape
        .clone();
    let in_features = *input_shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    let weight_shape = store
        .get(weight)
        .ok_or(AutogradError::InvalidTensorId(weight))?
        .shape
        .clone();
    if weight_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: weight_shape.len(),
        });
    }
    if weight_shape[1] != in_features {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![weight_shape[1]],
            got: vec![in_features],
        });
    }

    let prefix_elems = input_shape.iter().product::<usize>() / in_features;
    let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
    output_shape.push(weight_shape[0]);
    let flat_x = reshape(x, &[prefix_elems, in_features], store, tape)?;
    let weight_t = transpose(weight, 0, 1, store, tape)?;
    let flat_y = matmul(flat_x, weight_t, store, tape)?;
    reshape(flat_y, &output_shape, store, tape)
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

fn block_salt(index: usize, salt: u32) -> u32 {
    ((index as u32) << 8) ^ salt
}
