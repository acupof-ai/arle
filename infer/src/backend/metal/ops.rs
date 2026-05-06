use anyhow::{Result, ensure};

use super::mlx::{
    Dtype, MlxArray, add, as_dtype, async_eval, clear_cache, concatenate_axis, eval,
    gguf_quantized_matmul, matmul, multiply, quantized_matmul, reshape, rms_norm, silu, take_axis,
    transpose_axes, zeros,
};
use super::weights::WeightTensor;
use crate::ops::{OpsBackend, OpsBackendKind};
use crate::sampler::SamplingParams;

#[cfg(feature = "metal")]
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub(super) struct MetalOpsBackend;

#[cfg(feature = "metal")]
#[allow(dead_code)]
impl MetalOpsBackend {
    pub(super) fn new() -> Self {
        Self
    }
}

#[cfg(feature = "metal")]
impl OpsBackend for MetalOpsBackend {
    type Tensor = MlxArray;
    type TensorBatch = MlxArray;
    type Matrix = WeightTensor;
    type Embedding = MlxArray;
    type TokenIds = MlxArray;
    type SamplingScratch = MlxArray;
    type SamplingOutput = MlxArray;

    fn backend_kind(&self) -> OpsBackendKind {
        OpsBackendKind::Metal
    }

    fn rms_norm_into(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::Tensor,
    ) -> Result<()> {
        *out = rms_norm(x, weight, eps);
        Ok(())
    }

    fn fused_add_rms_norm_into(
        &self,
        hidden: &mut Self::Tensor,
        residual: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::Tensor,
    ) -> Result<()> {
        *hidden = add(hidden, residual);
        *out = rms_norm(hidden, weight, eps);
        Ok(())
    }

    fn rms_norm_batch_into(
        &self,
        x: &Self::TensorBatch,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::TensorBatch,
    ) -> Result<()> {
        *out = rms_norm(x, weight, eps);
        Ok(())
    }

    fn fused_add_rms_norm_batch_into(
        &self,
        hidden: &mut Self::TensorBatch,
        residual: &Self::TensorBatch,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::TensorBatch,
    ) -> Result<()> {
        *hidden = add(hidden, residual);
        *out = rms_norm(hidden, weight, eps);
        Ok(())
    }

    fn linear_vec_into(
        &self,
        weight: &Self::Matrix,
        input: &Self::Tensor,
        output: &mut Self::Tensor,
    ) -> Result<()> {
        *output = linear(input, weight);
        Ok(())
    }

    fn linear_batch_into(
        &self,
        weight: &Self::Matrix,
        input: &Self::TensorBatch,
        output: &mut Self::TensorBatch,
    ) -> Result<()> {
        *output = linear(input, weight);
        Ok(())
    }

    fn fused_mlp_into(
        &self,
        input: &Self::Tensor,
        gate_proj: &Self::Matrix,
        up_proj: &Self::Matrix,
        down_proj: &Self::Matrix,
        act: &mut Self::Tensor,
        out: &mut Self::Tensor,
    ) -> Result<()> {
        let gate = linear(input, gate_proj);
        let up = linear(input, up_proj);
        *act = multiply(&silu(&gate), &up);
        *out = linear(act, down_proj);
        Ok(())
    }

    fn add_batch_into(
        &self,
        a: &Self::TensorBatch,
        b: &Self::TensorBatch,
        out: &mut Self::TensorBatch,
    ) -> Result<()> {
        *out = add(a, b);
        Ok(())
    }

    fn silu_mul_batch_into(
        &self,
        gate: &Self::TensorBatch,
        up: &Self::TensorBatch,
        out: &mut Self::TensorBatch,
    ) -> Result<()> {
        *out = multiply(&silu(gate), up);
        Ok(())
    }

    fn extract_vec_into(
        &self,
        batch: &Self::TensorBatch,
        token_idx: usize,
        out: &mut Self::Tensor,
    ) -> Result<()> {
        let token_idx = i32::try_from(token_idx)?;
        let idx = MlxArray::from_slice_i32(&[token_idx], &[1]);
        *out = take_axis(batch, &idx, 0);
        Ok(())
    }

    fn embedding_decode_into(
        &self,
        embed: &Self::Embedding,
        token_ids: &Self::TokenIds,
        out: &mut Self::Tensor,
    ) -> Result<()> {
        *out = take_axis(embed, token_ids, 0);
        Ok(())
    }

    fn embedding_batch_into(
        &self,
        embed: &Self::Embedding,
        token_ids: &Self::TokenIds,
        out: &mut Self::TensorBatch,
    ) -> Result<()> {
        *out = take_axis(embed, token_ids, 0);
        Ok(())
    }

    fn sample_token_into(
        &self,
        logits: &Self::Tensor,
        scratch: &mut Self::SamplingScratch,
        out: &mut Self::SamplingOutput,
        params: &SamplingParams,
        random_val: f32,
    ) -> Result<u32> {
        let _ = random_val;
        super::sampling::validate_metal_sampling_params(params)?;
        *scratch = MlxArray::scalar_f32(0.0);
        *out = super::sampling::gpu_sample_token(logits, params);
        eval(&[&*out]);
        Ok(out.item_i32() as u32)
    }

    fn argmax_with_logprob(
        &self,
        logits: &Self::Tensor,
        out_idx: &mut Self::SamplingOutput,
        out_logprob: &mut Self::SamplingScratch,
    ) -> Result<(u32, f32)> {
        let (tokens, logprobs) = greedy_tokens_with_logprobs(logits, 1)?;
        *out_idx = MlxArray::from_slice_i32(&tokens, &[1]);
        *out_logprob = MlxArray::from_slice_f32(&logprobs, &[1]);
        Ok((tokens[0] as u32, logprobs[0]))
    }

    fn argmax_batch_logprob_launch(
        &self,
        logits: &Self::TensorBatch,
        out_ids: &mut Self::SamplingOutput,
        out_logprobs: &mut Self::SamplingScratch,
        batch_size: usize,
    ) -> Result<()> {
        let (tokens, logprobs) = greedy_tokens_with_logprobs(logits, batch_size)?;
        let batch_size_i32 = i32::try_from(batch_size)?;
        *out_ids = MlxArray::from_slice_i32(&tokens, &[batch_size_i32]);
        *out_logprobs = MlxArray::from_slice_f32(&logprobs, &[batch_size_i32]);
        async_eval(&[&*out_ids, &*out_logprobs]);
        Ok(())
    }

    fn argmax_batch_readback_into(
        &self,
        out: &Self::SamplingOutput,
        dst: &mut [i32],
        batch_size: usize,
    ) -> Result<()> {
        ensure!(
            dst.len() >= batch_size,
            "Metal argmax readback dst too small: {} < {}",
            dst.len(),
            batch_size
        );
        eval(&[out]);
        let values = out.as_slice_i32();
        ensure!(
            values.len() >= batch_size,
            "Metal argmax output too small: {} < {}",
            values.len(),
            batch_size
        );
        dst[..batch_size].copy_from_slice(&values[..batch_size]);
        Ok(())
    }
}

#[cfg(feature = "metal")]
pub(super) fn metal_async_eval(arr: &MlxArray) {
    async_eval(&[arr]);
}

#[cfg(feature = "metal")]
pub(super) fn clear_metal_cache() {
    clear_cache();
}

#[cfg(feature = "metal")]
pub(super) fn extend_kv_cache(cache: &mut MlxArray, n_kv_heads: i32, head_dim: i32, new_cap: i32) {
    let current_cap = cache.shape().get(2).copied().unwrap_or_default();
    if new_cap <= current_cap {
        return;
    }

    // Inherit the batch dim from the existing cache — in the packed-decode
    // path this cache holds multiple rows stacked along axis 0.
    let batch = cache.shape().first().copied().unwrap_or(1);
    let extra = zeros(
        &[batch, n_kv_heads, new_cap - current_cap, head_dim],
        cache.dtype(),
    );
    *cache = concatenate_axis(&[cache.clone(), extra], 2);
}

/// `x @ weight.T` — no bias, dispatches to dense matmul or quantized matmul.
///
/// For `Dense(w_t)`, `w_t` is already transposed at load time (shape `[in, out]`),
/// so this is just `matmul(x, w_t)` without an extra transpose.
#[cfg(feature = "metal")]
#[inline]
pub(super) fn linear(x: &MlxArray, weight: &WeightTensor) -> MlxArray {
    match weight {
        WeightTensor::Dense(w_t) => {
            // w_t is pre-transposed [in, out]; direct matmul, no per-call transpose.
            matmul(x, w_t)
        }
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => {
            // w stored as [out, in] packed uint32; transpose=true → x @ w.T
            quantized_matmul(x, w, scales, biases, true, *group_size, *bits)
        }
        WeightTensor::GgufPacked {
            w,
            format,
            rows,
            cols,
        } => gguf_quantized_matmul(x, w, format.as_i32(), *rows, *cols),
        WeightTensor::GgufPackedInputReordered {
            w,
            format,
            rows,
            cols,
            num_key_heads,
            num_value_heads_per_key,
            head_dim,
        } => {
            let x_reordered =
                reorder_qwen35_v_cols_input(x, *num_key_heads, *num_value_heads_per_key, *head_dim);
            gguf_quantized_matmul(&x_reordered, w, format.as_i32(), *rows, *cols)
        }
    }
}

#[cfg(feature = "metal")]
fn reorder_qwen35_v_cols_input(
    x: &MlxArray,
    num_key_heads: i32,
    num_value_heads_per_key: i32,
    head_dim: i32,
) -> MlxArray {
    if num_value_heads_per_key <= 1 {
        return x.clone();
    }

    let shape = x.shape();
    let Some(&cols) = shape.last() else {
        return x.clone();
    };
    assert_eq!(
        cols,
        num_key_heads * num_value_heads_per_key * head_dim,
        "Qwen3.5 GGUF value-head input reorder dimension mismatch"
    );

    let prefix_ndim = shape.len() - 1;
    let mut expanded = shape[..prefix_ndim].to_vec();
    expanded.extend([num_key_heads, num_value_heads_per_key, head_dim]);

    let mut axes: Vec<i32> = (0..i32::try_from(prefix_ndim).expect("ndim fits i32")).collect();
    let base = i32::try_from(prefix_ndim).expect("ndim fits i32");
    axes.extend([base + 1, base, base + 2]);

    let expanded_x = reshape(x, &expanded);
    reshape(&transpose_axes(&expanded_x, &axes), shape)
}

#[cfg(feature = "metal")]
fn greedy_tokens_with_logprobs(
    logits: &MlxArray,
    batch_size: usize,
) -> Result<(Vec<i32>, Vec<f32>)> {
    let logits_f32 = as_dtype(logits, Dtype::Float32);
    eval(&[&logits_f32]);
    let shape = logits_f32.shape();
    let vocab = *shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("Metal argmax logits must have at least one dimension"))?;
    ensure!(vocab > 0, "Metal argmax logits vocab dimension is empty");
    let vocab = usize::try_from(vocab)?;
    let values = logits_f32.as_slice_f32();
    ensure!(
        values.len() >= batch_size * vocab,
        "Metal argmax logits too small: {} values for batch_size={} vocab={}",
        values.len(),
        batch_size,
        vocab
    );

    let mut tokens = Vec::with_capacity(batch_size);
    let mut logprobs = Vec::with_capacity(batch_size);
    for row in 0..batch_size {
        let start = row * vocab;
        let row_values = &values[start..start + vocab];
        let (argmax_idx, max_val) = row_values.iter().copied().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |best, (idx, value)| {
                if value > best.1 { (idx, value) } else { best }
            },
        );
        let exp_sum: f32 = row_values
            .iter()
            .map(|value| (*value - max_val).exp())
            .sum();
        tokens.push(i32::try_from(argmax_idx)?);
        logprobs.push(-exp_sum.ln());
    }

    Ok((tokens, logprobs))
}
