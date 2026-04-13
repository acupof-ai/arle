use anyhow::{Context, Result};

use super::kv_pool::MetalKVPool;
use super::mlx::MlxArray;
use super::weights::{StandardMetalLayerWeights, StandardMetalWeights};
use crate::sampler::SamplingParams;

/// Build one forward-pass compute graph (lazy — no GPU work until eval/async_eval).
///
/// Returns an unsynchronised token `MlxArray` (greedy argmax or categorical sample).
/// The caller must `async_eval` or `eval` then call `.item_i32()` to materialise the token.
// GPU required: all ops register Metal-backed lazy computation nodes.
#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_forward_graph(
    current_ids: &[u32],
    weights: &StandardMetalWeights,
    k_caches: &mut [MlxArray],
    v_caches: &mut [MlxArray],
    cache_len: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    eps: f32,
    mut metal_kv_pool: Option<&mut MetalKVPool>,
    request_id: usize,
    params: &SamplingParams,
) -> Result<MlxArray> {
    use super::mlx::{rms_norm, take_axis};

    let seq = current_ids.len() as i32;
    if let Some(pool) = metal_kv_pool.as_deref_mut() {
        pool.alloc_tokens(request_id, seq as usize)
            .context("alloc MetalKVPool slots")?;
    }

    // ── Embedding lookup ─────────────────────────────────────────────────────
    let idx_i32: Vec<i32> = current_ids.iter().map(|&t| t as i32).collect();
    let idx_arr = MlxArray::from_slice_i32(&idx_i32, &[seq]);
    let mut x = take_axis(&weights.embed_tokens, &idx_arr, 0);

    // ── Transformer layers ────────────────────────────────────────────────────
    for (li, layer) in weights.layers.iter().enumerate() {
        x = rust_transformer_layer(
            x,
            layer,
            li,
            k_caches,
            v_caches,
            seq,
            cache_len,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale,
            rope_base,
            eps,
            metal_kv_pool.as_deref_mut(),
            request_id,
        )?;
    }

    // ── Final norm + lm_head ─────────────────────────────────────────────────
    let last_idx = MlxArray::from_slice_i32(&[seq - 1], &[1]);
    let last_x = take_axis(&x, &last_idx, 0);
    let last_x = rms_norm(&last_x, &weights.norm, eps);
    let logits = super::ops::linear(&last_x, &weights.lm_head); // [1, vocab]

    // P4: GPU-side sampling — stays on GPU, only scalar crosses on .item() later.
    super::sampling::gpu_sample_token(&logits, params)
}

/// Single transformer layer for the maintained Rust/MLX Qwen3 path.
// GPU required
#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
pub(super) fn rust_transformer_layer(
    x: MlxArray,
    layer: &StandardMetalLayerWeights,
    li: usize,
    k_caches: &mut [MlxArray],
    v_caches: &mut [MlxArray],
    seq: i32,
    cache_len: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    eps: f32,
    mut metal_kv_pool: Option<&mut MetalKVPool>,
    request_id: usize,
) -> Result<MlxArray> {
    use super::mlx::{
        add, multiply, reshape, rms_norm, rope, scaled_dot_product_attention, silu, slice,
        slice_update, transpose_axes,
    };

    // 1. Input norm + residual
    let residual = x.clone();
    let x = rms_norm(&x, &layer.input_layernorm, eps);
    // 2. QKV projections
    let (q_raw, k_raw, v_raw) = layer.attention_inputs.project(&x);

    // 3+4+5+6. Reshape → per-head norm → transpose → RoPE.
    //
    // rope expects (B, *, T, D) where T is the second-to-last dim (sequence axis).
    // Transpose to [1, n_heads, seq, d] BEFORE rope so T = seq (correct positions).
    let q = reshape(&q_raw, &[1, seq, n_heads, head_dim]);
    let q = rms_norm(&q, &layer.q_norm, eps);
    let q = transpose_axes(&q, &[0, 2, 1, 3]); // [1, n_heads, seq, d]
    let q = rope(&q, head_dim, false, rope_base, 1.0f32, cache_len);

    let k = reshape(&k_raw, &[1, seq, n_kv_heads, head_dim]);
    let k = rms_norm(&k, &layer.k_norm, eps);
    let k = transpose_axes(&k, &[0, 2, 1, 3]); // [1, n_kv, seq, d]
    let k = rope(&k, head_dim, false, rope_base, 1.0f32, cache_len);

    let v = reshape(&v_raw, &[1, seq, n_kv_heads, head_dim]);
    let v = transpose_axes(&v, &[0, 2, 1, 3]); // [1, n_kv, seq, d]

    // 7. KV cache update
    let (k_full, v_full) = if let Some(pool) = metal_kv_pool.as_deref_mut() {
        let k_rows = transpose_axes(&k, &[0, 2, 1, 3]);
        let k_rows = reshape(&k_rows, &[seq, n_kv_heads * head_dim]);
        let v_rows = transpose_axes(&v, &[0, 2, 1, 3]);
        let v_rows = reshape(&v_rows, &[seq, n_kv_heads * head_dim]);
        pool.write_kv(li, request_id, &k_rows, &v_rows)
            .context("write MetalKVPool")?;
        pool.gather_kv(li, request_id)
            .context("gather MetalKVPool")?
    } else {
        let end_pos = cache_len + seq;
        k_caches[li] = slice_update(
            &mut k_caches[li],
            &k,
            &[0, 0, cache_len, 0],
            &[1, n_kv_heads, end_pos, head_dim],
        );
        v_caches[li] = slice_update(
            &mut v_caches[li],
            &v,
            &[0, 0, cache_len, 0],
            &[1, n_kv_heads, end_pos, head_dim],
        );
        // Read back the full KV sequence
        let k_full = slice(
            &k_caches[li],
            &[0, 0, 0, 0],
            &[1, n_kv_heads, end_pos, head_dim],
            &[1, 1, 1, 1],
        );
        let v_full = slice(
            &v_caches[li],
            &[0, 0, 0, 0],
            &[1, n_kv_heads, end_pos, head_dim],
            &[1, 1, 1, 1],
        );
        (k_full, v_full)
    };

    // 8. Attention
    let use_causal = cache_len == 0 && seq > 1;
    let mask_arg = if use_causal { Some("causal") } else { None };
    let attn_out = scaled_dot_product_attention(&q, &k_full, &v_full, attn_scale, mask_arg);

    // 9. Reshape + output proj + residual
    let attn_out = transpose_axes(&attn_out, &[0, 2, 1, 3]);
    let attn_out = reshape(&attn_out, &[seq, n_heads * head_dim]);
    let attn_out = super::ops::linear(&attn_out, &layer.o_proj);
    let x = add(&residual, &attn_out);

    // 10. MLP
    let residual2 = x.clone();
    let xn = rms_norm(&x, &layer.post_attention_layernorm, eps);
    let (gate_raw, up) = layer.mlp_inputs.project(&xn);
    let gate = silu(&gate_raw);
    let mlp = super::ops::linear(&multiply(&gate, &up), &layer.down_proj);
    Ok(add(&residual2, &mlp))
}
