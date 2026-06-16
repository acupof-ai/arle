use smallvec::smallvec;

type AttentionGradTriplet = (Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>);

use crate::{
    AutogradError, Result,
    ops::{add_broadcast, matmul, mul_scalar, reshape, softmax, transpose},
    tape::{BackwardOp, GradPairs, SavedContext, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn repeat_kv(
    x: TensorId,
    n_rep: usize,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    if n_rep == 0 {
        return Err(AutogradError::InvalidIndicesLen {
            expected: 1,
            got: 0,
        });
    }
    if n_rep == 1 {
        return Ok(x);
    }

    let x_shape = store.tensor(x)?.shape.clone();
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }

    let expanded = vec![x_shape[0], x_shape[1], n_rep, x_shape[2], x_shape[3]];
    let reshaped = reshape(
        x,
        &[x_shape[0], x_shape[1], 1, x_shape[2], x_shape[3]],
        store,
        tape,
    )?;
    let zeros = store.alloc(Tensor::new(
        vec![0.0; expanded.iter().product()],
        expanded,
        false,
    )?);
    let repeated = add_broadcast(zeros, reshaped, store, tape)?;
    reshape(
        repeated,
        &[x_shape[0], x_shape[1] * n_rep, x_shape[2], x_shape[3]],
        store,
        tape,
    )
}

pub fn causal_sdpa(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();
    validate_attention_shapes(&q_shape, &k_shape, &v_shape)?;

    let batch = q_shape[0];
    let heads = q_shape[1];
    let seq_len = q_shape[2];
    let head_dim = q_shape[3];
    let merged_heads = batch * heads;

    let q_3d = reshape(q, &[merged_heads, seq_len, head_dim], store, tape)?;
    let k_3d = reshape(k, &[merged_heads, seq_len, head_dim], store, tape)?;
    let v_3d = reshape(v, &[merged_heads, seq_len, head_dim], store, tape)?;
    let k_t = transpose(k_3d, 1, 2, store, tape)?;
    let scores = matmul(q_3d, k_t, store, tape)?;
    let scaled = mul_scalar(scores, 1.0 / (head_dim as f32).sqrt(), store, tape)?;
    let mask = causal_mask(seq_len, store)?;
    let masked = add_broadcast(scaled, mask, store, tape)?;
    let probs = softmax(masked, store, tape)?;
    let context = matmul(probs, v_3d, store, tape)?;
    reshape(context, &[batch, heads, seq_len, head_dim], store, tape)
}

pub fn causal_sdpa_recompute(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();
    validate_attention_shapes(&q_shape, &k_shape, &v_shape)?;

    let requires_grad = store.tensor(q)?.requires_grad
        || store.tensor(k)?.requires_grad
        || store.tensor(v)?.requires_grad;

    let mut inner_tape = crate::Tape::new();
    inner_tape.set_enabled(false);
    let output_id = causal_sdpa(q, k, v, store, &mut inner_tape)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if tape.enabled && requires_grad {
        tape.record(crate::TapeEntry {
            op: BackwardOp::CausalSdpaRecompute,
            output_id,
            input_ids: smallvec![q, k, v],
            saved: SavedContext::CausalSdpaRecomputeCtx { q, k, v },
        });
    }

    Ok(output_id)
}

pub fn causal_sdpa_with_q_start(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_start: usize,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();
    validate_cached_attention_shapes(&q_shape, &k_shape, &v_shape, q_start)?;

    let batch = q_shape[0];
    let heads = q_shape[1];
    let q_len = q_shape[2];
    let kv_len = k_shape[2];
    let head_dim = q_shape[3];
    if q_start == 0 && q_len == kv_len {
        return causal_sdpa(q, k, v, store, tape);
    }

    let merged_heads = batch * heads;
    let q_3d = reshape(q, &[merged_heads, q_len, head_dim], store, tape)?;
    let k_3d = reshape(k, &[merged_heads, kv_len, head_dim], store, tape)?;
    let v_3d = reshape(v, &[merged_heads, kv_len, head_dim], store, tape)?;
    let k_t = transpose(k_3d, 1, 2, store, tape)?;
    let scores = matmul(q_3d, k_t, store, tape)?;
    let scaled = mul_scalar(scores, 1.0 / (head_dim as f32).sqrt(), store, tape)?;
    let masked = if q_len == 1 && q_start + 1 == kv_len {
        scaled
    } else {
        let mask = causal_mask_window(q_len, kv_len, q_start, store)?;
        add_broadcast(scaled, mask, store, tape)?
    };
    let probs = softmax(masked, store, tape)?;
    let context = matmul(probs, v_3d, store, tape)?;
    reshape(context, &[batch, heads, q_len, head_dim], store, tape)
}

pub fn causal_sdpa_decode_gqa(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_start: usize,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    if tape.enabled {
        return Err(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa is rollout-only and requires tape disabled",
        ));
    }

    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();

    store.ensure_device(q)?;
    store.ensure_device(k)?;
    store.ensure_device(v)?;
    let q_handle = store
        .tensor(q)?
        .device_handle
        .clone()
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa: q missing device handle",
        ))?;
    let k_handle = store
        .tensor(k)?
        .device_handle
        .clone()
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa: k missing device handle",
        ))?;
    let v_handle = store
        .tensor(v)?
        .device_handle
        .clone()
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa: v missing device handle",
        ))?;
    let (out_handle, out_shape) = store.backend().causal_sdpa_decode_gqa(
        &q_handle, &q_shape, &k_handle, &k_shape, &v_handle, &v_shape, q_start,
    )?;
    store.alloc_device_tensor(out_shape, out_handle)
}

pub(crate) fn causal_sdpa_recompute_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::CausalSdpaRecomputeCtx { q, k, v } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "causal_sdpa_recompute backward missing saved context",
        ));
    };

    let q_tensor = store.tensor_host(q)?;
    let k_tensor = store.tensor_host(k)?;
    let v_tensor = store.tensor_host(v)?;
    let upstream = store.tensor_host(output_grad_id)?;
    validate_attention_shapes(&q_tensor.shape, &k_tensor.shape, &v_tensor.shape)?;
    if upstream.shape != q_tensor.shape {
        return Err(AutogradError::ShapeMismatch {
            expected: q_tensor.shape.clone(),
            got: upstream.shape,
        });
    }

    let need_grad_q = store.tensor(q)?.requires_grad;
    let need_grad_k = store.tensor(k)?.requires_grad;
    let need_grad_v = store.tensor(v)?.requires_grad;
    let mut grads = GradPairs::new();
    if !need_grad_q && !need_grad_k && !need_grad_v {
        return Ok(grads);
    }

    let (grad_q, grad_k, grad_v) = causal_sdpa_backward_host(
        &q_tensor.data,
        &k_tensor.data,
        &v_tensor.data,
        &upstream.data,
        &q_tensor.shape,
        need_grad_q,
        need_grad_k,
        need_grad_v,
    )?;

    if let Some(data) = grad_q {
        grads.push((
            q,
            store.alloc(Tensor::new(data, q_tensor.shape.clone(), false)?),
        ));
    }
    if let Some(data) = grad_k {
        grads.push((
            k,
            store.alloc(Tensor::new(data, k_tensor.shape.clone(), false)?),
        ));
    }
    if let Some(data) = grad_v {
        grads.push((
            v,
            store.alloc(Tensor::new(data, v_tensor.shape.clone(), false)?),
        ));
    }

    Ok(grads)
}

fn causal_sdpa_backward_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    upstream: &[f32],
    shape: &[usize],
    need_grad_q: bool,
    need_grad_k: bool,
    need_grad_v: bool,
) -> Result<AttentionGradTriplet> {
    if shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: shape.len(),
        });
    }
    let batch = shape[0];
    let heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut grad_q = need_grad_q.then(|| vec![0.0; q.len()]);
    let mut grad_k = need_grad_k.then(|| vec![0.0; k.len()]);
    let mut grad_v = need_grad_v.then(|| vec![0.0; v.len()]);
    let mut scores = vec![0.0_f32; seq_len];
    let mut probs = vec![0.0_f32; seq_len];
    let mut d_probs = vec![0.0_f32; seq_len];

    for b in 0..batch {
        for h in 0..heads {
            for row in 0..seq_len {
                let mut max_score = f32::NEG_INFINITY;
                for col in 0..=row {
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q[offset4(b, h, row, d, heads, seq_len, head_dim)]
                            * k[offset4(b, h, col, d, heads, seq_len, head_dim)];
                    }
                    let score = dot * scale;
                    scores[col] = score;
                    max_score = max_score.max(score);
                }

                let mut denom = 0.0_f32;
                for col in 0..=row {
                    let p = (scores[col] - max_score).exp();
                    probs[col] = p;
                    denom += p;
                }
                for prob in probs.iter_mut().take(row + 1) {
                    *prob /= denom;
                }

                for col in 0..=row {
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += upstream[offset4(b, h, row, d, heads, seq_len, head_dim)]
                            * v[offset4(b, h, col, d, heads, seq_len, head_dim)];
                    }
                    d_probs[col] = dot;
                }

                let mut softmax_dot = 0.0_f32;
                for col in 0..=row {
                    softmax_dot += d_probs[col] * probs[col];
                }

                for col in 0..=row {
                    let d_score = probs[col] * (d_probs[col] - softmax_dot);
                    if let Some(grad_v) = grad_v.as_mut() {
                        for d in 0..head_dim {
                            grad_v[offset4(b, h, col, d, heads, seq_len, head_dim)] += probs[col]
                                * upstream[offset4(b, h, row, d, heads, seq_len, head_dim)];
                        }
                    }
                    if let Some(grad_q) = grad_q.as_mut() {
                        for d in 0..head_dim {
                            grad_q[offset4(b, h, row, d, heads, seq_len, head_dim)] += scale
                                * d_score
                                * k[offset4(b, h, col, d, heads, seq_len, head_dim)];
                        }
                    }
                    if let Some(grad_k) = grad_k.as_mut() {
                        for d in 0..head_dim {
                            grad_k[offset4(b, h, col, d, heads, seq_len, head_dim)] += scale
                                * d_score
                                * q[offset4(b, h, row, d, heads, seq_len, head_dim)];
                        }
                    }
                }
            }
        }
    }

    Ok((grad_q, grad_k, grad_v))
}

#[inline]
fn offset4(
    batch: usize,
    head: usize,
    token: usize,
    dim: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> usize {
    (((batch * heads + head) * seq_len + token) * head_dim) + dim
}

fn causal_mask(seq_len: usize, store: &mut TensorStore) -> Result<TensorId> {
    let mut data = vec![0.0; seq_len * seq_len];
    for row in 0..seq_len {
        for col in (row + 1)..seq_len {
            data[(row * seq_len) + col] = f32::NEG_INFINITY;
        }
    }
    Ok(store.alloc(Tensor::new(data, vec![1, seq_len, seq_len], false)?))
}

fn causal_mask_window(
    q_len: usize,
    kv_len: usize,
    q_start: usize,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let mut data = vec![0.0; q_len * kv_len];
    for row in 0..q_len {
        let max_visible = q_start + row;
        for col in (max_visible + 1)..kv_len {
            data[(row * kv_len) + col] = f32::NEG_INFINITY;
        }
    }
    Ok(store.alloc(Tensor::new(data, vec![1, q_len, kv_len], false)?))
}

fn validate_attention_shapes(
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
) -> Result<()> {
    for shape in [q_shape, k_shape, v_shape] {
        if shape.len() != 4 {
            return Err(AutogradError::InvalidRank {
                expected: "4",
                got: shape.len(),
            });
        }
    }

    if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[2] != k_shape[2] || q_shape[2] != v_shape[2] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }

    Ok(())
}

fn validate_cached_attention_shapes(
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
    q_start: usize,
) -> Result<()> {
    for shape in [q_shape, k_shape, v_shape] {
        if shape.len() != 4 {
            return Err(AutogradError::InvalidRank {
                expected: "4",
                got: shape.len(),
            });
        }
    }

    if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if k_shape[2] != v_shape[2] {
        return Err(AutogradError::ShapeMismatch {
            expected: k_shape.to_vec(),
            got: v_shape.to_vec(),
        });
    }
    if q_start + q_shape[2] > k_shape[2] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![q_start + q_shape[2]],
            got: vec![k_shape[2]],
        });
    }

    Ok(())
}
