use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::cpu_causal_sdpa_recompute_backward,
    ops::{add_broadcast, matmul, mul_scalar, reshape, softmax, softmax_backward, transpose},
    tape::{BackwardOp, GradPairs, SavedContext, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
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

    store.ensure_device(q)?;
    store.ensure_device(k)?;
    store.ensure_device(v)?;

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

    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    validate_attention_shapes(&q_shape, &k_shape, &v_shape)?;
    if upstream_shape != q_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.clone(),
            got: upstream_shape,
        });
    }

    let need_grad_q = store.tensor(q)?.requires_grad;
    let need_grad_k = store.tensor(k)?.requires_grad;
    let need_grad_v = store.tensor(v)?.requires_grad;
    let mut grads = GradPairs::new();
    if !need_grad_q && !need_grad_k && !need_grad_v {
        return Ok(grads);
    }

    let device_path_ok = {
        let q_tensor = store.tensor(q)?;
        let k_tensor = store.tensor(k)?;
        let v_tensor = store.tensor(v)?;
        let upstream = store.tensor(output_grad_id)?;
        q_tensor.dirty != Dirty::Host
            && q_tensor.device_handle.is_some()
            && k_tensor.dirty != Dirty::Host
            && k_tensor.device_handle.is_some()
            && v_tensor.dirty != Dirty::Host
            && v_tensor.device_handle.is_some()
            && upstream.dirty != Dirty::Host
            && upstream.device_handle.is_some()
    };
    if device_path_ok {
        return causal_sdpa_recompute_backward_device(
            q,
            k,
            v,
            output_grad_id,
            &q_shape,
            need_grad_q,
            need_grad_k,
            need_grad_v,
            store,
        );
    }

    let q_tensor = store.tensor_host(q)?;
    let k_tensor = store.tensor_host(k)?;
    let v_tensor = store.tensor_host(v)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let (grad_q, grad_k, grad_v) = cpu_causal_sdpa_recompute_backward(
        &q_tensor.data,
        &k_tensor.data,
        &v_tensor.data,
        &upstream.data,
        &q_shape,
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

#[allow(clippy::too_many_arguments)]
fn causal_sdpa_recompute_backward_device(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    upstream: TensorId,
    shape: &[usize],
    need_grad_q: bool,
    need_grad_k: bool,
    need_grad_v: bool,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let batch = shape[0];
    let heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];
    let merged_heads = batch * heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut tape = crate::Tape::new();
    tape.set_enabled(false);

    let q_3d = reshape(q, &[merged_heads, seq_len, head_dim], store, &mut tape)?;
    let k_3d = reshape(k, &[merged_heads, seq_len, head_dim], store, &mut tape)?;
    let v_3d = reshape(v, &[merged_heads, seq_len, head_dim], store, &mut tape)?;
    let upstream_3d = reshape(
        upstream,
        &[merged_heads, seq_len, head_dim],
        store,
        &mut tape,
    )?;

    let k_t = transpose(k_3d, 1, 2, store, &mut tape)?;
    let scores = matmul(q_3d, k_t, store, &mut tape)?;
    let scaled = mul_scalar(scores, scale, store, &mut tape)?;
    let mask = causal_mask(seq_len, store)?;
    let masked = add_broadcast(scaled, mask, store, &mut tape)?;
    let probs = softmax(masked, store, &mut tape)?;

    let mut grads = GradPairs::new();
    if need_grad_v {
        let probs_t = transpose(probs, 1, 2, store, &mut tape)?;
        let grad_v_3d = matmul(probs_t, upstream_3d, store, &mut tape)?;
        let grad_v = reshape(grad_v_3d, shape, store, &mut tape)?;
        grads.push((v, grad_v));
    }

    if need_grad_q || need_grad_k {
        let v_t = transpose(v_3d, 1, 2, store, &mut tape)?;
        let d_probs = matmul(upstream_3d, v_t, store, &mut tape)?;
        let softmax_entry = TapeEntry {
            op: BackwardOp::Softmax,
            output_id: probs,
            input_ids: smallvec![masked],
            saved: SavedContext::SoftmaxCtx { y: probs },
        };
        let d_scores_pairs = softmax_backward(&softmax_entry, d_probs, store)?;
        let d_scores = d_scores_pairs
            .iter()
            .find_map(|(input_id, grad_id)| (*input_id == masked).then_some(*grad_id))
            .ok_or(AutogradError::TapeInvariant(
                "causal_sdpa_recompute device backward missing softmax grad",
            ))?;

        if need_grad_q {
            let grad_q_3d = matmul(d_scores, k_3d, store, &mut tape)?;
            let grad_q_3d = mul_scalar(grad_q_3d, scale, store, &mut tape)?;
            let grad_q = reshape(grad_q_3d, shape, store, &mut tape)?;
            grads.push((q, grad_q));
        }
        if need_grad_k {
            let d_scores_t = transpose(d_scores, 1, 2, store, &mut tape)?;
            let grad_k_3d = matmul(d_scores_t, q_3d, store, &mut tape)?;
            let grad_k_3d = mul_scalar(grad_k_3d, scale, store, &mut tape)?;
            let grad_k = reshape(grad_k_3d, shape, store, &mut tape)?;
            grads.push((k, grad_k));
        }
    }

    Ok(grads)
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
