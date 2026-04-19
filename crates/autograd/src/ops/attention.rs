use crate::{
    AutogradError, Result,
    ops::{add_broadcast, matmul, mul_scalar, reshape, softmax, transpose},
    tensor::{GpuTensor, TensorId, TensorStore},
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
    let zeros = store.alloc(GpuTensor::new(
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

fn causal_mask(seq_len: usize, store: &mut TensorStore) -> Result<TensorId> {
    let mut data = vec![0.0; seq_len * seq_len];
    for row in 0..seq_len {
        for col in (row + 1)..seq_len {
            data[(row * seq_len) + col] = f32::NEG_INFINITY;
        }
    }
    Ok(store.alloc(GpuTensor::new(data, vec![1, seq_len, seq_len], false)?))
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
