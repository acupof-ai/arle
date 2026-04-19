use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn rope(
    x: TensorId,
    cos: TensorId,
    sin: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x_tensor = store.tensor(x)?.clone();
    let cos_tensor = store.tensor(cos)?.clone();
    let sin_tensor = store.tensor(sin)?.clone();
    validate_shapes(&x_tensor.shape, &cos_tensor.shape, &sin_tensor.shape)?;

    let batch = x_tensor.shape[0];
    let heads = x_tensor.shape[1];
    let seq_len = x_tensor.shape[2];
    let head_dim = x_tensor.shape[3];
    let half_dim = head_dim / 2;
    let mut output = vec![0.0; x_tensor.size];

    for batch_idx in 0..batch {
        for head_idx in 0..heads {
            for token_idx in 0..seq_len {
                let rope_base = token_idx * half_dim;
                let x_base = (((batch_idx * heads + head_idx) * seq_len) + token_idx) * head_dim;
                // NeoX / `rotate_half` layout: pair element `i` with element
                // `i + half_dim`. This matches Qwen3 HF weights + the infer-side
                // `precompute_rope` (infer/src/weight_loader.rs:450).
                for pair_idx in 0..half_dim {
                    let x0 = x_tensor.data[x_base + pair_idx];
                    let x1 = x_tensor.data[x_base + pair_idx + half_dim];
                    let cos_value = cos_tensor.data[rope_base + pair_idx];
                    let sin_value = sin_tensor.data[rope_base + pair_idx];
                    output[x_base + pair_idx] = (x0 * cos_value) - (x1 * sin_value);
                    output[x_base + pair_idx + half_dim] = (x1 * cos_value) + (x0 * sin_value);
                }
            }
        }
    }

    let output_id = store.alloc(Tensor::new(
        output,
        x_tensor.shape.clone(),
        x_tensor.requires_grad,
    )?);
    if x_tensor.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::RoPE,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::RoPECtx { cos, sin },
        });
    }

    Ok(output_id)
}

pub(crate) fn rope_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("rope missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::RoPECtx { cos, sin } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "rope backward missing saved context",
        ));
    };

    let x_shape = store.tensor(x)?.shape.clone();
    let cos_tensor = store.tensor(cos)?.clone();
    let sin_tensor = store.tensor(sin)?.clone();
    validate_shapes(&x_shape, &cos_tensor.shape, &sin_tensor.shape)?;

    let upstream = store.tensor(output_grad_id)?.clone();
    if upstream.shape != x_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: x_shape.clone(),
            got: upstream.shape,
        });
    }

    let batch = x_shape[0];
    let heads = x_shape[1];
    let seq_len = x_shape[2];
    let head_dim = x_shape[3];
    let half_dim = head_dim / 2;
    let mut grad_x = vec![0.0; upstream.size];

    for batch_idx in 0..batch {
        for head_idx in 0..heads {
            for token_idx in 0..seq_len {
                let rope_base = token_idx * half_dim;
                let grad_base = (((batch_idx * heads + head_idx) * seq_len) + token_idx) * head_dim;
                for pair_idx in 0..half_dim {
                    let grad0 = upstream.data[grad_base + pair_idx];
                    let grad1 = upstream.data[grad_base + pair_idx + half_dim];
                    let cos_value = cos_tensor.data[rope_base + pair_idx];
                    let sin_value = sin_tensor.data[rope_base + pair_idx];
                    grad_x[grad_base + pair_idx] = (grad0 * cos_value) + (grad1 * sin_value);
                    grad_x[grad_base + pair_idx + half_dim] =
                        (grad1 * cos_value) - (grad0 * sin_value);
                }
            }
        }
    }

    let grad_id = store.alloc(Tensor::new(grad_x, x_shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

fn validate_shapes(x_shape: &[usize], cos_shape: &[usize], sin_shape: &[usize]) -> Result<()> {
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }
    if cos_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: cos_shape.len(),
        });
    }
    if sin_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: sin_shape.len(),
        });
    }

    let head_dim = x_shape[3];
    if !head_dim.is_multiple_of(2) {
        return Err(AutogradError::InvalidRank {
            expected: "even head dim",
            got: head_dim,
        });
    }

    let expected_cache_shape = vec![x_shape[2], head_dim / 2];
    if cos_shape != expected_cache_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_cache_shape.clone(),
            got: cos_shape.to_vec(),
        });
    }
    if sin_shape != expected_cache_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_cache_shape,
            got: sin_shape.to_vec(),
        });
    }

    Ok(())
}
