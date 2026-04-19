use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::matmul_output_shape as backend_matmul_output_shape,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn matmul(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(a)?;
    store.ensure_device(b)?;
    let a_handle = store
        .tensor(a)?
        .device_handle
        .as_ref()
        .expect("ensure_device")
        .clone();
    let b_handle = store
        .tensor(b)?
        .device_handle
        .as_ref()
        .expect("ensure_device")
        .clone();
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    let requires_grad = store.tensor(a)?.requires_grad || store.tensor(b)?.requires_grad;
    let (out_handle, out_shape) = store
        .backend()
        .matmul(&a_handle, &a_shape, &b_handle, &b_shape)?;
    let output_id = store.alloc_device_tensor(out_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Matmul,
            output_id,
            input_ids: smallvec![a, b],
            saved: SavedContext::MatmulCtx { a, b },
        });
    }

    Ok(output_id)
}

pub(crate) fn matmul_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::MatmulCtx { a, b } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "matmul backward missing saved context",
        ));
    };

    let upstream = store.tensor(output_grad_id)?.clone();
    let a_tensor = store.tensor(a)?.clone();
    let b_tensor = store.tensor(b)?.clone();
    let expected_shape = matmul_output_shape(&a_tensor.shape, &b_tensor.shape)?;
    if upstream.shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream.shape,
        });
    }

    let mut grads = GradPairs::new();
    match (a_tensor.shape.len(), b_tensor.shape.len()) {
        (2, 2) | (3, 3) => {
            if a_tensor.requires_grad {
                // grad_a = upstream @ b^T
                let (b_t, b_t_shape) = transpose_last_two(&b_tensor.data, &b_tensor.shape);
                let (grad_a, grad_a_shape) = store.backend().matmul_forward(
                    &upstream.data,
                    &upstream.shape,
                    &b_t,
                    &b_t_shape,
                )?;
                debug_assert_eq!(grad_a_shape, a_tensor.shape);
                let grad_id = store.alloc(Tensor::new(grad_a, grad_a_shape, false)?);
                grads.push((a, grad_id));
            }

            if b_tensor.requires_grad {
                // grad_b = a^T @ upstream
                let (a_t, a_t_shape) = transpose_last_two(&a_tensor.data, &a_tensor.shape);
                let (grad_b, grad_b_shape) = store.backend().matmul_forward(
                    &a_t,
                    &a_t_shape,
                    &upstream.data,
                    &upstream.shape,
                )?;
                debug_assert_eq!(grad_b_shape, b_tensor.shape);
                let grad_id = store.alloc(Tensor::new(grad_b, grad_b_shape, false)?);
                grads.push((b, grad_id));
            }
        }
        _ => {
            return Err(AutogradError::InvalidRank {
                expected: "both operands must be rank-2 or rank-3",
                got: a_tensor.shape.len().max(b_tensor.shape.len()),
            });
        }
    }

    Ok(grads)
}

/// Physically transpose the inner-most two axes of a rank-2 or rank-3 row-major
/// tensor. Returns `(data, shape)` for the transposed tensor. O(N) memory
/// movement — kept on CPU since it is not a FLOP-bound op.
fn transpose_last_two(data: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    match shape.len() {
        2 => {
            let rows = shape[0];
            let cols = shape[1];
            let mut out = vec![0.0f32; rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    out[col * rows + row] = data[row * cols + col];
                }
            }
            (out, vec![cols, rows])
        }
        3 => {
            let batch = shape[0];
            let rows = shape[1];
            let cols = shape[2];
            let plane = rows * cols;
            let mut out = vec![0.0f32; batch * plane];
            for batch_index in 0..batch {
                let base = batch_index * plane;
                for row in 0..rows {
                    for col in 0..cols {
                        out[base + col * rows + row] = data[base + row * cols + col];
                    }
                }
            }
            (out, vec![batch, cols, rows])
        }
        _ => {
            // Unreachable in practice: matmul_backward validates rank before
            // calling in. Fall through to an identity to keep this helper
            // total.
            (data.to_vec(), shape.to_vec())
        }
    }
}

fn matmul_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    backend_matmul_output_shape(a_shape, b_shape)
}
