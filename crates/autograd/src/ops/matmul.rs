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
        (2, 2) => {
            let m = a_tensor.shape[0];
            let k = a_tensor.shape[1];
            let n = b_tensor.shape[1];

            if a_tensor.requires_grad {
                let mut grad_a = vec![0.0; a_tensor.size];
                for row in 0..m {
                    for inner in 0..k {
                        let mut acc = 0.0;
                        for col in 0..n {
                            acc +=
                                upstream.data[(row * n) + col] * b_tensor.data[(inner * n) + col];
                        }
                        grad_a[(row * k) + inner] = acc;
                    }
                }
                let grad_id = store.alloc(Tensor::new(grad_a, a_tensor.shape.clone(), false)?);
                grads.push((a, grad_id));
            }

            if b_tensor.requires_grad {
                let mut grad_b = vec![0.0; b_tensor.size];
                for inner in 0..k {
                    for col in 0..n {
                        let mut acc = 0.0;
                        for row in 0..m {
                            acc +=
                                a_tensor.data[(row * k) + inner] * upstream.data[(row * n) + col];
                        }
                        grad_b[(inner * n) + col] = acc;
                    }
                }
                let grad_id = store.alloc(Tensor::new(grad_b, b_tensor.shape.clone(), false)?);
                grads.push((b, grad_id));
            }
        }
        (3, 3) => {
            let batch = a_tensor.shape[0];
            let m = a_tensor.shape[1];
            let k = a_tensor.shape[2];
            let n = b_tensor.shape[2];
            let a_batch_stride = m * k;
            let b_batch_stride = k * n;
            let out_batch_stride = m * n;

            if a_tensor.requires_grad {
                let mut grad_a = vec![0.0; a_tensor.size];
                for batch_index in 0..batch {
                    let a_base = batch_index * a_batch_stride;
                    let b_base = batch_index * b_batch_stride;
                    let out_base = batch_index * out_batch_stride;
                    for row in 0..m {
                        for inner in 0..k {
                            let mut acc = 0.0;
                            for col in 0..n {
                                acc += upstream.data[out_base + (row * n) + col]
                                    * b_tensor.data[b_base + (inner * n) + col];
                            }
                            grad_a[a_base + (row * k) + inner] = acc;
                        }
                    }
                }
                let grad_id = store.alloc(Tensor::new(grad_a, a_tensor.shape.clone(), false)?);
                grads.push((a, grad_id));
            }

            if b_tensor.requires_grad {
                let mut grad_b = vec![0.0; b_tensor.size];
                for batch_index in 0..batch {
                    let a_base = batch_index * a_batch_stride;
                    let b_base = batch_index * b_batch_stride;
                    let out_base = batch_index * out_batch_stride;
                    for inner in 0..k {
                        for col in 0..n {
                            let mut acc = 0.0;
                            for row in 0..m {
                                acc += a_tensor.data[a_base + (row * k) + inner]
                                    * upstream.data[out_base + (row * n) + col];
                            }
                            grad_b[b_base + (inner * n) + col] = acc;
                        }
                    }
                }
                let grad_id = store.alloc(Tensor::new(grad_b, b_tensor.shape.clone(), false)?);
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

fn matmul_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    backend_matmul_output_shape(a_shape, b_shape)
}
