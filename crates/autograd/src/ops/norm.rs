use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn rmsnorm(
    x: TensorId,
    weight: TensorId,
    eps: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x_tensor = store.tensor(x)?.clone();
    let weight_tensor = store.tensor(weight)?.clone();
    let hidden = *x_tensor.shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if weight_tensor.shape != vec![hidden] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![hidden],
            got: weight_tensor.shape,
        });
    }

    let rows = x_tensor.size / hidden;
    let mut output = vec![0.0; x_tensor.size];
    let mut inv_rms = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * hidden;
        let mut sum_sq = 0.0;
        for col in 0..hidden {
            let value = x_tensor.data[base + col];
            sum_sq += value * value;
        }
        let inv = 1.0 / ((sum_sq / hidden as f32) + eps).sqrt();
        inv_rms.push(inv);
        for col in 0..hidden {
            output[base + col] = x_tensor.data[base + col] * inv * weight_tensor.data[col];
        }
    }

    let requires_grad = x_tensor.requires_grad || weight_tensor.requires_grad;
    let output_id = store.alloc(Tensor::new(output, x_tensor.shape.clone(), requires_grad)?);
    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::RMSNorm,
            output_id,
            input_ids: smallvec![x, weight],
            saved: SavedContext::RMSNormCtx { x, weight, inv_rms },
        });
    }

    Ok(output_id)
}

pub(crate) fn rmsnorm_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::RMSNormCtx { x, weight, inv_rms } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "rmsnorm backward missing saved context",
        ));
    };

    let upstream = store.tensor(output_grad_id)?.clone();
    let x_tensor = store.tensor(x)?.clone();
    let weight_tensor = store.tensor(weight)?.clone();
    if upstream.shape != x_tensor.shape {
        return Err(AutogradError::ShapeMismatch {
            expected: x_tensor.shape.clone(),
            got: upstream.shape,
        });
    }

    let hidden = *x_tensor.shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    let rows = x_tensor.size / hidden;
    if inv_rms.len() != rows {
        return Err(AutogradError::TapeInvariant(
            "rmsnorm inverse-rms rows mismatch",
        ));
    }

    let mut grads = GradPairs::new();
    if x_tensor.requires_grad {
        let mut grad_x = vec![0.0; x_tensor.size];
        for (row, &inv) in inv_rms.iter().enumerate() {
            let base = row * hidden;
            let mut dot = 0.0;
            for col in 0..hidden {
                dot +=
                    upstream.data[base + col] * weight_tensor.data[col] * x_tensor.data[base + col];
            }
            let correction = inv * inv * dot / hidden as f32;
            for col in 0..hidden {
                let scaled_grad = upstream.data[base + col] * weight_tensor.data[col];
                grad_x[base + col] =
                    (inv * scaled_grad) - (x_tensor.data[base + col] * inv * correction);
            }
        }
        let grad_id = store.alloc(Tensor::new(grad_x, x_tensor.shape.clone(), false)?);
        grads.push((x, grad_id));
    }

    if weight_tensor.requires_grad {
        let mut grad_weight = vec![0.0; hidden];
        for (row, &inv) in inv_rms.iter().enumerate() {
            let base = row * hidden;
            for (col, grad_slot) in grad_weight.iter_mut().enumerate() {
                *grad_slot += upstream.data[base + col] * x_tensor.data[base + col] * inv;
            }
        }
        let grad_id = store.alloc(Tensor::new(grad_weight, weight_tensor.shape, false)?);
        grads.push((weight, grad_id));
    }

    Ok(grads)
}
