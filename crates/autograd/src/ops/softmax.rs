use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let last_dim = last_dim(&input.shape)?;
    let rows = input.size / last_dim;
    let mut output = vec![0.0; input.size];

    for row in 0..rows {
        let base = row * last_dim;
        let slice = &input.data[base..base + last_dim];
        let max_value = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = slice
            .iter()
            .map(|value| (*value - max_value).exp())
            .sum::<f32>();
        for col in 0..last_dim {
            output[base + col] = (slice[col] - max_value).exp() / denom;
        }
    }

    let output_id = store.alloc(Tensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);
    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Softmax,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::SoftmaxCtx { y: output_id },
        });
    }

    Ok(output_id)
}

pub fn log_softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let last_dim = last_dim(&input.shape)?;
    let rows = input.size / last_dim;
    let mut output = vec![0.0; input.size];

    for row in 0..rows {
        let base = row * last_dim;
        let slice = &input.data[base..base + last_dim];
        let max_value = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = slice
            .iter()
            .map(|value| (*value - max_value).exp())
            .sum::<f32>();
        let log_denom = denom.ln();
        for col in 0..last_dim {
            output[base + col] = (slice[col] - max_value) - log_denom;
        }
    }

    let output_id = store.alloc(Tensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);
    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::LogSoftmax,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::LogSoftmaxCtx { y: output_id },
        });
    }

    Ok(output_id)
}

pub(crate) fn softmax_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("softmax missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::SoftmaxCtx { y } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "softmax backward missing saved output",
        ));
    };
    let output = store.tensor(y)?.clone();
    let upstream = store.tensor(output_grad_id)?.clone();
    if output.shape != upstream.shape {
        return Err(AutogradError::ShapeMismatch {
            expected: output.shape,
            got: upstream.shape,
        });
    }

    let last_dim = last_dim(&output.shape)?;
    let rows = output.size / last_dim;
    let mut grad = vec![0.0; output.size];
    for row in 0..rows {
        let base = row * last_dim;
        let mut dot = 0.0;
        for col in 0..last_dim {
            dot += upstream.data[base + col] * output.data[base + col];
        }
        for col in 0..last_dim {
            let y_value = output.data[base + col];
            grad[base + col] = y_value * (upstream.data[base + col] - dot);
        }
    }

    let grad_id = store.alloc(Tensor::new(grad, output.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

pub(crate) fn log_softmax_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("log_softmax missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::LogSoftmaxCtx { y } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "log_softmax backward missing saved output",
        ));
    };
    let output = store.tensor(y)?.clone();
    let upstream = store.tensor(output_grad_id)?.clone();
    if output.shape != upstream.shape {
        return Err(AutogradError::ShapeMismatch {
            expected: output.shape,
            got: upstream.shape,
        });
    }

    let last_dim = last_dim(&output.shape)?;
    let rows = output.size / last_dim;
    let mut grad = vec![0.0; output.size];
    for row in 0..rows {
        let base = row * last_dim;
        let mut sum_grad = 0.0;
        for col in 0..last_dim {
            sum_grad += upstream.data[base + col];
        }
        for col in 0..last_dim {
            grad[base + col] = upstream.data[base + col] - output.data[base + col].exp() * sum_grad;
        }
    }

    let grad_id = store.alloc(Tensor::new(grad, output.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

fn last_dim(shape: &[usize]) -> Result<usize> {
    shape.last().copied().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })
}
