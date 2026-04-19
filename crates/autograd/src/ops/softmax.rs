use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let _ = last_dim(&input.shape)?;
    let output = store
        .backend()
        .softmax_forward_last_axis(&input.data, &input.shape)?;

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
    let _ = last_dim(&input.shape)?;
    let output = store
        .backend()
        .log_softmax_forward_last_axis(&input.data, &input.shape)?;

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

    // dL/dx = y * (dL/dy - sum(dL/dy * y, axis=-1, keepdim))
    // Stream row-wise so we only allocate the output buffer — full-vocab logits
    // on training paths make intermediate `mul`/`sub` materializations cost
    // 3-4× peak memory (codex review 2026-04-19).
    let last = last_dim(&output.shape)?;
    let rows = output.data.len() / last;
    let mut grad = vec![0.0_f32; output.data.len()];
    for row in 0..rows {
        let base = row * last;
        let mut dot = 0.0_f32;
        for col in 0..last {
            dot += upstream.data[base + col] * output.data[base + col];
        }
        for col in 0..last {
            grad[base + col] = output.data[base + col] * (upstream.data[base + col] - dot);
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

    // dL/dx = dL/dy - softmax(x) * sum(dL/dy, axis=-1, keepdim)
    // Saved output is log_softmax(x), so softmax(x) = exp(output). Stream
    // row-wise to avoid materializing `exp(output)` + `scaled` + `grad` at
    // full-vocab scale (codex review 2026-04-19).
    let last = last_dim(&output.shape)?;
    let rows = output.data.len() / last;
    let mut grad = vec![0.0_f32; output.data.len()];
    for row in 0..rows {
        let base = row * last;
        let mut sum_grad = 0.0_f32;
        for col in 0..last {
            sum_grad += upstream.data[base + col];
        }
        for col in 0..last {
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
