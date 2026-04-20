use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn sum(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Pull only the metadata we need (shape + requires_grad) so the input's
    // device handle stays untouched: Metal keeps the matmul output as a
    // lazy MLX node, and `backend.sum_all` composes a `reshape -> sum_axis`
    // onto that node without forcing an `mlx_eval`. No `Tensor::clone` here
    // — that would assert against `Dirty::Device`.
    store.ensure_device(a)?;
    let (input_shape, requires_grad) = {
        let tensor = store.tensor(a)?;
        (tensor.shape.clone(), tensor.requires_grad)
    };
    let input_handle = store
        .tensor(a)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "sum: ensure_device left tensor without a device handle",
        ))?
        .clone();

    let out_handle = store.backend().sum_all(&input_handle, &input_shape)?;
    let output_id = store.alloc_device_tensor(Vec::new(), out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Sum,
            output_id,
            input_ids: smallvec![a],
            saved: SavedContext::Shape(input_shape),
        });
    }

    Ok(output_id)
}

pub fn mean(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor(a)?.clone();
    let value = input.data.iter().sum::<f32>() / input.size as f32;
    let output_id = store.alloc(Tensor::new(vec![value], Vec::new(), input.requires_grad)?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Mean,
            output_id,
            input_ids: smallvec![a],
            saved: SavedContext::MeanCtx {
                input: a,
                numel: input.size,
            },
        });
    }

    Ok(output_id)
}

pub(crate) fn sum_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let a = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("sum missing input"))?;
    if !store.tensor(a)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::Shape(shape) = &entry.saved else {
        return Err(AutogradError::TapeInvariant(
            "sum backward missing saved shape",
        ));
    };
    let output_grad = store.tensor(output_grad_id)?;
    if output_grad.shape != Vec::<usize>::new() || output_grad.data.len() != 1 {
        return Err(AutogradError::ShapeMismatch {
            expected: Vec::new(),
            got: output_grad.shape.clone(),
        });
    }

    let grad_value = output_grad.data[0];
    let size = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let grad_id = store.alloc(Tensor::new(vec![grad_value; size], shape.clone(), false)?);
    Ok(smallvec![(a, grad_id)])
}

pub(crate) fn mean_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let a = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("mean missing input"))?;
    if !store.tensor(a)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::MeanCtx { input, numel } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "mean backward missing saved context",
        ));
    };
    if input != a {
        return Err(AutogradError::TapeInvariant("mean backward input mismatch"));
    }

    let output_grad = store.tensor(output_grad_id)?;
    if output_grad.shape != Vec::<usize>::new() || output_grad.data.len() != 1 {
        return Err(AutogradError::ShapeMismatch {
            expected: Vec::new(),
            got: output_grad.shape.clone(),
        });
    }

    let input_shape = store.tensor(a)?.shape.clone();
    let grad_value = output_grad.data[0] / numel as f32;
    let grad_id = store.alloc(Tensor::new(vec![grad_value; numel], input_shape, false)?);
    Ok(smallvec![(a, grad_id)])
}
