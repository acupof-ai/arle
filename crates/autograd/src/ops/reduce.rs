use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{GpuTensor, TensorId, TensorStore},
};

pub fn sum(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor(a)?.clone();
    let value = input.data.iter().sum();
    let output_id = store.alloc(GpuTensor::new(vec![value], Vec::new(), input.requires_grad)?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Sum,
            output_id,
            input_ids: smallvec![a],
            saved: SavedContext::Shape(input.shape),
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
        return Err(AutogradError::TapeInvariant("sum backward missing saved shape"));
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
    let grad_id = store.alloc(GpuTensor::new(vec![grad_value; size], shape.clone(), false)?);
    Ok(smallvec![(a, grad_id)])
}
