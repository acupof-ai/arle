use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{GpuTensor, TensorId, TensorStore},
};

pub fn add(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let (a_data, a_shape, a_requires_grad) = {
        let tensor = store.tensor(a)?;
        (tensor.data.clone(), tensor.shape.clone(), tensor.requires_grad)
    };
    let (b_data, b_shape, b_requires_grad) = {
        let tensor = store.tensor(b)?;
        (tensor.data.clone(), tensor.shape.clone(), tensor.requires_grad)
    };
    if a_shape != b_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape,
            got: b_shape,
        });
    }

    let data = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect();
    let requires_grad = a_requires_grad || b_requires_grad;
    let output_id = store.alloc(GpuTensor::new(data, a_shape, requires_grad)?);

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Add,
            output_id,
            input_ids: smallvec![a, b],
            saved: SavedContext::None,
        });
    }

    Ok(output_id)
}

pub fn mul(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let (a_data, a_shape, a_requires_grad) = {
        let tensor = store.tensor(a)?;
        (tensor.data.clone(), tensor.shape.clone(), tensor.requires_grad)
    };
    let (b_data, b_shape, b_requires_grad) = {
        let tensor = store.tensor(b)?;
        (tensor.data.clone(), tensor.shape.clone(), tensor.requires_grad)
    };
    if a_shape != b_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape,
            got: b_shape,
        });
    }

    let data = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .collect();
    let requires_grad = a_requires_grad || b_requires_grad;
    let output_id = store.alloc(GpuTensor::new(data, a_shape, requires_grad)?);

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Mul,
            output_id,
            input_ids: smallvec![a, b],
            saved: SavedContext::Tensors(smallvec![a, b]),
        });
    }

    Ok(output_id)
}

pub fn mul_scalar(
    a: TensorId,
    k: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let (input_data, input_shape, requires_grad) = {
        let tensor = store.tensor(a)?;
        (tensor.data.clone(), tensor.shape.clone(), tensor.requires_grad)
    };

    let data = input_data.iter().map(|value| value * k).collect();
    let output_id = store.alloc(GpuTensor::new(data, input_shape, requires_grad)?);

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::MulScalar,
            output_id,
            input_ids: smallvec![a],
            saved: SavedContext::TensorAndScalar(a, k),
        });
    }

    Ok(output_id)
}

pub(crate) fn add_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let a = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("add missing lhs input"))?;
    let b = *entry
        .input_ids
        .get(1)
        .ok_or(AutogradError::TapeInvariant("add missing rhs input"))?;

    let mut grads = GradPairs::new();
    if store.tensor(a)?.requires_grad {
        grads.push((a, store.clone_tensor(output_grad_id)?));
    }
    if store.tensor(b)?.requires_grad {
        grads.push((b, store.clone_tensor(output_grad_id)?));
    }
    Ok(grads)
}

pub(crate) fn mul_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::Tensors(saved) = &entry.saved else {
        return Err(AutogradError::TapeInvariant("mul backward missing saved tensors"));
    };
    let a = *saved
        .first()
        .ok_or(AutogradError::TapeInvariant("mul missing lhs input"))?;
    let b = *saved
        .get(1)
        .ok_or(AutogradError::TapeInvariant("mul missing rhs input"))?;

    let upstream = store.to_host(output_grad_id)?;
    let a_tensor = store.tensor(a)?.clone();
    let b_tensor = store.tensor(b)?.clone();
    if a_tensor.shape != b_tensor.shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_tensor.shape,
            got: b_tensor.shape,
        });
    }

    let mut grads = GradPairs::new();
    if a_tensor.requires_grad {
        let grad_a = upstream
            .iter()
            .zip(b_tensor.data.iter())
            .map(|(grad, rhs)| grad * rhs)
            .collect();
        let grad_id = store.alloc(GpuTensor::new(grad_a, a_tensor.shape.clone(), false)?);
        grads.push((a, grad_id));
    }
    if b_tensor.requires_grad {
        let grad_b = upstream
            .iter()
            .zip(a_tensor.data.iter())
            .map(|(grad, lhs)| grad * lhs)
            .collect();
        let grad_id = store.alloc(GpuTensor::new(grad_b, b_tensor.shape.clone(), false)?);
        grads.push((b, grad_id));
    }

    Ok(grads)
}

pub(crate) fn mul_scalar_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::TensorAndScalar(a, k) = entry.saved else {
        return Err(AutogradError::TapeInvariant(
            "mul_scalar backward missing saved tensor/scalar",
        ));
    };

    if !store.tensor(a)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let upstream = store.to_host(output_grad_id)?;
    let input_shape = store.tensor(a)?.shape.clone();
    let grad = upstream.iter().map(|value| value * k).collect();
    let grad_id = store.alloc(GpuTensor::new(grad, input_shape, false)?);
    Ok(smallvec![(a, grad_id)])
}
