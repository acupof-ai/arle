use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn add(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    if a_shape != b_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape,
            got: b_shape,
        });
    }
    let requires_grad = store.tensor(a)?.requires_grad || store.tensor(b)?.requires_grad;

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

    let out_handle = store.backend().add(&a_handle, &b_handle, &a_shape)?;
    let output_id = store.alloc_device_tensor(a_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

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

pub fn mul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.17: dispatch is OR-lazy — if EITHER operand is device-resident,
    // upload the other and stay on the MLX graph. Same rationale as
    // `add_broadcast`: the hot path is `attn * gate` and `silu(gate) * up`
    // in Qwen3.5, where both operands are Dirty::Device chained from prior
    // matmul/sigmoid/silu nodes. Forcing a readback on either side would
    // flush the whole upstream graph.
    let a_use_lazy = {
        let t = store.tensor(a)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    let b_use_lazy = {
        let t = store.tensor(b)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    if a_use_lazy || b_use_lazy {
        mul_device_lazy(a, b, store, tape)
    } else {
        mul_host_eager(a, b, store, tape)
    }
}

fn mul_device_lazy(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let (a_shape, a_requires_grad) = {
        let t = store.tensor(a)?;
        (t.shape.clone(), t.requires_grad)
    };
    let (b_shape, b_requires_grad) = {
        let t = store.tensor(b)?;
        (t.shape.clone(), t.requires_grad)
    };
    if a_shape != b_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape,
            got: b_shape,
        });
    }

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

    let out_handle = store.backend().mul(&a_handle, &b_handle, &a_shape)?;
    let requires_grad = a_requires_grad || b_requires_grad;
    let output_id = store.alloc_device_tensor(a_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

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

fn mul_host_eager(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Mirror `add_broadcast_host_eager`: even on CPU backend, operands may
    // be Dirty::Device-on-CPU-handle, so ensure_host synchronizes before
    // `.clone()` (which asserts `dirty != Device`).
    store.ensure_host(a)?;
    store.ensure_host(b)?;
    let (a_data, a_shape, a_requires_grad) = {
        let tensor = store.tensor(a)?;
        (
            tensor.data.clone(),
            tensor.shape.clone(),
            tensor.requires_grad,
        )
    };
    let (b_data, b_shape, b_requires_grad) = {
        let tensor = store.tensor(b)?;
        (
            tensor.data.clone(),
            tensor.shape.clone(),
            tensor.requires_grad,
        )
    };
    if a_shape != b_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape,
            got: b_shape,
        });
    }

    let data = store.backend().mul_forward(&a_data, &b_data)?;
    let requires_grad = a_requires_grad || b_requires_grad;
    let output_id = store.alloc(Tensor::new(data, a_shape, requires_grad)?);

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
    let tensor = store.tensor(a)?;
    let use_lazy = tensor.device_handle.is_some() && tensor.dirty != Dirty::Host;
    if use_lazy {
        mul_scalar_device_lazy(a, k, store, tape)
    } else {
        mul_scalar_host_eager(a, k, store, tape)
    }
}

fn mul_scalar_device_lazy(
    a: TensorId,
    k: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let (input_shape, requires_grad) = {
        let tensor = store.tensor(a)?;
        (tensor.shape.clone(), tensor.requires_grad)
    };

    store.ensure_device(a)?;
    let a_handle = store
        .tensor(a)?
        .device_handle
        .as_ref()
        .expect("ensure_device")
        .clone();

    let out_handle = store.backend().mul_scalar(&a_handle, k, &input_shape)?;
    let output_id = store.alloc_device_tensor(input_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

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

fn mul_scalar_host_eager(
    a: TensorId,
    k: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let (input_data, input_shape, requires_grad) = {
        let tensor = store.tensor(a)?;
        (
            tensor.data.clone(),
            tensor.shape.clone(),
            tensor.requires_grad,
        )
    };

    let data = store.backend().mul_scalar_forward(&input_data, k)?;
    let output_id = store.alloc(Tensor::new(data, input_shape, requires_grad)?);

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
        return Err(AutogradError::TapeInvariant(
            "mul backward missing saved tensors",
        ));
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
        let grad_a = store.backend().mul_forward(&upstream, &b_tensor.data)?;
        let grad_id = store.alloc(Tensor::new(grad_a, a_tensor.shape.clone(), false)?);
        grads.push((a, grad_id));
    }
    if b_tensor.requires_grad {
        let grad_b = store.backend().mul_forward(&upstream, &a_tensor.data)?;
        let grad_id = store.alloc(Tensor::new(grad_b, b_tensor.shape.clone(), false)?);
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
    let grad = store.backend().mul_scalar_forward(&upstream, k)?;
    let grad_id = store.alloc(Tensor::new(grad, input_shape, false)?);
    Ok(smallvec![(a, grad_id)])
}
