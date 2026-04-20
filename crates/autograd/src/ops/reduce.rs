use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn sum(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.1: route Dirty::Device inputs through the lazy `backend.sum_all`
    // (composes `reshape -> sum_axis` into the MLX graph with no eval),
    // but keep Dirty::Host and Dirty::Both inputs on the host fast path
    // so we don't pay an unnecessary upload+device-reduce+readback for
    // scalars whose producer already lives on host (e.g. train_sft's
    // `sum(masked, ...)` right after `mul`). Codex-flagged P1 regression
    // that this branch closes.
    let dirty = store.tensor(a)?.dirty.clone();
    match dirty {
        Dirty::Device => sum_device_lazy(a, store, tape),
        Dirty::Host | Dirty::Both => sum_host_eager(a, store, tape),
    }
}

fn sum_device_lazy(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Metal lazy path. `ensure_device` is a no-op here because the caller
    // already routed in a Dirty::Device tensor; we re-call it defensively
    // so a future Dirty::Both path lands on the correct side without silent
    // drift. We extract scalar metadata in a scoped borrow so we never hit
    // the `Tensor::clone` assert against `Dirty::Device`.
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

fn sum_host_eager(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Pre-M5.3b.1 fast path for Dirty::Host / Dirty::Both inputs. Keeps
    // host-resident reductions purely host-side — no FFI, no upload, no
    // device scalar that the next op will have to pull back down.
    let input = store.tensor(a)?.clone();
    let value = input.data.iter().sum::<f32>();
    let output_id = store.alloc(Tensor::new(vec![value], Vec::new(), input.requires_grad)?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Sum,
            output_id,
            input_ids: smallvec![a],
            saved: SavedContext::Shape(input.shape.clone()),
        });
    }

    Ok(output_id)
}

pub fn mean(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.19: route Dirty::Device inputs through a lazy `sum_all + mul_scalar`
    // compose on the MLX graph (no new Backend trait method — reuses the
    // existing lazy `sum_all` from M5.3b.1 and `mul_scalar` from M5.3b.13).
    // Hot path: CE-loss head `log_softmax → gather_last_dim → mean → mul_scalar`
    // — without this the CE loss per-step flushes the full log-probs tensor
    // back to host, reversing every upstream M5.3b lazy win. Dirty::Host and
    // Dirty::Both stay on the host fast path so host-resident scalars don't
    // pay an upload+device-reduce+readback.
    let dirty = store.tensor(a)?.dirty.clone();
    match dirty {
        Dirty::Device => mean_device_lazy(a, store, tape),
        Dirty::Host | Dirty::Both => mean_host_eager(a, store, tape),
    }
}

fn mean_device_lazy(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_device(a)?;
    let (input_shape, numel, requires_grad) = {
        let tensor = store.tensor(a)?;
        (tensor.shape.clone(), tensor.size, tensor.requires_grad)
    };
    let input_handle = store
        .tensor(a)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "mean: ensure_device left tensor without a device handle",
        ))?
        .clone();

    let sum_handle = store.backend().sum_all(&input_handle, &input_shape)?;
    let inv_numel = if numel == 0 { 0.0 } else { 1.0 / numel as f32 };
    let out_handle = store.backend().mul_scalar(&sum_handle, inv_numel, &[])?;
    let output_id = store.alloc_device_tensor(Vec::new(), out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Mean,
            output_id,
            input_ids: smallvec![a],
            saved: SavedContext::MeanCtx { input: a, numel },
        });
    }

    Ok(output_id)
}

fn mean_host_eager(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
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
