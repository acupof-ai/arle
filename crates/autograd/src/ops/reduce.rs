use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn sum(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Route Dirty::Device through the lazy `backend.sum_all` (composes
    // `reshape -> sum_axis` into the MLX graph with no eval); keep
    // Dirty::Host / Dirty::Both on the host fast path to avoid an
    // upload+device-reduce+readback for host-resident scalars.
    let dirty = store.tensor(a)?.dirty.clone();
    match dirty {
        Dirty::Device => sum_device_lazy(a, store, tape),
        Dirty::Host | Dirty::Both => sum_host_eager(a, store, tape),
    }
}

fn sum_device_lazy(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // `ensure_device` is a no-op here (caller routed in Dirty::Device);
    // re-called defensively so a future Dirty::Both path lands correctly
    // without silent drift. Scalar metadata is extracted in a scoped borrow
    // to avoid the `Tensor::clone` assert against `Dirty::Device`.
    store.ensure_device(a)?;
    let input_shape = store.tensor(a)?.shape.clone();
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

    TapeEntry {
        op: BackwardOp::Sum,
        output_id,
        input_ids: smallvec![a],
        saved: SavedContext::Shape(input_shape),
    }
    .record(store, tape)?;

    Ok(output_id)
}

fn sum_host_eager(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Keeps host-resident reductions purely host-side — no FFI, no upload,
    // no device scalar the next op would have to pull back down.
    let input = store.tensor_host(a)?;
    let value = input.data.iter().sum::<f32>();
    let output_id = store.alloc(Tensor::new(vec![value], Vec::new(), input.requires_grad)?);

    TapeEntry {
        op: BackwardOp::Sum,
        output_id,
        input_ids: smallvec![a],
        saved: SavedContext::Shape(input.shape.clone()),
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub fn mean(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Route Dirty::Device through a lazy `sum_all + mul_scalar` compose on
    // the MLX graph (no new Backend trait method). Hot path: CE-loss head
    // `log_softmax → gather_last_dim → mean → mul_scalar` — without this
    // the per-step loss flushes the full log-probs tensor back to host,
    // reversing every upstream lazy win. Dirty::Host / Dirty::Both stay
    // host-side.
    let dirty = store.tensor(a)?.dirty.clone();
    match dirty {
        Dirty::Device => mean_device_lazy(a, store, tape),
        Dirty::Host | Dirty::Both => mean_host_eager(a, store, tape),
    }
}

fn mean_device_lazy(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_device(a)?;
    let (input_shape, numel) = {
        let tensor = store.tensor(a)?;
        (tensor.shape.clone(), tensor.size)
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

    TapeEntry {
        op: BackwardOp::Mean,
        output_id,
        input_ids: smallvec![a],
        saved: SavedContext::MeanCtx { input: a, numel },
    }
    .record(store, tape)?;

    Ok(output_id)
}

fn mean_host_eager(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor_host(a)?;
    let value = input.data.iter().sum::<f32>() / input.size as f32;
    let output_id = store.alloc(Tensor::new(vec![value], Vec::new(), input.requires_grad)?);

    TapeEntry {
        op: BackwardOp::Mean,
        output_id,
        input_ids: smallvec![a],
        saved: SavedContext::MeanCtx {
            input: a,
            numel: input.size,
        },
    }
    .record(store, tape)?;

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
    let device_path_ok = {
        let output_grad = store.tensor(output_grad_id)?;
        output_grad.dirty != Dirty::Host && output_grad.device_handle.is_some()
    };
    if device_path_ok {
        let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
        if upstream_shape != Vec::<usize>::new() {
            return Err(AutogradError::ShapeMismatch {
                expected: Vec::new(),
                got: upstream_shape,
            });
        }
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle = store
            .backend()
            .sum_backward_device(&upstream_handle, shape)?;
        let grad_id = store.alloc_device_tensor(shape.clone(), grad_handle)?;
        return Ok(smallvec![(a, grad_id)]);
    }

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

    // Route Dirty::Device upstream through `mean_backward_device` so the
    // scalar grad is broadcast-scaled on-device. The host fallback (readback
    // scalar + alloc `vec![v; N]`) is the *first* host op in the CE-loss
    // backward chain — its Dirty::Host output would demote every downstream
    // device override (`matmul_backward_device`, `log_softmax_last_axis_backward`,
    // `gather_last_dim_backward`, `add_into_device`) to host, dragging the
    // `[B, S, V] ≈ 1 GB` logits tile back through DtoH per step.
    let input_shape = store.tensor(a)?.shape.clone();
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        upstream.dirty != Dirty::Host && upstream.device_handle.is_some()
    };
    if device_path_ok {
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle =
            store
                .backend()
                .mean_backward_device(&upstream_handle, &input_shape, numel)?;
        let grad_id = store.alloc_device_tensor(input_shape, grad_handle)?;
        return Ok(smallvec![(a, grad_id)]);
    }

    let output_grad = store.tensor(output_grad_id)?;
    if output_grad.shape != Vec::<usize>::new() || output_grad.data.len() != 1 {
        return Err(AutogradError::ShapeMismatch {
            expected: Vec::new(),
            got: output_grad.shape.clone(),
        });
    }

    let grad_value = output_grad.data[0] / numel as f32;
    let grad_id = store.alloc(Tensor::new(vec![grad_value; numel], input_shape, false)?);
    Ok(smallvec![(a, grad_id)])
}
