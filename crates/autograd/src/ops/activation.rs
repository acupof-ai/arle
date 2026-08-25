use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn silu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Route Dirty::Device through the lazy `backend.silu` (composes
    // `mlx_multiply(x, mlx_sigmoid(x))` with no eval); Dirty::Host /
    // Dirty::Both stay host-side. Backward is host-only — `silu_backward`
    // clones `x` and forces a host readback of whatever Dirty state it is in.
    let dirty = store.tensor(x)?.dirty.clone();
    match dirty {
        Dirty::Device => silu_device_lazy(x, store, tape),
        Dirty::Host | Dirty::Both => silu_host_eager(x, store, tape),
    }
}

fn silu_device_lazy(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Defensive `ensure_device`: the caller already routed a Dirty::Device
    // tensor, but re-calling guards a future Dirty::Both path from silent
    // drift (mirrors `softmax_device_lazy`).
    store.ensure_device(x)?;
    let input_shape = store.tensor(x)?.shape.clone();
    let input_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "silu: ensure_device left tensor without a device handle",
        ))?
        .clone();

    let out_handle = store.backend().silu(&input_handle, &input_shape)?;
    let output_id = store.alloc_device_tensor(input_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::Silu,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::SiluCtx { x },
    }
    .record(store, tape)?;

    Ok(output_id)
}

fn silu_host_eager(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor_host(x)?;
    let output = store.backend().silu_forward(&input.data)?;
    let output_id = store.alloc(Tensor::new(output, input.shape.clone(), false)?);

    TapeEntry {
        op: BackwardOp::Silu,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::SiluCtx { x },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub fn sigmoid(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Route Dirty::Device through the lazy `backend.sigmoid` (one
    // `mlx_sigmoid` node, no eval); dispatch covers Dirty::Both so
    // post-matmul / post-reshape inputs stay lazy. Backward reads the
    // saved output `y` via `tape.backward`'s pre-walk flush, so
    // `sigmoid_backward` always sees Dirty::Host even when forward stays lazy.
    let has_device_handle = {
        let t = store.tensor(x)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    if has_device_handle {
        sigmoid_device_lazy(x, store, tape)
    } else {
        sigmoid_host_eager(x, store, tape)
    }
}

fn sigmoid_device_lazy(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_device(x)?;
    let input_shape = store.tensor(x)?.shape.clone();
    let input_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "sigmoid: ensure_device left tensor without a device handle",
        ))?
        .clone();

    let out_handle = store.backend().sigmoid(&input_handle, &input_shape)?;
    let output_id = store.alloc_device_tensor(input_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::Sigmoid,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::SigmoidCtx { y: output_id },
    }
    .record(store, tape)?;

    Ok(output_id)
}

fn sigmoid_host_eager(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor_host(x)?;
    let output = store.backend().sigmoid_forward(&input.data)?;
    let output_id = store.alloc(Tensor::new(output, input.shape.clone(), false)?);

    TapeEntry {
        op: BackwardOp::Sigmoid,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::SigmoidCtx { y: output_id },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn silu_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::SiluCtx { x } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "silu backward missing saved input",
        ));
    };
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    let x_shape = store.tensor(x)?.shape.clone();
    if x_shape != upstream_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: x_shape,
            got: upstream_shape,
        });
    }
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        let saved = store.tensor(x)?;
        upstream.dirty != Dirty::Host
            && upstream.device_handle.is_some()
            && saved.dirty != Dirty::Host
            && saved.device_handle.is_some()
    };
    if device_path_ok {
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let x_handle = store
            .tensor(x)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle =
            store
                .backend()
                .silu_backward_device(&upstream_handle, &x_handle, &x_shape)?;
        let grad_id = store.alloc_device_tensor(x_shape, grad_handle)?;
        return Ok(smallvec![(x, grad_id)]);
    }

    let input = store.tensor_host(x)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let grad = input
        .data
        .iter()
        .zip(upstream.data.iter())
        .map(|(&value, &grad_out)| {
            let sigmoid = 1.0 / (1.0 + (-value).exp());
            let derivative = sigmoid + (value * sigmoid * (1.0 - sigmoid));
            grad_out * derivative
        })
        .collect();
    let grad_id = store.alloc(Tensor::new(grad, input.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

pub(crate) fn sigmoid_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("sigmoid missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::SigmoidCtx { y } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "sigmoid backward missing saved output",
        ));
    };

    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    let y_shape = store.tensor(y)?.shape.clone();
    if y_shape != upstream_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: y_shape,
            got: upstream_shape,
        });
    }
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        let saved = store.tensor(y)?;
        upstream.dirty != Dirty::Host
            && upstream.device_handle.is_some()
            && saved.dirty != Dirty::Host
            && saved.device_handle.is_some()
    };
    if device_path_ok {
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let y_handle = store
            .tensor(y)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle =
            store
                .backend()
                .sigmoid_backward_device(&upstream_handle, &y_handle, &y_shape)?;
        let grad_id = store.alloc_device_tensor(y_shape, grad_handle)?;
        return Ok(smallvec![(x, grad_id)]);
    }

    let output = store.tensor_host(y)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let grad = output
        .data
        .iter()
        .zip(upstream.data.iter())
        .map(|(&value, &grad_out)| grad_out * value * (1.0 - value))
        .collect();
    let grad_id = store.alloc(Tensor::new(grad, output.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}
