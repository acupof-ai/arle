use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::broadcast_offset,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn add_broadcast(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // If EITHER operand is lazily device-resident, go lazy and upload the
    // other. The alternative (host_eager with `ensure_host` on the
    // device-resident side) would force an eval on the upstream lazy
    // graph (e.g. a matmul output) — exactly the readback we are trying
    // to eliminate. Uploading a small host side (bias, mask) is cheap;
    // forcing a readback of a large activation is not.
    let a_use_lazy = {
        let t = store.tensor(a)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    let b_use_lazy = {
        let t = store.tensor(b)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    if a_use_lazy || b_use_lazy {
        add_broadcast_device_lazy(a, b, store, tape)
    } else {
        add_broadcast_host_eager(a, b, store, tape)
    }
}

fn add_broadcast_device_lazy(
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

    let out_handle = store
        .backend()
        .add_broadcast(&a_handle, &a_shape, &b_handle, &b_shape)?;
    let requires_grad = a_requires_grad || b_requires_grad;
    let output_id = store.alloc_device_tensor(a_shape.clone(), out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::AddBroadcast,
            output_id,
            input_ids: smallvec![a, b],
            saved: SavedContext::AddBroadcastCtx { a_shape, b_shape },
        });
    }

    Ok(output_id)
}

fn add_broadcast_host_eager(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Mixed-residency fallback: at least one input is on the host side,
    // or one is device-resident while the other is host-only (e.g. the
    // Linear-bias case where `a` is a matmul output on the device and
    // `b` is a freshly-initialized host bias). Sync both to host before
    // we clone + call the host-side `add_broadcast_forward`.
    store.ensure_host(a)?;
    store.ensure_host(b)?;
    let a_tensor = store.tensor_host(a)?;
    let b_tensor = store.tensor_host(b)?;

    let output = store.backend().add_broadcast_forward(
        &a_tensor.data,
        &a_tensor.shape,
        &b_tensor.data,
        &b_tensor.shape,
    )?;

    let requires_grad = a_tensor.requires_grad || b_tensor.requires_grad;
    let output_id = store.alloc(Tensor::new(output, a_tensor.shape.clone(), requires_grad)?);
    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::AddBroadcast,
            output_id,
            input_ids: smallvec![a, b],
            saved: SavedContext::AddBroadcastCtx {
                a_shape: a_tensor.shape,
                b_shape: b_tensor.shape,
            },
        });
    }

    Ok(output_id)
}

pub(crate) fn add_broadcast_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let a = *entry.input_ids.first().ok_or(AutogradError::TapeInvariant(
        "add_broadcast missing lhs input",
    ))?;
    let b = *entry.input_ids.get(1).ok_or(AutogradError::TapeInvariant(
        "add_broadcast missing rhs input",
    ))?;

    let SavedContext::AddBroadcastCtx { a_shape, b_shape } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "add_broadcast backward missing saved shapes",
        ));
    };
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    if upstream_shape != a_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape.clone(),
            got: upstream_shape,
        });
    }

    let a_requires_grad = store.tensor(a)?.requires_grad;
    let b_requires_grad = store.tensor(b)?.requires_grad;

    // Wave 2 Commit A: route Dirty::Device upstream through
    // `add_broadcast_backward_device` so the `[B, S, H]` upstream tensor
    // and the `[H]`-shaped b-grad stay on-device. grad_a is the upstream
    // tensor itself (no reduce); just share its handle when device-resident
    // — that avoids both a memcpy and an extra device alloc.
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        upstream.dirty != Dirty::Host && upstream.device_handle.is_some()
    };
    if device_path_ok {
        let mut grads = GradPairs::new();
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        if a_requires_grad {
            // grad_a = upstream (identity through broadcast-add). Share the
            // same device handle — the tape will accumulate into `a`'s grad
            // via the device-resident `add_into_device` path.
            let grad_a_id = store.alloc_device_tensor(a_shape.clone(), upstream_handle.clone())?;
            grads.push((a, grad_a_id));
        }
        if b_requires_grad {
            let grad_b_handle = store.backend().add_broadcast_backward_device(
                &upstream_handle,
                &a_shape,
                &b_shape,
            )?;
            let grad_b_id = store.alloc_device_tensor(b_shape, grad_b_handle)?;
            grads.push((b, grad_b_id));
        }
        return Ok(grads);
    }

    let upstream = store.tensor_host(output_grad_id)?;
    let mut grads = GradPairs::new();
    if a_requires_grad {
        let grad_id = store.alloc(Tensor::new(upstream.data.clone(), a_shape, false)?);
        grads.push((a, grad_id));
    }

    if b_requires_grad {
        let b_size = if b_shape.is_empty() {
            1
        } else {
            b_shape.iter().product()
        };
        let mut grad_b = vec![0.0; b_size];
        for (index, grad_value) in upstream.data.iter().enumerate() {
            let offset = broadcast_offset(index, &entry.output_id_shape(store)?, &b_shape);
            grad_b[offset] += *grad_value;
        }
        let grad_id = store.alloc(Tensor::new(grad_b, b_shape, false)?);
        grads.push((b, grad_id));
    }

    Ok(grads)
}

trait OutputShapeExt {
    fn output_id_shape(&self, store: &TensorStore) -> Result<Vec<usize>>;
}

impl OutputShapeExt for TapeEntry {
    fn output_id_shape(&self, store: &TensorStore) -> Result<Vec<usize>> {
        Ok(store.tensor(self.output_id)?.shape.clone())
    }
}
