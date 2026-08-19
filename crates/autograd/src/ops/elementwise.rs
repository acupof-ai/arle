use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::DeviceHandle,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, TapeDtype, Tensor, TensorId, TensorStore},
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

    TapeEntry {
        op: BackwardOp::Add,
        output_id,
        input_ids: smallvec![a, b],
        saved: SavedContext::None,
    }
    .record(store, tape)?;

    Ok(output_id)
}

/// `a + b` with `b`'s device buffer as the destination, so the sum costs no new
/// allocation. `b` is dead after this call — its handle now holds the sum, and
/// only the returned id may be read. Falls back to the allocating `add` unless
/// the buffer is provably sole-owned.
pub fn add_consuming_rhs(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    if a_shape != b_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape,
            got: b_shape,
        });
    }

    store.ensure_device(b)?;
    // Probe before any clone bumps the count (same discriminator as `merge_grad`):
    // `Some(1)` is the only proof no sibling aliases `b`'s buffer. A bf16 tape
    // makes `add` emit a bf16 sum, which an f32 destination cannot hold.
    let reuse_b = store.tape_dtype() == TapeDtype::F32
        && store
            .tensor(b)?
            .device_handle
            .as_ref()
            .and_then(DeviceHandle::device_buffer_strong_count)
            == Some(1);
    let a_handle = store.device_handle(a)?;
    let b_handle = store.device_handle(b)?;

    let out_handle = if reuse_b {
        store
            .backend()
            .accumulate_into_device(&b_handle, &a_handle, &a_shape)?
    } else {
        store.backend().add(&a_handle, &b_handle, &a_shape)?
    };
    let output_id = store.alloc_device_tensor(a_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::Add,
        output_id,
        input_ids: smallvec![a, b],
        saved: SavedContext::None,
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub fn mul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Dispatch is OR-lazy — if EITHER operand is device-resident,
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
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
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
    let output_id = store.alloc_device_tensor(a_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::Mul,
        output_id,
        input_ids: smallvec![a, b],
        saved: SavedContext::Tensors(smallvec![a, b]),
    }
    .record(store, tape)?;

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
    let (a_data, a_shape) = {
        let tensor = store.tensor(a)?;
        (tensor.data.clone(), tensor.shape.clone())
    };
    let (b_data, b_shape) = {
        let tensor = store.tensor(b)?;
        (tensor.data.clone(), tensor.shape.clone())
    };
    if a_shape != b_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape,
            got: b_shape,
        });
    }

    let data = store.backend().mul_forward(&a_data, &b_data)?;
    let output_id = store.alloc(Tensor::new(data, a_shape, false)?);

    TapeEntry {
        op: BackwardOp::Mul,
        output_id,
        input_ids: smallvec![a, b],
        saved: SavedContext::Tensors(smallvec![a, b]),
    }
    .record(store, tape)?;

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
    let input_shape = store.tensor(a)?.shape.clone();

    store.ensure_device(a)?;
    let a_handle = store
        .tensor(a)?
        .device_handle
        .as_ref()
        .expect("ensure_device")
        .clone();

    let out_handle = store.backend().mul_scalar(&a_handle, k, &input_shape)?;
    let output_id = store.alloc_device_tensor(input_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::MulScalar,
        output_id,
        input_ids: smallvec![a],
        saved: SavedContext::TensorAndScalar(a, k),
    }
    .record(store, tape)?;

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

    TapeEntry {
        op: BackwardOp::MulScalar,
        output_id,
        input_ids: smallvec![a],
        saved: SavedContext::TensorAndScalar(a, k),
    }
    .record(store, tape)?;

    Ok(output_id)
}

/// Elementwise `|x|`. The L1 subgradient the DSpark TV term needs, so it can
/// stay device-side instead of folding `sign` into a host-materialized weight.
pub fn abs(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let t = store.tensor(x)?;
    if t.device_handle.is_some() && t.dirty != Dirty::Host {
        abs_device_lazy(x, store, tape)
    } else {
        abs_host_eager(x, store, tape)
    }
}

fn abs_device_lazy(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_device(x)?;
    let input_shape = store.tensor(x)?.shape.clone();
    let input_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "abs: ensure_device left tensor without a device handle",
        ))?
        .clone();

    let out_handle = store.backend().abs(&input_handle, &input_shape)?;
    let output_id = store.alloc_device_tensor(input_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::Abs,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::Tensor(x),
    }
    .record(store, tape)?;

    Ok(output_id)
}

fn abs_host_eager(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor_host(x)?;
    let output = store.backend().abs_forward(&input.data)?;
    let output_id = store.alloc(Tensor::new(output, input.shape.clone(), false)?);

    TapeEntry {
        op: BackwardOp::Abs,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::Tensor(x),
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn abs_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::Tensor(x) = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "abs backward missing saved input",
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
                .abs_backward_device(&upstream_handle, &x_handle, &x_shape)?;
        let grad_id = store.alloc_device_tensor(x_shape, grad_handle)?;
        return Ok(smallvec![(x, grad_id)]);
    }

    let input = store.tensor_host(x)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let grad = input
        .data
        .iter()
        .zip(upstream.data.iter())
        .map(|(&value, &grad_out)| grad_out * crate::backend::cpu_sign(value))
        .collect();
    let grad_id = store.alloc(Tensor::new(grad, input.shape, false)?);
    Ok(smallvec![(x, grad_id)])
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

    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    if a_shape != b_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape,
            got: b_shape,
        });
    }
    let need_grad_a = store.tensor(a)?.requires_grad;
    let need_grad_b = store.tensor(b)?.requires_grad;
    if !need_grad_a && !need_grad_b {
        return Ok(GradPairs::new());
    }

    // Route through `mul_backward_device` whenever upstream and
    // both saved operands are device-resident. The host path does
    // `to_host(upstream) + tensor_host(a) + tensor_host(b)`, forcing
    // three readbacks per layer (Qwen3.5 hot paths `attn * gate` and
    // `silu(gate) * up` × 28 layers).
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        let a_t = store.tensor(a)?;
        let b_t = store.tensor(b)?;
        upstream.dirty != Dirty::Host
            && upstream.device_handle.is_some()
            && a_t.dirty != Dirty::Host
            && a_t.device_handle.is_some()
            && b_t.dirty != Dirty::Host
            && b_t.device_handle.is_some()
    };
    if device_path_ok {
        if upstream_shape != a_shape {
            return Err(AutogradError::ShapeMismatch {
                expected: a_shape,
                got: upstream_shape,
            });
        }
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let a_handle = store
            .tensor(a)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let b_handle = store
            .tensor(b)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let (grad_a_handle, grad_b_handle) = store.backend().mul_backward_device(
            &upstream_handle,
            &a_handle,
            &b_handle,
            &a_shape,
            need_grad_a,
            need_grad_b,
        )?;
        let mut grads = GradPairs::new();
        if let Some(h) = grad_a_handle {
            let grad_id = store.alloc_device_tensor(a_shape.clone(), h)?;
            grads.push((a, grad_id));
        }
        if let Some(h) = grad_b_handle {
            let grad_id = store.alloc_device_tensor(b_shape, h)?;
            grads.push((b, grad_id));
        }
        return Ok(grads);
    }

    let upstream = store.to_host(output_grad_id)?;
    let a_tensor = store.tensor_host(a)?;
    let b_tensor = store.tensor_host(b)?;

    let mut grads = GradPairs::new();
    if need_grad_a {
        let grad_a = store.backend().mul_forward(&upstream, &b_tensor.data)?;
        let grad_id = store.alloc(Tensor::new(grad_a, a_tensor.shape.clone(), false)?);
        grads.push((a, grad_id));
    }
    if need_grad_b {
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

    // Route Dirty::Device upstream through `mul_scalar_backward_device`
    // so the gradient stays on-device. The host path does
    // `to_host(upstream) → mul_scalar_forward → alloc Tensor::new`, which
    // (combined with `mean_backward`'s host fallback) is the *first*
    // host op in the CE-loss backward chain. Keeping this on-device unblocks
    // every downstream `device_path_ok` gate (matmul / softmax / gather /
    // accumulate_grad).
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    let input_shape = store.tensor(a)?.shape.clone();
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        upstream.dirty != Dirty::Host && upstream.device_handle.is_some()
    };
    if device_path_ok {
        if upstream_shape != input_shape {
            return Err(AutogradError::ShapeMismatch {
                expected: input_shape,
                got: upstream_shape,
            });
        }
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle =
            store
                .backend()
                .mul_scalar_backward_device(&upstream_handle, k, &input_shape)?;
        let grad_id = store.alloc_device_tensor(input_shape, grad_handle)?;
        return Ok(smallvec![(a, grad_id)]);
    }

    let upstream = store.to_host(output_grad_id)?;
    let grad = store.backend().mul_scalar_forward(&upstream, k)?;
    let grad_id = store.alloc(Tensor::new(grad, input_shape, false)?);
    Ok(smallvec![(a, grad_id)])
}
