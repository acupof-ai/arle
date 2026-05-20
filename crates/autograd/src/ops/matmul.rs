use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::{
        cpu_matmul_bt_backward, matmul_bt_output_shape as backend_matmul_bt_output_shape,
        matmul_output_shape as backend_matmul_output_shape,
    },
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn matmul(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
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
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    let requires_grad = store.tensor(a)?.requires_grad || store.tensor(b)?.requires_grad;
    let (out_handle, out_shape) = store
        .backend()
        .matmul(&a_handle, &a_shape, &b_handle, &b_shape)?;
    let output_id = store.alloc_device_tensor(out_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Matmul,
            output_id,
            input_ids: smallvec![a, b],
            saved: SavedContext::MatmulCtx { a, b },
        });
    }

    Ok(output_id)
}

pub fn matmul_bt(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
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
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    let requires_grad = store.tensor(a)?.requires_grad || store.tensor(b)?.requires_grad;
    let (out_handle, out_shape) = store
        .backend()
        .matmul_bt(&a_handle, &a_shape, &b_handle, &b_shape)?;
    let output_id = store.alloc_device_tensor(out_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::MatmulBT,
            output_id,
            input_ids: smallvec![a, b],
            saved: SavedContext::MatmulBTCtx { a, b },
        });
    }

    Ok(output_id)
}

pub(crate) fn matmul_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::MatmulCtx { a, b } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "matmul backward missing saved context",
        ));
    };

    // Shape gate — borrow-only so we don't force a host readback when both
    // sides are still device-resident. Cloning here would panic on
    // `Dirty::Device` and (worse for `Dirty::Both`) materialise the 1 GB
    // upstream gradient on host even when the device path is taken.
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    let need_grad_a = store.tensor(a)?.requires_grad;
    let need_grad_b = store.tensor(b)?.requires_grad;
    let expected_shape = matmul_output_shape(&a_shape, &b_shape)?;
    if upstream_shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream_shape,
        });
    }

    let mut grads = GradPairs::new();
    if !need_grad_a && !need_grad_b {
        return Ok(grads);
    }
    match (a_shape.len(), b_shape.len()) {
        (2, 2) | (3, 3) => {}
        _ => {
            return Err(AutogradError::InvalidRank {
                expected: "both operands must be rank-2 or rank-3",
                got: a_shape.len().max(b_shape.len()),
            });
        }
    }

    // P2 (device-resident gradient tape): when all three operands are still
    // device-resident, dispatch through `matmul_backward_device` so the
    // saved hidden / weight buffers and the upstream gradient never round-
    // trip through host. This is the contract change that retires the
    // 1 GB DtoH that Wave 1 surfaced — the LM-head GEMM's
    // `grad_out: &[f32]` was the single largest readback per step
    // (`docs/research/2026-05-17-cuda-training-architectural-correction.md`).
    let device_path_ok = {
        let a_t = store.tensor(a)?;
        let b_t = store.tensor(b)?;
        let g_t = store.tensor(output_grad_id)?;
        a_t.dirty != Dirty::Host
            && a_t.device_handle.is_some()
            && b_t.dirty != Dirty::Host
            && b_t.device_handle.is_some()
            && g_t.dirty != Dirty::Host
            && g_t.device_handle.is_some()
    };
    if device_path_ok {
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
        let g_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let (grad_a_handle, grad_b_handle) = store.backend().matmul_backward_device(
            &a_handle,
            &a_shape,
            &b_handle,
            &b_shape,
            &g_handle,
            &upstream_shape,
            need_grad_a,
            need_grad_b,
        )?;
        if let Some(handle) = grad_a_handle {
            let grad_id = store.alloc_device_tensor(a_shape.clone(), handle)?;
            grads.push((a, grad_id));
        }
        if let Some(handle) = grad_b_handle {
            let grad_id = store.alloc_device_tensor(b_shape.clone(), handle)?;
            grads.push((b, grad_id));
        }
        return Ok(grads);
    }

    // Host-eager fallback. Any operand without a device handle (or already
    // marked `Dirty::Host`) drops the whole call back onto the legacy
    // `matmul_backward(&[f32], …)` contract, matching pre-P2 behaviour.
    let a_tensor = store.tensor_host(a)?;
    let b_tensor = store.tensor_host(b)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let (grad_a_data, grad_b_data) = store.backend().matmul_backward(
        &a_tensor.data,
        &a_tensor.shape,
        &b_tensor.data,
        &b_tensor.shape,
        &upstream.data,
        &upstream.shape,
        need_grad_a,
        need_grad_b,
    )?;
    if need_grad_a {
        let grad_id = store.alloc(Tensor::new(grad_a_data, a_tensor.shape.clone(), false)?);
        grads.push((a, grad_id));
    }
    if need_grad_b {
        let grad_id = store.alloc(Tensor::new(grad_b_data, b_tensor.shape.clone(), false)?);
        grads.push((b, grad_id));
    }

    Ok(grads)
}

pub(crate) fn matmul_bt_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::MatmulBTCtx { a, b } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "matmul_bt backward missing saved context",
        ));
    };

    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    let need_grad_a = store.tensor(a)?.requires_grad;
    let need_grad_b = store.tensor(b)?.requires_grad;
    let expected_shape = matmul_bt_output_shape(&a_shape, &b_shape)?;
    if upstream_shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream_shape,
        });
    }

    let mut grads = GradPairs::new();
    if !need_grad_a && !need_grad_b {
        return Ok(grads);
    }
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2",
            got: a_shape.len().max(b_shape.len()),
        });
    }

    let device_path_ok = {
        let a_t = store.tensor(a)?;
        let b_t = store.tensor(b)?;
        let g_t = store.tensor(output_grad_id)?;
        a_t.dirty != Dirty::Host
            && a_t.device_handle.is_some()
            && b_t.dirty != Dirty::Host
            && b_t.device_handle.is_some()
            && g_t.dirty != Dirty::Host
            && g_t.device_handle.is_some()
    };
    if device_path_ok {
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
        let g_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let (grad_a_handle, grad_b_handle) = store.backend().matmul_bt_backward_device(
            &a_handle,
            &a_shape,
            &b_handle,
            &b_shape,
            &g_handle,
            &upstream_shape,
            need_grad_a,
            need_grad_b,
        )?;
        if let Some(handle) = grad_a_handle {
            let grad_id = store.alloc_device_tensor(a_shape.clone(), handle)?;
            grads.push((a, grad_id));
        }
        if let Some(handle) = grad_b_handle {
            let grad_id = store.alloc_device_tensor(b_shape.clone(), handle)?;
            grads.push((b, grad_id));
        }
        return Ok(grads);
    }

    let a_tensor = store.tensor_host(a)?;
    let b_tensor = store.tensor_host(b)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let (grad_a_data, grad_b_data) = cpu_matmul_bt_backward(
        &a_tensor.data,
        &a_tensor.shape,
        &b_tensor.data,
        &b_tensor.shape,
        &upstream.data,
        &upstream.shape,
        need_grad_a,
        need_grad_b,
    )?;
    if need_grad_a {
        let grad_id = store.alloc(Tensor::new(grad_a_data, a_tensor.shape.clone(), false)?);
        grads.push((a, grad_id));
    }
    if need_grad_b {
        let grad_id = store.alloc(Tensor::new(grad_b_data, b_tensor.shape.clone(), false)?);
        grads.push((b, grad_id));
    }

    Ok(grads)
}

fn matmul_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    backend_matmul_output_shape(a_shape, b_shape)
}

fn matmul_bt_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    backend_matmul_bt_output_shape(a_shape, b_shape)
}
