use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::Device,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn rmsnorm(
    x: TensorId,
    weight: TensorId,
    eps: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Dispatch on device-handle presence (mirrors rope so Dirty::Both after
    // `ensure_device` stays lazy too). The lazy path skips the host-side
    // inv_rms computation — `rmsnorm_backward` recomputes inv_rms from x
    // (materialized to host by `tape.backward`'s pre-walk flush); an empty
    // saved `inv_rms` signals "recompute". weight is always host-resident
    // (shape [hidden]): `backend.rms_norm` takes it as a host slice, and the
    // per-call upload inside the Metal FFI wrapper is cheaper than a
    // device-handle code path for the weight.
    let has_device_handle = {
        let t = store.tensor(x)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    if has_device_handle {
        rmsnorm_device_lazy(x, weight, eps, store, tape)
    } else {
        rmsnorm_host_eager(x, weight, eps, store, tape)
    }
}

fn rmsnorm_device_lazy(
    x: TensorId,
    weight: TensorId,
    eps: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(weight)?;
    store.ensure_device(weight)?;
    store.ensure_device(x)?;

    let x_shape = store.tensor(x)?.shape.clone();
    let hidden = *x_shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    let weight_tensor = store.tensor_host(weight)?;
    if weight_tensor.shape != vec![hidden] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![hidden],
            got: weight_tensor.shape,
        });
    }

    let x_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "rmsnorm: ensure_device left x without a device handle",
        ))?
        .clone();
    let out_handle = store
        .backend()
        .rms_norm(&x_handle, &weight_tensor.data, &x_shape, eps)?;
    let output_id = store.alloc_device_tensor(x_shape, out_handle)?;

    // Empty inv_rms = "recompute from x in backward". x is Dirty::Device
    // here; tape.backward's batch-flush makes it Dirty::Both first.
    TapeEntry {
        op: BackwardOp::RMSNorm,
        output_id,
        input_ids: smallvec![x, weight],
        saved: SavedContext::RMSNormCtx {
            x,
            weight,
            inv_rms: Vec::new(),
            eps,
        },
    }
    .record(store, tape)?;

    Ok(output_id)
}

fn rmsnorm_host_eager(
    x: TensorId,
    weight: TensorId,
    eps: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x_tensor = store.tensor_host(x)?;
    let weight_tensor = store.tensor_host(weight)?;
    let hidden = *x_tensor.shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if weight_tensor.shape != vec![hidden] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![hidden],
            got: weight_tensor.shape,
        });
    }

    let requires_grad = x_tensor.requires_grad || weight_tensor.requires_grad;
    let output = store.backend().rms_norm_forward(
        &x_tensor.data,
        &weight_tensor.data,
        &x_tensor.shape,
        eps,
    )?;

    let output_id = store.alloc(Tensor::new(output, x_tensor.shape.clone(), requires_grad)?);
    if requires_grad {
        let rows = x_tensor.size / hidden;
        let inv_rms: Vec<f32> = (0..rows)
            .map(|row| {
                let sum_sq: f32 = x_tensor.data[row * hidden..(row + 1) * hidden]
                    .iter()
                    .map(|&v| v * v)
                    .sum();
                1.0 / ((sum_sq / hidden as f32) + eps).sqrt()
            })
            .collect();
        TapeEntry {
            op: BackwardOp::RMSNorm,
            output_id,
            input_ids: smallvec![x, weight],
            saved: SavedContext::RMSNormCtx {
                x,
                weight,
                inv_rms,
                eps,
            },
        }
        .record(store, tape)?;
    }

    Ok(output_id)
}

pub(crate) fn rmsnorm_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::RMSNormCtx {
        x,
        weight,
        inv_rms,
        eps,
    } = entry.saved.clone()
    else {
        return Err(AutogradError::TapeInvariant(
            "rmsnorm backward missing saved context",
        ));
    };

    let x_shape = store.tensor(x)?.shape.clone();
    let weight_shape = store.tensor(weight)?.shape.clone();
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    if upstream_shape != x_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: x_shape,
            got: upstream_shape,
        });
    }
    let hidden = *x_shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    let need_grad_x = store.tensor(x)?.requires_grad;
    let need_grad_w = store.tensor(weight)?.requires_grad;
    if !need_grad_x && !need_grad_w {
        return Ok(GradPairs::new());
    }

    // Route through `rms_norm_backward_device` when upstream, x, and weight
    // are all device-resident. A host path would `ensure_host(x)` for all
    // three operands, demoting x before any downstream device op could see
    // it. Heal a host-resident upstream grad first (mirrors matmul_backward)
    // so one host grad upstream doesn't demote this norm — and everything
    // downstream.
    if store.backend().device() != Device::Cpu {
        store.ensure_device(output_grad_id)?;
        store.ensure_device(x)?;
        store.ensure_device(weight)?;
    }
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        let x_t = store.tensor(x)?;
        let w_t = store.tensor(weight)?;
        upstream.dirty != Dirty::Host
            && upstream.device_handle.is_some()
            && x_t.dirty != Dirty::Host
            && x_t.device_handle.is_some()
            && w_t.dirty != Dirty::Host
            && w_t.device_handle.is_some()
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
        let w_handle = store
            .tensor(weight)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let (grad_x_handle, grad_w_handle) = store.backend().rms_norm_backward_device(
            &upstream_handle,
            &x_handle,
            &w_handle,
            &x_shape,
            eps,
            need_grad_x,
            need_grad_w,
        )?;
        let mut grads = GradPairs::new();
        if let Some(h) = grad_x_handle {
            let grad_id = store.alloc_device_tensor(x_shape.clone(), h)?;
            grads.push((x, grad_id));
        }
        if let Some(h) = grad_w_handle {
            let grad_id = store.alloc_device_tensor(weight_shape, h)?;
            grads.push((weight, grad_id));
        }
        return Ok(grads);
    }

    // Host fallback (CPU/Metal). A lazy-device forward saved inv_rms empty —
    // recompute it from the now-host x.
    store.ensure_host(x)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let x_tensor = store.tensor_host(x)?;
    let weight_tensor = store.tensor_host(weight)?;

    let rows = x_tensor.size / hidden;
    let inv_rms = if inv_rms.is_empty() {
        (0..rows)
            .map(|row| {
                let sum_sq: f32 = x_tensor.data[row * hidden..(row + 1) * hidden]
                    .iter()
                    .map(|&v| v * v)
                    .sum();
                1.0 / ((sum_sq / hidden as f32) + eps).sqrt()
            })
            .collect()
    } else if inv_rms.len() != rows {
        return Err(AutogradError::TapeInvariant(
            "rmsnorm inverse-rms rows mismatch",
        ));
    } else {
        inv_rms
    };

    let mut grads = GradPairs::new();
    if need_grad_x {
        let mut grad_x = vec![0.0; x_tensor.size];
        for (row, &inv) in inv_rms.iter().enumerate() {
            let base = row * hidden;
            let mut dot = 0.0;
            for col in 0..hidden {
                dot +=
                    upstream.data[base + col] * weight_tensor.data[col] * x_tensor.data[base + col];
            }
            let correction = inv * inv * dot / hidden as f32;
            for col in 0..hidden {
                let scaled_grad = upstream.data[base + col] * weight_tensor.data[col];
                grad_x[base + col] =
                    (inv * scaled_grad) - (x_tensor.data[base + col] * inv * correction);
            }
        }
        let grad_id = store.alloc(Tensor::new(grad_x, x_tensor.shape.clone(), false)?);
        grads.push((x, grad_id));
    }

    if need_grad_w {
        let mut grad_weight = vec![0.0; hidden];
        for (row, &inv) in inv_rms.iter().enumerate() {
            let base = row * hidden;
            for (col, grad_slot) in grad_weight.iter_mut().enumerate() {
                *grad_slot += upstream.data[base + col] * x_tensor.data[base + col] * inv;
            }
        }
        let grad_id = store.alloc(Tensor::new(grad_weight, weight_tensor.shape, false)?);
        grads.push((weight, grad_id));
    }

    Ok(grads)
}
