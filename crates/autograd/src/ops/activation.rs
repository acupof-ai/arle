use libm::erff;
use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

const INV_SQRT_2: f32 = 0.707_106_77;
const INV_SQRT_2PI: f32 = 0.398_942_3;

pub fn exp(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.4: route Dirty::Device inputs through the lazy `backend.exp`
    // (pipes a single `mlx_exp` node into the MLX graph, no eval).
    // Dirty::Host / Dirty::Both stay on the host fast path so
    // host-resident producers don't pay an upload+device-compute+readback.
    // Mirrors the M5.3b.3 silu dispatch shape. Backward reads the saved
    // output via `tape.backward`'s pre-walk flush, so `exp_backward`
    // always sees Dirty::Host even when the forward stays lazy.
    let dirty = store.tensor(x)?.dirty.clone();
    match dirty {
        Dirty::Device => exp_device_lazy(x, store, tape),
        Dirty::Host | Dirty::Both => exp_host_eager(x, store, tape),
    }
}

fn exp_device_lazy(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Defensive `ensure_device`: caller already routed a Dirty::Device
    // tensor, but re-calling guards a future Dirty::Both path from silent
    // drift (mirrors `silu_device_lazy`).
    store.ensure_device(x)?;
    let (input_shape, requires_grad) = {
        let tensor = store.tensor(x)?;
        (tensor.shape.clone(), tensor.requires_grad)
    };
    let input_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "exp: ensure_device left tensor without a device handle",
        ))?
        .clone();

    let out_handle = store.backend().exp(&input_handle, &input_shape)?;
    let output_id = store.alloc_device_tensor(input_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Exp,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::Tensor(output_id),
        });
    }

    Ok(output_id)
}

fn exp_host_eager(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor_host(x)?;
    let output = store.backend().exp_forward(&input.data)?;
    let output_id = store.alloc(Tensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Exp,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::Tensor(output_id),
        });
    }

    Ok(output_id)
}

pub fn gelu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.8: route Dirty::Device inputs through the lazy `backend.gelu`
    // (erf-form, composed from `mlx_multiply → mlx_erf → mlx_add →
    // mlx_multiply` on the MLX graph). Dispatch covers both Dirty::Device
    // and Dirty::Both so post-matmul and reshape-reentry inputs both stay
    // lazy. `gelu_backward` uses the erf derivative of the saved input;
    // tape.backward's pre-walk batch flush materializes the input before
    // the backward walk, so saving `x` here is safe.
    let has_device_handle = {
        let t = store.tensor(x)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    if has_device_handle {
        gelu_device_lazy(x, store, tape)
    } else {
        gelu_host_eager(x, store, tape)
    }
}

fn gelu_device_lazy(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_device(x)?;
    let (input_shape, requires_grad) = {
        let tensor = store.tensor(x)?;
        (tensor.shape.clone(), tensor.requires_grad)
    };
    let input_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "gelu: ensure_device left tensor without a device handle",
        ))?
        .clone();

    let out_handle = store.backend().gelu(&input_handle, &input_shape)?;
    let output_id = store.alloc_device_tensor(input_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Gelu,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::GeluCtx { x },
        });
    }

    Ok(output_id)
}

fn gelu_host_eager(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor_host(x)?;
    let output = input
        .data
        .iter()
        .map(|&value| 0.5 * value * (1.0 + erff(value * INV_SQRT_2)))
        .collect();
    let output_id = store.alloc(Tensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Gelu,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::GeluCtx { x },
        });
    }

    Ok(output_id)
}

pub fn silu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.3: route Dirty::Device inputs through the lazy `backend.silu`
    // (composes `mlx_multiply(x, mlx_sigmoid(x))` into the MLX graph with
    // no eval). Dirty::Host / Dirty::Both stay on the host fast path so
    // host-resident producers don't pay an upload+device-compute+readback.
    // Mirrors the M5.3b.1 sum / M5.3b.2 softmax dispatch shape. Backward
    // stays host-only — `silu_backward` clones `x` and forces a host
    // readback of whatever Dirty state it is in, matching the pre-M5.3b.3
    // behavior.
    let dirty = store.tensor(x)?.dirty.clone();
    match dirty {
        Dirty::Device => silu_device_lazy(x, store, tape),
        Dirty::Host | Dirty::Both => silu_host_eager(x, store, tape),
    }
}

fn silu_device_lazy(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Defensive `ensure_device`: caller already routed a Dirty::Device
    // tensor, but re-calling guards a future Dirty::Both path from silent
    // drift (mirrors `softmax_device_lazy`).
    store.ensure_device(x)?;
    let (input_shape, requires_grad) = {
        let tensor = store.tensor(x)?;
        (tensor.shape.clone(), tensor.requires_grad)
    };
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
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Silu,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::SiluCtx { x },
        });
    }

    Ok(output_id)
}

fn silu_host_eager(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor_host(x)?;
    let output = store.backend().silu_forward(&input.data)?;
    let output_id = store.alloc(Tensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Silu,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::SiluCtx { x },
        });
    }

    Ok(output_id)
}

pub fn sigmoid(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.18: route Dirty::Device inputs through the lazy `backend.sigmoid`
    // (pipes a single `mlx_sigmoid` node into the MLX graph, no eval).
    // Dirty::Host stays on the host fast path. Dispatch covers Dirty::Both
    // so post-matmul / post-reshape inputs also stay lazy. Backward reads
    // the saved output `y` via `tape.backward`'s pre-walk flush, so
    // `sigmoid_backward` always sees Dirty::Host even when forward stays
    // lazy. Mirrors the M5.3b.4 exp dispatch shape.
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
    let (input_shape, requires_grad) = {
        let tensor = store.tensor(x)?;
        (tensor.shape.clone(), tensor.requires_grad)
    };
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
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Sigmoid,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::SigmoidCtx { y: output_id },
        });
    }

    Ok(output_id)
}

fn sigmoid_host_eager(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor_host(x)?;
    let output = store.backend().sigmoid_forward(&input.data)?;
    let output_id = store.alloc(Tensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Sigmoid,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::SigmoidCtx { y: output_id },
        });
    }

    Ok(output_id)
}

pub(crate) fn exp_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("exp missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::Tensor(y_id) = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "exp backward missing saved output",
        ));
    };

    // Wave 2.1: route the (upstream, saved-output) pair through
    // `exp_backward_device` when both tensors are device-resident. Pre-2.1
    // this op did `tensor_host(y) + tensor_host(upstream) → mul_forward →
    // alloc`, which (via `tensor_host`'s `ensure_host`) demoted the saved
    // output from Dirty::Device → Dirty::Both, poisoning every downstream
    // op that re-read `y`. Keeping both on-device keeps the post-P3.1
    // backward chain unbroken.
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    let y_shape = store.tensor(y_id)?.shape.clone();
    if y_shape != upstream_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: y_shape,
            got: upstream_shape,
        });
    }
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        let saved = store.tensor(y_id)?;
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
            .tensor(y_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle =
            store
                .backend()
                .exp_backward_device(&upstream_handle, &y_handle, &y_shape)?;
        let grad_id = store.alloc_device_tensor(y_shape, grad_handle)?;
        return Ok(smallvec![(x, grad_id)]);
    }

    let output = store.tensor_host(y_id)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let grad = store.backend().mul_forward(&output.data, &upstream.data)?;
    let grad_id = store.alloc(Tensor::new(grad, output.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

pub(crate) fn gelu_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::GeluCtx { x } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "gelu backward missing saved input",
        ));
    };
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    // Wave 2.1: route through `gelu_backward_device` whenever upstream and
    // saved input are both device-resident.
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
                .gelu_backward_device(&upstream_handle, &x_handle, &x_shape)?;
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
            let erf_term = erff(value * INV_SQRT_2);
            let exp_term = (-0.5 * value * value).exp();
            let derivative = 0.5 * (1.0 + erf_term) + (value * INV_SQRT_2PI * exp_term);
            grad_out * derivative
        })
        .collect();
    let grad_id = store.alloc(Tensor::new(grad, input.shape, false)?);
    Ok(smallvec![(x, grad_id)])
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

    // Wave 2.1: route through `silu_backward_device` whenever upstream and
    // saved input are both device-resident.
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

    // Wave 2.1: route through `sigmoid_backward_device` whenever upstream
    // and saved output are both device-resident.
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
