use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.2: route Dirty::Device inputs through the lazy
    // `backend.softmax_last_axis` (composes `mlx_softmax_axis` into the MLX
    // graph with no eval). Dirty::Host / Dirty::Both stay on the host fast
    // path so host-resident producers don't pay an upload+device-reduce
    // +readback. Mirrors the M5.3b.1 `sum` dispatch shape.
    let dirty = store.tensor(x)?.dirty.clone();
    match dirty {
        Dirty::Device => softmax_device_lazy(x, store, tape, SoftmaxKind::Softmax),
        Dirty::Host | Dirty::Both => softmax_host_eager(x, store, tape, SoftmaxKind::Softmax),
    }
}

pub fn log_softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // See `softmax` for the dispatch rationale (M5.3b.2).
    let dirty = store.tensor(x)?.dirty.clone();
    match dirty {
        Dirty::Device => softmax_device_lazy(x, store, tape, SoftmaxKind::LogSoftmax),
        Dirty::Host | Dirty::Both => softmax_host_eager(x, store, tape, SoftmaxKind::LogSoftmax),
    }
}

#[derive(Copy, Clone)]
enum SoftmaxKind {
    Softmax,
    LogSoftmax,
}

impl SoftmaxKind {
    fn backward_op(self) -> BackwardOp {
        match self {
            SoftmaxKind::Softmax => BackwardOp::Softmax,
            SoftmaxKind::LogSoftmax => BackwardOp::LogSoftmax,
        }
    }

    fn saved(self, y: TensorId) -> SavedContext {
        match self {
            SoftmaxKind::Softmax => SavedContext::SoftmaxCtx { y },
            SoftmaxKind::LogSoftmax => SavedContext::LogSoftmaxCtx { y },
        }
    }
}

fn softmax_device_lazy(
    x: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
    kind: SoftmaxKind,
) -> Result<TensorId> {
    // Defensive `ensure_device`: caller already routed a Dirty::Device
    // tensor, but re-calling guards a future Dirty::Both path from silent
    // drift (mirrors `sum_device_lazy`).
    store.ensure_device(x)?;
    let (input_shape, requires_grad) = {
        let tensor = store.tensor(x)?;
        let _ = last_dim(&tensor.shape)?;
        (tensor.shape.clone(), tensor.requires_grad)
    };
    let input_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "softmax: ensure_device left tensor without a device handle",
        ))?
        .clone();

    let out_handle = match kind {
        SoftmaxKind::Softmax => store
            .backend()
            .softmax_last_axis(&input_handle, &input_shape)?,
        SoftmaxKind::LogSoftmax => store
            .backend()
            .log_softmax_last_axis(&input_handle, &input_shape)?,
    };
    let output_id = store.alloc_device_tensor(input_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: kind.backward_op(),
            output_id,
            input_ids: smallvec![x],
            saved: kind.saved(output_id),
        });
    }

    Ok(output_id)
}

fn softmax_host_eager(
    x: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
    kind: SoftmaxKind,
) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let _ = last_dim(&input.shape)?;
    let output = match kind {
        SoftmaxKind::Softmax => store
            .backend()
            .softmax_forward_last_axis(&input.data, &input.shape)?,
        SoftmaxKind::LogSoftmax => store
            .backend()
            .log_softmax_forward_last_axis(&input.data, &input.shape)?,
    };

    let output_id = store.alloc(Tensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);
    if input.requires_grad {
        tape.record(TapeEntry {
            op: kind.backward_op(),
            output_id,
            input_ids: smallvec![x],
            saved: kind.saved(output_id),
        });
    }

    Ok(output_id)
}

pub(crate) fn softmax_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("softmax missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::SoftmaxCtx { y } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "softmax backward missing saved output",
        ));
    };
    let output = store.tensor(y)?.clone();
    let upstream = store.tensor(output_grad_id)?.clone();
    if output.shape != upstream.shape {
        return Err(AutogradError::ShapeMismatch {
            expected: output.shape,
            got: upstream.shape,
        });
    }

    // dL/dx = y * (dL/dy - sum(dL/dy * y, axis=-1, keepdim))
    // Stream row-wise so we only allocate the output buffer — full-vocab logits
    // on training paths make intermediate `mul`/`sub` materializations cost
    // 3-4× peak memory (codex review 2026-04-19).
    let last = last_dim(&output.shape)?;
    let rows = output.data.len() / last;
    let mut grad = vec![0.0_f32; output.data.len()];
    for row in 0..rows {
        let base = row * last;
        let mut dot = 0.0_f32;
        for col in 0..last {
            dot += upstream.data[base + col] * output.data[base + col];
        }
        for col in 0..last {
            grad[base + col] = output.data[base + col] * (upstream.data[base + col] - dot);
        }
    }

    let grad_id = store.alloc(Tensor::new(grad, output.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

pub(crate) fn log_softmax_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("log_softmax missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::LogSoftmaxCtx { y } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "log_softmax backward missing saved output",
        ));
    };

    // Shape check uses borrowed access only — `.clone()` on a
    // `Dirty::Device` tensor would panic, and the device-aware fast path
    // (below) does not need a clone.
    let output_shape = store.tensor(y)?.shape.clone();
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    if output_shape != upstream_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: output_shape,
            got: upstream_shape,
        });
    }

    // Wave 1 (post-M5.3b nsys attribution): fast-path the backward when
    // BOTH the saved forward output and the upstream gradient are still
    // device-resident. This is the production CUDA path: the saved
    // `LogSoftmaxCtx { y }` is produced by `softmax_device_lazy` on a
    // `Dirty::Device` tensor, and the upstream gradient flows from
    // `gather_last_dim_backward`'s device override. Skipping the host
    // round-trip here kills the `[B, S, V] × 4 B ≈ 1 GB` DtoH that nsys
    // identified as the single largest readback per training step (see
    // `docs/research/2026-05-17-cuda-training-step-nsys-attribution.md`).
    let saved_on_device =
        store.tensor(y)?.dirty == Dirty::Device && store.tensor(y)?.device_handle.is_some();
    let upstream_on_device = store.tensor(output_grad_id)?.dirty == Dirty::Device
        && store.tensor(output_grad_id)?.device_handle.is_some();
    if saved_on_device && upstream_on_device {
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let output_handle = store
            .tensor(y)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle = store.backend().log_softmax_last_axis_backward(
            &upstream_handle,
            &output_handle,
            &output_shape,
        )?;
        let grad_id = store.alloc_device_tensor(output_shape, grad_handle)?;
        return Ok(smallvec![(x, grad_id)]);
    }

    // Host-eager fallback: any backend (CPU/Metal) plus any case where
    // the upstream or saved tensor is already host-resident. Mirrors the
    // pre-Wave-1 reference and stays in lock-step with
    // `cpu_log_softmax_backward` so device + host produce byte-identical
    // grads up to fp rounding.
    let output = store.tensor(y)?.clone();
    let upstream = store.tensor(output_grad_id)?.clone();
    let last = last_dim(&output.shape)?;
    let rows = output.data.len() / last;
    let mut grad = vec![0.0_f32; output.data.len()];
    for row in 0..rows {
        let base = row * last;
        let mut sum_grad = 0.0_f32;
        for col in 0..last {
            sum_grad += upstream.data[base + col];
        }
        for col in 0..last {
            grad[base + col] = upstream.data[base + col] - output.data[base + col].exp() * sum_grad;
        }
    }

    let grad_id = store.alloc(Tensor::new(grad, output.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

fn last_dim(shape: &[usize]) -> Result<usize> {
    shape.last().copied().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })
}
