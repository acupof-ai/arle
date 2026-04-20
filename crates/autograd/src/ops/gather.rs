// Index side-channel: indices stored as Vec<usize> in SavedContext, not in TensorStore.
// Avoids infrastructure sprawl (Option A).

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn gather_last_dim(
    src: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // M5.3b.9: dispatch on device-handle presence (same gate as rope /
    // rmsnorm / embedding — `device_handle.is_some() && dirty != Host`).
    // `gather_last_dim` is the final op in the CE-loss path: logits come
    // straight out of the output matmul, so staying on-device through the
    // per-row gather keeps the entire forward lazy. Dirty::Host inputs
    // take the eager host path for parity.
    let has_device_handle = {
        let t = store.tensor(src)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    if has_device_handle {
        gather_last_dim_device_lazy(src, indices, store, tape)
    } else {
        gather_last_dim_host_eager(src, indices, store, tape)
    }
}

fn gather_last_dim_device_lazy(
    src: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(src)?;

    let (src_shape, requires_grad) = {
        let t = store.tensor(src)?;
        (t.shape.clone(), t.requires_grad)
    };
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("shape checked above");
    let output_shape = src_shape[..src_shape.len() - 1].to_vec();
    let prefix_elems = if output_shape.is_empty() {
        1
    } else {
        output_shape.iter().product()
    };
    if indices.len() != prefix_elems {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix_elems,
            got: indices.len(),
        });
    }
    // Bounds-check once here; the lazy helper re-checks but the early
    // error carries the original `usize` index the caller passed.
    for &index in indices {
        if index >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index,
                upper: vocab,
            });
        }
    }

    let ids_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let src_handle = store
        .tensor(src)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "gather_last_dim: ensure_device left src without a device handle",
        ))?
        .clone();

    let out_handle = store
        .backend()
        .gather_last_dim(&src_handle, &src_shape, &ids_i32)?;
    let output_id = store.alloc_device_tensor(output_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Gather,
            output_id,
            input_ids: smallvec![src],
            saved: SavedContext::GatherCtx {
                indices: indices.to_vec(),
                src_shape,
            },
        });
    }

    Ok(output_id)
}

fn gather_last_dim_host_eager(
    src: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let src_tensor = store.tensor(src)?.clone();
    if src_tensor.shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }

    let vocab = *src_tensor.shape.last().expect("shape checked above");
    let output_shape = src_tensor.shape[..src_tensor.shape.len() - 1].to_vec();
    let prefix_elems = if output_shape.is_empty() {
        1
    } else {
        output_shape.iter().product()
    };
    if indices.len() != prefix_elems {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix_elems,
            got: indices.len(),
        });
    }

    // Bounds-check here so the error surfaces the original `usize` index
    // (the CUDA kernel zero-fills on OOB to keep the device path branch-free).
    for &index in indices {
        if index >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index,
                upper: vocab,
            });
        }
    }
    let ids_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let output =
        store
            .backend()
            .gather_last_dim_forward(&src_tensor.data, &src_tensor.shape, &ids_i32)?;
    debug_assert_eq!(output.len(), prefix_elems);

    let output_id = store.alloc(Tensor::new(output, output_shape, src_tensor.requires_grad)?);
    if src_tensor.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Gather,
            output_id,
            input_ids: smallvec![src],
            saved: SavedContext::GatherCtx {
                indices: indices.to_vec(),
                src_shape: src_tensor.shape,
            },
        });
    }

    Ok(output_id)
}

pub(crate) fn gather_last_dim_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let src = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("gather missing input"))?;
    if !store.tensor(src)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::GatherCtx { indices, src_shape } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "gather backward missing saved context",
        ));
    };
    let upstream = store.tensor(output_grad_id)?.clone();
    let output_shape = src_shape[..src_shape.len() - 1].to_vec();
    if upstream.shape != output_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: output_shape,
            got: upstream.shape,
        });
    }

    let vocab = *src_shape.last().ok_or(AutogradError::TapeInvariant(
        "gather missing source last dim",
    ))?;
    // Gather's backward is a per-prefix point-write into a `[prefix_rows, vocab]`
    // buffer. Each `(prefix_index, original_index)` pair is unique, so we can
    // flatten the target to `[prefix_rows * vocab]` and dispatch a single
    // scatter-add with remapped indices `i * vocab + original_indices[i]`.
    // One trait call, one GPU launch — the same shape the old inline loop wrote.
    let prefix_rows = indices.len();
    let flat_vocab = prefix_rows * vocab;
    let flat_ids: Vec<i32> = indices
        .iter()
        .enumerate()
        .map(|(i, &index)| (i * vocab + index) as i32)
        .collect();
    let grad = store.backend().scatter_add_rows_forward(
        &upstream.data,
        prefix_rows,
        1,
        &flat_ids,
        flat_vocab,
    )?;
    debug_assert_eq!(grad.len(), src_shape.iter().product::<usize>());

    let grad_id = store.alloc(Tensor::new(grad, src_shape, false)?);
    Ok(smallvec![(src, grad_id)])
}
