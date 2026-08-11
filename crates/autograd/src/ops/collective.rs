use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::CommAxis,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{TensorId, TensorStore},
};

/// Differentiable all-reduce sum over the world group.
///
/// Forward computes `y = sum_rank(x_rank)` through the backend collective.
/// Backward applies the adjoint collective, so a loss evaluated on every rank
/// returns the gradient of the distributed total loss.
pub fn all_reduce_sum(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_device(x)?;
    let (shape, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "all_reduce_sum: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), handle)
    };

    let out_handle =
        store
            .backend()
            .all_reduce_sum_device(&input_handle, &shape, CommAxis::World)?;
    let output_id = store.alloc_device_tensor(shape.clone(), out_handle)?;

    TapeEntry {
        op: BackwardOp::AllReduceSum,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::Shape(shape),
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn all_reduce_sum_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("all_reduce_sum missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::Shape(shape) = &entry.saved else {
        return Err(AutogradError::TapeInvariant(
            "all_reduce_sum backward missing saved shape",
        ));
    };
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    if upstream_shape != *shape {
        return Err(AutogradError::ShapeMismatch {
            expected: shape.clone(),
            got: upstream_shape,
        });
    }

    store.ensure_device(output_grad_id)?;
    let upstream_handle = store
        .tensor(output_grad_id)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "all_reduce_sum backward: upstream missing device handle",
        ))?
        .clone();
    let grad_handle =
        store
            .backend()
            .all_reduce_sum_device(&upstream_handle, shape, CommAxis::World)?;
    let grad_id = store.alloc_device_tensor(shape.clone(), grad_handle)?;
    Ok(smallvec![(x, grad_id)])
}

/// Differentiable all-gather over the sequence axis (context parallelism).
///
/// Forward gathers each rank's local shard `[1, S/N, H]` into the full
/// `[1, S, H]`; `full_shape` is the gathered result. Backward reduce-scatters the
/// upstream grad back to this rank's shard (the adjoint of a gather). Single-rank
/// semantics are identity (`full_shape == local_shape`).
pub fn all_gather_seq(
    x: TensorId,
    full_shape: Vec<usize>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(x)?;
    let (local_shape, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "all_gather_seq: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), handle)
    };

    let out_handle =
        store
            .backend()
            .all_gather_seq_device(&input_handle, &local_shape, CommAxis::Seq)?;
    let output_id = store.alloc_device_tensor(full_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::AllGatherSeq,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::Shape(local_shape),
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn all_gather_seq_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("all_gather_seq missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }
    let SavedContext::Shape(local_shape) = &entry.saved else {
        return Err(AutogradError::TapeInvariant(
            "all_gather_seq backward missing saved shape",
        ));
    };
    let local_shape = local_shape.clone();

    store.ensure_device(output_grad_id)?;
    let upstream_handle = store
        .tensor(output_grad_id)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "all_gather_seq backward: upstream missing device handle",
        ))?
        .clone();
    let grad_handle =
        store
            .backend()
            .reduce_scatter_sum_device(&upstream_handle, &local_shape, CommAxis::Seq)?;
    let grad_id = store.alloc_device_tensor(local_shape, grad_handle)?;
    Ok(smallvec![(x, grad_id)])
}

/// Differentiable reduce-scatter sum over the sequence axis — the adjoint pair of
/// [`all_gather_seq`]. Forward sums the full `[1, S, H]` across ranks and keeps
/// this rank's `[1, S/N, H]` row slice; backward all-gathers the upstream grad
/// back to full. `local_shape` is this rank's output shard.
pub fn reduce_scatter_sum(
    x: TensorId,
    local_shape: Vec<usize>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(x)?;
    let (full_shape, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "reduce_scatter_sum: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), handle)
    };

    let out_handle =
        store
            .backend()
            .reduce_scatter_sum_device(&input_handle, &local_shape, CommAxis::Seq)?;
    let output_id = store.alloc_device_tensor(local_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::ReduceScatterSum,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::Shape(full_shape),
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn reduce_scatter_sum_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry.input_ids.first().ok_or(AutogradError::TapeInvariant(
        "reduce_scatter_sum missing input",
    ))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }
    let SavedContext::Shape(full_shape) = &entry.saved else {
        return Err(AutogradError::TapeInvariant(
            "reduce_scatter_sum backward missing saved shape",
        ));
    };
    let full_shape = full_shape.clone();

    store.ensure_device(output_grad_id)?;
    let upstream_handle = store
        .tensor(output_grad_id)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "reduce_scatter_sum backward: upstream missing device handle",
        ))?
        .clone();
    let grad_handle =
        store
            .backend()
            .all_gather_seq_device(&upstream_handle, &full_shape, CommAxis::Seq)?;
    let grad_id = store.alloc_device_tensor(full_shape, grad_handle)?;
    Ok(smallvec![(x, grad_id)])
}

/// Differentiable all-to-all that swaps a *scatter* axis for a *gather* axis —
/// the linear-attention CP transport (Megatron `MambaContextParallel`). Forward
/// splits `scatter_axis` across ranks and concatenates each rank's slice along
/// `gather_axis`: `[seq/N, b, hidden]` → `[seq, b, hidden/N]` (full sequence for
/// 1/N of the heads). It is self-adjoint with the two axes swapped, so backward
/// is the same op reversed. Single-rank / CPU is identity (`out_shape ==
/// in_shape`), which is the locally-verifiable core; the multi-rank NCCL shuffle
/// is pod-gated like the other collectives.
pub fn all_to_all(
    x: TensorId,
    scatter_axis: usize,
    gather_axis: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(x)?;
    let (in_shape, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "all_to_all: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), handle)
    };

    let (out_handle, out_shape) = store.backend().all_to_all_device(
        &input_handle,
        &in_shape,
        scatter_axis,
        gather_axis,
        CommAxis::Seq,
    )?;
    let output_id = store.alloc_device_tensor(out_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::AllToAll,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::AllToAllCtx {
            in_shape,
            scatter_axis,
            gather_axis,
        },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn all_to_all_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("all_to_all missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }
    let SavedContext::AllToAllCtx {
        in_shape,
        scatter_axis,
        gather_axis,
    } = &entry.saved
    else {
        return Err(AutogradError::TapeInvariant(
            "all_to_all backward missing saved context",
        ));
    };
    let (in_shape, scatter_axis, gather_axis) = (in_shape.clone(), *scatter_axis, *gather_axis);

    store.ensure_device(output_grad_id)?;
    let (upstream_handle, upstream_shape) = {
        let tensor = store.tensor(output_grad_id)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "all_to_all backward: upstream missing device handle",
            ))?
            .clone();
        (handle, tensor.shape.clone())
    };
    // Adjoint = the same shuffle with scatter/gather swapped, mapping the upstream
    // grad back to this rank's input shape.
    let (grad_handle, grad_shape) = store.backend().all_to_all_device(
        &upstream_handle,
        &upstream_shape,
        gather_axis,
        scatter_axis,
        CommAxis::Seq,
    )?;
    if grad_shape != in_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: in_shape,
            got: grad_shape,
        });
    }
    let grad_id = store.alloc_device_tensor(in_shape, grad_handle)?;
    Ok(smallvec![(x, grad_id)])
}
