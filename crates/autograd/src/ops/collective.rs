use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::CommAxis,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{TensorId, TensorStore},
};

/// Differentiable all-reduce sum over the world group.
///
/// Forward is `y = sum_rank(x_rank)`. Backward applies the adjoint collective,
/// so a loss evaluated on every rank returns the gradient of the distributed
/// total loss.
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
/// `[1, S, H]` (`full_shape`). Backward reduce-scatters the upstream grad
/// back to this rank's shard (the adjoint of a gather). Single-rank is
/// identity (`full_shape == local_shape`).
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
