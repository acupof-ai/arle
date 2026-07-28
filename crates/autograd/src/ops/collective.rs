use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{TensorId, TensorStore},
};

/// Differentiable all-reduce sum.
///
/// Forward computes `y = sum_rank(x_rank)` through the backend collective.
/// Backward applies the adjoint collective, so a loss evaluated on every rank
/// returns the gradient of the distributed total loss.
pub fn all_reduce_sum(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_device(x)?;
    let (shape, requires_grad, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "all_reduce_sum: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), tensor.requires_grad, handle)
    };

    let out_handle = store
        .backend()
        .all_reduce_sum_device(&input_handle, &shape)?;
    let output_id = store.alloc_device_tensor(shape.clone(), out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::AllReduceSum,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::Shape(shape),
        });
    }

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
    let grad_handle = store
        .backend()
        .all_reduce_sum_device(&upstream_handle, shape)?;
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
    let (local_shape, requires_grad, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "all_gather_seq: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), tensor.requires_grad, handle)
    };

    let out_handle = store
        .backend()
        .all_gather_seq_device(&input_handle, &local_shape)?;
    let output_id = store.alloc_device_tensor(full_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::AllGatherSeq,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::Shape(local_shape),
        });
    }

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
    let grad_handle = store
        .backend()
        .reduce_scatter_sum_device(&upstream_handle, &local_shape)?;
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
    let (full_shape, requires_grad, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "reduce_scatter_sum: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), tensor.requires_grad, handle)
    };

    let out_handle = store
        .backend()
        .reduce_scatter_sum_device(&input_handle, &local_shape)?;
    let output_id = store.alloc_device_tensor(local_shape, out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::ReduceScatterSum,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::Shape(full_shape),
        });
    }

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
    let grad_handle = store
        .backend()
        .all_gather_seq_device(&upstream_handle, &full_shape)?;
    let grad_id = store.alloc_device_tensor(full_shape, grad_handle)?;
    Ok(smallvec![(x, grad_id)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tensor, ops, tape::Tape, tensor::TensorStore};

    #[test]
    fn all_reduce_sum_single_rank_forward_backward_is_identity() -> Result<()> {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let x = store.alloc(Tensor::new(vec![1.0, -2.0, 3.5], vec![3], true)?);
        let y = all_reduce_sum(x, &mut store, &mut tape)?;
        let yy = ops::mul(y, y, &mut store, &mut tape)?;
        let loss = ops::sum(yy, &mut store, &mut tape)?;

        let grads = tape.backward(loss, &mut store)?;
        let grad_id = grads[&x];
        let grad = store.to_host(grad_id)?;

        assert_eq!(store.to_host(y)?, vec![1.0, -2.0, 3.5]);
        assert_eq!(grad, vec![2.0, -4.0, 7.0]);
        Ok(())
    }

    // Single-rank (world==1) identity: all_gather_seq / reduce_scatter_sum are
    // no-ops on shape and value, and the pair is a correct adjoint. Tier-1 CP
    // parity gate, runnable locally on CPU before any NCCL wiring (full==local).
    #[test]
    fn all_gather_seq_single_rank_is_identity() -> Result<()> {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let x = store.alloc(Tensor::new(vec![1.0, -2.0, 3.5, 4.0], vec![1, 4, 1], true)?);
        let y = all_gather_seq(x, vec![1, 4, 1], &mut store, &mut tape)?;
        let yy = ops::mul(y, y, &mut store, &mut tape)?;
        let loss = ops::sum(yy, &mut store, &mut tape)?;

        let grads = tape.backward(loss, &mut store)?;
        assert_eq!(store.to_host(y)?, vec![1.0, -2.0, 3.5, 4.0]);
        assert_eq!(store.to_host(grads[&x])?, vec![2.0, -4.0, 7.0, 8.0]);
        Ok(())
    }

    #[test]
    fn reduce_scatter_sum_single_rank_is_identity() -> Result<()> {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let x = store.alloc(Tensor::new(vec![2.0, 5.0, -1.0], vec![1, 3, 1], true)?);
        let y = reduce_scatter_sum(x, vec![1, 3, 1], &mut store, &mut tape)?;
        let loss = ops::sum(y, &mut store, &mut tape)?;

        let grads = tape.backward(loss, &mut store)?;
        assert_eq!(store.to_host(y)?, vec![2.0, 5.0, -1.0]);
        assert_eq!(store.to_host(grads[&x])?, vec![1.0, 1.0, 1.0]);
        Ok(())
    }
}
