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
}
