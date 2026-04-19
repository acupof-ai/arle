use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::broadcast_offset,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn add_broadcast(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let a_tensor = store.tensor(a)?.clone();
    let b_tensor = store.tensor(b)?.clone();

    let output = store.backend().add_broadcast_forward(
        &a_tensor.data,
        &a_tensor.shape,
        &b_tensor.data,
        &b_tensor.shape,
    )?;

    let requires_grad = a_tensor.requires_grad || b_tensor.requires_grad;
    let output_id = store.alloc(Tensor::new(output, a_tensor.shape.clone(), requires_grad)?);
    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::AddBroadcast,
            output_id,
            input_ids: smallvec![a, b],
            saved: SavedContext::AddBroadcastCtx {
                a_shape: a_tensor.shape,
                b_shape: b_tensor.shape,
            },
        });
    }

    Ok(output_id)
}

pub(crate) fn add_broadcast_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let a = *entry.input_ids.first().ok_or(AutogradError::TapeInvariant(
        "add_broadcast missing lhs input",
    ))?;
    let b = *entry.input_ids.get(1).ok_or(AutogradError::TapeInvariant(
        "add_broadcast missing rhs input",
    ))?;

    let SavedContext::AddBroadcastCtx { a_shape, b_shape } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "add_broadcast backward missing saved shapes",
        ));
    };
    let upstream = store.tensor(output_grad_id)?.clone();
    if upstream.shape != a_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape.clone(),
            got: upstream.shape,
        });
    }

    let mut grads = GradPairs::new();
    if store.tensor(a)?.requires_grad {
        let grad_id = store.alloc(Tensor::new(upstream.data.clone(), a_shape, false)?);
        grads.push((a, grad_id));
    }

    if store.tensor(b)?.requires_grad {
        let b_size = if b_shape.is_empty() {
            1
        } else {
            b_shape.iter().product()
        };
        let mut grad_b = vec![0.0; b_size];
        for (index, grad_value) in upstream.data.iter().enumerate() {
            let offset = broadcast_offset(index, &entry.output_id_shape(store)?, &b_shape);
            grad_b[offset] += *grad_value;
        }
        let grad_id = store.alloc(Tensor::new(grad_b, b_shape, false)?);
        grads.push((b, grad_id));
    }

    Ok(grads)
}

trait OutputShapeExt {
    fn output_id_shape(&self, store: &TensorStore) -> Result<Vec<usize>>;
}

impl OutputShapeExt for TapeEntry {
    fn output_id_shape(&self, store: &TensorStore) -> Result<Vec<usize>> {
        Ok(store.tensor(self.output_id)?.shape.clone())
    }
}
