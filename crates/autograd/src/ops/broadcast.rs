use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{GpuTensor, TensorId, TensorStore},
};

pub fn add_broadcast(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let a_tensor = store.tensor(a)?.clone();
    let b_tensor = store.tensor(b)?.clone();
    validate_broadcast(&a_tensor.shape, &b_tensor.shape)?;

    let mut output = vec![0.0; a_tensor.size];
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = a_tensor.data[index]
            + b_tensor.data[broadcast_offset(index, &a_tensor.shape, &b_tensor.shape)];
    }

    let requires_grad = a_tensor.requires_grad || b_tensor.requires_grad;
    let output_id = store.alloc(GpuTensor::new(
        output,
        a_tensor.shape.clone(),
        requires_grad,
    )?);
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
        let grad_id = store.alloc(GpuTensor::new(upstream.data.clone(), a_shape, false)?);
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
        let grad_id = store.alloc(GpuTensor::new(grad_b, b_shape, false)?);
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

fn validate_broadcast(a_shape: &[usize], b_shape: &[usize]) -> Result<()> {
    if b_shape.len() > a_shape.len() {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        });
    }

    let rank_offset = a_shape.len() - b_shape.len();
    for (index, &dim) in b_shape.iter().enumerate() {
        let target = a_shape[rank_offset + index];
        if dim != 1 && dim != target {
            return Err(AutogradError::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            });
        }
    }

    Ok(())
}

fn broadcast_offset(out_index: usize, out_shape: &[usize], b_shape: &[usize]) -> usize {
    if b_shape.is_empty() {
        return 0;
    }

    let coords = linear_to_coords(out_index, out_shape);
    let rank_offset = out_shape.len() - b_shape.len();
    let b_strides = contiguous_strides(b_shape);
    let mut offset = 0usize;
    for (index, stride) in b_strides.iter().enumerate() {
        let coord = if b_shape[index] == 1 {
            0
        } else {
            coords[rank_offset + index]
        };
        offset += coord * stride;
    }
    offset
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (index, dim) in shape.iter().enumerate().rev() {
        strides[index] = stride;
        stride *= *dim;
    }
    strides
}

fn linear_to_coords(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut coords = vec![0; shape.len()];
    for index in (0..shape.len()).rev() {
        let dim = shape[index];
        coords[index] = linear % dim;
        linear /= dim;
    }
    coords
}
