use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{GpuTensor, TensorId, TensorStore},
};

pub fn transpose(
    x: TensorId,
    axis1: usize,
    axis2: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let (data, shape) = transpose_data(&input.data, &input.shape, axis1, axis2)?;
    let output_id = store.alloc(GpuTensor::new(data, shape, input.requires_grad)?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Transpose,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::TransposeCtx { axis1, axis2 },
        });
    }

    Ok(output_id)
}

pub(crate) fn transpose_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("transpose missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::TransposeCtx { axis1, axis2 } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "transpose backward missing saved axes",
        ));
    };
    let upstream = store.tensor(output_grad_id)?.clone();
    let (data, shape) = transpose_data(&upstream.data, &upstream.shape, axis1, axis2)?;
    let grad_id = store.alloc(GpuTensor::new(data, shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

fn transpose_data(
    data: &[f32],
    shape: &[usize],
    axis1: usize,
    axis2: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let rank = shape.len();
    if axis1 >= rank {
        return Err(AutogradError::AxisOutOfBounds { axis: axis1, rank });
    }
    if axis2 >= rank {
        return Err(AutogradError::AxisOutOfBounds { axis: axis2, rank });
    }
    if axis1 == axis2 {
        return Ok((data.to_vec(), shape.to_vec()));
    }

    let mut output_shape = shape.to_vec();
    output_shape.swap(axis1, axis2);
    let input_strides = contiguous_strides(shape);
    let mut output = vec![0.0; data.len()];
    for (out_index, slot) in output.iter_mut().enumerate() {
        let mut coords = linear_to_coords(out_index, &output_shape);
        coords.swap(axis1, axis2);
        let input_index: usize = coords
            .iter()
            .zip(input_strides.iter())
            .map(|(coord, stride)| coord * stride)
            .sum();
        *slot = data[input_index];
    }
    Ok((output, output_shape))
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
