use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn reshape(
    x: TensorId,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    if shape_numel(shape) != input.size {
        return Err(AutogradError::ShapeMismatch {
            expected: input.shape,
            got: shape.to_vec(),
        });
    }

    let output_id = store.alloc(Tensor::new(
        input.data,
        shape.to_vec(),
        input.requires_grad,
    )?);
    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Reshape,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::ReshapeCtx {
                input_shape: input.shape,
            },
        });
    }

    Ok(output_id)
}

pub fn transpose(
    x: TensorId,
    axis1: usize,
    axis2: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let (data, shape) = transpose_data(&input.data, &input.shape, axis1, axis2)?;
    let output_id = store.alloc(Tensor::new(data, shape, input.requires_grad)?);

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

pub fn slice(
    x: TensorId,
    starts: &[usize],
    ends: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let (data, shape) = slice_data(&input.data, &input.shape, starts, ends)?;
    let output_id = store.alloc(Tensor::new(data, shape, input.requires_grad)?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Slice,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::SliceCtx {
                input_shape: input.shape,
                starts: starts.to_vec(),
                ends: ends.to_vec(),
            },
        });
    }

    Ok(output_id)
}

pub(crate) fn reshape_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("reshape missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::ReshapeCtx { input_shape } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "reshape backward missing saved shape",
        ));
    };
    let upstream = store.tensor(output_grad_id)?.clone();
    if shape_numel(&input_shape) != upstream.size {
        return Err(AutogradError::ShapeMismatch {
            expected: input_shape,
            got: upstream.shape,
        });
    }

    let grad_id = store.alloc(Tensor::new(upstream.data, input_shape, false)?);
    Ok(smallvec![(x, grad_id)])
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
    let grad_id = store.alloc(Tensor::new(data, shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

pub(crate) fn slice_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("slice missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::SliceCtx {
        input_shape,
        starts,
        ends,
    } = entry.saved.clone()
    else {
        return Err(AutogradError::TapeInvariant(
            "slice backward missing saved bounds",
        ));
    };

    let upstream = store.tensor(output_grad_id)?.clone();
    let expected_shape: Vec<usize> = starts
        .iter()
        .zip(ends.iter())
        .map(|(&start, &end)| end - start)
        .collect();
    if upstream.shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream.shape,
        });
    }

    let input_strides = contiguous_strides(&input_shape);
    let mut grad = vec![0.0; shape_numel(&input_shape)];
    for (out_index, &grad_value) in upstream.data.iter().enumerate() {
        let out_coords = linear_to_coords(out_index, &upstream.shape);
        let input_index: usize = out_coords
            .iter()
            .enumerate()
            .map(|(axis, &coord)| (coord + starts[axis]) * input_strides[axis])
            .sum();
        grad[input_index] += grad_value;
    }

    let grad_id = store.alloc(Tensor::new(grad, input_shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

fn slice_data(
    data: &[f32],
    shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<(Vec<f32>, Vec<usize>)> {
    validate_slice_bounds(shape, starts, ends)?;
    let output_shape: Vec<usize> = starts
        .iter()
        .zip(ends.iter())
        .map(|(&start, &end)| end - start)
        .collect();
    let input_strides = contiguous_strides(shape);
    let mut output = vec![0.0; shape_numel(&output_shape)];
    for (out_index, slot) in output.iter_mut().enumerate() {
        let out_coords = linear_to_coords(out_index, &output_shape);
        let input_index: usize = out_coords
            .iter()
            .enumerate()
            .map(|(axis, &coord)| (coord + starts[axis]) * input_strides[axis])
            .sum();
        *slot = data[input_index];
    }
    Ok((output, output_shape))
}

fn validate_slice_bounds(shape: &[usize], starts: &[usize], ends: &[usize]) -> Result<()> {
    if starts.len() != shape.len() {
        return Err(AutogradError::InvalidIndicesLen {
            expected: shape.len(),
            got: starts.len(),
        });
    }
    if ends.len() != shape.len() {
        return Err(AutogradError::InvalidIndicesLen {
            expected: shape.len(),
            got: ends.len(),
        });
    }

    for ((&start, &end), &dim) in starts.iter().zip(ends.iter()).zip(shape.iter()) {
        if start > end {
            return Err(AutogradError::TapeInvariant(
                "slice start must be <= end for every axis",
            ));
        }
        if end > dim {
            return Err(AutogradError::IndexOutOfBounds {
                index: end,
                upper: dim,
            });
        }
        if start > dim {
            return Err(AutogradError::IndexOutOfBounds {
                index: start,
                upper: dim,
            });
        }
    }

    Ok(())
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

fn shape_numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}
