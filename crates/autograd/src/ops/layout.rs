use smallvec::smallvec;

use crate::{
    AutogradError, DeviceHandle, Result,
    backend::{Device, broadcast_offset, validate_broadcast},
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn reshape(
    x: TensorId,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Dispatch on device-handle presence (same gate as rope /
    // embedding / gather / AdamW). Reshape is free on MLX — `mlx_reshape`
    // is metadata-only — so taking the lazy branch when x is device-
    // resident keeps the whole forward chain on-device. Qwen3.5 hits this
    // ~6× per attention layer (q/k/v projection + attn-out reshape) × 28
    // layers = ~168 evals/step.
    let has_device_handle = {
        let t = store.tensor(x)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    if has_device_handle {
        reshape_device_lazy(x, shape, store, tape)
    } else {
        reshape_host_eager(x, shape, store, tape)
    }
}

fn reshape_device_lazy(
    x: TensorId,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(x)?;

    let input_shape = store.tensor(x)?.shape.clone();
    if shape_numel(shape) != shape_numel(&input_shape) {
        return Err(AutogradError::ShapeMismatch {
            expected: input_shape,
            got: shape.to_vec(),
        });
    }

    let x_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "reshape: ensure_device left x without a device handle",
        ))?
        .clone();

    let out_handle = store.backend().reshape(&x_handle, shape)?;
    let output_id = store.alloc_device_tensor(shape.to_vec(), out_handle)?;

    TapeEntry {
        op: BackwardOp::Reshape,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::ReshapeCtx { input_shape },
    }
    .record(store, tape)?;

    Ok(output_id)
}

fn reshape_host_eager(
    x: TensorId,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let input = store.tensor_host(x)?;
    if shape_numel(shape) != input.size {
        return Err(AutogradError::ShapeMismatch {
            expected: input.shape,
            got: shape.to_vec(),
        });
    }

    let output_id = store.alloc(Tensor::new(input.data, shape.to_vec(), false)?);
    TapeEntry {
        op: BackwardOp::Reshape,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::ReshapeCtx {
            input_shape: input.shape,
        },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub fn transpose(
    x: TensorId,
    axis1: usize,
    axis2: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Same gate as `reshape`. `mlx_transpose_axes` is a lazy
    // view op — MLX fuses it into downstream GEMMs, so we pay nothing
    // for staying on device. Qwen3.5 q/k/v projections transpose once
    // each × 3 × 28 layers = 84 evals/step.
    let has_device_handle = {
        let t = store.tensor(x)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    if has_device_handle {
        transpose_device_lazy(x, axis1, axis2, store, tape)
    } else {
        transpose_host_eager(x, axis1, axis2, store, tape)
    }
}

fn transpose_device_lazy(
    x: TensorId,
    axis1: usize,
    axis2: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(x)?;

    let input_shape = store.tensor(x)?.shape.clone();
    let rank = input_shape.len();
    if axis1 >= rank {
        return Err(AutogradError::AxisOutOfBounds { axis: axis1, rank });
    }
    if axis2 >= rank {
        return Err(AutogradError::AxisOutOfBounds { axis: axis2, rank });
    }

    let x_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "transpose: ensure_device left x without a device handle",
        ))?
        .clone();

    let (out_handle, new_shape) =
        store
            .backend()
            .transpose_axes_swap(&x_handle, &input_shape, axis1, axis2)?;
    let output_id = store.alloc_device_tensor(new_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::Transpose,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::TransposeCtx { axis1, axis2 },
    }
    .record(store, tape)?;

    Ok(output_id)
}

fn transpose_host_eager(
    x: TensorId,
    axis1: usize,
    axis2: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let input = store.tensor_host(x)?;
    let (data, shape) = transpose_data(&input.data, &input.shape, axis1, axis2)?;
    let output_id = store.alloc(Tensor::new(data, shape, input.requires_grad)?);

    TapeEntry {
        op: BackwardOp::Transpose,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::TransposeCtx { axis1, axis2 },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub fn slice(
    x: TensorId,
    starts: &[usize],
    ends: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Dispatch on device-handle presence (same gate as reshape /
    // transpose / rope / embedding). `mlx_slice` is lazy on Metal — fuses
    // with downstream matmul. Qwen3.5 hits this 2× per attention layer
    // (q/gate split from the fused q_full projection) × 28 layers = 56
    // evals/step.
    let has_device_handle = {
        let t = store.tensor(x)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    if has_device_handle {
        slice_device_lazy(x, starts, ends, store, tape)
    } else {
        slice_host_eager(x, starts, ends, store, tape)
    }
}

fn slice_device_lazy(
    x: TensorId,
    starts: &[usize],
    ends: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(x)?;

    let input_shape = store.tensor(x)?.shape.clone();
    let rank = input_shape.len();
    if starts.len() != rank {
        return Err(AutogradError::InvalidIndicesLen {
            expected: rank,
            got: starts.len(),
        });
    }
    if ends.len() != rank {
        return Err(AutogradError::InvalidIndicesLen {
            expected: rank,
            got: ends.len(),
        });
    }
    for ((&start, &end), &dim) in starts.iter().zip(ends.iter()).zip(input_shape.iter()) {
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
    }
    let new_shape: Vec<usize> = starts
        .iter()
        .zip(ends.iter())
        .map(|(&s, &e)| e - s)
        .collect();

    let x_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "slice: ensure_device left x without a device handle",
        ))?
        .clone();

    let out_handle = store
        .backend()
        .slice(&x_handle, &input_shape, starts, ends)?;
    let output_id = store.alloc_device_tensor(new_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::Slice,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::SliceCtx {
            input_shape,
            starts: starts.to_vec(),
            ends: ends.to_vec(),
        },
    }
    .record(store, tape)?;

    Ok(output_id)
}

fn slice_host_eager(
    x: TensorId,
    starts: &[usize],
    ends: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(x)?;
    // Borrow, not `tensor_host`: cloning the full source costs a full-seq memcpy
    // per chunk in the offloaded replay.
    let input = store.tensor(x)?;
    let (data, shape) = slice_data(&input.data, &input.shape, starts, ends)?;
    let input_shape = input.shape.clone();
    let requires_grad = input.requires_grad;
    let output_id = store.alloc(Tensor::new(data, shape, requires_grad)?);

    TapeEntry {
        op: BackwardOp::Slice,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::SliceCtx {
            input_shape,
            starts: starts.to_vec(),
            ends: ends.to_vec(),
        },
    }
    .record(store, tape)?;

    Ok(output_id)
}

/// Concatenate N tensors of equal rank along `axis`. All inputs must match on
/// every axis except `axis`. General over rank and axis (unlike the rank-4-only
/// `cat_seq`/`cat_heads`): the CP linear-attn transport concats rank-3 activations
/// on the feature axis and the rank-2 packed conv weight on axis 0. Host path —
/// this is layout glue, and the CP shuffle it feeds is pod-only regardless.
pub fn cat(
    inputs: &[TensorId],
    axis: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if inputs.is_empty() {
        return Err(AutogradError::InvalidIndicesLen {
            expected: 1,
            got: 0,
        });
    }
    if inputs.len() == 1 {
        return Ok(inputs[0]);
    }
    let first = store.tensor(inputs[0])?.shape.clone();
    if axis >= first.len() {
        return Err(AutogradError::AxisOutOfBounds {
            axis,
            rank: first.len(),
        });
    }
    let input_shapes: Vec<Vec<usize>> = inputs
        .iter()
        .map(|&id| {
            let s = store.tensor(id)?.shape.clone();
            if s.len() != first.len()
                || s.iter()
                    .enumerate()
                    .any(|(a, &d)| a != axis && d != first[a])
            {
                return Err(AutogradError::ShapeMismatch {
                    expected: first.clone(),
                    got: s.clone(),
                });
            }
            Ok(s)
        })
        .collect::<Result<_>>()?;

    // Device-lazy when every input is device-resident (same dispatch as slice),
    // else host — a host round-trip here dominates the 256K CP step.
    let all_device = inputs.iter().all(|&id| {
        store
            .tensor(id)
            .is_ok_and(|t| t.dirty != Dirty::Host && t.device_handle.is_some())
    });
    let output_id = if all_device {
        let handles: Vec<DeviceHandle> = inputs
            .iter()
            .map(|&id| store.device_handle(id))
            .collect::<Result<_>>()?;
        let parts: Vec<(&DeviceHandle, &[usize])> = handles
            .iter()
            .zip(input_shapes.iter())
            .map(|(h, s)| (h, s.as_slice()))
            .collect();
        let (out_handle, produced) = store.backend().concat(&parts, axis)?;
        store.alloc_device_tensor(produced, out_handle)?
    } else {
        let outer: usize = first[..axis].iter().product();
        let inner: usize = first[axis + 1..].iter().product();
        let axis_total: usize = input_shapes.iter().map(|s| s[axis]).sum();
        let requires_grad = inputs
            .iter()
            .any(|&id| store.tensor(id).is_ok_and(|t| t.requires_grad));
        let mut out_shape = first.clone();
        out_shape[axis] = axis_total;

        let mut data = vec![0.0_f32; outer * axis_total * inner];
        let mut axis_off = 0usize;
        for (i, &id) in inputs.iter().enumerate() {
            let axis_i = input_shapes[i][axis];
            let src = store.tensor_host(id)?;
            for o in 0..outer {
                let src_base = o * axis_i * inner;
                let dst_base = (o * axis_total + axis_off) * inner;
                let len = axis_i * inner;
                data[dst_base..dst_base + len].copy_from_slice(&src.data[src_base..src_base + len]);
            }
            axis_off += axis_i;
        }
        store.alloc(Tensor::new(data, out_shape, requires_grad)?)
    };

    TapeEntry {
        op: BackwardOp::Cat,
        output_id,
        input_ids: inputs.iter().copied().collect(),
        saved: SavedContext::CatCtx { axis, input_shapes },
    }
    .record(store, tape)?;
    Ok(output_id)
}

pub(crate) fn cat_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::CatCtx { axis, input_shapes } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "cat backward missing saved context",
        ));
    };
    let axis_total: usize = input_shapes.iter().map(|s| s[axis]).sum();

    // Each input's grad is the upstream sliced to its axis range — on-device when
    // the upstream is device-resident, else host (same round-trip concern as forward).
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        upstream.dirty != Dirty::Host && upstream.device_handle.is_some()
    };
    if device_path_ok {
        let mut full_shape = input_shapes[0].clone();
        full_shape[axis] = axis_total;
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let mut grads = GradPairs::new();
        let mut axis_off = 0usize;
        for (i, &input_id) in entry.input_ids.iter().enumerate() {
            let axis_i = input_shapes[i][axis];
            if store.tensor(input_id)?.requires_grad {
                let mut starts = vec![0usize; full_shape.len()];
                let mut ends = full_shape.clone();
                starts[axis] = axis_off;
                ends[axis] = axis_off + axis_i;
                let grad_handle =
                    store
                        .backend()
                        .slice(&upstream_handle, &full_shape, &starts, &ends)?;
                let grad_id = store.alloc_device_tensor(input_shapes[i].clone(), grad_handle)?;
                grads.push((input_id, grad_id));
            }
            axis_off += axis_i;
        }
        return Ok(grads);
    }

    let upstream = store.tensor_host(output_grad_id)?.data.clone();
    let inner: usize = input_shapes[0][axis + 1..].iter().product();
    let outer: usize = input_shapes[0][..axis].iter().product();

    let mut grads = GradPairs::new();
    let mut axis_off = 0usize;
    for (i, &input_id) in entry.input_ids.iter().enumerate() {
        let axis_i = input_shapes[i][axis];
        if store.tensor(input_id)?.requires_grad {
            let mut grad = vec![0.0_f32; outer * axis_i * inner];
            for o in 0..outer {
                let src_base = (o * axis_total + axis_off) * inner;
                let dst_base = o * axis_i * inner;
                let len = axis_i * inner;
                grad[dst_base..dst_base + len].copy_from_slice(&upstream[src_base..src_base + len]);
            }
            let grad_id = store.alloc(Tensor::new(grad, input_shapes[i].clone(), false)?);
            grads.push((input_id, grad_id));
        }
        axis_off += axis_i;
    }
    Ok(grads)
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
    // Heal a host-resident upstream grad before the gate (mirrors
    // matmul_backward) so one host grad upstream doesn't demote this reshape —
    // and everything downstream — to the host contract with a full-buffer clone.
    if store.backend().device() != Device::Cpu {
        store.ensure_device(output_grad_id)?;
    }
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        upstream.dirty != Dirty::Host && upstream.device_handle.is_some()
    };
    if device_path_ok {
        let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
        if shape_numel(&input_shape) != shape_numel(&upstream_shape) {
            return Err(AutogradError::ShapeMismatch {
                expected: input_shape,
                got: upstream_shape,
            });
        }
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle = store.backend().reshape(&upstream_handle, &input_shape)?;
        let grad_id = store.alloc_device_tensor(input_shape, grad_handle)?;
        return Ok(smallvec![(x, grad_id)]);
    }

    let upstream = store.tensor_host(output_grad_id)?;
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
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        upstream.dirty != Dirty::Host && upstream.device_handle.is_some()
    };
    if device_path_ok {
        let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let (grad_handle, grad_shape) =
            store
                .backend()
                .transpose_axes_swap(&upstream_handle, &upstream_shape, axis1, axis2)?;
        let grad_id = store.alloc_device_tensor(grad_shape, grad_handle)?;
        return Ok(smallvec![(x, grad_id)]);
    }

    let upstream = store.tensor_host(output_grad_id)?;
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

    let expected_shape: Vec<usize> = starts
        .iter()
        .zip(ends.iter())
        .map(|(&start, &end)| end - start)
        .collect();
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    if upstream_shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream_shape,
        });
    }

    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        upstream.dirty != Dirty::Host && upstream.device_handle.is_some()
    };
    if device_path_ok {
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle = store.backend().slice_backward_device(
            &upstream_handle,
            &input_shape,
            &starts,
            &ends,
        )?;
        let grad_id = store.alloc_device_tensor(input_shape, grad_handle)?;
        return Ok(smallvec![(x, grad_id)]);
    }

    let upstream = store.tensor_host(output_grad_id)?;
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

/// Broadcast-copy `src` up to `target_shape` (right-aligned; each src axis is 1
/// or equal). Replaces the `add_broadcast(zeros, src)` idiom in `repeat_kv`: no
/// full-size zeros carrier on the tape. Gradient = sum-reduce over the expanded
/// axes, bit-identical to the old add-broadcast path.
pub fn broadcast_expand(
    src: TensorId,
    target_shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let src_shape = store.tensor(src)?.shape.clone();
    validate_broadcast(target_shape, &src_shape)?;

    let use_lazy = {
        let t = store.tensor(src)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    let output_id = if use_lazy {
        store.ensure_device(src)?;
        let src_handle = store
            .tensor(src)?
            .device_handle
            .as_ref()
            .expect("ensure_device")
            .clone();
        let out = store
            .backend()
            .broadcast_expand(&src_handle, &src_shape, target_shape)?;
        store.alloc_device_tensor(target_shape.to_vec(), out)?
    } else {
        let src_tensor = store.tensor_host(src)?;
        let out = store.backend().broadcast_expand_forward(
            &src_tensor.data,
            &src_tensor.shape,
            target_shape,
        )?;
        store.alloc(Tensor::new(out, target_shape.to_vec(), false)?)
    };

    TapeEntry {
        op: BackwardOp::BroadcastExpand,
        output_id,
        input_ids: smallvec![src],
        saved: SavedContext::BroadcastExpandCtx { src_shape },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn broadcast_expand_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let src = *entry.input_ids.first().ok_or(AutogradError::TapeInvariant(
        "broadcast_expand missing input",
    ))?;
    if !store.tensor(src)?.requires_grad {
        return Ok(GradPairs::new());
    }
    let SavedContext::BroadcastExpandCtx { src_shape } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "broadcast_expand backward missing saved shape",
        ));
    };
    let target_shape = store.tensor(output_grad_id)?.shape.clone();

    // grad_src = sum-reduce upstream over the expanded axes — identical to the
    // `b`-branch of `add_broadcast_backward`.
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        upstream.dirty != Dirty::Host && upstream.device_handle.is_some()
    };
    let mut grads = GradPairs::new();
    if device_path_ok {
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle = store.backend().add_broadcast_backward_device(
            &upstream_handle,
            &target_shape,
            &src_shape,
        )?;
        let grad_id = store.alloc_device_tensor(src_shape, grad_handle)?;
        grads.push((src, grad_id));
        return Ok(grads);
    }

    let upstream = store.tensor_host(output_grad_id)?;
    let src_size = shape_numel(&src_shape);
    let mut grad_src = vec![0.0; src_size];
    for (index, value) in upstream.data.iter().enumerate() {
        let offset = broadcast_offset(index, &target_shape, &src_shape);
        grad_src[offset] += *value;
    }
    let grad_id = store.alloc(Tensor::new(grad_src, src_shape, false)?);
    grads.push((src, grad_id));
    Ok(grads)
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

#[cfg(test)]
mod cat_tests {
    use super::*;
    use crate::{Tensor, ops, tape::Tape, tensor::TensorStore};

    // slice → cat round-trips a rank-3 tensor on the feature axis, and grads flow
    // back to each split unchanged (cat's adjoint is the inverse split). This is the
    // strided-copy branch the CP linear-attn transport relies on.
    #[test]
    fn cat_last_axis_roundtrip_and_grad() -> Result<()> {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        // [b=1, seq=2, dim=6], three feature regions of width 2.
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = store.alloc(Tensor::new(data.clone(), vec![1, 2, 6], true)?);
        let a = ops::slice(x, &[0, 0, 0], &[1, 2, 2], &mut store, &mut tape)?;
        let b = ops::slice(x, &[0, 0, 2], &[1, 2, 4], &mut store, &mut tape)?;
        let c = ops::slice(x, &[0, 0, 4], &[1, 2, 6], &mut store, &mut tape)?;
        let y = cat(&[a, b, c], 2, &mut store, &mut tape)?;
        assert_eq!(store.tensor(y)?.shape, vec![1, 2, 6]);
        assert_eq!(store.to_host(y)?, data);

        let sq = ops::mul(y, y, &mut store, &mut tape)?;
        let loss = ops::sum(sq, &mut store, &mut tape)?;
        let grads = tape.backward(loss, &mut store)?;
        let want: Vec<f32> = data.iter().map(|v| 2.0 * v).collect();
        assert_eq!(store.to_host(grads[&x])?, want);
        Ok(())
    }

    // Axis-0 concat of rank-2 tensors — the packed-conv-weight regime.
    #[test]
    fn cat_axis0_rank2() -> Result<()> {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let a = store.alloc(Tensor::new(vec![1.0, 2.0], vec![1, 2], false)?);
        let b = store.alloc(Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2], false)?);
        let y = cat(&[a, b], 0, &mut store, &mut tape)?;
        assert_eq!(store.tensor(y)?.shape, vec![3, 2]);
        assert_eq!(store.to_host(y)?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }
}
