// Index side-channel: indices stored as Vec<usize> in SavedContext, not in TensorStore.
// Avoids infrastructure sprawl (Option A).

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn gather_last_dim(
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
    let mut grad = vec![0.0; src_shape.iter().product()];
    for (prefix_index, &index) in indices.iter().enumerate() {
        grad[(prefix_index * vocab) + index] += upstream.data[prefix_index];
    }

    let grad_id = store.alloc(Tensor::new(grad, src_shape, false)?);
    Ok(smallvec![(src, grad_id)])
}
