// Index side-channel: indices stored as Vec<usize> in SavedContext, not in TensorStore.
// Avoids infrastructure sprawl (Option A).

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{GpuTensor, TensorId, TensorStore},
};

pub fn embedding(
    table: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let table_tensor = store.tensor(table)?.clone();
    if table_tensor.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: table_tensor.shape.len(),
        });
    }

    let vocab = table_tensor.shape[0];
    let hidden = table_tensor.shape[1];
    let seq_len = indices.len();
    let mut output = vec![0.0; seq_len * hidden];
    for (position, &index) in indices.iter().enumerate() {
        if index >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index,
                upper: vocab,
            });
        }
        let src_base = index * hidden;
        let dst_base = position * hidden;
        output[dst_base..dst_base + hidden]
            .copy_from_slice(&table_tensor.data[src_base..src_base + hidden]);
    }

    // Raw indices do not carry an explicit [B, S] shape, so M1 treats them as a
    // single batch row `[1, S]` instead of introducing a separate integer tensor store.
    let output_shape = vec![1, seq_len, hidden];
    let output_id = store.alloc(GpuTensor::new(
        output,
        output_shape,
        table_tensor.requires_grad,
    )?);
    if table_tensor.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Embedding,
            output_id,
            input_ids: smallvec![table],
            saved: SavedContext::EmbeddingCtx {
                indices: indices.to_vec(),
                table_shape: table_tensor.shape,
            },
        });
    }

    Ok(output_id)
}

pub(crate) fn embedding_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let table = *entry.input_ids.first().ok_or(AutogradError::TapeInvariant(
        "embedding missing table input",
    ))?;
    if !store.tensor(table)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::EmbeddingCtx {
        indices,
        table_shape,
    } = entry.saved.clone()
    else {
        return Err(AutogradError::TapeInvariant(
            "embedding backward missing saved context",
        ));
    };

    let upstream = store.tensor(output_grad_id)?.clone();
    let hidden = table_shape[1];
    let expected_shape = vec![1, indices.len(), hidden];
    if upstream.shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream.shape,
        });
    }

    let mut grad_table = vec![0.0; table_shape.iter().product()];
    for (position, &index) in indices.iter().enumerate() {
        let src_base = position * hidden;
        let dst_base = index * hidden;
        for col in 0..hidden {
            grad_table[dst_base + col] += upstream.data[src_base + col];
        }
    }

    let grad_id = store.alloc(GpuTensor::new(grad_table, table_shape, false)?);
    Ok(smallvec![(table, grad_id)])
}
