// Index side-channel: indices stored as Vec<usize> in SavedContext, not in TensorStore.
// Avoids infrastructure sprawl (Option A).

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn embedding(
    table: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // M5.3b.7: dispatch on device-handle presence (same shape as the rope
    // gate — `device_handle.is_some() && dirty != Host`). Embedding is
    // the very first op in a forward pass; taking the lazy branch here
    // lets downstream lazy ops (rmsnorm, matmul, silu, rope, exp,
    // softmax) compose end-to-end with no intermediate eval. The table
    // is typically Dirty::Host on first call but can become Dirty::Both
    // after an AdamW step that uploads updated weights — both cases
    // route to the lazy path.
    let has_device_handle = {
        let t = store.tensor(table)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    if has_device_handle {
        embedding_device_lazy(table, indices, store, tape)
    } else {
        embedding_host_eager(table, indices, store, tape)
    }
}

fn embedding_device_lazy(
    table: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(table)?;

    let (table_shape, requires_grad) = {
        let t = store.tensor(table)?;
        (t.shape.clone(), t.requires_grad)
    };
    if table_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: table_shape.len(),
        });
    }
    let vocab = table_shape[0];
    let hidden = table_shape[1];
    let seq_len = indices.len();
    // Bounds-check once here; the lazy helper re-checks but the early
    // error carries the original `usize` index the caller passed.
    for &index in indices {
        if index >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index,
                upper: vocab,
            });
        }
    }

    let ids_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let table_handle = store
        .tensor(table)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "embedding: ensure_device left table without a device handle",
        ))?
        .clone();

    let out_handle = store
        .backend()
        .embedding(&table_handle, &table_shape, &ids_i32)?;
    let output_id = store.alloc_device_tensor(vec![1, seq_len, hidden], out_handle)?;
    store.set_requires_grad(output_id, requires_grad)?;

    if requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Embedding,
            output_id,
            input_ids: smallvec![table],
            saved: SavedContext::EmbeddingCtx {
                indices: indices.to_vec(),
                table_shape,
            },
        });
    }

    Ok(output_id)
}

fn embedding_host_eager(
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
    // Bounds-check here so the error carries the original `usize` index (the
    // backend kernel silently zero-fills OOB rows for well-defined behavior).
    for &index in indices {
        if index >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index,
                upper: vocab,
            });
        }
    }
    let ids_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let output = store
        .backend()
        .embedding_forward(&table_tensor.data, vocab, hidden, &ids_i32)?;
    debug_assert_eq!(output.len(), seq_len * hidden);

    // Raw indices do not carry an explicit [B, S] shape, so M1 treats them as a
    // single batch row `[1, S]` instead of introducing a separate integer tensor store.
    let output_shape = vec![1, seq_len, hidden];
    let output_id = store.alloc(Tensor::new(
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

    let vocab = table_shape[0];
    let ids_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let grad_table = store.backend().scatter_add_rows_forward(
        &upstream.data,
        indices.len(),
        hidden,
        &ids_i32,
        vocab,
    )?;

    let grad_id = store.alloc(Tensor::new(grad_table, table_shape, false)?);
    Ok(smallvec![(table, grad_id)])
}
