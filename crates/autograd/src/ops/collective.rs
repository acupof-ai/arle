use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::CommAxis,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{TensorId, TensorStore},
};

/// Differentiable all-reduce sum over the world group.
///
/// Forward computes `y = sum_rank(x_rank)` through the backend collective.
/// Backward applies the adjoint collective, so a loss evaluated on every rank
/// returns the gradient of the distributed total loss.
pub fn all_reduce_sum(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_device(x)?;
    let (shape, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "all_reduce_sum: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), handle)
    };

    let out_handle =
        store
            .backend()
            .all_reduce_sum_device(&input_handle, &shape, CommAxis::World)?;
    let output_id = store.alloc_device_tensor(shape.clone(), out_handle)?;

    TapeEntry {
        op: BackwardOp::AllReduceSum,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::Shape(shape),
    }
    .record(store, tape)?;

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
    let grad_handle =
        store
            .backend()
            .all_reduce_sum_device(&upstream_handle, shape, CommAxis::World)?;
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
    let (local_shape, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "all_gather_seq: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), handle)
    };

    let out_handle =
        store
            .backend()
            .all_gather_seq_device(&input_handle, &local_shape, CommAxis::Seq)?;
    let output_id = store.alloc_device_tensor(full_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::AllGatherSeq,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::Shape(local_shape),
    }
    .record(store, tape)?;

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
    let grad_handle =
        store
            .backend()
            .reduce_scatter_sum_device(&upstream_handle, &local_shape, CommAxis::Seq)?;
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
    let (full_shape, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "reduce_scatter_sum: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), handle)
    };

    let out_handle =
        store
            .backend()
            .reduce_scatter_sum_device(&input_handle, &local_shape, CommAxis::Seq)?;
    let output_id = store.alloc_device_tensor(local_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::ReduceScatterSum,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::Shape(full_shape),
    }
    .record(store, tape)?;

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
    let grad_handle =
        store
            .backend()
            .all_gather_seq_device(&upstream_handle, &full_shape, CommAxis::Seq)?;
    let grad_id = store.alloc_device_tensor(full_shape, grad_handle)?;
    Ok(smallvec![(x, grad_id)])
}

/// Differentiable all-to-all that swaps a *scatter* axis for a *gather* axis —
/// the linear-attention CP transport (Megatron `MambaContextParallel`). Forward
/// splits `scatter_axis` across ranks and concatenates each rank's slice along
/// `gather_axis`: `[seq/N, b, hidden]` → `[seq, b, hidden/N]` (full sequence for
/// 1/N of the heads). It is self-adjoint with the two axes swapped, so backward
/// is the same op reversed. Single-rank / CPU is identity (`out_shape ==
/// in_shape`), which is the locally-verifiable core; the multi-rank NCCL shuffle
/// is pod-gated like the other collectives.
pub fn all_to_all(
    x: TensorId,
    scatter_axis: usize,
    gather_axis: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(x)?;
    let (in_shape, input_handle) = {
        let tensor = store.tensor(x)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "all_to_all: ensure_device left tensor without a device handle",
            ))?
            .clone();
        (tensor.shape.clone(), handle)
    };

    let (out_handle, out_shape) = store.backend().all_to_all_device(
        &input_handle,
        &in_shape,
        scatter_axis,
        gather_axis,
        CommAxis::Seq,
    )?;
    let output_id = store.alloc_device_tensor(out_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::AllToAll,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::AllToAllCtx {
            in_shape,
            scatter_axis,
            gather_axis,
        },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn all_to_all_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("all_to_all missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }
    let SavedContext::AllToAllCtx {
        in_shape,
        scatter_axis,
        gather_axis,
    } = &entry.saved
    else {
        return Err(AutogradError::TapeInvariant(
            "all_to_all backward missing saved context",
        ));
    };
    let (in_shape, scatter_axis, gather_axis) = (in_shape.clone(), *scatter_axis, *gather_axis);

    store.ensure_device(output_grad_id)?;
    let (upstream_handle, upstream_shape) = {
        let tensor = store.tensor(output_grad_id)?;
        let handle = tensor
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "all_to_all backward: upstream missing device handle",
            ))?
            .clone();
        (handle, tensor.shape.clone())
    };
    // Adjoint = the same shuffle with scatter/gather swapped, mapping the upstream
    // grad back to this rank's input shape.
    let (grad_handle, grad_shape) = store.backend().all_to_all_device(
        &upstream_handle,
        &upstream_shape,
        gather_axis,
        scatter_axis,
        CommAxis::Seq,
    )?;
    if grad_shape != in_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: in_shape,
            got: grad_shape,
        });
    }
    let grad_id = store.alloc_device_tensor(in_shape, grad_handle)?;
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

    // world==1: all_to_all is identity on shape and value (no rank to shuffle to),
    // and self-adjoint — the backward is the same op with axes swapped, which at
    // N=1 is also identity. Tier-1 linear-attn CP gate, runnable on CPU before any
    // NCCL wiring. `[seq/N,b,hidden] -> [seq,b,hidden/N]` collapses to identity.
    #[test]
    fn all_to_all_single_rank_is_identity() -> Result<()> {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let x = store.alloc(Tensor::new(
            vec![1.0, -2.0, 3.5, 4.0, 5.0, -6.0],
            vec![3, 1, 2],
            true,
        )?);
        // scatter seq (axis 0) into head (axis 2): identity at world==1.
        let y = all_to_all(x, 0, 2, &mut store, &mut tape)?;
        let yy = ops::mul(y, y, &mut store, &mut tape)?;
        let loss = ops::sum(yy, &mut store, &mut tape)?;

        let grads = tape.backward(loss, &mut store)?;
        assert_eq!(store.tensor(y)?.shape, vec![3, 1, 2]);
        assert_eq!(store.to_host(y)?, vec![1.0, -2.0, 3.5, 4.0, 5.0, -6.0]);
        assert_eq!(
            store.to_host(grads[&x])?,
            vec![2.0, -4.0, 7.0, 8.0, 10.0, -12.0]
        );
        Ok(())
    }

    // Host reference for the world>1 all_to_all permutation — the correctness
    // spec the CUDA `all_to_all_device` body mirrors (backend_cuda.rs), runnable
    // on Mac where the device path can't. Not a co-evolving oracle: `a2a_ref`
    // assembles per-rank (slice each input along gather, concat along scatter, the
    // body's own steps) while `expected` is built by an INDEPENDENT path (slice the
    // full tensor along gather). The two agree only if the permutation is right.
    fn nd_strides(shape: &[usize]) -> Vec<usize> {
        let mut s = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    }

    fn slice_axis(
        src: &[f32],
        src_shape: &[usize],
        axis: usize,
        start: usize,
        end: usize,
    ) -> (Vec<f32>, Vec<usize>) {
        let mut out_shape = src_shape.to_vec();
        out_shape[axis] = end - start;
        let (ss, os) = (nd_strides(src_shape), nd_strides(&out_shape));
        let total: usize = out_shape.iter().product();
        let mut out = vec![0f32; total];
        for (flat, slot) in out.iter_mut().enumerate() {
            let (mut rem, mut sflat) = (flat, 0usize);
            for d in 0..out_shape.len() {
                let c = rem / os[d] + if d == axis { start } else { 0 };
                sflat += c * ss[d];
                rem %= os[d];
            }
            *slot = src[sflat];
        }
        (out, out_shape)
    }

    // Write `src` into `dst` at `off` along `axis` (the concat primitive).
    fn write_at_axis(
        dst: &mut [f32],
        dst_shape: &[usize],
        src: &[f32],
        src_shape: &[usize],
        axis: usize,
        off: usize,
    ) {
        let (ds, ss) = (nd_strides(dst_shape), nd_strides(src_shape));
        for (flat, &v) in src.iter().enumerate() {
            let (mut rem, mut dflat) = (flat, 0usize);
            for d in 0..src_shape.len() {
                let c = rem / ss[d] + if d == axis { off } else { 0 };
                dflat += c * ds[d];
                rem %= ss[d];
            }
            dst[dflat] = v;
        }
    }

    fn a2a_ref(
        inputs: &[(Vec<f32>, Vec<usize>)],
        scatter: usize,
        gather: usize,
        n: usize,
    ) -> Vec<(Vec<f32>, Vec<usize>)> {
        let in_shape = &inputs[0].1;
        let (sc, g) = (in_shape[scatter], in_shape[gather] / n);
        let mut out_shape = in_shape.to_vec();
        out_shape[scatter] = sc * n;
        out_shape[gather] = g;
        (0..n)
            .map(|j| {
                let mut out = vec![0f32; out_shape.iter().product()];
                for (r, (data, shape)) in inputs.iter().enumerate() {
                    let (chunk, chunk_shape) = slice_axis(data, shape, gather, j * g, (j + 1) * g);
                    write_at_axis(&mut out, &out_shape, &chunk, &chunk_shape, scatter, r * sc);
                }
                (out, out_shape.clone())
            })
            .collect()
    }

    fn check_a2a(full_shape: &[usize], scatter: usize, gather: usize, n: usize) {
        let total: usize = full_shape.iter().product();
        let full: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let sc = full_shape[scatter] / n;
        let g = full_shape[gather] / n;
        // Each rank owns a scatter-shard of the full tensor.
        let inputs: Vec<(Vec<f32>, Vec<usize>)> = (0..n)
            .map(|r| slice_axis(&full, full_shape, scatter, r * sc, (r + 1) * sc))
            .collect();
        let got = a2a_ref(&inputs, scatter, gather, n);
        for (j, rank) in got.iter().enumerate() {
            // Independent expected: rank j holds the FULL sequence for head-slice j.
            let expected = slice_axis(&full, full_shape, gather, j * g, (j + 1) * g);
            assert_eq!(*rank, expected, "a2a forward mismatch at rank {j}");
        }
        // Self-adjoint: applying the shuffle with axes swapped reconstructs inputs.
        let back = a2a_ref(&got, gather, scatter, n);
        assert_eq!(back, inputs, "a2a is not self-adjoint");
    }

    #[test]
    fn all_to_all_world2_permutation_matches_full_tensor_view() {
        // [2,4,6]: forward callsite (scatter=seq axis 1, gather=dim axis 2) and the
        // restore callsite (scatter=2 innermost → strided assembly, gather=1).
        check_a2a(&[2, 4, 6], 1, 2, 2);
        check_a2a(&[2, 4, 6], 2, 1, 2);
        // world=4 and a non-trivial batch axis.
        check_a2a(&[2, 8, 4], 1, 2, 4);
    }
}
