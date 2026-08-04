use std::collections::HashSet;

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::{CausalSdpaDeviceBackwardArgs, Device, cpu_causal_sdpa_recompute_backward},
    ops::{
        add_broadcast, broadcast_expand,
        chunk_accum::{ChunkSum, SeqAccum},
        matmul, mul_scalar, reshape, slice, softmax, softmax_backward, transpose,
    },
    tape::{BackwardOp, GradPairs, SavedContext, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn repeat_kv(
    x: TensorId,
    n_rep: usize,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    if n_rep == 0 {
        return Err(AutogradError::InvalidIndicesLen {
            expected: 1,
            got: 0,
        });
    }
    if n_rep == 1 {
        return Ok(x);
    }

    let x_shape = store.tensor(x)?.shape.clone();
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }

    // Fused prefill SDPA is GQA-native (reads kv_heads directly); this expand
    // exists only because the non-GQA recompute backward requires q/k/v at
    // equal head count. `broadcast_expand` copies without a zeros carrier.
    let reshaped = reshape(
        x,
        &[x_shape[0], x_shape[1], 1, x_shape[2], x_shape[3]],
        store,
        tape,
    )?;
    let expanded = broadcast_expand(
        reshaped,
        &[x_shape[0], x_shape[1], n_rep, x_shape[2], x_shape[3]],
        store,
        tape,
    )?;
    reshape(
        expanded,
        &[x_shape[0], x_shape[1] * n_rep, x_shape[2], x_shape[3]],
        store,
        tape,
    )
}

/// Fused flash-style prefill on the device backend (the adopted inference
/// kernel — no `[seq, seq]` score transient). `None` => compose from
/// primitives. Forward-only: callers own the tape entry (recompute backward
/// never consumes the forward output, so the fused output is a drop-in).
fn try_causal_sdpa_prefill_device(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_start: usize,
    store: &mut TensorStore,
) -> Result<Option<TensorId>> {
    if store.backend().device() != Device::Cuda {
        return Ok(None);
    }
    for id in [q, k, v] {
        store.ensure_device(id)?;
    }
    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();
    let (Some(q_h), Some(k_h), Some(v_h)) = (
        store.tensor(q)?.device_handle.clone(),
        store.tensor(k)?.device_handle.clone(),
        store.tensor(v)?.device_handle.clone(),
    ) else {
        return Ok(None);
    };
    let fused = store
        .backend()
        .causal_sdpa_prefill_device(&q_h, &q_shape, &k_h, &k_shape, &v_h, &v_shape, q_start)?;
    // The backend rejects out-of-envelope shapes with a silent None; surface
    // the verdict + shapes under the trace flag so a wrong guard is visible.
    if std::env::var("ARLE_SDPA_TRACE").is_ok() {
        eprintln!(
            "[sdpa-trace] fused={} q={q_shape:?} k={k_shape:?} q_start={q_start}",
            if fused.is_some() { "TAKEN" } else { "REJECTED" }
        );
    }
    let Some((out, out_shape)) = fused else {
        return Ok(None);
    };
    Ok(Some(store.alloc_device_tensor(out_shape, out)?))
}

pub fn causal_sdpa(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();
    validate_attention_shapes(&q_shape, &k_shape, &v_shape)?;

    let batch = q_shape[0];
    let heads = q_shape[1];
    let seq_len = q_shape[2];
    let head_dim = q_shape[3];
    let merged_heads = batch * heads;

    let q_3d = reshape(q, &[merged_heads, seq_len, head_dim], store, tape)?;
    let k_3d = reshape(k, &[merged_heads, seq_len, head_dim], store, tape)?;
    let v_3d = reshape(v, &[merged_heads, seq_len, head_dim], store, tape)?;
    let k_t = transpose(k_3d, 1, 2, store, tape)?;
    let scores = matmul(q_3d, k_t, store, tape)?;
    let scaled = mul_scalar(scores, 1.0 / (head_dim as f32).sqrt(), store, tape)?;
    let mask = causal_mask(seq_len, store)?;
    let masked = add_broadcast(scaled, mask, store, tape)?;
    let probs = softmax(masked, store, tape)?;
    let context = matmul(probs, v_3d, store, tape)?;
    reshape(context, &[batch, heads, seq_len, head_dim], store, tape)
}

pub fn causal_sdpa_recompute(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    causal_sdpa_recompute_with_q_start(q, k, v, 0, store, tape)
}

pub fn causal_sdpa_recompute_with_q_start(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_start: usize,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();
    validate_cached_attention_shapes(&q_shape, &k_shape, &v_shape, q_start)?;

    store.ensure_device(q)?;
    store.ensure_device(k)?;
    store.ensure_device(v)?;

    let output_id = if let Some(fused) = try_causal_sdpa_prefill_device(q, k, v, q_start, store)? {
        fused
    } else if store.backend().device() == Device::Cuda && q_shape[0] == 1 && q_shape[2] > 65_535 {
        let live_before = store.live_ids().into_iter().collect::<HashSet<_>>();
        let mut inner_tape = crate::Tape::new();
        inner_tape.set_enabled(false);
        let (batch, heads, seq, dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
        let mut accum = SeqAccum::new(q_shape.clone(), 2, store)?;
        let mut q0 = 0;
        while q0 < seq {
            let q1 = (q0 + 65_535).min(seq);
            let q_part = slice(
                q,
                &[0, 0, q0, 0],
                &[batch, heads, q1, dim],
                store,
                &mut inner_tape,
            )?;
            let k_part = slice(
                k,
                &[0, 0, 0, 0],
                &[batch, heads, q_start + q1, dim],
                store,
                &mut inner_tape,
            )?;
            let v_part = slice(
                v,
                &[0, 0, 0, 0],
                &[batch, heads, q_start + q1, dim],
                store,
                &mut inner_tape,
            )?;
            let out = try_causal_sdpa_prefill_device(q_part, k_part, v_part, q_start + q0, store)?
                .ok_or(AutogradError::TapeInvariant(
                    "long causal SDPA chunk rejected by fused prefill",
                ))?;
            accum.write_rows(q0, out, store)?;
            store.free_new_except(&live_before, &HashSet::from([accum.id()]))?;
            q0 = q1;
        }
        let out = accum.finish();
        let keep = HashSet::from([q, k, v, out]);
        store.free_new_except(&live_before, &keep)?;
        out
    } else {
        let live_before = store.live_ids().into_iter().collect::<HashSet<_>>();
        let mut inner_tape = crate::Tape::new();
        inner_tape.set_enabled(false);
        let out = causal_sdpa_with_q_start(q, k, v, q_start, store, &mut inner_tape)?;
        let keep = HashSet::from([q, k, v, out]);
        store.free_new_except(&live_before, &keep)?;
        out
    };

    crate::TapeEntry {
        op: BackwardOp::CausalSdpaRecompute,
        output_id,
        input_ids: smallvec![q, k, v],
        saved: SavedContext::CausalSdpaRecomputeCtx { q, k, v, q_start },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub fn causal_sdpa_with_q_start(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_start: usize,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();
    validate_cached_attention_shapes(&q_shape, &k_shape, &v_shape, q_start)?;

    let batch = q_shape[0];
    let heads = q_shape[1];
    let q_len = q_shape[2];
    let kv_len = k_shape[2];
    let head_dim = q_shape[3];
    if !tape.enabled
        && let Some(fused) = try_causal_sdpa_prefill_device(q, k, v, q_start, store)?
    {
        return Ok(fused);
    }
    // Bound the four score-sized transients.
    let chunk = sdpa_head_chunk(heads, q_len, kv_len);
    if chunk < heads {
        return sdpa_head_chunked(q, k, v, q_start, chunk, store, tape);
    }
    if q_start == 0 && q_len == kv_len {
        return causal_sdpa(q, k, v, store, tape);
    }

    let merged_heads = batch * heads;
    let q_3d = reshape(q, &[merged_heads, q_len, head_dim], store, tape)?;
    let k_3d = reshape(k, &[merged_heads, kv_len, head_dim], store, tape)?;
    let v_3d = reshape(v, &[merged_heads, kv_len, head_dim], store, tape)?;
    let k_t = transpose(k_3d, 1, 2, store, tape)?;
    let scores = matmul(q_3d, k_t, store, tape)?;
    let scaled = mul_scalar(scores, 1.0 / (head_dim as f32).sqrt(), store, tape)?;
    let masked = if q_len == 1 && q_start + 1 == kv_len {
        scaled
    } else {
        let mask = causal_mask_window(q_len, kv_len, q_start, store)?;
        add_broadcast(scaled, mask, store, tape)?
    };
    let probs = softmax(masked, store, tape)?;
    let context = matmul(probs, v_3d, store, tape)?;
    reshape(context, &[batch, heads, q_len, head_dim], store, tape)
}

pub fn causal_sdpa_decode_gqa(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_start: usize,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    if tape.enabled {
        return Err(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa is rollout-only and requires tape disabled",
        ));
    }

    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();

    store.ensure_device(q)?;
    store.ensure_device(k)?;
    store.ensure_device(v)?;
    let q_handle = store
        .tensor(q)?
        .device_handle
        .clone()
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa: q missing device handle",
        ))?;
    let k_handle = store
        .tensor(k)?
        .device_handle
        .clone()
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa: k missing device handle",
        ))?;
    let v_handle = store
        .tensor(v)?
        .device_handle
        .clone()
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa: v missing device handle",
        ))?;
    let (out_handle, out_shape) = store.backend().causal_sdpa_decode_gqa(
        &q_handle, &q_shape, &k_handle, &k_shape, &v_handle, &v_shape, q_start,
    )?;
    store.alloc_device_tensor(out_shape, out_handle)
}

pub(crate) fn causal_sdpa_recompute_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::CausalSdpaRecomputeCtx { q, k, v, q_start } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "causal_sdpa_recompute backward missing saved context",
        ));
    };

    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let v_shape = store.tensor(v)?.shape.clone();
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    if upstream_shape != q_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.clone(),
            got: upstream_shape,
        });
    }

    let need_grad_q = store.tensor(q)?.requires_grad;
    let need_grad_k = store.tensor(k)?.requires_grad;
    let need_grad_v = store.tensor(v)?.requires_grad;
    let mut grads = GradPairs::new();
    if !need_grad_q && !need_grad_k && !need_grad_v {
        return Ok(grads);
    }

    // Context-parallel query shard: q attends a gathered KV prefix (q_len < kv_len,
    // q_start > 0). The fused-square device kernel and the CPU path both assume a
    // square window, so route straight to the rectangular chunked recompute (the
    // only path that honors q_start + kv_len != q_len).
    if q_start != 0 || q_shape[2] != k_shape[2] {
        validate_cached_attention_shapes(&q_shape, &k_shape, &v_shape, q_start)?;
        let merged_heads = q_shape[0] * q_shape[1];
        let q_chunk = sdpa_backward_q_chunk(merged_heads, k_shape[2]);
        return causal_sdpa_recompute_backward_device_chunked(
            q,
            k,
            v,
            output_grad_id,
            &q_shape,
            &k_shape,
            q_start,
            need_grad_q,
            need_grad_k,
            need_grad_v,
            q_chunk,
            store,
        );
    }
    validate_attention_shapes(&q_shape, &k_shape, &v_shape)?;

    let device_path_ok = {
        let q_tensor = store.tensor(q)?;
        let k_tensor = store.tensor(k)?;
        let v_tensor = store.tensor(v)?;
        let upstream = store.tensor(output_grad_id)?;
        q_tensor.dirty != Dirty::Host
            && q_tensor.device_handle.is_some()
            && k_tensor.dirty != Dirty::Host
            && k_tensor.device_handle.is_some()
            && v_tensor.dirty != Dirty::Host
            && v_tensor.device_handle.is_some()
            && upstream.dirty != Dirty::Host
            && upstream.device_handle.is_some()
    };
    if device_path_ok {
        let q_handle = store
            .tensor(q)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let k_handle = store
            .tensor(k)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let v_handle = store
            .tensor(v)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let (grad_q, grad_k, grad_v) = store.backend().causal_sdpa_recompute_backward_device(
            CausalSdpaDeviceBackwardArgs {
                q: &q_handle,
                k: &k_handle,
                v: &v_handle,
                upstream: &upstream_handle,
                shape: &q_shape,
                need_grad_q,
                need_grad_k,
                need_grad_v,
            },
        )?;
        if let Some(handle) = grad_q {
            grads.push((q, store.alloc_device_tensor(q_shape.clone(), handle)?));
        }
        if let Some(handle) = grad_k {
            grads.push((k, store.alloc_device_tensor(k_shape.clone(), handle)?));
        }
        if let Some(handle) = grad_v {
            grads.push((v, store.alloc_device_tensor(v_shape.clone(), handle)?));
        }
        return Ok(grads);
    }

    let q_tensor = store.tensor_host(q)?;
    let k_tensor = store.tensor_host(k)?;
    let v_tensor = store.tensor_host(v)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let (grad_q, grad_k, grad_v) = cpu_causal_sdpa_recompute_backward(
        &q_tensor.data,
        &k_tensor.data,
        &v_tensor.data,
        &upstream.data,
        &q_shape,
        need_grad_q,
        need_grad_k,
        need_grad_v,
    )?;

    if let Some(data) = grad_q {
        grads.push((
            q,
            store.alloc(Tensor::new(data, q_tensor.shape.clone(), false)?),
        ));
    }
    if let Some(data) = grad_k {
        grads.push((
            k,
            store.alloc(Tensor::new(data, k_tensor.shape.clone(), false)?),
        ));
    }
    if let Some(data) = grad_v {
        grads.push((
            v,
            store.alloc(Tensor::new(data, v_tensor.shape.clone(), false)?),
        ));
    }

    Ok(grads)
}

/// Fit backward score transients in 4 GiB.
fn sdpa_backward_q_chunk(merged_heads: usize, seq_len: usize) -> usize {
    let per_row = merged_heads
        .saturating_mul(seq_len)
        .saturating_mul(4)
        .max(1);
    ((4usize << 30) / per_row.saturating_mul(6)).clamp(1, seq_len)
}

#[allow(clippy::too_many_arguments)]
fn causal_sdpa_recompute_backward_device_chunked(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    upstream: TensorId,
    q_shape: &[usize],
    kv_shape: &[usize],
    q_start: usize,
    need_grad_q: bool,
    need_grad_k: bool,
    need_grad_v: bool,
    q_chunk: usize,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let batch = q_shape[0];
    let heads = q_shape[1];
    let q_len = q_shape[2];
    let kv_len = kv_shape[2];
    let head_dim = q_shape[3];
    let merged_heads = batch * heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_chunk = q_chunk.clamp(1, q_len.max(1));
    let mut tape = crate::Tape::new();
    tape.set_enabled(false);

    let q_3d = reshape(q, &[merged_heads, q_len, head_dim], store, &mut tape)?;
    let k_3d = reshape(k, &[merged_heads, kv_len, head_dim], store, &mut tape)?;
    let v_3d = reshape(v, &[merged_heads, kv_len, head_dim], store, &mut tape)?;
    let upstream_3d = reshape(upstream, &[merged_heads, q_len, head_dim], store, &mut tape)?;

    // Shared across chunks (no O(seq²) cost): k^T and v^T are `[merged_heads, *, *]`.
    let k_t = transpose(k_3d, 1, 2, store, &mut tape)?;
    let v_t = if need_grad_q || need_grad_k {
        Some(transpose(v_3d, 1, 2, store, &mut tape)?)
    } else {
        None
    };

    // Anchor for per-chunk transient cleanup: everything alloc'd from here on
    // (except the kept accumulators / grad_q chunks) is freed after each chunk.
    let live_before: HashSet<TensorId> = store.live_ids().into_iter().collect();

    // Rows are the *local* q shard (q_len), not kv_len — a CP shard has q_len < kv_len.
    let mut grad_q_3d = need_grad_q
        .then(|| SeqAccum::new(vec![merged_heads, q_len, head_dim], 1, store))
        .transpose()?;
    let mut grad_k_acc = ChunkSum::new();
    let mut grad_v_acc = ChunkSum::new();

    let mut q0 = 0usize;
    while q0 < q_len {
        let q1 = (q0 + q_chunk).min(q_len);
        let rows = q1 - q0;

        let q_chunk_3d = slice(
            q_3d,
            &[0, q0, 0],
            &[merged_heads, q1, head_dim],
            store,
            &mut tape,
        )?;
        let up_chunk_3d = slice(
            upstream_3d,
            &[0, q0, 0],
            &[merged_heads, q1, head_dim],
            store,
            &mut tape,
        )?;

        // scores/probs for this q-chunk: `[merged_heads, rows, kv_len]`.
        let scores = matmul(q_chunk_3d, k_t, store, &mut tape)?;
        let scaled = mul_scalar(scores, scale, store, &mut tape)?;
        // Windowed causal mask `[1, rows, kv_len]`; absolute q position = q_start+q0,
        // so a shard's local q rows attend the right prefix of the gathered KV.
        let mask = causal_mask_window(rows, kv_len, q_start + q0, store)?;
        let masked = add_broadcast(scaled, mask, store, &mut tape)?;
        let probs = softmax(masked, store, &mut tape)?;

        // grad_v contribution: probs^T @ upstream_chunk → `[merged_heads, seq, head_dim]`.
        if need_grad_v {
            let probs_t = transpose(probs, 1, 2, store, &mut tape)?;
            let grad_v_part = matmul(probs_t, up_chunk_3d, store, &mut tape)?;
            grad_v_acc.add(grad_v_part, store)?;
        }

        if need_grad_q || need_grad_k {
            let v_t = v_t.expect("v_t computed when need_grad_q || need_grad_k");
            let d_probs = matmul(up_chunk_3d, v_t, store, &mut tape)?;
            let softmax_entry = TapeEntry {
                op: BackwardOp::Softmax,
                output_id: probs,
                input_ids: smallvec![masked],
                saved: SavedContext::SoftmaxCtx { y: probs },
            };
            let d_scores_pairs = softmax_backward(&softmax_entry, d_probs, store)?;
            let d_scores = d_scores_pairs
                .iter()
                .find_map(|(input_id, grad_id)| (*input_id == masked).then_some(*grad_id))
                .ok_or(AutogradError::TapeInvariant(
                    "causal_sdpa_recompute device backward missing softmax grad",
                ))?;

            // grad_q chunk: scale·(d_scores @ k) → `[merged_heads, rows, head_dim]`.
            if let Some(acc) = grad_q_3d.as_mut() {
                let grad_q_part = matmul(d_scores, k_3d, store, &mut tape)?;
                let grad_q_part = mul_scalar(grad_q_part, scale, store, &mut tape)?;
                acc.write_rows(q0, grad_q_part, store)?;
            }
            // grad_k contribution: scale·(d_scores^T @ q_chunk) → `[merged_heads, seq, head_dim]`.
            if need_grad_k {
                let d_scores_t = transpose(d_scores, 1, 2, store, &mut tape)?;
                let grad_k_part = matmul(d_scores_t, q_chunk_3d, store, &mut tape)?;
                let grad_k_part = mul_scalar(grad_k_part, scale, store, &mut tape)?;
                grad_k_acc.add(grad_k_part, store)?;
            }
        }

        // Free this chunk's `[*, rows, seq]` transients (scores/scaled/masked/
        // probs/d_probs/d_scores + q/upstream slices); keep only the running
        // accumulators. Bounds peak to O(q_chunk·seq).
        let keep = grad_q_3d
            .as_ref()
            .map(SeqAccum::id)
            .into_iter()
            .chain(grad_k_acc.id())
            .chain(grad_v_acc.id())
            .collect();
        store.free_new_except(&live_before, &keep)?;

        q0 = q1;
    }

    let mut grads = GradPairs::new();
    if need_grad_v {
        let grad_v_3d = grad_v_acc
            .finish(store)?
            .ok_or(AutogradError::TapeInvariant(
                "causal_sdpa_recompute device backward: grad_v requested but no chunk produced it",
            ))?;
        let grad_v = reshape(grad_v_3d, kv_shape, store, &mut tape)?;
        grads.push((v, grad_v));
    }
    if let Some(acc) = grad_q_3d {
        let grad_q = reshape(acc.finish(), q_shape, store, &mut tape)?;
        grads.push((q, grad_q));
    }
    if need_grad_k {
        let grad_k_3d = grad_k_acc
            .finish(store)?
            .ok_or(AutogradError::TapeInvariant(
                "causal_sdpa_recompute device backward: grad_k requested but no chunk produced it",
            ))?;
        let grad_k = reshape(grad_k_3d, kv_shape, store, &mut tape)?;
        grads.push((k, grad_k));
    }

    Ok(grads)
}

/// Concatenate rank-4 `[batch, heads_i, seq, head_dim]` tensors along the
/// head axis (axis 1). Host-only path — runs ~once per layer, not in the
/// inner softmax loop, so correctness over speed (mirrors `slice_host_eager`).
pub fn cat_heads(
    inputs: &[TensorId],
    store: &mut TensorStore,
    tape: &mut crate::Tape,
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

    // Validate ranks + matching batch/seq/head_dim; heads may differ.
    let first_shape = store.tensor(inputs[0])?.shape.clone();
    if first_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: first_shape.len(),
        });
    }
    let (batch, seq, head_dim) = (first_shape[0], first_shape[2], first_shape[3]);
    let head_counts: Vec<usize> = inputs
        .iter()
        .map(|&id| {
            let t = store.tensor(id)?;
            if t.shape.len() != 4 {
                return Err(AutogradError::InvalidRank {
                    expected: "4",
                    got: t.shape.len(),
                });
            }
            if t.shape[0] != batch || t.shape[2] != seq || t.shape[3] != head_dim {
                return Err(AutogradError::ShapeMismatch {
                    expected: first_shape.clone(),
                    got: t.shape.clone(),
                });
            }
            Ok(t.shape[1])
        })
        .collect::<Result<Vec<_>>>()?;
    let total_heads: usize = head_counts.iter().sum();
    let requires_grad = inputs
        .iter()
        .any(|&id| store.tensor(id).is_ok_and(|t| t.requires_grad));

    let out_shape = vec![batch, total_heads, seq, head_dim];
    let mut data = vec![0.0_f32; batch * total_heads * seq * head_dim];
    let block = seq * head_dim;
    for (i, &id) in inputs.iter().enumerate() {
        let heads_i = head_counts[i];
        let head_offset: usize = head_counts[..i].iter().sum();
        let src = store.tensor_host(id)?;
        for batch_idx in 0..batch {
            let out_base = (batch_idx * total_heads + head_offset) * block;
            let src_base = (batch_idx * heads_i) * block;
            let len = heads_i * block;
            data[out_base..out_base + len].copy_from_slice(&src.data[src_base..src_base + len]);
        }
    }

    let output_id = store.alloc(Tensor::new(data, out_shape, requires_grad)?);

    TapeEntry {
        op: BackwardOp::CatHeads,
        output_id,
        input_ids: inputs.iter().copied().collect(),
        saved: SavedContext::CatHeadsCtx { head_counts },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn cat_heads_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::CatHeadsCtx { head_counts } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "cat_heads backward missing saved context",
        ));
    };

    let upstream = store.tensor_host(output_grad_id)?;
    let upstream_shape = upstream.shape.clone();
    if upstream_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: upstream_shape.len(),
        });
    }
    let (batch, total_heads, seq, head_dim) = (
        upstream_shape[0],
        upstream_shape[1],
        upstream_shape[2],
        upstream_shape[3],
    );
    let block = seq * head_dim;

    let mut grads = GradPairs::new();
    for (i, &input_id) in entry.input_ids.iter().enumerate() {
        if !store.tensor(input_id)?.requires_grad {
            continue;
        }
        let heads_i = head_counts[i];
        let head_offset: usize = head_counts[..i].iter().sum();
        let mut grad = vec![0.0_f32; batch * heads_i * block];
        for batch_idx in 0..batch {
            let src_base = (batch_idx * total_heads + head_offset) * block;
            let dst_base = (batch_idx * heads_i) * block;
            let len = heads_i * block;
            grad[dst_base..dst_base + len]
                .copy_from_slice(&upstream.data[src_base..src_base + len]);
        }
        let grad_id = store.alloc(Tensor::new(
            grad,
            vec![batch, heads_i, seq, head_dim],
            false,
        )?);
        grads.push((input_id, grad_id));
    }

    Ok(grads)
}

/// Heads per chunk for the composed SDPA: bound its 4 simultaneous
/// `[chunk, q_len, kv_len]` f32 buffers (scores/scaled/masked/probs) to ~8 GiB.
fn sdpa_head_chunk(heads: usize, q_len: usize, kv_len: usize) -> usize {
    let per_head = q_len.saturating_mul(kv_len).saturating_mul(4).max(1);
    ((8usize << 30) / per_head.saturating_mul(4)).clamp(1, heads.max(1))
}

/// Composed SDPA over head chunks — heads are independent, so chunking is
/// numerically exact while bounding the live `[chunk, q_len, kv_len]` scores.
fn sdpa_head_chunked(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_start: usize,
    chunk: usize,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    let q_shape = store.tensor(q)?.shape.clone();
    let k_shape = store.tensor(k)?.shape.clone();
    let (batch, heads, q_len, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let kv_len = k_shape[2];

    // When the tape is disabled the per-chunk SDPA intermediates have no backward
    // dependency but nothing frees them until a checkpoint boundary, so they pile
    // up across the loop — free each chunk's transients before the next iteration.
    let free_chunk_transients = !tape.enabled;
    let mut chunks: Vec<TensorId> = Vec::new();
    let mut h0 = 0usize;
    while h0 < heads {
        let h1 = (h0 + chunk).min(heads);
        let live_before = free_chunk_transients
            .then(|| store.live_ids().into_iter().collect::<HashSet<TensorId>>());
        let q_chunk = slice(
            q,
            &[0, h0, 0, 0],
            &[batch, h1, q_len, head_dim],
            store,
            tape,
        )?;
        let k_chunk = slice(
            k,
            &[0, h0, 0, 0],
            &[batch, h1, kv_len, head_dim],
            store,
            tape,
        )?;
        let v_chunk = slice(
            v,
            &[0, h0, 0, 0],
            &[batch, h1, kv_len, head_dim],
            store,
            tape,
        )?;
        let out = causal_sdpa_with_q_start(q_chunk, k_chunk, v_chunk, q_start, store, tape)?;
        if let Some(live_before) = live_before {
            let keep = HashSet::from([out]);
            store.free_new_except(&live_before, &keep)?;
        }
        chunks.push(out);
        h0 = h1;
    }
    cat_heads(&chunks, store, tape)
}

/// Concatenate two rank-4 `[batch, heads, seq_i, head_dim]` tensors along the
/// SEQUENCE axis (axis 2). Taped: grad flows back to each input's seq slice (a
/// `requires_grad = false` input — e.g. the OPD frozen prompt KV — is simply
/// skipped in backward). Batch / heads / head_dim must match; only seq differs.
pub fn cat_seq(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut crate::Tape,
) -> Result<TensorId> {
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    for shape in [&a_shape, &b_shape] {
        if shape.len() != 4 {
            return Err(AutogradError::InvalidRank {
                expected: "4",
                got: shape.len(),
            });
        }
    }
    if a_shape[0] != b_shape[0] || a_shape[1] != b_shape[1] || a_shape[3] != b_shape[3] {
        return Err(AutogradError::ShapeMismatch {
            expected: a_shape.clone(),
            got: b_shape.clone(),
        });
    }
    let (batch, heads, head_dim) = (a_shape[0], a_shape[1], a_shape[3]);
    let (seq_a, seq_b) = (a_shape[2], b_shape[2]);
    let total_seq = seq_a + seq_b;
    let requires_grad = store.tensor(a)?.requires_grad || store.tensor(b)?.requires_grad;
    let device_path = [a, b].into_iter().all(|id| {
        store
            .tensor(id)
            .is_ok_and(|t| t.dirty != Dirty::Host && t.device_handle.is_some())
    });
    let output_id = if device_path {
        let a_handle = store.tensor(a)?.device_handle.as_ref().expect("device");
        let b_handle = store.tensor(b)?.device_handle.as_ref().expect("device");
        let (handle, shape) = store
            .backend()
            .concat_axis2(a_handle, &a_shape, b_handle, &b_shape)?;
        store.alloc_device_tensor(shape, handle)?
    } else {
        let mut data = vec![0.0_f32; batch * heads * total_seq * head_dim];
        let a_host = store.tensor_host(a)?;
        let b_host = store.tensor_host(b)?;
        for bh in 0..(batch * heads) {
            let out_base = bh * total_seq * head_dim;
            let a_len = seq_a * head_dim;
            let b_len = seq_b * head_dim;
            let a_base = bh * a_len;
            let b_base = bh * b_len;
            data[out_base..out_base + a_len].copy_from_slice(&a_host.data[a_base..a_base + a_len]);
            data[out_base + a_len..out_base + a_len + b_len]
                .copy_from_slice(&b_host.data[b_base..b_base + b_len]);
        }
        store.alloc(Tensor::new(
            data,
            vec![batch, heads, total_seq, head_dim],
            requires_grad,
        )?)
    };

    TapeEntry {
        op: BackwardOp::CatSeq,
        output_id,
        input_ids: smallvec![a, b],
        saved: SavedContext::CatSeqCtx {
            seq_counts: vec![seq_a, seq_b],
        },
    }
    .record(store, tape)?;
    Ok(output_id)
}

pub(crate) fn cat_seq_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::CatSeqCtx { seq_counts } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "cat_seq backward missing saved context",
        ));
    };
    let upstream = store.tensor_host(output_grad_id)?;
    let upstream_shape = upstream.shape.clone();
    if upstream_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: upstream_shape.len(),
        });
    }
    let (batch, heads, total_seq, head_dim) = (
        upstream_shape[0],
        upstream_shape[1],
        upstream_shape[2],
        upstream_shape[3],
    );

    let mut grads = GradPairs::new();
    let mut seq_offset = 0usize;
    for (i, &input_id) in entry.input_ids.iter().enumerate() {
        let seq_i = seq_counts[i];
        if !store.tensor(input_id)?.requires_grad {
            seq_offset += seq_i;
            continue;
        }
        let mut grad = vec![0.0_f32; batch * heads * seq_i * head_dim];
        for bh in 0..(batch * heads) {
            let src_base = (bh * total_seq + seq_offset) * head_dim;
            let dst_base = bh * seq_i * head_dim;
            let len = seq_i * head_dim;
            grad[dst_base..dst_base + len]
                .copy_from_slice(&upstream.data[src_base..src_base + len]);
        }
        let grad_id = store.alloc(Tensor::new(
            grad,
            vec![batch, heads, seq_i, head_dim],
            false,
        )?);
        grads.push((input_id, grad_id));
        seq_offset += seq_i;
    }
    Ok(grads)
}

fn causal_mask(seq_len: usize, store: &mut TensorStore) -> Result<TensorId> {
    let mut data = vec![0.0; seq_len * seq_len];
    for row in 0..seq_len {
        for col in (row + 1)..seq_len {
            data[(row * seq_len) + col] = f32::NEG_INFINITY;
        }
    }
    Ok(store.alloc(Tensor::new(data, vec![1, seq_len, seq_len], false)?))
}

fn causal_mask_window(
    q_len: usize,
    kv_len: usize,
    q_start: usize,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let mut data = vec![0.0; q_len * kv_len];
    for row in 0..q_len {
        let max_visible = q_start + row;
        for col in (max_visible + 1)..kv_len {
            data[(row * kv_len) + col] = f32::NEG_INFINITY;
        }
    }
    Ok(store.alloc(Tensor::new(data, vec![1, q_len, kv_len], false)?))
}

fn validate_attention_shapes(
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
) -> Result<()> {
    for shape in [q_shape, k_shape, v_shape] {
        if shape.len() != 4 {
            return Err(AutogradError::InvalidRank {
                expected: "4",
                got: shape.len(),
            });
        }
    }

    if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[2] != k_shape[2] || q_shape[2] != v_shape[2] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }

    Ok(())
}

fn validate_cached_attention_shapes(
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
    q_start: usize,
) -> Result<()> {
    for shape in [q_shape, k_shape, v_shape] {
        if shape.len() != 4 {
            return Err(AutogradError::InvalidRank {
                expected: "4",
                got: shape.len(),
            });
        }
    }

    if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3] {
        return Err(AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if k_shape[2] != v_shape[2] {
        return Err(AutogradError::ShapeMismatch {
            expected: k_shape.to_vec(),
            got: v_shape.to_vec(),
        });
    }
    if q_start + q_shape[2] > k_shape[2] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![q_start + q_shape[2]],
            got: vec![k_shape[2]],
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu_causal_sdpa_recompute_backward;

    fn det_vec(len: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = (state >> 8) as f32 / (u32::MAX >> 8) as f32;
                (unit - 0.5) * scale
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    // Head-chunking only regroups independent heads, so the chunked composed
    // forward must match the whole-tensor one exactly; and with the tape off
    // each chunk's scores/scaled/masked/probs must be freed before the next
    // (the writeback-OOM fix) — live growth is bounded by the chunk outputs.
    #[test]
    fn head_chunked_sdpa_matches_unchunked() -> Result<()> {
        const HEADS: usize = 4;
        const SEQ: usize = 8;
        const HEAD_DIM: usize = 16;
        let shape = vec![1, HEADS, SEQ, HEAD_DIM];
        let size: usize = shape.iter().product();

        let mut store = TensorStore::default();
        let mut tape = crate::Tape::new();
        tape.set_enabled(false);
        let q = store.alloc(Tensor::new(
            det_vec(size, 0x4d59_5df4, 0.1),
            shape.clone(),
            false,
        )?);
        let k = store.alloc(Tensor::new(
            det_vec(size, 0x7a35_2b19, 0.1),
            shape.clone(),
            false,
        )?);
        let v = store.alloc(Tensor::new(
            det_vec(size, 0x1f12_d9e7, 0.3),
            shape.clone(),
            false,
        )?);
        let reference_id = causal_sdpa_with_q_start(q, k, v, 0, &mut store, &mut tape)?;
        let reference = store.to_host(reference_id)?;

        for chunk in [2usize, 1] {
            let live_before = store.live_tensor_count();
            let out = sdpa_head_chunked(q, k, v, 0, chunk, &mut store, &mut tape)?;
            let got = store.to_host(out)?;
            let diff = max_abs_diff(&reference, &got);
            assert!(diff < 1e-4, "chunk={chunk} diverged: max_abs={diff:.3e}");
            let grew = store.live_tensor_count() - live_before;
            let n_chunks = HEADS.div_ceil(chunk);
            assert!(
                grew <= n_chunks + 4,
                "chunk={chunk}: {grew} tensors live; per-chunk SDPA transients not freed"
            );
        }
        Ok(())
    }

    // The chunked device backward must be numerically identical to the unchunked
    // host reference (`cpu_causal_sdpa_recompute_backward`), the writeback-OOM fix
    // gate: q-chunking only reorders an exact sum, so any divergence is a bug.
    #[test]
    fn chunked_sdpa_backward_matches_unchunked() -> Result<()> {
        const BATCH: usize = 1;
        const HEADS: usize = 3;
        const SEQ: usize = 20;
        const HEAD_DIM: usize = 8;
        let shape = vec![BATCH, HEADS, SEQ, HEAD_DIM];
        let size: usize = shape.iter().product();

        let q_data = det_vec(size, 0x4d59_5df4, 0.4);
        let k_data = det_vec(size, 0x7a35_2b19, 0.4);
        let v_data = det_vec(size, 0x1f12_d9e7, 0.6);
        let up_data = det_vec(size, 0x2c91_7b03, 0.5);

        // Host reference grads (unchunked oracle).
        let (ref_gq, ref_gk, ref_gv) = cpu_causal_sdpa_recompute_backward(
            &q_data, &k_data, &v_data, &up_data, &shape, true, true, true,
        )?;
        let ref_gq = ref_gq.expect("grad_q");
        let ref_gk = ref_gk.expect("grad_k");
        let ref_gv = ref_gv.expect("grad_v");

        // Exercise both a multi-chunk run (q_chunk=3 → 7 chunks over SEQ=20) and
        // the single-chunk run (q_chunk=SEQ); both must match the oracle exactly.
        for q_chunk in [3usize, SEQ] {
            let mut store = TensorStore::default();
            let q = store.alloc(Tensor::new(q_data.clone(), shape.clone(), true)?);
            let k = store.alloc(Tensor::new(k_data.clone(), shape.clone(), true)?);
            let v = store.alloc(Tensor::new(v_data.clone(), shape.clone(), true)?);
            let upstream = store.alloc(Tensor::new(up_data.clone(), shape.clone(), false)?);
            // Force the device path (Dirty::Both + handle) like the writeback runtime.
            for id in [q, k, v, upstream] {
                store.ensure_device(id)?;
            }

            let grads = causal_sdpa_recompute_backward_device_chunked(
                q, k, v, upstream, &shape, &shape, 0, true, true, true, q_chunk, &mut store,
            )?;

            let find = |target: TensorId| -> TensorId {
                grads
                    .iter()
                    .find_map(|(id, gid)| (*id == target).then_some(*gid))
                    .expect("grad present")
            };
            let got_gq = store.to_host(find(q))?;
            let got_gk = store.to_host(find(k))?;
            let got_gv = store.to_host(find(v))?;

            let dq = max_abs_diff(&ref_gq, &got_gq);
            let dk = max_abs_diff(&ref_gk, &got_gk);
            let dv = max_abs_diff(&ref_gv, &got_gv);
            eprintln!(
                "chunked_sdpa_backward q_chunk={q_chunk} dq={dq:.3e} dk={dk:.3e} dv={dv:.3e}"
            );
            assert!(
                dq < 1e-4,
                "q_chunk={q_chunk} grad_q diverged: max_abs={dq:.3e}"
            );
            assert!(
                dk < 1e-4,
                "q_chunk={q_chunk} grad_k diverged: max_abs={dk:.3e}"
            );
            assert!(
                dv < 1e-4,
                "q_chunk={q_chunk} grad_v diverged: max_abs={dv:.3e}"
            );
        }

        Ok(())
    }

    // Selective-grad paths (e.g. frozen K/V) must also match the oracle and not
    // touch the un-requested grad accumulators.
    #[test]
    fn chunked_sdpa_backward_grad_q_only() -> Result<()> {
        const SEQ: usize = 14;
        const HEAD_DIM: usize = 8;
        let shape = vec![1usize, 2, SEQ, HEAD_DIM];
        let size: usize = shape.iter().product();
        let q_data = det_vec(size, 0xa1b2_c3d4, 0.4);
        let k_data = det_vec(size, 0x1357_9bdf, 0.4);
        let v_data = det_vec(size, 0x2468_ace0, 0.6);
        let up_data = det_vec(size, 0x0f0f_0f0f, 0.5);

        let (ref_gq, _, _) = cpu_causal_sdpa_recompute_backward(
            &q_data, &k_data, &v_data, &up_data, &shape, true, false, false,
        )?;
        let ref_gq = ref_gq.expect("grad_q");

        let mut store = TensorStore::default();
        let q = store.alloc(Tensor::new(q_data, shape.clone(), true)?);
        let k = store.alloc(Tensor::new(k_data, shape.clone(), true)?);
        let v = store.alloc(Tensor::new(v_data, shape.clone(), true)?);
        let upstream = store.alloc(Tensor::new(up_data, shape.clone(), false)?);
        for id in [q, k, v, upstream] {
            store.ensure_device(id)?;
        }

        let grads = causal_sdpa_recompute_backward_device_chunked(
            q, k, v, upstream, &shape, &shape, 0, true, false, false, 4, &mut store,
        )?;
        assert_eq!(grads.len(), 1, "only grad_q requested");
        let gid = grads[0].1;
        let got_gq = store.to_host(gid)?;
        let dq = max_abs_diff(&ref_gq, &got_gq);
        assert!(dq < 1e-4, "grad_q-only diverged: max_abs={dq:.3e}");
        Ok(())
    }

    // Rectangular backward (q_len < kv_len, q_start > 0) — the context-parallel
    // shape where a rank's local q attends the gathered KV prefix. Oracle = the
    // composed primitives (matmul/softmax/matmul via causal_sdpa_with_q_start on an
    // enabled tape, each with its own backward); the chunked device backward must
    // match it bit-for-bit up to add-order (q-chunking only reorders the sum).
    #[test]
    fn chunked_sdpa_backward_rectangular_matches_composed() -> Result<()> {
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 8;
        const Q_LEN: usize = 4;
        const KV_LEN: usize = 10;
        const Q_START: usize = KV_LEN - Q_LEN; // last shard: q rows 6..10 over kv 0..10
        let q_shape = vec![1usize, HEADS, Q_LEN, HEAD_DIM];
        let kv_shape = vec![1usize, HEADS, KV_LEN, HEAD_DIM];
        let q_data = det_vec(HEADS * Q_LEN * HEAD_DIM, 0x51a9_3c7e, 0.4);
        let k_data = det_vec(HEADS * KV_LEN * HEAD_DIM, 0x2e7d_44b1, 0.4);
        let v_data = det_vec(HEADS * KV_LEN * HEAD_DIM, 0x6b03_1af2, 0.6);
        let up_data = det_vec(HEADS * Q_LEN * HEAD_DIM, 0x1c4f_9082, 0.5);

        // Oracle: composed forward on an enabled tape + autograd backward.
        let (ref_gq, ref_gk, ref_gv) = {
            let mut store = TensorStore::default();
            let mut tape = crate::Tape::new();
            let q = store.alloc(Tensor::new(q_data.clone(), q_shape.clone(), true)?);
            let k = store.alloc(Tensor::new(k_data.clone(), kv_shape.clone(), true)?);
            let v = store.alloc(Tensor::new(v_data.clone(), kv_shape.clone(), true)?);
            let out = causal_sdpa_with_q_start(q, k, v, Q_START, &mut store, &mut tape)?;
            // Seed the same upstream grad the device path receives (loss = <out, up>).
            let up = store.alloc(Tensor::new(up_data.clone(), q_shape.clone(), false)?);
            let weighted = crate::ops::mul(out, up, &mut store, &mut tape)?;
            let loss = crate::ops::sum(weighted, &mut store, &mut tape)?;
            let grads = tape.backward(loss, &mut store)?;
            (
                store.to_host(grads[&q])?,
                store.to_host(grads[&k])?,
                store.to_host(grads[&v])?,
            )
        };

        // Chunked device backward on the same rectangular case (q_chunk=3 → 2 chunks).
        let mut store = TensorStore::default();
        let q = store.alloc(Tensor::new(q_data, q_shape.clone(), true)?);
        let k = store.alloc(Tensor::new(k_data, kv_shape.clone(), true)?);
        let v = store.alloc(Tensor::new(v_data, kv_shape.clone(), true)?);
        let upstream = store.alloc(Tensor::new(up_data, q_shape.clone(), false)?);
        for id in [q, k, v, upstream] {
            store.ensure_device(id)?;
        }
        let grads = causal_sdpa_recompute_backward_device_chunked(
            q, k, v, upstream, &q_shape, &kv_shape, Q_START, true, true, true, 3, &mut store,
        )?;
        let find = |target: TensorId| -> TensorId {
            grads
                .iter()
                .find_map(|(id, gid)| (*id == target).then_some(*gid))
                .expect("grad present")
        };
        let dq = max_abs_diff(&ref_gq, &store.to_host(find(q))?);
        let dk = max_abs_diff(&ref_gk, &store.to_host(find(k))?);
        let dv = max_abs_diff(&ref_gv, &store.to_host(find(v))?);
        eprintln!("rect_sdpa_backward dq={dq:.3e} dk={dk:.3e} dv={dv:.3e}");
        assert!(dq < 1e-4, "rect grad_q diverged: {dq:.3e}");
        assert!(dk < 1e-4, "rect grad_k diverged: {dk:.3e}");
        assert!(dv < 1e-4, "rect grad_v diverged: {dv:.3e}");
        Ok(())
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn causal_sdpa_recompute_with_q_start_matches_composed() -> Result<()> {
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 8;
        const Q_LEN: usize = 5;
        const KV_LEN: usize = 12;
        const Q_START: usize = KV_LEN - Q_LEN;
        let q_shape = vec![1usize, HEADS, Q_LEN, HEAD_DIM];
        let kv_shape = vec![1usize, HEADS, KV_LEN, HEAD_DIM];
        let q_data = det_vec(HEADS * Q_LEN * HEAD_DIM, 0x7c2e_11a5, 0.4);
        let k_data = det_vec(HEADS * KV_LEN * HEAD_DIM, 0x3b91_ee02, 0.4);
        let v_data = det_vec(HEADS * KV_LEN * HEAD_DIM, 0x59d0_74c3, 0.6);
        let up_data = det_vec(HEADS * Q_LEN * HEAD_DIM, 0x0a5f_3d18, 0.5);

        let run = |owning: bool| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
            let mut store = TensorStore::default();
            let mut tape = crate::Tape::new();
            let q = store.alloc(Tensor::new(q_data.clone(), q_shape.clone(), true)?);
            let k = store.alloc(Tensor::new(k_data.clone(), kv_shape.clone(), true)?);
            let v = store.alloc(Tensor::new(v_data.clone(), kv_shape.clone(), true)?);
            let out = if owning {
                causal_sdpa_recompute_with_q_start(q, k, v, Q_START, &mut store, &mut tape)?
            } else {
                causal_sdpa_with_q_start(q, k, v, Q_START, &mut store, &mut tape)?
            };
            let out_host = store.to_host(out)?;
            let up = store.alloc(Tensor::new(up_data.clone(), q_shape.clone(), false)?);
            let weighted = crate::ops::mul(out, up, &mut store, &mut tape)?;
            let loss = crate::ops::sum(weighted, &mut store, &mut tape)?;
            let grads = tape.backward(loss, &mut store)?;
            Ok((
                out_host,
                store.to_host(grads[&q])?,
                store.to_host(grads[&k])?,
                store.to_host(grads[&v])?,
            ))
        };

        let (ref_out, ref_gq, ref_gk, ref_gv) = run(false)?;
        let (out, gq, gk, gv) = run(true)?;
        assert!(max_abs_diff(&ref_out, &out) < 1e-4);
        assert!(max_abs_diff(&ref_gq, &gq) < 1e-4);
        assert!(max_abs_diff(&ref_gk, &gk) < 1e-4);
        assert!(max_abs_diff(&ref_gv, &gv) < 1e-4);
        Ok(())
    }
}
