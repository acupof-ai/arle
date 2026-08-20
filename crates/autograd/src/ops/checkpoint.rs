use std::{collections::HashSet, sync::Arc};

use smallvec::SmallVec;

use crate::{
    AutogradError, Result,
    ops::chunk_accum::SeqAccum,
    tape::{BackwardOp, CheckpointFn, SavedContext, Tape, TapeEntry},
    tensor::{TensorId, TensorStore},
};

pub fn checkpoint<F>(
    input_ids: Vec<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
    replay: F,
) -> Result<TensorId>
where
    F: Fn(&mut TensorStore, &mut Tape, &[TensorId]) -> Result<TensorId> + Send + Sync + 'static,
{
    if !tape.enabled {
        return replay(store, tape, &input_ids);
    }

    let input_ids = input_ids.into_iter().collect::<SmallVec<[TensorId; 2]>>();
    let live_before = store.live_ids().into_iter().collect::<HashSet<_>>();
    let was_enabled = tape.enabled;
    tape.enabled = false;
    let forward_result = replay(store, tape, &input_ids);
    tape.enabled = was_enabled;
    let output_id = match forward_result {
        Ok(output_id) => output_id,
        Err(err) => {
            let _ = store.free_new_except(&live_before, &HashSet::new());
            return Err(err);
        }
    };

    let mut keep = HashSet::from([output_id]);
    keep.extend(input_ids.iter().copied());
    store.free_new_except(&live_before, &keep)?;

    if store.any_requires_grad(&input_ids) {
        // Offload the saved inputs to host RAM: each layer's input is the prior
        // layer's output and is untouched until backward replay (which re-fetches
        // via ensure_device), so this frees the ~30 GB of grad-checkpoints a long
        // training forward would otherwise pin in VRAM.
        if tape.offload_checkpoints {
            let skip_hidden = tape.take_skip_next_checkpoint_input_offload();
            for (idx, &id) in input_ids.iter().enumerate() {
                if skip_hidden && idx == 0 {
                    continue;
                }
                store.offload_checkpoint_to_host(id)?;
            }
        }
        let checkpoint_fn: CheckpointFn = Arc::new(move |st, tp, _start, inp| replay(st, tp, inp));
        let function_id = tape.register_checkpoint_fn(checkpoint_fn);
        TapeEntry {
            op: BackwardOp::Checkpoint,
            output_id,
            input_ids,
            saved: SavedContext::CheckpointCtx { function_id },
        }
        .record(store, tape)?;
    } else if tape.offload_checkpoints {
        let _ = tape.take_skip_next_checkpoint_input_offload();
        // FROZEN group (no trainable param → no backward replay, no tape entry):
        // its input hidden (`input_ids[0]`, = the PRIOR group's output) is the
        // unbounded leak. It's in `keep` here (this group's saved input) so the
        // free above can't touch it, and it then lands in EVERY later group's
        // `live_before` — so no later `free_new_except` reclaims it either.
        // ~+156 MiB/group at seq=8000 (`[1, seq, hidden]`), unbounded over the
        // frozen prefix → writeback OOM. Nothing reads it once this group's
        // output exists (frozen ⇒ no replay), so drop its DEVICE residency. Only
        // `input_ids[0]` (the hidden); `input_ids[1..]` are shared frozen weights.
        // Gated to the OPD offload path; the default forward never sets
        // `offload_checkpoints`, so it stays byte-identical.
        if let Some(&hidden_id) = input_ids.first()
            && hidden_id != output_id
        {
            store.drop_device_residency(hidden_id)?;
        }
    }

    Ok(output_id)
}

/// Run `num_layers` sequential layers, checkpointing one layer per group.
/// Each group's saved inputs are `[hidden, ...deduped params of the group's
/// layers]`; offload (if enabled on the tape) then moves K layers' inputs to
/// host in one shot. `detach_at`, if set, forces a group boundary there and
/// detaches the hidden at that index (preserving a model's frozen/LoRA
/// boundary) — no group spans it, so it stays numerically exact.
/// `layer_fn(idx, hidden, store, tape)` runs layer `idx`; `layer_params(idx)`
/// returns that layer's trainable param ids.
pub fn checkpoint_sequential<FF, PF>(
    input: TensorId,
    num_layers: usize,
    detach_at: Option<usize>,
    store: &mut TensorStore,
    tape: &mut Tape,
    layer_params: PF,
    layer_fn: FF,
) -> Result<TensorId>
where
    FF: Fn(usize, TensorId, &mut TensorStore, &mut Tape) -> Result<TensorId>
        + Clone
        + Send
        + Sync
        + 'static,
    PF: Fn(usize) -> Vec<TensorId>,
{
    let trace_vram = trace_checkpoint_group_vram();
    let mut hidden = input;
    let mut li = 0;
    while li < num_layers {
        let mut end = (li + 1).min(num_layers);
        // Don't let a group span the frozen/LoRA boundary.
        if let Some(b) = detach_at
            && li < b
            && b < end
        {
            end = b;
        }
        // Boundary detach, outside the recompute (matches per-layer detach).
        if detach_at == Some(li) {
            hidden = store.detach(hidden)?;
        }

        // Saved inputs = [hidden, ...deduped group param ids].
        let mut input_ids = vec![hidden];
        for idx in li..end {
            for id in layer_params(idx) {
                if !input_ids.contains(&id) {
                    input_ids.push(id);
                }
            }
        }

        let f = layer_fn.clone();
        let (start, stop) = (li, end);
        if tape.offload_checkpoints && end == num_layers {
            tape.set_skip_next_checkpoint_input_offload(true);
        }
        let checkpoint_fn = trace_vram.then(|| tape.checkpoint_fn_count());
        hidden = checkpoint(input_ids, store, tape, move |s, t, inp| {
            let mut h = *inp.first().ok_or(AutogradError::TapeInvariant(
                "checkpoint_sequential missing hidden input",
            ))?;
            for idx in start..stop {
                h = f(idx, h, s, t)?;
            }
            Ok(h)
        })?;
        if let Some(checkpoint_fn) = checkpoint_fn
            && tape.checkpoint_fn_count() == checkpoint_fn + 1
        {
            eprintln!("[checkpoint-map] checkpoint_fn={checkpoint_fn} layers={start}..{stop}");
        }
        li = end;
        if trace_vram && let Some((free, total)) = store.backend().device_mem_info() {
            let pool = store.backend().mem_pool_stats();
            let fmt_pool = |value: Option<u64>| {
                value.map_or_else(|| "n/a".to_string(), |bytes| format!("{}MiB", bytes >> 20))
            };
            eprintln!(
                "[ckpt-group-vram] after_group end={end}/{num_layers} used={}MiB free={}MiB \
                 pool_reserved={} pool_used_current={} live_tensors={}",
                (total - free) >> 20,
                free >> 20,
                fmt_pool(pool.map(|(reserved, _)| reserved)),
                fmt_pool(pool.map(|(_, used)| used)),
                store.live_tensor_count(),
            );
        }
    }
    Ok(hidden)
}

/// Trim when the pool hoards more than 2 GB it is not re-cutting.
pub(crate) fn trim_if_hoarding(store: &TensorStore) -> Result<()> {
    const HOARD_TRIM_BYTES: u64 = 2 << 30;
    if let Some((reserved, used)) = store.backend().mem_pool_stats()
        && reserved.saturating_sub(used) > HOARD_TRIM_BYTES
    {
        store.backend().trim_memory_pool()?;
    }
    Ok(())
}

fn trace_checkpoint_group_vram() -> bool {
    std::env::var("ARLE_OPD_VRAM_TRACE").is_ok_and(|v| v != "0" && v != "false")
}

/// Chunk a position-wise block in forward and backward.
pub fn checkpoint_seq_chunked<F>(
    input: TensorId,
    param_ids: Vec<TensorId>,
    chunk: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
    replay: F,
) -> Result<TensorId>
where
    F: Fn(&mut TensorStore, &mut Tape, usize, &[TensorId]) -> Result<TensorId>
        + Send
        + Sync
        + 'static,
{
    let mut input_ids = vec![input];
    input_ids.extend(param_ids);
    if chunk == 0 {
        return replay(store, tape, 0, &input_ids);
    }
    let shape = store.tensor(input)?.shape.clone();
    let &[batch, seq, dim] = shape.as_slice() else {
        return Err(AutogradError::InvalidRank {
            expected: "rank-3 [batch, seq, dim]",
            got: shape.len(),
        });
    };

    let requires_grad = store.any_requires_grad(&input_ids);
    let outer_enabled = tape.enabled;
    let live_before = store.live_ids().into_iter().collect::<HashSet<_>>();
    tape.enabled = false;
    let forward = (|| {
        // Output width comes from the first chunk — the block may project dim.
        let mut out: Option<SeqAccum> = None;
        for start in (0..seq).step_by(chunk) {
            let end = (start + chunk).min(seq);
            let x = crate::ops::slice(input, &[0, start, 0], &[batch, end, dim], store, tape)?;
            let mut chunk_inputs = vec![x];
            chunk_inputs.extend_from_slice(&input_ids[1..]);
            let y = replay(store, tape, start, &chunk_inputs)?;
            if out.is_none() {
                let out_dim =
                    *store
                        .tensor(y)?
                        .shape
                        .last()
                        .ok_or(AutogradError::TapeInvariant(
                            "seq-chunked block returned rank-0 output",
                        ))?;
                let acc = SeqAccum::new(vec![batch, seq, out_dim], 1, store)?;
                // Not left to `record` below: that only fires on an enabled outer tape.
                store.set_requires_grad(acc.id(), requires_grad)?;
                out = Some(acc);
            }
            let acc = out.as_mut().expect("set above");
            acc.write_rows(start, y, store)?;
            let keep = HashSet::from([acc.id()]);
            store.free_new_except(&live_before, &keep)?;
            // The pool does not re-cut freed chunk blocks for the next chunk
            // (reserved +33 GB for +7 GB live at 131,072, sync or not); trim
            // whenever the hoard passes the threshold so reserved stays at one
            // chunk's working set.
            trim_if_hoarding(store)?;
        }
        let out = out.ok_or(AutogradError::TapeInvariant(
            "seq-chunked block on empty seq",
        ))?;
        Ok(out.finish())
    })();
    tape.enabled = outer_enabled;
    let output_id = match forward {
        Ok(id) => id,
        Err(err) => {
            let _ = store.free_new_except(&live_before, &HashSet::new());
            return Err(err);
        }
    };
    let out_dim = *store.tensor(output_id)?.shape.last().unwrap_or(&dim);
    let mut keep = HashSet::from([output_id]);
    keep.extend(input_ids.iter().copied());
    store.free_new_except(&live_before, &keep)?;

    if outer_enabled && requires_grad {
        let effective_chunk = chunk.min(seq);
        let function_id = tape.register_checkpoint_fn(Arc::new(replay));
        TapeEntry {
            op: BackwardOp::SeqChunkedRecompute,
            output_id,
            input_ids: input_ids.into_iter().collect(),
            saved: SavedContext::SeqChunkedRecomputeCtx {
                function_id,
                batch,
                seq,
                dim,
                out_dim,
                chunk: effective_chunk,
            },
        }
        .record(store, tape)?;
    }
    Ok(output_id)
}
