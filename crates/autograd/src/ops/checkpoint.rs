use std::{collections::HashSet, sync::Arc};

use smallvec::SmallVec;

use crate::{
    AutogradError, Result,
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

    let requires_grad = input_ids
        .iter()
        .any(|&id| store.get(id).is_some_and(|tensor| tensor.requires_grad));
    if requires_grad {
        // Offload the saved inputs to host RAM: each layer's input is the prior
        // layer's output and is untouched until backward replay (which re-fetches
        // via ensure_device), so this frees the ~30 GB of grad-checkpoints a long
        // training forward would otherwise pin in VRAM.
        if tape.offload_checkpoints {
            for &id in &input_ids {
                store.offload_to_host(id)?;
            }
        }
        let checkpoint_fn: CheckpointFn = Arc::new(replay);
        let function_id = tape.register_checkpoint_fn(checkpoint_fn);
        tape.record(TapeEntry {
            op: BackwardOp::Checkpoint,
            output_id,
            input_ids,
            saved: SavedContext::CheckpointCtx { function_id },
        });
    }

    Ok(output_id)
}

/// Run `num_layers` sequential layers, checkpointing them in groups of
/// `group_size` (vs one checkpoint per layer). Each group's saved inputs are
/// `[hidden, ...deduped params of the group's layers]`; offload (if enabled on
/// the tape) then moves K layers' inputs to host in one shot. `detach_at`, if
/// set, forces a group boundary there and detaches the hidden at that index
/// (preserving a model's frozen/LoRA boundary) — no group spans it, so it stays
/// numerically exact. `layer_fn(idx, hidden, store, tape)` runs layer `idx`;
/// `layer_params(idx)` returns that layer's trainable param ids.
pub fn checkpoint_sequential<FF, PF>(
    input: TensorId,
    num_layers: usize,
    group_size: usize,
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
    let mut hidden = input;
    let mut li = 0;
    while li < num_layers {
        let mut end = (li + group_size.max(1)).min(num_layers);
        // Don't let a group span the frozen/LoRA boundary.
        if let Some(b) = detach_at {
            if li < b && b < end {
                end = b;
            }
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
        hidden = checkpoint(input_ids, store, tape, move |s, t, inp| {
            let mut h = *inp.first().ok_or(AutogradError::TapeInvariant(
                "checkpoint_sequential missing hidden input",
            ))?;
            for idx in start..stop {
                h = f(idx, h, s, t)?;
            }
            Ok(h)
        })?;
        li = end;
    }
    Ok(hidden)
}
