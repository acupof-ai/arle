use std::{collections::HashSet, sync::Arc};

use smallvec::SmallVec;

use crate::{
    Result,
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
