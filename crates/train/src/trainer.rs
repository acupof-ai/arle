pub use crate::grad_clip::clip_grad_norm;
pub use crate::loss::cross_entropy_loss;

use std::collections::HashSet;

use autograd::{Tape, TensorId, TensorStore};

/// Unconditionally re-enables the tape — correct post-backward, but an eval
/// loop that wants the tape disabled across windows must re-disable with
/// `tape.set_enabled(false)` after each call.
pub fn cleanup_after_backward(
    store: &mut TensorStore,
    tape: &mut Tape,
    params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
) {
    tape.entries.clear();
    tape.set_enabled(true);
    let mut keep = keep_extra.clone();
    extend_keep_with_params_and_grads(&mut keep, params.iter().copied(), store);
    store.retain_ids(&keep);
}

pub fn retained_param_and_grad_ids(params: &[TensorId], store: &TensorStore) -> HashSet<TensorId> {
    let mut keep = HashSet::with_capacity(params.len() * 2);
    extend_keep_with_params_and_grads(&mut keep, params.iter().copied(), store);
    keep
}

pub fn extend_keep_with_params_and_grads<I>(
    keep: &mut HashSet<TensorId>,
    params: I,
    store: &TensorStore,
) where
    I: IntoIterator<Item = TensorId>,
{
    for param_id in params {
        keep.insert(param_id);
        if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
            keep.insert(grad_id);
        }
    }
}
