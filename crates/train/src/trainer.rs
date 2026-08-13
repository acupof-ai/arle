//! Free post-backward cleanup helpers shared by the OPD training loops.

pub use crate::grad_clip::clip_grad_norm;
pub use crate::loss::cross_entropy_loss;

use std::collections::HashSet;

use autograd::{Tape, TensorId, TensorStore};

/// Post-backward cleanup: prune the store down to `keep_extra ∪ params ∪ grads`.
///
/// Exposed `pub` so OPD eval closures that produce multi-forward activations
/// can prune the store between windows. Note: this unconditionally re-enables
/// the tape, which is correct for the post-backward path but NOT for an eval
/// loop that wants the tape disabled across windows — the caller must
/// re-disable with `tape.set_enabled(false)` after each invocation in that case.
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
