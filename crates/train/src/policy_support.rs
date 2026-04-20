use std::collections::HashSet;

use autograd::{TensorId, TensorStore};

use crate::policy::GrpoPolicy;
use crate::trainer::extend_keep_with_params_and_grads;

pub fn retained_ids<P: GrpoPolicy>(models: &[&P], store: &TensorStore) -> HashSet<TensorId> {
    let mut keep = HashSet::new();
    for model in models {
        extend_keep_with_params_and_grads(&mut keep, model.all_parameter_ids(), store);
    }
    keep
}

pub fn trainable_param_ids<P: GrpoPolicy>(model: &P, store: &TensorStore) -> Vec<TensorId> {
    let mut params = model
        .all_parameter_ids()
        .into_iter()
        .filter(|tensor_id| {
            store
                .get(*tensor_id)
                .is_some_and(|tensor| tensor.requires_grad)
        })
        .collect::<Vec<_>>();
    params.sort_unstable();
    params.dedup();
    params
}
