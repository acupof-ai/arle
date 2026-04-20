use std::collections::{HashMap, HashSet};

use autograd::{Result, SafetensorsRegistry, Tape, TensorId, TensorStore};

use crate::{policy::GrpoPolicy, trainer::extend_keep_with_params_and_grads};

pub trait CausalLm: GrpoPolicy {
    fn forward_with_positions(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> Result<TensorId>;

    fn param_name_map(&self) -> HashMap<&'static str, TensorId>;
}

pub fn build_registry<M: CausalLm>(model: &M) -> SafetensorsRegistry {
    let mut registry = SafetensorsRegistry::new();
    for (name, tensor_id) in model.param_name_map() {
        registry.insert(name, tensor_id);
    }
    registry
}

pub fn trainable_params<M: CausalLm>(model: &M, store: &TensorStore) -> Vec<TensorId> {
    let mut params = model
        .param_name_map()
        .into_values()
        .collect::<HashSet<_>>()
        .into_iter()
        .filter(|tensor_id| {
            store
                .get(*tensor_id)
                .is_some_and(|tensor| tensor.requires_grad)
        })
        .collect::<Vec<_>>();
    params.sort_unstable();
    params
}

pub fn live_tensor_ids(store: &TensorStore) -> HashSet<TensorId> {
    store
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(tensor_id, slot)| slot.as_ref().map(|_| tensor_id))
        .collect()
}

pub fn retained_ids(
    model_ids: &HashSet<TensorId>,
    params: &[TensorId],
    store: &TensorStore,
) -> HashSet<TensorId> {
    let mut keep = model_ids.clone();
    extend_keep_with_params_and_grads(&mut keep, params.iter().copied(), store);
    keep
}
