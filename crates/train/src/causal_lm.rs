use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

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

    fn adapter_name_map(&self) -> HashMap<&'static str, TensorId> {
        HashMap::new()
    }

    fn materialized_param_name_map(
        &self,
        _store: &mut TensorStore,
        _tape: &mut Tape,
    ) -> Result<HashMap<&'static str, TensorId>> {
        Ok(self.param_name_map())
    }
}

pub fn build_registry<M: CausalLm>(model: &M) -> SafetensorsRegistry {
    let mut registry = SafetensorsRegistry::new();
    for (name, tensor_id) in model.param_name_map() {
        registry.insert(name, tensor_id);
    }
    registry
}

pub fn build_adapter_registry<M: CausalLm>(model: &M) -> SafetensorsRegistry {
    let mut registry = SafetensorsRegistry::new();
    for (name, tensor_id) in model.adapter_name_map() {
        registry.insert(name, tensor_id);
    }
    registry
}

pub fn build_materialized_registry<M: CausalLm>(
    model: &M,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<SafetensorsRegistry> {
    let mut registry = SafetensorsRegistry::new();
    for (name, tensor_id) in model.materialized_param_name_map(store, tape)? {
        registry.insert(name, tensor_id);
    }
    Ok(registry)
}

pub fn save_materialized_registry<M: CausalLm>(
    model: &M,
    store: &mut TensorStore,
    tape: &mut Tape,
    path: &Path,
    bf16: bool,
) -> Result<()> {
    let mut registry = SafetensorsRegistry::new();
    for (name, tensor_id) in model.materialized_param_name_map(store, tape)? {
        registry.insert(name, tensor_id);
    }
    if bf16 {
        registry.save_from_bf16(store, path)
    } else {
        registry.save_from(store, path)
    }
}

pub fn trainable_params<M: CausalLm>(model: &M, store: &TensorStore) -> Vec<TensorId> {
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

pub fn trainable_param_name_map<M: CausalLm>(
    model: &M,
    store: &TensorStore,
) -> Vec<(TensorId, String)> {
    let trainable: HashSet<TensorId> = trainable_params(model, store).into_iter().collect();
    let mut named = model
        .adapter_name_map()
        .into_iter()
        .chain(model.param_name_map())
        .filter(|(_, tensor_id)| trainable.contains(tensor_id))
        .map(|(name, tensor_id)| (tensor_id, name.to_string()))
        .collect::<Vec<_>>();
    named.sort_unstable_by(|(id_a, name_a), (id_b, name_b)| {
        name_a.cmp(name_b).then_with(|| id_a.cmp(id_b))
    });
    named.dedup_by(|(id_a, _), (id_b, _)| id_a == id_b);
    named
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
