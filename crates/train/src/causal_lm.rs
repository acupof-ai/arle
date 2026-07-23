use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use autograd::{Result, SafetensorsRegistry, Tape, TensorId, TensorStore};

pub trait CausalLm {
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

    fn all_parameter_ids(&self) -> Vec<TensorId>;
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

pub fn live_tensor_ids(store: &TensorStore) -> HashSet<TensorId> {
    store
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(tensor_id, slot)| slot.as_ref().map(|_| tensor_id))
        .collect()
}
