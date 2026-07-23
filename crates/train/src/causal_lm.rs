use std::{collections::HashSet, path::Path};

use autograd::{Result, SafetensorsRegistry, Tape, TensorId, TensorStore};

use crate::qwen35::{Qwen35Model, qwen35_to_autograd};

pub fn save_materialized_registry(
    model: &Qwen35Model,
    store: &mut TensorStore,
    _tape: &mut Tape,
    path: &Path,
    bf16: bool,
) -> Result<()> {
    let mut registry = SafetensorsRegistry::new();
    for (name, tensor_id) in model
        .materialized_param_name_map(store)
        .map_err(qwen35_to_autograd)?
    {
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
