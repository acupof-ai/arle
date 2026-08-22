//! Opaque serializable codec for [`AdamW`] moments.
//!
//! Callers own the param-name ↔ `TensorId` mapping; the codec works off a
//! `&[(TensorId, String)]` handed in at export/import time. The count of
//! internal entries that lacked a name during `export_state` is returned via
//! [`AdamWState::skipped_export`] (not stderr) so callers control logging.

use serde::{Deserialize, Serialize};

use crate::{TensorId, optim::AdamW};

/// `shape` is captured at export time.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AdamWParamState {
    pub name: String,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub shape: Vec<usize>,
}

/// `skipped_export` counts internal AdamW entries omitted from the last export
/// (the caller's `names` slice did not cover them); `0` after deserialization.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AdamWState {
    pub step: u64,
    pub params: Vec<AdamWParamState>,
    /// Round-trips through serde so callers can propagate it across save/load.
    #[serde(default)]
    pub skipped_export: usize,
}

impl AdamW {
    /// Tracked `TensorId`s missing from `names` are skipped; the count lands in
    /// [`AdamWState::skipped_export`]. Names with no state yet are omitted.
    pub fn export_state(&self, names: &[(TensorId, String)]) -> AdamWState {
        let params: Vec<AdamWParamState> = names
            .iter()
            .filter_map(|(id, name)| {
                let (m, v) = self.moments_host(*id)?;
                let shape = self.param_shape(*id).unwrap_or_else(|| vec![m.len()]);
                Some(AdamWParamState {
                    name: name.clone(),
                    m,
                    v,
                    shape,
                })
            })
            .collect();

        let total_internal = self.state_len();
        let skipped_export = total_internal.saturating_sub(params.len());

        AdamWState {
            step: self.step_count() as u64,
            params,
            skipped_export,
        }
    }

    /// Returns the count of params restored. Entries whose `name` isn't in
    /// `names` are skipped (caller-side mapping is authoritative). Shape
    /// mismatch is a hard error.
    pub fn import_state(
        &mut self,
        state: &AdamWState,
        names: &[(TensorId, String)],
    ) -> anyhow::Result<usize> {
        use std::collections::HashMap;

        let lookup: HashMap<&str, TensorId> = names
            .iter()
            .map(|(id, name)| (name.as_str(), *id))
            .collect();

        let mut restored = 0usize;
        for param in &state.params {
            let Some(&id) = lookup.get(param.name.as_str()) else {
                continue;
            };

            if let Some(existing_shape) = self.param_shape(id)
                && existing_shape != param.shape
            {
                anyhow::bail!(
                    "AdamW shape mismatch for '{}' (id {id}): existing {:?}, loaded {:?}",
                    param.name,
                    existing_shape,
                    param.shape,
                );
            }

            let expected_len: usize = if param.shape.is_empty() {
                1
            } else {
                param.shape.iter().product()
            };
            if param.m.len() != expected_len || param.v.len() != expected_len {
                anyhow::bail!(
                    "AdamW moment length mismatch for '{}' (id {id}): shape {:?} => {} elems, m {} v {}",
                    param.name,
                    param.shape,
                    expected_len,
                    param.m.len(),
                    param.v.len(),
                );
            }

            self.set_state(id, param.m.clone(), param.v.clone(), param.shape.clone());
            restored += 1;
        }

        self.set_step_count(state.step as i32);
        Ok(restored)
    }
}
