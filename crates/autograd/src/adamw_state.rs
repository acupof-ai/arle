//! Opaque serializable codec for [`AdamW`] moments.
//!
//! Callers own their param-name ↔ `TensorId` mapping. The codec does not; it
//! works off a `&[(TensorId, String)]` handed in at export/import time.
//!
//! Skipped-entry reporting channel: the count of internal AdamW state entries
//! that lacked a name in the `names` slice during `export_state` is returned
//! via [`AdamWState::skipped_export`] (not via stderr). This keeps callers in
//! control of logging — they can surface, assert, or ignore.

use serde::{Deserialize, Serialize};

use crate::{TensorId, optim::AdamW};

/// Per-parameter AdamW moment state (first + second moment), tagged with the
/// caller-supplied `name` and the shape at export time.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AdamWParamState {
    pub name: String,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Full AdamW optimizer state: the step counter plus every known param's
/// moments. `skipped_export` records how many internal AdamW entries were
/// omitted during the most recent export (because the caller's `names` slice
/// did not cover them). Defaults to 0 for freshly deserialized states.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AdamWState {
    pub step: u64,
    pub params: Vec<AdamWParamState>,
    /// Count of internal AdamW state entries not covered by the caller's
    /// `names` slice during export. Populated by [`AdamW::export_state`];
    /// `0` for any state that was freshly deserialized (the field round-trips
    /// through serde so callers can propagate it across a save/load cycle).
    #[serde(default)]
    pub skipped_export: usize,
}

impl AdamW {
    /// Serialize moments for every `TensorId` listed in `names`.
    ///
    /// Any `TensorId` currently tracked in AdamW's internal state but missing
    /// from `names` is skipped; the count of those skips is recorded in
    /// [`AdamWState::skipped_export`]. Names whose `TensorId` has no state
    /// yet (e.g. param never received a gradient) are silently omitted — they
    /// have nothing to save.
    pub fn export_state(&self, names: &[(TensorId, String)]) -> AdamWState {
        let mut params = Vec::with_capacity(names.len());
        let mut covered = 0usize;

        for (id, name) in names {
            let Some((m, v)) = self.state_for(*id) else {
                // No moments recorded yet for this TensorId. Nothing to save.
                continue;
            };
            covered += 1;
            let shape = self.param_shape(*id).unwrap_or_else(|| vec![m.len()]);
            params.push(AdamWParamState {
                name: name.clone(),
                m: m.clone(),
                v: v.clone(),
                shape,
            });
        }

        let total_internal = self.state_len();
        let skipped_export = total_internal.saturating_sub(covered);

        AdamWState {
            step: self.step_count() as u64,
            params,
            skipped_export,
        }
    }

    /// Restore moments for entries whose `name` matches one in `names`.
    ///
    /// Returns the count of params successfully restored. Entries in `state`
    /// whose `name` has no corresponding `TensorId` in `names` (caller-side
    /// mapping is authoritative — it's the up-to-date one) are silently
    /// skipped. A shape mismatch between `state` and the current parameter
    /// shape tracked by AdamW is a hard error.
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
                // Caller's mapping doesn't cover this name. Skip silently.
                continue;
            };

            if let Some(existing_shape) = self.param_shape(id) {
                if existing_shape != param.shape {
                    anyhow::bail!(
                        "AdamW shape mismatch for '{}' (id {id}): existing {:?}, loaded {:?}",
                        param.name,
                        existing_shape,
                        param.shape,
                    );
                }
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

            self.set_state(id, param.m.clone(), param.v.clone());
            restored += 1;
        }

        self.set_step_count(state.step as i32);
        Ok(restored)
    }
}
