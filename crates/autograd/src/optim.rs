use std::collections::HashMap;
use std::sync::Arc;

use crate::adamw_state::AdamWState;
use crate::backend::{Backend, DeviceHandle};
use crate::tensor::Dirty;
use crate::{Result, TensorId, tensor::TensorStore};

/// Per-parameter moment storage. The device path keeps `m`/`v` resident on
/// the backend across steps so the update stays in the MLX lazy graph and
/// the param never takes a re-upload round-trip.
#[derive(Debug)]
enum MomentStorage {
    Host(Vec<f32>),
    Device(DeviceHandle),
}

#[derive(Debug)]
struct ParamMoments {
    m: MomentStorage,
    v: MomentStorage,
    shape: Vec<usize>,
}

/// AdamW optimizer with two moment-storage paths: host `Vec<f32>` (default,
/// [`AdamW::new`]) and device-resident handles ([`AdamW::new_with_device`]).
/// The device path folds the update into one MLX lazy graph with a single
/// terminal eval per step, eliminating the per-param re-upload churn the
/// host path's `Dirty::Host` flag causes on Metal. The on-disk codec
/// (`AdamWState`) is unchanged — device moments readback to host on export.
pub struct AdamW {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    wd: f32,
    step: i32,
    state: HashMap<TensorId, ParamMoments>,
    /// Present when constructed via `new_with_device`; owns the device
    /// bridge for `adamw_step` and device↔host moment migration.
    backend: Option<Arc<dyn Backend + Send + Sync>>,
}

impl std::fmt::Debug for AdamW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW")
            .field("lr", &self.lr)
            .field("betas", &self.betas)
            .field("eps", &self.eps)
            .field("wd", &self.wd)
            .field("step", &self.step)
            .field("params_tracked", &self.state.len())
            .field("device_backed", &self.backend.is_some())
            .finish()
    }
}

impl AdamW {
    /// Host-path constructor; moments stay as `Vec<f32>`.
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, wd: f32) -> Self {
        Self {
            lr,
            betas,
            eps,
            wd,
            step: 0,
            state: HashMap::new(),
            backend: None,
        }
    }

    /// Device-path constructor. Use only when the store's backend overrides
    /// `adamw_step` to stay device-resident — CPU/default-trait backends
    /// silently fall back to readback→host→upload, strictly slower than the
    /// host path.
    pub fn new_with_device(
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        wd: f32,
        backend: Arc<dyn Backend + Send + Sync>,
    ) -> Self {
        Self {
            lr,
            betas,
            eps,
            wd,
            step: 0,
            state: HashMap::new(),
            backend: Some(backend),
        }
    }

    pub fn step(&mut self, params: &[TensorId], store: &mut TensorStore) {
        self.step += 1;
        let (beta1, beta2) = self.betas;
        let bc1 = 1.0 - beta1.powi(self.step);
        let bc2 = 1.0 - beta2.powi(self.step);

        if self.backend.is_some() {
            self.step_device(params, store, beta1, beta2, bc1, bc2);
        } else {
            self.step_host(params, store, beta1, beta2, bc1, bc2);
        }
    }

    fn step_host(
        &mut self,
        params: &[TensorId],
        store: &mut TensorStore,
        beta1: f32,
        beta2: f32,
        bc1: f32,
        bc2: f32,
    ) {
        for &param_id in params {
            let (grad_id, param_len, param_shape) = {
                let Some(param_snapshot) = store.get(param_id) else {
                    panic!("adamw parameter {param_id} does not exist");
                };
                let Some(grad_id) = param_snapshot.grad else {
                    continue;
                };
                (
                    grad_id,
                    param_snapshot.data.len().max(param_snapshot.size),
                    param_snapshot.shape.clone(),
                )
            };

            let grad = store
                .to_host(grad_id)
                .expect("gradient tensor should be readable from the store");
            let moments = self.state.entry(param_id).or_insert_with(|| ParamMoments {
                m: MomentStorage::Host(vec![0.0; param_len]),
                v: MomentStorage::Host(vec![0.0; param_len]),
                shape: param_shape,
            });
            let (m, v) = match (&mut moments.m, &mut moments.v) {
                (MomentStorage::Host(m), MomentStorage::Host(v)) => (m, v),
                _ => panic!(
                    "host AdamW path encountered device-resident moments for param {param_id}; \
                     use `new_with_device` on the optimizer or drop the device moments first"
                ),
            };
            let param = store
                .get_mut(param_id)
                .expect("parameter tensor should still exist when stepping");

            assert_eq!(
                grad.len(),
                param.data.len(),
                "AdamW grad length must match parameter length for param {param_id}"
            );
            if self.wd > 0.0 {
                let decay = 1.0 - (self.lr * self.wd);
                for value in &mut param.data {
                    *value *= decay;
                }
            }

            let step_size = self.lr / bc1;
            let inv_bc2 = 1.0 / bc2;
            let one_minus_beta1 = 1.0 - beta1;
            let one_minus_beta2 = 1.0 - beta2;
            for ((param_value, &g), (m_value, v_value)) in param
                .data
                .iter_mut()
                .zip(&grad)
                .zip(m.iter_mut().zip(v.iter_mut()))
            {
                let m_next = (beta1 * *m_value) + (one_minus_beta1 * g);
                let v_next = (beta2 * *v_value) + (one_minus_beta2 * g * g);
                *m_value = m_next;
                *v_value = v_next;
                let denom = (v_next * inv_bc2).sqrt() + self.eps;
                *param_value -= step_size * m_next / denom;
            }
        }
    }

    fn step_device(
        &mut self,
        params: &[TensorId],
        store: &mut TensorStore,
        beta1: f32,
        beta2: f32,
        bc1: f32,
        bc2: f32,
    ) {
        let backend = self
            .backend
            .as_ref()
            .expect("step_device called without a backend")
            .clone();

        // Collect every param's handle clones during the loop and fire a
        // single terminal `backend.eval(...)` after — one eval per step
        // regardless of param count. Per-param chains share no sub-node, so
        // batching is safe; Metal handle clones are cheap Arc ref-counts.
        let mut pending_eval: Vec<DeviceHandle> = Vec::with_capacity(params.len() * 3);

        for &param_id in params {
            let (grad_id, param_shape) = {
                let Some(param_snapshot) = store.get(param_id) else {
                    panic!("adamw parameter {param_id} does not exist");
                };
                let Some(grad_id) = param_snapshot.grad else {
                    continue;
                };
                (grad_id, param_snapshot.shape.clone())
            };

            // Peek at grad residency before any `to_host`: a device-resident
            // grad routes through `adamw_step_device`, skipping the DtoH that
            // measured a +1.8% wash (3423 extra DtoH calls, 41.5 GB/step).
            // The host fallback stays for grads still produced on host.
            let grad_device_handle = {
                let grad_tensor = store
                    .tensor(grad_id)
                    .expect("gradient tensor should exist in the store");
                if grad_tensor.dirty != Dirty::Host {
                    grad_tensor.device_handle.clone()
                } else {
                    None
                }
            };

            // Clone the handle so the backend call borrows it without
            // holding `store` hostage.
            store
                .ensure_device(param_id)
                .expect("ensure_device for adamw param");
            let param_handle = store
                .tensors
                .get(param_id)
                .and_then(|slot| slot.as_ref())
                .and_then(|t| t.device_handle.clone())
                .expect("param device_handle after ensure_device");

            let entry = self.state.entry(param_id).or_insert_with(|| ParamMoments {
                m: MomentStorage::Device(
                    backend
                        .zeros(&param_shape)
                        .expect("allocate zero m on first adamw step"),
                ),
                v: MomentStorage::Device(
                    backend
                        .zeros(&param_shape)
                        .expect("allocate zero v on first adamw step"),
                ),
                shape: param_shape.clone(),
            });

            // A prior host path may have left host moments; migrate them.
            if let MomentStorage::Host(host_m) = &entry.m {
                let handle = backend
                    .upload(host_m, &entry.shape)
                    .expect("upload host m to device");
                entry.m = MomentStorage::Device(handle);
            }
            if let MomentStorage::Host(host_v) = &entry.v {
                let handle = backend
                    .upload(host_v, &entry.shape)
                    .expect("upload host v to device");
                entry.v = MomentStorage::Device(handle);
            }

            let (m_handle, v_handle) = match (&entry.m, &entry.v) {
                (MomentStorage::Device(m), MomentStorage::Device(v)) => (m.clone(), v.clone()),
                _ => unreachable!("moments migrated to Device above"),
            };

            let (new_param, new_m, new_v) = if let Some(grad_h) = grad_device_handle {
                backend
                    .adamw_step_device(
                        &param_handle,
                        &m_handle,
                        &v_handle,
                        &grad_h,
                        &entry.shape,
                        self.lr,
                        beta1,
                        beta2,
                        self.eps,
                        self.wd,
                        bc1,
                        bc2,
                    )
                    .expect("backend adamw_step_device")
            } else {
                // Host fallback: grad still authoritative on host.
                let grad = store
                    .to_host(grad_id)
                    .expect("gradient tensor should be readable from the store");
                backend
                    .adamw_step(
                        &param_handle,
                        &m_handle,
                        &v_handle,
                        &grad,
                        &entry.shape,
                        self.lr,
                        beta1,
                        beta2,
                        self.eps,
                        self.wd,
                        bc1,
                        bc2,
                    )
                    .expect("backend adamw_step")
            };

            pending_eval.push(new_param.clone());
            pending_eval.push(new_m.clone());
            pending_eval.push(new_v.clone());

            // Install the new param handle WITHOUT going through `get_mut`
            // (which would ensure_host → mark Dirty::Host → force re-upload).
            store
                .replace_device_handle(param_id, new_param)
                .expect("replace_device_handle for adamw param");

            entry.m = MomentStorage::Device(new_m);
            entry.v = MomentStorage::Device(new_v);
        }

        // One terminal eval for the whole step: `adamw_step` returned every
        // triple unevaluated, so without this the graph accumulates until
        // the next forward's `ensure_host` forces a catch-up eval. The CPU
        // default `Backend::eval` is a no-op.
        if !pending_eval.is_empty() {
            let refs: Vec<&DeviceHandle> = pending_eval.iter().collect();
            backend.eval(&refs).expect("batched adamw terminal eval");
        }
    }

    pub fn zero_grad(&mut self, params: &[TensorId], store: &mut TensorStore) {
        if self.backend.is_some() {
            for &param_id in params {
                let grad_id = store.get(param_id).and_then(|tensor| tensor.grad);
                if let Some(grad_id) = grad_id {
                    store
                        .set_grad(param_id, None)
                        .expect("clear device-backed grad id");
                    store.free(grad_id).expect("free device-backed grad tensor");
                }
            }
            return;
        }

        for &param_id in params {
            let grad_id = store.get(param_id).and_then(|tensor| tensor.grad);
            if let Some(grad_id) = grad_id {
                store
                    .set_grad(param_id, None)
                    .expect("clear host-backed grad id");
                store.free(grad_id).expect("free host-backed grad tensor");
            }
        }
    }

    /// Drop the optimizer moments for the given parameter ids, returning how
    /// many entries were actually removed.
    ///
    /// SOPD rollback ([`EmaSelfTeacher::restore`](../../train/src/ema_self_teacher.rs))
    /// needs this: [`AdamWState::import_state`](crate::adamw_state::AdamWState::import_state)
    /// only re-installs the entries serialized in the snapshot, so a step that
    /// gets rejected *after* the snapshot — e.g. the first gated step, taken
    /// before any moments existed — would otherwise leave its freshly-created
    /// m/v behind. Clearing the adapter ids first makes restore exact.
    pub fn clear_param_state(&mut self, ids: &[TensorId]) -> usize {
        let mut removed = 0;
        for id in ids {
            if self.state.remove(id).is_some() {
                removed += 1;
            }
        }
        removed
    }

    // Accessors for the opaque state codec in `adamw_state.rs`; they
    // deliberately avoid exposing the private `ParamMoments` struct.

    /// Materialize `(m, v)` as owned host vectors, reading device-resident
    /// moments back through the stored backend.
    pub(crate) fn moments_host(&self, id: TensorId) -> Option<(Vec<f32>, Vec<f32>)> {
        let moments = self.state.get(&id)?;
        let m = match &moments.m {
            MomentStorage::Host(m) => m.clone(),
            MomentStorage::Device(handle) => self
                .backend
                .as_ref()
                .expect("device moments require a backend")
                .readback(handle)
                .expect("readback device m for export"),
        };
        let v = match &moments.v {
            MomentStorage::Host(v) => v.clone(),
            MomentStorage::Device(handle) => self
                .backend
                .as_ref()
                .expect("device moments require a backend")
                .readback(handle)
                .expect("readback device v for export"),
        };
        Some((m, v))
    }

    pub(crate) fn state_len(&self) -> usize {
        self.state.len()
    }

    pub(crate) fn param_shape(&self, id: TensorId) -> Option<Vec<usize>> {
        self.state.get(&id).map(|p| p.shape.clone())
    }

    pub(crate) fn step_count(&self) -> i32 {
        self.step
    }

    pub(crate) fn set_step_count(&mut self, step: i32) {
        self.step = step;
    }

    pub(crate) fn set_state(
        &mut self,
        id: TensorId,
        m: Vec<f32>,
        v: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<()> {
        debug_assert_eq!(m.len(), v.len(), "m and v must share length");
        let (m_store, v_store) = if let Some(backend) = self.backend.as_ref() {
            let m_handle = backend.upload(&m, &shape)?;
            let v_handle = backend.upload(&v, &shape)?;
            (
                MomentStorage::Device(m_handle),
                MomentStorage::Device(v_handle),
            )
        } else {
            (MomentStorage::Host(m), MomentStorage::Host(v))
        };
        self.state.insert(
            id,
            ParamMoments {
                m: m_store,
                v: v_store,
                shape,
            },
        );
        Ok(())
    }
}

// Hand-rolled Clone (API compat with the former derive): host moments clone;
// device moments drop to zero-initialized host vectors — callers that relied
// on the derive never touched device state.
impl Clone for AdamW {
    fn clone(&self) -> Self {
        let cloned_state: HashMap<TensorId, ParamMoments> = self
            .state
            .iter()
            .map(|(id, moments)| {
                let m = match &moments.m {
                    MomentStorage::Host(v) => MomentStorage::Host(v.clone()),
                    MomentStorage::Device(_) => {
                        MomentStorage::Host(vec![
                            0.0;
                            moments.shape.iter().product::<usize>().max(1)
                        ])
                    }
                };
                let v = match &moments.v {
                    MomentStorage::Host(v) => MomentStorage::Host(v.clone()),
                    MomentStorage::Device(_) => {
                        MomentStorage::Host(vec![
                            0.0;
                            moments.shape.iter().product::<usize>().max(1)
                        ])
                    }
                };
                (
                    *id,
                    ParamMoments {
                        m,
                        v,
                        shape: moments.shape.clone(),
                    },
                )
            })
            .collect();
        Self {
            lr: self.lr,
            betas: self.betas,
            eps: self.eps,
            wd: self.wd,
            step: self.step,
            state: cloned_state,
            backend: self.backend.clone(),
        }
    }
}

/// Trait-level optimizer view; AdamW is the only implementor. The state
/// codec surface is AdamW-shaped — [`AdamWState`] is the on-disk format,
/// and future optimizers bump the schema tag when they arrive.
///
/// Argument order: the trait takes `store` before `params` (context-first);
/// the concrete `AdamW::step` keeps `(params, store)` for source compat
/// with the training binaries, and the impl swaps them. Trait `step`
/// always returns `Ok(())` — the concrete method panics on invariant
/// violations unreachable from shipped call sites.
pub trait Optimizer: Send {
    fn step(&mut self, store: &mut TensorStore, params: &[TensorId]) -> Result<()>;
    fn zero_grad(&mut self, store: &mut TensorStore, params: &[TensorId]);
    fn set_lr(&mut self, lr: f32);
    fn lr(&self) -> f32;

    /// Schema tag for the on-disk state doc. e.g. `"adamw-v1"`. Used by the
    /// checkpoint codec to validate on import.
    fn state_schema(&self) -> &'static str;

    /// Export moments + scalars keyed by caller-supplied name.
    fn export_state(&self, names: &[(TensorId, String)]) -> AdamWState;

    /// Restore moments; shape mismatch is a hard error; unknown names are
    /// silently skipped. Returns the count of entries actually restored.
    fn import_state(
        &mut self,
        doc: &AdamWState,
        names: &[(TensorId, String)],
    ) -> anyhow::Result<usize>;
}

impl Optimizer for AdamW {
    fn step(&mut self, store: &mut TensorStore, params: &[TensorId]) -> Result<()> {
        AdamW::step(self, params, store);
        Ok(())
    }

    fn zero_grad(&mut self, store: &mut TensorStore, params: &[TensorId]) {
        AdamW::zero_grad(self, params, store);
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn state_schema(&self) -> &'static str {
        "adamw-v1"
    }

    fn export_state(&self, names: &[(TensorId, String)]) -> AdamWState {
        AdamW::export_state(self, names)
    }

    fn import_state(
        &mut self,
        doc: &AdamWState,
        names: &[(TensorId, String)],
    ) -> anyhow::Result<usize> {
        AdamW::import_state(self, doc, names)
    }
}
