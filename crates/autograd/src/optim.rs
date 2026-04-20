use std::collections::HashMap;
use std::sync::Arc;

use crate::adamw_state::AdamWState;
use crate::backend::{Backend, DeviceHandle};
use crate::{Result, TensorId, tensor::TensorStore};

/// Per-parameter moment storage. Host is the long-standing path; Device is
/// the M5.3b.10 opt-in path that keeps `m` / `v` resident on the backend
/// across steps so the optimizer's update can stay in the MLX lazy graph
/// and the param never takes a re-upload round-trip.
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

/// AdamW optimizer. Two code paths live side-by-side:
///
/// - **Host path** (default, [`AdamW::new`]): moments live as `Vec<f32>`,
///   gradients read back host-side, param mutated through `get_mut`
///   (which auto-triggers `ensure_host`). This is the pre-M5.3b.10
///   behavior — correct everywhere, optimal on CPU.
/// - **Device path** ([`AdamW::new_with_device`]): moments live as
///   `DeviceHandle`s, the configured backend's `adamw_step` performs the
///   update on-device, and the param is re-installed via
///   `TensorStore::replace_device_handle` (never round-tripping through
///   host). On Metal this folds the entire EMA + bias-correction + update
///   into one MLX lazy graph with a single terminal `mlx_eval` per step,
///   eliminating the ~200-param-per-step re-upload churn that the
///   host-path's `Dirty::Host` flag caused on Qwen3.5-class models.
///
/// The on-disk state codec (`AdamWState` in `adamw_state.rs`) is unchanged
/// by the device path — device-resident moments are readback'd to host via
/// the stored backend during export, and uploaded during import.
pub struct AdamW {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    wd: f32,
    step: i32,
    state: HashMap<TensorId, ParamMoments>,
    /// Present when constructed via `new_with_device`. The backend owns the
    /// MLX device bridge (on Metal) used by `adamw_step` and for
    /// device↔host moment migration.
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
    /// Host-path constructor. Moments stay as `Vec<f32>`; gradient + param
    /// updates use the host loop. Every existing caller goes through this.
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

    /// Device-path constructor. Moments live on the backend; `step()`
    /// dispatches through `Backend::adamw_step` and installs updated
    /// params via `TensorStore::replace_device_handle`. Use when the
    /// store's backend is Metal (or any backend whose `adamw_step` is
    /// overridden to stay device-resident) — CPU/default-trait-impl
    /// backends will silently do the readback→host→upload fallback,
    /// which is strictly slower than the host path, so don't wire this
    /// up unless the backend actually overrides `adamw_step`.
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

            if self.wd > 0.0 {
                let decay = 1.0 - (self.lr * self.wd);
                for value in &mut param.data {
                    *value *= decay;
                }
            }

            for index in 0..param.data.len() {
                let g = grad[index];
                m[index] = (beta1 * m[index]) + ((1.0 - beta1) * g);
                v[index] = (beta2 * v[index]) + ((1.0 - beta2) * g * g);
                let m_hat = m[index] / bc1;
                let v_hat = v[index] / bc2;
                param.data[index] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
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

        // M5.3b.11: collect every param's (new_param, new_m, new_v) handle
        // clones during the loop and fire a single terminal
        // `backend.eval(...)` after — one eval per optimizer step regardless
        // of parameter count. The MLX chains for independent params share no
        // sub-node, so batching is safe. `DeviceHandle::Metal(MlxHandle)`
        // clones are cheap Arc ref-counts, so holding 3 × num_params handles
        // through the loop costs ~negligible memory.
        let mut pending_eval: Vec<DeviceHandle> = Vec::with_capacity(params.len() * 3);

        for &param_id in params {
            let (grad_id, param_shape, param_size) = {
                let Some(param_snapshot) = store.get(param_id) else {
                    panic!("adamw parameter {param_id} does not exist");
                };
                let Some(grad_id) = param_snapshot.grad else {
                    continue;
                };
                (grad_id, param_snapshot.shape.clone(), param_snapshot.size)
            };

            // Grad stays host-side — matmul_backward returns host Vec<f32>.
            let grad = store
                .to_host(grad_id)
                .expect("gradient tensor should be readable from the store");

            // Param: ensure it's on the device (upload if currently Host
            // or if this is the first step). Then clone the handle so the
            // backend call borrows it without holding `store` hostage.
            store
                .ensure_device(param_id)
                .expect("ensure_device for adamw param");
            let param_handle = store
                .tensors
                .get(param_id)
                .and_then(|slot| slot.as_ref())
                .and_then(|t| t.device_handle.clone())
                .expect("param device_handle after ensure_device");

            // Initialize moments on first touch: upload zeros through
            // the backend. Subsequent steps reuse the device handles.
            let entry = self.state.entry(param_id).or_insert_with(|| ParamMoments {
                m: MomentStorage::Device(
                    backend
                        .upload(&vec![0.0_f32; param_size], &param_shape)
                        .expect("upload zero m on first adamw step"),
                ),
                v: MomentStorage::Device(
                    backend
                        .upload(&vec![0.0_f32; param_size], &param_shape)
                        .expect("upload zero v on first adamw step"),
                ),
                shape: param_shape.clone(),
            });

            // If a prior host path left host moments behind, migrate them
            // up to the device now so the update formula sees device state.
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

            let (new_param, new_m, new_v) = backend
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
                .expect("backend adamw_step");

            // Record cheap Arc-clones for the terminal batched eval below.
            pending_eval.push(new_param.clone());
            pending_eval.push(new_m.clone());
            pending_eval.push(new_v.clone());

            // Install the new param handle WITHOUT going through get_mut
            // (which would ensure_host → mark Dirty::Host → force re-upload).
            store
                .replace_device_handle(param_id, new_param)
                .expect("replace_device_handle for adamw param");

            entry.m = MomentStorage::Device(new_m);
            entry.v = MomentStorage::Device(new_v);
        }

        // M5.3b.11: one terminal eval for the whole optimizer step. The
        // Metal backend's `adamw_step` returned every triple unevaluated,
        // so without this line the MLX graph would accumulate until the
        // next forward pass's `ensure_host` forces a catch-up eval —
        // correctness-equivalent but gives up the batching benefit for
        // the eval counter. The CPU default `Backend::eval` is a no-op.
        if !pending_eval.is_empty() {
            let refs: Vec<&DeviceHandle> = pending_eval.iter().collect();
            backend.eval(&refs).expect("batched adamw terminal eval");
        }
    }

    pub fn zero_grad(&mut self, params: &[TensorId], store: &mut TensorStore) {
        for &param_id in params {
            let grad_id = store.get(param_id).and_then(|tensor| tensor.grad);
            if let Some(grad_id) = grad_id
                && let Some(grad) = store.get_mut(grad_id)
            {
                grad.data.fill(0.0);
            }
        }
    }

    // ------------------------------------------------------------------
    // Accessors used by the opaque state codec in `adamw_state.rs`.
    // They deliberately avoid exposing the private `ParamMoments` struct.
    // Device-resident moments readback through the stored backend.
    // ------------------------------------------------------------------

    /// Materialize `(m, v)` as owned host vectors for the caller, regardless
    /// of whether the moments are currently host- or device-resident.
    /// Device readback uses the optimizer's stored backend.
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

    /// Install imported moments into the optimizer. On the device path the
    /// moments upload through the stored backend so the next `step()` stays
    /// on-device; on the host path they land as `Vec<f32>`.
    pub(crate) fn set_state(&mut self, id: TensorId, m: Vec<f32>, v: Vec<f32>, shape: Vec<usize>) {
        debug_assert_eq!(m.len(), v.len(), "m and v must share length");
        let (m_store, v_store) = if let Some(backend) = self.backend.as_ref() {
            let m_handle = backend
                .upload(&m, &shape)
                .expect("upload m on import to device");
            let v_handle = backend
                .upload(&v, &shape)
                .expect("upload v on import to device");
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
    }
}

// Equivalent of the previous derive(Clone) — kept for API compat but skips
// device handles (they'd need a backend clone + FFI). The host path round-
// trips perfectly; device-path clones drop to zero-initialized moments so
// callers that relied on the derive were never touching device state anyway.
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

/// Trait-level view of an optimizer. Today AdamW is the only implementor; the
/// trait exists so the in-progress training runtime (see
/// `docs/plans/train-runtime-architecture-v1.md` §4.1) can dispatch
/// polymorphically over future Lion/Muon/SGD impls without forking every
/// binary. The state-codec surface (`state_schema` + `export_state` +
/// `import_state`) is AdamW-shaped on purpose — the [`AdamWState`] value is
/// the on-disk format, and alternative optimizers will extend the doc schema
/// (e.g. `"lion-v1"`) when they arrive.
///
/// Note on argument order: the trait takes `store` before `params`, which
/// matches the plan's signature and the conventional "context-first" Rust
/// style. The concrete `AdamW::step` kept the original `(params, store)`
/// order for source compatibility with the 4 training binaries; the trait
/// impl below swaps the two. A trait dispatch always returns `Ok(())` — the
/// concrete method panics on internal invariant violations (missing
/// parameter, unreadable grad), and those panics are not reachable from the
/// well-formed call sites we ship today. If a future optimizer wants real
/// `Err` paths, it can wire them in without the concrete `AdamW` signature
/// changing.
pub trait Optimizer: Send {
    fn step(&mut self, store: &mut TensorStore, params: &[TensorId]) -> Result<()>;
    fn zero_grad(&mut self, store: &mut TensorStore, params: &[TensorId]);
    fn set_lr(&mut self, lr: f32);
    fn lr(&self) -> f32;

    /// Schema tag for the on-disk state doc. e.g. `"adamw-v1"`. Used by the
    /// checkpoint codec to validate on import.
    fn state_schema(&self) -> &'static str;

    /// Export moments + scalars keyed by caller-supplied name. Today the doc
    /// type is AdamW-specific — future optimizers that need a different
    /// layout will bump the schema tag and/or introduce a new doc variant.
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
        // Concrete signature is (&params, &mut store); adapt and wrap. The
        // concrete method panics on invariant violations, which the trait
        // contract lets propagate — callers of the trait see the same
        // behavior as callers of the concrete impl.
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
