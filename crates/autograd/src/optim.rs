use std::collections::HashMap;

use crate::adamw_state::AdamWState;
use crate::{Result, TensorId, tensor::TensorStore};

#[derive(Debug, Clone)]
struct ParamMoments {
    m: Vec<f32>,
    v: Vec<f32>,
    shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct AdamW {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    wd: f32,
    step: i32,
    state: HashMap<TensorId, ParamMoments>,
}

impl AdamW {
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, wd: f32) -> Self {
        Self {
            lr,
            betas,
            eps,
            wd,
            step: 0,
            state: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &[TensorId], store: &mut TensorStore) {
        self.step += 1;
        let (beta1, beta2) = self.betas;
        let bc1 = 1.0 - beta1.powi(self.step);
        let bc2 = 1.0 - beta2.powi(self.step);

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
                    param_snapshot.data.len(),
                    param_snapshot.shape.clone(),
                )
            };

            let grad = store
                .to_host(grad_id)
                .expect("gradient tensor should be readable from the store");
            let moments = self.state.entry(param_id).or_insert_with(|| ParamMoments {
                m: vec![0.0; param_len],
                v: vec![0.0; param_len],
                shape: param_shape,
            });
            let ParamMoments { m, v, .. } = moments;
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
    // ------------------------------------------------------------------

    pub(crate) fn state_for(&self, id: TensorId) -> Option<(&Vec<f32>, &Vec<f32>)> {
        self.state.get(&id).map(|p| (&p.m, &p.v))
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

    pub(crate) fn set_state(&mut self, id: TensorId, m: Vec<f32>, v: Vec<f32>, shape: Vec<usize>) {
        debug_assert_eq!(m.len(), v.len(), "m and v must share length");
        self.state.insert(id, ParamMoments { m, v, shape });
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
