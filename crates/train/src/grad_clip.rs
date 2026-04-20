//! Gradient clipping — free function (`clip_grad_norm`) kept for existing
//! call sites + `GradClip` trait surface used by the Phase 2 `Trainer`.
//!
//! See `docs/plans/train-runtime-architecture-v1.md` §4.4.

use autograd::{Result, TensorId, TensorStore};

/// Pre-clip global L2 norm across every param's gradient.
///
/// Missing grads are skipped (matches `clip_grad_norm`'s traversal).
fn compute_global_norm(params: &[TensorId], store: &TensorStore) -> f32 {
    let mut total_sq_norm = 0.0_f32;
    for &param_id in params {
        let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) else {
            continue;
        };
        let Some(grad) = store.get(grad_id) else {
            continue;
        };
        total_sq_norm += grad.data.iter().map(|value| value * value).sum::<f32>();
    }
    total_sq_norm.sqrt()
}

pub fn clip_grad_norm(params: &[TensorId], max_norm: f32, store: &mut TensorStore) {
    if max_norm <= 0.0 {
        return;
    }

    let total_norm = compute_global_norm(params, store);
    if total_norm <= max_norm || total_norm == 0.0 {
        return;
    }

    let scale = max_norm / total_norm;
    for &param_id in params {
        let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) else {
            continue;
        };
        let Some(grad) = store.get_mut(grad_id) else {
            continue;
        };
        for value in &mut grad.data {
            *value *= scale;
        }
    }
}

pub trait GradClip: Send {
    /// Clip gradients in-place. Return pre-clip global L2 norm for logging.
    fn clip(&mut self, store: &mut TensorStore, params: &[TensorId]) -> Result<f32>;
}

pub struct NoClip;

impl GradClip for NoClip {
    fn clip(&mut self, _store: &mut TensorStore, _params: &[TensorId]) -> Result<f32> {
        Ok(0.0)
    }
}

pub struct GlobalNorm {
    pub max_norm: f32,
}

impl GradClip for GlobalNorm {
    fn clip(&mut self, store: &mut TensorStore, params: &[TensorId]) -> Result<f32> {
        let pre_clip_norm = compute_global_norm(params, store);
        clip_grad_norm(params, self.max_norm, store);
        Ok(pre_clip_norm)
    }
}
