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
    // Non-positive / non-finite max_norm is treated as disabling gradient
    // clipping. NaN/inf used to silently propagate into the scale factor
    // and poison every gradient (codex review ef24ca6 P2).
    if !(max_norm > 0.0 && max_norm.is_finite()) {
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
    fn clip(&mut self, store: &mut TensorStore, params: &[TensorId]) -> Result<f32> {
        // Report the true pre-clip global L2 norm so unclipped baselines
        // still see explode/vanish gradients in logs.
        Ok(compute_global_norm(params, store))
    }
}

pub struct GlobalNorm {
    pub max_norm: f32,
}

impl GlobalNorm {
    /// Construct a `GlobalNorm` clipper. Panics if `max_norm <= 0.0` to fail
    /// fast rather than silently becoming a no-op (see `clip_grad_norm`).
    pub fn new(max_norm: f32) -> Self {
        assert!(
            max_norm > 0.0 && max_norm.is_finite(),
            "GlobalNorm::new: max_norm must be > 0.0 and finite, got {max_norm}"
        );
        Self { max_norm }
    }
}

impl GradClip for GlobalNorm {
    fn clip(&mut self, store: &mut TensorStore, params: &[TensorId]) -> Result<f32> {
        let pre_clip_norm = compute_global_norm(params, store);
        clip_grad_norm(params, self.max_norm, store);
        Ok(pre_clip_norm)
    }
}
