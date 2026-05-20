//! Gradient clipping — free function (`clip_grad_norm`) kept for existing
//! call sites + `GradClip` trait surface used by the Phase 2 `Trainer`.
//!
//! See `docs/plans/train-runtime-architecture-v1.md` §4.4.

use autograd::{Device, Result, TensorId, TensorStore, tensor::Dirty};

/// Pre-clip global L2 norm across every param's gradient.
///
/// Missing grads are skipped (matches `clip_grad_norm`'s traversal).
fn compute_global_norm_f64(params: &[TensorId], store: &TensorStore) -> f64 {
    let mut total_sq_norm = 0.0_f64;
    for &param_id in params {
        let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) else {
            continue;
        };
        let Some(grad) = store.get(grad_id) else {
            continue;
        };
        if store.backend().device() != Device::Cpu
            && grad.dirty != Dirty::Host
            && let Some(handle) = grad.device_handle.as_ref()
        {
            total_sq_norm += store
                .backend()
                .sum_squares(handle, &grad.shape)
                .expect("device grad norm should be computable");
        } else {
            total_sq_norm += grad
                .data
                .iter()
                .map(|&value| {
                    let value = f64::from(value);
                    value * value
                })
                .sum::<f64>();
        }
    }
    total_sq_norm.sqrt()
}

fn compute_global_norm(params: &[TensorId], store: &TensorStore) -> f32 {
    compute_global_norm_f64(params, store) as f32
}

pub fn clip_grad_norm(params: &[TensorId], max_norm: f32, store: &mut TensorStore) {
    // Non-positive / non-finite max_norm is treated as disabling gradient
    // clipping. NaN/inf used to silently propagate into the scale factor
    // and poison every gradient (codex review ef24ca6 P2).
    if !(max_norm > 0.0 && max_norm.is_finite()) {
        return;
    }

    if try_clip_grad_norm_device(params, max_norm, store) {
        return;
    }

    let total_norm = compute_global_norm_f64(params, store);
    if total_norm <= f64::from(max_norm) || total_norm == 0.0 {
        return;
    }

    let scale = f64::from(max_norm) / total_norm;
    for &param_id in params {
        let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) else {
            continue;
        };
        let device_grad = {
            let Some(grad) = store.get(grad_id) else {
                continue;
            };
            if store.backend().device() != Device::Cpu && grad.dirty != Dirty::Host {
                grad.device_handle
                    .as_ref()
                    .map(|handle| (handle.clone(), grad.shape.clone()))
            } else {
                None
            }
        };
        if let Some((handle, shape)) = device_grad {
            let scaled = store
                .backend()
                .mul_scalar(&handle, scale as f32, &shape)
                .expect("device grad scale should be computable");
            store
                .replace_device_handle(grad_id, scaled)
                .expect("scaled device grad should be installable");
            continue;
        }
        let Some(grad) = store.get_mut(grad_id) else {
            continue;
        };
        for value in &mut grad.data {
            *value *= scale as f32;
        }
    }
}

fn try_clip_grad_norm_device(params: &[TensorId], max_norm: f32, store: &mut TensorStore) -> bool {
    if store.backend().device() == Device::Cpu {
        return false;
    }

    let mut grad_ids = Vec::new();
    let mut device_grads = Vec::new();
    let mut saw_grad = false;
    for &param_id in params {
        let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) else {
            continue;
        };
        saw_grad = true;
        let Some(grad) = store.get(grad_id) else {
            continue;
        };
        if grad.dirty == Dirty::Host {
            return false;
        }
        let Some(handle) = grad.device_handle.as_ref() else {
            return false;
        };
        grad_ids.push(grad_id);
        device_grads.push((handle.clone(), grad.shape.clone()));
    }

    if !saw_grad || device_grads.is_empty() {
        return true;
    }

    let result = store
        .backend()
        .clip_grad_norm_device(&device_grads, max_norm)
        .expect("device grad clip should be computable");
    let Some(result) = result else {
        return false;
    };
    let _pre_clip_norm = result.pre_clip_norm;
    let Some(clipped_grads) = result.clipped_grads else {
        return true;
    };
    assert_eq!(
        clipped_grads.len(),
        grad_ids.len(),
        "device grad clip returned mismatched gradient handle count"
    );
    for (grad_id, handle) in grad_ids.into_iter().zip(clipped_grads) {
        store
            .replace_device_handle(grad_id, handle)
            .expect("clipped device grad should be installable");
    }
    true
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
