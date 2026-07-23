//! Gradient clipping — free functions (`clip_grad_norm` /
//! `compute_global_norm_f64`) used by the OPD training loops.
//!
//! See `docs/plans/train-runtime-architecture-v1.md` §4.4.

use autograd::{Device, TensorId, TensorStore, tensor::Dirty};

/// Pre-clip global L2 norm across every param's gradient.
///
/// Missing grads are skipped (matches `clip_grad_norm`'s traversal).
pub fn compute_global_norm_f64(params: &[TensorId], store: &TensorStore) -> f64 {
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

pub fn clip_grad_norm(params: &[TensorId], max_norm: f32, store: &mut TensorStore) {
    // Non-positive / non-finite max_norm is treated as disabling gradient
    // clipping. NaN/inf used to silently propagate into the scale factor
    // and poison every gradient (codex review ef24ca6 P2).
    if !(max_norm > 0.0 && max_norm.is_finite()) {
        return;
    }

    // Diagnostic (opt-in, env-gated to avoid per-step stderr spam): surface the
    // pre-clip global L2 norm so we can falsify whether the batchmean KL fix
    // (grads now ~vocab x larger) trips `--grad-clip` every step — which would
    // re-introduce an LR cut via clipping instead of AdamW eps. The prod clip
    // path otherwise discards this norm, leaving the question unobservable.
    if std::env::var("ARLE_OPD_LOG_GRAD_NORM").is_ok() {
        let pre_clip_norm = compute_global_norm_f64(params, store);
        eprintln!(
            "[grad-clip] pre_clip_norm={pre_clip_norm:.6e} max_norm={max_norm:.3e} clipped={}",
            pre_clip_norm > f64::from(max_norm)
        );
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
