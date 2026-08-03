//! Gradient clipping — free functions (`clip_grad_norm` /
//! `compute_global_norm_f64`) used by the OPD training loops.

use autograd::{AutogradError, CommAxis, Device, Optimizer, TensorId, TensorStore, tensor::Dirty};

#[derive(Debug, thiserror::Error)]
pub enum FiniteStepError {
    #[error("training loss became non-finite ({0})")]
    NonFiniteLoss(f32),
    #[error("global gradient norm became non-finite ({0})")]
    NonFiniteGradNorm(f64),
    #[error(transparent)]
    Optimizer(#[from] AutogradError),
}

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

pub fn ensure_finite_loss<O: Optimizer>(
    loss: f32,
    params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
) -> Result<(), FiniteStepError> {
    if loss.is_finite() {
        return Ok(());
    }
    optimizer.zero_grad(store, params);
    Err(FiniteStepError::NonFiniteLoss(loss))
}

/// Commit one optimizer update only when both the scalar loss and the complete
/// accumulated gradient set are finite. Any rejected update clears pending grads.
pub fn finite_optimizer_step<O: Optimizer>(
    loss: f32,
    params: &[TensorId],
    max_norm: f32,
    optimizer: &mut O,
    store: &mut TensorStore,
) -> Result<f64, FiniteStepError> {
    ensure_finite_loss(loss, params, optimizer, store)?;

    let global_norm = compute_global_norm_f64(params, store);
    if !global_norm.is_finite() {
        optimizer.zero_grad(store, params);
        return Err(FiniteStepError::NonFiniteGradNorm(global_norm));
    }

    clip_grad_norm(params, max_norm, store);
    if let Err(error) = optimizer.step(store, params) {
        optimizer.zero_grad(store, params);
        return Err(FiniteStepError::Optimizer(error));
    }
    optimizer.zero_grad(store, params);
    Ok(global_norm)
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

/// Sum a per-rank integer count across the collective group (DP global-target
/// count for the mean-CE `inv_n`). Uploads the local count as a 1-elem tensor,
/// all-reduce-sums it, reads it back. world==1 is identity (the backend's
/// `all_reduce_sum_device` no-ops without an NCCL comm), so single-card returns
/// `local` unchanged — byte-identical.
pub fn dp_group_sum_count(local: usize, store: &mut TensorStore) -> Result<usize, AutogradError> {
    dp_group_sum_scalar(local as f32, store).map(|v| v.round() as usize)
}

/// Sum a host scalar over the world comm. Callers on a CP subgroup contribute
/// from cp rank 0 only so replica-shared values aren't multiplied by cp_size.
pub fn dp_group_sum_scalar(local: f32, store: &mut TensorStore) -> Result<f32, AutogradError> {
    let handle = store.backend().upload(&[local], &[1])?;
    let reduced = store
        .backend()
        .all_reduce_sum_device(&handle, &[1], CommAxis::World)?;
    let summed = store.backend().readback(&reduced)?;
    Ok(summed.first().copied().unwrap_or(local))
}

/// All-reduce-sum every trainable param's gradient across the context-parallel
/// group. CP replicates weights and shards the sequence, so each rank holds its
/// shard's contribution to the (replicated) weight grad; summing over the group
/// gives the exact single-card grad. Must run AFTER backward and BEFORE the
/// optimizer step, and only reduce `trainable_params` (activation grads stay
/// local).
///
/// A rank whose sequence shard owns zero masked targets produces NO grad for a
/// param — but it MUST still enter every collective or NCCL deadlocks the group.
/// So a missing grad is materialized as a zero grad of the param's shape and
/// reduced like any other (contributing 0 to the sum).
///
/// The per-position element counts are checked identical across ranks first
/// (`assert_cp_param_layout_agrees`): NCCL has no size rendezvous, so an order
/// mismatch would otherwise spin the GPU forever with no error.
pub fn all_reduce_cp_grads(
    params: &[TensorId],
    store: &mut TensorStore,
) -> Result<(), AutogradError> {
    assert_cp_param_layout_agrees(params, store)?;
    for &param_id in params {
        let grad_id = match store.get(param_id).and_then(|tensor| tensor.grad) {
            Some(grad_id) => grad_id,
            None => {
                // Zero grad of the param's shape so this rank still participates.
                let shape = store
                    .get(param_id)
                    .ok_or(AutogradError::InvalidTensorId(param_id))?
                    .shape
                    .clone();
                let zero = store.backend().zeros(&shape)?;
                let zero_id = store.alloc_device_tensor(shape, zero)?;
                store.accumulate_grad(param_id, zero_id)?;
                store
                    .get(param_id)
                    .and_then(|tensor| tensor.grad)
                    .expect("grad set by accumulate_grad")
            }
        };
        store.ensure_device(grad_id)?;
        let (handle, shape) = {
            let grad = store
                .get(grad_id)
                .ok_or(AutogradError::InvalidTensorId(grad_id))?;
            let handle = grad
                .device_handle
                .as_ref()
                .ok_or(AutogradError::TapeInvariant(
                    "all_reduce_cp_grads: grad missing device handle after ensure_device",
                ))?
                .clone();
            (handle, grad.shape.clone())
        };
        let reduced = store
            .backend()
            .all_reduce_sum_device(&handle, &shape, CommAxis::World)?;
        store.replace_device_handle(grad_id, reduced)?;
    }
    Ok(())
}

/// Fail fast if the ranks disagree on the per-position param element counts.
///
/// `all_reduce_cp_grads` issues one collective per param by index; NCCL matches
/// nothing about shape, so a divergent order (e.g. a HashMap-seeded param list)
/// silently wedges the GPU. This gathers each rank's fixed-length count vector
/// (same length on every rank — one entry per param — so the gather itself can
/// never mismatch) and rejects the step the moment two ranks differ.
fn assert_cp_param_layout_agrees(
    params: &[TensorId],
    store: &mut TensorStore,
) -> Result<(), AutogradError> {
    let n = params.len();
    let local: Vec<f32> = params
        .iter()
        .map(|&id| store.get(id).map_or(0.0, |t| t.size as f32))
        .collect();
    let handle = store.backend().upload(&local, &[n])?;
    let gathered = store
        .backend()
        .all_gather_seq_device(&handle, &[n], CommAxis::World)?;
    let gathered = store.backend().readback(&gathered)?;
    let world = gathered.len() / n; // world==1 → identity, loop is a no-op
    for rank in 1..world {
        for i in 0..n {
            if gathered[rank * n + i] != gathered[i] {
                return Err(AutogradError::TapeInvariant(
                    "CP param layout diverges across ranks: gradient all-reduce \
                     would pair mismatched shapes into one NCCL collective. The \
                     per-rank trainable-param order must be identical (no HashMap \
                     iteration in its construction).",
                ));
            }
        }
    }
    Ok(())
}
