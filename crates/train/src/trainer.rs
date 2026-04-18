use autograd::{
    Result, Tape, TensorId, TensorStore,
    ops::{gather_last_dim, log_softmax, mean, mul_scalar},
};

pub fn cross_entropy_loss(
    logits_id: TensorId,
    targets: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let log_probs = log_softmax(logits_id, store, tape)?;
    let target_log_probs = gather_last_dim(log_probs, targets, store, tape)?;
    let mean_log_prob = mean(target_log_probs, store, tape)?;
    mul_scalar(mean_log_prob, -1.0, store, tape)
}

pub fn clip_grad_norm(params: &[TensorId], max_norm: f32, store: &mut TensorStore) {
    if max_norm <= 0.0 {
        return;
    }

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

    let total_norm = total_sq_norm.sqrt();
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
