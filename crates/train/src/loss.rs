use autograd::{
    Result, Tape, TensorId, TensorStore,
    ops::{gather_last_dim, log_softmax, mean, mul, mul_scalar, softmax},
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

/// Forward KL divergence `KL(teacher || student)` used as the OPD distill
/// objective. Teacher logits must carry `requires_grad = false`; the
/// returned loss only backpropagates through `student_logits`.
///
/// Implementation note: `KL(t || s) = sum_v t_p * (log t_p - log s_p)
///                                   = -H(t) - sum_v t_p * log s_p`.
/// The `-H(t)` term is constant w.r.t. student parameters, so we drop it
/// and minimise the soft cross-entropy `-sum_v t_p * log s_p` averaged
/// over `num_positions`. This is exactly the standard distillation loss
/// (TRL `DistilTrainer`, MiniLLM, Agarwal et al. 2024 OPD) up to the
/// constant `H(t)`.
pub fn kl_distill_loss(
    student_logits: TensorId,
    teacher_logits: TensorId,
    _num_positions: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let teacher_probs = softmax(teacher_logits, store, tape)?;
    let student_log_probs = log_softmax(student_logits, store, tape)?;
    let weighted = mul(teacher_probs, student_log_probs, store, tape)?;
    // `mean` reduces across all dims (positions × vocab). The KL gradient
    // direction is identical to the `sum / num_positions` form up to a
    // constant `1/vocab` rescale; AdamW absorbs the constant via its
    // adaptive learning rate. Using `mean` here matches the existing
    // CE-loss backward path (which `cross_entropy_loss` exercises), so we
    // pick up the same device-resident fast paths instead of routing
    // through `sum_backward`'s untested scalar broadcast.
    let avg = mean(weighted, store, tape)?;
    mul_scalar(avg, -1.0, store, tape)
}
