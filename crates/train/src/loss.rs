use autograd::{
    AutogradError, Result, Tape, TensorId, TensorStore,
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
/// objective. Student logits must carry `requires_grad = true`; teacher
/// logits must carry `requires_grad = false`; the returned loss only
/// backpropagates through `student_logits`.
///
/// Implementation note: `KL(t || s) = sum_v t_p * (log t_p - log s_p)
///                                   = -H(t) - sum_v t_p * log s_p`.
/// The `-H(t)` term is constant w.r.t. student parameters, so we drop it
/// and minimise the soft cross-entropy `-sum_v t_p * log s_p`.
///
/// `num_positions` is validated against `logits.numel() / vocab` so a stale
/// rollout length does not silently train against the wrong tensor shape. The
/// current numeric path uses the existing mean-over-all-logits normalization
/// (`positions * vocab`), which is a constant `1 / vocab` rescale of
/// `sum_v / num_positions`; changing that scale is an OPD semantics decision.
pub fn kl_distill_loss(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    validate_kl_distill_inputs(student_logits, teacher_logits, num_positions, store)?;
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

fn validate_kl_distill_inputs(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    store: &TensorStore,
) -> Result<()> {
    let student = store
        .get(student_logits)
        .ok_or(AutogradError::InvalidTensorId(student_logits))?;
    let teacher = store
        .get(teacher_logits)
        .ok_or(AutogradError::InvalidTensorId(teacher_logits))?;
    if !student.requires_grad {
        return Err(AutogradError::TapeInvariant(
            "kl_distill_loss: student_logits must have requires_grad=true. \
             Hint: pass logits from the trainable OPD student forward; a \
             frozen student loss would not produce gradients.",
        ));
    }
    if teacher.requires_grad {
        return Err(AutogradError::TapeInvariant(
            "kl_distill_loss: teacher_logits must have requires_grad=false. \
             Hint: pass logits from a frozen teacher/eval forward; OPD must \
             not backpropagate into the teacher.",
        ));
    }
    if student.shape != teacher.shape {
        return Err(AutogradError::TapeInvariant(
            "kl_distill_loss: student_logits and teacher_logits must have identical shapes. \
             Hint: pass logits from the same OPD rollout scored by compatible teacher and \
             student Qwen3.5-family models with matching vocab_size.",
        ));
    }

    let vocab = student
        .shape
        .last()
        .copied()
        .ok_or(AutogradError::TapeInvariant(
            "kl_distill_loss: logits must have at least one dimension with vocab on the last axis. \
         Hint: pass Qwen35Model forward logits shaped [..., vocab_size].",
        ))?;
    if vocab == 0 {
        return Err(AutogradError::TapeInvariant(
            "kl_distill_loss: logits last dimension (vocab) must be non-zero. \
             Hint: verify teacher/student config.json vocab_size before running OPD.",
        ));
    }
    if num_positions == 0 {
        return Err(AutogradError::TapeInvariant(
            "kl_distill_loss: num_positions must be > 0. Hint: pass rollout.len() \
             for OPD batch=1, or batch * seq_len for batched logits.",
        ));
    }
    let actual_positions = student.size / vocab;
    if actual_positions != num_positions {
        return Err(AutogradError::TapeInvariant(
            "kl_distill_loss: num_positions must match logits.numel() / vocab. \
             Hint: pass rollout.len() for OPD batch=1, or batch * seq_len for \
             batched logits.",
        ));
    }

    Ok(())
}
