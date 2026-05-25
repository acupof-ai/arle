use autograd::{
    AutogradError, Result, Tape, TensorId, TensorStore,
    ops::{add, gather_last_dim, log_softmax, mean, mul, mul_scalar, slice, softmax},
};

pub const DEFAULT_KL_CHUNK_SIZE: usize = 32;

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

/// Chunked sibling of [`kl_distill_loss`] that preserves the baseline
/// mean-over-positions-and-vocab scale while limiting KL intermediates to
/// `[prefix..., chunk, vocab]`.
///
/// The input logits may still be full-sequence tensors. This entrypoint
/// chunks the loss graph only; OPD/eval callers must stop materializing full
/// forward logits separately before this becomes an end-to-end peak-memory
/// fix.
pub fn kl_distill_loss_chunked(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    chunk_size: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let shape = validate_kl_distill_inputs(student_logits, teacher_logits, num_positions, store)?;
    if chunk_size == 0 {
        return Err(AutogradError::TapeInvariant(
            "kl_distill_loss_chunked: chunk_size must be > 0. \
             Hint: pass the maximum sequence positions to score per KL chunk.",
        ));
    }
    if shape.rank < 2 {
        return Err(AutogradError::TapeInvariant(
            "kl_distill_loss_chunked: logits must be shaped [..., seq_len, vocab]. \
             Hint: pass Qwen35Model forward logits shaped [batch, seq_len, vocab].",
        ));
    }
    if shape.seq_len == 0 {
        return Err(AutogradError::TapeInvariant(
            "kl_distill_loss_chunked: seq_len must be > 0. \
             Hint: pass at least one prompt or rollout token.",
        ));
    }

    let mut total = None;
    for seq_start in (0..shape.seq_len).step_by(chunk_size) {
        let seq_end = seq_start.saturating_add(chunk_size).min(shape.seq_len);
        let chunk_len = seq_end - seq_start;
        let chunk_positions =
            shape
                .prefix_positions
                .checked_mul(chunk_len)
                .ok_or(AutogradError::TapeInvariant(
                    "kl_distill_loss_chunked: chunk position count overflow",
                ))?;
        let chunk_weight = chunk_positions as f32 / num_positions as f32;

        let mut starts = vec![0; shape.rank];
        let mut ends = shape.dims.clone();
        starts[shape.seq_axis] = seq_start;
        ends[shape.seq_axis] = seq_end;

        let teacher_chunk = slice(teacher_logits, &starts, &ends, store, tape)?;
        let student_chunk = slice(student_logits, &starts, &ends, store, tape)?;
        let teacher_probs = softmax(teacher_chunk, store, tape)?;
        let student_log_probs = log_softmax(student_chunk, store, tape)?;
        let weighted = mul(teacher_probs, student_log_probs, store, tape)?;
        let chunk_avg = mean(weighted, store, tape)?;
        let weighted_chunk = mul_scalar(chunk_avg, chunk_weight, store, tape)?;
        total = Some(match total {
            Some(previous) => add(previous, weighted_chunk, store, tape)?,
            None => weighted_chunk,
        });
    }

    let total = total.ok_or(AutogradError::TapeInvariant(
        "kl_distill_loss_chunked: no chunks were produced",
    ))?;
    mul_scalar(total, -1.0, store, tape)
}

#[derive(Debug, Clone)]
struct KlDistillShape {
    dims: Vec<usize>,
    rank: usize,
    seq_axis: usize,
    seq_len: usize,
    prefix_positions: usize,
}

fn validate_kl_distill_inputs(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    store: &TensorStore,
) -> Result<KlDistillShape> {
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

    let rank = student.shape.len();
    let seq_axis = rank.saturating_sub(2);
    let seq_len = student.shape.get(seq_axis).copied().unwrap_or(0);
    let prefix_positions = if rank >= 2 {
        student.shape[..seq_axis].iter().product()
    } else {
        0
    };

    Ok(KlDistillShape {
        dims: student.shape.clone(),
        rank,
        seq_axis,
        seq_len,
        prefix_positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::{Tensor, TensorStore};

    fn deterministic_logits(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let mixed = (i.wrapping_mul(37).wrapping_add(salt * 19)) % 257;
                (mixed as f32 - 128.0) / 32.0
            })
            .collect()
    }

    fn loss_and_student_grad(
        student_logits: &[f32],
        teacher_logits: &[f32],
        shape: &[usize],
        chunk_size: Option<usize>,
    ) -> (f32, Vec<f32>) {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let student = store.alloc(
            Tensor::new(student_logits.to_vec(), shape.to_vec(), true).expect("student logits"),
        );
        let teacher = store.alloc(
            Tensor::new(teacher_logits.to_vec(), shape.to_vec(), false).expect("teacher logits"),
        );
        let vocab = shape.last().copied().expect("vocab dim");
        let num_positions = student_logits.len() / vocab;
        let loss = match chunk_size {
            Some(chunk_size) => kl_distill_loss_chunked(
                student,
                teacher,
                num_positions,
                chunk_size,
                &mut store,
                &mut tape,
            ),
            None => kl_distill_loss(student, teacher, num_positions, &mut store, &mut tape),
        }
        .expect("kl loss");
        let loss_value = store.to_host(loss).expect("loss host value")[0];
        tape.backward(loss, &mut store).expect("backward");

        let grad = store
            .get(student)
            .and_then(|tensor| tensor.grad)
            .expect("student logits gradient");
        let grad_values = store.to_host(grad).expect("gradient host value");
        (loss_value, grad_values)
    }

    fn assert_close(lhs: f32, rhs: f32, eps: f32, label: &str) {
        let abs = (lhs - rhs).abs();
        assert!(
            abs <= eps,
            "{label} mismatch: lhs={lhs:.10e} rhs={rhs:.10e} abs={abs:.3e} eps={eps:.3e}"
        );
    }

    fn assert_slice_close(lhs: &[f32], rhs: &[f32], eps: f32, label: &str) {
        assert_eq!(lhs.len(), rhs.len(), "{label} length mismatch");
        let mut worst = (0usize, 0.0_f32, 0.0_f32, 0.0_f32);
        for (i, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let abs = (a - b).abs();
            if abs > worst.3 {
                worst = (i, a, b, abs);
            }
        }
        assert!(
            worst.3 <= eps,
            "{label} mismatch at {}: lhs={:.10e} rhs={:.10e} abs={:.3e} eps={:.3e}",
            worst.0,
            worst.1,
            worst.2,
            worst.3,
            eps
        );
    }

    #[test]
    fn chunked_kl_matches_baseline_forward_and_student_grad() {
        let shape = [1, 64, 1024];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 3);
        let teacher_logits = deterministic_logits(len, 11);

        let (baseline_loss, baseline_grad) =
            loss_and_student_grad(&student_logits, &teacher_logits, &shape, None);
        let (chunked_loss, chunked_grad) =
            loss_and_student_grad(&student_logits, &teacher_logits, &shape, Some(8));

        assert_close(baseline_loss, chunked_loss, 1.0e-5, "chunk_size=8 loss");
        assert_slice_close(
            &baseline_grad,
            &chunked_grad,
            1.0e-5,
            "chunk_size=8 student gradient",
        );
    }

    #[test]
    fn chunked_kl_single_chunk_degenerates_to_baseline() {
        let shape = [1, 64, 1024];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 5);
        let teacher_logits = deterministic_logits(len, 17);

        let (baseline_loss, baseline_grad) =
            loss_and_student_grad(&student_logits, &teacher_logits, &shape, None);
        let (chunked_loss, chunked_grad) =
            loss_and_student_grad(&student_logits, &teacher_logits, &shape, Some(64));

        assert_close(baseline_loss, chunked_loss, 1.0e-5, "single-chunk loss");
        assert_slice_close(
            &baseline_grad,
            &chunked_grad,
            1.0e-5,
            "single-chunk student gradient",
        );
    }

    #[test]
    fn chunked_kl_chunk_size_one_boundary_is_finite() {
        let shape = [1, 64, 1024];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 7);
        let teacher_logits = deterministic_logits(len, 23);

        let (loss, grad) = loss_and_student_grad(&student_logits, &teacher_logits, &shape, Some(1));

        assert!(loss.is_finite(), "chunk_size=1 loss must be finite");
        assert_eq!(grad.len(), len);
        assert!(
            grad.iter().all(|value| value.is_finite()),
            "chunk_size=1 gradient must be finite"
        );
    }

    #[test]
    fn chunked_kl_synthetic_memory_sanity_for_512_token_qwen35_logits() {
        let bytes_per_f32 = std::mem::size_of::<f32>();
        let full_tensor_bytes = 512usize * 248_320usize * bytes_per_f32;
        let chunked_tensor_bytes = 64usize * 248_320usize * bytes_per_f32;

        assert_eq!(full_tensor_bytes, 508_559_360);
        assert_eq!(chunked_tensor_bytes, 63_569_920);
        assert_eq!(full_tensor_bytes / chunked_tensor_bytes, 8);
        assert_eq!(full_tensor_bytes * 2, 1_017_118_720);
        assert_eq!(chunked_tensor_bytes * 2, 127_139_840);
    }
}
