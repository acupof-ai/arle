use autograd::{
    AutogradError, Result, Tape, TensorId, TensorStore,
    ops::{add, gather_last_dim, log_softmax, mean, mul, mul_scalar, slice, softmax},
};

pub const DEFAULT_KL_CHUNK_SIZE: usize = 32;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum KlDirection {
    #[default]
    Forward,
    Reverse,
}

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
/// rollout length does not silently train against the wrong tensor shape.
///
/// Normalization = **`batchmean`** (sum over vocab, mean over positions):
/// `sum_v / num_positions`, the mathematically-correct KL reduction PyTorch's
/// `KLDivLoss` documents (`reduction='mean'`, i.e. dividing by `positions *
/// vocab` too, is the form PyTorch warns is *not* the correct KL). The earlier
/// `mean`-over-all-logits path was a constant `1 / vocab` (≈1/152k) rescale of
/// this; that is **not** absorbed by AdamW, because the per-parameter gradient
/// second moment `sqrt(v_hat)` is then pushed near/below `eps=1e-8`, so AdamW
/// degenerates from adaptive normalization to `eps`-dominated scaled-SGD and the
/// effective LR collapses by up to ~vocab×; it also makes `--grad-clip` never
/// fire and the displayed loss `1/vocab` too small. We multiply the `mean` by
/// `vocab` to recover `batchmean` while keeping the tested `mul_scalar`+`mean`
/// fast path (no `sum_backward` scalar-broadcast).
pub fn kl_distill_loss(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    temperature: f32,
    direction: KlDirection,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let shape = validate_kl_distill_inputs(student_logits, teacher_logits, num_positions, store)?;
    let vocab_scale = shape.vocab as f32;
    validate_kl_temperature(temperature)?;
    let inv_temperature = 1.0 / temperature;
    let temperature_sq = temperature * temperature;
    let teacher_logits = mul_scalar(teacher_logits, inv_temperature, store, tape)?;
    let student_logits = mul_scalar(student_logits, inv_temperature, store, tape)?;
    match direction {
        KlDirection::Forward => {
            let teacher_probs = softmax(teacher_logits, store, tape)?;
            let student_log_probs = log_softmax(student_logits, store, tape)?;
            let weighted = mul(teacher_probs, student_log_probs, store, tape)?;
            // `mean` reduces across all dims (positions × vocab); multiplying by
            // `vocab` recovers the `batchmean` reduction (`sum_v / positions`)
            // while reusing the tested `mul_scalar`+`mean` device path instead of
            // `sum_backward`'s untested scalar broadcast. The `1/vocab` factor is
            // NOT optimizer-invariant (see fn doc): it would push the gradient
            // below AdamW `eps` and collapse the effective LR.
            let avg = mean(weighted, store, tape)?;
            mul_scalar(avg, -temperature_sq * vocab_scale, store, tape)
        }
        KlDirection::Reverse => {
            let student_probs = softmax(student_logits, store, tape)?;
            let student_log_probs = log_softmax(student_logits, store, tape)?;
            let teacher_log_probs = log_softmax(teacher_logits, store, tape)?;
            let q_logq = mul(student_probs, student_log_probs, store, tape)?;
            let q_logp = mul(student_probs, teacher_log_probs, store, tape)?;
            let neg_q_logp = mul_scalar(q_logp, -1.0, store, tape)?;
            let elem = add(q_logq, neg_q_logp, store, tape)?;
            let avg = mean(elem, store, tape)?;
            mul_scalar(avg, temperature_sq * vocab_scale, store, tape)
        }
    }
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
    temperature: f32,
    direction: KlDirection,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let shape = validate_kl_distill_inputs(student_logits, teacher_logits, num_positions, store)?;
    validate_kl_temperature(temperature)?;
    let inv_temperature = 1.0 / temperature;
    let temperature_sq = temperature * temperature;
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
        let teacher_chunk = mul_scalar(teacher_chunk, inv_temperature, store, tape)?;
        let student_chunk = mul_scalar(student_chunk, inv_temperature, store, tape)?;
        let chunk_avg = match direction {
            KlDirection::Forward => {
                let teacher_probs = softmax(teacher_chunk, store, tape)?;
                let student_log_probs = log_softmax(student_chunk, store, tape)?;
                let weighted = mul(teacher_probs, student_log_probs, store, tape)?;
                mean(weighted, store, tape)?
            }
            KlDirection::Reverse => {
                let student_probs = softmax(student_chunk, store, tape)?;
                let student_log_probs = log_softmax(student_chunk, store, tape)?;
                let teacher_log_probs = log_softmax(teacher_chunk, store, tape)?;
                let q_logq = mul(student_probs, student_log_probs, store, tape)?;
                let q_logp = mul(student_probs, teacher_log_probs, store, tape)?;
                let neg_q_logp = mul_scalar(q_logp, -1.0, store, tape)?;
                let elem = add(q_logq, neg_q_logp, store, tape)?;
                mean(elem, store, tape)?
            }
        };
        let weighted_chunk = mul_scalar(chunk_avg, chunk_weight, store, tape)?;
        total = Some(match total {
            Some(previous) => add(previous, weighted_chunk, store, tape)?,
            None => weighted_chunk,
        });
    }

    let total = total.ok_or(AutogradError::TapeInvariant(
        "kl_distill_loss_chunked: no chunks were produced",
    ))?;
    // Multiply by `vocab` so the chunked path matches the `batchmean`
    // (`sum_v / positions`) scale of `kl_distill_loss`; each `chunk_avg` is a
    // `mean` over (positions × vocab), so the recovered scale is identical.
    let vocab_scale = shape.vocab as f32;
    match direction {
        KlDirection::Forward => mul_scalar(total, -temperature_sq * vocab_scale, store, tape),
        KlDirection::Reverse => mul_scalar(total, temperature_sq * vocab_scale, store, tape),
    }
}

fn validate_kl_temperature(temperature: f32) -> Result<()> {
    if temperature.is_finite() && temperature > 0.0 {
        return Ok(());
    }
    Err(AutogradError::TapeInvariant(
        "kl_distill_loss: temperature must be finite and > 0.0. \
         Hint: use 1.0 for baseline KL or a positive value for opt-in \
         pure-OPD temperature softening.",
    ))
}

#[derive(Debug, Clone)]
struct KlDistillShape {
    dims: Vec<usize>,
    rank: usize,
    seq_axis: usize,
    seq_len: usize,
    prefix_positions: usize,
    vocab: usize,
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
        vocab,
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
        direction: KlDirection,
    ) -> (f32, Vec<f32>) {
        loss_and_student_grad_with_temperature(
            student_logits,
            teacher_logits,
            shape,
            chunk_size,
            1.0,
            direction,
        )
    }

    fn loss_and_student_grad_with_temperature(
        student_logits: &[f32],
        teacher_logits: &[f32],
        shape: &[usize],
        chunk_size: Option<usize>,
        temperature: f32,
        direction: KlDirection,
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
                temperature,
                direction,
                &mut store,
                &mut tape,
            ),
            None => kl_distill_loss(
                student,
                teacher,
                num_positions,
                temperature,
                direction,
                &mut store,
                &mut tape,
            ),
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

    fn untempered_loss_and_student_grad(
        student_logits: &[f32],
        teacher_logits: &[f32],
        shape: &[usize],
        direction: KlDirection,
    ) -> (f32, Vec<f32>) {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let student = store.alloc(
            Tensor::new(student_logits.to_vec(), shape.to_vec(), true).expect("student logits"),
        );
        let teacher = store.alloc(
            Tensor::new(teacher_logits.to_vec(), shape.to_vec(), false).expect("teacher logits"),
        );
        // Mirror `kl_distill_loss`'s `batchmean` scale: `mean` over (positions ×
        // vocab) then `* vocab` to recover `sum_v / positions`. -1.0 * vocab and
        // +1.0 * vocab match the real path's `-temperature_sq * vocab_scale` /
        // `temperature_sq * vocab_scale` at temperature == 1.0 byte-for-byte.
        let vocab = shape.last().copied().expect("vocab dim") as f32;
        let loss = match direction {
            KlDirection::Forward => {
                let teacher_probs = softmax(teacher, &mut store, &mut tape).expect("teacher probs");
                let student_log_probs =
                    log_softmax(student, &mut store, &mut tape).expect("student log probs");
                let weighted =
                    mul(teacher_probs, student_log_probs, &mut store, &mut tape).expect("weighted");
                let avg = mean(weighted, &mut store, &mut tape).expect("mean");
                mul_scalar(avg, -1.0 * vocab, &mut store, &mut tape).expect("forward loss")
            }
            KlDirection::Reverse => {
                let student_probs = softmax(student, &mut store, &mut tape).expect("student probs");
                let student_log_probs =
                    log_softmax(student, &mut store, &mut tape).expect("student log probs");
                let teacher_log_probs =
                    log_softmax(teacher, &mut store, &mut tape).expect("teacher log probs");
                let q_logq =
                    mul(student_probs, student_log_probs, &mut store, &mut tape).expect("q log q");
                let q_logp =
                    mul(student_probs, teacher_log_probs, &mut store, &mut tape).expect("q log p");
                let neg_q_logp = mul_scalar(q_logp, -1.0, &mut store, &mut tape).expect("-q log p");
                let elem = add(q_logq, neg_q_logp, &mut store, &mut tape).expect("reverse elem");
                let avg = mean(elem, &mut store, &mut tape).expect("reverse mean");
                mul_scalar(avg, vocab, &mut store, &mut tape).expect("reverse loss")
            }
        };
        let loss_value = store.to_host(loss).expect("loss host value")[0];
        tape.backward(loss, &mut store).expect("backward");

        let grad = store
            .get(student)
            .and_then(|tensor| tensor.grad)
            .expect("student logits gradient");
        let grad_values = store.to_host(grad).expect("gradient host value");
        (loss_value, grad_values)
    }

    /// Relative slack added on top of the absolute floor `eps`. The chunked vs
    /// non-chunked equivalence is a *relative* property (different summation
    /// order over softmax-of-1024 rows leaves ~2e-5 relative noise); since the
    /// `batchmean` rescale multiplies the loss/grad magnitude by `vocab`, a pure
    /// absolute bound is scale-fragile. Tolerance = `eps + REL_TOL * |rhs|`.
    const REL_TOL: f32 = 2.0e-4;

    fn assert_close(lhs: f32, rhs: f32, eps: f32, label: &str) {
        let abs = (lhs - rhs).abs();
        let tol = eps + REL_TOL * rhs.abs();
        assert!(
            abs <= tol,
            "{label} mismatch: lhs={lhs:.10e} rhs={rhs:.10e} abs={abs:.3e} tol={tol:.3e}"
        );
    }

    fn assert_slice_close(lhs: &[f32], rhs: &[f32], eps: f32, label: &str) {
        assert_eq!(lhs.len(), rhs.len(), "{label} length mismatch");
        let mut worst = (0usize, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        for (i, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let abs = (a - b).abs();
            let tol = eps + REL_TOL * b.abs();
            // track the largest abs-over-tol violation, not the largest abs
            let slack = abs - tol;
            if slack > worst.4 {
                worst = (i, a, b, abs, slack);
            }
        }
        assert!(
            worst.4 <= 0.0,
            "{label} mismatch at {}: lhs={:.10e} rhs={:.10e} abs={:.3e} slack={:.3e}",
            worst.0,
            worst.1,
            worst.2,
            worst.3,
            worst.4
        );
    }

    fn loss_value(
        student_logits: &[f32],
        teacher_logits: &[f32],
        shape: &[usize],
        direction: KlDirection,
    ) -> f32 {
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
        let loss = kl_distill_loss(
            student,
            teacher,
            num_positions,
            1.0,
            direction,
            &mut store,
            &mut tape,
        )
        .expect("kl loss");
        store.to_host(loss).expect("loss host value")[0]
    }

    fn reference_reverse_kl_mean(
        student_logits: &[f32],
        teacher_logits: &[f32],
        vocab: usize,
    ) -> f64 {
        student_logits
            .chunks_exact(vocab)
            .zip(teacher_logits.chunks_exact(vocab))
            .flat_map(|(student_row, teacher_row)| {
                let student_log_probs = reference_log_softmax(student_row);
                let teacher_log_probs = reference_log_softmax(teacher_row);
                student_log_probs
                    .into_iter()
                    .zip(teacher_log_probs)
                    .map(|(student_log_prob, teacher_log_prob)| {
                        student_log_prob.exp() * (student_log_prob - teacher_log_prob)
                    })
                    .collect::<Vec<_>>()
            })
            .sum::<f64>()
            // `batchmean`: sum over vocab, mean over positions (= len / vocab),
            // matching `kl_distill_loss`'s `vocab`-rescaled normalization.
            / (student_logits.len() / vocab) as f64
    }

    /// Independent f64 reference for the FORWARD path. The production forward
    /// loss minimizes the soft cross-entropy `-sum_v t_p * log s_p` (the `-H(t)`
    /// term is dropped, see `kl_distill_loss` doc), reduced `batchmean` (sum over
    /// vocab, mean over positions). Recomputed from scratch (softmax the teacher,
    /// log_softmax the student) so it does NOT reuse the production
    /// `-1.0 * vocab` construction — this is the independent forward check the
    /// suite previously lacked (only `reference_reverse_kl_mean` existed).
    fn reference_forward_kl_mean(
        student_logits: &[f32],
        teacher_logits: &[f32],
        vocab: usize,
    ) -> f64 {
        student_logits
            .chunks_exact(vocab)
            .zip(teacher_logits.chunks_exact(vocab))
            .map(|(student_row, teacher_row)| {
                let student_log_probs = reference_log_softmax(student_row);
                let teacher_log_probs = reference_log_softmax(teacher_row);
                student_log_probs
                    .into_iter()
                    .zip(teacher_log_probs)
                    .map(|(student_log_prob, teacher_log_prob)| {
                        -teacher_log_prob.exp() * student_log_prob
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            / (student_logits.len() / vocab) as f64
    }

    fn reference_log_softmax(row: &[f32]) -> Vec<f64> {
        let max_value = row
            .iter()
            .copied()
            .map(f64::from)
            .fold(f64::NEG_INFINITY, f64::max);
        let denom = row
            .iter()
            .map(|&value| (f64::from(value) - max_value).exp())
            .sum::<f64>();
        let log_denom = denom.ln();
        row.iter()
            .map(|&value| f64::from(value) - max_value - log_denom)
            .collect()
    }

    #[test]
    fn chunked_kl_matches_baseline_forward_and_student_grad() {
        let shape = [1, 64, 1024];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 3);
        let teacher_logits = deterministic_logits(len, 11);

        let (baseline_loss, baseline_grad) = loss_and_student_grad(
            &student_logits,
            &teacher_logits,
            &shape,
            None,
            KlDirection::Forward,
        );
        let (chunked_loss, chunked_grad) = loss_and_student_grad(
            &student_logits,
            &teacher_logits,
            &shape,
            Some(8),
            KlDirection::Forward,
        );

        assert_close(baseline_loss, chunked_loss, 1.0e-5, "chunk_size=8 loss");
        assert_slice_close(
            &baseline_grad,
            &chunked_grad,
            1.0e-5,
            "chunk_size=8 student gradient",
        );
    }

    #[test]
    fn chunked_kl_matches_baseline_reverse_and_student_grad() {
        let shape = [1, 64, 1024];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 13);
        let teacher_logits = deterministic_logits(len, 31);

        let (baseline_loss, baseline_grad) = loss_and_student_grad(
            &student_logits,
            &teacher_logits,
            &shape,
            None,
            KlDirection::Reverse,
        );
        let (chunked_loss, chunked_grad) = loss_and_student_grad(
            &student_logits,
            &teacher_logits,
            &shape,
            Some(8),
            KlDirection::Reverse,
        );

        assert_close(
            baseline_loss,
            chunked_loss,
            1.0e-5,
            "reverse chunk_size=8 loss",
        );
        assert_slice_close(
            &baseline_grad,
            &chunked_grad,
            1.0e-5,
            "reverse chunk_size=8 student gradient",
        );
    }

    #[test]
    fn chunked_kl_single_chunk_degenerates_to_baseline() {
        let shape = [1, 64, 1024];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 5);
        let teacher_logits = deterministic_logits(len, 17);

        let (baseline_loss, baseline_grad) = loss_and_student_grad(
            &student_logits,
            &teacher_logits,
            &shape,
            None,
            KlDirection::Forward,
        );
        let (chunked_loss, chunked_grad) = loss_and_student_grad(
            &student_logits,
            &teacher_logits,
            &shape,
            Some(64),
            KlDirection::Forward,
        );

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

        let (loss, grad) = loss_and_student_grad(
            &student_logits,
            &teacher_logits,
            &shape,
            Some(1),
            KlDirection::Forward,
        );

        assert!(loss.is_finite(), "chunk_size=1 loss must be finite");
        assert_eq!(grad.len(), len);
        assert!(
            grad.iter().all(|value| value.is_finite()),
            "chunk_size=1 gradient must be finite"
        );
    }

    #[test]
    fn kl_temperature_one_is_byte_identical_to_untempered_forward_and_reverse() {
        let shape = [2, 3, 5];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 37);
        let teacher_logits = deterministic_logits(len, 43);

        for direction in [KlDirection::Forward, KlDirection::Reverse] {
            let (baseline_loss, baseline_grad) = untempered_loss_and_student_grad(
                &student_logits,
                &teacher_logits,
                &shape,
                direction,
            );
            let (temperature_one_loss, temperature_one_grad) =
                loss_and_student_grad_with_temperature(
                    &student_logits,
                    &teacher_logits,
                    &shape,
                    None,
                    1.0,
                    direction,
                );

            assert_eq!(
                temperature_one_loss.to_bits(),
                baseline_loss.to_bits(),
                "temperature=1.0 loss must be byte-identical for {direction:?}"
            );
            assert_eq!(
                temperature_one_grad.len(),
                baseline_grad.len(),
                "temperature=1.0 gradient len mismatch for {direction:?}"
            );
            for (idx, (&actual, &expected)) in temperature_one_grad
                .iter()
                .zip(baseline_grad.iter())
                .enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "temperature=1.0 grad[{idx}] must be byte-identical for {direction:?}"
                );
            }
        }
    }

    #[test]
    fn chunked_kl_matches_baseline_with_temperature() {
        let shape = [1, 17, 31];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 47);
        let teacher_logits = deterministic_logits(len, 53);

        for direction in [KlDirection::Forward, KlDirection::Reverse] {
            let (baseline_loss, baseline_grad) = loss_and_student_grad_with_temperature(
                &student_logits,
                &teacher_logits,
                &shape,
                None,
                2.0,
                direction,
            );
            let (chunked_loss, chunked_grad) = loss_and_student_grad_with_temperature(
                &student_logits,
                &teacher_logits,
                &shape,
                Some(5),
                2.0,
                direction,
            );

            assert_close(
                baseline_loss,
                chunked_loss,
                1.0e-5,
                &format!("temperature=2 chunked loss {direction:?}"),
            );
            assert_slice_close(
                &baseline_grad,
                &chunked_grad,
                1.0e-5,
                &format!("temperature=2 chunked student gradient {direction:?}"),
            );
        }
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

    #[test]
    fn reverse_kl_matches_hand_computed_two_row_reference() {
        let shape = [2, 4];
        let student_logits = vec![0.3, -0.7, 1.2, 0.0, -0.2, 0.9, 0.1, -1.1];
        let teacher_logits = vec![-0.4, 0.6, 0.2, -0.1, 1.3, -0.8, 0.4, 0.0];

        let actual = loss_value(
            &student_logits,
            &teacher_logits,
            &shape,
            KlDirection::Reverse,
        );
        let expected = reference_reverse_kl_mean(&student_logits, &teacher_logits, shape[1]);

        assert_close(actual, expected as f32, 1.0e-5, "reverse KL reference");
    }

    #[test]
    fn forward_kl_matches_hand_computed_reference() {
        // Independent absolute-value check for the FORWARD batchmean path,
        // closing the audit-found test gap (forward previously rested only on
        // chunked-equivalence + byte-identity to a helper that reused the same
        // `-1.0 * vocab` construction = self-consistency, not independence).
        let shape = [2, 4];
        let student_logits = vec![0.3, -0.7, 1.2, 0.0, -0.2, 0.9, 0.1, -1.1];
        let teacher_logits = vec![-0.4, 0.6, 0.2, -0.1, 1.3, -0.8, 0.4, 0.0];

        let actual = loss_value(
            &student_logits,
            &teacher_logits,
            &shape,
            KlDirection::Forward,
        );
        let expected = reference_forward_kl_mean(&student_logits, &teacher_logits, shape[1]);

        assert_close(actual, expected as f32, 1.0e-5, "forward KL reference");
    }

    #[test]
    fn reverse_kl_identical_logits_is_zero() {
        let shape = [2, 4];
        let logits = vec![0.3, -0.7, 1.2, 0.0, -0.2, 0.9, 0.1, -1.1];

        let actual = loss_value(&logits, &logits, &shape, KlDirection::Reverse);

        assert!(
            actual.abs() <= 1.0e-6,
            "identical-logit reverse KL should be zero, got {actual:.10e}"
        );
    }

    #[test]
    fn reverse_kl_random_finite_logits_is_non_negative() {
        let shape = [3, 4];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 29);
        let teacher_logits = deterministic_logits(len, 41);

        let actual = loss_value(
            &student_logits,
            &teacher_logits,
            &shape,
            KlDirection::Reverse,
        );

        assert!(actual.is_finite(), "reverse KL must be finite");
        assert!(
            actual >= -1.0e-6,
            "reverse KL should be non-negative within numerical tolerance, got {actual:.10e}"
        );
    }
}
