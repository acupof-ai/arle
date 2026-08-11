use autograd::{
    AutogradError, Result, Tape, TensorId, TensorStore,
    ops::fused_linear_distill::{
        FusedLinearDistillDirection,
        fused_linear_distill_loss as autograd_fused_linear_distill_loss,
        fused_linear_distill_loss_sparse as autograd_fused_linear_distill_loss_sparse,
        generalized_jsd_loss as autograd_generalized_jsd_loss,
    },
    ops::{add, gather_last_dim, log_softmax, mean, mul, mul_scalar, slice, softmax},
};

pub const DEFAULT_KL_CHUNK_SIZE: usize = 32;

/// `batchmean` correction: `mean` reduces over positions×vocab, so `×vocab`
/// recovers `sum_v / positions`. Load-bearing — dropping it collapses the AdamW
/// effective LR by ~vocab× (`errors/2026-06-16-opd-kl-vocab-reduction-lr-collapse.md`).
#[inline]
fn kl_batchmean_scale(vocab: usize) -> f32 {
    debug_assert!(vocab > 0, "kl_batchmean_scale: vocab must be > 0");
    vocab as f32
}

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

/// Forward KL `KL(teacher || student)`. Drops the constant `-H(t)`, minimises
/// soft cross-entropy `-Σ_v t·log s`. Reduction: `batchmean` (see `kl_batchmean_scale`).
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
    let vocab_scale = kl_batchmean_scale(shape.vocab);
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

/// Chunked sibling of `kl_distill_loss` that limits KL intermediates to
/// `[prefix..., chunk, vocab]` while preserving the `batchmean` scale.
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
    // Each `chunk_avg` is a `mean` over positions×vocab, so `×vocab` recovers
    // the same `batchmean` scale as `kl_distill_loss`.
    let vocab_scale = kl_batchmean_scale(shape.vocab);
    match direction {
        KlDirection::Forward => mul_scalar(total, -temperature_sq * vocab_scale, store, tape),
        KlDirection::Reverse => mul_scalar(total, temperature_sq * vocab_scale, store, tape),
    }
}

pub fn generalized_beta_jsd_loss(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    temperature: f32,
    beta: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Endpoint semantics are explicit KL dispatches; do not use them as an interior-JSD continuity proof.
    validate_kl_beta(beta)?;
    if beta == 0.0 {
        return kl_distill_loss(
            student_logits,
            teacher_logits,
            num_positions,
            temperature,
            KlDirection::Forward,
            store,
            tape,
        );
    }
    if beta == 1.0 {
        return kl_distill_loss(
            student_logits,
            teacher_logits,
            num_positions,
            temperature,
            KlDirection::Reverse,
            store,
            tape,
        );
    }
    autograd_generalized_jsd_loss(
        student_logits,
        teacher_logits,
        num_positions,
        temperature,
        beta,
        store,
        tape,
    )
}

pub fn generalized_beta_jsd_loss_chunked(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    chunk_size: usize,
    temperature: f32,
    beta: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    validate_kl_beta(beta)?;
    if beta == 0.0 {
        return kl_distill_loss_chunked(
            student_logits,
            teacher_logits,
            num_positions,
            chunk_size,
            temperature,
            KlDirection::Forward,
            store,
            tape,
        );
    }
    if beta == 1.0 {
        return kl_distill_loss_chunked(
            student_logits,
            teacher_logits,
            num_positions,
            chunk_size,
            temperature,
            KlDirection::Reverse,
            store,
            tape,
        );
    }
    let shape = validate_kl_distill_inputs(student_logits, teacher_logits, num_positions, store)?;
    validate_kl_temperature(temperature)?;
    if chunk_size == 0 {
        return Err(AutogradError::TapeInvariant(
            "generalized_beta_jsd_loss_chunked: chunk_size must be > 0",
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
                    "generalized_beta_jsd_loss_chunked: chunk position count overflow",
                ))?;
        let chunk_weight = chunk_positions as f32 / num_positions as f32;

        let mut starts = vec![0; shape.rank];
        let mut ends = shape.dims.clone();
        starts[shape.seq_axis] = seq_start;
        ends[shape.seq_axis] = seq_end;

        let teacher_chunk = slice(teacher_logits, &starts, &ends, store, tape)?;
        let student_chunk = slice(student_logits, &starts, &ends, store, tape)?;
        let chunk_loss = generalized_beta_jsd_loss(
            student_chunk,
            teacher_chunk,
            chunk_positions,
            temperature,
            beta,
            store,
            tape,
        )?;
        let weighted_chunk = mul_scalar(chunk_loss, chunk_weight, store, tape)?;
        total = Some(match total {
            Some(previous) => add(previous, weighted_chunk, store, tape)?,
            None => weighted_chunk,
        });
    }

    total.ok_or(AutogradError::TapeInvariant(
        "generalized_beta_jsd_loss_chunked: no chunks were produced",
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn fused_linear_distill_loss(
    hidden: TensorId,
    lm_head: TensorId,
    teacher_logits: TensorId,
    row_start: usize,
    num_positions: usize,
    chunk_size: usize,
    temperature: f32,
    direction: KlDirection,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let direction = match direction {
        KlDirection::Forward => FusedLinearDistillDirection::Forward,
        KlDirection::Reverse => FusedLinearDistillDirection::Reverse,
    };
    autograd_fused_linear_distill_loss(
        hidden,
        lm_head,
        teacher_logits,
        row_start,
        num_positions,
        chunk_size,
        temperature,
        direction,
        store,
        tape,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fused_linear_distill_loss_sparse(
    hidden: TensorId,
    lm_head: TensorId,
    teacher_topk_log_probs: TensorId,
    teacher_topk_indices: &[i32],
    row_start: usize,
    num_positions: usize,
    chunk_size: usize,
    temperature: f32,
    direction: KlDirection,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if direction != KlDirection::Forward {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: sparse teacher top-k is forward-KL only",
        ));
    }
    autograd_fused_linear_distill_loss_sparse(
        hidden,
        lm_head,
        teacher_topk_log_probs,
        teacher_topk_indices,
        row_start,
        num_positions,
        chunk_size,
        temperature,
        store,
        tape,
    )
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

fn validate_kl_beta(beta: f32) -> Result<()> {
    if beta.is_finite() && (0.0..=1.0).contains(&beta) {
        return Ok(());
    }
    Err(AutogradError::TapeInvariant(
        "generalized_beta_jsd_loss: beta must be finite and in [0.0, 1.0]",
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
