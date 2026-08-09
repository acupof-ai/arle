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

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::{Tensor, TensorStore, ops::matmul_bt};

    fn deterministic_logits(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let mixed = (i.wrapping_mul(37).wrapping_add(salt * 19)) % 257;
                (mixed as f32 - 128.0) / 32.0
            })
            .collect()
    }

    fn linear_logits(
        hidden: &[f32],
        weight: &[f32],
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
    ) -> Vec<f32> {
        let mut logits = vec![0.0_f32; rows * vocab];
        for row in 0..rows {
            for vocab_index in 0..vocab {
                let mut dot = 0.0_f32;
                for hidden_index in 0..hidden_dim {
                    dot += hidden[row * hidden_dim + hidden_index]
                        * weight[vocab_index * hidden_dim + hidden_index];
                }
                logits[row * vocab + vocab_index] = dot;
            }
        }
        logits
    }

    fn dense_linear_kl_loss_and_grads(
        hidden_values: &[f32],
        weight_values: &[f32],
        teacher_logits: &[f32],
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        direction: KlDirection,
    ) -> (f32, Vec<f32>, Vec<f32>) {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let hidden = store.alloc(
            Tensor::new(hidden_values.to_vec(), vec![rows, hidden_dim], true).expect("hidden"),
        );
        let weight = store.alloc(
            Tensor::new(weight_values.to_vec(), vec![vocab, hidden_dim], true).expect("weight"),
        );
        let teacher = store.alloc(
            Tensor::new(teacher_logits.to_vec(), vec![rows, vocab], false).expect("teacher logits"),
        );
        let student_logits = matmul_bt(hidden, weight, &mut store, &mut tape).expect("matmul bt");
        let loss = kl_distill_loss(
            student_logits,
            teacher,
            rows,
            temperature,
            direction,
            &mut store,
            &mut tape,
        )
        .expect("dense KL");
        let loss_value = store.to_host(loss).expect("loss host")[0];
        tape.backward(loss, &mut store).expect("dense backward");
        let hidden_grad = store
            .get(hidden)
            .and_then(|tensor| tensor.grad)
            .expect("hidden grad");
        let weight_grad = store
            .get(weight)
            .and_then(|tensor| tensor.grad)
            .expect("weight grad");
        (
            loss_value,
            store.to_host(hidden_grad).expect("hidden grad host"),
            store.to_host(weight_grad).expect("weight grad host"),
        )
    }

    fn fused_linear_kl_loss_and_grads(
        hidden_values: &[f32],
        weight_values: &[f32],
        teacher_logits: &[f32],
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        direction: KlDirection,
    ) -> (f32, Vec<f32>, Vec<f32>) {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let hidden = store.alloc(
            Tensor::new(hidden_values.to_vec(), vec![rows, hidden_dim], true).expect("hidden"),
        );
        let weight = store.alloc(
            Tensor::new(weight_values.to_vec(), vec![vocab, hidden_dim], true).expect("weight"),
        );
        let teacher = store.alloc(
            Tensor::new(teacher_logits.to_vec(), vec![rows, vocab], false).expect("teacher logits"),
        );
        let loss = fused_linear_distill_loss(
            hidden,
            weight,
            teacher,
            0,
            rows,
            3,
            temperature,
            direction,
            &mut store,
            &mut tape,
        )
        .expect("fused linear distill");
        let loss_value = store.to_host(loss).expect("loss host")[0];
        tape.backward(loss, &mut store).expect("fused backward");
        let hidden_grad = store
            .get(hidden)
            .and_then(|tensor| tensor.grad)
            .expect("hidden grad");
        let weight_grad = store
            .get(weight)
            .and_then(|tensor| tensor.grad)
            .expect("weight grad");
        (
            loss_value,
            store.to_host(hidden_grad).expect("hidden grad host"),
            store.to_host(weight_grad).expect("weight grad host"),
        )
    }

    fn sparse_linear_kl_loss_and_grads(
        hidden_values: &[f32],
        weight_values: &[f32],
        teacher_topk_log_probs: &[f32],
        teacher_topk_indices: &[i32],
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        topk: usize,
        temperature: f32,
    ) -> (f32, Vec<f32>, Vec<f32>) {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let hidden = store.alloc(
            Tensor::new(hidden_values.to_vec(), vec![rows, hidden_dim], true).expect("hidden"),
        );
        let weight = store.alloc(
            Tensor::new(weight_values.to_vec(), vec![vocab, hidden_dim], true).expect("weight"),
        );
        let teacher = store.alloc(
            Tensor::new(teacher_topk_log_probs.to_vec(), vec![rows, topk], false)
                .expect("teacher top-k log-probs"),
        );
        let loss = fused_linear_distill_loss_sparse(
            hidden,
            weight,
            teacher,
            teacher_topk_indices,
            0,
            rows,
            3,
            temperature,
            KlDirection::Forward,
            &mut store,
            &mut tape,
        )
        .expect("sparse fused linear distill");
        let loss_value = store.to_host(loss).expect("loss host")[0];
        tape.backward(loss, &mut store).expect("sparse backward");
        let hidden_grad = store
            .get(hidden)
            .and_then(|tensor| tensor.grad)
            .expect("hidden grad");
        let weight_grad = store
            .get(weight)
            .and_then(|tensor| tensor.grad)
            .expect("weight grad");
        (
            loss_value,
            store.to_host(hidden_grad).expect("hidden grad host"),
            store.to_host(weight_grad).expect("weight grad host"),
        )
    }

    fn sparse_linear_kl_loss_value(
        hidden_values: &[f32],
        weight_values: &[f32],
        teacher_topk_log_probs: &[f32],
        teacher_topk_indices: &[i32],
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        topk: usize,
        temperature: f32,
    ) -> f32 {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let hidden = store.alloc(
            Tensor::new(hidden_values.to_vec(), vec![rows, hidden_dim], true).expect("hidden"),
        );
        let weight = store.alloc(
            Tensor::new(weight_values.to_vec(), vec![vocab, hidden_dim], true).expect("weight"),
        );
        let teacher = store.alloc(
            Tensor::new(teacher_topk_log_probs.to_vec(), vec![rows, topk], false)
                .expect("teacher top-k log-probs"),
        );
        let loss = fused_linear_distill_loss_sparse(
            hidden,
            weight,
            teacher,
            teacher_topk_indices,
            0,
            rows,
            2,
            temperature,
            KlDirection::Forward,
            &mut store,
            &mut tape,
        )
        .expect("sparse fused linear distill");
        store.to_host(loss).expect("loss host")[0]
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

    fn beta_jsd_loss_and_student_grad(
        student_logits: &[f32],
        teacher_logits: &[f32],
        shape: &[usize],
        beta: f32,
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
        let loss = generalized_beta_jsd_loss(
            student,
            teacher,
            num_positions,
            1.0,
            beta,
            &mut store,
            &mut tape,
        )
        .expect("beta JSD loss");
        let loss_value = store.to_host(loss).expect("loss host value")[0];
        tape.backward(loss, &mut store).expect("backward");

        let grad = store
            .get(student)
            .and_then(|tensor| tensor.grad)
            .expect("student logits gradient");
        let grad_values = store.to_host(grad).expect("gradient host value");
        (loss_value, grad_values)
    }

    fn beta_jsd_loss_and_student_grad_chunked(
        student_logits: &[f32],
        teacher_logits: &[f32],
        shape: &[usize],
        chunk_size: Option<usize>,
        beta: f32,
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
            Some(chunk_size) => generalized_beta_jsd_loss_chunked(
                student,
                teacher,
                num_positions,
                chunk_size,
                1.0,
                beta,
                &mut store,
                &mut tape,
            ),
            None => generalized_beta_jsd_loss(
                student,
                teacher,
                num_positions,
                1.0,
                beta,
                &mut store,
                &mut tape,
            ),
        }
        .expect("beta JSD loss");
        let loss_value = store.to_host(loss).expect("loss host value")[0];
        tape.backward(loss, &mut store).expect("backward");

        let grad = store
            .get(student)
            .and_then(|tensor| tensor.grad)
            .expect("student logits gradient");
        let grad_values = store.to_host(grad).expect("gradient host value");
        (loss_value, grad_values)
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
        let vocab = shape.last().copied().expect("vocab dim") as f32;
        let loss = match direction {
            KlDirection::Forward => {
                let teacher_probs = softmax(teacher, &mut store, &mut tape).expect("teacher probs");
                let student_log_probs =
                    log_softmax(student, &mut store, &mut tape).expect("student log probs");
                let weighted =
                    mul(teacher_probs, student_log_probs, &mut store, &mut tape).expect("weighted");
                let avg = mean(weighted, &mut store, &mut tape).expect("mean");
                mul_scalar(avg, -vocab, &mut store, &mut tape).expect("forward loss")
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
            / (student_logits.len() / vocab) as f64
    }

    /// Independent f64 reference for the forward path (softmax teacher, log_softmax student).
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

    fn reference_generalized_jsd_mean(
        student_logits: &[f32],
        teacher_logits: &[f32],
        vocab: usize,
        beta: f64,
    ) -> f64 {
        student_logits
            .chunks_exact(vocab)
            .zip(teacher_logits.chunks_exact(vocab))
            .map(|(student_row, teacher_row)| {
                let student_log_probs = reference_log_softmax(student_row);
                let teacher_log_probs = reference_log_softmax(teacher_row);
                student_log_probs
                    .iter()
                    .zip(teacher_log_probs.iter())
                    .map(|(&student_log_prob, &teacher_log_prob)| {
                        let student_prob = student_log_prob.exp();
                        let teacher_prob = teacher_log_prob.exp();
                        let mixture = (1.0 - beta) * student_prob + beta * teacher_prob;
                        beta * teacher_prob * (teacher_log_prob - mixture.ln())
                            + (1.0 - beta) * student_prob * (student_log_prob - mixture.ln())
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

    fn log_softmax_row_f32(row: &[f32]) -> Vec<f32> {
        let max_value = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = row
            .iter()
            .map(|value| (*value - max_value).exp())
            .sum::<f32>();
        let log_denom = denom.ln();
        row.iter()
            .map(|value| (*value - max_value) - log_denom)
            .collect()
    }

    fn topk_log_probs_from_teacher_logits(
        teacher_logits: &[f32],
        rows: usize,
        vocab: usize,
        topk: usize,
        temperature: f32,
    ) -> (Vec<f32>, Vec<i32>, f64) {
        let inv_temperature = 1.0 / temperature;
        let mut topk_log_probs = Vec::with_capacity(rows * topk);
        let mut topk_indices = Vec::with_capacity(rows * topk);
        let mut max_missing_mass = 0.0_f64;
        for row in 0..rows {
            let base = row * vocab;
            let scaled = teacher_logits[base..base + vocab]
                .iter()
                .map(|value| *value * inv_temperature)
                .collect::<Vec<_>>();
            let log_probs = log_softmax_row_f32(&scaled);
            let mut order = (0..vocab).collect::<Vec<_>>();
            order.sort_by(|&lhs, &rhs| log_probs[rhs].total_cmp(&log_probs[lhs]));
            let mut selected_mass = 0.0_f64;
            for &vocab_index in order.iter().take(topk) {
                topk_log_probs.push(log_probs[vocab_index]);
                topk_indices.push(vocab_index as i32);
                selected_mass += f64::from(log_probs[vocab_index]).exp();
            }
            max_missing_mass = max_missing_mass.max((1.0 - selected_mass).max(0.0));
        }
        (topk_log_probs, topk_indices, max_missing_mass)
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
    fn generalized_beta_jsd_matches_trl_endpoints_and_midpoint() {
        let shape = [2, 3, 11];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 61);
        let teacher_logits = deterministic_logits(len, 67);

        let (forward_loss, forward_grad) = loss_and_student_grad(
            &student_logits,
            &teacher_logits,
            &shape,
            None,
            KlDirection::Forward,
        );
        let (beta_zero_loss, beta_zero_grad) =
            beta_jsd_loss_and_student_grad(&student_logits, &teacher_logits, &shape, 0.0);
        assert_close(beta_zero_loss, forward_loss, 0.0, "beta=0 forward KL loss");
        assert_slice_close(
            &beta_zero_grad,
            &forward_grad,
            0.0,
            "beta=0 forward KL grad",
        );

        let (reverse_loss, reverse_grad) = loss_and_student_grad(
            &student_logits,
            &teacher_logits,
            &shape,
            None,
            KlDirection::Reverse,
        );
        let (beta_one_loss, beta_one_grad) =
            beta_jsd_loss_and_student_grad(&student_logits, &teacher_logits, &shape, 1.0);
        assert_close(beta_one_loss, reverse_loss, 0.0, "beta=1 reverse KL loss");
        assert_slice_close(&beta_one_grad, &reverse_grad, 0.0, "beta=1 reverse KL grad");

        let (midpoint_loss, _) =
            beta_jsd_loss_and_student_grad(&student_logits, &teacher_logits, &shape, 0.5);
        let expected_midpoint =
            reference_generalized_jsd_mean(&student_logits, &teacher_logits, shape[2], 0.5);
        assert_close(
            midpoint_loss,
            expected_midpoint as f32,
            1.0e-5,
            "beta=0.5 generalized JSD midpoint",
        );
    }

    #[test]
    fn generalized_beta_jsd_midpoint_student_grad_matches_finite_difference() {
        let shape = [2, 2, 5];
        let len = shape.iter().product();
        let mut student_logits = deterministic_logits(len, 97);
        let teacher_logits = deterministic_logits(len, 109);
        let beta = 0.5;
        let (_loss, grad) =
            beta_jsd_loss_and_student_grad(&student_logits, &teacher_logits, &shape, beta);
        let eps = 1.0e-2_f32;

        for index in 0..student_logits.len() {
            let original = student_logits[index];
            student_logits[index] = original + eps;
            let plus = reference_generalized_jsd_mean(
                &student_logits,
                &teacher_logits,
                shape[2],
                f64::from(beta),
            );
            student_logits[index] = original - eps;
            let minus = reference_generalized_jsd_mean(
                &student_logits,
                &teacher_logits,
                shape[2],
                f64::from(beta),
            );
            student_logits[index] = original;

            let finite_diff = ((plus - minus) / (2.0 * f64::from(eps))) as f32;
            assert_close(
                grad[index],
                finite_diff,
                2.0e-3,
                &format!("beta=0.5 student grad[{index}]"),
            );
        }
    }

    #[test]
    fn chunked_generalized_beta_jsd_matches_unchunked_midpoint() {
        let shape = [1, 19, 13];
        let len = shape.iter().product();
        let student_logits = deterministic_logits(len, 101);
        let teacher_logits = deterministic_logits(len, 103);

        let (baseline_loss, baseline_grad) = beta_jsd_loss_and_student_grad_chunked(
            &student_logits,
            &teacher_logits,
            &shape,
            None,
            0.5,
        );
        let (chunked_loss, chunked_grad) = beta_jsd_loss_and_student_grad_chunked(
            &student_logits,
            &teacher_logits,
            &shape,
            Some(8),
            0.5,
        );

        assert_close(baseline_loss, chunked_loss, 1.0e-5, "chunked beta-JSD loss");
        assert_slice_close(
            &baseline_grad,
            &chunked_grad,
            1.0e-5,
            "chunked beta-JSD student grad",
        );
    }

    #[test]
    fn fused_linear_distill_matches_dense_linear_kl_loss_and_grads() {
        let rows = 11;
        let hidden_dim = 7;
        let vocab = 19;
        let hidden = deterministic_logits(rows * hidden_dim, 5);
        let weight = deterministic_logits(vocab * hidden_dim, 17);
        let teacher_logits = deterministic_logits(rows * vocab, 29);

        for direction in [KlDirection::Forward, KlDirection::Reverse] {
            let (dense_loss, dense_hidden_grad, dense_weight_grad) = dense_linear_kl_loss_and_grads(
                &hidden,
                &weight,
                &teacher_logits,
                rows,
                hidden_dim,
                vocab,
                1.7,
                direction,
            );
            let (fused_loss, fused_hidden_grad, fused_weight_grad) = fused_linear_kl_loss_and_grads(
                &hidden,
                &weight,
                &teacher_logits,
                rows,
                hidden_dim,
                vocab,
                1.7,
                direction,
            );

            assert_close(
                fused_loss,
                dense_loss,
                2.0e-5,
                &format!("fused linear loss {direction:?}"),
            );
            assert_slice_close(
                &fused_hidden_grad,
                &dense_hidden_grad,
                2.0e-5,
                &format!("fused hidden grad {direction:?}"),
            );
            assert_slice_close(
                &fused_weight_grad,
                &dense_weight_grad,
                2.0e-5,
                &format!("fused weight grad {direction:?}"),
            );
        }
    }

    #[test]
    fn fused_linear_sparse_topk_matches_dense_when_missing_mass_tends_to_zero() {
        let rows = 7;
        let hidden_dim = 5;
        let vocab = 11;
        let topk = 4;
        let temperature = 1.6;
        let hidden = deterministic_logits(rows * hidden_dim, 61);
        let weight = deterministic_logits(vocab * hidden_dim, 67);
        let mut teacher_logits = vec![-42.0_f32; rows * vocab];
        for row in 0..rows {
            for slot in 0..topk {
                let vocab_index = (row * 3 + slot * 2) % vocab;
                teacher_logits[row * vocab + vocab_index] =
                    2.5 - slot as f32 * 0.7 + row as f32 * 0.03;
            }
        }
        let (topk_log_probs, topk_indices, missing_mass) =
            topk_log_probs_from_teacher_logits(&teacher_logits, rows, vocab, topk, temperature);
        assert!(
            missing_mass <= 1.0e-6,
            "synthetic top-k teacher tail mass must be ~0, got {missing_mass:.3e}"
        );

        let (dense_loss, dense_hidden_grad, dense_weight_grad) = dense_linear_kl_loss_and_grads(
            &hidden,
            &weight,
            &teacher_logits,
            rows,
            hidden_dim,
            vocab,
            temperature,
            KlDirection::Forward,
        );
        let (sparse_loss, sparse_hidden_grad, sparse_weight_grad) = sparse_linear_kl_loss_and_grads(
            &hidden,
            &weight,
            &topk_log_probs,
            &topk_indices,
            rows,
            hidden_dim,
            vocab,
            topk,
            temperature,
        );

        assert_close(sparse_loss, dense_loss, 2.0e-5, "sparse top-k loss");
        assert_slice_close(
            &sparse_hidden_grad,
            &dense_hidden_grad,
            2.0e-5,
            "sparse top-k hidden grad",
        );
        assert_slice_close(
            &sparse_weight_grad,
            &dense_weight_grad,
            2.0e-5,
            "sparse top-k weight grad",
        );
    }

    #[test]
    fn fused_linear_sparse_partial_mass_grad_matches_finite_difference() {
        let rows = 2;
        let hidden_dim = 3;
        let vocab = 5;
        let topk = 3;
        let temperature = 1.3;
        let hidden = deterministic_logits(rows * hidden_dim, 71);
        let weight = deterministic_logits(vocab * hidden_dim, 73);
        let teacher_topk_log_probs = vec![
            0.45_f32.ln(),
            0.20_f32.ln(),
            0.10_f32.ln(),
            0.30_f32.ln(),
            0.25_f32.ln(),
            0.15_f32.ln(),
        ];
        let teacher_topk_indices = vec![0, 2, 4, 1, 3, 4];
        for row in 0..rows {
            let mass = teacher_topk_log_probs[row * topk..(row + 1) * topk]
                .iter()
                .map(|value| value.exp())
                .sum::<f32>();
            let missing = 1.0 - mass;
            assert!(
                (0.1..=0.4).contains(&missing),
                "test must exercise partial teacher mass, got m={missing:.3}"
            );
        }

        let (_loss, hidden_grad, weight_grad) = sparse_linear_kl_loss_and_grads(
            &hidden,
            &weight,
            &teacher_topk_log_probs,
            &teacher_topk_indices,
            rows,
            hidden_dim,
            vocab,
            topk,
            temperature,
        );
        let eps = 1.0e-2_f32;

        for index in 0..hidden.len() {
            let mut plus = hidden.clone();
            let mut minus = hidden.clone();
            plus[index] += eps;
            minus[index] -= eps;
            let finite_diff = (sparse_linear_kl_loss_value(
                &plus,
                &weight,
                &teacher_topk_log_probs,
                &teacher_topk_indices,
                rows,
                hidden_dim,
                vocab,
                topk,
                temperature,
            ) - sparse_linear_kl_loss_value(
                &minus,
                &weight,
                &teacher_topk_log_probs,
                &teacher_topk_indices,
                rows,
                hidden_dim,
                vocab,
                topk,
                temperature,
            )) / (2.0 * eps);
            assert_close(
                hidden_grad[index],
                finite_diff,
                2.0e-3,
                &format!("partial-mass hidden grad[{index}]"),
            );
        }

        for index in 0..weight.len() {
            let mut plus = weight.clone();
            let mut minus = weight.clone();
            plus[index] += eps;
            minus[index] -= eps;
            let finite_diff = (sparse_linear_kl_loss_value(
                &hidden,
                &plus,
                &teacher_topk_log_probs,
                &teacher_topk_indices,
                rows,
                hidden_dim,
                vocab,
                topk,
                temperature,
            ) - sparse_linear_kl_loss_value(
                &hidden,
                &minus,
                &teacher_topk_log_probs,
                &teacher_topk_indices,
                rows,
                hidden_dim,
                vocab,
                topk,
                temperature,
            )) / (2.0 * eps);
            assert_close(
                weight_grad[index],
                finite_diff,
                2.0e-3,
                &format!("partial-mass weight grad[{index}]"),
            );
        }
    }

    #[test]
    fn fused_linear_sparse_rejects_duplicate_topk_indices_per_row() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let hidden =
            store.alloc(Tensor::new(vec![0.1, 0.2], vec![1, 2], true).expect("hidden tensor"));
        let weight = store.alloc(
            Tensor::new(vec![0.3, -0.2, 0.1, 0.4, -0.5, 0.6], vec![3, 2], true)
                .expect("weight tensor"),
        );
        let teacher = store.alloc(
            Tensor::new(
                vec![0.5_f32.ln(), 0.3_f32.ln(), 0.1_f32.ln()],
                vec![1, 3],
                false,
            )
            .expect("teacher top-k tensor"),
        );
        let err = fused_linear_distill_loss_sparse(
            hidden,
            weight,
            teacher,
            &[0, 1, 1],
            0,
            1,
            1,
            1.0,
            KlDirection::Forward,
            &mut store,
            &mut tape,
        )
        .expect_err("duplicate per-row top-k index must be rejected");

        assert!(format!("{err}").contains("unique per row"));
    }

    #[test]
    fn fused_linear_sparse_rejects_nonfinite_topk_log_probs() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let hidden =
            store.alloc(Tensor::new(vec![0.1, 0.2], vec![1, 2], true).expect("hidden tensor"));
        let weight = store.alloc(
            Tensor::new(vec![0.3, -0.2, 0.1, 0.4, -0.5, 0.6], vec![3, 2], true)
                .expect("weight tensor"),
        );
        let teacher = store.alloc(
            Tensor::new(vec![0.5_f32.ln(), f32::NAN], vec![1, 2], false)
                .expect("teacher top-k tensor"),
        );
        let err = fused_linear_distill_loss_sparse(
            hidden,
            weight,
            teacher,
            &[0, 1],
            0,
            1,
            1,
            1.0,
            KlDirection::Forward,
            &mut store,
            &mut tape,
        )
        .expect_err("nonfinite top-k log-prob must be rejected");

        assert!(format!("{err}").contains("log-probs must be finite"));
    }

    #[test]
    fn fused_linear_distill_matches_dense_kl_for_nonzero_row_start() {
        let total_rows = 17;
        let row_start = 4;
        let rows = 9;
        let hidden_dim = 7;
        let vocab = 19;
        let hidden = deterministic_logits(total_rows * hidden_dim, 41);
        let hidden_window =
            hidden[row_start * hidden_dim..(row_start + rows) * hidden_dim].to_vec();
        let weight = deterministic_logits(vocab * hidden_dim, 43);
        let teacher_logits = deterministic_logits(rows * vocab, 47);

        for direction in [KlDirection::Forward, KlDirection::Reverse] {
            let (dense_loss, dense_hidden_grad, dense_weight_grad) = dense_linear_kl_loss_and_grads(
                &hidden_window,
                &weight,
                &teacher_logits,
                rows,
                hidden_dim,
                vocab,
                1.3,
                direction,
            );

            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let hidden_id = store.alloc(
                Tensor::new(hidden.clone(), vec![total_rows, hidden_dim], true).expect("hidden"),
            );
            let weight_id = store
                .alloc(Tensor::new(weight.clone(), vec![vocab, hidden_dim], true).expect("weight"));
            let teacher_id = store.alloc(
                Tensor::new(teacher_logits.clone(), vec![rows, vocab], false)
                    .expect("teacher logits"),
            );
            let fused_loss = fused_linear_distill_loss(
                hidden_id, weight_id, teacher_id, row_start, rows, 4, 1.3, direction, &mut store,
                &mut tape,
            )
            .expect("fused linear distill");
            let fused_loss_value = store.to_host(fused_loss).expect("loss host")[0];
            tape.backward(fused_loss, &mut store)
                .expect("fused backward");
            let hidden_grad_id = store
                .get(hidden_id)
                .and_then(|tensor| tensor.grad)
                .expect("hidden grad");
            let weight_grad_id = store
                .get(weight_id)
                .and_then(|tensor| tensor.grad)
                .expect("weight grad");
            let fused_hidden_grad = store.to_host(hidden_grad_id).expect("hidden grad host");
            let fused_weight_grad = store.to_host(weight_grad_id).expect("weight grad host");

            assert_close(
                fused_loss_value,
                dense_loss,
                2.0e-5,
                &format!("nonzero row_start loss {direction:?}"),
            );
            assert_slice_close(
                &fused_hidden_grad[row_start * hidden_dim..(row_start + rows) * hidden_dim],
                &dense_hidden_grad,
                2.0e-5,
                &format!("nonzero row_start hidden grad {direction:?}"),
            );
            assert!(
                fused_hidden_grad[..row_start * hidden_dim]
                    .iter()
                    .chain(fused_hidden_grad[(row_start + rows) * hidden_dim..].iter())
                    .all(|value| value.abs() <= 1.0e-8),
                "hidden rows outside the fused window must keep zero gradient for {direction:?}"
            );
            assert_slice_close(
                &fused_weight_grad,
                &dense_weight_grad,
                2.0e-5,
                &format!("nonzero row_start weight grad {direction:?}"),
            );
        }
    }

    #[test]
    fn fused_linear_distill_respects_requires_grad_combinations() {
        let rows = 3;
        let hidden_dim = 2;
        let vocab = 5;
        let hidden_values = deterministic_logits(rows * hidden_dim, 111);
        let weight_values = deterministic_logits(vocab * hidden_dim, 113);
        let teacher_logits = deterministic_logits(rows * vocab, 117);

        for (hidden_requires_grad, weight_requires_grad) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let hidden = store.alloc(
                Tensor::new(
                    hidden_values.clone(),
                    vec![rows, hidden_dim],
                    hidden_requires_grad,
                )
                .expect("hidden"),
            );
            let weight = store.alloc(
                Tensor::new(
                    weight_values.clone(),
                    vec![vocab, hidden_dim],
                    weight_requires_grad,
                )
                .expect("weight"),
            );
            let teacher = store.alloc(
                Tensor::new(teacher_logits.clone(), vec![rows, vocab], false)
                    .expect("teacher logits"),
            );
            let tape_len = tape.entries.len();
            let loss = fused_linear_distill_loss(
                hidden,
                weight,
                teacher,
                0,
                rows,
                2,
                1.0,
                KlDirection::Forward,
                &mut store,
                &mut tape,
            )
            .expect("fused linear distill");
            let requires_grad = hidden_requires_grad || weight_requires_grad;
            assert_eq!(
                store.get(loss).expect("loss tensor").requires_grad,
                requires_grad
            );
            assert_eq!(
                tape.entries.len(),
                tape_len + usize::from(requires_grad),
                "tape record mismatch for hidden={hidden_requires_grad} weight={weight_requires_grad}"
            );
            if requires_grad {
                tape.backward(loss, &mut store).expect("backward");
            }
            assert_eq!(
                store.get(hidden).and_then(|tensor| tensor.grad).is_some(),
                hidden_requires_grad
            );
            assert_eq!(
                store.get(weight).and_then(|tensor| tensor.grad).is_some(),
                weight_requires_grad
            );
        }
    }

    #[test]
    fn fused_linear_distill_uses_less_live_host_memory_than_dense_logits_kl() {
        let rows = 32;
        let hidden_dim = 4;
        let vocab = 128;
        let hidden = deterministic_logits(rows * hidden_dim, 3);
        let weight = deterministic_logits(vocab * hidden_dim, 13);
        let teacher_logits = deterministic_logits(rows * vocab, 23);
        let student_logits = linear_logits(&hidden, &weight, rows, hidden_dim, vocab);

        let dense_extra = {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let _hidden = store
                .alloc(Tensor::new(hidden.clone(), vec![rows, hidden_dim], true).expect("hidden"));
            let _weight = store
                .alloc(Tensor::new(weight.clone(), vec![vocab, hidden_dim], true).expect("weight"));
            let teacher = store.alloc(
                Tensor::new(teacher_logits.clone(), vec![rows, vocab], false)
                    .expect("teacher logits"),
            );
            let base_bytes = store.live_host_bytes();
            let student = store.alloc(
                Tensor::new(student_logits, vec![rows, vocab], true).expect("student logits"),
            );
            let _loss = kl_distill_loss(
                student,
                teacher,
                rows,
                1.0,
                KlDirection::Forward,
                &mut store,
                &mut tape,
            )
            .expect("dense KL");
            store.live_host_bytes() - base_bytes
        };

        let fused_extra = {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let hidden =
                store.alloc(Tensor::new(hidden, vec![rows, hidden_dim], true).expect("hidden"));
            let weight =
                store.alloc(Tensor::new(weight, vec![vocab, hidden_dim], true).expect("weight"));
            let teacher = store.alloc(
                Tensor::new(teacher_logits, vec![rows, vocab], false).expect("teacher logits"),
            );
            let base_bytes = store.live_host_bytes();
            let _loss = fused_linear_distill_loss(
                hidden,
                weight,
                teacher,
                0,
                rows,
                5,
                1.0,
                KlDirection::Forward,
                &mut store,
                &mut tape,
            )
            .expect("fused linear distill");
            store.live_host_bytes() - base_bytes
        };

        let fused_upper_bound =
            (rows * hidden_dim + vocab * hidden_dim + 1) * std::mem::size_of::<f32>();
        assert!(
            fused_extra <= fused_upper_bound,
            "fused extra host bytes ({fused_extra}) must stay within grad_hidden+grad_weight+loss bound ({fused_upper_bound}); dense extra was {dense_extra}"
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
