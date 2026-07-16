use std::collections::HashSet;

use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::Device,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedLinearDistillDirection {
    Forward,
    Reverse,
}

pub fn generalized_jsd_loss(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    temperature: f32,
    beta: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let shape = validate_logits_inputs(student_logits, teacher_logits, num_positions, store)?;
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(AutogradError::TapeInvariant(
            "generalized_jsd_loss: temperature must be finite and > 0.0",
        ));
    }
    if !(beta > 0.0 && beta < 1.0 && beta.is_finite()) {
        return Err(AutogradError::TapeInvariant(
            "generalized_jsd_loss: beta must be finite and strictly between 0.0 and 1.0",
        ));
    }

    // ponytail: beta-JSD is opt-in host-eager; add device kernels only if
    // recipe A/B makes --kl-beta worth default-path performance work.
    let student_data = store.to_host(student_logits)?;
    let teacher_data = store.to_host(teacher_logits)?;
    let inv_temperature = 1.0 / temperature;
    let loss_scale = temperature * temperature / num_positions as f32;
    let grad_scale = temperature / num_positions as f32;
    let student_mix = 1.0 - beta;
    let log_student_mix = student_mix.ln();
    let log_teacher_mix = beta.ln();
    let mut loss_sum = 0.0_f32;
    let mut grad_student = vec![0.0_f32; student_data.len()];

    for row in 0..num_positions {
        let base = row * shape.vocab;
        let mut student_scaled = vec![0.0_f32; shape.vocab];
        let mut teacher_scaled = vec![0.0_f32; shape.vocab];
        for vocab_index in 0..shape.vocab {
            student_scaled[vocab_index] = student_data[base + vocab_index] * inv_temperature;
            teacher_scaled[vocab_index] = teacher_data[base + vocab_index] * inv_temperature;
        }

        let student_log_probs = log_softmax_row(&student_scaled);
        let teacher_log_probs = log_softmax_row(&teacher_scaled);
        let mut mixture_log_probs = vec![0.0_f32; shape.vocab];
        let mut student_kl = 0.0_f32;
        let mut teacher_kl = 0.0_f32;
        for vocab_index in 0..shape.vocab {
            mixture_log_probs[vocab_index] = logaddexp(
                student_log_probs[vocab_index] + log_student_mix,
                teacher_log_probs[vocab_index] + log_teacher_mix,
            );
            let student_prob = student_log_probs[vocab_index].exp();
            let teacher_prob = teacher_log_probs[vocab_index].exp();
            student_kl +=
                student_prob * (student_log_probs[vocab_index] - mixture_log_probs[vocab_index]);
            teacher_kl +=
                teacher_prob * (teacher_log_probs[vocab_index] - mixture_log_probs[vocab_index]);
        }
        loss_sum += beta * teacher_kl + student_mix * student_kl;
        for vocab_index in 0..shape.vocab {
            let student_prob = student_log_probs[vocab_index].exp();
            grad_student[base + vocab_index] = grad_scale
                * student_mix
                * student_prob
                * (student_log_probs[vocab_index] - mixture_log_probs[vocab_index] - student_kl);
        }
    }

    let loss_id = store.alloc(Tensor::new(vec![loss_sum * loss_scale], Vec::new(), true)?);
    let grad_id = store.alloc(Tensor::new(grad_student, shape.student_shape, false)?);
    tape.record(TapeEntry {
        op: BackwardOp::GeneralizedJsd,
        output_id: loss_id,
        input_ids: smallvec![student_logits],
        saved: SavedContext::GeneralizedJsdCtx {
            grad_student: grad_id,
        },
    });
    Ok(loss_id)
}

pub(crate) fn generalized_jsd_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::GeneralizedJsdCtx { grad_student } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "generalized_jsd backward missing saved context",
        ));
    };
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    if !upstream_shape.is_empty() {
        return Err(AutogradError::ShapeMismatch {
            expected: Vec::new(),
            got: upstream_shape,
        });
    }
    let upstream = store.to_host(output_grad_id)?[0];
    Ok(smallvec![(
        entry.input_ids[0],
        scale_saved_grad(grad_student, upstream, store)?
    )])
}

#[allow(clippy::too_many_arguments)]
pub fn fused_linear_distill_loss(
    hidden: TensorId,
    weight: TensorId,
    teacher_logits: TensorId,
    row_start: usize,
    num_rows: usize,
    chunk_rows: usize,
    temperature: f32,
    direction: FusedLinearDistillDirection,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Not bit-identical to matmul+KL; tests gate ~2e-5 equivalence. Default OPD flip still needs H20 needle + same-binary A/B + wins.
    let shape = validate_inputs(hidden, weight, teacher_logits, row_start, num_rows, store)?;
    if chunk_rows == 0 {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss: chunk_rows must be > 0",
        ));
    }
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss: temperature must be finite and > 0.0",
        ));
    }

    let need_hidden_grad = store.tensor(hidden)?.requires_grad;
    let need_weight_grad = store.tensor(weight)?.requires_grad;
    let requires_grad = need_hidden_grad || need_weight_grad;
    let hidden_data = store.to_host(hidden)?;
    let weight_data = store.to_host(weight)?;
    let teacher_data = store.to_host(teacher_logits)?;

    let mut grad_hidden = need_hidden_grad.then(|| vec![0.0_f32; hidden_data.len()]);
    let mut grad_weight = need_weight_grad.then(|| vec![0.0_f32; weight_data.len()]);
    let inv_temperature = 1.0 / temperature;
    let grad_scale = temperature / num_rows as f32;
    let loss_scale = temperature * temperature / num_rows as f32;
    let mut loss_sum = 0.0_f32;

    for chunk_start in (0..num_rows).step_by(chunk_rows) {
        let chunk_end = chunk_start.saturating_add(chunk_rows).min(num_rows);
        for local_row in chunk_start..chunk_end {
            let hidden_row = row_start + local_row;
            let hidden_base = hidden_row * shape.hidden_dim;
            let teacher_base = local_row * shape.vocab;
            let hidden_slice = &hidden_data[hidden_base..hidden_base + shape.hidden_dim];

            let mut student_scaled = vec![0.0_f32; shape.vocab];
            let mut teacher_scaled = vec![0.0_f32; shape.vocab];
            for vocab_index in 0..shape.vocab {
                let weight_base = vocab_index * shape.hidden_dim;
                let mut dot = 0.0_f32;
                for hidden_index in 0..shape.hidden_dim {
                    dot += hidden_slice[hidden_index] * weight_data[weight_base + hidden_index];
                }
                student_scaled[vocab_index] = dot * inv_temperature;
                teacher_scaled[vocab_index] =
                    teacher_data[teacher_base + vocab_index] * inv_temperature;
            }

            let student_log_probs = log_softmax_row(&student_scaled);
            let teacher_log_probs = log_softmax_row(&teacher_scaled);
            let mut dlogits = vec![0.0_f32; shape.vocab];

            match direction {
                FusedLinearDistillDirection::Forward => {
                    let mut row_ce = 0.0_f32;
                    for vocab_index in 0..shape.vocab {
                        let teacher_prob = teacher_log_probs[vocab_index].exp();
                        row_ce -= teacher_prob * student_log_probs[vocab_index];
                        dlogits[vocab_index] =
                            grad_scale * (student_log_probs[vocab_index].exp() - teacher_prob);
                    }
                    loss_sum += row_ce;
                }
                FusedLinearDistillDirection::Reverse => {
                    let mut row_kl = 0.0_f32;
                    for vocab_index in 0..shape.vocab {
                        let student_prob = student_log_probs[vocab_index].exp();
                        row_kl += student_prob
                            * (student_log_probs[vocab_index] - teacher_log_probs[vocab_index]);
                    }
                    for vocab_index in 0..shape.vocab {
                        let student_prob = student_log_probs[vocab_index].exp();
                        dlogits[vocab_index] = grad_scale
                            * student_prob
                            * (student_log_probs[vocab_index]
                                - teacher_log_probs[vocab_index]
                                - row_kl);
                    }
                    loss_sum += row_kl;
                }
            }

            if let Some(grad_hidden) = grad_hidden.as_mut() {
                for hidden_index in 0..shape.hidden_dim {
                    let mut grad = 0.0_f32;
                    for (vocab_index, &dlogit) in dlogits.iter().enumerate() {
                        grad += dlogit * weight_data[vocab_index * shape.hidden_dim + hidden_index];
                    }
                    grad_hidden[hidden_base + hidden_index] += grad;
                }
            }
            if let Some(grad_weight) = grad_weight.as_mut() {
                for (vocab_index, &grad) in dlogits.iter().enumerate() {
                    let weight_base = vocab_index * shape.hidden_dim;
                    for hidden_index in 0..shape.hidden_dim {
                        grad_weight[weight_base + hidden_index] +=
                            grad * hidden_slice[hidden_index];
                    }
                }
            }
        }
    }

    let loss_id = store.alloc(Tensor::new(
        vec![loss_sum * loss_scale],
        Vec::new(),
        requires_grad,
    )?);
    if requires_grad {
        let grad_hidden_id = grad_hidden
            .map(|data| Ok(store.alloc(Tensor::new(data, shape.hidden_shape.clone(), false)?)))
            .transpose()?;
        let grad_weight_id = grad_weight
            .map(|data| Ok(store.alloc(Tensor::new(data, shape.weight_shape.clone(), false)?)))
            .transpose()?;
        tape.record(TapeEntry {
            op: BackwardOp::FusedLinearDistill,
            output_id: loss_id,
            input_ids: smallvec![hidden, weight],
            saved: SavedContext::FusedLinearDistillCtx {
                grad_hidden: grad_hidden_id,
                grad_weight: grad_weight_id,
            },
        });
    }

    Ok(loss_id)
}

#[allow(clippy::too_many_arguments)]
pub fn fused_linear_distill_loss_sparse(
    hidden: TensorId,
    weight: TensorId,
    teacher_topk_log_probs: TensorId,
    teacher_topk_indices: &[i32],
    row_start: usize,
    num_rows: usize,
    chunk_rows: usize,
    temperature: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let (shape, teacher_topk_log_prob_data) = validate_sparse_inputs(
        hidden,
        weight,
        teacher_topk_log_probs,
        teacher_topk_indices,
        row_start,
        num_rows,
        store,
    )?;
    if chunk_rows == 0 {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: chunk_rows must be > 0",
        ));
    }
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: temperature must be finite and > 0.0",
        ));
    }

    let need_hidden_grad = store.tensor(hidden)?.requires_grad;
    let need_weight_grad = store.tensor(weight)?.requires_grad;
    let requires_grad = need_hidden_grad || need_weight_grad;
    let hidden_data = store.to_host(hidden)?;
    let weight_data = store.to_host(weight)?;

    let mut grad_hidden = need_hidden_grad.then(|| vec![0.0_f32; hidden_data.len()]);
    let mut grad_weight = need_weight_grad.then(|| vec![0.0_f32; weight_data.len()]);
    let inv_temperature = 1.0 / temperature;
    let grad_scale = temperature / num_rows as f32;
    let loss_scale = temperature * temperature / num_rows as f32;
    let mut loss_sum = 0.0_f32;

    for chunk_start in (0..num_rows).step_by(chunk_rows) {
        let chunk_end = chunk_start.saturating_add(chunk_rows).min(num_rows);
        for local_row in chunk_start..chunk_end {
            let hidden_row = row_start + local_row;
            let hidden_base = hidden_row * shape.hidden_dim;
            let topk_base = local_row * shape.topk;
            let hidden_slice = &hidden_data[hidden_base..hidden_base + shape.hidden_dim];

            let mut student_scaled = vec![0.0_f32; shape.vocab];
            for (vocab_index, student_value) in student_scaled.iter_mut().enumerate() {
                let weight_base = vocab_index * shape.hidden_dim;
                let mut dot = 0.0_f32;
                for hidden_index in 0..shape.hidden_dim {
                    dot += hidden_slice[hidden_index] * weight_data[weight_base + hidden_index];
                }
                *student_value = dot * inv_temperature;
            }

            let student_log_probs = log_softmax_row(&student_scaled);
            let mut row_ce = 0.0_f32;
            let mut row_teacher_mass = 0.0_f32;
            for topk_offset in 0..shape.topk {
                let slot = topk_base + topk_offset;
                let teacher_log_prob = teacher_topk_log_prob_data[slot];
                let vocab_index = teacher_topk_indices[slot] as usize;
                let teacher_prob = teacher_log_prob.exp();
                row_ce -= teacher_prob * student_log_probs[vocab_index];
                row_teacher_mass += teacher_prob;
            }
            let mut dlogits = vec![0.0_f32; shape.vocab];
            for vocab_index in 0..shape.vocab {
                dlogits[vocab_index] =
                    grad_scale * row_teacher_mass * student_log_probs[vocab_index].exp();
            }
            for topk_offset in 0..shape.topk {
                let slot = topk_base + topk_offset;
                let vocab_index = teacher_topk_indices[slot] as usize;
                let teacher_prob = teacher_topk_log_prob_data[slot].exp();
                dlogits[vocab_index] -= grad_scale * teacher_prob;
            }
            loss_sum += row_ce;

            if let Some(grad_hidden) = grad_hidden.as_mut() {
                for hidden_index in 0..shape.hidden_dim {
                    let mut grad = 0.0_f32;
                    for (vocab_index, &dlogit) in dlogits.iter().enumerate() {
                        grad += dlogit * weight_data[vocab_index * shape.hidden_dim + hidden_index];
                    }
                    grad_hidden[hidden_base + hidden_index] += grad;
                }
            }
            if let Some(grad_weight) = grad_weight.as_mut() {
                for (vocab_index, &grad) in dlogits.iter().enumerate() {
                    let weight_base = vocab_index * shape.hidden_dim;
                    for hidden_index in 0..shape.hidden_dim {
                        grad_weight[weight_base + hidden_index] +=
                            grad * hidden_slice[hidden_index];
                    }
                }
            }
        }
    }

    let loss_id = store.alloc(Tensor::new(
        vec![loss_sum * loss_scale],
        Vec::new(),
        requires_grad,
    )?);
    if requires_grad {
        let grad_hidden_id = grad_hidden
            .map(|data| Ok(store.alloc(Tensor::new(data, shape.hidden_shape.clone(), false)?)))
            .transpose()?;
        let grad_weight_id = grad_weight
            .map(|data| Ok(store.alloc(Tensor::new(data, shape.weight_shape.clone(), false)?)))
            .transpose()?;
        tape.record(TapeEntry {
            op: BackwardOp::FusedLinearDistill,
            output_id: loss_id,
            input_ids: smallvec![hidden, weight],
            saved: SavedContext::FusedLinearDistillCtx {
                grad_hidden: grad_hidden_id,
                grad_weight: grad_weight_id,
            },
        });
    }

    Ok(loss_id)
}

/// Forward-once chunked fused cross-entropy on HARD target tokens at a sparse
/// set of hidden positions — the masked-CE writeback core (agent-OPD).
///
/// Mirrors [`fused_linear_distill_loss_sparse`] but replaces the per-row
/// teacher-distribution KL with single-target CE, and takes the loss positions
/// as an explicit `position_indices` list into `hidden`'s rows (the masked /
/// LLM-generated predicting positions are a non-contiguous subset, so a
/// `row_start..num_rows` range cannot express them). Per chunk it materializes
/// only `[chunk_len, vocab]` logits in f32, never the full `[seq, vocab]` tile,
/// and frees each chunk before the next — peak transient ≈ `chunk * vocab * 4`
/// bytes instead of `seq * vocab * 4`.
///
/// `hidden`: `[1, seq, hidden_dim]` (or any shape whose last dim is
/// `hidden_dim`); `weight`: `[vocab, hidden_dim]` (the lm_head, `logits =
/// hidden @ weightᵀ`). `position_indices[i]` selects hidden row `p_i`;
/// `target_tokens[i]` is its hard target token id. Loss = mean over the
/// supplied positions of `-log_softmax(logits)[target]`; gradient
/// `d_logits = (softmax - onehot(target)) / N` (N = #positions). `d_hidden`
/// flows always (if `hidden.requires_grad`); `d_weight` only when the lm_head is
/// trainable — in `--share-frozen-base` the head is frozen/tied, so only
/// `d_hidden` is produced. f32 numerics throughout.
#[allow(clippy::too_many_arguments)]
pub fn fused_linear_ce_loss_indexed(
    hidden: TensorId,
    weight: TensorId,
    position_indices: &[i32],
    target_tokens: &[i32],
    chunk_rows: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let shape = validate_ce_indexed_inputs(hidden, weight, position_indices, target_tokens, store)?;
    if chunk_rows == 0 {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_ce_loss_indexed: chunk_rows must be > 0",
        ));
    }

    // GPU backends (CUDA/Metal) run the chunked CE through the existing
    // device-resident ops (embedding row-gather + matmul_bt + log_softmax +
    // gather + sum); the host scalar loop below stays the CPU reference path.
    if !matches!(store.backend().device(), Device::Cpu) {
        return fused_linear_ce_loss_indexed_device(
            hidden,
            weight,
            position_indices,
            target_tokens,
            chunk_rows,
            &shape,
            store,
            tape,
        );
    }

    let need_hidden_grad = store.tensor(hidden)?.requires_grad;
    let need_weight_grad = store.tensor(weight)?.requires_grad;
    let requires_grad = need_hidden_grad || need_weight_grad;
    let hidden_data = store.to_host(hidden)?;
    let weight_data = store.to_host(weight)?;

    let mut grad_hidden = need_hidden_grad.then(|| vec![0.0_f32; hidden_data.len()]);
    let mut grad_weight = need_weight_grad.then(|| vec![0.0_f32; weight_data.len()]);
    let num_targets = position_indices.len();
    // CE with temperature=1: loss = mean over positions of -log p[target];
    // d_logits = (softmax - onehot) / N. Both share the 1/N mean scale.
    let inv_n = 1.0_f32 / num_targets as f32;
    let mut loss_sum = 0.0_f32;

    for chunk_start in (0..num_targets).step_by(chunk_rows) {
        let chunk_end = chunk_start.saturating_add(chunk_rows).min(num_targets);
        for local in chunk_start..chunk_end {
            let hidden_row = position_indices[local] as usize;
            let target = target_tokens[local] as usize;
            let hidden_base = hidden_row * shape.hidden_dim;
            let hidden_slice = &hidden_data[hidden_base..hidden_base + shape.hidden_dim];

            let mut logits = vec![0.0_f32; shape.vocab];
            for (vocab_index, logit) in logits.iter_mut().enumerate() {
                let weight_base = vocab_index * shape.hidden_dim;
                let mut dot = 0.0_f32;
                for hidden_index in 0..shape.hidden_dim {
                    dot += hidden_slice[hidden_index] * weight_data[weight_base + hidden_index];
                }
                *logit = dot;
            }

            let log_probs = log_softmax_row(&logits);
            loss_sum -= log_probs[target];

            let mut dlogits = vec![0.0_f32; shape.vocab];
            for (vocab_index, dlogit) in dlogits.iter_mut().enumerate() {
                *dlogit = inv_n * log_probs[vocab_index].exp();
            }
            dlogits[target] -= inv_n;

            if let Some(grad_hidden) = grad_hidden.as_mut() {
                for hidden_index in 0..shape.hidden_dim {
                    let mut grad = 0.0_f32;
                    for (vocab_index, &dlogit) in dlogits.iter().enumerate() {
                        grad += dlogit * weight_data[vocab_index * shape.hidden_dim + hidden_index];
                    }
                    grad_hidden[hidden_base + hidden_index] += grad;
                }
            }
            if let Some(grad_weight) = grad_weight.as_mut() {
                for (vocab_index, &grad) in dlogits.iter().enumerate() {
                    let weight_base = vocab_index * shape.hidden_dim;
                    for hidden_index in 0..shape.hidden_dim {
                        grad_weight[weight_base + hidden_index] +=
                            grad * hidden_slice[hidden_index];
                    }
                }
            }
        }
    }

    let loss_id = store.alloc(Tensor::new(
        vec![loss_sum * inv_n],
        Vec::new(),
        requires_grad,
    )?);
    if requires_grad {
        let grad_hidden_id = grad_hidden
            .map(|data| Ok(store.alloc(Tensor::new(data, shape.hidden_shape.clone(), false)?)))
            .transpose()?;
        let grad_weight_id = grad_weight
            .map(|data| Ok(store.alloc(Tensor::new(data, shape.weight_shape.clone(), false)?)))
            .transpose()?;
        tape.record(TapeEntry {
            op: BackwardOp::FusedLinearDistill,
            output_id: loss_id,
            input_ids: smallvec![hidden, weight],
            saved: SavedContext::FusedLinearDistillCtx {
                grad_hidden: grad_hidden_id,
                grad_weight: grad_weight_id,
            },
        });
    }

    Ok(loss_id)
}

/// GPU path for [`fused_linear_ce_loss_indexed`] — same masked-CE math and same
/// per-chunk free discipline, but built from the existing device-resident
/// autograd ops instead of a host scalar loop (no `to_host` of the
/// `[vocab, hidden]` lm_head or the `[seq, hidden]` activations). Per
/// position-chunk: `embedding(hidden_rows, chunk_positions)` (row-gather) →
/// `matmul_bt(rows, weight)` → `log_softmax` → `gather(targets)` → `sum`, scaled
/// by `-1/N`; a per-chunk sub-tape `backward_collect` yields the chunk's
/// `grad_hidden` (and `grad_weight` if the head is trainable), accumulated into a
/// running full-shape device grad and the chunk's `[chunk, vocab]` logits freed
/// before the next chunk. The accumulated grads are saved on the `FusedLinearDistill`
/// tape entry, so outer backward scales them by the upstream exactly like the host
/// path.
#[allow(clippy::too_many_arguments)]
fn fused_linear_ce_loss_indexed_device(
    hidden: TensorId,
    weight: TensorId,
    position_indices: &[i32],
    target_tokens: &[i32],
    chunk_rows: usize,
    shape: &FusedLinearDistillShape,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let need_hidden_grad = store.tensor(hidden)?.requires_grad;
    let need_weight_grad = store.tensor(weight)?.requires_grad;
    let requires_grad = need_hidden_grad || need_weight_grad;
    let num_targets = position_indices.len();
    let inv_n = 1.0_f32 / num_targets as f32;

    // 2D view [total_rows, hidden_dim] so `embedding` can row-gather chunk
    // positions. reshape is a metadata-only lazy op; grad flows back to `hidden`.
    let total_rows = store.tensor(hidden)?.size / shape.hidden_dim;
    let mut view_tape = Tape::new();
    view_tape.set_enabled(false);
    let hidden_2d = crate::ops::reshape(
        hidden,
        &[total_rows, shape.hidden_dim],
        store,
        &mut view_tape,
    )?;

    let mut loss_sum = 0.0_f32;
    let mut grad_hidden_2d_accum: Option<TensorId> = None;
    let mut grad_weight_accum: Option<TensorId> = None;

    for chunk_start in (0..num_targets).step_by(chunk_rows) {
        let live_before = store.live_ids().into_iter().collect::<HashSet<_>>();
        let chunk_end = chunk_start.saturating_add(chunk_rows).min(num_targets);
        let chunk_positions: Vec<usize> = position_indices[chunk_start..chunk_end]
            .iter()
            .map(|&p| p as usize)
            .collect();
        let chunk_targets: Vec<usize> = target_tokens[chunk_start..chunk_end]
            .iter()
            .map(|&t| t as usize)
            .collect();

        // Per-chunk sub-tape so each chunk's [chunk, vocab] logits/intermediates
        // are dropped after its backward, never retained across chunks.
        let mut chunk_tape = Tape::new();
        chunk_tape.set_enabled(requires_grad);

        // `embedding` row-gathers but emits a [1, chunk, hidden] (implicit batch
        // dim); reshape to rank-2 [chunk, hidden] so matmul_bt's rank-2 backward
        // applies. reshape grad flows straight through to the embedding output.
        let chunk_len = chunk_positions.len();
        let hidden_rows_3d =
            crate::ops::embedding(hidden_2d, &chunk_positions, store, &mut chunk_tape)?;
        let hidden_rows = crate::ops::reshape(
            hidden_rows_3d,
            &[chunk_len, shape.hidden_dim],
            store,
            &mut chunk_tape,
        )?;
        let logits = crate::ops::matmul_bt(hidden_rows, weight, store, &mut chunk_tape)?;
        let log_probs = crate::ops::log_softmax(logits, store, &mut chunk_tape)?;
        let gathered =
            crate::ops::gather_last_dim(log_probs, &chunk_targets, store, &mut chunk_tape)?;
        let summed = crate::ops::sum(gathered, store, &mut chunk_tape)?;
        // loss contribution = -(Σ log p[target]) / N; backward seed 1.0 then
        // gives d_logits = (softmax - onehot)/N, matching the host path.
        let chunk_loss = crate::ops::mul_scalar(summed, -inv_n, store, &mut chunk_tape)?;
        loss_sum += store.to_host(chunk_loss)?[0];

        if requires_grad {
            let grads = chunk_tape.backward_collect(chunk_loss, store)?;
            if need_hidden_grad && let Some(&g) = grads.get(&hidden_2d) {
                grad_hidden_2d_accum = Some(match grad_hidden_2d_accum.take() {
                    None => store.clone_tensor(g)?,
                    Some(acc) => {
                        let next = crate::ops::add(acc, g, store, &mut view_tape)?;
                        if store.get(acc).is_some() {
                            store.free(acc)?;
                        }
                        next
                    }
                });
            }
            if need_weight_grad && let Some(&g) = grads.get(&weight) {
                grad_weight_accum = Some(match grad_weight_accum.take() {
                    None => store.clone_tensor(g)?,
                    Some(acc) => {
                        let next = crate::ops::add(acc, g, store, &mut view_tape)?;
                        if store.get(acc).is_some() {
                            store.free(acc)?;
                        }
                        next
                    }
                });
            }
        }

        let keep = [grad_hidden_2d_accum, grad_weight_accum]
            .into_iter()
            .flatten()
            .collect::<HashSet<_>>();
        store.free_new_except(&live_before, &keep)?;
    }

    let loss_id = store.alloc(Tensor::new(vec![loss_sum], Vec::new(), requires_grad)?);
    if requires_grad {
        let grad_hidden_id = match grad_hidden_2d_accum {
            Some(g) => {
                let reshaped = crate::ops::reshape(g, &shape.hidden_shape, store, &mut view_tape)?;
                if store.get(g).is_some() {
                    store.free(g)?;
                }
                Some(reshaped)
            }
            None => None,
        };
        tape.record(TapeEntry {
            op: BackwardOp::FusedLinearDistill,
            output_id: loss_id,
            input_ids: smallvec![hidden, weight],
            saved: SavedContext::FusedLinearDistillCtx {
                grad_hidden: grad_hidden_id,
                grad_weight: grad_weight_accum,
            },
        });
    }

    Ok(loss_id)
}

/// How the (detached) importance ratio `r` gates the base weight inside
/// [`fused_linear_pg_loss_indexed`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightForm {
    /// Binary DIS gate: keep iff `(1−lo) < r < (1+hi)`, else drop the token.
    HardGate { lo: f32, hi: f32 },
    /// CISPO-family: the clamped ratio IS the gate factor, `clamp(r, 1−lo, 1+hi)`.
    SoftClamp { lo: f32, hi: f32 },
    /// Caller already folded any ratio factor into the weight (e.g. GSPO's
    /// sequence-level clamp); the op applies the weight as-is.
    Precomputed,
}

/// Off-policy diagnostics accumulated by [`fused_linear_pg_loss_indexed`]:
/// Schulman k3 KL(π_b‖πθ) `Σ(r−1−ln r)`, the gate/clamp reject count, and the
/// raw importance-ratio sum/max (off-policy blow-up telemetry).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PgStats {
    pub kl_sum: f64,
    pub clipped: usize,
    pub tokens: usize,
    pub ratio_sum: f64,
    pub ratio_max: f64,
}

impl PgStats {
    pub fn merge(&mut self, other: PgStats) {
        self.kl_sum += other.kl_sum;
        self.clipped += other.clipped;
        self.tokens += other.tokens;
        self.ratio_sum += other.ratio_sum;
        self.ratio_max = self.ratio_max.max(other.ratio_max);
    }
    pub fn kl_mean(&self) -> f64 {
        self.kl_sum / self.tokens.max(1) as f64
    }
    pub fn clip_frac(&self) -> f64 {
        self.clipped as f64 / self.tokens.max(1) as f64
    }
    pub fn ratio_mean(&self) -> f64 {
        self.ratio_sum / self.tokens.max(1) as f64
    }
    fn observe(&mut self, logp: f32, rollout_logp: f32, clipped: bool) {
        let d = f64::from(logp - rollout_logp);
        let r = d.exp();
        self.kl_sum += r - 1.0 - d;
        self.clipped += usize::from(clipped);
        self.tokens += 1;
        self.ratio_sum += r;
        self.ratio_max = self.ratio_max.max(r);
    }
}

/// `w = base·gate(r) − kl_coef·(r−1)`, all DETACHED (k3 KL gradient w.r.t. logp
/// is `r−1`; the sign makes descending the loss decrease KL(π_b‖πθ)). Single
/// source of truth for the host and device paths. Returns `(w, ratio_clipped)`.
pub fn pg_token_weight(form: WeightForm, base: f32, r: f32, kl_coef: f32) -> (f32, bool) {
    let (gate, clipped) = match form {
        WeightForm::HardGate { lo, hi } => {
            if (1.0 - lo) < r && r < (1.0 + hi) {
                (1.0, false)
            } else {
                (0.0, true)
            }
        }
        WeightForm::SoftClamp { lo, hi } => {
            let clamped = r.clamp(1.0 - lo, 1.0 + hi);
            (clamped, clamped != r)
        }
        WeightForm::Precomputed => (1.0, false),
    };
    let mut w = base * gate;
    // Guarded so kl_coef=0 stays byte-identical even at r=inf (0.0·inf = NaN).
    if kl_coef != 0.0 {
        w -= kl_coef * (r - 1.0);
    }
    (w, clipped)
}

/// Per-token-weighted policy-gradient twin of [`fused_linear_ce_loss_indexed`]
/// — a REINFORCE surrogate `Σ_p (-w_p · logπθ(t_p))` at a sparse set of hidden
/// positions, chunked so it never materializes `[seq, vocab]`. Structure,
/// dispatch, and grad-writeback mirror the CE op exactly; the ONLY difference is
/// the per-position scale: the CE op uses a uniform `1/N`, this op uses
/// `w_p = pg_token_weight(form, token_weight[p], r_p, kl_coef)` with
/// `r = exp(logp − rollout_logprobs[p])` (importance ratio vs the behavior
/// policy). `w` is a DETACHED weight: `loss += −w · logp`;
/// `d_logits_p = w · (softmax(logits_p) − onehot(t))` — the gradient flows
/// solely through `logp`, never through the ratio. `d_hidden` flows always (if
/// `hidden.requires_grad`); `d_weight` only when the lm_head is trainable. f32
/// numerics throughout. Also returns the k3 KL / clip diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn fused_linear_pg_loss_indexed(
    hidden: TensorId,
    weight: TensorId,
    position_indices: &[i32],
    target_tokens: &[i32],
    rollout_logprobs: &[f32],
    token_weight: &[f32],
    form: WeightForm,
    kl_coef: f32,
    chunk_rows: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<(TensorId, PgStats)> {
    let shape = validate_ce_indexed_inputs(hidden, weight, position_indices, target_tokens, store)?;
    if chunk_rows == 0 {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_pg_loss_indexed: chunk_rows must be > 0",
        ));
    }
    if rollout_logprobs.len() != position_indices.len() {
        return Err(AutogradError::InvalidIndicesLen {
            expected: position_indices.len(),
            got: rollout_logprobs.len(),
        });
    }
    if token_weight.len() != position_indices.len() {
        return Err(AutogradError::InvalidIndicesLen {
            expected: position_indices.len(),
            got: token_weight.len(),
        });
    }
    let (lo, hi) = match form {
        WeightForm::HardGate { lo, hi } | WeightForm::SoftClamp { lo, hi } => (lo, hi),
        WeightForm::Precomputed => (0.0, 0.0),
    };
    if !(token_weight.iter().all(|a| a.is_finite())
        && lo.is_finite()
        && hi.is_finite()
        && kl_coef.is_finite())
    {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_pg_loss_indexed: token_weight/clip bounds/kl_coef must be finite",
        ));
    }
    if rollout_logprobs.iter().any(|value| !value.is_finite()) {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_pg_loss_indexed: rollout_logprobs must be finite",
        ));
    }

    if !matches!(store.backend().device(), Device::Cpu) {
        return fused_linear_pg_loss_indexed_device(
            hidden,
            weight,
            position_indices,
            target_tokens,
            rollout_logprobs,
            token_weight,
            form,
            kl_coef,
            chunk_rows,
            &shape,
            store,
            tape,
        );
    }

    let need_hidden_grad = store.tensor(hidden)?.requires_grad;
    let need_weight_grad = store.tensor(weight)?.requires_grad;
    let requires_grad = need_hidden_grad || need_weight_grad;
    let hidden_data = store.to_host(hidden)?;
    let weight_data = store.to_host(weight)?;

    let mut grad_hidden = need_hidden_grad.then(|| vec![0.0_f32; hidden_data.len()]);
    let mut grad_weight = need_weight_grad.then(|| vec![0.0_f32; weight_data.len()]);
    let num_targets = position_indices.len();
    let mut loss_sum = 0.0_f32;
    let mut stats = PgStats::default();

    for chunk_start in (0..num_targets).step_by(chunk_rows) {
        let chunk_end = chunk_start.saturating_add(chunk_rows).min(num_targets);
        for local in chunk_start..chunk_end {
            let hidden_row = position_indices[local] as usize;
            let target = target_tokens[local] as usize;
            let hidden_base = hidden_row * shape.hidden_dim;
            let hidden_slice = &hidden_data[hidden_base..hidden_base + shape.hidden_dim];

            let mut logits = vec![0.0_f32; shape.vocab];
            for (vocab_index, logit) in logits.iter_mut().enumerate() {
                let weight_base = vocab_index * shape.hidden_dim;
                let mut dot = 0.0_f32;
                for hidden_index in 0..shape.hidden_dim {
                    dot += hidden_slice[hidden_index] * weight_data[weight_base + hidden_index];
                }
                *logit = dot;
            }

            let log_probs = log_softmax_row(&logits);
            let logp = log_probs[target];
            // Detached weight: ratio/gate/base are constants, no grad path.
            let r = (logp - rollout_logprobs[local]).exp();
            let (w, clipped) = pg_token_weight(form, token_weight[local], r, kl_coef);
            stats.observe(logp, rollout_logprobs[local], clipped);
            loss_sum -= w * logp;

            // d_logits = w · (softmax − onehot), the CE gradient scaled by w.
            let mut dlogits = vec![0.0_f32; shape.vocab];
            for (vocab_index, dlogit) in dlogits.iter_mut().enumerate() {
                *dlogit = w * log_probs[vocab_index].exp();
            }
            dlogits[target] -= w;

            if let Some(grad_hidden) = grad_hidden.as_mut() {
                for hidden_index in 0..shape.hidden_dim {
                    let mut grad = 0.0_f32;
                    for (vocab_index, &dlogit) in dlogits.iter().enumerate() {
                        grad += dlogit * weight_data[vocab_index * shape.hidden_dim + hidden_index];
                    }
                    grad_hidden[hidden_base + hidden_index] += grad;
                }
            }
            if let Some(grad_weight) = grad_weight.as_mut() {
                for (vocab_index, &grad) in dlogits.iter().enumerate() {
                    let weight_base = vocab_index * shape.hidden_dim;
                    for hidden_index in 0..shape.hidden_dim {
                        grad_weight[weight_base + hidden_index] +=
                            grad * hidden_slice[hidden_index];
                    }
                }
            }
        }
    }

    let loss_id = store.alloc(Tensor::new(vec![loss_sum], Vec::new(), requires_grad)?);
    if requires_grad {
        let grad_hidden_id = grad_hidden
            .map(|data| Ok(store.alloc(Tensor::new(data, shape.hidden_shape.clone(), false)?)))
            .transpose()?;
        let grad_weight_id = grad_weight
            .map(|data| Ok(store.alloc(Tensor::new(data, shape.weight_shape.clone(), false)?)))
            .transpose()?;
        tape.record(TapeEntry {
            op: BackwardOp::FusedLinearDistill,
            output_id: loss_id,
            input_ids: smallvec![hidden, weight],
            saved: SavedContext::FusedLinearDistillCtx {
                grad_hidden: grad_hidden_id,
                grad_weight: grad_weight_id,
            },
        });
    }

    Ok((loss_id, stats))
}

/// GPU path for [`fused_linear_pg_loss_indexed`] — mirrors
/// [`fused_linear_ce_loss_indexed_device`], but replaces the uniform
/// `mul_scalar(summed, -1/N)` with a per-position detached weight vector
/// `neg_w_p = -pg_token_weight(...)` applied elementwise to the gathered
/// target log-probs before the sum. The ratio is evaluated on host from a
/// per-chunk readback of the gathered log-probs (the only extra readback vs
/// the CE path); the multiply feeds `d_logits_p = w_p·(softmax − onehot)` back
/// through log_softmax, matching the host path.
#[allow(clippy::too_many_arguments)]
fn fused_linear_pg_loss_indexed_device(
    hidden: TensorId,
    weight: TensorId,
    position_indices: &[i32],
    target_tokens: &[i32],
    rollout_logprobs: &[f32],
    token_weight: &[f32],
    form: WeightForm,
    kl_coef: f32,
    chunk_rows: usize,
    shape: &FusedLinearDistillShape,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<(TensorId, PgStats)> {
    let need_hidden_grad = store.tensor(hidden)?.requires_grad;
    let need_weight_grad = store.tensor(weight)?.requires_grad;
    let requires_grad = need_hidden_grad || need_weight_grad;
    let num_targets = position_indices.len();

    let total_rows = store.tensor(hidden)?.size / shape.hidden_dim;
    let mut view_tape = Tape::new();
    view_tape.set_enabled(false);
    let hidden_2d = crate::ops::reshape(
        hidden,
        &[total_rows, shape.hidden_dim],
        store,
        &mut view_tape,
    )?;

    let mut loss_sum = 0.0_f32;
    let mut stats = PgStats::default();
    let mut grad_hidden_2d_accum: Option<TensorId> = None;
    let mut grad_weight_accum: Option<TensorId> = None;

    for chunk_start in (0..num_targets).step_by(chunk_rows) {
        let live_before = store.live_ids().into_iter().collect::<HashSet<_>>();
        let chunk_end = chunk_start.saturating_add(chunk_rows).min(num_targets);
        let chunk_positions: Vec<usize> = position_indices[chunk_start..chunk_end]
            .iter()
            .map(|&p| p as usize)
            .collect();
        let chunk_targets: Vec<usize> = target_tokens[chunk_start..chunk_end]
            .iter()
            .map(|&t| t as usize)
            .collect();

        let mut chunk_tape = Tape::new();
        chunk_tape.set_enabled(requires_grad);

        let chunk_len = chunk_positions.len();
        let hidden_rows_3d =
            crate::ops::embedding(hidden_2d, &chunk_positions, store, &mut chunk_tape)?;
        let hidden_rows = crate::ops::reshape(
            hidden_rows_3d,
            &[chunk_len, shape.hidden_dim],
            store,
            &mut chunk_tape,
        )?;
        let logits = crate::ops::matmul_bt(hidden_rows, weight, store, &mut chunk_tape)?;
        let log_probs = crate::ops::log_softmax(logits, store, &mut chunk_tape)?;
        let gathered =
            crate::ops::gather_last_dim(log_probs, &chunk_targets, store, &mut chunk_tape)?;

        // Detached per-position weight negated for the -w·logp loss; read the
        // gathered log-probs back to evaluate the ratio gate/clamp on host.
        let gathered_logp = store.to_host(gathered)?;
        let mut neg_w = Vec::with_capacity(gathered_logp.len());
        for (offset, &logp) in gathered_logp.iter().enumerate() {
            let idx = chunk_start + offset;
            let r = (logp - rollout_logprobs[idx]).exp();
            let (w, clipped) = pg_token_weight(form, token_weight[idx], r, kl_coef);
            stats.observe(logp, rollout_logprobs[idx], clipped);
            neg_w.push(-w);
        }
        let neg_w_id = store.from_slice(&neg_w, &[chunk_len])?;
        // gather_last_dim can return [chunk_len, 1]; flatten to rank-1 so the
        // per-position weight mul (and its backward) stays rank-1 — otherwise the
        // broadcast emits a rank-3 grad into matmul_bt's rank-2 backward. CE sums
        // first then scales by a scalar and sidesteps this; the DIS weight can't.
        let gathered_1d = crate::ops::reshape(gathered, &[chunk_len], store, &mut chunk_tape)?;
        let weighted = crate::ops::mul(gathered_1d, neg_w_id, store, &mut chunk_tape)?;
        let chunk_loss = crate::ops::sum(weighted, store, &mut chunk_tape)?;
        loss_sum += store.to_host(chunk_loss)?[0];

        if requires_grad {
            let grads = chunk_tape.backward_collect(chunk_loss, store)?;
            if need_hidden_grad && let Some(&g) = grads.get(&hidden_2d) {
                grad_hidden_2d_accum = Some(match grad_hidden_2d_accum.take() {
                    None => store.clone_tensor(g)?,
                    Some(acc) => {
                        let next = crate::ops::add(acc, g, store, &mut view_tape)?;
                        if store.get(acc).is_some() {
                            store.free(acc)?;
                        }
                        next
                    }
                });
            }
            if need_weight_grad && let Some(&g) = grads.get(&weight) {
                grad_weight_accum = Some(match grad_weight_accum.take() {
                    None => store.clone_tensor(g)?,
                    Some(acc) => {
                        let next = crate::ops::add(acc, g, store, &mut view_tape)?;
                        if store.get(acc).is_some() {
                            store.free(acc)?;
                        }
                        next
                    }
                });
            }
        }

        let keep = [grad_hidden_2d_accum, grad_weight_accum]
            .into_iter()
            .flatten()
            .collect::<HashSet<_>>();
        store.free_new_except(&live_before, &keep)?;
    }

    let loss_id = store.alloc(Tensor::new(vec![loss_sum], Vec::new(), requires_grad)?);
    if requires_grad {
        let grad_hidden_id = match grad_hidden_2d_accum {
            Some(g) => {
                let reshaped = crate::ops::reshape(g, &shape.hidden_shape, store, &mut view_tape)?;
                if store.get(g).is_some() {
                    store.free(g)?;
                }
                Some(reshaped)
            }
            None => None,
        };
        tape.record(TapeEntry {
            op: BackwardOp::FusedLinearDistill,
            output_id: loss_id,
            input_ids: smallvec![hidden, weight],
            saved: SavedContext::FusedLinearDistillCtx {
                grad_hidden: grad_hidden_id,
                grad_weight: grad_weight_accum,
            },
        });
    }

    Ok((loss_id, stats))
}

pub(crate) fn fused_linear_distill_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::FusedLinearDistillCtx {
        grad_hidden,
        grad_weight,
    } = entry.saved.clone()
    else {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill backward missing saved context",
        ));
    };
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    if !upstream_shape.is_empty() {
        return Err(AutogradError::ShapeMismatch {
            expected: Vec::new(),
            got: upstream_shape,
        });
    }
    let upstream = store.to_host(output_grad_id)?[0];
    let mut grads = GradPairs::new();
    if let Some(grad_id) = grad_hidden {
        let scaled = scale_saved_grad(grad_id, upstream, store)?;
        grads.push((entry.input_ids[0], scaled));
    }
    if let Some(grad_id) = grad_weight {
        let scaled = scale_saved_grad(grad_id, upstream, store)?;
        grads.push((entry.input_ids[1], scaled));
    }
    Ok(grads)
}

fn scale_saved_grad(grad_id: TensorId, scale: f32, store: &mut TensorStore) -> Result<TensorId> {
    if (scale - 1.0).abs() < f32::EPSILON {
        return store.clone_tensor(grad_id);
    }
    let device_path_ok = {
        let grad = store.tensor(grad_id)?;
        grad.dirty != crate::tensor::Dirty::Host && grad.device_handle.is_some()
    };
    if device_path_ok {
        let (shape, handle) = {
            let grad = store.tensor(grad_id)?;
            (
                grad.shape.clone(),
                grad.device_handle.as_ref().expect("checked above").clone(),
            )
        };
        let scaled = store.backend().mul_scalar(&handle, scale, &shape)?;
        return store.alloc_device_tensor(shape, scaled);
    }
    let grad = store.tensor_host(grad_id)?;
    let data = grad.data.into_iter().map(|value| value * scale).collect();
    Ok(store.alloc(Tensor::new(data, grad.shape, false)?))
}

#[derive(Debug, Clone)]
struct FusedLinearDistillShape {
    hidden_shape: Vec<usize>,
    weight_shape: Vec<usize>,
    hidden_dim: usize,
    vocab: usize,
}

#[derive(Debug, Clone)]
struct SparseFusedLinearDistillShape {
    hidden_shape: Vec<usize>,
    weight_shape: Vec<usize>,
    hidden_dim: usize,
    vocab: usize,
    topk: usize,
}

#[derive(Debug, Clone)]
struct GeneralizedJsdShape {
    student_shape: Vec<usize>,
    vocab: usize,
}

fn validate_logits_inputs(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    store: &TensorStore,
) -> Result<GeneralizedJsdShape> {
    let student = store
        .get(student_logits)
        .ok_or(AutogradError::InvalidTensorId(student_logits))?;
    let teacher = store
        .get(teacher_logits)
        .ok_or(AutogradError::InvalidTensorId(teacher_logits))?;
    if !student.requires_grad {
        return Err(AutogradError::TapeInvariant(
            "generalized_jsd_loss: student_logits must have requires_grad=true",
        ));
    }
    if teacher.requires_grad {
        return Err(AutogradError::TapeInvariant(
            "generalized_jsd_loss: teacher_logits must have requires_grad=false",
        ));
    }
    if student.shape != teacher.shape {
        return Err(AutogradError::ShapeMismatch {
            expected: student.shape.clone(),
            got: teacher.shape.clone(),
        });
    }
    let vocab = *student.shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if vocab == 0 || num_positions == 0 || student.size != num_positions * vocab {
        return Err(AutogradError::TapeInvariant(
            "generalized_jsd_loss: num_positions must match logits.numel() / vocab",
        ));
    }
    Ok(GeneralizedJsdShape {
        student_shape: student.shape.clone(),
        vocab,
    })
}

fn validate_inputs(
    hidden: TensorId,
    weight: TensorId,
    teacher_logits: TensorId,
    row_start: usize,
    num_rows: usize,
    store: &TensorStore,
) -> Result<FusedLinearDistillShape> {
    let hidden_tensor = store
        .get(hidden)
        .ok_or(AutogradError::InvalidTensorId(hidden))?;
    let weight_tensor = store
        .get(weight)
        .ok_or(AutogradError::InvalidTensorId(weight))?;
    let teacher_tensor = store
        .get(teacher_logits)
        .ok_or(AutogradError::InvalidTensorId(teacher_logits))?;
    if hidden_tensor.shape.len() < 2 {
        return Err(AutogradError::InvalidRank {
            expected: "at least 2",
            got: hidden_tensor.shape.len(),
        });
    }
    if weight_tensor.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: weight_tensor.shape.len(),
        });
    }
    if teacher_tensor.requires_grad {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss: teacher_logits must have requires_grad=false",
        ));
    }
    if num_rows == 0 {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss: num_rows must be > 0",
        ));
    }

    let hidden_dim = *hidden_tensor
        .shape
        .last()
        .ok_or(AutogradError::InvalidRank {
            expected: "at least 2",
            got: 0,
        })?;
    let vocab = weight_tensor.shape[0];
    if hidden_dim == 0 || vocab == 0 {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss: hidden_dim and vocab must be non-zero",
        ));
    }
    if weight_tensor.shape[1] != hidden_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![vocab, hidden_dim],
            got: weight_tensor.shape.clone(),
        });
    }
    if teacher_tensor.size != num_rows * vocab {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![num_rows, vocab],
            got: teacher_tensor.shape.clone(),
        });
    }
    let total_rows = hidden_tensor.size / hidden_dim;
    let row_end = row_start
        .checked_add(num_rows)
        .ok_or(AutogradError::TapeInvariant(
            "fused_linear_distill_loss: row_start + num_rows overflows",
        ))?;
    if row_end > total_rows {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss: row_start + num_rows exceeds hidden rows",
        ));
    }

    Ok(FusedLinearDistillShape {
        hidden_shape: hidden_tensor.shape.clone(),
        weight_shape: weight_tensor.shape.clone(),
        hidden_dim,
        vocab,
    })
}

fn validate_sparse_inputs(
    hidden: TensorId,
    weight: TensorId,
    teacher_topk_log_probs: TensorId,
    teacher_topk_indices: &[i32],
    row_start: usize,
    num_rows: usize,
    store: &mut TensorStore,
) -> Result<(SparseFusedLinearDistillShape, Vec<f32>)> {
    let hidden_tensor = store
        .get(hidden)
        .ok_or(AutogradError::InvalidTensorId(hidden))?;
    let weight_tensor = store
        .get(weight)
        .ok_or(AutogradError::InvalidTensorId(weight))?;
    let teacher_tensor = store
        .get(teacher_topk_log_probs)
        .ok_or(AutogradError::InvalidTensorId(teacher_topk_log_probs))?;
    if hidden_tensor.shape.len() < 2 {
        return Err(AutogradError::InvalidRank {
            expected: "at least 2",
            got: hidden_tensor.shape.len(),
        });
    }
    if weight_tensor.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: weight_tensor.shape.len(),
        });
    }
    if teacher_tensor.requires_grad {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: teacher_topk_log_probs must have requires_grad=false",
        ));
    }
    if num_rows == 0 {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: num_rows must be > 0",
        ));
    }

    let hidden_dim = *hidden_tensor
        .shape
        .last()
        .ok_or(AutogradError::InvalidRank {
            expected: "at least 2",
            got: 0,
        })?;
    let vocab = weight_tensor.shape[0];
    if hidden_dim == 0 || vocab == 0 {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: hidden_dim and vocab must be non-zero",
        ));
    }
    if weight_tensor.shape[1] != hidden_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![vocab, hidden_dim],
            got: weight_tensor.shape.clone(),
        });
    }

    let topk = *teacher_tensor
        .shape
        .last()
        .ok_or(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        })?;
    if topk == 0 || topk > vocab {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: top-k must be in 1..=vocab",
        ));
    }
    let expected_teacher_size = num_rows
        .checked_mul(topk)
        .ok_or(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: num_rows * top-k overflows",
        ))?;
    if teacher_tensor.size != expected_teacher_size {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![num_rows, topk],
            got: teacher_tensor.shape.clone(),
        });
    }
    if teacher_topk_indices.len() != expected_teacher_size {
        return Err(AutogradError::InvalidIndicesLen {
            expected: expected_teacher_size,
            got: teacher_topk_indices.len(),
        });
    }
    for row in 0..num_rows {
        let mut seen = vec![false; vocab];
        for topk_offset in 0..topk {
            let slot = row * topk + topk_offset;
            let index = teacher_topk_indices[slot];
            if index < 0 {
                return Err(AutogradError::TapeInvariant(
                    "fused_linear_distill_loss_sparse: teacher top-k indices must be non-negative",
                ));
            }
            let index = index as usize;
            if index >= vocab {
                return Err(AutogradError::IndexOutOfBounds {
                    index,
                    upper: vocab,
                });
            }
            if seen[index] {
                return Err(AutogradError::TapeInvariant(
                    "fused_linear_distill_loss_sparse: teacher top-k indices must be unique per row",
                ));
            }
            seen[index] = true;
        }
    }

    let total_rows = hidden_tensor.size / hidden_dim;
    let row_end = row_start
        .checked_add(num_rows)
        .ok_or(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: row_start + num_rows overflows",
        ))?;
    if row_end > total_rows {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: row_start + num_rows exceeds hidden rows",
        ));
    }

    let shape = SparseFusedLinearDistillShape {
        hidden_shape: hidden_tensor.shape.clone(),
        weight_shape: weight_tensor.shape.clone(),
        hidden_dim,
        vocab,
        topk,
    };
    let teacher_topk_log_prob_data = store.to_host(teacher_topk_log_probs)?;
    if teacher_topk_log_prob_data
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_distill_loss_sparse: teacher top-k log-probs must be finite",
        ));
    }

    Ok((shape, teacher_topk_log_prob_data))
}

fn validate_ce_indexed_inputs(
    hidden: TensorId,
    weight: TensorId,
    position_indices: &[i32],
    target_tokens: &[i32],
    store: &TensorStore,
) -> Result<FusedLinearDistillShape> {
    let hidden_tensor = store
        .get(hidden)
        .ok_or(AutogradError::InvalidTensorId(hidden))?;
    let weight_tensor = store
        .get(weight)
        .ok_or(AutogradError::InvalidTensorId(weight))?;
    if hidden_tensor.shape.len() < 2 {
        return Err(AutogradError::InvalidRank {
            expected: "at least 2",
            got: hidden_tensor.shape.len(),
        });
    }
    if weight_tensor.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: weight_tensor.shape.len(),
        });
    }
    if position_indices.is_empty() {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_ce_loss_indexed: position_indices must be non-empty",
        ));
    }
    if position_indices.len() != target_tokens.len() {
        return Err(AutogradError::InvalidIndicesLen {
            expected: position_indices.len(),
            got: target_tokens.len(),
        });
    }

    let hidden_dim = *hidden_tensor
        .shape
        .last()
        .ok_or(AutogradError::InvalidRank {
            expected: "at least 2",
            got: 0,
        })?;
    let vocab = weight_tensor.shape[0];
    if hidden_dim == 0 || vocab == 0 {
        return Err(AutogradError::TapeInvariant(
            "fused_linear_ce_loss_indexed: hidden_dim and vocab must be non-zero",
        ));
    }
    if weight_tensor.shape[1] != hidden_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![vocab, hidden_dim],
            got: weight_tensor.shape.clone(),
        });
    }

    let total_rows = hidden_tensor.size / hidden_dim;
    for &row in position_indices {
        if row < 0 {
            return Err(AutogradError::TapeInvariant(
                "fused_linear_ce_loss_indexed: position indices must be non-negative",
            ));
        }
        if row as usize >= total_rows {
            return Err(AutogradError::IndexOutOfBounds {
                index: row as usize,
                upper: total_rows,
            });
        }
    }
    for &token in target_tokens {
        if token < 0 {
            return Err(AutogradError::TapeInvariant(
                "fused_linear_ce_loss_indexed: target tokens must be non-negative",
            ));
        }
        if token as usize >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index: token as usize,
                upper: vocab,
            });
        }
    }

    Ok(FusedLinearDistillShape {
        hidden_shape: hidden_tensor.shape.clone(),
        weight_shape: weight_tensor.shape.clone(),
        hidden_dim,
        vocab,
    })
}

fn log_softmax_row(row: &[f32]) -> Vec<f32> {
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

fn logaddexp(a: f32, b: f32) -> f32 {
    let max_value = a.max(b);
    max_value + ((a - max_value).exp() + (b - max_value).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn deterministic_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((state >> 33) as f32) / ((1u64 << 31) as f32);
                (u - 1.0) * 0.5
            })
            .collect()
    }

    /// The model's own current log-prob at each (position, target) — the
    /// rollout-policy log-probs that make r = 1 (gate on) in the parity test.
    fn model_target_logprobs(
        hidden_data: &[f32],
        weight_data: &[f32],
        hidden_dim: usize,
        vocab: usize,
        positions: &[i32],
        targets: &[i32],
    ) -> Vec<f32> {
        positions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| {
                let base = p as usize * hidden_dim;
                let logits: Vec<f32> = (0..vocab)
                    .map(|v| {
                        (0..hidden_dim)
                            .map(|h| hidden_data[base + h] * weight_data[v * hidden_dim + h])
                            .sum::<f32>()
                    })
                    .collect();
                log_softmax_row(&logits)[t as usize]
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_ce(
        hidden_data: &[f32],
        weight_data: &[f32],
        seq: usize,
        hidden_dim: usize,
        vocab: usize,
        positions: &[i32],
        targets: &[i32],
        chunk_rows: usize,
    ) -> Result<(f32, Vec<f32>, Vec<f32>)> {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let hidden = store.from_slice(hidden_data, &[1, seq, hidden_dim])?;
        store.get_mut(hidden).expect("hidden").requires_grad = true;
        let weight = store.from_slice(weight_data, &[vocab, hidden_dim])?;
        store.get_mut(weight).expect("weight").requires_grad = true;
        let loss = fused_linear_ce_loss_indexed(
            hidden, weight, positions, targets, chunk_rows, &mut store, &mut tape,
        )?;
        let loss_value = store.to_host(loss)?[0];
        let grads: HashMap<_, _> = tape.backward(loss, &mut store)?;
        let d_hidden = store.to_host(*grads.get(&hidden).expect("d_hidden"))?;
        let d_weight = store.to_host(*grads.get(&weight).expect("d_weight"))?;
        Ok((loss_value, d_hidden, d_weight))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_pg(
        hidden_data: &[f32],
        weight_data: &[f32],
        seq: usize,
        hidden_dim: usize,
        vocab: usize,
        positions: &[i32],
        targets: &[i32],
        rollout_logprobs: &[f32],
        token_weight: &[f32],
        form: WeightForm,
        chunk_rows: usize,
    ) -> Result<(f32, Vec<f32>, Vec<f32>)> {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let hidden = store.from_slice(hidden_data, &[1, seq, hidden_dim])?;
        store.get_mut(hidden).expect("hidden").requires_grad = true;
        let weight = store.from_slice(weight_data, &[vocab, hidden_dim])?;
        store.get_mut(weight).expect("weight").requires_grad = true;
        let (loss, _stats) = fused_linear_pg_loss_indexed(
            hidden,
            weight,
            positions,
            targets,
            rollout_logprobs,
            token_weight,
            form,
            0.0,
            chunk_rows,
            &mut store,
            &mut tape,
        )?;
        let loss_value = store.to_host(loss)?[0];
        let grads: HashMap<_, _> = tape.backward(loss, &mut store)?;
        let d_hidden = store.to_host(*grads.get(&hidden).expect("d_hidden"))?;
        let d_weight = store.to_host(*grads.get(&weight).expect("d_weight"))?;
        Ok((loss_value, d_hidden, d_weight))
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Unit-weight reduction: with `advantage = 1/N` and `rollout_logprobs` set
    /// to the model's own current log-probs (so r = 1 → gate = 1 → w_p = 1/N),
    /// the weighted PG op must equal the uniform-`1/N` CE op on loss AND grads.
    #[test]
    fn pg_reduces_to_ce_at_unit_weight() -> Result<()> {
        let (seq, hidden_dim, vocab) = (8usize, 16usize, 32usize);
        let hidden_data = deterministic_vec(seq * hidden_dim, 7);
        let weight_data = deterministic_vec(vocab * hidden_dim, 11);
        let positions: Vec<i32> = vec![0, 2, 3, 5, 6];
        let targets: Vec<i32> = positions
            .iter()
            .map(|&p| ((p * 5 + 3) % 32) as i32)
            .collect();
        let n = positions.len();

        let rollout = model_target_logprobs(
            &hidden_data,
            &weight_data,
            hidden_dim,
            vocab,
            &positions,
            &targets,
        );

        let (ce_loss, ce_dh, ce_dw) = run_ce(
            &hidden_data,
            &weight_data,
            seq,
            hidden_dim,
            vocab,
            &positions,
            &targets,
            3,
        )?;
        let (pg_loss, pg_dh, pg_dw) = run_pg(
            &hidden_data,
            &weight_data,
            seq,
            hidden_dim,
            vocab,
            &positions,
            &targets,
            &rollout,
            &vec![1.0 / n as f32; n],
            WeightForm::HardGate { lo: 0.2, hi: 0.2 },
            3,
        )?;

        let loss_err = (ce_loss - pg_loss).abs();
        let dh_err = max_abs(&ce_dh, &pg_dh);
        let dw_err = max_abs(&ce_dw, &pg_dw);
        assert!(loss_err < 1e-5, "loss mismatch: ce={ce_loss} pg={pg_loss}");
        assert!(dh_err < 1e-5, "d_hidden mismatch: max_abs={dh_err}");
        assert!(dw_err < 1e-5, "d_weight mismatch: max_abs={dw_err}");
        Ok(())
    }
}
