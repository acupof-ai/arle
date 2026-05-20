//! On-Policy Distillation step function.
//!
//! Per the 2026-05-18 OPD-only pivot, ARLE's one in-tree training surface
//! is OPD: a frozen teacher `Qwen35Model` and a trainable student
//! `Qwen35Model` (optionally LoRA-adapted) share a single `TensorStore`;
//! the student samples a rollout greedily, the teacher re-scores the
//! same rollout, and the forward-KL distill loss drives backward through
//! the student parameters.
//!
//! Smoke pattern (`crates/train/tests/test_opd_step.rs`):
//! - Two `Qwen35Model::new` calls into the same store, the teacher copy
//!   pinned via `clone_frozen` so its parameter ids report
//!   `requires_grad = false`.
//! - `opd_step` is invoked per training step; on return, the tape and
//!   ephemeral tensors are pruned by the function itself.
//!
//! Production wiring (`crates/cli/src/train_cli.rs::run_opd`):
//! - Teacher loaded from a separate HF/ModelScope checkpoint via
//!   `crates/train/src/qwen35_checkpoint.rs`.
//! - Student initialised from a smaller checkpoint with LoRA adapter
//!   layered on via `Qwen35Model::new_with_lora`.

use autograd::{AutogradError, Tape, TensorId, TensorStore, optim::Optimizer};
use std::collections::HashSet;

use crate::{
    grad_clip::clip_grad_norm,
    loss::kl_distill_loss,
    qwen35::{Qwen35Error, Qwen35Model},
    trainer::{cleanup_after_backward, retained_param_and_grad_ids},
};

#[derive(Debug, thiserror::Error)]
pub enum OpdError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Qwen35(#[from] Qwen35Error),
    #[error("{0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, OpdError>;

#[derive(Debug, Clone, Copy)]
pub struct OpdStepConfig {
    /// Tokens to roll out greedily from the student starting from the prompt.
    pub rollout_len: usize,
    /// Gradient L2 norm clip threshold.
    pub grad_clip: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct OpdStepOutcome {
    pub loss: f32,
    pub rollout_len: usize,
}

/// Greedy-argmax the last-position row of a `[1, seq_len, vocab]` logits
/// tensor. The student's lm-head returns dense logits per position; OPD
/// only needs the next-token row.
fn greedy_next_token(
    logits_id: TensorId,
    seq_len: usize,
    vocab: usize,
    store: &mut TensorStore,
) -> Result<u32> {
    let host = store.to_host(logits_id)?;
    if seq_len == 0 || vocab == 0 {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout cannot sample next token with seq_len={seq_len}, vocab={vocab}. \
             Hint: pass a non-empty prompt and a Qwen35Config with vocab_size > 0; \
             see docs/projects/2026-05-18-opd-only-pivot.md."
        )));
    }
    let expected_len = seq_len.checked_mul(vocab).ok_or_else(|| {
        OpdError::InvalidInput(format!(
            "OPD rollout logits shape overflow for seq_len={seq_len}, vocab={vocab}. \
             Hint: check the prompt length and Qwen35Config::vocab_size before calling opd_step."
        ))
    })?;
    if host.len() != expected_len {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout logits length mismatch: logits_len={}, expected exactly \
             seq_len * vocab = {expected_len} ({seq_len} * {vocab}). Hint: check \
             Qwen35Model::forward output shape and Qwen35Config::vocab_size.",
            host.len()
        )));
    }
    let last_row_start = (seq_len - 1) * vocab;
    let row = &host[last_row_start..last_row_start + vocab];
    let mut best_idx: usize = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() {
            return Err(OpdError::InvalidInput(format!(
                "OPD rollout logits contain non-finite value at last-row vocab index {i}: {v}. \
                 Hint: check student forward numerics, checkpoint dtype conversion, and \
                 learning-rate stability before sampling the next token."
            )));
        }
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

fn validate_student_params(student_params: &[TensorId], store: &TensorStore) -> Result<()> {
    if student_params.is_empty() {
        return Err(OpdError::InvalidInput(
            "OPD step requires at least one student parameter id. Hint: pass \
             student.all_parameter_ids() from the trainable student model; an empty \
             parameter list makes the optimizer step a no-op."
                .to_owned(),
        ));
    }

    let mut trainable = 0usize;
    let mut seen = std::collections::HashSet::new();
    for (index, &param_id) in student_params.iter().enumerate() {
        if !seen.insert(param_id) {
            return Err(OpdError::InvalidInput(format!(
                "OPD student_params[{index}]={param_id} duplicates an earlier \
                 parameter id. Hint: pass each trainable student parameter \
                 exactly once; duplicate ids would apply grad clipping and \
                 optimizer updates more than once."
            )));
        }
        let tensor = store.get(param_id).ok_or_else(|| {
            OpdError::InvalidInput(format!(
                "OPD student_params[{index}]={param_id} does not exist in the TensorStore. \
                 Hint: pass parameter ids from the same student Qwen35Model and TensorStore \
                 used for this opd_step call."
            ))
        })?;
        if tensor.requires_grad {
            trainable += 1;
        }
    }

    if trainable == 0 {
        return Err(OpdError::InvalidInput(
            "OPD student_params contains no trainable tensors (requires_grad=true). \
             Hint: build the student with Qwen35Model::new for scratch training or \
             Qwen35Model::new_with_lora for LoRA; frozen teacher/eval parameter ids \
             make the OPD optimizer step a no-op."
                .to_owned(),
        ));
    }

    Ok(())
}

fn validate_student_param_ownership(
    student_params: &[TensorId],
    student_model_params: &[TensorId],
    teacher_params: &[TensorId],
) -> Result<()> {
    let student_model_param_set: HashSet<TensorId> = student_model_params.iter().copied().collect();
    let teacher_param_set: HashSet<TensorId> = teacher_params.iter().copied().collect();
    for (index, &param_id) in student_params.iter().enumerate() {
        if teacher_param_set.contains(&param_id) {
            return Err(OpdError::InvalidInput(format!(
                "OPD student_params[{index}]={param_id} belongs to the frozen \
                 teacher model. Hint: pass student parameter ids from \
                 student.all_parameter_ids() or the student's LoRA adapter ids; \
                 teacher weights must not be optimized."
            )));
        }
        if !student_model_param_set.contains(&param_id) {
            return Err(OpdError::InvalidInput(format!(
                "OPD student_params[{index}]={param_id} is not owned by the \
                 student Qwen35Model passed to opd_step. Hint: build \
                 student_params from that exact student's all_parameter_ids() \
                 or adapter ids, using the same TensorStore."
            )));
        }
    }

    Ok(())
}

fn validate_teacher_params(teacher_params: &[TensorId], store: &TensorStore) -> Result<()> {
    if teacher_params.is_empty() {
        return Err(OpdError::InvalidInput(
            "OPD teacher exposes no parameter ids. Hint: pass a Qwen35Model \
             built by Qwen35Model::new_for_eval or load_qwen35_from_hf_dir."
                .to_owned(),
        ));
    }

    for (index, &param_id) in teacher_params.iter().enumerate() {
        let tensor = store.get(param_id).ok_or_else(|| {
            OpdError::InvalidInput(format!(
                "OPD teacher parameter ids must belong to the same TensorStore, \
                 but teacher_params[{index}]={param_id} is missing. Hint: build \
                 teacher and student in the TensorStore passed to opd_step."
            ))
        })?;
        if tensor.requires_grad {
            return Err(OpdError::InvalidInput(format!(
                "OPD teacher parameter teacher_params[{index}]={param_id} has \
                 requires_grad=true. Hint: build the teacher with \
                 Qwen35Model::new_for_eval, load_qwen35_from_hf_dir, or \
                 student.clone_frozen; OPD must not optimize teacher weights."
            )));
        }
    }

    Ok(())
}

fn validate_loss_value(loss_value: f32) -> Result<()> {
    if loss_value.is_finite() {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD KL loss became non-finite ({loss_value}). Hint: check teacher/student logits \
         for NaN or Inf, verify both checkpoints use the same tokenizer/model family, and \
         reduce the learning rate before resuming. See \
         docs/projects/2026-05-18-opd-only-pivot.md."
    )))
}

fn validate_step_config(cfg: OpdStepConfig) -> Result<()> {
    if cfg.grad_clip.is_finite() {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD step requires cfg.grad_clip to be finite, got {}. Hint: pass \
         a positive finite threshold to enable clipping, or pass 0.0 to \
         disable clipping explicitly.",
        cfg.grad_clip
    )))
}

fn validate_rollout_shape(prompt_len: usize, rollout_len: usize, vocab: usize) -> Result<()> {
    let total_len = prompt_len.checked_add(rollout_len).ok_or_else(|| {
        OpdError::InvalidInput(format!(
            "OPD rollout length overflow: prompt_len={prompt_len}, \
             cfg.rollout_len={rollout_len}. Hint: reduce --rollout-len or \
             split the prompt before calling opd_step."
        ))
    })?;
    if total_len > u32::MAX as usize {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout total length {total_len} exceeds u32::MAX position ids. \
             Hint: reduce --rollout-len or prompt length; the current OPD \
             Qwen3.5 path uses u32 position ids."
        )));
    }
    if vocab > u32::MAX as usize {
        return Err(OpdError::InvalidInput(format!(
            "OPD student.config().vocab_size={vocab} exceeds u32::MAX token ids. \
             Hint: verify Qwen35Config::vocab_size; greedy rollout returns u32 \
             token ids."
        )));
    }
    Ok(())
}

fn map_qwen35_forward_error(stage: &str, err: Qwen35Error) -> OpdError {
    match err {
        Qwen35Error::InputLenMismatch {
            input_len,
            expected_len,
        } => OpdError::InvalidInput(format!(
            "OPD {stage} Qwen3.5 forward input length mismatch: got \
             {input_len}, expected {expected_len}. Hint: verify prompt_ids, \
             generated rollout length, and position ids were built from the \
             same rollout."
        )),
        Qwen35Error::PositionOutOfBounds { position, upper } => OpdError::InvalidInput(format!(
            "OPD {stage} Qwen3.5 forward position id {position} is outside \
             rope cache size {upper}. Hint: reduce prompt length or \
             --rollout-len, or load/build a Qwen35Config with a larger \
             rope_cache_len_hint."
        )),
        Qwen35Error::InvalidConfig(reason) => OpdError::InvalidInput(format!(
            "OPD {stage} Qwen3.5 forward config error: {reason}. Hint: verify \
             Qwen35Config matches the checkpoint and that rope_cache_len_hint \
             covers prompt length plus rollout length."
        )),
        other => OpdError::Qwen35(other),
    }
}

/// Run one OPD step:
/// 1. Greedy-rollout `cfg.rollout_len` tokens from `student` starting from `prompt_ids`.
/// 2. Forward `teacher` on the full rollout (tape disabled).
/// 3. Forward `student` on the full rollout (tape enabled).
/// 4. `kl_distill_loss(student_logits, teacher_logits, rollout.len(), …)`.
/// 5. Backward + grad-clip + optimizer step.
/// 6. Clear ephemeral tensors so the next step starts from a clean store.
pub fn opd_step<O: Optimizer>(
    student: &Qwen35Model,
    teacher: &Qwen35Model,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<OpdStepOutcome> {
    validate_step_config(cfg)?;
    let vocab = student.config().vocab_size;
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "OPD step requires a non-empty prompt_ids slice. Hint: pass at least \
             one BOS/chat token; the OPD substrate does not synthesize prompts. \
             See docs/projects/2026-05-18-opd-only-pivot.md."
                .to_owned(),
        ));
    }
    if vocab == 0 {
        return Err(OpdError::InvalidInput(
            "OPD step requires student.config().vocab_size > 0. Hint: verify the \
             loaded Qwen35Config before constructing the student model."
                .to_owned(),
        ));
    }
    validate_rollout_shape(prompt_ids.len(), cfg.rollout_len, vocab)?;
    let teacher_vocab = teacher.config().vocab_size;
    if teacher_vocab != vocab {
        return Err(OpdError::InvalidInput(format!(
            "OPD requires teacher/student vocab_size to match, got \
             teacher.config().vocab_size={teacher_vocab} and \
             student.config().vocab_size={vocab}. Hint: use model directories \
             that share the same tokenizer before running OPD. See \
             docs/projects/2026-05-18-opd-only-pivot.md."
        )));
    }
    if let Some((index, token_id)) = prompt_ids
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, token_id)| token_id as usize >= vocab)
    {
        return Err(OpdError::InvalidInput(format!(
            "OPD prompt token id {token_id} at prompt_ids[{index}] is outside \
             student.config().vocab_size={vocab}. Hint: verify the tokenizer and \
             student model directory match before running OPD. See \
            docs/projects/2026-05-18-opd-only-pivot.md."
        )));
    }
    let teacher_params = teacher.all_parameter_ids();
    let student_model_params = student.all_parameter_ids();
    validate_teacher_params(&teacher_params, store)?;
    validate_student_params(student_params, store)?;
    validate_student_param_ownership(student_params, &student_model_params, &teacher_params)?;
    let keep_extra = retained_param_and_grad_ids(&teacher_params, store);

    // 1. Greedy rollout — tape disabled, no backward graph for sample tokens.
    tape.entries.clear();
    tape.set_enabled(false);
    let mut rollout: Vec<u32> = prompt_ids.to_vec();
    for _ in 0..cfg.rollout_len {
        let positions: Vec<u32> = (0..rollout.len() as u32).collect();
        let logits = student
            .forward(store, tape, &rollout, &positions)
            .map_err(|err| map_qwen35_forward_error("student rollout", err))?;
        let next = greedy_next_token(logits, rollout.len(), vocab, store)?;
        rollout.push(next);
    }
    // Note: rollout ephemerals (logits per iteration) stay in the store
    // until the post-step `retain_ids` prune below. For long rollouts
    // (>16 steps) we'd want per-iteration pruning, but that requires a
    // `keep` set covering both teacher AND student parameters, not just
    // `student_params`. Deferred until the production path benches.

    // 2. Teacher forward — still tape-disabled. Teacher params carry
    //    `requires_grad = false` so no entries record even if tape was on,
    //    but disabling cheap-defends against any rogue grad-bearing weight.
    let positions: Vec<u32> = (0..rollout.len() as u32).collect();
    let teacher_logits = teacher
        .forward(store, tape, &rollout, &positions)
        .map_err(|err| map_qwen35_forward_error("teacher scoring", err))?;

    // 3. Student forward — tape enabled now so backward can flow.
    tape.set_enabled(true);
    let student_logits = student
        .forward(store, tape, &rollout, &positions)
        .map_err(|err| map_qwen35_forward_error("student KL", err))?;

    // 4. KL distill loss.
    let loss = kl_distill_loss(student_logits, teacher_logits, rollout.len(), store, tape)?;
    let loss_value = store.to_host(loss)?[0];
    validate_loss_value(loss_value)?;

    // 5. Backward + grad clip + optimizer step.
    optimizer.zero_grad(store, student_params);
    tape.backward(loss, store)?;
    clip_grad_norm(student_params, cfg.grad_clip, store);
    optimizer.step(store, student_params)?;

    // 6. Prune rollout/teacher/student forward temporaries. Teacher params
    //    live in `keep_extra`; student params and their persistent grads are
    //    retained by `cleanup_after_backward`.
    cleanup_after_backward(store, tape, student_params, &keep_extra);

    Ok(OpdStepOutcome {
        loss: loss_value,
        rollout_len: rollout.len(),
    })
}

#[cfg(test)]
mod tests {
    use autograd::{Tensor, TensorStore};

    use super::{
        OpdError, OpdStepConfig, greedy_next_token, validate_loss_value, validate_rollout_shape,
        validate_step_config, validate_student_param_ownership, validate_student_params,
        validate_teacher_params,
    };

    #[test]
    fn greedy_next_token_rejects_logits_len_mismatch() {
        let mut store = TensorStore::default();
        let logits =
            store.alloc(Tensor::new(vec![0.0; 8], vec![1, 2, 4], false).expect("logits tensor"));

        let err = greedy_next_token(logits, 1, 4, &mut store)
            .expect_err("extra logits rows must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("logits length mismatch"));
        assert!(message.contains("expected exactly"));
        assert!(message.contains("1 * 4"));
    }

    #[test]
    fn greedy_next_token_rejects_non_finite_logits() {
        let mut store = TensorStore::default();
        let logits = store.alloc(
            Tensor::new(vec![0.0, 0.5, f32::NAN, 0.25], vec![1, 1, 4], false)
                .expect("logits tensor"),
        );

        let err = greedy_next_token(logits, 1, 4, &mut store)
            .expect_err("NaN logits must not silently sample token 0");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("non-finite"));
        assert!(message.contains("vocab index 2"));
        assert!(message.contains("student forward numerics"));
    }

    #[test]
    fn validate_student_params_rejects_empty_list() {
        let store = TensorStore::default();
        let err = validate_student_params(&[], &store)
            .expect_err("empty student parameter list must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("at least one student parameter id"));
        assert!(message.contains("optimizer step a no-op"));
    }

    #[test]
    fn validate_student_params_rejects_missing_tensor_id() {
        let store = TensorStore::default();
        let err = validate_student_params(&[17], &store)
            .expect_err("missing TensorStore id must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("student_params[0]=17"));
        assert!(message.contains("same student Qwen35Model"));
    }

    #[test]
    fn validate_student_params_rejects_duplicate_tensor_id() {
        let mut store = TensorStore::default();
        let trainable =
            store.alloc(Tensor::new(vec![0.0; 4], vec![2, 2], true).expect("trainable tensor"));

        let err = validate_student_params(&[trainable, trainable], &store)
            .expect_err("duplicate student parameter ids must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("student_params[1]"));
        assert!(message.contains("duplicates"));
        assert!(message.contains("exactly once"));
        assert!(message.contains("optimizer updates more than once"));
    }

    #[test]
    fn validate_student_params_rejects_frozen_only_params() {
        let mut store = TensorStore::default();
        let frozen =
            store.alloc(Tensor::new(vec![0.0; 4], vec![2, 2], false).expect("frozen tensor"));

        let err = validate_student_params(&[frozen], &store)
            .expect_err("frozen-only parameter list must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("no trainable tensors"));
        assert!(message.contains("requires_grad=true"));
        assert!(message.contains("optimizer step a no-op"));
    }

    #[test]
    fn validate_student_param_ownership_rejects_teacher_param_ids() {
        let student_param = 10;
        let teacher_param = 20;
        let err = validate_student_param_ownership(
            &[student_param, teacher_param],
            &[student_param],
            &[teacher_param],
        )
        .expect_err("teacher ids must not be accepted as student params");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("student_params[1]=20"));
        assert!(message.contains("frozen teacher"));
        assert!(message.contains("must not be optimized"));
    }

    #[test]
    fn validate_student_param_ownership_rejects_non_student_param_ids() {
        let student_param = 10;
        let foreign_param = 30;
        let err = validate_student_param_ownership(&[foreign_param], &[student_param], &[])
            .expect_err("foreign ids must not be accepted as student params");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("student_params[0]=30"));
        assert!(message.contains("not owned by the student"));
        assert!(message.contains("same TensorStore"));
    }

    #[test]
    fn validate_teacher_params_rejects_trainable_params() {
        let mut store = TensorStore::default();
        let trainable_teacher =
            store.alloc(Tensor::new(vec![0.0; 4], vec![2, 2], true).expect("teacher tensor"));

        let err = validate_teacher_params(&[trainable_teacher], &store)
            .expect_err("trainable teacher parameters must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("teacher_params[0]"));
        assert!(message.contains("requires_grad=true"));
        assert!(message.contains("new_for_eval"));
        assert!(message.contains("must not optimize teacher weights"));
    }

    #[test]
    fn validate_loss_value_rejects_non_finite_loss() {
        for loss_value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_loss_value(loss_value)
                .expect_err("non-finite OPD loss must be rejected before backward");

            let OpdError::InvalidInput(message) = err else {
                panic!("expected InvalidInput, got {err:?}");
            };
            assert!(message.contains("non-finite"));
            assert!(message.contains("teacher/student logits"));
            assert!(message.contains("learning rate"));
            assert!(message.contains("2026-05-18-opd-only-pivot.md"));
        }
    }

    #[test]
    fn validate_step_config_rejects_non_finite_grad_clip() {
        for grad_clip in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_step_config(OpdStepConfig {
                rollout_len: 1,
                grad_clip,
            })
            .expect_err("non-finite OPD grad_clip must not silently disable clipping");

            let OpdError::InvalidInput(message) = err else {
                panic!("expected InvalidInput, got {err:?}");
            };
            assert!(message.contains("cfg.grad_clip"));
            assert!(message.contains("finite"));
            assert!(message.contains("0.0"));
        }
    }

    #[test]
    fn validate_rollout_shape_rejects_total_len_overflow() {
        let err = validate_rollout_shape(2, usize::MAX, 16)
            .expect_err("rollout length overflow must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("rollout length overflow"));
        assert!(message.contains("prompt_len=2"));
        assert!(message.contains("rollout-len"));
    }

    #[test]
    fn validate_rollout_shape_rejects_u32_position_overflow() {
        let err = validate_rollout_shape(u32::MAX as usize, 1, 16)
            .expect_err("position id overflow must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("exceeds u32::MAX position ids"));
        assert!(message.contains("u32 position ids"));
    }

    #[test]
    fn validate_rollout_shape_rejects_u32_vocab_overflow() {
        let err = validate_rollout_shape(1, 1, u32::MAX as usize + 1)
            .expect_err("token id overflow must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("vocab_size"));
        assert!(message.contains("u32 token ids"));
    }
}
