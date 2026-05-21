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

use autograd::{AutogradError, Device, Tape, TensorId, TensorStore, optim::Optimizer};
use std::collections::HashSet;

use crate::{
    grad_clip::clip_grad_norm,
    loss::kl_distill_loss,
    qwen35::{
        Qwen35Error, Qwen35KvCache, Qwen35Model, forward_rollout_cached,
        forward_rollout_cached_device_token,
    },
    teacher_infer::{InProcessTeacher, TeacherForward, TeacherForwardError},
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

fn device_argmax_token(
    logits_id: TensorId,
    vocab: usize,
    store: &mut TensorStore,
) -> Result<TensorId> {
    if vocab == 0 {
        return Err(OpdError::InvalidInput(
            "OPD rollout cannot sample next token with vocab=0. Hint: verify \
             Qwen35Config::vocab_size before calling opd_step."
                .to_owned(),
        ));
    }
    let shape = store
        .get(logits_id)
        .ok_or(AutogradError::InvalidTensorId(logits_id))?
        .shape
        .clone();
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim != vocab {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout logits last dim mismatch: got {last_dim}, expected \
             vocab={vocab}. Hint: check Qwen35Model::forward output shape and \
             Qwen35Config::vocab_size."
        )));
    }
    let total = shape.iter().product::<usize>();
    let rows = total / vocab;
    if rows != 1 {
        return Err(OpdError::InvalidInput(format!(
            "OPD device rollout expects exactly one logits row, got {rows}. \
             Hint: rollout KV cache should return only the final next-token \
             logits row."
        )));
    }
    store.ensure_device(logits_id)?;
    let logits_handle = store
        .get(logits_id)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "device_argmax_token: logits missing device handle",
        ))?;
    let token_handle = store.backend().argmax_last_dim(&logits_handle, &shape)?;
    Ok(store.alloc_device_tensor(vec![rows], token_handle)?)
}

fn write_rollout_token(
    buffer_id: TensorId,
    token_id: TensorId,
    rollout_len: usize,
    step: usize,
    store: &mut TensorStore,
) -> Result<TensorId> {
    store.ensure_device(buffer_id)?;
    store.ensure_device(token_id)?;
    let buffer_handle = store
        .get(buffer_id)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "write_rollout_token: rollout buffer missing device handle",
        ))?;
    let token_handle = store
        .get(token_id)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "write_rollout_token: token missing device handle",
        ))?;
    let next_handle =
        store
            .backend()
            .write_scalar_at(&buffer_handle, &token_handle, rollout_len, step)?;
    Ok(store.alloc_device_tensor(vec![rollout_len], next_handle)?)
}

fn read_generated_rollout_tokens(
    buffer_id: TensorId,
    rollout_len: usize,
    vocab: usize,
    store: &mut TensorStore,
) -> Result<Vec<u32>> {
    let host = store.to_host(buffer_id)?;
    if host.len() != rollout_len {
        return Err(OpdError::InvalidInput(format!(
            "OPD generated rollout token buffer length mismatch: got {}, \
             expected {rollout_len}. Hint: device argmax rollout buffer shape \
             should match cfg.rollout_len.",
            host.len()
        )));
    }
    let mut out = Vec::with_capacity(rollout_len);
    for (index, &value) in host.iter().enumerate() {
        if !value.is_finite() {
            return Err(OpdError::InvalidInput(format!(
                "OPD generated rollout token at index {index} is non-finite ({value}). \
                 Hint: check CUDA argmax output and student forward numerics."
            )));
        }
        let rounded = value.round();
        if (value - rounded).abs() > 0.0 {
            return Err(OpdError::InvalidInput(format!(
                "OPD generated rollout token at index {index} is not an exact \
                 integer id ({value}). Hint: CUDA argmax should write exact \
                 f32 token ids."
            )));
        }
        if rounded < 0.0 || rounded as usize >= vocab {
            return Err(OpdError::InvalidInput(format!(
                "OPD generated rollout token id {rounded} at index {index} is \
                 outside student.config().vocab_size={vocab}. Hint: check the \
                 argmax kernel bounds and model vocab size."
            )));
        }
        out.push(rounded as u32);
    }
    Ok(out)
}

fn use_device_rollout_argmax(store: &TensorStore, rollout_len: usize, vocab: usize) -> bool {
    matches!(store.backend().device(), Device::Cuda) && (rollout_len >= 4 || vocab >= 65_536)
}

fn rollout_full_forward(
    student: &Qwen35Model,
    rollout: &mut Vec<u32>,
    rollout_len: usize,
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<()> {
    for _ in 0..rollout_len {
        let positions = (0..rollout.len() as u32).collect::<Vec<_>>();
        let logits = student
            .forward(store, tape, rollout, &positions)
            .map_err(|err| map_qwen35_forward_error("student rollout", err))?;
        let next = greedy_next_token(logits, rollout.len(), vocab, store)?;
        rollout.push(next);
    }
    Ok(())
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
    if cfg.grad_clip >= 0.0 && cfg.grad_clip.is_finite() {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD step requires cfg.grad_clip to be non-negative and finite, got {}. Hint: pass \
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
        Qwen35Error::Autograd(err) => OpdError::InvalidInput(format!(
            "OPD {stage} Qwen3.5 forward autograd error: {err}. Hint: verify \
             the checkpoint tensor shapes match config.json, that teacher and \
             student use compatible Qwen3.5-family layouts, and include this \
             stage name in the OPD loader/model follow-up report."
        )),
        Qwen35Error::Config(err) => OpdError::InvalidInput(format!(
            "OPD {stage} Qwen3.5 config error: {err}. Hint: verify config.json \
             is a supported Qwen3/Qwen3.5-family config before running OPD."
        )),
    }
}

fn map_teacher_forward_error(stage: &str, err: TeacherForwardError) -> OpdError {
    match err {
        TeacherForwardError::Qwen35(err) => map_qwen35_forward_error(stage, err),
        TeacherForwardError::Autograd(err) => OpdError::InvalidInput(format!(
            "OPD {stage} teacher forward autograd error: {err}. Hint: verify \
             the teacher runtime shares the same TensorStore backend and returns \
             device-resident logits compatible with the student KL path."
        )),
        TeacherForwardError::InvalidInput(reason) => OpdError::InvalidInput(format!(
            "OPD {stage} teacher forward input error: {reason}. Hint: verify \
             prompt_ids, rollout ids, and positions are aligned before scoring \
             the rollout."
        )),
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
    let teacher = InProcessTeacher::new(teacher);
    opd_step_with_teacher_forward(
        student,
        &teacher,
        prompt_ids,
        cfg,
        student_params,
        optimizer,
        store,
        tape,
    )
}

pub fn opd_step_with_teacher_forward<O: Optimizer, T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
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
    let teacher_vocab = teacher.vocab_size();
    if teacher_vocab != vocab {
        return Err(OpdError::InvalidInput(format!(
            "OPD requires teacher/student vocab_size to match, got \
             teacher.vocab_size()={teacher_vocab} and \
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
    let teacher_params = teacher.parameter_ids().to_vec();
    let student_model_params = student.all_parameter_ids();
    if !teacher_params.is_empty() {
        validate_teacher_params(&teacher_params, store)?;
    }
    validate_student_params(student_params, store)?;
    validate_student_param_ownership(student_params, &student_model_params, &teacher_params)?;
    let keep_extra = retained_param_and_grad_ids(&teacher_params, store);

    let result = (|| {
        // 1. Greedy rollout — tape disabled, no backward graph for sample tokens.
        tape.entries.clear();
        tape.set_enabled(false);
        let mut rollout: Vec<u32> = prompt_ids.to_vec();
        let use_rollout_kv_cache = student.supports_rollout_kv_cache();
        if use_rollout_kv_cache && use_device_rollout_argmax(store, cfg.rollout_len, vocab) {
            let mut rollout_cache = Qwen35KvCache::new(student);
            let mut generated_tokens = if cfg.rollout_len == 0 {
                None
            } else {
                let handle = store.backend().zeros(&[cfg.rollout_len])?;
                Some(store.alloc_device_tensor(vec![cfg.rollout_len], handle)?)
            };
            let mut current_device_token: Option<TensorId> = None;
            for step in 0..cfg.rollout_len {
                let logits = if step == 0 {
                    let positions = (0..prompt_ids.len() as u32).collect::<Vec<_>>();
                    forward_rollout_cached(
                        student,
                        store,
                        tape,
                        prompt_ids,
                        &positions,
                        &mut rollout_cache,
                    )
                    .map_err(|err| map_qwen35_forward_error("student rollout", err))?
                } else {
                    let token_id = current_device_token.ok_or_else(|| {
                        OpdError::InvalidInput(
                            "OPD rollout cache cannot decode from an empty rollout. Hint: pass a \
                             non-empty prompt before calling opd_step."
                                .to_owned(),
                        )
                    })?;
                    let position = (prompt_ids.len() + step - 1) as u32;
                    forward_rollout_cached_device_token(
                        student,
                        store,
                        tape,
                        token_id,
                        position,
                        &mut rollout_cache,
                    )
                    .map_err(|err| map_qwen35_forward_error("student rollout", err))?
                };
                let next_token = device_argmax_token(logits, vocab, store)?;
                if let Some(buffer_id) = generated_tokens {
                    generated_tokens = Some(write_rollout_token(
                        buffer_id,
                        next_token,
                        cfg.rollout_len,
                        step,
                        store,
                    )?);
                }
                current_device_token = Some(next_token);
            }
            if let Some(buffer_id) = generated_tokens {
                rollout.extend(read_generated_rollout_tokens(
                    buffer_id,
                    cfg.rollout_len,
                    vocab,
                    store,
                )?);
            }
        } else if use_rollout_kv_cache {
            let mut rollout_cache = Qwen35KvCache::new(student);
            for step in 0..cfg.rollout_len {
                let (input_ids, positions, logits_seq_len) = if step == 0 {
                    (
                        rollout.clone(),
                        (0..rollout.len() as u32).collect::<Vec<_>>(),
                        1,
                    )
                } else {
                    let last = *rollout.last().ok_or_else(|| {
                        OpdError::InvalidInput(
                            "OPD rollout cache cannot decode from an empty rollout. Hint: pass a \
                             non-empty prompt before calling opd_step."
                                .to_owned(),
                        )
                    })?;
                    let position = (rollout.len() - 1) as u32;
                    (vec![last], vec![position], 1)
                };
                let logits = forward_rollout_cached(
                    student,
                    store,
                    tape,
                    &input_ids,
                    &positions,
                    &mut rollout_cache,
                )
                .map_err(|err| map_qwen35_forward_error("student rollout", err))?;
                let next = greedy_next_token(logits, logits_seq_len, vocab, store)?;
                rollout.push(next);
            }
        } else {
            rollout_full_forward(student, &mut rollout, cfg.rollout_len, vocab, store, tape)?;
        }

        // 2. Teacher forward — still tape-disabled. Teacher params carry
        //    `requires_grad = false` so no entries record even if tape was on,
        //    but disabling cheap-defends against any rogue grad-bearing weight.
        let positions: Vec<u32> = (0..rollout.len() as u32).collect();
        let teacher_logits = teacher
            .forward_logits_device(&rollout, &positions, store, tape)
            .map_err(|err| map_teacher_forward_error("teacher scoring", err))?;
        let expected_teacher_shape = vec![1, rollout.len(), vocab];
        if teacher_logits.shape != expected_teacher_shape {
            return Err(OpdError::InvalidInput(format!(
                "OPD teacher logits shape mismatch: got {:?}, expected {:?}. \
                 Hint: the TeacherForward implementation must return \
                 [batch=1, seq_len, vocab] logits for the exact rollout \
                 scored by the student.",
                teacher_logits.shape, expected_teacher_shape
            )));
        }

        // 3. Student forward — tape enabled now so backward can flow.
        tape.set_enabled(true);
        let student_logits = student
            .forward(store, tape, &rollout, &positions)
            .map_err(|err| map_qwen35_forward_error("student KL", err))?;

        // 4. KL distill loss.
        let loss = kl_distill_loss(
            student_logits,
            teacher_logits.tensor_id,
            rollout.len(),
            store,
            tape,
        )?;
        let loss_value = store.to_host(loss)?[0];
        validate_loss_value(loss_value)?;

        // 5. Backward + grad clip + optimizer step.
        optimizer.zero_grad(store, student_params);
        tape.backward(loss, store)?;
        clip_grad_norm(student_params, cfg.grad_clip, store);
        optimizer.step(store, student_params)?;

        Ok(OpdStepOutcome {
            loss: loss_value,
            rollout_len: rollout.len(),
        })
    })();

    // 6. Prune rollout/teacher/student forward temporaries on both success
    //    and failure. Teacher params live in `keep_extra`. Retain the full
    //    student model, not just the optimizer target slice, because LoRA-only
    //    OPD optimizes adapter ids while still needing frozen base weights for
    //    the next forward pass.
    cleanup_after_backward(store, tape, &student_model_params, &keep_extra);
    result
}

#[cfg(test)]
mod tests {
    use autograd::{AutogradError, Tensor, TensorStore};

    use super::{
        OpdError, OpdStepConfig, greedy_next_token, map_qwen35_forward_error, validate_loss_value,
        validate_rollout_shape, validate_step_config, validate_student_param_ownership,
        validate_student_params, validate_teacher_params,
    };
    use crate::qwen35::Qwen35Error;

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
    fn validate_step_config_rejects_negative_grad_clip() {
        let err = validate_step_config(OpdStepConfig {
            rollout_len: 1,
            grad_clip: -1.0,
        })
        .expect_err("negative OPD grad_clip must not silently disable clipping");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("cfg.grad_clip"));
        assert!(message.contains("non-negative"));
        assert!(message.contains("0.0"));
    }

    #[test]
    fn map_qwen35_forward_error_wraps_autograd_errors_with_opd_context() {
        let err = map_qwen35_forward_error(
            "student KL",
            Qwen35Error::Autograd(AutogradError::ShapeMismatch {
                expected: vec![1, 2, 3],
                got: vec![1, 2],
            }),
        );

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("OPD student KL"));
        assert!(message.contains("autograd error"));
        assert!(message.contains("checkpoint tensor shapes"));
        assert!(message.contains("OPD loader/model follow-up"));
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
