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
use std::{collections::HashSet, time::Instant};

use crate::{
    grad_clip::clip_grad_norm,
    loss::{cross_entropy_loss, kl_distill_loss, kl_distill_loss_chunked},
    qwen35::{
        Qwen35Error, Qwen35KvCache, Qwen35Model, SequenceWindow, forward_rollout_cached,
        forward_rollout_cached_device_token,
    },
    teacher_infer::{InProcessTeacher, TeacherForward, TeacherForwardError},
    trainer::{cleanup_after_backward, retained_param_and_grad_ids},
};
use autograd::ops::{add, mul_scalar, slice};

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

#[derive(Debug, Default, Clone, Copy)]
pub struct OpdStepProfile {
    pub total_seconds: f64,
    pub student_rollout_seconds: f64,
    pub teacher_forward_seconds: f64,
    pub student_forward_seconds: f64,
    pub kl_loss_seconds: f64,
    pub optimizer_zero_grad_seconds: f64,
    pub backward_seconds: f64,
    pub grad_clip_seconds: f64,
    pub optimizer_step_seconds: f64,
    pub post_step_cleanup_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GkdSftAnchor {
    StudentRollout,
    CorpusTruth,
}

#[derive(Debug, Clone, Copy)]
pub struct GkdLossConfig<'a> {
    pub lambda: f32,
    pub sft_anchor: GkdSftAnchor,
    pub corpus_tokens: Option<&'a [u32]>,
    pub kl_chunk_size: Option<usize>,
    pub logits_window_size: Option<usize>,
}

impl Default for GkdLossConfig<'_> {
    fn default() -> Self {
        Self {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: None,
            logits_window_size: None,
        }
    }
}

fn record_profile(
    profile: &mut Option<&mut OpdStepProfile>,
    update: impl FnOnce(&mut OpdStepProfile),
) {
    if let Some(profile) = profile.as_deref_mut() {
        update(profile);
    }
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

fn validate_logits_shape(stage: &str, shape: &[usize], seq_len: usize, vocab: usize) -> Result<()> {
    let expected_shape = vec![1, seq_len, vocab];
    if shape == expected_shape {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD {stage} logits shape mismatch: got {shape:?}, expected \
         {expected_shape:?}. Hint: windowed Route B requires each teacher and \
         student forward to return [batch=1, window_len, vocab] for exactly \
         the current logits window."
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

fn validate_gkd_lambda(gkd_lambda: f32) -> Result<()> {
    if (0.0..=1.0).contains(&gkd_lambda) && gkd_lambda.is_finite() {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD GKD lambda must be finite and in [0.0, 1.0], got {gkd_lambda}. \
         Hint: pass --gkd-lambda 0.0 for pure OPD, 0.3 for the literature \
         SFT/OPD blend probe, or 1.0 for pure hard-token SFT proxy."
    )))
}

fn validate_gkd_loss_config(config: GkdLossConfig<'_>) -> Result<()> {
    validate_gkd_lambda(config.lambda)?;
    if config.kl_chunk_size == Some(0) {
        return Err(OpdError::InvalidInput(
            "OPD KL chunk size must be > 0 when set. Hint: pass \
             --kl-chunk-size 64 for the 512-token real-corpus smoke, or omit \
             it to keep the baseline full-logits KL path."
                .to_owned(),
        ));
    }
    if config.logits_window_size == Some(0) {
        return Err(OpdError::InvalidInput(
            "OPD logits window size must be > 0 when set. Hint: pass \
             --logits-window-size 64 with --kl-chunk-size 64 for the \
             512-token real-corpus Route B smoke, or omit it to keep the \
             baseline full-logits path."
                .to_owned(),
        ));
    }
    if config.lambda == 0.0 {
        return Ok(());
    }
    if config.sft_anchor == GkdSftAnchor::CorpusTruth
        && config.corpus_tokens.is_none_or(|tokens| tokens.is_empty())
    {
        return Err(OpdError::InvalidInput(
            "GKD corpus-truth SFT anchor requires non-empty corpus completion \
             tokens when lambda > 0. Hint: add a `completion` or `target` \
             field to each training row in --prompts-file, or use \
             --sft-anchor student-rollout."
                .to_owned(),
        ));
    }
    Ok(())
}

fn kl_distill_loss_for_config(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    kl_chunk_size: Option<usize>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    match kl_chunk_size {
        Some(chunk_size) => kl_distill_loss_chunked(
            student_logits,
            teacher_logits,
            num_positions,
            chunk_size,
            store,
            tape,
        )
        .map_err(OpdError::from),
        None => kl_distill_loss(student_logits, teacher_logits, num_positions, store, tape)
            .map_err(OpdError::from),
    }
}

fn sequence_windows(total_positions: usize, window_size: usize) -> Result<Vec<SequenceWindow>> {
    if total_positions == 0 {
        return Err(OpdError::InvalidInput(
            "OPD windowed logits path requires at least one position. Hint: \
             pass a non-empty prompt/completion sequence before enabling \
             --logits-window-size."
                .to_owned(),
        ));
    }
    if window_size == 0 {
        return Err(OpdError::InvalidInput(
            "OPD logits window size must be > 0 when set. Hint: pass \
             --logits-window-size 64 or omit the flag for full logits."
                .to_owned(),
        ));
    }
    let mut windows = Vec::new();
    let mut start = 0usize;
    while start < total_positions {
        let end = start.saturating_add(window_size).min(total_positions);
        windows.push(SequenceWindow { start, end });
        start = end;
    }
    Ok(windows)
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

fn next_token_sft_loss_from_logits(
    student_logits: TensorId,
    logits_seq_len: usize,
    start_position: usize,
    target_tokens: &[u32],
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if target_tokens.is_empty() {
        return Err(OpdError::InvalidInput(
            "GKD SFT proxy requires at least one target token. Hint: provide \
             non-empty corpus completion tokens or use a rollout with at \
             least two tokens."
                .to_owned(),
        ));
    }
    let end_position = start_position
        .checked_add(target_tokens.len())
        .ok_or_else(|| {
            OpdError::InvalidInput(
                "GKD SFT proxy logits slice overflowed. Hint: check prompt \
                 and completion lengths before mixing SFT into OPD."
                    .to_owned(),
            )
        })?;
    if end_position > logits_seq_len {
        return Err(OpdError::InvalidInput(format!(
            "GKD SFT proxy target slice [{}..{}) exceeds logits_seq_len={}. \
             Hint: completion-token CE should use logits from the prompt's \
             final token through the completion prefix.",
            start_position, end_position, logits_seq_len
        )));
    }
    let shape = store
        .get(student_logits)
        .ok_or(AutogradError::InvalidTensorId(student_logits))?
        .shape
        .clone();
    let expected_shape = vec![1, logits_seq_len, vocab];
    if shape != expected_shape {
        return Err(OpdError::InvalidInput(format!(
            "GKD SFT proxy expected student logits shape {:?}, got {:?}. \
             Hint: pass logits from the same sequence used for the SFT \
             target tokens.",
            expected_shape, shape
        )));
    }
    let mut targets = Vec::with_capacity(target_tokens.len());
    for (index, &token_id) in target_tokens.iter().enumerate() {
        if token_id as usize >= vocab {
            return Err(OpdError::InvalidInput(format!(
                "GKD SFT proxy target token {token_id} at target[{index}] is \
                 outside vocab={vocab}. Hint: verify tokenizer/model vocab \
                 alignment before mixing hard-token SFT into OPD."
            )));
        }
        targets.push(token_id as usize);
    }
    let shifted_logits = slice(
        student_logits,
        &[0, start_position, 0],
        &[1, end_position, vocab],
        store,
        tape,
    )?;
    let token_mean_ce = cross_entropy_loss(shifted_logits, &targets, store, tape)?;
    // `kl_distill_loss` intentionally uses mean over positions * vocab.
    // Scale the hard-label CE to the same internal normalization before
    // mixing, otherwise lambda=0.3 would dominate KL by roughly vocab_size.
    mul_scalar(token_mean_ce, 1.0 / vocab as f32, store, tape).map_err(OpdError::from)
}

fn shifted_rollout_sft_loss(
    student_logits: TensorId,
    rollout: &[u32],
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if rollout.len() < 2 {
        return Err(OpdError::InvalidInput(
            "GKD student-rollout SFT anchor requires at least two rollout \
             tokens so logits can be trained against next-token labels. Hint: \
             use a prompt with length >= 2 or set --rollout-len > 0 when \
             --gkd-lambda > 0."
                .to_owned(),
        ));
    }
    next_token_sft_loss_from_logits(
        student_logits,
        rollout.len(),
        0,
        &rollout[1..],
        vocab,
        store,
        tape,
    )
}

fn corpus_truth_sft_loss(
    student: &Qwen35Model,
    prompt_ids: &[u32],
    corpus_tokens: &[u32],
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "GKD corpus-truth SFT anchor requires a non-empty prompt. Hint: \
             OPD prompts should include at least one context token."
                .to_owned(),
        ));
    }
    if corpus_tokens.is_empty() {
        return Err(OpdError::InvalidInput(
            "GKD corpus-truth SFT anchor requires non-empty completion tokens. \
             Hint: add a `completion` or `target` field to each training row \
             in --prompts-file."
                .to_owned(),
        ));
    }
    let total_len = prompt_ids
        .len()
        .checked_add(corpus_tokens.len())
        .ok_or_else(|| {
            OpdError::InvalidInput(
                "GKD corpus-truth SFT prompt+completion length overflowed. \
                 Hint: reduce prompt/completion max tokens."
                    .to_owned(),
            )
        })?;
    if total_len > u32::MAX as usize {
        return Err(OpdError::InvalidInput(format!(
            "GKD corpus-truth SFT sequence length {total_len} exceeds u32::MAX \
             RoPE position range. Hint: reduce prompt/completion max tokens."
        )));
    }
    for (index, &token_id) in corpus_tokens.iter().enumerate() {
        if token_id as usize >= vocab {
            return Err(OpdError::InvalidInput(format!(
                "GKD corpus-truth SFT completion token {token_id} at \
                 completion[{index}] is outside vocab={vocab}. Hint: verify \
                 tokenizer/model vocab alignment before training."
            )));
        }
    }
    let mut sft_sequence = Vec::with_capacity(total_len);
    sft_sequence.extend_from_slice(prompt_ids);
    sft_sequence.extend_from_slice(corpus_tokens);
    let positions = (0..total_len as u32).collect::<Vec<_>>();
    let student_logits = student
        .forward(store, tape, &sft_sequence, &positions)
        .map_err(|err| map_qwen35_forward_error("student corpus SFT", err))?;
    next_token_sft_loss_from_logits(
        student_logits,
        total_len,
        prompt_ids.len() - 1,
        corpus_tokens,
        vocab,
        store,
        tape,
    )
}

fn gkd_sft_loss(
    config: GkdLossConfig<'_>,
    student: &Qwen35Model,
    prompt_ids: &[u32],
    student_logits: TensorId,
    rollout: &[u32],
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    match config.sft_anchor {
        GkdSftAnchor::StudentRollout => {
            shifted_rollout_sft_loss(student_logits, rollout, vocab, store, tape)
        }
        GkdSftAnchor::CorpusTruth => corpus_truth_sft_loss(
            student,
            prompt_ids,
            config.corpus_tokens.ok_or_else(|| {
                OpdError::InvalidInput(
                    "GKD corpus-truth SFT anchor requires corpus completion \
                     tokens. Hint: add completion/target fields to \
                     --prompts-file."
                        .to_owned(),
                )
            })?,
            vocab,
            store,
            tape,
        ),
    }
}

fn mix_gkd_losses(
    kl_loss: TensorId,
    sft_loss: TensorId,
    gkd_lambda: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    validate_gkd_lambda(gkd_lambda)?;
    if gkd_lambda == 0.0 {
        return Ok(kl_loss);
    }
    if gkd_lambda == 1.0 {
        return Ok(sft_loss);
    }
    let weighted_kl = mul_scalar(kl_loss, 1.0 - gkd_lambda, store, tape)?;
    let weighted_sft = mul_scalar(sft_loss, gkd_lambda, store, tape)?;
    add(weighted_kl, weighted_sft, store, tape).map_err(OpdError::from)
}

fn backward_weighted_window_loss(
    loss: TensorId,
    weight: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<f32> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(OpdError::InvalidInput(format!(
            "OPD window loss weight must be finite and non-negative, got {weight}. \
             Hint: verify lambda and window/target counts before Route B backward."
        )));
    }
    let loss_started = Instant::now();
    let weighted_loss = if (weight - 1.0).abs() < f32::EPSILON {
        loss
    } else {
        mul_scalar(loss, weight, store, tape)?
    };
    let loss_value = store.to_host(weighted_loss)?[0];
    validate_loss_value(loss_value)?;
    record_profile(profile, |profile| {
        profile.kl_loss_seconds += loss_started.elapsed().as_secs_f64();
    });

    let phase_started = Instant::now();
    tape.backward(weighted_loss, store)?;
    record_profile(profile, |profile| {
        profile.backward_seconds += phase_started.elapsed().as_secs_f64();
    });
    Ok(loss_value)
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
        TeacherForwardError::ApiRuntime(reason) => OpdError::InvalidInput(format!(
            "OPD {stage} API teacher runtime error: {reason}. Hint: verify \
             the API teacher endpoint is reachable, returns full logits for \
             every requested token position, and uses the same tokenizer/vocab \
             as the student."
        )),
        TeacherForwardError::ApiDecode(reason) => OpdError::InvalidInput(format!(
            "OPD {stage} API teacher logits decode error: {reason}. Hint: verify \
             the response shape is [seq,vocab] or [1,seq,vocab], dtype is f32 \
             or bf16, and logits_b64 is little-endian."
        )),
        #[cfg(feature = "cuda")]
        TeacherForwardError::InferRuntime(reason) => OpdError::InvalidInput(format!(
            "OPD {stage} infer teacher runtime error: {reason}. Hint: verify \
             the infer teacher model is loaded on CUDA, raw logits export is \
             available, and the token positions are contiguous from zero for \
             the current Path B bridge."
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn backward_windowed_gkd_loss<T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    rollout: &[u32],
    positions: &[u32],
    vocab: usize,
    gkd_config: GkdLossConfig<'_>,
    window_size: usize,
    student_model_params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<f32> {
    let mut total_loss = 0.0f32;
    let mut backward_windows = 0usize;

    if gkd_config.lambda < 1.0 {
        for window in sequence_windows(rollout.len(), window_size)? {
            tape.entries.clear();
            tape.set_enabled(false);

            let phase_started = Instant::now();
            let teacher_logits = teacher
                .forward_logits_window_device(rollout, positions, window, store, tape)
                .map_err(|err| map_teacher_forward_error("teacher windowed KL", err))?;
            record_profile(profile, |profile| {
                profile.teacher_forward_seconds += phase_started.elapsed().as_secs_f64();
            });
            validate_logits_shape(
                "teacher windowed KL",
                &teacher_logits.shape,
                window.len(),
                vocab,
            )?;

            tape.set_enabled(true);
            let phase_started = Instant::now();
            let student_logits = student
                .forward_logits_window(store, tape, rollout, positions, window)
                .map_err(|err| map_qwen35_forward_error("student windowed KL", err))?;
            let student_shape = store
                .get(student_logits)
                .ok_or(AutogradError::InvalidTensorId(student_logits))?
                .shape
                .clone();
            validate_logits_shape("student windowed KL", &student_shape, window.len(), vocab)?;
            record_profile(profile, |profile| {
                profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
            });

            let phase_started = Instant::now();
            let kl_loss = kl_distill_loss_for_config(
                student_logits,
                teacher_logits.tensor_id,
                window.len(),
                gkd_config.kl_chunk_size,
                store,
                tape,
            )?;
            record_profile(profile, |profile| {
                profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
            });
            let weight = (1.0 - gkd_config.lambda) * (window.len() as f32 / rollout.len() as f32);
            total_loss += backward_weighted_window_loss(kl_loss, weight, store, tape, profile)?;
            backward_windows += 1;
            cleanup_after_backward(store, tape, student_model_params, keep_extra);
        }
    }

    if gkd_config.lambda > 0.0 {
        match gkd_config.sft_anchor {
            GkdSftAnchor::StudentRollout => {
                let target_count = rollout.len().checked_sub(1).ok_or_else(|| {
                    OpdError::InvalidInput(
                        "GKD student-rollout SFT anchor requires at least two rollout \
                         tokens so logits can be trained against next-token labels."
                            .to_owned(),
                    )
                })?;
                if target_count == 0 {
                    return Err(OpdError::InvalidInput(
                        "GKD student-rollout SFT anchor requires at least two rollout \
                         tokens so logits can be trained against next-token labels."
                            .to_owned(),
                    ));
                }
                for target_window in sequence_windows(target_count, window_size)? {
                    tape.entries.clear();
                    tape.set_enabled(true);

                    let logits_window = target_window;
                    let phase_started = Instant::now();
                    let student_logits = student
                        .forward_logits_window(store, tape, rollout, positions, logits_window)
                        .map_err(|err| {
                            map_qwen35_forward_error("student windowed rollout SFT", err)
                        })?;
                    let student_shape = store
                        .get(student_logits)
                        .ok_or(AutogradError::InvalidTensorId(student_logits))?
                        .shape
                        .clone();
                    validate_logits_shape(
                        "student windowed rollout SFT",
                        &student_shape,
                        logits_window.len(),
                        vocab,
                    )?;
                    record_profile(profile, |profile| {
                        profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
                    });

                    let target_tokens = &rollout[target_window.start + 1..target_window.end + 1];
                    let phase_started = Instant::now();
                    let sft_loss = next_token_sft_loss_from_logits(
                        student_logits,
                        logits_window.len(),
                        0,
                        target_tokens,
                        vocab,
                        store,
                        tape,
                    )?;
                    record_profile(profile, |profile| {
                        profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
                    });
                    let weight =
                        gkd_config.lambda * (target_tokens.len() as f32 / target_count as f32);
                    total_loss +=
                        backward_weighted_window_loss(sft_loss, weight, store, tape, profile)?;
                    backward_windows += 1;
                    cleanup_after_backward(store, tape, student_model_params, keep_extra);
                }
            }
            GkdSftAnchor::CorpusTruth => {
                let corpus_tokens = gkd_config.corpus_tokens.ok_or_else(|| {
                    OpdError::InvalidInput(
                        "GKD corpus-truth SFT anchor requires corpus completion \
                         tokens. Hint: add completion/target fields to \
                         --prompts-file."
                            .to_owned(),
                    )
                })?;
                if prompt_ids.is_empty() || corpus_tokens.is_empty() {
                    return Err(OpdError::InvalidInput(
                        "GKD corpus-truth SFT anchor requires non-empty prompt and \
                         completion tokens."
                            .to_owned(),
                    ));
                }
                let total_len = prompt_ids
                    .len()
                    .checked_add(corpus_tokens.len())
                    .ok_or_else(|| {
                        OpdError::InvalidInput(
                            "GKD corpus-truth SFT prompt+completion length overflowed. \
                             Hint: reduce prompt/completion max tokens."
                                .to_owned(),
                        )
                    })?;
                if total_len > u32::MAX as usize {
                    return Err(OpdError::InvalidInput(format!(
                        "GKD corpus-truth SFT sequence length {total_len} exceeds \
                         u32::MAX RoPE position range. Hint: reduce prompt/completion \
                         max tokens."
                    )));
                }
                let mut sft_sequence = Vec::with_capacity(total_len);
                sft_sequence.extend_from_slice(prompt_ids);
                sft_sequence.extend_from_slice(corpus_tokens);
                let sft_positions = (0..total_len as u32).collect::<Vec<_>>();
                for target_window in sequence_windows(corpus_tokens.len(), window_size)? {
                    let logits_window = SequenceWindow {
                        start: prompt_ids.len() - 1 + target_window.start,
                        end: prompt_ids.len() - 1 + target_window.end,
                    };

                    tape.entries.clear();
                    tape.set_enabled(true);
                    let phase_started = Instant::now();
                    let student_logits = student
                        .forward_logits_window(
                            store,
                            tape,
                            &sft_sequence,
                            &sft_positions,
                            logits_window,
                        )
                        .map_err(|err| {
                            map_qwen35_forward_error("student windowed corpus SFT", err)
                        })?;
                    let student_shape = store
                        .get(student_logits)
                        .ok_or(AutogradError::InvalidTensorId(student_logits))?
                        .shape
                        .clone();
                    validate_logits_shape(
                        "student windowed corpus SFT",
                        &student_shape,
                        logits_window.len(),
                        vocab,
                    )?;
                    record_profile(profile, |profile| {
                        profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
                    });

                    let target_tokens = &corpus_tokens[target_window.start..target_window.end];
                    let phase_started = Instant::now();
                    let sft_loss = next_token_sft_loss_from_logits(
                        student_logits,
                        logits_window.len(),
                        0,
                        target_tokens,
                        vocab,
                        store,
                        tape,
                    )?;
                    record_profile(profile, |profile| {
                        profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
                    });
                    let weight = gkd_config.lambda
                        * (target_tokens.len() as f32 / corpus_tokens.len() as f32);
                    total_loss +=
                        backward_weighted_window_loss(sft_loss, weight, store, tape, profile)?;
                    backward_windows += 1;
                    cleanup_after_backward(store, tape, student_model_params, keep_extra);
                }
            }
        }
    }

    if backward_windows == 0 {
        return Err(OpdError::InvalidInput(
            "OPD windowed Route B built zero backward windows. Hint: verify \
             lambda, prompt length, rollout length, and --logits-window-size."
                .to_owned(),
        ));
    }
    Ok(total_loss)
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
    opd_step_with_teacher_forward_profiled(
        student,
        teacher,
        prompt_ids,
        cfg,
        student_params,
        optimizer,
        store,
        tape,
        None,
    )
}

pub fn opd_step_with_teacher_forward_profiled<O: Optimizer, T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    opd_step_with_teacher_forward_profiled_gkd(
        student,
        teacher,
        prompt_ids,
        cfg,
        student_params,
        optimizer,
        store,
        tape,
        0.0,
        profile,
    )
}

pub fn opd_step_with_teacher_forward_profiled_gkd<O: Optimizer, T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    gkd_lambda: f32,
    profile: Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    opd_step_with_teacher_forward_profiled_gkd_anchor(
        student,
        teacher,
        prompt_ids,
        cfg,
        student_params,
        optimizer,
        store,
        tape,
        GkdLossConfig {
            lambda: gkd_lambda,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: None,
            logits_window_size: None,
        },
        profile,
    )
}

pub fn opd_step_with_teacher_forward_profiled_gkd_anchor<
    O: Optimizer,
    T: TeacherForward + ?Sized,
>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    gkd_config: GkdLossConfig<'_>,
    profile: Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    let mut profile = profile;
    if let Some(profile) = profile.as_deref_mut() {
        *profile = OpdStepProfile::default();
    }
    let total_started = Instant::now();
    validate_step_config(cfg)?;
    validate_gkd_loss_config(gkd_config)?;
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
        let phase_started = Instant::now();
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
        record_profile(&mut profile, |profile| {
            profile.student_rollout_seconds += phase_started.elapsed().as_secs_f64();
        });

        // 2. Teacher forward — still tape-disabled. Teacher params carry
        //    `requires_grad = false` so no entries record even if tape was on,
        //    but disabling cheap-defends against any rogue grad-bearing weight.
        let positions: Vec<u32> = (0..rollout.len() as u32).collect();
        if let Some(window_size) = gkd_config.logits_window_size {
            let phase_started = Instant::now();
            optimizer.zero_grad(store, student_params);
            record_profile(&mut profile, |profile| {
                profile.optimizer_zero_grad_seconds += phase_started.elapsed().as_secs_f64();
            });

            let loss_value = backward_windowed_gkd_loss(
                student,
                teacher,
                prompt_ids,
                &rollout,
                &positions,
                vocab,
                gkd_config,
                window_size,
                &student_model_params,
                &keep_extra,
                store,
                tape,
                &mut profile,
            )?;

            let phase_started = Instant::now();
            clip_grad_norm(student_params, cfg.grad_clip, store);
            record_profile(&mut profile, |profile| {
                profile.grad_clip_seconds += phase_started.elapsed().as_secs_f64();
            });
            let phase_started = Instant::now();
            optimizer.step(store, student_params)?;
            record_profile(&mut profile, |profile| {
                profile.optimizer_step_seconds += phase_started.elapsed().as_secs_f64();
            });

            return Ok(OpdStepOutcome {
                loss: loss_value,
                rollout_len: rollout.len(),
            });
        }

        let phase_started = Instant::now();
        let teacher_logits = teacher
            .forward_logits_device(&rollout, &positions, store, tape)
            .map_err(|err| map_teacher_forward_error("teacher scoring", err))?;
        record_profile(&mut profile, |profile| {
            profile.teacher_forward_seconds += phase_started.elapsed().as_secs_f64();
        });
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
        let phase_started = Instant::now();
        let student_logits = student
            .forward(store, tape, &rollout, &positions)
            .map_err(|err| map_qwen35_forward_error("student KL", err))?;
        record_profile(&mut profile, |profile| {
            profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
        });

        // 4. KL distill loss, optionally blended with a GKD hard-token
        //    SFT anchor. The legacy anchor uses the on-policy rollout;
        //    corpus-truth mode re-forwards the student over prompt+target.
        let phase_started = Instant::now();
        let kl_loss = kl_distill_loss_for_config(
            student_logits,
            teacher_logits.tensor_id,
            rollout.len(),
            gkd_config.kl_chunk_size,
            store,
            tape,
        )?;
        let loss = if gkd_config.lambda == 0.0 {
            kl_loss
        } else {
            let sft_loss = gkd_sft_loss(
                gkd_config,
                student,
                prompt_ids,
                student_logits,
                &rollout,
                vocab,
                store,
                tape,
            )?;
            mix_gkd_losses(kl_loss, sft_loss, gkd_config.lambda, store, tape)?
        };
        let loss_value = store.to_host(loss)?[0];
        validate_loss_value(loss_value)?;
        record_profile(&mut profile, |profile| {
            profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
        });

        // 5. Backward + grad clip + optimizer step.
        let phase_started = Instant::now();
        optimizer.zero_grad(store, student_params);
        record_profile(&mut profile, |profile| {
            profile.optimizer_zero_grad_seconds += phase_started.elapsed().as_secs_f64();
        });
        let phase_started = Instant::now();
        tape.backward(loss, store)?;
        record_profile(&mut profile, |profile| {
            profile.backward_seconds += phase_started.elapsed().as_secs_f64();
        });
        let phase_started = Instant::now();
        clip_grad_norm(student_params, cfg.grad_clip, store);
        record_profile(&mut profile, |profile| {
            profile.grad_clip_seconds += phase_started.elapsed().as_secs_f64();
        });
        let phase_started = Instant::now();
        optimizer.step(store, student_params)?;
        record_profile(&mut profile, |profile| {
            profile.optimizer_step_seconds += phase_started.elapsed().as_secs_f64();
        });

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
    let phase_started = Instant::now();
    cleanup_after_backward(store, tape, &student_model_params, &keep_extra);
    record_profile(&mut profile, |profile| {
        profile.post_step_cleanup_seconds += phase_started.elapsed().as_secs_f64();
        profile.total_seconds = total_started.elapsed().as_secs_f64();
    });
    result
}

#[cfg(test)]
mod tests {
    use autograd::{AutogradError, Tape, Tensor, TensorStore};

    use super::{
        GkdLossConfig, GkdSftAnchor, OpdError, OpdStepConfig, greedy_next_token,
        kl_distill_loss_for_config, map_qwen35_forward_error, mix_gkd_losses,
        next_token_sft_loss_from_logits, sequence_windows, shifted_rollout_sft_loss,
        validate_gkd_lambda, validate_gkd_loss_config, validate_loss_value, validate_rollout_shape,
        validate_step_config, validate_student_param_ownership, validate_student_params,
        validate_teacher_params,
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
    fn validate_gkd_lambda_rejects_out_of_range_values() {
        for gkd_lambda in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            let err =
                validate_gkd_lambda(gkd_lambda).expect_err("invalid GKD lambda must be rejected");

            let OpdError::InvalidInput(message) = err else {
                panic!("expected InvalidInput, got {err:?}");
            };
            assert!(message.contains("GKD lambda"));
            assert!(message.contains("[0.0, 1.0]"));
        }
    }

    #[test]
    fn validate_gkd_loss_config_requires_corpus_tokens_for_corpus_anchor() {
        let missing = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.3,
            sft_anchor: GkdSftAnchor::CorpusTruth,
            corpus_tokens: None,
            kl_chunk_size: None,
            logits_window_size: None,
        })
        .expect_err("corpus anchor must require target tokens");
        let OpdError::InvalidInput(message) = missing else {
            panic!("expected InvalidInput, got {missing:?}");
        };
        assert!(message.contains("corpus-truth"));
        assert!(message.contains("completion"));

        let empty = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.3,
            sft_anchor: GkdSftAnchor::CorpusTruth,
            corpus_tokens: Some(&[]),
            kl_chunk_size: None,
            logits_window_size: None,
        })
        .expect_err("empty corpus anchor tokens must be rejected");
        let OpdError::InvalidInput(message) = empty else {
            panic!("expected InvalidInput, got {empty:?}");
        };
        assert!(message.contains("non-empty"));

        validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::CorpusTruth,
            corpus_tokens: None,
            kl_chunk_size: None,
            logits_window_size: None,
        })
        .expect("lambda=0 should not require unused corpus targets");
        validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.3,
            sft_anchor: GkdSftAnchor::CorpusTruth,
            corpus_tokens: Some(&[1, 2]),
            kl_chunk_size: None,
            logits_window_size: None,
        })
        .expect("non-empty corpus targets should be accepted");
    }

    #[test]
    fn validate_gkd_loss_config_rejects_zero_kl_chunk_size() {
        let err = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: Some(0),
            logits_window_size: None,
        })
        .expect_err("zero KL chunk size must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("KL chunk size"));
        assert!(message.contains("> 0"));
        assert!(message.contains("--kl-chunk-size"));
    }

    #[test]
    fn validate_gkd_loss_config_rejects_zero_logits_window_size() {
        let err = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: None,
            logits_window_size: Some(0),
        })
        .expect_err("zero logits window size must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("logits window size"));
        assert!(message.contains("> 0"));
        assert!(message.contains("--logits-window-size"));
    }

    #[test]
    fn sequence_windows_cover_tail_without_overlap() {
        let windows = sequence_windows(10, 4).expect("windows");
        let spans = windows
            .into_iter()
            .map(|window| (window.start, window.end))
            .collect::<Vec<_>>();
        assert_eq!(spans, vec![(0, 4), (4, 8), (8, 10)]);
    }

    #[test]
    fn kl_distill_loss_for_config_accepts_chunked_path() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let student = store.alloc(
            Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 2, 2], true).expect("student logits"),
        );
        let teacher = store.alloc(
            Tensor::new(vec![0.4, 0.3, 0.2, 0.1], vec![1, 2, 2], false).expect("teacher logits"),
        );

        let loss = kl_distill_loss_for_config(student, teacher, 2, Some(1), &mut store, &mut tape)
            .expect("chunked KL config should run");
        let value = store.to_host(loss).expect("loss host")[0];
        assert!(value.is_finite(), "chunked KL loss must be finite");
    }

    #[test]
    fn mix_gkd_losses_respects_lambda_endpoints_and_weighted_blend() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let kl = store.alloc(Tensor::new(vec![2.0], vec![1], true).expect("kl scalar"));
        let sft = store.alloc(Tensor::new(vec![10.0], vec![1], true).expect("sft scalar"));

        let pure_kl = mix_gkd_losses(kl, sft, 0.0, &mut store, &mut tape)
            .expect("lambda=0 should produce pure KL");
        assert_eq!(store.to_host(pure_kl).expect("pure kl host")[0], 2.0);

        let pure_sft = mix_gkd_losses(kl, sft, 1.0, &mut store, &mut tape)
            .expect("lambda=1 should produce pure SFT");
        assert_eq!(store.to_host(pure_sft).expect("pure sft host")[0], 10.0);

        let midpoint = mix_gkd_losses(kl, sft, 0.5, &mut store, &mut tape)
            .expect("lambda=0.5 should mix losses evenly");
        let value = store.to_host(midpoint).expect("midpoint host")[0];
        assert!((value - 6.0).abs() < 1.0e-6, "got {value}");

        let lambda03 = mix_gkd_losses(kl, sft, 0.3, &mut store, &mut tape)
            .expect("lambda=0.3 should mix losses with SFT weight 0.3");
        let value = store.to_host(lambda03).expect("lambda03 host")[0];
        assert!((value - 4.4).abs() < 1.0e-6, "got {value}");
    }

    #[test]
    fn shifted_rollout_sft_loss_uses_kl_internal_vocab_scale() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let logits = store.alloc(
            Tensor::new(
                vec![
                    0.0, 1.0, 2.0, 3.0, // position 0, target 2
                    4.0, 3.0, 2.0, 1.0, // position 1, target 1
                    0.5, 0.25, 0.0, -0.25, // position 2 ignored
                ],
                vec![1, 3, 4],
                true,
            )
            .expect("student logits"),
        );

        let loss = shifted_rollout_sft_loss(logits, &[0, 2, 1], 4, &mut store, &mut tape)
            .expect("shifted sft loss");
        let value = store.to_host(loss).expect("loss host")[0];

        // Manual CE over positions 0..1, then divided by vocab=4 to match
        // the current KL internal normalization.
        let ce0 = (0.0f32.exp() + 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln() - 2.0;
        let ce1 = (4.0f32.exp() + 3.0f32.exp() + 2.0f32.exp() + 1.0f32.exp()).ln() - 3.0;
        let expected = ((ce0 + ce1) * 0.5) / 4.0;
        assert!(
            (value - expected).abs() < 1.0e-6,
            "got {value}, expected {expected}"
        );
    }

    #[test]
    fn corpus_target_sft_loss_uses_completion_logits_after_prompt() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let logits = store.alloc(
            Tensor::new(
                vec![
                    0.0, 0.0, 0.0, 0.0, // prompt position 0 ignored
                    0.0, 1.0, 2.0, 3.0, // prompt final position -> target 3
                    4.0, 3.0, 2.0, 1.0, // completion prefix -> target 1
                    0.5, 0.25, 0.0, -0.25, // final position ignored
                ],
                vec![1, 4, 4],
                true,
            )
            .expect("student logits"),
        );

        let loss = next_token_sft_loss_from_logits(logits, 4, 1, &[3, 1], 4, &mut store, &mut tape)
            .expect("corpus sft loss");
        let value = store.to_host(loss).expect("loss host")[0];

        let ce0 = (0.0f32.exp() + 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln() - 3.0;
        let ce1 = (4.0f32.exp() + 3.0f32.exp() + 2.0f32.exp() + 1.0f32.exp()).ln() - 3.0;
        let expected = ((ce0 + ce1) * 0.5) / 4.0;
        assert!(
            (value - expected).abs() < 1.0e-6,
            "got {value}, expected {expected}"
        );
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
