//! Input, config, and numeric-shape checks for the OPD step surface.

use std::collections::HashSet;

use autograd::{TensorId, TensorStore};

use crate::loss::KlDirection;

use super::{GkdLossConfig, GkdSftAnchor, OpdError, OpdStepConfig, Result};

pub(super) fn validate_token_ids(context: &str, tokens: &[u32], vocab: usize) -> Result<()> {
    if let Some((index, token_id)) = tokens
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, token_id)| token_id as usize >= vocab)
    {
        return Err(OpdError::InvalidInput(format!(
            "{context} token id {token_id} at {context}[{index}] is outside \
             student.config().vocab_size={vocab}. Hint: verify the tokenizer and \
             student model directory match before running OPD."
        )));
    }
    Ok(())
}

pub(super) fn validate_forced_rollout(
    forced_rollout: &[u32],
    prompt_ids: &[u32],
    rollout_len: usize,
    vocab: usize,
) -> Result<()> {
    let expected_len = prompt_ids
        .len()
        .checked_add(rollout_len)
        .ok_or_else(|| OpdError::InvalidInput("OPD forced_rollout length overflow".to_owned()))?;
    if forced_rollout.len() != expected_len {
        return Err(OpdError::InvalidInput(format!(
            "OPD forced_rollout length mismatch: got {}, expected {} \
             (= prompt_len {} + rollout_len {}). Hint: pass the full selected \
             prompt-plus-rollout trajectory.",
            forced_rollout.len(),
            expected_len,
            prompt_ids.len(),
            rollout_len
        )));
    }
    if !forced_rollout.starts_with(prompt_ids) {
        return Err(OpdError::InvalidInput(
            "OPD forced_rollout must start with the exact prompt_ids prefix. \
             Hint: select whole candidate trajectories; do not splice response \
             suffixes into a different prompt."
                .to_owned(),
        ));
    }
    validate_token_ids("forced_rollout", forced_rollout, vocab)
}

pub(super) fn validate_student_params(
    student_params: &[TensorId],
    store: &TensorStore,
) -> Result<()> {
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

pub(super) fn validate_student_param_ownership(
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

pub(super) fn validate_teacher_params(
    teacher_params: &[TensorId],
    store: &TensorStore,
) -> Result<()> {
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

pub(super) fn validate_loss_value(loss_value: f32) -> Result<()> {
    if loss_value.is_finite() {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD KL loss became non-finite ({loss_value}). Hint: check teacher/student logits \
         for NaN or Inf, verify both checkpoints use the same tokenizer/model family, and \
         reduce the learning rate before resuming."
    )))
}

pub(super) fn validate_logits_shape(
    stage: &str,
    shape: &[usize],
    seq_len: usize,
    vocab: usize,
) -> Result<()> {
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

pub(super) fn validate_step_config(cfg: &OpdStepConfig) -> Result<()> {
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

pub(super) fn validate_gkd_lambda(gkd_lambda: f32) -> Result<()> {
    if (0.0..=1.0).contains(&gkd_lambda) && gkd_lambda.is_finite() {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD GKD lambda must be finite and in [0.0, 1.0], got {gkd_lambda}. \
         Hint: pass --gkd-lambda 0.0 for pure OPD, 0.3 for the literature \
         SFT/OPD blend probe, or 1.0 for pure hard-token SFT proxy."
    )))
}

pub(super) fn validate_gkd_loss_config(config: GkdLossConfig<'_>) -> Result<()> {
    validate_gkd_lambda(config.lambda)?;
    if !config.kl_temperature.is_finite() || config.kl_temperature <= 0.0 {
        return Err(OpdError::InvalidInput(format!(
            "OPD KL temperature must be finite and > 0.0, got {}. Hint: use \
             --kl-temperature 1.0 for the baseline KL loss or a positive value \
             for pure-OPD temperature softening.",
            config.kl_temperature
        )));
    }
    if config.kl_temperature != 1.0 && config.lambda > 0.0 {
        return Err(OpdError::InvalidInput(format!(
            "OPD KL temperature is a pure-OPD lever only: got \
             kl_temperature={} with gkd_lambda={}. The SFT anchor is deliberately \
             1/vocab-scale-matched to the KL term, so T^2 KL compensation would \
             silently reweight the (1-lambda)KL + lambda*SFT blend. Set \
             --gkd-lambda 0.0 or --kl-temperature 1.0.",
            config.kl_temperature, config.lambda
        )));
    }
    if let Some(beta) = config.kl_beta
        && !(beta.is_finite() && (0.0..=1.0).contains(&beta))
    {
        return Err(OpdError::InvalidInput(format!(
            "OPD KL beta must be finite and in [0.0, 1.0], got {beta}. \
             Hint: omit --kl-beta to keep --kl-direction, or pass 0.5 for \
             TRL-style generalized JSD."
        )));
    }
    if config.kl_chunk_size == Some(0) {
        return Err(OpdError::InvalidInput(
            "OPD KL chunk size must be > 0 when set. Hint: pass \
             --kl-chunk-size 32 for the 256-token rollout bench, or set an \
             explicit larger value after a memory check."
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

pub(super) fn validate_rollout_shape(
    prompt_len: usize,
    rollout_len: usize,
    vocab: usize,
) -> Result<()> {
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
