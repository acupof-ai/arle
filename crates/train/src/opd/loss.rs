//! OPD objective terms: KL distill dispatch, the SFT anchors, and their blend.

use autograd::{
    AutogradError, Tape, TensorId, TensorStore,
    ops::{add, mul_scalar, slice},
};

use crate::{
    loss::{
        KlDirection, cross_entropy_loss, generalized_beta_jsd_loss,
        generalized_beta_jsd_loss_chunked, kl_distill_loss, kl_distill_loss_chunked,
    },
    qwen35::Qwen35Model,
};

use super::{
    GkdLossConfig, GkdSftAnchor, OpdError, Result, map_qwen35_forward_error,
    validation::validate_gkd_lambda,
};

pub(super) fn kl_distill_loss_for_config(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    kl_chunk_size: Option<usize>,
    kl_direction: KlDirection,
    kl_temperature: f32,
    kl_beta: Option<f32>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if let Some(beta) = kl_beta {
        return match kl_chunk_size {
            Some(chunk_size) => generalized_beta_jsd_loss_chunked(
                student_logits,
                teacher_logits,
                num_positions,
                chunk_size,
                kl_temperature,
                beta,
                store,
                tape,
            ),
            None => generalized_beta_jsd_loss(
                student_logits,
                teacher_logits,
                num_positions,
                kl_temperature,
                beta,
                store,
                tape,
            ),
        }
        .map_err(OpdError::from);
    }
    match kl_chunk_size {
        Some(chunk_size) => kl_distill_loss_chunked(
            student_logits,
            teacher_logits,
            num_positions,
            chunk_size,
            kl_temperature,
            kl_direction,
            store,
            tape,
        )
        .map_err(OpdError::from),
        None => kl_distill_loss(
            student_logits,
            teacher_logits,
            num_positions,
            kl_temperature,
            kl_direction,
            store,
            tape,
        )
        .map_err(OpdError::from),
    }
}

pub(super) fn next_token_sft_loss_from_logits(
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
    let targets: Vec<usize> = target_tokens
        .iter()
        .enumerate()
        .map(|(index, &token_id)| {
            if token_id as usize >= vocab {
                return Err(OpdError::InvalidInput(format!(
                    "GKD SFT proxy target token {token_id} at target[{index}] is \
                     outside vocab={vocab}. Hint: verify tokenizer/model vocab \
                     alignment before mixing hard-token SFT into OPD."
                )));
            }
            Ok(token_id as usize)
        })
        .collect::<Result<Vec<_>>>()?;
    let shifted_logits = slice(
        student_logits,
        &[0, start_position, 0],
        &[1, end_position, vocab],
        store,
        tape,
    )?;
    // `cross_entropy_loss` is already per-position (`mean` over the one gathered
    // target logit per position), which matches `kl_distill_loss`'s `batchmean`
    // (`sum_v / positions`) scale. No `1/vocab` rescale: both the KL and the
    // hard-label CE are per-position, so `--gkd-lambda` blends them at face
    // value. (Pre-batchmean-fix this was scaled by `1/vocab` to match the old
    // mean-over-positions*vocab KL; that compensation is removed with the fix.)
    cross_entropy_loss(shifted_logits, &targets, store, tape).map_err(OpdError::from)
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
    let sft_sequence: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(corpus_tokens.iter().copied())
        .collect();
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

pub(super) fn gkd_sft_loss(
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

pub(super) fn mix_gkd_losses(
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

/// Logits at sequence position `p` predict token `p + 1` (the
/// `next_token_sft_loss_from_logits` convention). A target counts iff:
/// - it lies inside the response span: `p + 1 >= prompt_len`, AND
/// - the response token at that index is LLM-generated:
///   `response_mask[(p + 1) - prompt_len] == 1`.
///
/// Tool / environment tokens (`response_mask == 0`) are excluded as *targets*
/// but still appear in the context that earlier predictions condition on, so
/// the student never learns to hallucinate tool output. Pure (no device work)
/// so it is unit-testable without a model.
pub(super) fn build_masked_loss_targets(
    full: &[u32],
    prompt_len: usize,
    response_mask: &[u8],
) -> Vec<(usize, usize)> {
    let seq_len = full.len();
    let mut targets = Vec::new();
    if seq_len < 2 || prompt_len == 0 {
        return targets;
    }
    let p_start = prompt_len - 1;
    for p in p_start..=(seq_len - 2) {
        let target_pos = p + 1;
        let resp_idx = target_pos - prompt_len;
        if resp_idx < response_mask.len() && response_mask[resp_idx] == 1 {
            targets.push((p, full[target_pos] as usize));
        }
    }
    targets
}
