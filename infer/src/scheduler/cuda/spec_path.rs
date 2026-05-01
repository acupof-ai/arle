use super::{ModelForward, Scheduler};
use crate::model::SpecVerifyRequest;
use crate::scheduler::DraftMode;
use crate::server_engine::FinishReason;

pub(super) struct SpecPath;

struct SpecRow {
    slot_idx: usize,
    request_id: u64,
    draft_start_position: usize,
    original_target_len: usize,
    input_tokens: Vec<u32>,
    draft_tokens: Vec<u32>,
}

impl SpecPath {
    pub(super) fn draft_then_verify<M: ModelForward>(scheduler: &mut Scheduler<M>) {
        if !matches!(scheduler.config.spec_draft_model, DraftMode::External(_)) {
            scheduler.step_spec_decode_launch_from_path();
            return;
        }
        if scheduler.draft_engine.is_none() || scheduler.config.spec_draft_k == 0 {
            scheduler.step_decode_launch();
            return;
        }

        let started = std::time::Instant::now();
        let (mut decode_indices, mut token_ids) = scheduler.collect_decode_batch_inputs();
        if decode_indices.is_empty() {
            return;
        }
        let verifier_tokens = scheduler.config.spec_draft_k.saturating_add(1);
        let extra_verifier_pages = decode_indices
            .iter()
            .map(|&slot_idx| {
                scheduler
                    .additional_pages_needed_for_slot(slot_idx, verifier_tokens)
                    .saturating_sub(scheduler.additional_pages_needed_for_slot(slot_idx, 1))
            })
            .sum();
        scheduler.retract_decode_to_fit(&mut decode_indices, &mut token_ids, extra_verifier_pages);
        if decode_indices.is_empty() {
            return;
        }
        let verifier_pages_needed: usize = decode_indices
            .iter()
            .map(|&slot_idx| scheduler.additional_pages_needed_for_slot(slot_idx, verifier_tokens))
            .sum();
        if verifier_pages_needed > scheduler.effective_pool_free_pages() {
            release_slot_draft_states(scheduler, &decode_indices);
            scheduler.step_decode_launch();
            return;
        }

        if !decode_indices.iter().all(|&slot_idx| {
            scheduler.request(slot_idx).is_some_and(|req| {
                !req.spec_decode_disabled
                    && !req.has_stop_sequences()
                    && req.sampling.is_greedy()
                    && !req.sampling.has_penalties()
            })
        }) {
            release_slot_draft_states(scheduler, &decode_indices);
            scheduler.step_decode_launch();
            return;
        }

        let mut rows = Vec::with_capacity(decode_indices.len());
        for &slot_idx in &decode_indices {
            let Some((request_id, prompt_tokens, generated_tokens, max_tokens, last_token)) =
                scheduler.request(slot_idx).and_then(|req| {
                    req.generated_tokens.last().copied().map(|last| {
                        (
                            req.id,
                            req.prompt_tokens.clone(),
                            req.generated_tokens.clone(),
                            req.max_tokens,
                            last,
                        )
                    })
                })
            else {
                continue;
            };

            let draft_engine = scheduler
                .draft_engine
                .as_ref()
                .expect("checked draft engine before spec path");
            if !draft_engine.has_request_state(request_id) {
                let mut prefix = Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
                prefix.extend_from_slice(&prompt_tokens);
                prefix.extend_from_slice(&generated_tokens);
                let draft_max_seq_len = prompt_tokens
                    .len()
                    .saturating_add(max_tokens)
                    .saturating_add(scheduler.config.spec_draft_k)
                    .saturating_add(1);
                if let Err(err) =
                    draft_engine.create_request_state(request_id, &prefix, draft_max_seq_len)
                {
                    log::warn!("spec draft state init failed for request {request_id}: {err}");
                    release_draft_states(scheduler, &rows, Some(request_id));
                    scheduler.step_decode_launch();
                    return;
                }
            }

            let draft_start_position = draft_engine
                .request_position(request_id)
                .unwrap_or(prompt_tokens.len() + generated_tokens.len());
            let proposal =
                match draft_engine.draft_for_request(request_id, scheduler.config.spec_draft_k) {
                    Ok(proposal) => proposal,
                    Err(err) => {
                        log::warn!("spec draft failed for request {request_id}: {err}");
                        release_draft_states(scheduler, &rows, Some(request_id));
                        scheduler.step_decode_launch();
                        return;
                    }
                };
            if proposal.tokens.is_empty() {
                continue;
            }

            let mut input_tokens = Vec::with_capacity(proposal.tokens.len() + 1);
            input_tokens.push(last_token);
            input_tokens.extend_from_slice(&proposal.tokens);
            rows.push(SpecRow {
                slot_idx,
                request_id,
                draft_start_position,
                original_target_len: scheduler.paged_kv_pool.seq_len(slot_idx),
                input_tokens,
                draft_tokens: proposal.tokens,
            });
        }

        if rows.is_empty() {
            scheduler.step_decode_launch();
            return;
        }

        let verify_requests: Vec<SpecVerifyRequest<'_>> = rows
            .iter()
            .map(|row| SpecVerifyRequest {
                slot_idx: row.slot_idx,
                input_tokens: &row.input_tokens,
                draft_tokens: &row.draft_tokens,
            })
            .collect();
        let outputs = match scheduler.model.forward_spec_verify_batch(
            &verify_requests,
            &mut scheduler.states,
            &mut scheduler.paged_kv_pool,
        ) {
            Ok(outputs) => outputs,
            Err(err) => {
                if err
                    .to_string()
                    .contains("does not support speculative verifier batch")
                {
                    log::warn!("spec verifier unsupported by target model; falling back: {err}");
                    release_draft_states(scheduler, &rows, None);
                    scheduler.step_decode_launch();
                    return;
                }
                log::error!("spec verifier failed: {err}");
                for row in &rows {
                    scheduler.finish_slot(row.slot_idx);
                }
                return;
            }
        };

        let mut draft_total = 0usize;
        let mut verified_total = 0usize;
        let mut accepted_total = 0usize;
        for row in rows {
            let Some(output) = outputs
                .iter()
                .find(|output| output.slot_idx == row.slot_idx)
            else {
                scheduler.finish_slot(row.slot_idx);
                continue;
            };
            let result = crate::speculative::verify_tokens_greedy(
                &row.draft_tokens,
                &output.target_argmax_tokens,
            );
            let bonus = output
                .target_argmax_tokens
                .get(result.num_accepted)
                .copied()
                .unwrap_or_else(|| row.draft_tokens[result.num_accepted.saturating_sub(1)]);
            let keep_target_len = row
                .original_target_len
                .saturating_add(1)
                .saturating_add(result.num_accepted);
            if let Err(err) = scheduler
                .paged_kv_pool
                .truncate_slot(row.slot_idx, keep_target_len)
            {
                log::error!("spec target KV rollback failed: {err}");
                scheduler.finish_slot(row.slot_idx);
                continue;
            }
            if let Some(draft_engine) = scheduler.draft_engine.as_ref() {
                if let Err(err) = draft_engine.commit_request_state(
                    row.request_id,
                    row.draft_start_position,
                    result.num_accepted,
                    bonus,
                ) {
                    log::warn!(
                        "spec draft state commit failed for request {}: {err}",
                        row.request_id
                    );
                    draft_engine.release_request_state(row.request_id);
                }
            }

            draft_total = draft_total.saturating_add(row.draft_tokens.len());
            verified_total = verified_total.saturating_add(row.draft_tokens.len());
            accepted_total = accepted_total.saturating_add(result.num_accepted);

            let threshold = scheduler.config.spec_acceptance_threshold;
            if let Some(req) = scheduler.request_mut(row.slot_idx) {
                let tracker = req
                    .spec_acceptance_tracker
                    .get_or_insert_with(crate::speculative::AcceptanceTracker::default_window);
                tracker.observe_step(result.num_accepted, row.draft_tokens.len());
                if tracker.should_disable(threshold) {
                    req.spec_decode_disabled = true;
                }
            }

            for &token in result.accepted.iter().chain(std::iter::once(&bonus)) {
                let ignore_eos = scheduler
                    .request(row.slot_idx)
                    .is_some_and(|req| req.sampling.ignore_eos);
                if !ignore_eos && scheduler.model.is_stop_token(token) {
                    scheduler.finish_request(row.slot_idx, FinishReason::Stop);
                    break;
                }
                if let Some(req) = scheduler.request_mut(row.slot_idx) {
                    req.generated_tokens.push(token);
                }
                let reached_max = scheduler
                    .request(row.slot_idx)
                    .is_some_and(|req| req.generated_tokens.len() >= req.max_tokens);
                if reached_max {
                    if !scheduler.defer_finish_until_emit_gate(row.slot_idx, FinishReason::Length) {
                        scheduler.finish_request(row.slot_idx, FinishReason::Length);
                    }
                    break;
                }
            }
        }

        scheduler.metrics.record_spec_step(
            draft_total,
            verified_total,
            accepted_total,
            started.elapsed().as_micros() as u64,
        );
    }
}

fn release_draft_states<M: ModelForward>(
    scheduler: &Scheduler<M>,
    rows: &[SpecRow],
    current_request_id: Option<u64>,
) {
    let Some(draft_engine) = scheduler.draft_engine.as_ref() else {
        return;
    };
    for row in rows {
        draft_engine.release_request_state(row.request_id);
    }
    if let Some(request_id) = current_request_id {
        draft_engine.release_request_state(request_id);
    }
}

fn release_slot_draft_states<M: ModelForward>(scheduler: &Scheduler<M>, slot_indices: &[usize]) {
    let Some(draft_engine) = scheduler.draft_engine.as_ref() else {
        return;
    };
    for &slot_idx in slot_indices {
        if let Some(request_id) = scheduler.request(slot_idx).map(|req| req.id) {
            draft_engine.release_request_state(request_id);
        }
    }
}
