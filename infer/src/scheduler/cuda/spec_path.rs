use super::{ModelForward, Phase, Scheduler};
use crate::model::{SparseKvDraftView, SpecVerifyRequest};
use crate::prefix_cache::BlockId;
use crate::scheduler::DraftMode;
use crate::server_engine::FinishReason;
use std::collections::HashMap;

pub(super) struct SpecPath;

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SparseDraftView {
    pub(super) slot_idx: usize,
    pub(super) page_ids: Vec<BlockId>,
    pub(super) attached_page_ids: Vec<BlockId>,
    pub(super) sparse_total_tokens: usize,
    /// Recent active-slot tokens that P2.B.3 must resolve from paged-KV slot
    /// state, including generated tail tokens that are not present in radix.
    pub(super) active_recent_tokens: usize,
}

impl SparseDraftView {
    fn new(
        slot_idx: usize,
        page_ids: Vec<BlockId>,
        attached_page_ids: Vec<BlockId>,
        block_size: usize,
        active_recent_tokens: usize,
    ) -> Self {
        Self {
            slot_idx,
            sparse_total_tokens: page_ids.len().saturating_mul(block_size),
            active_recent_tokens,
            page_ids,
            attached_page_ids,
        }
    }
}

struct SpecRow {
    slot_idx: usize,
    request_id: u64,
    draft_start_position: usize,
    original_target_len: usize,
    input_tokens: Vec<u32>,
    draft_tokens: Vec<u32>,
}

impl SpecPath {
    pub(super) fn draft_then_verify<M: ModelForward>(
        scheduler: &mut Scheduler<M>,
        force_sparse_view: Option<SparseDraftView>,
    ) {
        if matches!(scheduler.config.spec_draft_model, DraftMode::SelfSpec) {
            if scheduler.config.spec_sparse_kv_enabled {
                Self::draft_self_sparse_then_verify(scheduler, force_sparse_view);
                return;
            }
            if let Some(_view) = force_sparse_view {
                scheduler.step_decode_launch();
                return;
            }
            scheduler.step_spec_decode_launch_from_path();
            return;
        }
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
                    && req.speculative.as_ref().and_then(|spec| spec.enabled) != Some(false)
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

    fn draft_self_sparse_then_verify<M: ModelForward>(
        scheduler: &mut Scheduler<M>,
        force_sparse_view: Option<SparseDraftView>,
    ) {
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
            scheduler.step_decode_launch();
            return;
        }

        if !decode_indices.iter().all(|&slot_idx| {
            scheduler.request(slot_idx).is_some_and(|req| {
                !req.spec_decode_disabled
                    && req.speculative.as_ref().is_none_or(|spec| {
                        spec.allows_sparse_self_spec(scheduler.config.spec_draft_k)
                    })
                    && !req.has_stop_sequences()
                    && req.sampling.is_greedy()
                    && !req.sampling.has_penalties()
            })
        }) {
            scheduler.step_decode_launch();
            return;
        }

        if scheduler.decode_bufs.is_none() {
            match scheduler.model.create_decode_context(
                scheduler.states.len(),
                scheduler.effective_max_seq_len,
                &scheduler.paged_kv_pool,
            ) {
                Ok(ctx) => scheduler.decode_bufs = Some(ctx),
                Err(err) => {
                    log::warn!("sparse draft decode context init failed: {err}");
                    scheduler.step_decode_launch();
                    return;
                }
            }
        }

        let mut owned_views = match force_sparse_view {
            Some(view) => vec![view],
            None => build_sparse_draft_views(scheduler),
        };
        if owned_views.is_empty() {
            scheduler.step_decode_launch();
            return;
        }
        let views: HashMap<usize, SparseDraftView> = owned_views
            .drain(..)
            .map(|view| (view.slot_idx, view))
            .collect();
        if !decode_indices
            .iter()
            .all(|slot_idx| views.contains_key(slot_idx))
        {
            scheduler.step_decode_launch();
            return;
        }
        for view in views.values() {
            let _dropped_sparse_pages = scheduler.prefix_cache.drop_pages_for_sparse_view(
                view.slot_idx,
                &view.page_ids,
                &view.attached_page_ids,
            );
        }

        let mut rows = Vec::with_capacity(decode_indices.len());
        for (&slot_idx, &last_token) in decode_indices.iter().zip(&token_ids) {
            let view = views
                .get(&slot_idx)
                .expect("checked every decode slot has a sparse view");
            let Some((request_id, max_tokens, generated_len)) = scheduler
                .request(slot_idx)
                .map(|req| (req.id, req.max_tokens, req.generated_tokens.len()))
            else {
                continue;
            };

            let original_target_len = scheduler.paged_kv_pool.seq_len(slot_idx);
            let mut draft_tokens = Vec::with_capacity(scheduler.config.spec_draft_k);
            let mut token = last_token;
            let page_ids = match scheduler.flattened_pages_for_blocks(&view.page_ids) {
                Ok(pages) => pages,
                Err(err) => {
                    log::warn!(
                        "sparse draft page expansion failed for request {request_id}: {err}"
                    );
                    scheduler.step_decode_launch();
                    return;
                }
            };

            for _ in 0..scheduler.config.spec_draft_k {
                if generated_len + draft_tokens.len() >= max_tokens {
                    break;
                }
                if let Err(err) = scheduler
                    .paged_kv_pool
                    .cow_tail_page_for_append(scheduler.model.device_context(), slot_idx)
                    .and_then(|_| {
                        scheduler
                            .paged_kv_pool
                            .alloc_tokens(slot_idx, 1)
                            .map(|_| ())
                    })
                {
                    log::warn!("sparse draft KV allocation failed for request {request_id}: {err}");
                    let _ = scheduler
                        .paged_kv_pool
                        .truncate_slot(slot_idx, original_target_len);
                    scheduler.step_decode_launch();
                    return;
                }

                let sparse_view = SparseKvDraftView {
                    slot_idx,
                    page_ids: &page_ids,
                    active_recent_tokens: view.active_recent_tokens,
                };
                let Some(decode_ctx) = scheduler.decode_bufs.as_mut() else {
                    scheduler.step_decode_launch();
                    return;
                };
                match scheduler.model.forward_sparse_decode_with_logits(
                    token,
                    &mut scheduler.states,
                    slot_idx,
                    &mut scheduler.paged_kv_pool,
                    decode_ctx,
                    sparse_view,
                ) {
                    Ok(next_token) => {
                        draft_tokens.push(next_token);
                        token = next_token;
                    }
                    Err(err) => {
                        log::warn!("sparse draft forward failed for request {request_id}: {err}");
                        let _ = scheduler
                            .paged_kv_pool
                            .truncate_slot(slot_idx, original_target_len);
                        scheduler.step_decode_launch();
                        return;
                    }
                }
            }

            if let Err(err) = scheduler
                .paged_kv_pool
                .truncate_slot(slot_idx, original_target_len)
            {
                log::error!("sparse draft rollback failed for request {request_id}: {err}");
                scheduler.finish_slot(slot_idx);
                continue;
            }
            if draft_tokens.is_empty() {
                continue;
            }

            let mut input_tokens = Vec::with_capacity(draft_tokens.len() + 1);
            input_tokens.push(last_token);
            input_tokens.extend_from_slice(&draft_tokens);
            rows.push(SpecRow {
                slot_idx,
                request_id,
                draft_start_position: original_target_len,
                original_target_len,
                input_tokens,
                draft_tokens,
            });
        }

        verify_and_commit_rows(scheduler, rows, started);
    }
}

fn verify_and_commit_rows<M: ModelForward>(
    scheduler: &mut Scheduler<M>,
    rows: Vec<SpecRow>,
    started: std::time::Instant,
) {
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
            log::warn!("sparse spec verifier failed, falling back to normal decode: {err}");
            for row in &rows {
                let _ = scheduler
                    .paged_kv_pool
                    .truncate_slot(row.slot_idx, row.original_target_len);
            }
            scheduler.step_decode_launch();
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
            log::error!("sparse spec target KV rollback failed: {err}");
            scheduler.finish_slot(row.slot_idx);
            continue;
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

#[allow(dead_code)]
fn build_sparse_draft_views<M: ModelForward>(scheduler: &mut Scheduler<M>) -> Vec<SparseDraftView> {
    let block_size = scheduler.prefix_cache.block_size();
    let mut views = Vec::new();
    for slot_idx in 0..scheduler.active.len() {
        let Some(req) = scheduler.active[slot_idx].as_ref() else {
            continue;
        };
        if !matches!(req.phase, Phase::Decoding) {
            continue;
        }
        let mut tokens = Vec::with_capacity(req.prompt_tokens.len() + req.generated_tokens.len());
        tokens.extend_from_slice(&req.prompt_tokens);
        tokens.extend_from_slice(&req.generated_tokens);
        let page_ids = scheduler
            .prefix_cache
            .select_sparse_pages_for_draft_tokens_with_attached(
                slot_idx,
                &tokens,
                scheduler.config.spec_sparse_recent_tokens,
                scheduler.config.spec_sparse_top_k_pages,
                &req.attached_prefix_blocks,
            );
        let active_recent_tokens = scheduler.config.spec_sparse_recent_tokens.min(tokens.len());
        if page_ids.is_empty() {
            continue;
        }
        views.push(SparseDraftView::new(
            slot_idx,
            page_ids,
            req.attached_prefix_blocks.clone(),
            block_size,
            active_recent_tokens,
        ));
    }
    views
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_draft_view_counts_selected_block_tokens() {
        let view = SparseDraftView::new(3, vec![BlockId(7), BlockId(9)], vec![BlockId(5)], 16, 24);

        assert_eq!(view.slot_idx, 3);
        assert_eq!(view.page_ids, vec![BlockId(7), BlockId(9)]);
        assert_eq!(view.attached_page_ids, vec![BlockId(5)]);
        assert_eq!(view.sparse_total_tokens, 32);
        assert_eq!(view.active_recent_tokens, 24);
    }
}
