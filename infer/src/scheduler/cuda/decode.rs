use super::{
    FinishReason, GenerationState, IncomingRequest, ModelForward, Phase, Scheduler, error, warn,
};
use crate::model::{MixedBatchRequest, PrefillBatchRequest};
use crate::scheduler::cuda::core::{
    PendingDecode, PendingMixedPrefill, PendingPrefill, PendingPrefillRow,
};
use crate::scheduler::cuda::execution::PrefillCandidate;
use crate::scheduler::cuda::runtime::WaitingInsertBias;

fn retract_victim_score(
    generated_tokens: usize,
    prompt_tokens: usize,
) -> (usize, std::cmp::Reverse<usize>) {
    (generated_tokens, std::cmp::Reverse(prompt_tokens))
}

#[cfg(test)]
fn mixed_prefill_pages_needed(seq_len: usize, prefill_tokens: usize, page_size: usize) -> usize {
    super::budget::additional_pages_needed(seq_len, prefill_tokens, page_size)
}

impl<M: ModelForward> Scheduler<M> {
    fn collect_decode_batch_inputs(&mut self) -> (Vec<usize>, Vec<u32>) {
        let decode_indices = self.running_decode_slots();
        let mut token_ids = Vec::with_capacity(decode_indices.len());
        let mut valid_decode_indices = Vec::with_capacity(decode_indices.len());
        for &slot_idx in &decode_indices {
            if let Some(&tok) = self
                .request(slot_idx)
                .and_then(|req| req.generated_tokens.last())
            {
                token_ids.push(tok);
                valid_decode_indices.push(slot_idx);
            } else {
                let req_id = self.request(slot_idx).map(|req| req.id).unwrap_or_default();
                error!(
                    "Request {}: Decoding state with no generated tokens - dropping",
                    req_id
                );
                self.finish_slot(slot_idx);
            }
        }
        (valid_decode_indices, token_ids)
    }

    fn allocate_decode_tokens(
        &mut self,
        decode_indices: &[usize],
        token_ids: &[u32],
    ) -> (Vec<usize>, Vec<u32>) {
        let mut alloc_ok_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        let mut alloc_ok_tokens: Vec<u32> = Vec::with_capacity(decode_indices.len());
        for (j, &slot_idx) in decode_indices.iter().enumerate() {
            if let Err(e) = self.alloc_pool_tokens_with_retry(slot_idx, 1) {
                let req_id = self.request(slot_idx).map(|req| req.id).unwrap_or_default();
                error!(
                    "Request {}: KV pool exhausted after preemption (slot {}): {} — finishing",
                    req_id, slot_idx, e
                );
                self.finish_request(slot_idx, FinishReason::Length);
            } else {
                alloc_ok_indices.push(slot_idx);
                alloc_ok_tokens.push(token_ids[j]);
            }
        }
        (alloc_ok_indices, alloc_ok_tokens)
    }

    fn decode_pages_needed(&self, slot_indices: &[usize]) -> usize {
        slot_indices
            .iter()
            .map(|&slot_idx| self.additional_pages_needed_for_slot(slot_idx, 1))
            .sum()
    }

    fn retract_victim_pos(&self, decode_indices: &[usize]) -> Option<usize> {
        decode_indices
            .iter()
            .enumerate()
            .min_by_key(|(_, slot_idx)| {
                self.request(**slot_idx)
                    .map_or((usize::MAX, std::cmp::Reverse(0)), |req| {
                        retract_victim_score(req.generated_tokens.len(), req.prompt_tokens.len())
                    })
            })
            .map(|(pos, _)| pos)
    }

    fn retract_decode_to_fit(
        &mut self,
        decode_indices: &mut Vec<usize>,
        token_ids: &mut Vec<u32>,
        extra_pages: usize,
    ) {
        while self.paged_kv_pool.is_active()
            && self
                .decode_pages_needed(decode_indices)
                .saturating_add(extra_pages)
                > self.effective_pool_free_pages()
            && decode_indices.len() > 1
        {
            let Some(victim_pos) = self.retract_victim_pos(decode_indices) else {
                break;
            };
            let victim_idx = decode_indices[victim_pos];
            self.requeue_preempted_decode(victim_idx);
            decode_indices.remove(victim_pos);
            token_ids.remove(victim_pos);
        }
    }

    pub(super) fn finish_request(&mut self, slot_idx: usize, reason: FinishReason) {
        self.queue_emit_finish(slot_idx, reason);
        if let Some(req) = self.request_mut(slot_idx) {
            req.phase = Phase::Finished;
        }
        self.finish_slot(slot_idx);
    }

    fn running_decode_slots(&mut self) -> Vec<usize> {
        let queued: Vec<usize> = self.running_batch.iter().copied().collect();
        let mut decode_slots = Vec::with_capacity(queued.len());
        for slot_idx in queued {
            let Some(req) = self.request(slot_idx) else {
                self.dequeue_running(slot_idx);
                continue;
            };
            if req.delta_tx.is_closed() {
                self.finish_slot(slot_idx);
                continue;
            }
            if self.slot_is_runnable_decode(slot_idx) {
                decode_slots.push(slot_idx);
            }
        }
        decode_slots
    }

    fn requeue_preempted_decode(&mut self, slot_idx: usize) {
        let (victim_id, generated_tokens, requeue) = {
            let victim = self
                .request_mut(slot_idx)
                .expect("preempted decode slot must hold a request");
            let generated_tokens = victim.generated_tokens.len();
            let requeue = IncomingRequest {
                prompt: std::mem::take(&mut victim.prompt),
                prompt_tokens: Some(std::mem::take(&mut victim.prompt_tokens)),
                max_tokens: victim.max_tokens,
                sampling: victim.sampling.clone(),
                stop: victim.stop.take(),
                priority: victim.priority,
                session_id: victim.session_id.clone(),
                trace_context: victim.trace_context,
                delta_tx: victim.delta_tx.clone(),
            };
            victim.phase = Phase::Finished;
            (victim.id, generated_tokens, requeue)
        };

        warn!(
            "Request {}: preempting (recompute) — {} generated tokens, pool free={}",
            victim_id,
            generated_tokens,
            self.paged_kv_pool.free_count()
        );
        self.paged_kv_pool.free_slot(slot_idx);
        if let Err(e) = self.states[slot_idx].reset() {
            error!(
                "Request {}: slot reset after preempt failed: {}",
                victim_id, e
            );
        }
        self.slot_materialized_prompt_lens[slot_idx] = 0;
        self.clear_slot_prefix_ownership(slot_idx);
        self.finish_slot(slot_idx);
        self.enqueue_waiting_request(requeue, WaitingInsertBias::BeforeEqual);
    }

    fn queue_pending_decode_launch(
        &mut self,
        decode_indices: Vec<usize>,
        slot_indices: Vec<usize>,
        greedy_launched: bool,
        mixed_prefill: Option<PendingMixedPrefill>,
    ) {
        let batch_size = decode_indices.len();
        let decode_spans = decode_indices
            .iter()
            .filter_map(|&slot_idx| {
                self.request(slot_idx).and_then(|req| {
                    req.begin_trace_span("decode_loop").map(|span| {
                        (
                            slot_idx,
                            span.with_properties(|| {
                                [
                                    ("slot_idx", slot_idx.to_string()),
                                    ("batch_size", batch_size.to_string()),
                                ]
                            }),
                        )
                    })
                })
            })
            .collect();

        self.pending_decode = Some(PendingDecode {
            decode_indices,
            slot_indices,
            greedy_launched,
            decode_spans,
            mixed_prefill,
        });
    }

    fn launch_decode_batch_from_tokens(
        &mut self,
        mut decode_indices: Vec<usize>,
        token_ids: Vec<u32>,
        decode_tokens_already_allocated: bool,
    ) {
        if decode_indices.is_empty() {
            return;
        }

        let token_ids = if decode_tokens_already_allocated {
            token_ids
        } else {
            let (alloc_ok_indices, alloc_ok_tokens) =
                self.allocate_decode_tokens(&decode_indices, &token_ids);
            decode_indices = alloc_ok_indices;
            alloc_ok_tokens
        };

        if decode_indices.is_empty() {
            return;
        }

        let slot_indices = decode_indices.clone();
        let sampling_params: Vec<crate::sampler::SamplingParams> = decode_indices
            .iter()
            .filter_map(|&slot_idx| self.request(slot_idx).map(|req| req.sampling.clone()))
            .collect();
        let all_greedy = sampling_params
            .iter()
            .all(|p| p.is_greedy() && !p.has_penalties());

        if self.decode_bufs.is_none() {
            match self.model.create_decode_context(
                self.states.len(),
                self.effective_max_seq_len,
                &self.paged_kv_pool,
            ) {
                Ok(ctx) => self.decode_bufs = Some(ctx),
                Err(e) => {
                    error!("Failed to create decode context: {}", e);
                    for &slot_idx in &decode_indices {
                        self.finish_slot(slot_idx);
                    }
                    return;
                }
            }
        }
        let decode_ctx = self.decode_bufs.as_mut().unwrap();

        let forward_result = self.model.forward_decode_batch(
            &token_ids,
            &mut self.states,
            &slot_indices,
            Some(&mut self.paged_kv_pool),
            decode_ctx,
            all_greedy,
        );

        if let Err(e) = forward_result {
            error!("Batched decode failed: {}", e);
            for &slot_idx in &decode_indices {
                self.finish_slot(slot_idx);
            }
            return;
        }

        let greedy_launched = if all_greedy {
            match self
                .model
                .sample_batch_greedy_launch(&slot_indices, decode_ctx)
            {
                Ok(true) => true,
                Ok(false) => {
                    if let Err(e) = self.model.prepare_batch_sampling_fallback(
                        &mut self.states,
                        &slot_indices,
                        decode_ctx,
                    ) {
                        error!("Preparing batched sampling fallback failed: {}", e);
                        for &slot_idx in &decode_indices {
                            self.finish_slot(slot_idx);
                        }
                        return;
                    }
                    false
                }
                Err(e) => {
                    error!("Batched greedy sampling launch failed: {}", e);
                    for &slot_idx in &decode_indices {
                        self.finish_slot(slot_idx);
                    }
                    return;
                }
            }
        } else {
            false
        };

        self.queue_pending_decode_launch(decode_indices, slot_indices, greedy_launched, None);
    }

    /// Batch all decode requests into a single GPU forward pass.
    pub(super) fn step_decode_launch(&mut self) {
        let (mut decode_indices, mut token_ids) = self.collect_decode_batch_inputs();
        if decode_indices.is_empty() {
            return;
        }
        // Preemption: if pool can't fit all decode requests, retract the
        // least-progressed request and, on ties, the one with the longer
        // prompt. This matches sglang's default decode retract heuristic.
        // Recompute mode: preempted request is re-queued and re-prefilled
        // when GPU memory frees up.
        self.retract_decode_to_fit(&mut decode_indices, &mut token_ids, 0);

        self.launch_decode_batch_from_tokens(decode_indices, token_ids, false);
    }

    pub(super) fn step_mixed_launch(&mut self, candidates: &[PrefillCandidate]) {
        if candidates.is_empty() {
            self.step_decode_launch();
            return;
        }

        let (mut decode_indices, mut token_ids) = self.collect_decode_batch_inputs();
        if decode_indices.is_empty() {
            self.step_prefill_batch(candidates);
            return;
        }

        let mut launch_candidates;
        loop {
            launch_candidates =
                self.select_mixed_launch_prefill_candidates(candidates, &decode_indices);
            if launch_candidates.is_empty() {
                self.launch_decode_batch_from_tokens(decode_indices, token_ids, false);
                return;
            }

            let extra_pages = launch_candidates
                .iter()
                .map(|candidate| {
                    self.additional_pages_needed_for_slot(
                        candidate.slot_idx,
                        candidate.reservation.prefill_tokens,
                    )
                })
                .sum();
            let before_retract = decode_indices.len();
            self.retract_decode_to_fit(&mut decode_indices, &mut token_ids, extra_pages);
            if decode_indices.is_empty() {
                self.step_prefill_batch(&launch_candidates);
                return;
            }
            if decode_indices.len() == before_retract {
                break;
            }
        }

        let (alloc_ok_indices, alloc_ok_tokens) =
            self.allocate_decode_tokens(&decode_indices, &token_ids);
        let decode_indices = alloc_ok_indices;
        let token_ids = alloc_ok_tokens;
        if decode_indices.is_empty() {
            self.step_prefill_batch(&launch_candidates);
            return;
        }

        launch_candidates =
            self.select_mixed_launch_prefill_candidates(candidates, &decode_indices);
        if launch_candidates.is_empty() {
            self.launch_decode_batch_from_tokens(decode_indices, token_ids, true);
            return;
        }

        let mut pending_rows = Vec::with_capacity(launch_candidates.len());
        let mut prefill_chunks = Vec::with_capacity(launch_candidates.len());
        let mut prefill_start_positions = Vec::with_capacity(launch_candidates.len());
        for candidate in &launch_candidates {
            let slot_idx = candidate.slot_idx;
            let Some(req) = self.request(slot_idx) else {
                continue;
            };
            if req.delta_tx.is_closed() {
                self.finish_slot(slot_idx);
                continue;
            }
            let (prefill_tokens, progress, total_tokens) = if let Phase::Prefilling {
                effective_tokens,
                progress,
            } = &req.phase
            {
                let total = effective_tokens.len();
                let chunk_end = (*progress + candidate.reservation.prefill_tokens).min(total);
                (
                    effective_tokens[*progress..chunk_end].to_vec(),
                    *progress,
                    total,
                )
            } else {
                continue;
            };
            if prefill_tokens.is_empty() {
                self.dequeue_prefill(slot_idx);
                continue;
            }
            pending_rows.push(PendingPrefillRow {
                slot_idx,
                total_tokens,
                next_progress: progress + prefill_tokens.len(),
            });
            prefill_start_positions.push(progress);
            prefill_chunks.push((slot_idx, prefill_tokens));
        }
        if prefill_chunks.is_empty() {
            self.launch_decode_batch_from_tokens(decode_indices, token_ids, true);
            return;
        }

        if self.decode_bufs.is_none() {
            match self.model.create_decode_context(
                self.states.len(),
                self.effective_max_seq_len,
                &self.paged_kv_pool,
            ) {
                Ok(ctx) => self.decode_bufs = Some(ctx),
                Err(e) => {
                    error!("Failed to create decode context: {}", e);
                    for row in &pending_rows {
                        self.finish_slot(row.slot_idx);
                    }
                    for &slot_idx in &decode_indices {
                        self.finish_slot(slot_idx);
                    }
                    return;
                }
            }
        }

        let slot_indices = decode_indices.clone();
        let batch_size = decode_indices.len() + prefill_chunks.len();
        let prefill_spans: Vec<(usize, fastrace::Span)> = prefill_chunks
            .iter()
            .filter_map(|(slot_idx, tokens)| {
                self.request(*slot_idx).and_then(|req| {
                    req.begin_trace_span("prefill").map(|span| {
                        (
                            *slot_idx,
                            span.with_properties(|| {
                                [
                                    ("slot_idx", slot_idx.to_string()),
                                    ("chunk_tokens", tokens.len().to_string()),
                                    ("batch_size", batch_size.to_string()),
                                ]
                            }),
                        )
                    })
                })
            })
            .collect();
        for (slot_idx, _) in &prefill_chunks {
            self.dequeue_prefill(*slot_idx);
        }

        let sampling_params: Vec<crate::sampler::SamplingParams> = decode_indices
            .iter()
            .filter_map(|&slot_idx| self.request(slot_idx).map(|req| req.sampling.clone()))
            .collect();
        let all_greedy = sampling_params
            .iter()
            .all(|params| params.is_greedy() && !params.has_penalties());
        let decode_ctx = self
            .decode_bufs
            .as_mut()
            .expect("decode context initialized before mixed launch");
        let prefills: Vec<PrefillBatchRequest<'_>> = prefill_chunks
            .iter()
            .map(|(slot_idx, tokens)| PrefillBatchRequest {
                slot_idx: *slot_idx,
                tokens,
            })
            .collect();
        let mixed_batch = MixedBatchRequest {
            decode_tokens: &token_ids,
            decode_slot_indices: &slot_indices,
            prefills: &prefills,
            prefill_start_positions: &prefill_start_positions,
        };
        match self.model.forward_mixed_batch(
            mixed_batch,
            &mut self.states,
            Some(&mut self.paged_kv_pool),
            decode_ctx,
        ) {
            Ok(true) => {}
            Ok(false) => {
                self.step_prefill_batch(&launch_candidates);
                for row in &pending_rows {
                    if self
                        .request(row.slot_idx)
                        .is_some_and(|req| matches!(req.phase, Phase::Prefilling { .. }))
                        && !self.slot_has_pending_gpu_work(row.slot_idx)
                    {
                        self.queue_prefill(row.slot_idx);
                    }
                }
                self.launch_decode_batch_from_tokens(decode_indices, token_ids, true);
                return;
            }
            Err(e) => {
                error!("Mixed batch launch failed: {}", e);
                for row in &pending_rows {
                    self.finish_slot(row.slot_idx);
                }
                for &slot_idx in &decode_indices {
                    self.finish_slot(slot_idx);
                }
                return;
            }
        }
        let greedy_launched = if all_greedy {
            match self
                .model
                .sample_batch_greedy_launch(&slot_indices, decode_ctx)
            {
                Ok(true) => true,
                Ok(false) => {
                    if let Err(e) = self.model.prepare_batch_sampling_fallback(
                        &mut self.states,
                        &slot_indices,
                        decode_ctx,
                    ) {
                        error!("Preparing batched sampling fallback failed: {}", e);
                        for row in &pending_rows {
                            self.finish_slot(row.slot_idx);
                        }
                        for &slot_idx in &decode_indices {
                            self.finish_slot(slot_idx);
                        }
                        return;
                    }
                    false
                }
                Err(e) => {
                    error!("Batched greedy sampling launch failed: {}", e);
                    for row in &pending_rows {
                        self.finish_slot(row.slot_idx);
                    }
                    for &slot_idx in &decode_indices {
                        self.finish_slot(slot_idx);
                    }
                    return;
                }
            }
        } else {
            if let Err(e) = self.model.prepare_batch_sampling_fallback(
                &mut self.states,
                &slot_indices,
                decode_ctx,
            ) {
                error!("Preparing mixed-batch sampling fallback failed: {}", e);
                for row in &pending_rows {
                    self.finish_slot(row.slot_idx);
                }
                for &slot_idx in &decode_indices {
                    self.finish_slot(slot_idx);
                }
                return;
            }
            false
        };

        self.queue_pending_decode_launch(
            decode_indices,
            slot_indices,
            greedy_launched,
            Some(PendingMixedPrefill {
                rows: pending_rows,
                uses_paged: self.model.prefill_uses_paged_pool() && self.paged_kv_pool.is_active(),
                prefill_spans,
            }),
        );
    }

    pub(super) fn step_decode_readback(&mut self) {
        let Some(pending) = self.pending_decode.take() else {
            return;
        };
        let decode_trace_contexts: std::collections::HashMap<
            usize,
            fastrace::collector::SpanContext,
        > = pending
            .decode_spans
            .iter()
            .filter_map(|(slot_idx, span)| {
                fastrace::collector::SpanContext::from_span(span)
                    .map(|context| (*slot_idx, context))
            })
            .collect();
        let sampling_params: Vec<crate::sampler::SamplingParams> = pending
            .decode_indices
            .iter()
            .filter_map(|&slot_idx| self.request(slot_idx).map(|req| req.sampling.clone()))
            .collect();
        let sampling_refs: Vec<&crate::sampler::SamplingParams> = sampling_params.iter().collect();

        let (sampled_result, logprobs_host) = if pending.greedy_launched {
            // Argmax kernel was launched — sync + readback.
            let decode_ctx = self.decode_bufs.as_mut().unwrap();
            match self
                .model
                .sample_batch_greedy_readback(&pending.slot_indices, decode_ctx)
            {
                Ok(Some(tokens)) => (
                    Ok(tokens),
                    Some(crate::model::DecodeContextOps::logprobs_host(&*decode_ctx).to_vec()),
                ),
                Ok(None) => (
                    self.model.select_tokens_batch(
                        &mut self.states,
                        &pending.slot_indices,
                        &sampling_refs,
                        &mut self.rng,
                    ),
                    None,
                ),
                Err(e) => (Err(e), None),
            }
        } else {
            (
                self.model.select_tokens_batch(
                    &mut self.states,
                    &pending.slot_indices,
                    &sampling_refs,
                    &mut self.rng,
                ),
                None,
            )
        };

        match sampled_result {
            Ok(sampled_tokens) => {
                for (j, &slot_idx) in pending.decode_indices.iter().enumerate() {
                    if let Some(req) = self.request_mut(slot_idx) {
                        req.trace_context = decode_trace_contexts
                            .get(&slot_idx)
                            .copied()
                            .or(req.trace_context);
                    }
                    let token = sampled_tokens[j];
                    if let Some(req) = self.request_mut(slot_idx) {
                        req.latest_logprob =
                            logprobs_host.as_ref().and_then(|lps| lps.get(j).copied());
                    }
                    let ignore_eos = self
                        .request(slot_idx)
                        .is_some_and(|req| req.sampling.ignore_eos);
                    if !ignore_eos && self.model.is_stop_token(token) {
                        self.finish_request(slot_idx, FinishReason::Stop);
                        continue;
                    }
                    if let Some(req) = self.request_mut(slot_idx) {
                        req.generated_tokens.push(token);
                    }
                    if matches!(
                        self.request(slot_idx).map(|req| &req.phase),
                        Some(Phase::Finished)
                    ) {
                        self.finish_slot(slot_idx);
                        continue;
                    }
                    let reached_max = self
                        .request(slot_idx)
                        .is_some_and(|req| req.generated_tokens.len() >= req.max_tokens);
                    if reached_max {
                        if !self.defer_finish_until_emit_gate(slot_idx, FinishReason::Length) {
                            self.finish_request(slot_idx, FinishReason::Length);
                        }
                    }
                }

                if let Some(mixed_prefill) = pending.mixed_prefill {
                    self.finish_prefill_batch(PendingPrefill {
                        rows: mixed_prefill.rows,
                        uses_paged: mixed_prefill.uses_paged,
                        prefill_spans: mixed_prefill.prefill_spans,
                    });
                }
            }
            Err(e) => {
                error!("Batched sampling failed: {}", e);
                for &slot_idx in &pending.decode_indices {
                    self.finish_slot(slot_idx);
                }
                if let Some(mixed_prefill) = pending.mixed_prefill {
                    for row in mixed_prefill.rows {
                        self.finish_slot(row.slot_idx);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{mixed_prefill_pages_needed, retract_victim_score};

    #[test]
    fn retract_prefers_less_progress_even_if_other_prompt_is_shorter() {
        assert!(
            retract_victim_score(2, 64) < retract_victim_score(5, 1024),
            "fewer generated tokens must retract first",
        );
    }

    #[test]
    fn retract_prefers_longer_prompt_when_progress_ties() {
        assert!(
            retract_victim_score(3, 1024) < retract_victim_score(3, 128),
            "when decode progress ties, the longer prompt must retract first",
        );
    }

    #[test]
    fn mixed_prefill_retract_budget_counts_pages_not_tokens() {
        assert_eq!(mixed_prefill_pages_needed(0, 16, 16), 1);
        assert_eq!(mixed_prefill_pages_needed(8, 4, 16), 0);
        assert_eq!(mixed_prefill_pages_needed(8, 12, 16), 1);
    }
}
