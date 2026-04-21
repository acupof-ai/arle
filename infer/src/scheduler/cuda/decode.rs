use super::{
    FinishReason, GenerationState, IncomingRequest, ModelForward, Phase, Scheduler, error, info,
    warn,
};
use crate::model::kv_cache::KVFormat;
use crate::scheduler::cuda::core::PendingDecode;

fn retract_victim_score(
    generated_tokens: usize,
    prompt_tokens: usize,
) -> (usize, std::cmp::Reverse<usize>) {
    (generated_tokens, std::cmp::Reverse(prompt_tokens))
}

fn expected_prefill_pool_seq(materialized_prefix_len: usize, progress: usize) -> usize {
    materialized_prefix_len.saturating_add(progress)
}

impl<M: ModelForward> Scheduler<M> {
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
                > self.pool_free_pages()
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
        {
            let Self {
                active, tokenizer, ..
            } = self;
            if let Some(req) = active[slot_idx].as_mut() {
                req.finish(reason, tokenizer);
            }
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
            if matches!(req.phase, Phase::Decoding) {
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
                max_tokens: victim.max_tokens,
                sampling: victim.sampling.clone(),
                stop: victim.stop.take(),
                priority: victim.priority,
                session_id: victim.session_id.clone(),
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
        self.waiting.push_front(requeue);
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
            match self
                .model
                .create_decode_context(self.states.len(), &self.paged_kv_pool)
            {
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

        {
            use crate::model::DecodeContextOps;
            let ctx = self.model.device_context();
            decode_ctx.set_batch_size(token_ids.len());
            if let Err(e) = decode_ctx.upload_token_ids(ctx, &token_ids) {
                error!("Pre-decode upload_token_ids failed: {}", e);
                for &slot_idx in &decode_indices {
                    self.finish_slot(slot_idx);
                }
                return;
            }
            match decode_ctx.update_metadata(ctx, &self.paged_kv_pool, &slot_indices) {
                Ok(reallocated) => {
                    if reallocated {
                        decode_ctx.invalidate_graph_cache(token_ids.len());
                    }
                }
                Err(e) => {
                    error!("Pre-decode update_metadata failed: {}", e);
                    for &slot_idx in &decode_indices {
                        self.finish_slot(slot_idx);
                    }
                    return;
                }
            }
            if let Err(e) = decode_ctx.plan_attention(
                ctx,
                token_ids.len(),
                self.model.num_q_heads(),
                self.model.num_kv_heads(),
                self.paged_kv_pool.page_size,
                self.model.head_dim(),
                self.paged_kv_pool.format,
            ) {
                error!("Pre-decode plan_attention failed: {}", e);
                for &slot_idx in &decode_indices {
                    self.finish_slot(slot_idx);
                }
                return;
            }
        }

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

        self.pending_decode = Some(PendingDecode {
            decode_indices,
            slot_indices,
            greedy_launched,
            mixed_prefill_request_idx: None,
            mixed_prefill_chunk_complete: false,
        });
    }

    pub(super) fn step_decode_launch_mixed(&mut self, prefill_slot_idx: usize) {
        let decode_indices = self.running_decode_slots();

        if decode_indices.is_empty()
            || !self.model.supports_mixed_batch()
            || self.paged_kv_pool.format != KVFormat::BF16
        {
            self.step_decode_launch();
            return;
        }

        let mut token_ids: Vec<u32> = Vec::with_capacity(decode_indices.len());
        let mut valid_decode_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
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
        let mut decode_indices = valid_decode_indices;
        if decode_indices.is_empty() {
            return;
        }

        let (prefill_total, prefill_progress, prefill_tokens) = match self.request(prefill_slot_idx)
        {
            Some(req) if !req.delta_tx.is_closed() => {
                if let Phase::Prefilling {
                    materialized_prefix_len,
                    effective_tokens,
                    progress,
                } = &req.phase
                {
                    let pool_seq = self.paged_kv_pool.seq_len(prefill_slot_idx);
                    let expected_pool_seq =
                        expected_prefill_pool_seq(*materialized_prefix_len, *progress);
                    if *progress >= effective_tokens.len() || pool_seq != expected_pool_seq {
                        self.step_decode_launch();
                        return;
                    }
                    let mixed_chunk_size = self
                        .prefill_chunk_size()
                        .min(self.config.max_prefill_tokens.max(1));
                    let end = (*progress + mixed_chunk_size).min(effective_tokens.len());
                    (
                        effective_tokens.len(),
                        *progress,
                        effective_tokens[*progress..end].to_vec(),
                    )
                } else {
                    self.step_decode_launch();
                    return;
                }
            }
            _ => {
                self.step_decode_launch();
                return;
            }
        };

        let prefill_token_count = prefill_tokens.len();
        if prefill_token_count == 0 {
            self.step_decode_launch();
            return;
        }

        let mixed_prefill_pages =
            self.additional_pages_needed_for_slot(prefill_slot_idx, prefill_token_count);
        self.retract_decode_to_fit(&mut decode_indices, &mut token_ids, mixed_prefill_pages);

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
        let decode_indices = alloc_ok_indices;
        let token_ids = alloc_ok_tokens;
        if decode_indices.is_empty() {
            return;
        }

        let slot_indices = decode_indices.clone();
        let sampling_params_greedy: Vec<bool> = decode_indices
            .iter()
            .filter_map(|&slot_idx| {
                self.request(slot_idx).map(|req| {
                    let p = &req.sampling;
                    p.is_greedy() && !p.has_penalties()
                })
            })
            .collect();
        let all_greedy = sampling_params_greedy.iter().all(|&g| g);

        if self.decode_bufs.is_none() {
            match self
                .model
                .create_decode_context(self.states.len(), &self.paged_kv_pool)
            {
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

        let forward_result = self.model.forward_mixed_batch(
            &token_ids,
            &prefill_tokens,
            &mut self.states,
            &slot_indices,
            prefill_slot_idx,
            prefill_progress,
            Some(&mut self.paged_kv_pool),
            decode_ctx,
        );

        let mixed_ok = match forward_result {
            Ok(true) => {
                info!(
                    "Mixed batch: B={} decode + C={} prefill (slot {})",
                    token_ids.len(),
                    prefill_token_count,
                    prefill_slot_idx
                );
                true
            }
            Ok(false) => {
                info!("Mixed batch: fallback (Ok(false))");
                self.launch_decode_batch_from_tokens(decode_indices, token_ids, true);
                return;
            }
            Err(e) => {
                error!(
                    "Mixed batched decode failed after decode-token allocation: {}",
                    e
                );
                for &slot_idx in &decode_indices {
                    self.finish_slot(slot_idx);
                }
                self.finish_slot(prefill_slot_idx);
                return;
            }
        };
        debug_assert!(mixed_ok);

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

        let new_progress = prefill_progress + prefill_token_count;
        if let Some(req) = self.request_mut(prefill_slot_idx)
            && let Phase::Prefilling { progress, .. } = &mut req.phase
        {
            *progress = new_progress;
        }

        self.pending_decode = Some(PendingDecode {
            decode_indices,
            slot_indices,
            greedy_launched,
            mixed_prefill_request_idx: Some(prefill_slot_idx),
            mixed_prefill_chunk_complete: new_progress >= prefill_total,
        });
    }

    /// Batch all decode requests into a single GPU forward pass.
    pub(super) fn step_decode_launch(&mut self) {
        let decode_indices = self.running_decode_slots();

        if decode_indices.is_empty() {
            return;
        }

        let mut token_ids: Vec<u32> = Vec::with_capacity(decode_indices.len());
        let mut valid_decode_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
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
        let mut decode_indices = valid_decode_indices;
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

    pub(super) fn step_decode_readback(&mut self) {
        let Some(pending) = self.pending_decode.take() else {
            return;
        };
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
                        self.finish_request(slot_idx, FinishReason::Length);
                    }
                }
            }
            Err(e) => {
                error!("Batched sampling failed: {}", e);
                for &slot_idx in &pending.decode_indices {
                    self.finish_slot(slot_idx);
                }
            }
        }

        if let Some(prefill_slot_idx) = pending.mixed_prefill_request_idx {
            if pending.mixed_prefill_chunk_complete {
                let req_id = self
                    .request(prefill_slot_idx)
                    .map(|req| req.id)
                    .unwrap_or_default();
                if let Err(e) = self.states[prefill_slot_idx].save_prefix_snapshot() {
                    warn!(
                        "Request {}: save prefix snapshot failed: {} (prefix cache disabled for this slot)",
                        req_id, e
                    );
                } else if let Some(req) = self.request_mut(prefill_slot_idx) {
                    req.mark_prompt_cacheable();
                }

                let sampling = self
                    .request(prefill_slot_idx)
                    .map(|req| req.sampling.clone())
                    .expect("mixed prefill completion requires live request");
                match self.model.select_token(
                    &mut self.states[prefill_slot_idx],
                    &sampling,
                    &mut self.rng,
                ) {
                    Ok(token) => {
                        let ignore_eos = self
                            .request(prefill_slot_idx)
                            .is_some_and(|req| req.sampling.ignore_eos);
                        if !ignore_eos && self.model.is_stop_token(token) {
                            self.finish_request(prefill_slot_idx, FinishReason::Stop);
                            return;
                        }
                        {
                            let Self {
                                active, tokenizer, ..
                            } = self;
                            if let Some(req) = active[prefill_slot_idx].as_mut() {
                                req.generated_tokens.push(token);
                                req.emit_delta(tokenizer);
                            }
                        }

                        if matches!(
                            self.request(prefill_slot_idx).map(|req| &req.phase),
                            Some(Phase::Finished)
                        ) {
                            self.finish_slot(prefill_slot_idx);
                            return;
                        }
                        let reached_max = self
                            .request(prefill_slot_idx)
                            .is_some_and(|req| req.generated_tokens.len() >= req.max_tokens);
                        if reached_max {
                            self.finish_request(prefill_slot_idx, FinishReason::Length);
                        } else {
                            if let Some(req) = self.request_mut(prefill_slot_idx)
                                && req.first_token_at.is_none()
                            {
                                req.first_token_at = Some(std::time::Instant::now());
                            }
                            self.move_to_decode(prefill_slot_idx);
                        }
                    }
                    Err(e) => {
                        error!(
                            "Request {}: select_token failed after mixed prefill: {}",
                            req_id, e
                        );
                        self.finish_slot(prefill_slot_idx);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{expected_prefill_pool_seq, retract_victim_score};

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
    fn mixed_prefill_progress_includes_materialized_prefix_base() {
        assert_eq!(expected_prefill_pool_seq(0, 3), 3);
        assert_eq!(expected_prefill_pool_seq(15, 0), 15);
        assert_eq!(expected_prefill_pool_seq(15, 4), 19);
    }
}
