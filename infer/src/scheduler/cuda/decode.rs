use super::*;
use crate::scheduler::cuda::core::PendingDecode;

impl<M: ModelForward> Scheduler<M> {
    /// Batch all decode requests into a single GPU forward pass.
    pub(super) fn step_decode_launch(&mut self) {
        let decode_indices: Vec<usize> = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, r)| matches!(r.phase, Phase::Decoding) && !r.delta_tx.is_closed())
            .map(|(i, _)| i)
            .collect();

        for i in 0..self.active.len() {
            if matches!(self.active[i].phase, Phase::Decoding)
                && self.active[i].delta_tx.is_closed()
            {
                self.active[i].phase = Phase::Finished;
            }
        }

        if decode_indices.is_empty() {
            return;
        }

        let mut token_ids: Vec<u32> = Vec::with_capacity(decode_indices.len());
        let mut valid_decode_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        for &i in &decode_indices {
            match self.active[i].generated_tokens.last() {
                Some(&tok) => {
                    token_ids.push(tok);
                    valid_decode_indices.push(i);
                }
                None => {
                    error!(
                        "Request {}: Decoding state with no generated tokens - dropping",
                        self.active[i].id
                    );
                    self.active[i].phase = Phase::Finished;
                }
            }
        }
        let mut decode_indices = valid_decode_indices;
        if decode_indices.is_empty() {
            return;
        }
        // Preemption: if pool can't fit all decode requests, preempt the
        // request with the most generated tokens (highest KV cost, preserves
        // shorter conversations that are closer to completion).
        // Recompute mode: preempted request is re-queued and re-prefilled
        // when GPU memory frees up.
        while self.paged_kv_pool.is_active()
            && self.paged_kv_pool.free_count() < decode_indices.len()
            && decode_indices.len() > 1
        {
            // Pick victim: most generated tokens = highest KV cost
            let victim_pos = decode_indices
                .iter()
                .enumerate()
                .max_by_key(|(_, i)| self.active[**i].generated_tokens.len())
                .map(|(pos, _)| pos)
                .unwrap();
            let victim_idx = decode_indices[victim_pos];
            let victim = &mut self.active[victim_idx];
            let victim_slot = victim.slot_idx;
            warn!(
                "Request {}: preempting (recompute) — {} generated tokens, pool free={}",
                victim.id,
                victim.generated_tokens.len(),
                self.paged_kv_pool.free_count()
            );

            // Free victim's paged pool tokens and reset slot state
            self.paged_kv_pool.free_slot(victim_slot);
            if let Err(e) = self.states[victim_slot].reset() {
                error!(
                    "Request {}: slot reset after preempt failed: {}",
                    victim.id, e
                );
            }
            self.slot_materialized_prompt_lens[victim_slot] = 0;

            // Re-queue as waiting (will be re-prefilled when re-admitted).
            // NOTE: preempted request loses generated tokens (recompute mode).
            // The prompt is re-tokenized on re-admission via assign_slots().
            let requeue = IncomingRequest {
                prompt: std::mem::take(&mut victim.prompt),
                max_tokens: victim.max_tokens,
                sampling: victim.sampling.clone(),
                stop: victim.stop.take(),
                priority: RequestPriority::Normal,
                session_id: victim.session_id.clone(),
                delta_tx: victim.delta_tx.clone(),
            };
            victim.phase = Phase::Finished;
            let _ = victim;
            self.clear_slot_prefix_ownership(victim_slot);

            // Remove from decode batch
            decode_indices.remove(victim_pos);
            token_ids.remove(victim_pos);

            // Push to front of waiting queue (priority re-admission)
            self.waiting.push_front(requeue);
        }

        // Allocate one pool token per remaining decode request.
        let mut alloc_ok_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        let mut alloc_ok_tokens: Vec<u32> = Vec::with_capacity(decode_indices.len());
        for (j, &i) in decode_indices.iter().enumerate() {
            let slot = self.active[i].slot_idx;
            if let Err(e) = self.alloc_pool_tokens_with_retry(slot, 1) {
                error!(
                    "Request {}: KV pool exhausted after preemption (slot {}): {} — finishing",
                    self.active[i].id, slot, e
                );
                self.active[i].finish(FinishReason::Length, &self.tokenizer);
            } else {
                alloc_ok_indices.push(i);
                alloc_ok_tokens.push(token_ids[j]);
            }
        }
        let decode_indices = alloc_ok_indices;
        let token_ids = alloc_ok_tokens;
        if decode_indices.is_empty() {
            return;
        }
        let slot_indices: Vec<usize> = decode_indices
            .iter()
            .map(|&i| self.active[i].slot_idx)
            .collect();

        let sampling_params: Vec<&crate::sampler::SamplingParams> = decode_indices
            .iter()
            .map(|&i| &self.active[i].sampling)
            .collect();
        let all_greedy = sampling_params
            .iter()
            .all(|p| p.is_greedy() && !p.has_penalties());

        // Lazy-init decode context on first batched decode.
        if self.decode_bufs.is_none() {
            match self
                .model
                .create_decode_context(self.states.len(), &self.paged_kv_pool)
            {
                Ok(ctx) => self.decode_bufs = Some(ctx),
                Err(e) => {
                    error!("Failed to create decode context: {}", e);
                    for &i in &decode_indices {
                        self.active[i].phase = Phase::Finished;
                    }
                    return;
                }
            }
        }
        let decode_ctx = self.decode_bufs.as_mut().unwrap();

        // Pre-decode: scheduler handles H2D, metadata, and FlashInfer plan
        // via DecodeContextOps. This decouples scheduler-level work from
        // model-level neural computation.
        {
            use crate::model::DecodeContextOps;
            let ctx = self.model.device_context();
            decode_ctx.set_batch_size(token_ids.len());
            if let Err(e) = decode_ctx.upload_token_ids(ctx, &token_ids) {
                error!("Pre-decode upload_token_ids failed: {}", e);
                for &i in &decode_indices {
                    self.active[i].phase = Phase::Finished;
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
                    for &i in &decode_indices {
                        self.active[i].phase = Phase::Finished;
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
                for &i in &decode_indices {
                    self.active[i].phase = Phase::Finished;
                }
                return;
            }
        }

        let prefill_pending = self
            .active
            .iter()
            .any(|req| matches!(req.phase, Phase::New | Phase::Prefilling { .. }));
        if prefill_pending {
            use crate::model::DecodeContextOps;
            decode_ctx.invalidate_graph_cache(token_ids.len());
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
            for &i in &decode_indices {
                self.active[i].phase = Phase::Finished;
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
                        for &req_idx in &decode_indices {
                            self.active[req_idx].phase = Phase::Finished;
                        }
                        return;
                    }
                    false
                }
                Err(e) => {
                    error!("Batched greedy sampling launch failed: {}", e);
                    for &req_idx in &decode_indices {
                        self.active[req_idx].phase = Phase::Finished;
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
            all_greedy,
            greedy_launched,
            sampling_params_greedy: sampling_params
                .iter()
                .map(|p| p.is_greedy() && !p.has_penalties())
                .collect(),
        });
    }

    pub(super) fn step_decode_readback(&mut self) {
        let pending = match self.pending_decode.take() {
            Some(pending) => pending,
            None => return,
        };
        let decode_ctx = self.decode_bufs.as_mut().unwrap();

        let sampled_result = if pending.greedy_launched {
            // Argmax kernel was launched — sync + readback
            match self
                .model
                .sample_batch_greedy_readback(&pending.slot_indices, decode_ctx)
            {
                Ok(Some(tokens)) => Ok(tokens),
                Ok(None) => {
                    let params: Vec<&crate::sampler::SamplingParams> = pending
                        .decode_indices
                        .iter()
                        .zip(&pending.sampling_params_greedy)
                        .map(|(&i, _)| &self.active[i].sampling)
                        .collect();
                    self.model.select_tokens_batch(
                        &mut self.states,
                        &pending.slot_indices,
                        &params,
                        &mut self.rng,
                    )
                }
                Err(e) => Err(e),
            }
        } else {
            let params: Vec<&crate::sampler::SamplingParams> = pending
                .decode_indices
                .iter()
                .zip(&pending.sampling_params_greedy)
                .map(|(&i, _)| &self.active[i].sampling)
                .collect();
            self.model.select_tokens_batch(
                &mut self.states,
                &pending.slot_indices,
                &params,
                &mut self.rng,
            )
        };

        match sampled_result {
            Ok(sampled_tokens) => {
                let logprobs_host: Option<&[f32]> = if pending.greedy_launched {
                    Some(crate::model::DecodeContextOps::logprobs_host(&*decode_ctx))
                } else {
                    None
                };
                for (j, &req_idx) in pending.decode_indices.iter().enumerate() {
                    let token = sampled_tokens[j];
                    let req = &mut self.active[req_idx];
                    req.latest_logprob = logprobs_host.and_then(|lps| lps.get(j).copied());
                    if !req.sampling.ignore_eos && self.model.is_stop_token(token) {
                        req.finish(FinishReason::Stop, &self.tokenizer);
                        continue;
                    }
                    req.generated_tokens.push(token);
                    if matches!(req.phase, Phase::Finished) {
                        continue;
                    }
                    if req.generated_tokens.len() >= req.max_tokens {
                        req.finish(FinishReason::Length, &self.tokenizer);
                    }
                }
            }
            Err(e) => {
                error!("Batched sampling failed: {}", e);
                for &req_idx in &pending.decode_indices {
                    self.active[req_idx].phase = Phase::Finished;
                }
            }
        }
    }
}
