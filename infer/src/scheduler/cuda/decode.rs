use super::*;

impl<M: ModelForward> Scheduler<M> {
    /// Batch all decode requests into a single GPU forward pass.
    pub(super) fn step_decode_batch(&mut self) {
        let Self {
            model,
            tokenizer,
            states,
            active,
            rng,
            paged_kv_pool,
            decode_bufs,
            waiting,
            cached_prompts,
            ..
        } = self;

        let decode_indices: Vec<usize> = active
            .iter()
            .enumerate()
            .filter(|(_, r)| matches!(r.phase, Phase::Decoding) && !r.delta_tx.is_closed())
            .map(|(i, _)| i)
            .collect();

        for i in 0..active.len() {
            if matches!(active[i].phase, Phase::Decoding) && active[i].delta_tx.is_closed() {
                active[i].phase = Phase::Finished;
            }
        }

        if decode_indices.is_empty() {
            return;
        }

        let mut token_ids: Vec<u32> = Vec::with_capacity(decode_indices.len());
        let mut valid_decode_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        for &i in &decode_indices {
            match active[i].generated_tokens.last() {
                Some(&tok) => {
                    token_ids.push(tok);
                    valid_decode_indices.push(i);
                }
                None => {
                    error!(
                        "Request {}: Decoding state with no generated tokens - dropping",
                        active[i].id
                    );
                    active[i].phase = Phase::Finished;
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
        while paged_kv_pool.is_active()
            && paged_kv_pool.free_count() < decode_indices.len()
            && decode_indices.len() > 1
        {
            // Pick victim: most generated tokens = highest KV cost
            let victim_pos = decode_indices
                .iter()
                .enumerate()
                .max_by_key(|(_, i)| active[**i].generated_tokens.len())
                .map(|(pos, _)| pos)
                .unwrap();
            let victim_idx = decode_indices[victim_pos];
            let victim = &mut active[victim_idx];
            warn!(
                "Request {}: preempting (recompute) — {} generated tokens, pool free={}",
                victim.id,
                victim.generated_tokens.len(),
                paged_kv_pool.free_count()
            );

            // Free victim's paged pool tokens and reset slot state
            paged_kv_pool.free_slot(victim.slot_idx);
            if let Err(e) = states[victim.slot_idx].reset() {
                error!(
                    "Request {}: slot reset after preempt failed: {}",
                    victim.id, e
                );
            }
            cached_prompts[victim.slot_idx].clear();

            // Re-queue as waiting (will be re-prefilled when re-admitted).
            // NOTE: preempted request loses generated tokens (recompute mode).
            // The prompt is re-tokenized on re-admission via assign_slots().
            let requeue = IncomingRequest {
                prompt: std::mem::take(&mut victim.prompt),
                max_tokens: victim.max_tokens,
                sampling: victim.sampling.clone(),
                stop: victim.stop.take(),
                priority: RequestPriority::Normal,
                delta_tx: victim.delta_tx.clone(),
            };
            victim.phase = Phase::Finished;

            // Remove from decode batch
            decode_indices.remove(victim_pos);
            token_ids.remove(victim_pos);

            // Push to front of waiting queue (priority re-admission)
            waiting.push_front(requeue);
        }

        // Allocate one pool token per remaining decode request.
        let mut alloc_ok_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        let mut alloc_ok_tokens: Vec<u32> = Vec::with_capacity(decode_indices.len());
        for (j, &i) in decode_indices.iter().enumerate() {
            let slot = active[i].slot_idx;
            if let Err(e) = paged_kv_pool.alloc_tokens(slot, 1) {
                error!(
                    "Request {}: KV pool exhausted after preemption (slot {}): {} — finishing",
                    active[i].id, slot, e
                );
                active[i].finish(FinishReason::Length, tokenizer);
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
        let slot_indices: Vec<usize> = decode_indices.iter().map(|&i| active[i].slot_idx).collect();

        let sampling_params: Vec<&crate::sampler::SamplingParams> = decode_indices
            .iter()
            .map(|&i| &active[i].sampling)
            .collect();
        let all_greedy = sampling_params
            .iter()
            .all(|p| p.is_greedy() && !p.has_penalties());

        // Lazy-init decode context on first batched decode.
        if decode_bufs.is_none() {
            match model.create_decode_context(states.len(), paged_kv_pool) {
                Ok(ctx) => *decode_bufs = Some(ctx),
                Err(e) => {
                    error!("Failed to create decode context: {}", e);
                    for &i in &decode_indices {
                        active[i].phase = Phase::Finished;
                    }
                    return;
                }
            }
        }
        let decode_ctx = decode_bufs.as_mut().unwrap();

        // Pre-decode: scheduler handles H2D, metadata, and FlashInfer plan
        // via DecodeContextOps. This decouples scheduler-level work from
        // model-level neural computation.
        {
            use crate::model::DecodeContextOps;
            let ctx = model.device_context();
            decode_ctx.set_batch_size(token_ids.len());
            if let Err(e) = decode_ctx.upload_token_ids(ctx, &token_ids) {
                error!("Pre-decode upload_token_ids failed: {}", e);
                for &i in &decode_indices {
                    active[i].phase = Phase::Finished;
                }
                return;
            }
            match decode_ctx.update_metadata(ctx, paged_kv_pool, &slot_indices) {
                Ok(reallocated) => {
                    if reallocated {
                        decode_ctx.invalidate_graph_cache(token_ids.len());
                    }
                }
                Err(e) => {
                    error!("Pre-decode update_metadata failed: {}", e);
                    for &i in &decode_indices {
                        active[i].phase = Phase::Finished;
                    }
                    return;
                }
            }
            if let Err(e) = decode_ctx.plan_attention(
                ctx,
                token_ids.len(),
                model.num_q_heads(),
                model.num_kv_heads(),
                1, // page_size: token-level pool
                model.head_dim(),
                paged_kv_pool.format,
            ) {
                error!("Pre-decode plan_attention failed: {}", e);
                for &i in &decode_indices {
                    active[i].phase = Phase::Finished;
                }
                return;
            }
        }

        let forward_result = model.forward_decode_batch(
            &token_ids,
            states,
            &slot_indices,
            Some(paged_kv_pool),
            decode_ctx,
            all_greedy,
        );

        if let Err(e) = forward_result {
            error!("Batched decode failed: {}", e);
            for &i in &decode_indices {
                active[i].phase = Phase::Finished;
            }
            return;
        }
        let sampled_result = if all_greedy {
            match model.sample_batch_greedy(&slot_indices, decode_ctx) {
                Ok(Some(tokens)) => Ok(tokens),
                Ok(None) => model.select_tokens_batch(states, &slot_indices, &sampling_params, rng),
                Err(e) => Err(e),
            }
        } else {
            model.select_tokens_batch(states, &slot_indices, &sampling_params, rng)
        };
        match sampled_result {
            Ok(sampled_tokens) => {
                // Read logprobs from decode context (set by sample_batch_greedy)
                let logprobs_host: Option<&[f32]> = if all_greedy {
                    Some(crate::model::DecodeContextOps::logprobs_host(&*decode_ctx))
                } else {
                    None
                };
                for (j, &req_idx) in decode_indices.iter().enumerate() {
                    let token = sampled_tokens[j];
                    let req = &mut active[req_idx];
                    req.latest_logprob = logprobs_host.map(|lps| lps[j]);
                    if !req.sampling.ignore_eos && model.is_stop_token(token) {
                        req.finish(FinishReason::Stop, tokenizer);
                        continue;
                    }
                    req.generated_tokens.push(token);
                    if matches!(req.phase, Phase::Finished) {
                        continue;
                    }
                    if req.generated_tokens.len() >= req.max_tokens {
                        req.finish(FinishReason::Length, tokenizer);
                    }
                }
            }
            Err(e) => {
                error!("Batched sampling failed: {}", e);
                for &req_idx in &decode_indices {
                    active[req_idx].phase = Phase::Finished;
                }
            }
        }
    }
}
