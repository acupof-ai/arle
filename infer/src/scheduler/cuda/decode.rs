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
        let decode_indices = valid_decode_indices;
        if decode_indices.is_empty() {
            return;
        }
        // Allocate one pool token per decode request. If allocation fails,
        // finish that request rather than continuing with stale pool state.
        let mut alloc_ok_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        let mut alloc_ok_tokens: Vec<u32> = Vec::with_capacity(decode_indices.len());
        for (j, &i) in decode_indices.iter().enumerate() {
            let slot = active[i].slot_idx;
            if let Err(e) = paged_kv_pool.alloc_tokens(slot, 1) {
                error!(
                    "Request {}: KV pool exhausted (slot {}): {} — finishing request",
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
                for (j, &req_idx) in decode_indices.iter().enumerate() {
                    let token = sampled_tokens[j];
                    let req = &mut active[req_idx];
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
