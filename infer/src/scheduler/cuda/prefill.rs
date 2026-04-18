use super::*;

fn is_full_prompt_reuse_hit(prompt_len: usize, prefix_len: usize) -> bool {
    prefix_len > 0 && prefix_len == prompt_len
}

fn is_exact_full_prefix_hit(prompt_len: usize, cached_len: usize, prefix_len: usize) -> bool {
    is_full_prompt_reuse_hit(prompt_len, prefix_len) && prefix_len == cached_len
}

fn is_prompt_prefix_of_cached_hit(prompt_len: usize, cached_len: usize, prefix_len: usize) -> bool {
    is_full_prompt_reuse_hit(prompt_len, prefix_len) && prefix_len < cached_len
}

impl<M: ModelForward> Scheduler<M> {
    /// Compute prefix cache for a new request and begin chunked prefill.
    pub(super) fn step_new(&mut self, idx: usize) {
        let default_chunk_size = self.prefill_chunk_size();
        if self.active[idx].delta_tx.is_closed() {
            self.active[idx].phase = Phase::Finished;
            return;
        }

        let si = self.active[idx].slot_idx;
        let req_id = self.active[idx].id;
        let prompt_len = self.active[idx].prompt_tokens.len();
        let raw_prefix_len = self.active[idx].reusable_prefix_len;
        let cached_prompt_len = self.active[idx].reusable_cached_prompt_len;

        // Hybrid models (e.g. Qwen3.5) cannot truncate recurrent state to an
        // arbitrary prefix length. Downgrade any partial hit (radix match
        // shorter than prompt) to MISS — only full-prompt hits benefit from
        // snapshot/restore. The previous `raw < cached` guard left a hole at
        // exact-block-aligned prompts where `raw == cached < prompt_len` fell
        // through to the `truncate_to + restore_prefix_snapshot` branch at
        // line 99, which zeroes recurrent state and depends on the snapshot
        // being valid. See docs/plans/paged-prefill-followups-2026-04-18.md §3.
        let (effective, pool_prefix_len) = {
            let req = &mut self.active[idx];
            let state = &mut self.states[si];

            let prefix_len = if raw_prefix_len > 0
                && raw_prefix_len < prompt_len
                && !state.supports_partial_prefix()
            {
                0
            } else {
                raw_prefix_len
            };
            let exact_full_prefix_hit =
                is_exact_full_prefix_hit(prompt_len, cached_prompt_len, prefix_len);
            let prompt_prefix_of_cached_hit =
                is_prompt_prefix_of_cached_hit(prompt_len, cached_prompt_len, prefix_len);
            let mut pool_prefix_len = prefix_len;

            let effective = if exact_full_prefix_hit {
                if state.supports_partial_prefix() {
                    // An exact prompt match can safely keep the prefix up to N-1
                    // tokens and replay only the final prompt token. This refreshes
                    // the next-token logits without duplicating it in KV.
                    let replay_from = prefix_len.saturating_sub(1);
                    info!(
                        "Request {}: prefix HIT {}/{} tokens (exact full match, replaying final token with {} reused)",
                        req.id, prefix_len, prompt_len, replay_from
                    );
                    if let Err(e) = state.truncate_to(replay_from) {
                        error!(
                            "Request {}: truncate on full prompt reuse hit failed: {}",
                            req.id, e
                        );
                        req.phase = Phase::Finished;
                        return;
                    }
                    self.slot_materialized_prompt_lens[si] = replay_from;
                    pool_prefix_len = replay_from;
                    req.prompt_tokens[replay_from..].to_vec()
                } else {
                    info!(
                        "Request {}: prefix HIT {}/{} tokens (exact full match, recomputing prompt to refresh logits)",
                        req.id, prefix_len, prompt_len
                    );
                    if let Err(e) = state.reset() {
                        error!("Request {}: reset failed: {}", req.id, e);
                        req.phase = Phase::Finished;
                        return;
                    }
                    self.slot_materialized_prompt_lens[si] = 0;
                    pool_prefix_len = 0;
                    req.prompt_tokens.clone()
                }
            } else if prompt_prefix_of_cached_hit {
                info!(
                    "Request {}: prefix HIT {}/{} tokens (cached prompt had extra suffix, recomputing prompt for correctness)",
                    req.id, prefix_len, prompt_len
                );
                if let Err(e) = state.reset() {
                    error!("Request {}: reset failed: {}", req.id, e);
                    req.phase = Phase::Finished;
                    return;
                }
                self.slot_materialized_prompt_lens[si] = 0;
                pool_prefix_len = 0;
                req.prompt_tokens.clone()
            } else if prefix_len > 0 && prefix_len == cached_prompt_len {
                // Truncate contiguous KV cache to prefix length — removes stale
                // decode tokens from the previous request before migration reads it.
                if let Err(e) = state.truncate_to(prefix_len) {
                    error!("Request {}: truncate on prefix hit failed: {}", req.id, e);
                    if let Err(e2) = state.reset() {
                        error!("Request {}: reset failed: {}", req.id, e2);
                    }
                    self.slot_materialized_prompt_lens[si] = 0;
                    req.phase = Phase::Prefilling {
                        effective_tokens: req.prompt_tokens.clone(),
                        progress: 0,
                    };
                    return;
                }

                // Full prefix hit — restore recurrent state snapshot to undo
                // decode-token contamination from the previous request.
                match state.restore_prefix_snapshot() {
                    Ok(true) => info!(
                        "Request {}: prefix HIT {}/{} tokens (recurrent state restored)",
                        req.id, prefix_len, prompt_len
                    ),
                    Ok(false) => info!(
                        "Request {}: prefix HIT {}/{} tokens",
                        req.id, prefix_len, prompt_len
                    ),
                    Err(e) => {
                        warn!(
                            "Request {}: prefix hit but snapshot restore failed ({}), falling back to MISS",
                            req.id, e
                        );
                        if let Err(e2) = state.reset() {
                            error!("Request {}: reset failed: {}", req.id, e2);
                            req.phase = Phase::Finished;
                            return;
                        }
                        self.slot_materialized_prompt_lens[si] = 0;
                        req.phase = Phase::Prefilling {
                            effective_tokens: req.prompt_tokens.clone(),
                            progress: 0,
                        };
                        return;
                    }
                }
                self.slot_materialized_prompt_lens[si] = prefix_len;
                req.prompt_tokens[prefix_len..].to_vec()
            } else if prefix_len > 0 {
                info!(
                    "Request {}: prefix PARTIAL {}/{} tokens",
                    req.id, prefix_len, prompt_len
                );
                if let Err(e) = state.truncate_to(prefix_len) {
                    error!("Request {}: truncate failed: {}", req.id, e);
                    req.phase = Phase::Finished;
                    return;
                }
                self.slot_materialized_prompt_lens[si] = prefix_len;
                req.prompt_tokens[prefix_len..].to_vec()
            } else {
                info!("Request {}: prefix MISS", req.id);
                if let Err(e) = state.reset() {
                    error!("Request {}: reset failed: {}", req.id, e);
                    req.phase = Phase::Finished;
                    return;
                }
                self.slot_materialized_prompt_lens[si] = 0;
                req.prompt_tokens.clone()
            };

            (effective, pool_prefix_len)
        };
        let prefix_hit = effective.len() < prompt_len;
        self.metrics.record_prefix_lookup(prefix_hit);

        if pool_prefix_len > 0 && self.paged_kv_pool.is_active() {
            match self.alloc_pool_tokens_with_retry(si, pool_prefix_len) {
                Err(e) => {
                    error!("Request {}: pool alloc for prefix failed: {}", req_id, e);
                }
                Ok(_new_pages) => {
                    let ctx = self.model.device_context();
                    if let Err(e) = self.states[si].migrate_kv_range_to_paged(
                        ctx,
                        &self.paged_kv_pool,
                        si,
                        0,
                        pool_prefix_len,
                    ) {
                        error!(
                            "Request {}: prefix KV migration to pool failed: {}",
                            req_id, e
                        );
                    }
                }
            }
        }

        info!(
            "Request {}: chunked prefill starting ({} effective tokens, chunk_size={})",
            req_id,
            effective.len(),
            default_chunk_size
        );

        self.active[idx].phase = Phase::Prefilling {
            effective_tokens: effective,
            progress: 0,
        };
    }

    /// Process one chunk of a prefill. When all chunks are done, sample the
    /// first token and transition to Decoding.
    pub(super) fn step_prefill_chunk(&mut self, idx: usize) {
        let chunk_size = self.prefill_chunk_size();
        if self.active[idx].delta_tx.is_closed() {
            self.active[idx].phase = Phase::Finished;
            return;
        }

        let slot_idx = self.active[idx].slot_idx;
        // Snapshot chunk inputs as owned values so we can release the phase
        // borrow before calling `&mut self` methods (alloc_pool_tokens_with_retry).
        let (chunk_tokens, progress_val, total) = match &self.active[idx].phase {
            Phase::Prefilling {
                effective_tokens,
                progress,
            } => {
                let total = effective_tokens.len();
                let chunk_end = (*progress + chunk_size).min(total);
                (
                    effective_tokens[*progress..chunk_end].to_vec(),
                    *progress,
                    total,
                )
            }
            _ => return,
        };
        let chunk_len = chunk_tokens.len();
        let chunk_end = progress_val + chunk_len;

        let uses_paged = self.model.prefill_uses_paged_pool() && self.paged_kv_pool.is_active();

        let forward_result = if uses_paged {
            // Paged prefill: pre-allocate pool pages for this chunk so the
            // forward can write K/V directly through the page table. We pass a
            // dummy CudaSlice for `new_token_indices` — Qwen3's paged
            // implementation reads the page table from the pool itself.
            match self.alloc_pool_tokens_with_retry(slot_idx, chunk_len) {
                Err(e) => {
                    let req_id = self.active[idx].id;
                    error!(
                        "Request {}: pool alloc for paged prefill failed: {}",
                        req_id, e
                    );
                    self.active[idx].phase = Phase::Finished;
                    return;
                }
                Ok(_new_pages) => {
                    let ctx = self.model.device_context();
                    match ctx.stream.clone_htod(&[0i32]) {
                        Ok(dummy_indices) => self.model.forward_prefill_with_pool(
                            &chunk_tokens,
                            &mut self.states[slot_idx],
                            &self.paged_kv_pool,
                            slot_idx,
                            &dummy_indices,
                        ),
                        Err(e) => Err(anyhow::anyhow!("dummy indices H2D failed: {e}")),
                    }
                }
            }
        } else {
            self.model
                .forward_prefill(&chunk_tokens, &mut self.states[slot_idx])
        };

        if let Err(e) = forward_result {
            let req_id = self.active[idx].id;
            error!("Request {}: prefill chunk failed: {}", req_id, e);
            self.active[idx].phase = Phase::Finished;
            return;
        }

        let new_progress = chunk_end;
        if new_progress < total {
            if let Phase::Prefilling { progress, .. } = &mut self.active[idx].phase {
                *progress = new_progress;
            }
            info!(
                "Request {}: prefill chunk {}/{} tokens",
                self.active[idx].id, new_progress, total
            );
            return;
        }

        // Post-forward KV migration (only when contiguous-KV prefill was used).
        // Paged prefill already wrote K/V into the pool; nothing to migrate.
        if !uses_paged && self.paged_kv_pool.is_active() {
            let pool_start = self.paged_kv_pool.seq_len(slot_idx);
            match self.alloc_pool_tokens_with_retry(slot_idx, total) {
                Err(e) => {
                    error!(
                        "Request {}: pool alloc for migration failed: {}",
                        self.active[idx].id, e
                    );
                }
                Ok(_new_pages) => {
                    let ctx = self.model.device_context();
                    if let Err(e) = self.states[slot_idx].migrate_kv_range_to_paged(
                        ctx,
                        &self.paged_kv_pool,
                        slot_idx,
                        pool_start,
                        total,
                    ) {
                        error!(
                            "Request {}: KV migration to pool failed: {}",
                            self.active[idx].id, e
                        );
                    }
                }
            }
        }

        // Snapshot auxiliary state (recurrent/SSM) after prefill completes.
        // On the next full prefix hit for this slot, restore_prefix_snapshot()
        // reverts to this clean state, avoiding decode-token contamination.
        if let Err(e) = self.states[slot_idx].save_prefix_snapshot() {
            warn!(
                "Request {}: save prefix snapshot failed: {} (prefix cache disabled for this slot)",
                self.active[idx].id, e
            );
        } else {
            self.active[idx].mark_prompt_cacheable();
        }

        let sampling = self.active[idx].sampling.clone();
        match self
            .model
            .select_token(&mut self.states[slot_idx], &sampling, &mut self.rng)
        {
            Ok(token) => {
                if !self.active[idx].sampling.ignore_eos && self.model.is_stop_token(token) {
                    self.active[idx].finish(FinishReason::Stop, &self.tokenizer);
                    return;
                }
                self.active[idx].generated_tokens.push(token);
                self.active[idx].emit_delta(&self.tokenizer);

                if matches!(self.active[idx].phase, Phase::Finished) {
                    return;
                }
                if self.active[idx].generated_tokens.len() >= self.active[idx].max_tokens {
                    self.active[idx].finish(FinishReason::Length, &self.tokenizer);
                } else {
                    if self.active[idx].first_token_at.is_none() {
                        self.active[idx].first_token_at = Some(std::time::Instant::now());
                    }
                    self.active[idx].phase = Phase::Decoding;
                }
            }
            Err(e) => {
                error!(
                    "Request {}: select_token failed: {}",
                    self.active[idx].id, e
                );
                self.active[idx].phase = Phase::Finished;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        is_exact_full_prefix_hit, is_full_prompt_reuse_hit, is_prompt_prefix_of_cached_hit,
    };

    #[test]
    fn exact_full_prefix_hit_detects_only_true_exact_matches() {
        assert!(is_exact_full_prefix_hit(4, 4, 4));
        assert!(!is_exact_full_prefix_hit(5, 4, 4));
        assert!(!is_exact_full_prefix_hit(4, 5, 4));
        assert!(!is_exact_full_prefix_hit(4, 4, 3));
    }

    #[test]
    fn full_prompt_reuse_hit_detects_exact_and_prefix_of_cached_cases() {
        assert!(is_full_prompt_reuse_hit(4, 4));
        assert!(is_full_prompt_reuse_hit(4, 4));
        assert!(!is_full_prompt_reuse_hit(4, 3));
        assert!(!is_full_prompt_reuse_hit(4, 0));
    }

    #[test]
    fn prompt_prefix_of_cached_hit_detects_only_shorter_prompt_case() {
        assert!(is_prompt_prefix_of_cached_hit(4, 6, 4));
        assert!(!is_prompt_prefix_of_cached_hit(4, 4, 4));
        assert!(!is_prompt_prefix_of_cached_hit(4, 6, 3));
        assert!(!is_prompt_prefix_of_cached_hit(4, 3, 3));
    }
}
