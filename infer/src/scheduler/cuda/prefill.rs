use super::{FinishReason, GenerationState, ModelForward, Phase, Scheduler, error, info, warn};

fn is_full_prompt_reuse_hit(prompt_len: usize, prefix_len: usize) -> bool {
    prefix_len > 0 && prefix_len == prompt_len
}

fn is_exact_full_prefix_hit(prompt_len: usize, cached_len: usize, prefix_len: usize) -> bool {
    is_full_prompt_reuse_hit(prompt_len, prefix_len) && prefix_len == cached_len
}

fn is_prompt_prefix_of_cached_hit(prompt_len: usize, cached_len: usize, prefix_len: usize) -> bool {
    is_full_prompt_reuse_hit(prompt_len, prefix_len) && prefix_len < cached_len
}

/// Returns true when the radix hit should be downgraded to MISS for a model
/// that cannot truncate state to an arbitrary prefix (e.g. Qwen3.5 hybrid).
///
/// Only full-prompt hits (`raw == prompt_len`) are safe for such models,
/// because the exact-match branch in `step_new` routes through `state.reset()`
/// + full re-prefill rather than `truncate_to + restore_prefix_snapshot`.
///
/// Any partial hit — including the exact-block-aligned `raw == cached < prompt_len`
/// case — must downgrade. See docs/plans/paged-prefill-followups-2026-04-18.md §3.
fn should_downgrade_partial_hit_to_miss(
    raw_prefix_len: usize,
    prompt_len: usize,
    supports_partial_prefix: bool,
) -> bool {
    raw_prefix_len > 0 && raw_prefix_len < prompt_len && !supports_partial_prefix
}

impl<M: ModelForward> Scheduler<M> {
    /// Compute prefix cache for a new request and begin chunked prefill.
    pub(super) fn step_new(&mut self, slot_idx: usize) {
        let default_chunk_size = self.prefill_chunk_size();
        let Some(req) = self.request(slot_idx) else {
            return;
        };
        if req.delta_tx.is_closed() {
            self.finish_slot(slot_idx);
            return;
        }

        let req_id = req.id;
        let prompt_tokens = req.prompt_tokens.clone();
        let prompt_len = prompt_tokens.len();
        let raw_prefix_len = req.reusable_prefix_len;
        let cached_prompt_len = req.reusable_cached_prompt_len;
        let attached_prefix_blocks = req.attached_prefix_blocks.clone();
        let si = slot_idx;

        if self.model.prefill_uses_paged_pool() && !attached_prefix_blocks.is_empty() {
            let attach_prefix_len = if is_full_prompt_reuse_hit(prompt_len, raw_prefix_len) {
                raw_prefix_len.saturating_sub(1)
            } else {
                raw_prefix_len
            };
            let effective = prompt_tokens[attach_prefix_len..].to_vec();

            if let Err(e) = self.states[si].reset() {
                error!(
                    "Request {}: reset before paged prefix attach failed: {}",
                    req_id, e
                );
                self.finish_slot(slot_idx);
                return;
            }
            self.slot_materialized_prompt_lens[si] = 0;

            if let Err(e) =
                self.attach_gpu_prefix_blocks(si, &attached_prefix_blocks, attach_prefix_len)
            {
                error!(
                    "Request {}: paged prefix attach failed for {} tokens: {}",
                    req_id, attach_prefix_len, e
                );
                self.finish_slot(slot_idx);
                return;
            }

            info!(
                "Request {}: paged prefix ATTACH {}/{} tokens",
                req_id, attach_prefix_len, prompt_len
            );
            info!(
                "Request {}: chunked prefill starting ({} effective tokens, chunk_size={})",
                req_id,
                effective.len(),
                default_chunk_size
            );
            if let Some(req) = self.request_mut(slot_idx) {
                req.phase = Phase::Prefilling {
                    materialized_prefix_len: attach_prefix_len,
                    effective_tokens: effective,
                    progress: 0,
                };
            }
            return;
        }

        // Hybrid models (e.g. Qwen3.5) cannot truncate recurrent state to an
        // arbitrary prefix length. Downgrade any partial hit (radix match
        // shorter than prompt) to MISS — only full-prompt hits benefit from
        // snapshot/restore. The previous `raw < cached` guard left a hole at
        // exact-block-aligned prompts where `raw == cached < prompt_len` fell
        // through to the `truncate_to + restore_prefix_snapshot` branch at
        // line 99, which zeroes recurrent state and depends on the snapshot
        // being valid. See docs/plans/paged-prefill-followups-2026-04-18.md §3.
        let (effective, pool_prefix_len) = {
            let state = &mut self.states[si];

            let prefix_len = if should_downgrade_partial_hit_to_miss(
                raw_prefix_len,
                prompt_len,
                state.supports_partial_prefix(),
            ) {
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
                        req_id, prefix_len, prompt_len, replay_from
                    );
                    if let Err(e) = state.truncate_to(replay_from) {
                        error!(
                            "Request {}: truncate on full prompt reuse hit failed: {}",
                            req_id, e
                        );
                        self.finish_slot(slot_idx);
                        return;
                    }
                    self.slot_materialized_prompt_lens[si] = replay_from;
                    pool_prefix_len = replay_from;
                    prompt_tokens[replay_from..].to_vec()
                } else {
                    info!(
                        "Request {}: prefix HIT {}/{} tokens (exact full match, recomputing prompt to refresh logits)",
                        req_id, prefix_len, prompt_len
                    );
                    if let Err(e) = state.reset() {
                        error!("Request {}: reset failed: {}", req_id, e);
                        self.finish_slot(slot_idx);
                        return;
                    }
                    self.slot_materialized_prompt_lens[si] = 0;
                    pool_prefix_len = 0;
                    prompt_tokens
                }
            } else if prompt_prefix_of_cached_hit {
                info!(
                    "Request {}: prefix HIT {}/{} tokens (cached prompt had extra suffix, recomputing prompt for correctness)",
                    req_id, prefix_len, prompt_len
                );
                if let Err(e) = state.reset() {
                    error!("Request {}: reset failed: {}", req_id, e);
                    self.finish_slot(slot_idx);
                    return;
                }
                self.slot_materialized_prompt_lens[si] = 0;
                pool_prefix_len = 0;
                prompt_tokens
            } else if prefix_len > 0 && prefix_len == cached_prompt_len {
                // Truncate contiguous KV cache to prefix length — removes stale
                // decode tokens from the previous request before migration reads it.
                if let Err(e) = state.truncate_to(prefix_len) {
                    error!("Request {}: truncate on prefix hit failed: {}", req_id, e);
                    if let Err(e2) = state.reset() {
                        error!("Request {}: reset failed: {}", req_id, e2);
                    }
                    self.slot_materialized_prompt_lens[si] = 0;
                    pool_prefix_len = 0;
                    prompt_tokens
                } else {
                    // Full prefix hit — restore recurrent state snapshot to undo
                    // decode-token contamination from the previous request.
                    let restored = match state.restore_prefix_snapshot() {
                        Ok(true) => {
                            info!(
                                "Request {}: prefix HIT {}/{} tokens (recurrent state restored)",
                                req_id, prefix_len, prompt_len
                            );
                            true
                        }
                        Ok(false) => {
                            info!(
                                "Request {}: prefix HIT {}/{} tokens",
                                req_id, prefix_len, prompt_len
                            );
                            true
                        }
                        Err(e) => {
                            warn!(
                                "Request {}: prefix hit but snapshot restore failed ({}), falling back to MISS",
                                req_id, e
                            );
                            if let Err(e2) = state.reset() {
                                error!("Request {}: reset failed: {}", req_id, e2);
                                self.finish_slot(slot_idx);
                                return;
                            }
                            self.slot_materialized_prompt_lens[si] = 0;
                            pool_prefix_len = 0;
                            false
                        }
                    };
                    if restored {
                        self.slot_materialized_prompt_lens[si] = prefix_len;
                        prompt_tokens[prefix_len..].to_vec()
                    } else {
                        prompt_tokens
                    }
                }
            } else if prefix_len > 0 {
                info!(
                    "Request {}: prefix PARTIAL {}/{} tokens",
                    req_id, prefix_len, prompt_len
                );
                if let Err(e) = state.truncate_to(prefix_len) {
                    error!("Request {}: truncate failed: {}", req_id, e);
                    self.finish_slot(slot_idx);
                    return;
                }
                self.slot_materialized_prompt_lens[si] = prefix_len;
                prompt_tokens[prefix_len..].to_vec()
            } else {
                info!("Request {}: prefix MISS", req_id);
                if let Err(e) = state.reset() {
                    error!("Request {}: reset failed: {}", req_id, e);
                    self.finish_slot(slot_idx);
                    return;
                }
                self.slot_materialized_prompt_lens[si] = 0;
                prompt_tokens
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

        if let Some(req) = self.request_mut(slot_idx) {
            req.phase = Phase::Prefilling {
                materialized_prefix_len: pool_prefix_len,
                effective_tokens: effective,
                progress: 0,
            };
        }
    }

    /// Process one chunk of a prefill. When all chunks are done, sample the
    /// first token and transition to Decoding.
    pub(super) fn step_prefill_chunk(&mut self, slot_idx: usize, chunk_size: usize) -> usize {
        let Some(req) = self.request(slot_idx) else {
            return 0;
        };
        if req.delta_tx.is_closed() {
            self.finish_slot(slot_idx);
            return 0;
        }

        let req_id = req.id;
        // Snapshot chunk inputs as owned values so we can release the phase
        // borrow before calling `&mut self` methods (alloc_pool_tokens_with_retry).
        let (chunk_tokens, progress_val, total) = match &req.phase {
            Phase::Prefilling {
                materialized_prefix_len: _,
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
            _ => return 0,
        };
        let chunk_len = chunk_tokens.len();
        if chunk_len == 0 {
            return 0;
        }
        let chunk_end = progress_val + chunk_len;

        let uses_paged = self.model.prefill_uses_paged_pool() && self.paged_kv_pool.is_active();

        let forward_result = if uses_paged {
            // Paged prefill: pre-allocate pool pages for this chunk so the
            // forward can write K/V directly through the page table. We pass a
            // dummy CudaSlice for `new_token_indices` — Qwen3's paged
            // implementation reads the page table from the pool itself.
            match self.alloc_pool_tokens_with_retry(slot_idx, chunk_len) {
                Err(e) => {
                    warn!(
                        "Request {}: deferring paged prefill chunk after pool alloc failed: {}",
                        req_id, e
                    );
                    return 0;
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
            error!("Request {}: prefill chunk failed: {}", req_id, e);
            self.finish_slot(slot_idx);
            return 0;
        }

        let new_progress = chunk_end;
        if new_progress < total {
            if let Some(req) = self.request_mut(slot_idx)
                && let Phase::Prefilling { progress, .. } = &mut req.phase
            {
                *progress = new_progress;
            }
            info!(
                "Request {}: prefill chunk {}/{} tokens",
                req_id, new_progress, total
            );
            return chunk_len;
        }

        // Post-forward KV migration (only when contiguous-KV prefill was used).
        // Paged prefill already wrote K/V into the pool; nothing to migrate.
        if !uses_paged && self.paged_kv_pool.is_active() {
            let pool_start = self.paged_kv_pool.seq_len(slot_idx);
            match self.alloc_pool_tokens_with_retry(slot_idx, total) {
                Err(e) => {
                    error!("Request {}: pool alloc for migration failed: {}", req_id, e);
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
                        error!("Request {}: KV migration to pool failed: {}", req_id, e);
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
                req_id, e
            );
        } else if let Some(req) = self.request_mut(slot_idx) {
            req.mark_prompt_cacheable();
        }

        let sampling = self
            .request(slot_idx)
            .map(|req| req.sampling.clone())
            .expect("prefill completion requires live request");
        match self
            .model
            .select_token(&mut self.states[slot_idx], &sampling, &mut self.rng)
        {
            Ok(token) => {
                let ignore_eos = self
                    .request(slot_idx)
                    .is_some_and(|req| req.sampling.ignore_eos);
                if !ignore_eos && self.model.is_stop_token(token) {
                    let Self {
                        active, tokenizer, ..
                    } = self;
                    if let Some(req) = active[slot_idx].as_mut() {
                        req.finish(FinishReason::Stop, tokenizer);
                    }
                    self.finish_slot(slot_idx);
                    return chunk_len;
                }
                let Self {
                    active, tokenizer, ..
                } = self;
                if let Some(req) = active[slot_idx].as_mut() {
                    req.generated_tokens.push(token);
                    req.emit_delta(tokenizer);
                }

                if matches!(
                    self.request(slot_idx).map(|req| &req.phase),
                    Some(Phase::Finished)
                ) {
                    self.finish_slot(slot_idx);
                    return chunk_len;
                }
                let reached_max = self
                    .request(slot_idx)
                    .is_some_and(|req| req.generated_tokens.len() >= req.max_tokens);
                if reached_max {
                    let Self {
                        active, tokenizer, ..
                    } = self;
                    if let Some(req) = active[slot_idx].as_mut() {
                        req.finish(FinishReason::Length, tokenizer);
                    }
                    self.finish_slot(slot_idx);
                } else {
                    if let Some(req) = self.request_mut(slot_idx)
                        && req.first_token_at.is_none()
                    {
                        req.first_token_at = Some(std::time::Instant::now());
                    }
                    self.move_to_decode(slot_idx);
                }
            }
            Err(e) => {
                error!("Request {}: select_token failed: {}", req_id, e);
                self.finish_slot(slot_idx);
            }
        }
        chunk_len
    }
}

#[cfg(test)]
mod tests {
    use super::{
        is_exact_full_prefix_hit, is_full_prompt_reuse_hit, is_prompt_prefix_of_cached_hit,
        should_downgrade_partial_hit_to_miss,
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

    #[test]
    fn hybrid_downgrade_fires_on_every_partial_hit() {
        // Non-hybrid models: never downgrade, even on partial hits.
        assert!(!should_downgrade_partial_hit_to_miss(4, 10, true));
        assert!(!should_downgrade_partial_hit_to_miss(10, 10, true));

        // Hybrid models: downgrade whenever the radix match is shorter than
        // the prompt. This is the safety invariant the fix locks in —
        // covers both `raw < cached` (common, block-remainder gap) and the
        // previously-slipped-through `raw == cached < prompt_len` case.
        for raw in 1..10 {
            for prompt in (raw + 1)..=16 {
                assert!(
                    should_downgrade_partial_hit_to_miss(raw, prompt, false),
                    "hybrid must downgrade when raw={raw} < prompt={prompt}",
                );
            }
        }

        // Full-prompt hit (`raw == prompt`) is the ONLY case safe for hybrid:
        // routes through the exact-match branch (state.reset + full re-prefill).
        for n in 1..=16 {
            assert!(
                !should_downgrade_partial_hit_to_miss(n, n, false),
                "hybrid must NOT downgrade full-prompt hits (raw == prompt == {n})",
            );
        }

        // Empty radix hit: nothing to downgrade (already effective MISS).
        assert!(!should_downgrade_partial_hit_to_miss(0, 16, false));
        assert!(!should_downgrade_partial_hit_to_miss(0, 0, false));

        // Exact-block-aligned partial hit — the slip-through the fix closes.
        // Pre-fix condition `raw < cached` missed this when cached was also
        // block-aligned equal to raw; the new `raw < prompt_len` check fires.
        assert!(should_downgrade_partial_hit_to_miss(16, 32, false));
        assert!(should_downgrade_partial_hit_to_miss(32, 48, false));
    }
}
