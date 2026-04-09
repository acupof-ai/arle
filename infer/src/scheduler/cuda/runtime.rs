use super::*;

impl<M: ModelForward> Scheduler<M> {
    /// Run the scheduler loop. Blocks until all handles are dropped.
    pub fn run(mut self) {
        self.warmup_cuda_graphs();
        info!("Scheduler run loop started");
        loop {
            while let Ok(req) = self.request_rx.try_recv() {
                self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                self.waiting.push_back(req);
            }

            if self.active.is_empty() && self.waiting.is_empty() {
                match self.request_rx.blocking_recv() {
                    Some(req) => {
                        self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                        self.waiting.push_back(req);
                    }
                    None => {
                        info!("Scheduler shutting down: all handles dropped");
                        break;
                    }
                }
            }

            let step_start = std::time::Instant::now();
            self.assign_slots();
            let assign_us = step_start.elapsed().as_micros();

            let step_t = std::time::Instant::now();
            self.step();
            let step_us = step_t.elapsed().as_micros();

            let clean_t = std::time::Instant::now();
            self.cleanup();
            let clean_us = clean_t.elapsed().as_micros();

            let total_us = step_start.elapsed().as_micros();
            if total_us > 50_000 {
                // Log slow iterations (>50ms)
                info!(
                    "Scheduler step: assign={}us step={}us cleanup={}us total={}us active={}",
                    assign_us,
                    step_us,
                    clean_us,
                    total_us,
                    self.active.len()
                );
            }
        }
    }

    fn assign_slots(&mut self) {
        while !self.waiting.is_empty() {
            let free_slots = self.free_slots();
            if free_slots.is_empty() {
                break;
            }

            let incoming = self.waiting.pop_front().expect("checked non-empty above");
            let prompt_tokens = match self.tokenizer.encode(&incoming.prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => {
                    error!("Empty prompt after tokenization, skipping");
                    continue;
                }
                Err(e) => {
                    error!("Tokenization error: {}", e);
                    continue;
                }
            };

            // Pick the free slot with the best prefix match to maximize KV reuse.
            let slot_idx = self.best_prefix_slot(&free_slots, &prompt_tokens);

            let id = self.next_id;
            self.next_id += 1;

            let prefix_len = self.cached_prompts[slot_idx]
                .iter()
                .zip(prompt_tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            if prefix_len > 0 {
                info!(
                    "Request {} → slot {} (prompt={} tokens, prefix_reuse={}, queue={})",
                    id,
                    slot_idx,
                    prompt_tokens.len(),
                    prefix_len,
                    self.waiting.len()
                );
            } else {
                info!(
                    "Request {} → slot {} (prompt={} tokens, queue={})",
                    id,
                    slot_idx,
                    prompt_tokens.len(),
                    self.waiting.len()
                );
            }

            self.active.push(ActiveRequest {
                id,
                slot_idx,
                prompt: incoming.prompt,
                prompt_tokens,
                generated_tokens: Vec::new(),
                max_tokens: incoming.max_tokens,
                sampling: incoming.sampling,
                stop: incoming.stop,
                delta_tx: incoming.delta_tx,
                full_decoded: String::new(),
                decoded_token_count: 0,
                sent_len: 0,
                phase: Phase::New,
                prefix_byte_len: 0,
                latest_logprob: None,
            });
        }
    }

    /// Find all free slot indices.
    fn free_slots(&self) -> Vec<usize> {
        let in_use: Vec<usize> = self.active.iter().map(|a| a.slot_idx).collect();
        (0..self.states.len())
            .filter(|i| !in_use.contains(i))
            .collect()
    }

    /// Pick the free slot whose cached prompt best matches the given tokens.
    /// Falls back to the first free slot if no prefix matches.
    fn best_prefix_slot(&self, free_slots: &[usize], prompt_tokens: &[u32]) -> usize {
        let mut best_slot = free_slots[0];
        let mut best_match = 0usize;

        for &slot in free_slots {
            let cached = &self.cached_prompts[slot];
            let match_len = cached
                .iter()
                .zip(prompt_tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            if match_len > best_match {
                best_match = match_len;
                best_slot = slot;
            }
        }
        best_slot
    }

    fn cleanup(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if matches!(self.active[i].phase, Phase::Finished) {
                let req = self.active.remove(i);
                let gen_tokens = req.generated_tokens.len() as u64;

                self.paged_kv_pool.free_slot(req.slot_idx);
                self.cached_prompts[req.slot_idx] = req.prompt_tokens;
                let _ = self.states[req.slot_idx].offload_kv_if_needed();

                self.total_completed += 1;
                self.total_generated_tokens += gen_tokens;

                info!(
                    "Request {} done: {} tokens (active={}, waiting={})",
                    req.id,
                    gen_tokens,
                    self.active.len(),
                    self.waiting.len()
                );

                if self.total_completed % STATS_LOG_INTERVAL == 0 {
                    info!(
                        "Scheduler stats: completed={}, generated_tokens={}, active={}, waiting={}",
                        self.total_completed,
                        self.total_generated_tokens,
                        self.active.len(),
                        self.waiting.len()
                    );
                }

                if self.last_served >= self.active.len() && !self.active.is_empty() {
                    self.last_served = 0;
                }
            } else {
                i += 1;
            }
        }
    }
}
