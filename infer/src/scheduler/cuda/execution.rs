use super::*;

impl<M: ModelForward> Scheduler<M> {
    pub(super) fn step(&mut self) {
        let num = self.active.len();
        if num == 0 {
            return;
        }

        // Phase 1: Batched decode (GPU forward pass + sampling)
        //
        // Launched BEFORE emit_delta so that GPU compute overlaps with
        // CPU tokenizer work in Phase 2. emit_delta only needs tokens
        // from the PREVIOUS step (already in generated_tokens), so
        // there's no data dependency with the current forward pass.
        let has_decode = self
            .active
            .iter()
            .any(|r| matches!(r.phase, Phase::Decoding));
        let decode_us = if has_decode {
            let t = std::time::Instant::now();
            self.step_decode_batch();
            t.elapsed().as_micros()
        } else {
            0
        };

        // Phase 2: Flush deferred deltas (CPU tokenizer decode)
        //
        // Runs after decode batch returns. When GPU forward is fast
        // (CUDA Graph replay), emit_delta dominates; when forward is
        // slow, emit_delta would have overlapped with GPU compute if
        // we split forward into launch + sync phases (future opt).
        let emit_t = std::time::Instant::now();
        {
            let Self {
                active, tokenizer, ..
            } = self;
            for req in active.iter_mut() {
                if matches!(req.phase, Phase::Decoding)
                    && req.decoded_token_count < req.generated_tokens.len()
                {
                    req.emit_delta(tokenizer);
                }
            }
        }
        let emit_us = emit_t.elapsed().as_micros();

        // Phase 3: Accept ALL new requests (prefix cache + start prefill)
        // Process all new requests in one step to avoid multi-iteration admission delay.
        let new_t = std::time::Instant::now();
        loop {
            let new_idx =
                (0..self.active.len()).find(|&i| matches!(self.active[i].phase, Phase::New));
            match new_idx {
                Some(idx) => self.step_new(idx),
                None => break,
            }
        }
        let new_us = new_t.elapsed().as_micros();

        // Phase 4: Prefill chunks — process a limited number of pending prefill
        // requests to avoid blocking decode for too long. When decode requests
        // are active, limit to 1 prefill per step (like SGLang). When idle,
        // process up to 8 to speed up batch admission.
        let prefill_t = std::time::Instant::now();
        let max_prefills = if has_decode { 1 } else { 8 };
        let prefill_indices: Vec<usize> = (0..self.active.len())
            .filter(|&i| matches!(self.active[i].phase, Phase::Prefilling { .. }))
            .take(max_prefills)
            .collect();
        for idx in prefill_indices {
            // Re-check phase since step_prefill_chunk may transition to Decoding/Finished
            if matches!(self.active[idx].phase, Phase::Prefilling { .. }) {
                self.step_prefill_chunk(idx, has_decode);
            }
        }
        let prefill_us = prefill_t.elapsed().as_micros();

        // Log slow steps for profiling
        let total_us = emit_us + decode_us + new_us + prefill_us;
        if total_us > 100_000 {
            info!(
                "step breakdown: emit={}us decode={}us new={}us prefill={}us total={}us batch={}",
                emit_us, decode_us, new_us, prefill_us, total_us, num
            );
        }
    }
}
