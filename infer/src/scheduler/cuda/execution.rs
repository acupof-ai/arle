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

        // Phase 4: Prefill chunks — adaptive rate based on decode batch size.
        //
        // During ramp-up (few decodes), allow more prefills to reduce TTFT.
        // At high concurrency, limit prefills to protect decode ITL.
        // SGLang uses a similar adaptive policy.
        let prefill_t = std::time::Instant::now();
        let decode_count = self
            .active
            .iter()
            .filter(|r| matches!(r.phase, Phase::Decoding))
            .count();
        let max_prefills = if !has_decode {
            8 // No decode: drain prefill queue fast
        } else if decode_count <= 4 {
            4 // Ramp-up: GPU has headroom, reduce TTFT
        } else if decode_count <= 16 {
            2 // Moderate load: balance TTFT vs ITL
        } else {
            1 // High concurrency: protect decode latency
        };
        let prefill_indices: Vec<usize> = (0..self.active.len())
            .filter(|&i| matches!(self.active[i].phase, Phase::Prefilling { .. }))
            .take(max_prefills)
            .collect();
        for idx in prefill_indices {
            // Re-check phase since step_prefill_chunk may transition to Decoding/Finished
            if matches!(self.active[idx].phase, Phase::Prefilling { .. }) {
                self.step_prefill_chunk(idx);
            }
        }
        let prefill_us = prefill_t.elapsed().as_micros();

        // Step timing — always tracked for /v1/stats and profiling.
        // EMA (exponential moving average) with α=0.1 smooths noise
        // while responding quickly to sustained changes.
        let total_us = decode_us + emit_us + new_us + prefill_us;
        const ALPHA: f64 = 0.1;
        let update_ema = |ema: &mut f64, val: u128| {
            let v = val as f64;
            if *ema == 0.0 {
                *ema = v;
            } else {
                *ema = ALPHA * v + (1.0 - ALPHA) * *ema;
            }
        };
        update_ema(&mut self.step_timing_decode_us, decode_us);
        update_ema(&mut self.step_timing_emit_us, emit_us);
        update_ema(&mut self.step_timing_prefill_us, prefill_us);
        update_ema(&mut self.step_timing_total_us, total_us);

        if total_us > 100_000 {
            info!(
                "step breakdown: decode={}us emit={}us new={}us prefill={}us total={}us batch={}",
                decode_us, emit_us, new_us, prefill_us, total_us, num
            );
        }
    }
}
