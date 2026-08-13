//! OPD frozen-prompt-KV: off-tape chunked prompt capture, then the taped gen-segment stack.

use super::*;

impl Qwen35Model {
    /// OPD frozen-prompt-KV writeback forward: forward+backward ONLY the
    /// generated segment (`gen_ids` = rows `gen_start..seq_len`), seeding each
    /// layer's attention from the prompt prefix (`prompt_ids` = rows
    /// `0..gen_start`) captured off-tape. Returns `[1, gen_len, hidden]`.
    ///
    /// Two phases:
    ///  1. OFF-TAPE prompt pass (no checkpoint, no offload): embed the prompt,
    ///     run the layers with the tape disabled; at each layer capture the
    ///     prefix K/V (full) or boundary state+conv (linear) AND propagate the
    ///     prompt hidden so the next layer's capture sees the correct input. The
    ///     prompt hidden is discarded after.
    ///  2. TAPED gen pass: embed ONLY `gen_ids` fresh → `[1, gen_len, hidden]`
    ///     (RMSNorm + MLP are position-local, so feeding `embed(gen_ids)` is
    ///     exact — only attention reads the prefix). Run the layers via
    ///     `checkpoint_sequential` (or a per-layer loop when checkpointing is
    ///     off), each consuming its captured prefix, then final_norm.
    pub fn forward_hidden_states_gen_segment(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        prompt_ids: &[u32],
        gen_ids: &[u32],
        prompt_positions: &[u32],
        gen_positions: &[u32],
        cp: crate::context_parallel::CpContext,
    ) -> Result<TensorId> {
        if prompt_ids.len() != prompt_positions.len() {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: prompt_ids.len(),
                expected_len: prompt_positions.len(),
            });
        }
        if gen_ids.len() != gen_positions.len() {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: gen_ids.len(),
                expected_len: gen_positions.len(),
            });
        }
        if gen_ids.is_empty() {
            return Err(Qwen35Error::InvalidConfig(
                "frozen-prompt-KV forward requires at least one generated token",
            ));
        }
        let gen_start = prompt_ids.len();
        let gen_len = gen_ids.len();
        let batch = 1usize;

        // ---- PHASE 1: off-tape prompt prefix capture ----
        // Chunked: the prompt is processed in OPD_SEQ_CHUNK-row pieces so the
        // per-layer hidden/MLP transients stay O(chunk × hidden) instead of
        // O(prompt × hidden). Each layer's K/V is accumulated across chunks
        // (prefix + chunk) and used as the attention K/V for the next chunk.
        let prefix_cache = if gen_start > 0 {
            let prompt_token_indices = prompt_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
            let prompt_pos = prompt_positions
                .iter()
                .map(|&id| id as usize)
                .collect::<Vec<_>>();

            let mut prefix_tape = Tape::new();
            prefix_tape.set_enabled(false);

            let chunk = crate::runtime_flags::OPD_SEQ_CHUNK;
            let num_chunks = gen_start.div_ceil(chunk);
            // Accumulated prefix per layer (None before the first chunk touches it).
            let mut layer_prefix: Vec<Option<LayerPrefix>> = vec![None; self.layers.len()];

            for c in 0..num_chunks {
                let start = c * chunk;
                let end = (start + chunk).min(gen_start);
                let chunk_len = end - start;
                let chunk_ids = &prompt_token_indices[start..end];
                let chunk_pos = &prompt_pos[start..end];
                let cos_chunk = select_cache_rows(self.cos_cache, chunk_pos, store)?;
                let sin_chunk = select_cache_rows(self.sin_cache, chunk_pos, store)?;

                let mut h = embedding(self.embed_tokens, chunk_ids, store, &mut prefix_tape)?;
                h = reshape(
                    h,
                    &[batch, chunk_len, self.config.hidden_size],
                    store,
                    &mut prefix_tape,
                )?;

                for (li, layer) in self.layers.iter().enumerate() {
                    let prefix_kv = match &layer_prefix[li] {
                        Some(LayerPrefix::Full(kv)) => Some(kv),
                        _ => None,
                    };
                    let (next, prefix) = layer.forward_capture_prefix(
                        h,
                        &self.config,
                        self.tp,
                        cos_chunk,
                        sin_chunk,
                        batch,
                        chunk_len,
                        start,
                        prefix_kv,
                        store,
                        &mut prefix_tape,
                    )?;
                    h = next;
                    layer_prefix[li] = Some(prefix);
                }
            }

            let layers = layer_prefix
                .into_iter()
                .map(|opt| opt.expect("every layer must have captured prefix"))
                .collect();
            WritebackPrefixCache { layers }
        } else {
            // gen_start == 0: no prefix; an empty cache forces the gen pass to seed
            // from zeros (equivalent to the full sequence with no prompt).
            return Err(Qwen35Error::InvalidConfig(
                "frozen-prompt-KV forward requires a non-empty prompt prefix",
            ));
        };

        // ---- PHASE 2: taped gen-segment forward ----
        let gen_token_indices = gen_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let gen_pos = gen_positions
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let cos_gen = select_cache_rows(self.cos_cache, &gen_pos, store)?;
        let sin_gen = select_cache_rows(self.sin_cache, &gen_pos, store)?;
        // CP: gen_positions are the absolute rows of this rank's gen shard; pass them
        // through so the ring kernel masks causally by true position. Off-CP: None.
        // Arc so checkpoint_sequential's per-group layer_fn.clone() is a refcount
        // bump, not a deep copy of the position vector.
        let cp_positions: Option<Arc<[usize]>> = cp.is_enabled().then(|| Arc::from(gen_pos));

        let mut hidden = embedding(self.embed_tokens, &gen_token_indices, store, tape)?;
        hidden = reshape(
            hidden,
            &[batch, gen_len, self.config.hidden_size],
            store,
            tape,
        )?;

        if self.should_checkpoint(batch, gen_len, store, tape) {
            let cache = Arc::new(prefix_cache);
            let layers = Arc::new(self.layers.clone());
            let cfg = self.config.clone();
            let tp = self.tp;
            let (cos_id, sin_id) = (cos_gen, sin_gen);
            let cp_positions = cp_positions.clone();
            let layer_fn = {
                let layers = Arc::clone(&layers);
                let cache = Arc::clone(&cache);
                move |idx: usize, h, s: &mut TensorStore, t: &mut Tape| {
                    layers[idx]
                        .forward_gen_segment(
                            h,
                            &cfg,
                            tp,
                            cos_id,
                            sin_id,
                            &cache.layers[idx],
                            batch,
                            gen_start,
                            gen_len,
                            cp,
                            cp_positions.as_deref(),
                            s,
                            t,
                        )
                        .map_err(qwen35_to_autograd)
                }
            };
            let param_ids: Vec<Vec<TensorId>> = self
                .layers
                .iter()
                .map(|l| l.checkpoint_param_ids(self.lora_skip_experts, store))
                .collect();
            hidden = self.checkpoint_layers(
                hidden,
                batch,
                gen_len,
                store,
                tape,
                |idx| param_ids[idx].clone(),
                layer_fn,
            )?;
        } else {
            for (layer_index, layer) in self.layers.iter().enumerate() {
                hidden = self.detach_before_lora_layer(hidden, layer_index, store, tape)?;
                hidden = layer.forward_gen_segment(
                    hidden,
                    &self.config,
                    self.tp,
                    cos_gen,
                    sin_gen,
                    &prefix_cache.layers[layer_index],
                    batch,
                    gen_start,
                    gen_len,
                    cp,
                    cp_positions.as_deref(),
                    store,
                    tape,
                )?;
            }
        }
        qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )
    }
}
