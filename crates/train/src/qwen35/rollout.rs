use super::*;

#[doc(hidden)]
pub fn forward_rollout_cached(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    input_ids: &[u32],
    position_ids: &[u32],
    cache: &mut Qwen35KvCache,
) -> Result<TensorId> {
    model.forward_rollout_cached(store, tape, input_ids, position_ids, cache)
}

#[doc(hidden)]
pub fn forward_rollout_cached_profiled(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    input_ids: &[u32],
    position_ids: &[u32],
    cache: &mut Qwen35KvCache,
) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
    model.forward_rollout_cached_profiled(store, tape, input_ids, position_ids, cache)
}

#[doc(hidden)]
pub fn forward_rollout_cached_device_token(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    token_id: TensorId,
    position_id: u32,
    cache: &mut Qwen35KvCache,
) -> Result<TensorId> {
    model.forward_rollout_cached_device_token(store, tape, token_id, position_id, cache)
}

#[doc(hidden)]
pub fn forward_rollout_cached_device_token_profiled(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    token_id: TensorId,
    position_id: u32,
    cache: &mut Qwen35KvCache,
) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
    model.forward_rollout_cached_device_token_profiled(store, tape, token_id, position_id, cache)
}

impl Qwen35Model {
    pub fn supports_rollout_kv_cache(&self) -> bool {
        !self.tp.is_enabled()
            && self
                .layers
                .iter()
                .all(|layer| matches!(layer.self_attn, Qwen35Attention::Full(_)))
    }

    pub(super) fn ensure_rollout_cache_supported(&self) -> Result<()> {
        if self.tp.is_enabled() {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache requires a non-tensor-parallel train model",
            ));
        }
        Ok(())
    }

    pub(super) fn forward_rollout_cached(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        cache: &mut Qwen35KvCache,
    ) -> Result<TensorId> {
        self.ensure_rollout_cache_supported()?;
        if input_ids.len() != position_ids.len() {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: position_ids.len(),
            });
        }
        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_indices_with_kv_cache(store, tape, &token_indices, &positions, cache)
    }

    pub(super) fn forward_rollout_cached_profiled(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        cache: &mut Qwen35KvCache,
    ) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
        self.ensure_rollout_cache_supported()?;
        if input_ids.len() != position_ids.len() {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: position_ids.len(),
            });
        }
        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_indices_with_kv_cache_profiled(
            store,
            tape,
            &token_indices,
            &positions,
            cache,
        )
    }

    pub(super) fn forward_rollout_cached_device_token(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_id: TensorId,
        position_id: u32,
        cache: &mut Qwen35KvCache,
    ) -> Result<TensorId> {
        self.ensure_rollout_cache_supported()?;
        if tape.enabled {
            return Err(Qwen35Error::InvalidConfig(
                "device-token rollout requires tape disabled",
            ));
        }
        if cache.layers.len() != self.layers.len() {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer count does not match model",
            ));
        }
        let position = position_id as usize;
        if position != cache.seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache device token requires position equal to cache length",
            ));
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if cache.seq_len + 1 > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache sequence length exceeds configured rope cache length",
            ));
        }

        let q_start = cache.seq_len;
        let cos = select_cache_rows(self.cos_cache, &[position], store)?;
        let sin = select_cache_rows(self.sin_cache, &[position], store)?;

        let mut hidden = embedding_device_f32_ids(self.embed_tokens, token_id, 1, store)?;
        hidden = reshape(hidden, &[1, 1, self.config.hidden_size], store, tape)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut cache.layers[layer_index];
            hidden = layer.forward_with_kv_cache(
                hidden,
                &self.config,
                cos,
                sin,
                layer_cache,
                q_start,
                store,
                tape,
            )?;
        }
        cache.seq_len += 1;
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        linear_forward(hidden, self.lm_head, store, tape)
    }

    pub(super) fn forward_rollout_cached_device_token_profiled(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_id: TensorId,
        position_id: u32,
        cache: &mut Qwen35KvCache,
    ) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
        self.ensure_rollout_cache_supported()?;
        let total_started = Instant::now();
        let mut profile = Qwen35RolloutForwardProfile::default();

        if tape.enabled {
            return Err(Qwen35Error::InvalidConfig(
                "device-token rollout requires tape disabled",
            ));
        }
        if cache.layers.len() != self.layers.len() {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer count does not match model",
            ));
        }
        let position = position_id as usize;
        if position != cache.seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache device token requires position equal to cache length",
            ));
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if cache.seq_len + 1 > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache sequence length exceeds configured rope cache length",
            ));
        }

        let q_start = cache.seq_len;
        let started = Instant::now();
        let cos = select_cache_rows(self.cos_cache, &[position], store)?;
        let sin = select_cache_rows(self.sin_cache, &[position], store)?;
        profile.cache_select += started.elapsed();

        let started = Instant::now();
        let mut hidden = embedding_device_f32_ids(self.embed_tokens, token_id, 1, store)?;
        hidden = reshape(hidden, &[1, 1, self.config.hidden_size], store, tape)?;
        profile.embedding += started.elapsed();

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut cache.layers[layer_index];
            let (next_hidden, layer_profile) = layer.forward_with_kv_cache_profiled(
                hidden,
                &self.config,
                cos,
                sin,
                layer_cache,
                q_start,
                store,
                tape,
            )?;
            hidden = next_hidden;
            profile.layers.push(layer_profile);
        }
        cache.seq_len += 1;

        let started = Instant::now();
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        profile.final_norm += started.elapsed();

        let started = Instant::now();
        let logits = linear_forward(hidden, self.lm_head, store, tape)?;
        profile.lm_head += started.elapsed();
        profile.total = total_started.elapsed();
        Ok((logits, profile))
    }

    pub(super) fn forward_batch_indices_with_kv_cache(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        cache: &mut Qwen35KvCache,
    ) -> Result<TensorId> {
        self.ensure_rollout_cache_supported()?;
        let seq_len = positions.len();
        if token_indices.len() != seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: token_indices.len(),
                expected_len: seq_len,
            });
        }
        if cache.layers.len() != self.layers.len() {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer count does not match model",
            ));
        }
        if seq_len == 0 {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache requires at least one token",
            ));
        }
        for (offset, &position) in positions.iter().enumerate() {
            if position != cache.seq_len + offset {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache requires contiguous positions starting at cache length",
                ));
            }
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if cache.seq_len + seq_len > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache sequence length exceeds configured rope cache length",
            ));
        }

        let q_start = cache.seq_len;
        let cos = select_cache_rows(self.cos_cache, positions, store)?;
        let sin = select_cache_rows(self.sin_cache, positions, store)?;

        let mut hidden = embedding(self.embed_tokens, token_indices, store, tape)?;
        hidden = reshape(hidden, &[1, seq_len, self.config.hidden_size], store, tape)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut cache.layers[layer_index];
            hidden = layer.forward_with_kv_cache(
                hidden,
                &self.config,
                cos,
                sin,
                layer_cache,
                q_start,
                store,
                tape,
            )?;
        }
        cache.seq_len += seq_len;
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        let hidden = if seq_len == 1 {
            hidden
        } else {
            slice(
                hidden,
                &[0, seq_len - 1, 0],
                &[1, seq_len, self.config.hidden_size],
                store,
                tape,
            )?
        };
        linear_forward(hidden, self.lm_head, store, tape)
    }

    pub(super) fn forward_batch_indices_with_kv_cache_profiled(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        cache: &mut Qwen35KvCache,
    ) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
        self.ensure_rollout_cache_supported()?;
        let total_started = Instant::now();
        let mut profile = Qwen35RolloutForwardProfile::default();
        let seq_len = positions.len();
        if token_indices.len() != seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: token_indices.len(),
                expected_len: seq_len,
            });
        }
        if cache.layers.len() != self.layers.len() {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer count does not match model",
            ));
        }
        if seq_len == 0 {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache requires at least one token",
            ));
        }
        for (offset, &position) in positions.iter().enumerate() {
            if position != cache.seq_len + offset {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache requires contiguous positions starting at cache length",
                ));
            }
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if cache.seq_len + seq_len > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache sequence length exceeds configured rope cache length",
            ));
        }

        let q_start = cache.seq_len;
        let started = Instant::now();
        let cos = select_cache_rows(self.cos_cache, positions, store)?;
        let sin = select_cache_rows(self.sin_cache, positions, store)?;
        profile.cache_select += started.elapsed();

        let started = Instant::now();
        let mut hidden = embedding(self.embed_tokens, token_indices, store, tape)?;
        hidden = reshape(hidden, &[1, seq_len, self.config.hidden_size], store, tape)?;
        profile.embedding += started.elapsed();

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut cache.layers[layer_index];
            let (next_hidden, layer_profile) = layer.forward_with_kv_cache_profiled(
                hidden,
                &self.config,
                cos,
                sin,
                layer_cache,
                q_start,
                store,
                tape,
            )?;
            hidden = next_hidden;
            profile.layers.push(layer_profile);
        }
        cache.seq_len += seq_len;

        let started = Instant::now();
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        profile.final_norm += started.elapsed();

        let hidden = if seq_len == 1 {
            hidden
        } else {
            slice(
                hidden,
                &[0, seq_len - 1, 0],
                &[1, seq_len, self.config.hidden_size],
                store,
                tape,
            )?
        };

        let started = Instant::now();
        let logits = linear_forward(hidden, self.lm_head, store, tape)?;
        profile.lm_head += started.elapsed();
        profile.total = total_started.elapsed();
        Ok((logits, profile))
    }
}

pub(super) fn embedding_device_f32_ids(
    table: TensorId,
    ids: TensorId,
    n_ids: usize,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let table_shape = store
        .get(table)
        .ok_or(AutogradError::InvalidTensorId(table))?
        .shape
        .clone();
    if table_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: table_shape.len(),
        }
        .into());
    }
    store.ensure_device(table)?;
    store.ensure_device(ids)?;
    let table_handle = store
        .get(table)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "embedding_device_f32_ids: table missing device handle",
        ))?;
    let ids_handle = store
        .get(ids)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "embedding_device_f32_ids: ids missing device handle",
        ))?;
    let out_handle =
        store
            .backend()
            .embedding_from_f32_ids(&table_handle, &table_shape, &ids_handle, n_ids)?;
    Ok(store.alloc_device_tensor(vec![1, n_ids, table_shape[1]], out_handle)?)
}
