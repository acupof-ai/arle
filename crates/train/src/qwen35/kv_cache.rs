use super::*;

impl Qwen35KvCache {
    pub fn new(model: &Qwen35Model, max_seq_len: usize) -> Self {
        Self {
            layers: vec![
                Qwen35LayerKvCache {
                    k: None,
                    v: None,
                    max_seq_len,
                    seq_cursor: 0,
                };
                model.layers.len()
            ],
            seq_len: 0,
        }
    }

    pub fn extend_tensor_ids(&self, keep: &mut HashSet<TensorId>) {
        for layer in &self.layers {
            if let Some(k) = layer.k {
                keep.insert(k);
            }
            if let Some(v) = layer.v {
                keep.insert(v);
            }
        }
    }
}

impl Qwen35Layer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_full_attention_with_kv_cache(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        layer_cache: &mut Qwen35LayerKvCache,
        q_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let q_full = attn.q_proj.forward(h, store, tape)?;
        let decode_prepare_fast = !tape.enabled
            && seq_len == 1
            && cfg.rotary_dim == cfg.head_dim
            && store.backend().device() == Device::Cuda;
        let (q, gate, k, v) = if decode_prepare_fast {
            let (q, gate) =
                qwen_decode_prepare_q(q_full, attn.q_norm, cos, sin, cfg, batch, store)?;
            let k = attn.k_proj.forward(h, store, tape)?;
            let v = attn.v_proj.forward(h, store, tape)?;
            let (k, v) = qwen_decode_prepare_kv(k, v, attn.k_norm, cos, sin, cfg, batch, store)?;
            (q, gate, k, v)
        } else {
            let (q, gate) = if cfg.full_attn_gated {
                let q_full = reshape(
                    q_full,
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim * 2],
                    store,
                    tape,
                )?;
                let q = slice(
                    q_full,
                    &[0, 0, 0, 0],
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim],
                    store,
                    tape,
                )?;
                let gate = slice(
                    q_full,
                    &[0, 0, 0, cfg.head_dim],
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim * 2],
                    store,
                    tape,
                )?;
                (
                    transpose(q, 1, 2, store, tape)?,
                    Some(transpose(gate, 1, 2, store, tape)?),
                )
            } else {
                let q = reshape(
                    q_full,
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim],
                    store,
                    tape,
                )?;
                (transpose(q, 1, 2, store, tape)?, None)
            };

            let k = attn.k_proj.forward(h, store, tape)?;
            let v = attn.v_proj.forward(h, store, tape)?;
            let k = split_heads(
                k,
                batch,
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                store,
                tape,
            )?;
            let v = split_heads(
                v,
                batch,
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                store,
                tape,
            )?;

            let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
            let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
            let q = rope(q, cos, sin, store, tape)?;
            let k = rope(k, cos, sin, store, tape)?;
            (q, gate, k, v)
        };

        if layer_cache.seq_cursor != q_start {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer cursor diverged from global cache length",
            ));
        }
        let prev_kv_len = layer_cache.seq_cursor;
        let k_cache = append_cached_kv(
            layer_cache.k,
            k,
            layer_cache.max_seq_len,
            prev_kv_len,
            store,
        )?;
        let v_cache = append_cached_kv(
            layer_cache.v,
            v,
            layer_cache.max_seq_len,
            prev_kv_len,
            store,
        )?;
        let kv_len = prev_kv_len + seq_len;
        layer_cache.k = Some(k_cache);
        layer_cache.v = Some(v_cache);
        layer_cache.seq_cursor = kv_len;

        let kv_repeat = cfg.num_attention_heads / cfg.num_key_value_heads;
        let attn_hidden = if !tape.enabled && seq_len == 1 && q_start + 1 == kv_len {
            causal_sdpa_decode_gqa_cached(q, k_cache, v_cache, kv_len, q_start, store, tape)?
        } else {
            if prev_kv_len != 0 {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache only supports one-token decode after the initial prompt",
                ));
            }
            let k_all = repeat_kv(k, kv_repeat, store, tape)?;
            let v_all = repeat_kv(v, kv_repeat, store, tape)?;
            causal_sdpa_with_q_start(q, k_all, v_all, q_start, store, tape)?
        };
        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            seq_len,
            cfg.num_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        Ok(attn.o_proj.forward(attn_hidden, store, tape)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_full_attention_with_kv_cache_profiled(
        &self,
        h: TensorId,
        attn: &Qwen35FullAttention,
        cfg: &Qwen35Config,
        cos: TensorId,
        sin: TensorId,
        batch: usize,
        seq_len: usize,
        layer_cache: &mut Qwen35LayerKvCache,
        q_start: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
        profile: &mut Qwen35AttentionForwardProfile,
    ) -> Result<TensorId> {
        let started = Instant::now();
        let q_full = attn.q_proj.forward(h, store, tape)?;
        profile.q_proj += started.elapsed();

        let decode_prepare_fast = !tape.enabled
            && seq_len == 1
            && cfg.rotary_dim == cfg.head_dim
            && store.backend().device() == Device::Cuda;
        let (q, gate, k, v) = if decode_prepare_fast {
            let started = Instant::now();
            let (q, gate) =
                qwen_decode_prepare_q(q_full, attn.q_norm, cos, sin, cfg, batch, store)?;
            profile.q_layout += started.elapsed();

            let started = Instant::now();
            let k = attn.k_proj.forward(h, store, tape)?;
            profile.k_proj += started.elapsed();

            let started = Instant::now();
            let v = attn.v_proj.forward(h, store, tape)?;
            profile.v_proj += started.elapsed();

            let started = Instant::now();
            let (k, v) = qwen_decode_prepare_kv(k, v, attn.k_norm, cos, sin, cfg, batch, store)?;
            profile.kv_split += started.elapsed();
            (q, gate, k, v)
        } else {
            let started = Instant::now();
            let (q, gate) = if cfg.full_attn_gated {
                let q_full = reshape(
                    q_full,
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim * 2],
                    store,
                    tape,
                )?;
                let q = slice(
                    q_full,
                    &[0, 0, 0, 0],
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim],
                    store,
                    tape,
                )?;
                let gate = slice(
                    q_full,
                    &[0, 0, 0, cfg.head_dim],
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim * 2],
                    store,
                    tape,
                )?;
                (
                    transpose(q, 1, 2, store, tape)?,
                    Some(transpose(gate, 1, 2, store, tape)?),
                )
            } else {
                let q = reshape(
                    q_full,
                    &[batch, seq_len, cfg.num_attention_heads, cfg.head_dim],
                    store,
                    tape,
                )?;
                (transpose(q, 1, 2, store, tape)?, None)
            };
            profile.q_layout += started.elapsed();

            let started = Instant::now();
            let k = attn.k_proj.forward(h, store, tape)?;
            profile.k_proj += started.elapsed();

            let started = Instant::now();
            let v = attn.v_proj.forward(h, store, tape)?;
            profile.v_proj += started.elapsed();

            let started = Instant::now();
            let k = split_heads(
                k,
                batch,
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                store,
                tape,
            )?;
            let v = split_heads(
                v,
                batch,
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                store,
                tape,
            )?;
            profile.kv_split += started.elapsed();

            let started = Instant::now();
            let q = qwen35_rmsnorm(q, attn.q_norm, cfg.rms_norm_eps, store, tape)?;
            let k = qwen35_rmsnorm(k, attn.k_norm, cfg.rms_norm_eps, store, tape)?;
            profile.qk_norm += started.elapsed();

            let started = Instant::now();
            let q = rope(q, cos, sin, store, tape)?;
            let k = rope(k, cos, sin, store, tape)?;
            profile.rope += started.elapsed();
            (q, gate, k, v)
        };

        let started = Instant::now();
        if layer_cache.seq_cursor != q_start {
            return Err(Qwen35Error::InvalidConfig(
                "rollout KV cache layer cursor diverged from global cache length",
            ));
        }
        let prev_kv_len = layer_cache.seq_cursor;
        let k_cache = append_cached_kv(
            layer_cache.k,
            k,
            layer_cache.max_seq_len,
            prev_kv_len,
            store,
        )?;
        let v_cache = append_cached_kv(
            layer_cache.v,
            v,
            layer_cache.max_seq_len,
            prev_kv_len,
            store,
        )?;
        let kv_len = prev_kv_len + seq_len;
        layer_cache.k = Some(k_cache);
        layer_cache.v = Some(v_cache);
        layer_cache.seq_cursor = kv_len;
        profile.append_kv += started.elapsed();

        let kv_repeat = cfg.num_attention_heads / cfg.num_key_value_heads;
        let attn_hidden = if !tape.enabled && seq_len == 1 && q_start + 1 == kv_len {
            let started = Instant::now();
            let out =
                causal_sdpa_decode_gqa_cached(q, k_cache, v_cache, kv_len, q_start, store, tape)?;
            profile.sdpa += started.elapsed();
            out
        } else {
            if prev_kv_len != 0 {
                return Err(Qwen35Error::InvalidConfig(
                    "rollout KV cache only supports one-token decode after the initial prompt",
                ));
            }
            let repeat_started = Instant::now();
            let k_all = repeat_kv(k, kv_repeat, store, tape)?;
            let v_all = repeat_kv(v, kv_repeat, store, tape)?;
            profile.repeat_kv += repeat_started.elapsed();
            let started = Instant::now();
            let out = causal_sdpa_with_q_start(q, k_all, v_all, q_start, store, tape)?;
            profile.sdpa += started.elapsed();
            out
        };

        let started = Instant::now();
        let attn_hidden = if let Some(gate) = gate {
            let gate = sigmoid(gate, store, tape)?;
            mul(attn_hidden, gate, store, tape)?
        } else {
            attn_hidden
        };
        profile.gate += started.elapsed();

        let started = Instant::now();
        let attn_hidden = merge_heads(
            attn_hidden,
            batch,
            seq_len,
            cfg.num_attention_heads,
            cfg.head_dim,
            store,
            tape,
        )?;
        profile.merge += started.elapsed();

        let started = Instant::now();
        let out = attn.o_proj.forward(attn_hidden, store, tape)?;
        profile.o_proj += started.elapsed();
        Ok(out)
    }
}

pub(super) fn append_cached_kv(
    cached: Option<TensorId>,
    next: TensorId,
    max_seq_len: usize,
    seq_cursor: usize,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let next_shape = store
        .get(next)
        .ok_or(AutogradError::InvalidTensorId(next))?
        .shape
        .clone();
    if next_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "rank-4 KV tensor [batch, heads, seq, dim]",
            got: next_shape.len(),
        }
        .into());
    }
    let next_seq_len = next_shape[2];
    if seq_cursor + next_seq_len > max_seq_len {
        return Err(Qwen35Error::InvalidConfig(
            "rollout KV cache append exceeds preallocated max_seq_len",
        ));
    }

    let cached = if let Some(cached) = cached {
        cached
    } else {
        let cache_shape = vec![next_shape[0], next_shape[1], max_seq_len, next_shape[3]];
        let handle = store.backend().zeros(&cache_shape)?;
        store.alloc_device_tensor(cache_shape, handle)?
    };

    let cached_shape = store
        .get(cached)
        .ok_or(AutogradError::InvalidTensorId(cached))?
        .shape
        .clone();

    store.ensure_device(cached)?;
    store.ensure_device(next)?;
    let cached_handle = store
        .get(cached)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "append_cached_kv: cached tensor missing device handle",
        ))?;
    let next_handle = store
        .get(next)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "append_cached_kv: next tensor missing device handle",
        ))?;
    let out_handle = store.backend().kv_cache_write_axis2(
        &cached_handle,
        &cached_shape,
        &next_handle,
        &next_shape,
        seq_cursor,
    )?;
    store.replace_device_handle(cached, out_handle)?;
    Ok(cached)
}

pub(super) fn causal_sdpa_decode_gqa_cached(
    q: TensorId,
    k_cache: TensorId,
    v_cache: TensorId,
    kv_len: usize,
    q_start: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if tape.enabled {
        return Err(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa_cached is rollout-only and requires tape disabled",
        )
        .into());
    }
    let q_shape = store
        .get(q)
        .ok_or(AutogradError::InvalidTensorId(q))?
        .shape
        .clone();
    let k_shape = store
        .get(k_cache)
        .ok_or(AutogradError::InvalidTensorId(k_cache))?
        .shape
        .clone();
    let v_shape = store
        .get(v_cache)
        .ok_or(AutogradError::InvalidTensorId(v_cache))?
        .shape
        .clone();
    store.ensure_device(q)?;
    store.ensure_device(k_cache)?;
    store.ensure_device(v_cache)?;
    let q_handle = store
        .get(q)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa_cached: q missing device handle",
        ))?;
    let k_handle = store
        .get(k_cache)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa_cached: k missing device handle",
        ))?;
    let v_handle = store
        .get(v_cache)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "causal_sdpa_decode_gqa_cached: v missing device handle",
        ))?;
    let (out_handle, out_shape) = store.backend().causal_sdpa_decode_gqa_cache(
        &q_handle, &q_shape, &k_handle, &k_shape, &v_handle, &v_shape, kv_len, q_start,
    )?;
    Ok(store.alloc_device_tensor(out_shape, out_handle)?)
}

pub(super) fn qwen_decode_prepare_q(
    q_full: TensorId,
    q_norm: TensorId,
    cos: TensorId,
    sin: TensorId,
    cfg: &Qwen35Config,
    batch: usize,
    store: &mut TensorStore,
) -> Result<(TensorId, Option<TensorId>)> {
    store.ensure_device(q_full)?;
    store.ensure_device(q_norm)?;
    store.ensure_device(cos)?;
    store.ensure_device(sin)?;

    let q_full_shape = store
        .get(q_full)
        .ok_or(AutogradError::InvalidTensorId(q_full))?
        .shape
        .clone();
    if q_full_shape.len() != 3 || q_full_shape[0] != batch || q_full_shape[1] != 1 {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![batch, 1, cfg.full_attn_q_proj_dim()],
            got: q_full_shape,
        }
        .into());
    }
    let q_norm_shape = store
        .get(q_norm)
        .ok_or(AutogradError::InvalidTensorId(q_norm))?
        .shape
        .clone();
    let cos_shape = store
        .get(cos)
        .ok_or(AutogradError::InvalidTensorId(cos))?
        .shape
        .clone();
    let sin_shape = store
        .get(sin)
        .ok_or(AutogradError::InvalidTensorId(sin))?
        .shape
        .clone();
    let q_full_handle = store
        .get(q_full)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_q: q_full missing device handle",
        ))?;
    let q_norm_handle = store
        .get(q_norm)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_q: q_norm missing device handle",
        ))?;
    let cos_handle = store
        .get(cos)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_q: cos missing device handle",
        ))?;
    let sin_handle = store
        .get(sin)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_q: sin missing device handle",
        ))?;

    let (q_handle, gate_handle, out_shape) = store.backend().qwen_decode_prepare_q(
        &q_full_handle,
        &q_full_shape,
        &q_norm_handle,
        &q_norm_shape,
        &cos_handle,
        &cos_shape,
        &sin_handle,
        &sin_shape,
        cfg.num_attention_heads,
        cfg.head_dim,
        cfg.full_attn_gated,
        cfg.rms_norm_eps,
    )?;
    let q = store.alloc_device_tensor(out_shape.clone(), q_handle)?;
    let gate = gate_handle
        .map(|handle| store.alloc_device_tensor(out_shape, handle))
        .transpose()?;
    Ok((q, gate))
}

pub(super) fn qwen_decode_prepare_kv(
    k_full: TensorId,
    v_full: TensorId,
    k_norm: TensorId,
    cos: TensorId,
    sin: TensorId,
    cfg: &Qwen35Config,
    batch: usize,
    store: &mut TensorStore,
) -> Result<(TensorId, TensorId)> {
    store.ensure_device(k_full)?;
    store.ensure_device(v_full)?;
    store.ensure_device(k_norm)?;
    store.ensure_device(cos)?;
    store.ensure_device(sin)?;

    let k_full_shape = store
        .get(k_full)
        .ok_or(AutogradError::InvalidTensorId(k_full))?
        .shape
        .clone();
    if k_full_shape.len() != 3
        || k_full_shape[0] != batch
        || k_full_shape[1] != 1
        || k_full_shape[2] != cfg.num_key_value_heads * cfg.head_dim
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![batch, 1, cfg.num_key_value_heads * cfg.head_dim],
            got: k_full_shape,
        }
        .into());
    }
    let v_full_shape = store
        .get(v_full)
        .ok_or(AutogradError::InvalidTensorId(v_full))?
        .shape
        .clone();
    let k_norm_shape = store
        .get(k_norm)
        .ok_or(AutogradError::InvalidTensorId(k_norm))?
        .shape
        .clone();
    let cos_shape = store
        .get(cos)
        .ok_or(AutogradError::InvalidTensorId(cos))?
        .shape
        .clone();
    let sin_shape = store
        .get(sin)
        .ok_or(AutogradError::InvalidTensorId(sin))?
        .shape
        .clone();
    let k_full_handle = store
        .get(k_full)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_kv: k_full missing device handle",
        ))?;
    let v_full_handle = store
        .get(v_full)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_kv: v_full missing device handle",
        ))?;
    let k_norm_handle = store
        .get(k_norm)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_kv: k_norm missing device handle",
        ))?;
    let cos_handle = store
        .get(cos)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_kv: cos missing device handle",
        ))?;
    let sin_handle = store
        .get(sin)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "qwen_decode_prepare_kv: sin missing device handle",
        ))?;

    let (k_handle, v_handle, out_shape) = store.backend().qwen_decode_prepare_kv(
        &k_full_handle,
        &k_full_shape,
        &v_full_handle,
        &v_full_shape,
        &k_norm_handle,
        &k_norm_shape,
        &cos_handle,
        &cos_shape,
        &sin_handle,
        &sin_shape,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.rms_norm_eps,
    )?;
    let k = store.alloc_device_tensor(out_shape.clone(), k_handle)?;
    let v = store.alloc_device_tensor(out_shape, v_handle)?;
    Ok((k, v))
}

pub(super) fn select_cache_rows(
    cache: TensorId,
    position_ids: &[usize],
    store: &mut TensorStore,
) -> Result<TensorId> {
    let cache_tensor = store
        .get(cache)
        .ok_or(AutogradError::InvalidTensorId(cache))?;
    if cache_tensor.shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: cache_tensor.shape.len(),
        }
        .into());
    }

    let rows = cache_tensor.shape[0];
    let cols = cache_tensor.shape[1];
    let mut data = Vec::with_capacity(position_ids.len() * cols);
    for &position in position_ids {
        if position >= rows {
            return Err(Qwen35Error::PositionOutOfBounds {
                position,
                upper: rows,
            });
        }
        let base = position * cols;
        data.extend_from_slice(&cache_tensor.data[base..base + cols]);
    }
    let output_shape = vec![position_ids.len(), cols];
    Ok(store.alloc(Tensor::new(data, output_shape, false)?))
}
