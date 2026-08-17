//! The taped training forward and its public entry points.

use super::*;

impl Qwen35Model {
    pub(super) fn detach_before_lora_layer(
        &self,
        hidden: TensorId,
        layer_index: usize,
        store: &mut TensorStore,
        tape: &Tape,
    ) -> Result<TensorId> {
        if tape.enabled && self.lora_layer_start == Some(layer_index) {
            Ok(store.detach(hidden)?)
        } else {
            Ok(hidden)
        }
    }

    pub fn forward_tokens(
        &self,
        input_ids: &[usize],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> autograd::Result<TensorId> {
        self.forward_batch_tokens(input_ids, 1, input_ids.len(), store, tape)
    }

    pub fn forward_batch_tokens(
        &self,
        input_ids: &[usize],
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> autograd::Result<TensorId> {
        let position_ids = (0..seq_len).collect::<Vec<_>>();
        self.forward_batch_tokens_with_positions(input_ids, &position_ids, batch, store, tape)
    }

    pub fn forward_batch_tokens_with_positions(
        &self,
        input_ids: &[usize],
        position_ids: &[usize],
        batch: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> autograd::Result<TensorId> {
        self.forward_batch_indices(store, tape, input_ids, position_ids, batch)
            .map_err(qwen35_to_autograd)
    }

    /// Batched sibling of [`Self::forward_hidden_states`]: post-final-norm hidden
    /// `[batch, seq_len, hidden]` for `input_ids` (row-major, len `batch*seq_len`),
    /// positions `0..seq_len` per row. The logits path (`forward_batch_tokens`) is
    /// this plus `lm_head`; callers that project only a few masked positions (fused
    /// chunked CE) take the hidden and skip the full `[batch, seq_len, vocab]` tile.
    pub fn forward_batch_hidden(
        &self,
        input_ids: &[usize],
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> autograd::Result<TensorId> {
        let position_ids = (0..seq_len).collect::<Vec<_>>();
        self.forward_batch_hidden_indices(
            store,
            tape,
            input_ids,
            &position_ids,
            batch,
            crate::context_parallel::CpContext::single(),
        )
        .map_err(qwen35_to_autograd)
    }

    pub fn forward_batch(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        batch: usize,
        seq_len: usize,
    ) -> Result<TensorId> {
        if input_ids.len() != batch * seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: batch * seq_len,
            });
        }
        if position_ids.len() != seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: position_ids.len(),
                expected_len: seq_len,
            });
        }
        let max_seq_len = self
            .config
            .rope_cache_len_hint
            .ok_or(Qwen35Error::InvalidConfig(
                "train-side qwen3.5 requires rope_cache_len_hint",
            ))?;
        if seq_len > max_seq_len {
            return Err(Qwen35Error::InvalidConfig(
                "sequence length exceeds configured rope cache length",
            ));
        }

        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_indices(store, tape, &token_indices, &positions, batch)
    }

    pub(super) fn forward_batch_indices(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        batch: usize,
    ) -> Result<TensorId> {
        let hidden = self.forward_batch_hidden_indices(
            store,
            tape,
            token_indices,
            positions,
            batch,
            crate::context_parallel::CpContext::single(),
        )?;
        linear_forward(hidden, self.lm_head, store, tape)
    }

    pub(super) fn forward_batch_indices_profiled(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        batch: usize,
        trace: bool,
    ) -> Result<(TensorId, Qwen35RolloutForwardProfile)> {
        let total_started = Instant::now();
        let mut profile = Qwen35RolloutForwardProfile::default();
        let seq_len = positions.len();
        if token_indices.len() != batch * seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: token_indices.len(),
                expected_len: batch * seq_len,
            });
        }

        let started = Instant::now();
        let cos = select_cache_rows(self.cos_cache, positions, store)?;
        let sin = select_cache_rows(self.sin_cache, positions, store)?;
        profile.cache_select += started.elapsed();
        trace_model_component(trace, "cache_select", profile.cache_select);

        let started = Instant::now();
        let mut hidden = embedding(self.embed_tokens, token_indices, store, tape)?;
        hidden = reshape(
            hidden,
            &[batch, seq_len, self.config.hidden_size],
            store,
            tape,
        )?;
        profile.embedding += started.elapsed();
        trace_model_component(trace, "embedding", profile.embedding);

        if self.should_checkpoint(batch, seq_len, store, tape) {
            // Checkpoint layers in adaptive groups (one host offload per
            // group). Numerically exact vs per-layer: same layer order, same
            // detach point (the LoRA boundary forces a group split), same params
            // in each checkpoint's saved inputs.
            let layers = Arc::new(self.layers.clone());
            let cfg = self.config.clone();
            let tp = self.tp;
            let (cos_id, sin_id) = (cos, sin);
            let layer_fn = {
                let layers = Arc::clone(&layers);
                move |idx: usize, h, s: &mut TensorStore, t: &mut Tape| {
                    layers[idx]
                        .forward(
                            h,
                            &cfg,
                            tp,
                            crate::context_parallel::CpContext::single(),
                            cos_id,
                            sin_id,
                            None,
                            s,
                            t,
                        )
                        .map_err(qwen35_to_autograd)
                }
            };
            // Param ids only read `requires_grad`; precompute so `layer_params`
            // doesn't alias the mutable `store` borrow during the call.
            let param_ids: Vec<Vec<TensorId>> = self
                .layers
                .iter()
                .map(|l| l.checkpoint_param_ids(self.lora_skip_experts, store))
                .collect();
            hidden = self.checkpoint_layers(
                hidden,
                batch,
                seq_len,
                store,
                tape,
                |idx| param_ids[idx].clone(),
                layer_fn,
            )?;
            // Grouped path records no per-layer profile; keep profile.layers len.
            profile
                .layers
                .resize_with(self.layers.len(), Qwen35LayerForwardProfile::default);
        } else {
            for (layer_index, layer) in self.layers.iter().enumerate() {
                hidden = self.detach_before_lora_layer(hidden, layer_index, store, tape)?;
                let started = Instant::now();
                let (next_hidden, layer_profile) = layer.forward_profiled(
                    layer_index,
                    hidden,
                    &self.config,
                    self.tp,
                    cos,
                    sin,
                    trace,
                    store,
                    tape,
                )?;
                hidden = next_hidden;
                trace_model_component(trace, "layer_total", started.elapsed());
                profile.layers.push(layer_profile);
            }
        }

        let started = Instant::now();
        let hidden = qwen35_rmsnorm(
            hidden,
            self.final_norm,
            self.config.rms_norm_eps,
            store,
            tape,
        )?;
        profile.final_norm += started.elapsed();
        trace_model_component(trace, "final_norm", profile.final_norm);

        let started = Instant::now();
        let logits = linear_forward(hidden, self.lm_head, store, tape)?;
        profile.lm_head += started.elapsed();
        trace_model_component(trace, "lm_head", profile.lm_head);
        profile.total = total_started.elapsed();
        trace_model_component(trace, "total", profile.total);
        Ok((logits, profile))
    }

    pub(super) fn forward_batch_hidden_indices(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        batch: usize,
        cp: crate::context_parallel::CpContext,
    ) -> Result<TensorId> {
        self.forward_batch_hidden_indices_retaining(
            store,
            tape,
            token_indices,
            positions,
            batch,
            cp,
            false,
        )
    }

    /// Like `forward_batch_hidden_indices`, but frees per-layer scratch from
    /// the store after each layer. The teacher full-sequence forward at 65K
    /// tokens OOMs without this: the store would otherwise hold every layer's
    /// linear-attention scratch (~3 GB/layer) until cleanup_after_backward.
    /// `retain_set` lists tensor IDs that must survive the per-layer pruning
    /// (e.g. student params, LoRA adapters, optimizer state).
    pub fn forward_hidden_freeing_intermediates(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        batch: usize,
        cp: crate::context_parallel::CpContext,
    ) -> Result<TensorId> {
        self.forward_batch_hidden_indices_retaining(
            store,
            tape,
            token_indices,
            positions,
            batch,
            cp,
            true,
        )
    }

    fn forward_batch_hidden_indices_retaining(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        token_indices: &[usize],
        positions: &[usize],
        batch: usize,
        cp: crate::context_parallel::CpContext,
        free_intermediates: bool,
    ) -> Result<TensorId> {
        let seq_len = positions.len();
        if token_indices.len() != batch * seq_len {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: token_indices.len(),
                expected_len: batch * seq_len,
            });
        }
        let cos = select_cache_rows(self.cos_cache, positions, store)?;
        let sin = select_cache_rows(self.sin_cache, positions, store)?;

        // Under CP the ring masks by the shard's absolute row positions — the SAME
        // slice that built cos/sin above. Shared as `Arc<[usize]>` so the `'static`
        // checkpoint closure can capture it; `None` off-CP (contiguous, unused).
        let cp_positions: Option<Arc<[usize]>> =
            cp.is_enabled().then(|| Arc::from(positions.to_vec()));

        let mut hidden = embedding(self.embed_tokens, token_indices, store, tape)?;
        hidden = reshape(
            hidden,
            &[batch, seq_len, self.config.hidden_size],
            store,
            tape,
        )?;
        if self.should_checkpoint(batch, seq_len, store, tape) {
            // Group-checkpoint (see forward_batch_indices_profiled) — exact vs
            // per-layer.
            let layers = Arc::new(self.layers.clone());
            let cfg = self.config.clone();
            let tp = self.tp;
            let (cos_id, sin_id) = (cos, sin);

            // Per-layer wall aggregation for ARLE_OPD_PROFILE=1 (this is the
            // masked-writeback forward path). Each layer_fn call is timed and —
            // when ARLE_OPD_PROFILE_SYNC=1 — stream-synced so the measurement
            // captures kernel wall not enqueue latency. Backward recompute
            // re-invokes layer_fn, so counts[idx] surfaces the recompute
            // multiplier. Both env vars default OFF: shipping path unchanged.
            let profile_enabled = std::env::var("ARLE_OPD_PROFILE").is_ok();
            let profile_sync = std::env::var("ARLE_OPD_PROFILE_SYNC").is_ok();
            let num_layers = self.layers.len();
            let layer_times: Arc<Mutex<Vec<Duration>>> =
                Arc::new(Mutex::new(vec![Duration::default(); num_layers]));
            let layer_counts: Arc<Mutex<Vec<usize>>> =
                Arc::new(Mutex::new(vec![0usize; num_layers]));

            let layer_fn = {
                let layers = Arc::clone(&layers);
                let layer_times = Arc::clone(&layer_times);
                let layer_counts = Arc::clone(&layer_counts);
                let cp_positions = cp_positions.clone();
                move |idx: usize, h, s: &mut TensorStore, t: &mut Tape| {
                    let pos = cp_positions.as_deref();
                    if !profile_enabled {
                        return layers[idx]
                            .forward(h, &cfg, tp, cp, cos_id, sin_id, pos, s, t)
                            .map_err(qwen35_to_autograd);
                    }
                    if profile_sync {
                        let _ = s.backend().stream_synchronize();
                    }
                    let t0 = Instant::now();
                    let result = layers[idx]
                        .forward(h, &cfg, tp, cp, cos_id, sin_id, pos, s, t)
                        .map_err(qwen35_to_autograd);
                    if profile_sync {
                        let _ = s.backend().stream_synchronize();
                    }
                    let dt = t0.elapsed();
                    if let Ok(mut times) = layer_times.lock() {
                        times[idx] += dt;
                    }
                    if let Ok(mut counts) = layer_counts.lock() {
                        counts[idx] += 1;
                    }
                    result
                }
            };
            // See forward_batch_indices_profiled: precompute param ids so the
            // closure doesn't alias the mutable `store` borrow.
            let param_ids: Vec<Vec<TensorId>> = self
                .layers
                .iter()
                .map(|l| l.checkpoint_param_ids(self.lora_skip_experts, store))
                .collect();
            hidden = self.checkpoint_layers(
                hidden,
                batch,
                seq_len,
                store,
                tape,
                |idx| param_ids[idx].clone(),
                layer_fn,
            )?;

            if profile_enabled
                && let (Ok(times), Ok(counts)) = (layer_times.lock(), layer_counts.lock())
            {
                let total: Duration = times.iter().sum();
                eprintln!(
                    "[opd-profile] masked-writeback forward layer wall \
                         (checkpointed, sync={profile_sync}); counts include backward recompute:"
                );
                for (idx, (t, c)) in times.iter().zip(counts.iter()).enumerate() {
                    eprintln!(
                        "[opd-profile]   layer[{idx:>2}] wall={:>10.3}ms calls={}",
                        t.as_secs_f64() * 1000.0,
                        c
                    );
                }
                eprintln!(
                    "[opd-profile] forward layers sum: {:.3}s across {num_layers} layers",
                    total.as_secs_f64()
                );
            }
        } else {
            // #170 attribution probe: full-tape per-layer VRAM ramp. free is
            // pool-reserved-aware (cuMemGetInfo), so slope = tape + transients.
            let vram_ramp = std::env::var("ARLE_OPD_VRAM_TRACE").is_ok();
            let layer_trace = std::env::var("ARLE_OPD_LAYER_TRACE").is_ok();
            // When freeing intermediates (teacher full-seq forward), snapshot
            // live IDs before the layer loop and free per-layer scratch after
            // each layer via free_new_except — only tensors created during
            // this forward are freed; pre-existing store entries are untouched.
            let live_before: Option<HashSet<TensorId>> =
                free_intermediates.then(|| store.live_ids().into_iter().collect());
            for (layer_index, layer) in self.layers.iter().enumerate() {
                hidden = self.detach_before_lora_layer(hidden, layer_index, store, tape)?;
                hidden = layer.forward(
                    hidden,
                    &self.config,
                    self.tp,
                    cp,
                    cos,
                    sin,
                    cp_positions.as_deref(),
                    store,
                    tape,
                )?;
                if let Some(before) = &live_before {
                    let _ = store.free_new_except(before, &HashSet::from([hidden]));
                }
                if vram_ramp && let Some((free, _)) = store.backend().device_mem_info() {
                    eprintln!("[vram-ramp] layer={layer_index} free={free}");
                }
                if layer_trace {
                    let sq = store.get(hidden).and_then(|t| {
                        t.device_handle
                            .as_ref()
                            .and_then(|h| store.backend().sum_squares(h, &t.shape).ok())
                    });
                    eprintln!("[layer-trace] layer={layer_index} hidden_sum_sq={sq:?}");
                }
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

    pub fn forward(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> Result<TensorId> {
        self.forward_batch(store, tape, input_ids, position_ids, 1, position_ids.len())
    }

    pub fn forward_hidden_states(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        cp: crate::context_parallel::CpContext,
    ) -> Result<TensorId> {
        if input_ids.len() != position_ids.len() {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: input_ids.len(),
                expected_len: position_ids.len(),
            });
        }
        if input_ids.is_empty() {
            return Err(Qwen35Error::InvalidConfig(
                "hidden-state forward requires at least one token",
            ));
        }
        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let positions = position_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        self.forward_batch_hidden_indices(store, tape, &token_indices, &positions, 1, cp)
    }

    pub fn logits_from_hidden_window(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        hidden: TensorId,
        window: SequenceWindow,
    ) -> Result<TensorId> {
        if window.start >= window.end {
            return Err(Qwen35Error::InvalidConfig(
                "hidden logits window must be non-empty",
            ));
        }
        let hidden_shape = store
            .get(hidden)
            .ok_or(AutogradError::InvalidTensorId(hidden))?
            .shape
            .clone();
        if hidden_shape.len() != 3 {
            return Err(AutogradError::InvalidRank {
                expected: "3",
                got: hidden_shape.len(),
            }
            .into());
        }
        if hidden_shape[0] != 1 || hidden_shape[2] != self.config.hidden_size {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![1, hidden_shape[1], self.config.hidden_size],
                got: hidden_shape.clone(),
            }
            .into());
        }
        if window.end > hidden_shape[1] {
            return Err(Qwen35Error::InputLenMismatch {
                input_len: window.end,
                expected_len: hidden_shape[1],
            });
        }
        let hidden_window = if window.start == 0 && window.end == hidden_shape[1] {
            hidden
        } else {
            slice(
                hidden,
                &[0, window.start, 0],
                &[1, window.end, self.config.hidden_size],
                store,
                tape,
            )?
        };
        linear_forward(hidden_window, self.lm_head, store, tape)
    }

    pub fn forward_logits_window(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        window: SequenceWindow,
    ) -> Result<TensorId> {
        validate_sequence_window(input_ids, position_ids, window)?;
        let prefix_len = window.end;
        let token_indices = input_ids[..prefix_len]
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let positions = position_ids[..prefix_len]
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let hidden = self.forward_batch_hidden_indices(
            store,
            tape,
            &token_indices,
            &positions,
            1,
            crate::context_parallel::CpContext::single(),
        )?;
        self.logits_from_hidden_window(store, tape, hidden, window)
    }
}
