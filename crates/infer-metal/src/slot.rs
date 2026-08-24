use super::*;

#[cfg(feature = "metal")]
pub struct MetalSlotState {
    pub(super) slot: usize,
    pub(super) slot_epoch: u64,
    /// Session position: number of tokens whose `step_session` has been issued.
    /// In pipeline mode this runs one ahead of `committed_len` (the prequeued
    /// step). In HEAD mode the two stay equal.
    pub(super) cache_len: usize,
    /// Tokens the engine has committed for this slot (its `kv_seq_len`). Decode
    /// admission is validated against this, not `cache_len`, so the prequeued
    /// step does not trip the seam's length invariant.
    pub(super) committed_len: usize,
    pub(super) kv_flat: Vec<mlx::MlxArray>,
    pub(super) gdr_flat: Vec<mlx::MlxArray>,
    pub(super) session_active: bool,
    /// Deferred sampled token (greedy argmax, async-evaluated) from the most
    /// recent step issued on this slot — the input the next prequeue feeds into
    /// `step_session`. `None` outside pipeline mode.
    pub(super) last_sampled: Option<mlx::MlxArray>,
    pub(super) dflash_target_hidden: Option<mlx::MlxArray>,
    pub(super) dflash_draft_state: Option<dflash::DFlashDraftState>,
    /// EWMA of accepted tokens per draft block (0..block_size).
    pub(super) dflash_ewma_accept: f32,
    /// Blocks to skip before retrying the draft (adaptive fallback).
    pub(super) dflash_skip_remaining: u32,
    /// Consecutive blocks where all draft tokens were rejected. When this
    /// reaches 3, DSpark is disabled for the rest of the session (the draft
    /// model is producing garbage, e.g. missing conv states).
    pub(super) dflash_consecutive_rejects: u32,
}

#[cfg(feature = "metal")]
impl MetalSlotState {
    pub(super) fn new(
        slot: usize,
        slot_epoch: u64,
        config: &config::MetalModelConfig,
        kv_cache_dtype: MetalKvCacheDtype,
        capacity_tokens: usize,
    ) -> Self {
        let capacity = round_up_capacity(capacity_tokens);
        let kv_flat = allocate_kv_flat(config, kv_cache_dtype, capacity);

        let gdr_flat: Vec<mlx::MlxArray> = if let Some(lfm2) = config.arch.lfm2.as_ref() {
            // LFM2: one conv state per gated conv layer — the last (kernel-1)
            // post-gate frames, channels-last to match the C++ conv1d path.
            (0..config.arch.num_conv_layers())
                .map(|_| {
                    mlx::zeros(
                        &[1, (lfm2.conv_kernel - 1) as i32, config.hidden_size as i32],
                        mlx::Dtype::Bfloat16,
                    )
                })
                .collect()
        } else {
            let la = &config.arch.linear;
            (0..config.arch.num_linear_attention_layers())
                .flat_map(|_| {
                    [
                        mlx::zeros(
                            &[
                                1,
                                la.num_value_heads as i32,
                                la.value_dim as i32,
                                la.key_dim as i32,
                            ],
                            mlx::Dtype::Float32,
                        ),
                        mlx::zeros(
                            &[1, (la.conv_kernel - 1) as i32, la.qkv_dim() as i32],
                            mlx::Dtype::Bfloat16,
                        ),
                    ]
                })
                .collect()
        };

        Self {
            slot,
            slot_epoch,
            cache_len: 0,
            committed_len: 0,
            kv_flat,
            gdr_flat,
            session_active: false,
            last_sampled: None,
            dflash_target_hidden: None,
            dflash_draft_state: None,
            dflash_ewma_accept: 4.0,
            dflash_skip_remaining: 0,
            dflash_consecutive_rejects: 0,
        }
    }

    pub(super) fn from_arrays(
        slot: usize,
        slot_epoch: u64,
        cache_len: usize,
        kv_flat: Vec<mlx::MlxArray>,
        gdr_flat: Vec<mlx::MlxArray>,
    ) -> Self {
        Self {
            slot,
            slot_epoch,
            cache_len,
            committed_len: cache_len,
            kv_flat,
            gdr_flat,
            session_active: false,
            last_sampled: None,
            dflash_target_hidden: None,
            dflash_draft_state: None,
            dflash_ewma_accept: 4.0,
            dflash_skip_remaining: 0,
            dflash_consecutive_rejects: 0,
        }
    }

    /// Append captured rows to the rolling target-hidden history (last 64).
    pub(super) fn roll_target_hidden(&mut self, new_rows: mlx::MlxArray) {
        let old = self
            .dflash_target_hidden
            .take()
            .unwrap_or_else(|| new_rows.clone());
        let combined = mlx::concatenate_axis(&[old, new_rows], 0);
        let len = combined.shape().first().copied().unwrap_or(0) as i32;
        self.dflash_target_hidden = Some(if len > 64 {
            let dim = combined.shape().get(1).copied().unwrap_or(0) as i32;
            mlx::slice(&combined, &[len - 64, 0], &[len, dim], &[1, 1])
        } else {
            combined
        });
    }

    pub(super) fn ensure_session_active(
        &mut self,
        model: &dyn CompiledMetalModel,
    ) -> anyhow::Result<()> {
        if self.session_active {
            return Ok(());
        }
        model.session_begin(&self.kv_flat, &self.gdr_flat)?;
        self.session_active = true;
        Ok(())
    }

    pub(super) fn drain_session(&mut self, model: &dyn CompiledMetalModel) -> anyhow::Result<()> {
        if !self.session_active {
            return Ok(());
        }
        let (kv_flat, gdr_flat) = model.session_end(self.kv_flat.len(), self.gdr_flat.len())?;
        self.kv_flat = kv_flat;
        self.gdr_flat = gdr_flat;
        self.session_active = false;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn bf16_prefix_read_inputs(
        &self,
        cache_len: usize,
    ) -> anyhow::Result<(Vec<mlx::MlxArray>, Vec<mlx::MlxArray>)> {
        anyhow::ensure!(
            cache_len <= self.cache_len,
            "paged KV read cache_len {cache_len} exceeds slot cache_len {}",
            self.cache_len
        );
        anyhow::ensure!(
            self.kv_flat.len().is_multiple_of(2),
            "bf16 slot cache must contain K/V pairs, got {} arrays",
            self.kv_flat.len()
        );

        let mut k_full = Vec::with_capacity(self.kv_flat.len() / 2);
        let mut v_full = Vec::with_capacity(self.kv_flat.len() / 2);
        for (layer_idx, pair) in self.kv_flat.chunks_exact(2).enumerate() {
            for (axis, array) in pair.iter().enumerate() {
                anyhow::ensure!(
                    array.dtype() == mlx::Dtype::Bfloat16,
                    "paged KV read expected bf16 layer {layer_idx} axis {axis}, got {:?}",
                    array.dtype()
                );
            }
            k_full.push(slice_kv_tokens(&pair[0], 0, cache_len)?);
            v_full.push(slice_kv_tokens(&pair[1], 0, cache_len)?);
        }
        Ok((k_full, v_full))
    }

    #[cfg(test)]
    pub(super) fn int8_prefix_read_inputs(
        &self,
        cache_len: usize,
    ) -> anyhow::Result<(Vec<mlx::MlxArray>, Vec<mlx::MlxArray>)> {
        anyhow::ensure!(
            cache_len <= self.cache_len,
            "paged INT8 KV read cache_len {cache_len} exceeds slot cache_len {}",
            self.cache_len
        );
        anyhow::ensure!(
            self.kv_flat.len().is_multiple_of(6),
            "int8 slot cache must contain K/V q/scale/bias sextets, got {} arrays",
            self.kv_flat.len()
        );

        let mut k_full = Vec::with_capacity(self.kv_flat.len() / 2);
        let mut v_full = Vec::with_capacity(self.kv_flat.len() / 2);
        for (layer_idx, sextet) in self.kv_flat.chunks_exact(6).enumerate() {
            let expected = [
                mlx::Dtype::Uint32,
                mlx::Dtype::Bfloat16,
                mlx::Dtype::Bfloat16,
                mlx::Dtype::Uint32,
                mlx::Dtype::Bfloat16,
                mlx::Dtype::Bfloat16,
            ];
            for (axis, (array, dtype)) in sextet.iter().zip(expected).enumerate() {
                anyhow::ensure!(
                    array.dtype() == dtype,
                    "paged INT8 KV read expected layer {layer_idx} axis {axis} dtype {:?}, got {:?}",
                    dtype,
                    array.dtype()
                );
            }
            for array in &sextet[..3] {
                k_full.push(slice_kv_tokens(array, 0, cache_len)?);
            }
            for array in &sextet[3..6] {
                v_full.push(slice_kv_tokens(array, 0, cache_len)?);
            }
        }
        Ok((k_full, v_full))
    }

    /// Guarantee the flat K/V cache can hold `cache_len + needed` tokens, growing
    /// the seq axis with zeros when the prefill reservation is exhausted.
    ///
    /// The C++ session writes each step's K/V with `slice_update`, which returns a
    /// *same-shape* array — so the session's capacity is frozen at `begin_session`
    /// and never grows on its own. The host KV pool already grows page-by-page for
    /// arbitrarily long generations; without this the executor's `kv_flat` lags
    /// behind, `slice_update` silently drops out-of-range writes (corrupt output),
    /// and `publish_slot` eventually hard-errors at a page boundary
    /// (`K/V slice token range [..] exceeds shape=[..]`). The prefix-wide
    /// recurrent/conv restore state is sequence-independent (see
    /// `MetalSlotState::new`) and is left untouched, exactly as
    /// `materialize_slot_from_prefix` treats it. Growing mutates `kv_flat`,
    /// which an open session owns, so the session is drained first; the caller
    /// re-activates it via `ensure_session_active`.
    pub(super) fn ensure_kv_capacity(
        &mut self,
        model: &dyn CompiledMetalModel,
        needed: usize,
    ) -> anyhow::Result<()> {
        let capacity = self
            .kv_flat
            .first()
            .map(|array| array.shape().get(2).copied().unwrap_or(0) as usize)
            .unwrap_or(0);
        let required = self.cache_len.saturating_add(needed);
        if capacity == 0 || required <= capacity {
            return Ok(());
        }
        // The open session holds these arrays; drain before reallocating so the
        // grown buffers are the ones the next `begin_session` binds.
        self.drain_session(model)?;
        let new_capacity = round_up_capacity(required.max(capacity.saturating_mul(2))) as usize;
        let grown: Vec<_> = self
            .kv_flat
            .iter()
            .map(|array| grow_kv_seq_axis(array, new_capacity))
            .collect::<anyhow::Result<Vec<_>>>()?;
        // Materialize before re-binding so the concatenation is not replayed
        // lazily on every subsequent step's forward graph.
        let refs: Vec<&mlx::MlxArray> = grown.iter().collect();
        mlx::eval(&refs);
        self.kv_flat = grown;
        Ok(())
    }
}
