//! Per-request slot state for the Metal executor.

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
    /// Session KV-recall plan for the next decode: when `Some`, the step attends
    /// only these token ranges (sink ∪ recalled blocks ∪ local) via the page-gather
    /// instead of the full `[0..cache_len]`. `None` = today's contiguous read.
    /// Set by the Engine from `infer_core::plan_recall` (#5); default off.
    pub(super) recall_ranges: Option<Vec<(usize, usize)>>,
    /// Resident per-middle-block mean-key reps for KV-recall scoring (#2). Each
    /// entry is the layer-0 K mean-pooled over its `l_bs` tokens, flattened to
    /// `nkv * hd` f32 — the resident representative that keeps an offloaded block
    /// scorable (`q · rep`). Index = middle block index (token base
    /// `n_init + i * l_bs`). Grown incrementally as whole blocks complete; empty
    /// unless recall is enabled for this slot.
    pub(super) block_reps: Vec<Vec<f32>>,
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

        let la = &config.arch.linear;
        let gdr_flat: Vec<mlx::MlxArray> = (0..config.arch.num_linear_attention_layers())
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
            .collect();

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
            recall_ranges: None,
            block_reps: Vec::new(),
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
            recall_ranges: None,
            block_reps: Vec::new(),
        }
    }

    pub(super) fn ensure_session_active(
        &mut self,
        model: &qwen35::CppQwen35Model,
    ) -> anyhow::Result<()> {
        if self.session_active {
            return Ok(());
        }
        model.begin_session(&self.kv_flat, &self.gdr_flat)?;
        self.session_active = true;
        Ok(())
    }

    pub(super) fn drain_session(&mut self, model: &qwen35::CppQwen35Model) -> anyhow::Result<()> {
        if !self.session_active {
            return Ok(());
        }
        let (kv_flat, gdr_flat) = model.end_session(self.kv_flat.len(), self.gdr_flat.len())?;
        self.kv_flat = kv_flat;
        self.gdr_flat = gdr_flat;
        self.session_active = false;
        Ok(())
    }

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

    /// Page-gather read for session KV-recall (Phase-1 #4 primitive). Builds the
    /// decode K/V from a SELECTED set of contiguous token ranges
    /// (sink ∪ recalled blocks ∪ local window) instead of the full `[0..cache_len]`,
    /// by slicing each range and concatenating along the token axis — reuses
    /// `slice_kv_tokens` + `concatenate_or_single`, no new MLX op. Which ranges to
    /// recall is the device-neutral policy (infer-core SessionMemory); this is only
    /// the executor primitive that attends them. `ranges` must be within `cache_len`.
    pub(super) fn bf16_recall_read_inputs(
        &self,
        ranges: &[(usize, usize)],
    ) -> anyhow::Result<(Vec<mlx::MlxArray>, Vec<mlx::MlxArray>)> {
        anyhow::ensure!(
            !ranges.is_empty(),
            "recall KV read requires at least one token range"
        );
        for &(s, e) in ranges {
            anyhow::ensure!(
                s <= e && e <= self.cache_len,
                "recall token range [{s}, {e}) exceeds slot cache_len {}",
                self.cache_len
            );
        }
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
                    "recall KV read expected bf16 layer {layer_idx} axis {axis}, got {:?}",
                    array.dtype()
                );
            }
            k_full.push(gather_kv_ranges(&pair[0], ranges)?);
            v_full.push(gather_kv_ranges(&pair[1], ranges)?);
        }
        Ok((k_full, v_full))
    }

    /// Session KV-recall reps (#2): mean-pool layer-0 K over each frozen middle
    /// block into a resident `[nkv, hd]` representative (flattened `nkv*hd` f32).
    /// A block is "frozen" once it has left the local window
    /// (`base + l_bs <= cache_len - n_local`), so its K is final and the rep is
    /// computed exactly once. This is the resident scoring substrate: even after
    /// a block's full KV is offloaded its rep stays here, keeping `q · rep`
    /// scorable. Cheap — only newly-completed blocks are recomputed each step.
    /// The mean is taken on the host from the (tiny) block K slice.
    pub(super) fn update_block_reps(
        &mut self,
        cfg: &infer_core::RecallConfig,
    ) -> anyhow::Result<()> {
        if cfg.l_bs == 0 || self.cache_len <= cfg.n_init + cfg.n_local {
            return Ok(());
        }
        let mid_span = self.cache_len - cfg.n_init - cfg.n_local;
        let frozen_blocks = mid_span / cfg.l_bs;
        if frozen_blocks <= self.block_reps.len() {
            return Ok(());
        }
        let Some(k0) = self.kv_flat.first() else {
            return Ok(());
        };
        anyhow::ensure!(
            k0.dtype() == mlx::Dtype::Bfloat16,
            "recall reps expect bf16 layer-0 K, got {:?}",
            k0.dtype()
        );
        let shape = k0.shape();
        anyhow::ensure!(shape.len() == 4, "recall reps expect rank-4 K");
        let nkv = shape[1] as usize;
        let hd = shape[3] as usize;
        let l_bs_f = cfg.l_bs as f32;
        for block in self.block_reps.len()..frozen_blocks {
            let base = cfg.n_init + block * cfg.l_bs;
            let slice = slice_kv_tokens(k0, base, base + cfg.l_bs)?; // [1, nkv, l_bs, hd]
            let f32_slice = mlx::as_dtype(&slice, mlx::Dtype::Float32);
            mlx::eval(&[&f32_slice]);
            let data = f32_slice.as_slice_f32(); // row-major [1, nkv, l_bs, hd]
            let mut rep = vec![0.0_f32; nkv * hd];
            for h in 0..nkv {
                for t in 0..cfg.l_bs {
                    let row = (h * cfg.l_bs + t) * hd;
                    let out = h * hd;
                    for d in 0..hd {
                        rep[out + d] += data[row + d];
                    }
                }
            }
            for v in &mut rep {
                *v /= l_bs_f;
            }
            self.block_reps.push(rep);
        }
        Ok(())
    }

    /// Session KV-recall plan (#2/#3): score the resident block reps against the
    /// just-emitted layer-0 decode query (`q · rep`, one step stale — licensed),
    /// run `infer_core::plan_recall`, and stash the result for the NEXT step.
    /// `recall_ranges` is set to `None` when the plan is the single contiguous
    /// range (session still fits the budget) so the default page-read stays
    /// byte-identical. Requires bf16 KV (the only recall-built path); the query
    /// is the C++ emit (`take_recall_query`).
    pub(super) fn recompute_recall_plan(
        &mut self,
        model: &qwen35::CppQwen35Model,
        cfg: &infer_core::RecallConfig,
    ) -> anyhow::Result<()> {
        self.update_block_reps(cfg)?;
        let Some(query) = model.take_recall_query()? else {
            // No query stashed yet (first step on the session) → leave ranges.
            return Ok(());
        };
        mlx::eval(&[&query]);
        let q = query.as_slice_f32(); // [nkv, hd] row-major
        let nb = self.block_reps.len();
        let mut scores = vec![0.0_f32; nb];
        for (i, rep) in self.block_reps.iter().enumerate() {
            // q · rep over the full [nkv, hd] vector (GQA-grouped query mean).
            let n = rep.len().min(q.len());
            let mut acc = 0.0_f32;
            for k in 0..n {
                acc += q[k] * rep[k];
            }
            scores[i] = acc;
        }
        let plan = infer_core::plan_recall(self.cache_len, &scores, cfg);
        // A single contiguous full range == today's default read; keep `None` so
        // the decode hot path stays byte-identical when the session fits.
        let is_full = plan.ranges.len() == 1 && plan.ranges[0] == (0, self.cache_len);
        self.recall_ranges = (!is_full).then_some(plan.ranges);
        // TODO(kv-recall L3): this is the RESIDENT variant — full KV stays in
        // `slot.kv_flat` (HBM) and recall restricts *attention* to the selected
        // ranges (the page-gather), saving decode compute. The plan-doc L3 tier
        // offload (demote the non-selected middle blocks' full KV to
        // kv-native-sys / `radix::demote_block`, keeping only the resident rep;
        // promote the selected blocks back) is not wired into the Metal slot KV
        // yet — that frees the HBM working set for unbounded history. Reps +
        // scoring + recall_ranges are fully live, so recall itself works now.
        Ok(())
    }

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
        model: &qwen35::CppQwen35Model,
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
