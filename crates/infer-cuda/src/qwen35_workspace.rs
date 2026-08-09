use super::*;

#[derive(Default)]
pub(crate) struct Qwen35Workspace {
    pub(crate) token_ids: SliceSlot<i32>,
    /// For the full-attn prep kernel — uploaded once per forward (identical
    /// for every full-attn layer; the old path uploaded one identical buffer
    /// per layer). The decode graph also reads it from the devpos attention
    /// kernel, so it is the single per-step position scalar staged pre-replay.
    pub(crate) start_pos: SliceSlot<i32>,
    pub(crate) hidden: HiddenSlot,
    pub(crate) normed: HiddenSlot,
    pub(crate) hidden_mid: HiddenSlot,
    pub(crate) attn_out: HiddenSlot,
    pub(crate) mlp_out: HiddenSlot,
    pub(crate) full: FullAttnScratch,
    pub(crate) linear: LinearAttnScratch,
    pub(crate) dense: DenseMlpScratch,
    pub(crate) moe: MoeForwardScratch,
    pub(crate) last_hidden: VecSlot,
    pub(crate) last_normed: VecSlot,
    pub(crate) logits: VecSlot,
    /// Persistent buffer for the greedy sampling tail — removes the last
    /// steady-state per-token device allocation (`ops::argmax`'s
    /// `alloc_zeros(1)`).
    pub(crate) argmax_out: SliceSlot<i32>,
    /// Buffer-address generation. Bumped whenever cached buffers are dropped
    /// wholesale ([`Self::release`]) — i.e. whenever previously-cached device
    /// ADDRESSES may change on the next `get`. The captured decode graph bakes
    /// buffer addresses, so it records this at capture and recaptures on
    /// mismatch instead of replaying against freed memory.
    pub(crate) epoch: u64,
}

#[derive(Default)]
pub(crate) struct FullAttnScratch {
    pub(crate) qkv_fused: HiddenSlot,
    pub(crate) q_full: HiddenSlot,
    pub(crate) k_batch: HiddenSlot,
    pub(crate) v_batch: HiddenSlot,
    pub(crate) q_prepped: HiddenSlot,
    pub(crate) attn_heads: HiddenSlot,
    pub(crate) fa3_lse: SliceSlot<f32>,
    pub(crate) fa3_oaccum: SliceSlot<f32>,
    pub(crate) fa3_lseaccum: SliceSlot<f32>,
    pub(crate) fa3_semaphore: SliceSlot<i32>,
    pub(crate) batch_partial_out: SliceSlot<f32>,
    pub(crate) batch_partial_m: SliceSlot<f32>,
    pub(crate) batch_partial_l: SliceSlot<f32>,
}

/// Paged full-attn forwarding context for Qwen3.6 — the DEFAULT path since the
/// shared-paged migration. Each full-attn layer reads/writes the shared
/// `PagedKVPool` (`full_attn_kv`) over `meta` (the page table) instead of a
/// per-slot contiguous cache. The default build hands a `for_slot` page table
/// over the slot's FULL resident pages (full attention, no eviction); the
/// `--kv-recall` cycle layers a working-set restriction on top of the SAME
/// pool. `Some` on `layer0_query` opts into the layer-0 post-RoPE query
/// readback for the recall score — a mid-forward D2H, so only the recall
/// prefill asks for it.
pub(crate) struct Qwen35RecallForward<'a> {
    pub(crate) pool: &'a mut PagedKVPool,
    pub(crate) meta: &'a crate::loader::PageMeta,
    pub(crate) layer0_query: Option<Vec<f32>>,
}

#[derive(Default)]
pub(crate) struct LinearAttnScratch {
    pub(crate) capture_copy: Qwen35CopyScratch,
    pub(crate) qkvz: HiddenSlot,
    pub(crate) qkv: HiddenSlot,
    pub(crate) z: HiddenSlot,
    pub(crate) ba: HiddenSlot,
    pub(crate) b_proj: HiddenSlot,
    pub(crate) a_proj: HiddenSlot,
    pub(crate) qkv_conv: HiddenSlot,
    pub(crate) gdr_out: HiddenSlot,
    pub(crate) normed_out: HiddenSlot,
    pub(crate) fq_q: HiddenSlot,
    pub(crate) fq_k: HiddenSlot,
    pub(crate) fq_v: HiddenSlot,
    pub(crate) fq_a: HiddenSlot,
    pub(crate) fq_g: SliceSlot<f32>,
    pub(crate) fq_g_cumsum: SliceSlot<f32>,
    pub(crate) fq_beta: SliceSlot<f32>,
    pub(crate) batch_ptrs: SliceSlot<u64>,
    pub(crate) batch_len: SliceSlot<i32>,
    pub(crate) batch_host: Vec<u64>,
    pub(crate) batch_len_host: Vec<i32>,
}

/// Rows this long or shorter take the batched recurrent core instead of
/// per-row FlashQLA: one chunk holds them, so there is no chunk parallelism
/// to win back against B times the launches.
pub(crate) const LINEAR_BATCH_MAX_LEN: usize = 64;

/// One slot's contiguous column range in a ragged batch. Its `len` token-major
/// columns advance THIS slot's state; its capture receives them from offset 0.
pub(crate) struct LinearRow<'a> {
    pub(crate) slot: &'a mut Qwen35SlotState,
    pub(crate) len: usize,
    pub(crate) capture: Option<&'a mut Qwen35LinearCapture>,
}

/// How [`Qwen35Model::linear_attention`] reaches per-slot conv + recurrent
/// state. Everything else in the layer runs once over all columns.
pub(crate) enum LinearCore<'a, 'r> {
    /// Ragged `B×T`: one single-slot multi-token launch per row.
    Rows(&'a mut [LinearRow<'r>]),
    /// Pure decode, one token per row: staged pointer tables advance all B
    /// states in ONE conv + ONE GDR launch. Row `r`'s channels sit at `r*C`;
    /// conv `[C, K-1]` and GDR `[Vh, Kd, Vd]` match the single-slot layout.
    Tables {
        conv: &'a CudaSlice<u64>,
        gdr: &'a CudaSlice<u64>,
    },
}

#[derive(Default)]
pub(crate) struct DenseMlpScratch {
    pub(crate) gate_up: HiddenSlot,
    pub(crate) act: HiddenSlot,
}

impl Qwen35Workspace {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Called by the executor after the OPD weight offload so the workspace
    /// does not hold prefill-shaped scratch while the student backward needs
    /// the headroom. The caller must have quiesced the device first
    /// (`offload_engine_weights` syncs). Bumps the address epoch: any
    /// captured decode graph over these buffers is stale after this.
    pub(crate) fn release(&mut self) {
        self.epoch += 1;
        let Self {
            token_ids,
            start_pos,
            hidden,
            normed,
            hidden_mid,
            attn_out,
            mlp_out,
            full,
            linear,
            dense,
            moe,
            last_hidden,
            last_normed,
            logits,
            argmax_out,
            epoch: _,
        } = self;
        token_ids.release();
        start_pos.release();
        hidden.release();
        normed.release();
        hidden_mid.release();
        attn_out.release();
        mlp_out.release();
        full.q_full.release();
        full.k_batch.release();
        full.v_batch.release();
        full.q_prepped.release();
        full.attn_heads.release();
        full.fa3_lse.release();
        full.fa3_oaccum.release();
        full.fa3_lseaccum.release();
        full.fa3_semaphore.release();
        full.batch_partial_out.release();
        full.batch_partial_m.release();
        full.batch_partial_l.release();
        linear.qkv.release();
        linear.z.release();
        linear.b_proj.release();
        linear.a_proj.release();
        linear.qkv_conv.release();
        linear.gdr_out.release();
        linear.normed_out.release();
        dense.gate_up.release();
        dense.act.release();
        moe.release();
        last_hidden.release();
        last_normed.release();
        logits.release();
        argmax_out.release();
    }
}

/// Persistent device state for the rows>1 BATCHED DECODE path (stage 1:
/// contiguous per-slot KV kept, no paged migration). Re-port of the deleted
/// monolith's proven design (`e81b98fb~1` `infer/src/model/qwen35/batch_decode.rs`,
/// `BatchDecodeBuffers35` + per-layer pointer tables, lines 87-89/780-815)
/// onto the rewrite's workspace slots, using DSv4's batched-decode executor
/// shape (`dsv4.rs` Step-A per-row attention) as the template.
///
/// Owns a DEDICATED forward workspace: it only ever sees `[*, B]` decode
/// shapes, so the main (prefill-reshaping) workspace never thrashes, and —
/// critically — every buffer that feeds `TpRuntime::all_reduce_sum` is an
/// EXACT-shape `[dim, B]` allocation. `all_reduce_sum` derives the collective
/// message length (and the one-shot-vs-NCCL choice) from `data.len()`
/// (see `workspace.rs`), so exact-shape buffers make the reduced message
/// exactly B valid columns BY CONSTRUCTION — no capacity tail of stale
/// columns can ever enter a reduction. (Deviation from the monolith's
/// capacity-sized `set_batch_size` buffers, deliberate: the monolith had a
/// length-honest collective API; the rewrite's does not.)
///
/// Pointer tables are capacity-sized (`[num_slots]` u64 per linear layer per
/// kind) and uploaded with B valid entries; the batch kernels read entries
/// `[0, gridDim.y)` = `[0, B)` only, so the dead tail is never dereferenced.
/// Tables are a pure function of `(slot_indices, layer)`: the per-slot conv
/// ring / GDR state `CudaSlice`s are allocated once at executor construction
/// and never re-allocated (`Qwen35SlotState::reset` memsets in place; the OPD
/// weight offload leaves slot state untouched), so restaging is needed only
/// when the row→slot mapping changes (monolith `TileLangDecodeMetadata.update`
/// pattern).
pub(crate) struct Qwen35BatchDecodeState {
    pub(crate) ws: Qwen35Workspace,
    pub(crate) positions: SliceSlot<i32>,
    pub(crate) seq_lens: SliceSlot<i32>,
    pub(crate) full_k_cache_ptrs: Vec<CudaSlice<u64>>,
    pub(crate) full_v_cache_ptrs: Vec<CudaSlice<u64>>,
    pub(crate) conv_state_ptrs: Vec<CudaSlice<u64>>,
    pub(crate) gdr_state_ptrs: Vec<CudaSlice<u64>>,
    /// Host staging vecs for the table uploads (monolith pattern: one
    /// `memcpy_htod` per layer per table, no per-row H2D).
    pub(crate) conv_host: Vec<u64>,
    pub(crate) gdr_host: Vec<u64>,
    pub(crate) full_k_host: Vec<u64>,
    pub(crate) full_v_host: Vec<u64>,
    pub(crate) staged_slot_indices: Vec<usize>,
    pub(crate) logits_batch: HiddenSlot,
    pub(crate) argmax: SliceSlot<i32>,
}

impl Qwen35BatchDecodeState {
    pub(crate) fn new(
        ctx: &DeviceContext,
        num_full_layers: usize,
        num_linear_layers: usize,
        max_batch: usize,
    ) -> Result<Self> {
        ensure!(
            max_batch > 0,
            "Qwen3.5 batched decode requires max_batch > 0"
        );
        let (full_k_cache_ptrs, full_v_cache_ptrs) =
            (0..num_full_layers)
                .map(|i| {
                    let k = ctx.stream.alloc_zeros::<u64>(max_batch).map_err(|e| {
                        anyhow!("alloc qwen35 batch full_k_cache_ptrs layer {i}: {e}")
                    })?;
                    let v = ctx.stream.alloc_zeros::<u64>(max_batch).map_err(|e| {
                        anyhow!("alloc qwen35 batch full_v_cache_ptrs layer {i}: {e}")
                    })?;
                    Ok((k, v))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip::<_, _, Vec<_>, Vec<_>>();
        let (conv_state_ptrs, gdr_state_ptrs) = (0..num_linear_layers)
            .map(|i| {
                let c = ctx
                    .stream
                    .alloc_zeros::<u64>(max_batch)
                    .map_err(|e| anyhow!("alloc qwen35 batch conv_state_ptrs layer {i}: {e}"))?;
                let g = ctx
                    .stream
                    .alloc_zeros::<u64>(max_batch)
                    .map_err(|e| anyhow!("alloc qwen35 batch gdr_state_ptrs layer {i}: {e}"))?;
                Ok((c, g))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Ok(Self {
            ws: Qwen35Workspace::new(),
            positions: SliceSlot::default(),
            seq_lens: SliceSlot::default(),
            full_k_cache_ptrs,
            full_v_cache_ptrs,
            conv_state_ptrs,
            gdr_state_ptrs,
            conv_host: vec![0u64; max_batch],
            gdr_host: vec![0u64; max_batch],
            full_k_host: vec![0u64; max_batch],
            full_v_host: vec![0u64; max_batch],
            staged_slot_indices: Vec::new(),
            logits_batch: HiddenSlot::default(),
            argmax: SliceSlot::default(),
        })
    }

    /// Re-upload the per-layer state pointer tables iff the row→slot mapping
    /// changed (tables are a pure function of `(slot_indices, layer)`; see
    /// struct docs for why the slot-state addresses themselves are stable).
    pub(crate) fn stage_pointer_tables(
        &mut self,
        ctx: &DeviceContext,
        slots: &mut [Qwen35SlotState],
        slot_indices: &[usize],
    ) -> Result<()> {
        if self.staged_slot_indices == slot_indices {
            return Ok(());
        }
        let b = slot_indices.len();
        ensure!(
            b <= self.conv_host.len(),
            "Qwen3.5 batched decode batch {} exceeds table capacity {}",
            b,
            self.conv_host.len()
        );
        for layer_idx in 0..self.full_k_cache_ptrs.len() {
            for (r, &si) in slot_indices.iter().enumerate() {
                let slot = &mut slots[si];
                ensure!(
                    layer_idx < slot.k_caches.len() && layer_idx < slot.v_caches.len(),
                    "Qwen3.5 batched decode full-attn layer {layer_idx} outside slot cache \
                     (k={}, v={})",
                    slot.k_caches.len(),
                    slot.v_caches.len()
                );
                let (k_ptr, _gk) = slot.k_caches[layer_idx].data.device_ptr_mut(&ctx.stream);
                let (v_ptr, _gv) = slot.v_caches[layer_idx].data.device_ptr_mut(&ctx.stream);
                self.full_k_host[r] = k_ptr;
                self.full_v_host[r] = v_ptr;
            }
            ctx.stream
                .memcpy_htod(
                    &self.full_k_host[..b],
                    &mut self.full_k_cache_ptrs[layer_idx],
                )
                .map_err(|e| anyhow!("H2D qwen35 full_k_cache_ptrs layer {layer_idx}: {e}"))?;
            ctx.stream
                .memcpy_htod(
                    &self.full_v_host[..b],
                    &mut self.full_v_cache_ptrs[layer_idx],
                )
                .map_err(|e| anyhow!("H2D qwen35 full_v_cache_ptrs layer {layer_idx}: {e}"))?;
        }
        for layer_idx in 0..self.conv_state_ptrs.len() {
            for (r, &si) in slot_indices.iter().enumerate() {
                let slot = &mut slots[si];
                ensure!(
                    layer_idx < slot.conv_states.len() && layer_idx < slot.gdr_states.len(),
                    "Qwen3.5 batched decode linear layer {layer_idx} outside slot state \
                     (conv={}, gdr={})",
                    slot.conv_states.len(),
                    slot.gdr_states.len()
                );
                let (conv_ptr, _gc) = slot.conv_states[layer_idx].data.device_ptr_mut(&ctx.stream);
                let (gdr_ptr, _gg) = slot.gdr_states[layer_idx].device_ptr_mut(&ctx.stream);
                self.conv_host[r] = conv_ptr;
                self.gdr_host[r] = gdr_ptr;
            }
            ctx.stream
                .memcpy_htod(&self.conv_host[..b], &mut self.conv_state_ptrs[layer_idx])
                .map_err(|e| anyhow!("H2D qwen35 conv_state_ptrs layer {layer_idx}: {e}"))?;
            ctx.stream
                .memcpy_htod(&self.gdr_host[..b], &mut self.gdr_state_ptrs[layer_idx])
                .map_err(|e| anyhow!("H2D qwen35 gdr_state_ptrs layer {layer_idx}: {e}"))?;
        }
        self.staged_slot_indices = slot_indices.to_vec();
        Ok(())
    }

    /// Recurrent-only pointer staging for the PAGED batched-decode lane: stage
    /// the conv-ring + GDR-state tables (linear-attn layers) but SKIP the
    /// contiguous full-attn `k_caches`/`v_caches` tables, which the shared-paged
    /// default never allocates (touching them would deref an empty slice). Paged
    /// full attention reads the shared pool via the per-step `PageMeta` instead,
    /// so the conv/GDR tables are the only per-slot device pointers it needs.
    /// Same `staged_slot_indices` cache key as the contiguous path; both lanes
    /// share the invalidation hook. The two stagers never interleave on one
    /// executor (a build is either paged or contiguous), so the cache key is
    /// unambiguous.
    pub(crate) fn stage_recurrent_pointer_tables(
        &mut self,
        ctx: &DeviceContext,
        slots: &mut [Qwen35SlotState],
        slot_indices: &[usize],
    ) -> Result<()> {
        if self.staged_slot_indices == slot_indices {
            return Ok(());
        }
        let b = slot_indices.len();
        ensure!(
            b <= self.conv_host.len(),
            "Qwen3.5 paged batched decode batch {} exceeds table capacity {}",
            b,
            self.conv_host.len()
        );
        for layer_idx in 0..self.conv_state_ptrs.len() {
            for (r, &si) in slot_indices.iter().enumerate() {
                let slot = &mut slots[si];
                ensure!(
                    layer_idx < slot.conv_states.len() && layer_idx < slot.gdr_states.len(),
                    "Qwen3.5 paged batched decode linear layer {layer_idx} outside slot state \
                     (conv={}, gdr={})",
                    slot.conv_states.len(),
                    slot.gdr_states.len()
                );
                let (conv_ptr, _gc) = slot.conv_states[layer_idx].data.device_ptr_mut(&ctx.stream);
                let (gdr_ptr, _gg) = slot.gdr_states[layer_idx].device_ptr_mut(&ctx.stream);
                self.conv_host[r] = conv_ptr;
                self.gdr_host[r] = gdr_ptr;
            }
            ctx.stream
                .memcpy_htod(&self.conv_host[..b], &mut self.conv_state_ptrs[layer_idx])
                .map_err(|e| anyhow!("H2D qwen35 paged conv_state_ptrs layer {layer_idx}: {e}"))?;
            ctx.stream
                .memcpy_htod(&self.gdr_host[..b], &mut self.gdr_state_ptrs[layer_idx])
                .map_err(|e| anyhow!("H2D qwen35 paged gdr_state_ptrs layer {layer_idx}: {e}"))?;
        }
        self.staged_slot_indices = slot_indices.to_vec();
        Ok(())
    }

    /// Invalidate the staged pointer-table cache so the next decode batch
    /// restages from the slots' CURRENT recurrent-block addresses. Required at a
    /// request boundary: with the free-list pool a slot's `gdr_states`/
    /// `conv_states` `CudaSlice`s change identity when a new request acquires a
    /// different block (vs the old upfront alloc, where they were fixed for the
    /// executor's lifetime). The cache keys on `slot_indices` alone, so without
    /// this a same-mapping batch would dereference the prior occupant's block.
    pub(crate) fn invalidate_staged_pointers(&mut self) {
        self.staged_slot_indices.clear();
    }

    /// OPD weight time-share hook. The pointer tables and the staged mapping
    /// stay: the per-slot state addresses they hold are executor-owned and
    /// untouched by the weight offload, so they remain valid across an
    /// offload/reload cycle (and they are ~KB-scale).
    pub(crate) fn release(&mut self) {
        self.ws.release();
        self.positions.release();
        self.seq_lens.release();
        self.logits_batch.release();
        self.argmax.release();
    }
}
