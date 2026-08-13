//! The N-slot batched forward drivers: layer-major decode and layer-major chain
//! verify over N slots.

use super::layer_block::HcHalf;
use super::*;

impl Dsv4Model {
    /// Layer-major batched decode over N independent slots (one decode token
    /// each). Row `r` decodes slot `slot_ids[r]` at `start_positions[r]`. The
    /// point-wise pipeline runs over the whole `seq_len = N` batch — those ops are
    /// token-independent, so stacking N rows is math-identical to N single-row
    /// forwards. B>1 uses the batched FlashMLA sparse decode lane; B=1 stays on
    /// the single-row path.
    pub(crate) fn forward_decode_batch(
        &self,
        slots: &mut [Dsv4SlotState],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_ids: &[usize],
        tokens: &[u32],
        start_positions: &[usize],
        positions: &[u64],
        params: &[SamplingParams],
        penalties: &[infer_plan::PenaltyHistory<'_>],
        // false on the spec-off decode lane: nothing reads these taps.
        capture_taps: bool,
    ) -> Result<Vec<u32>> {
        let n = slot_ids.len();
        ensure!(n > 0, "DSv4 batched decode requires at least one row");
        ensure!(
            tokens.len() == n
                && start_positions.len() == n
                && positions.len() == n
                && params.len() == n
                && penalties.len() == n,
            "DSv4 batched decode surface length mismatch (slots {n}, tokens {}, starts {}, positions {}, params {}, penalty histories {})",
            tokens.len(),
            start_positions.len(),
            positions.len(),
            params.len(),
            penalties.len()
        );
        let (stream, mut keepalive) = self.forward_decode_batch_stream_impl(
            slots,
            kv_adapter,
            slot_ids,
            tokens,
            start_positions,
            capture_taps,
        )?;
        let fast_head = params.iter().all(SamplingParams::is_raw_argmax);
        let out_tokens = crate::profile::profile_op(&self.ctx, "lm_head", None, n, || {
            if fast_head {
                let logits = self.verify_logits_from_stream(&stream, n, &mut keepalive)?;
                self.mtp_argmax_batch(&logits)
            } else {
                // `forward_stream_last_token` folds row `seq_len - 1`.
                (0..n)
                    .map(|r| {
                        self.forward_stream_last_token(
                            &stream,
                            r + 1,
                            &params[r],
                            positions[r],
                            penalties[r],
                            None,
                            &mut keepalive,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            }
        })?;
        std::hint::black_box(keepalive.len());
        drop(keepalive);
        Ok(out_tokens)
    }

    pub(super) fn forward_decode_batch_stream_impl(
        &self,
        slots: &mut [Dsv4SlotState],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_ids: &[usize],
        tokens: &[u32],
        start_positions: &[usize],
        capture_taps: bool,
    ) -> Result<(HiddenStates, Dsv4ForwardKeepalive)> {
        let n = slot_ids.len();
        let mega_epoch = self.begin_mega_moe_forward(n)?;
        // Scopes the per-GEMV stats to ONE decode forward (self-gates on
        // ARLE_DSV4_LINEAR_PROFILE). When enabled it syncs per call, inflating
        // absolute step ms — read it for the RELATIVE per-GEMV split.
        crate::linear_profile::reset();
        for r in 0..n {
            let slot = &slots[slot_ids[r]];
            ensure!(
                slot.seq_len == start_positions[r],
                "DSv4 batched decode slot {} seq_len {} != start_pos {}; decode requires contiguous appends",
                slot_ids[r],
                slot.seq_len,
                start_positions[r]
            );
            let next_len = start_positions[r]
                .checked_add(1)
                .ok_or_else(|| anyhow!("DSv4 batched decode start_pos overflow"))?;
            ensure!(
                start_positions[r] < slot.max_seq_len,
                "DSv4 batched decode slot {} sequence {} exceeds max_seq_len {}",
                slot_ids[r],
                next_len,
                slot.max_seq_len
            );
        }

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let seq_len = n; // batch dimension: N independent decode rows
        // capture only when the caller consumes taps; plain decode never reads them.
        let dspark = capture_taps && self.config.is_dspark();
        let use_deepep_transport = crate::runtime_flags::dsv4_moe_transport()?.is_deepep();
        // N>1: the per-token decode scratch / comm-overlap fast paths are seq_len==1
        // only.
        let mut keepalive = Dsv4ForwardKeepalive::new(false);
        let ctx = &self.ctx;

        // Batched (b=N) FlashMLA decode lane; canonical for B>1 when the
        // model-wide batched scratch exists. N=1 never reaches this function.
        let batched_attn_lane = crate::attention::dsv4_flashmla_decode_enabled()?
            && kv_adapter.has_flashmla_batch_scratch();

        // Per-slot decode position scalars (each row's attention reads its own).
        for r in 0..n {
            let start_pos_i32 = i32::try_from(start_positions[r])
                .map_err(|_| anyhow!("DSv4 start_pos {} overflows i32", start_positions[r]))?;
            let slot = &mut slots[slot_ids[r]];
            ctx.stream
                .memcpy_htod(&[start_pos_i32], &mut slot.start_pos_device)
                .map_err(|e| anyhow!("DSv4 batched start_pos H2D failed: {e}"))?;
        }

        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = crate::ops::upload_i32(&self.ctx, &token_ids_host)?;
        keepalive.keep_i32(&token_ids);
        // SAFETY: embed_stream writes the full stream buffer.
        let mut stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
        self.embed_stream(&token_ids, seq_len, &mut stream, &mut keepalive)?;

        // Reusable [hidden,1] scratch for the per-row attention copy-in/out.
        // Reuse across rows/layers is safe because all ops run on `ctx.stream`
        // (WAR/RAW resolved by stream ordering).
        // SAFETY: fully written by the copy-in / mla_attention each row before read.
        let mut normed_row = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, 1)? };
        // SAFETY: uninit device scratch; fully written before first read.
        let mut attn_out_row = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, 1)? };
        keepalive.keep_hidden(&normed_row);
        keepalive.keep_hidden(&attn_out_row);

        // Contiguous [N] absolute-position array for the batched projection
        // pre-pass RoPE. Uploaded once, reused every layer (positions are fixed
        // across the layer loop). Only allocated on the batched lane.
        let batched_positions = if batched_attn_lane {
            let positions_host: Vec<i32> = (0..n)
                .map(|r| {
                    i32::try_from(start_positions[r])
                        .map_err(|_| anyhow!("DSv4 start_pos {} overflows i32", start_positions[r]))
                })
                .collect::<Result<Vec<_>>>()?;
            let buf = crate::ops::upload_i32(&self.ctx, &positions_host)?;
            keepalive.keep_i32(&buf);
            Some(buf)
        } else {
            None
        };
        // Per-row [*, 1] scratch for the batched projection slice copy-in.
        // Reused across rows/layers (WAR resolved by stream order).
        let local_width = self.layers[0].attention.wq_b.rows;
        let q_lora_rank = self.config.q_lora_rank;
        let mla_head_dim = self.config.head_dim;
        // Reused per-row [q_lora_rank, 1] scratch for this row's batched
        // c_q_normed slice. q_prepared / k_prepared rows go into fresh OWNED
        // buffers instead (taken by value into Dsv4MlaPrepared).
        let mut c_q_normed_row = if batched_attn_lane {
            // SAFETY: fully written by the per-row slice copy-in before read.
            let cq = unsafe { HiddenStates::uninit(&self.ctx, q_lora_rank, 1)? };
            keepalive.keep_hidden(&cq);
            cq
        } else {
            // Unused (per-row lane); zero-sized stand-in keeps the binding typed.
            // SAFETY: zero-size placeholder; never read.
            unsafe { HiddenStates::uninit(&self.ctx, 0, 1)? }
        };

        // FINISH F1 batched destination: ONE contiguous token-major
        // `[n, local_width]` buffer for `slice_out_batched`, reused across layers.
        // local_width is uniform across DSv4 layers. Held in keepalive
        // (premature-free hazard under disabled events).
        let mut local_attn_batched = if batched_attn_lane {
            // SAFETY: every [r*local_width, (r+1)*local_width) span is written by
            // slice_out_batched before the finish kernels / O-LoRA read it.
            let lab = unsafe { HiddenStates::uninit(&self.ctx, local_width, n)? };
            keepalive.keep_hidden(&lab);
            lab
        } else {
            // SAFETY: zero-size placeholder; never read.
            unsafe { HiddenStates::uninit(&self.ctx, 0, 1)? }
        };
        // Full-flatten batches compressor state update, inverse-RoPE and SW-window
        // write into one launch over N rows; SparseIndexed has no main compressor
        // and is excluded below. `ptr_keepalive` holds the host-uploaded per-row
        // device-pointer ARRAYS past their kernel launch (the N>1 keepalive is
        // inert; premature-free hazard under disabled event tracking).
        let mut ptr_keepalive: Vec<CudaSlice<u64>> = Vec::new();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let full_flatten = layer.mode != DeepSeekV4AttentionMode::SparseIndexed;

            // SAFETY: uninit device scratch; fully written before first read.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            let attn_mhc = self.hc_pre_norm(
                layer,
                HcHalf::Attn,
                layer_idx,
                seq_len,
                &stream,
                &mut normed,
                &mut keepalive,
            )?;
            keepalive.keep_hidden(&normed);

            // SAFETY: every [r*hidden, (r+1)*hidden) span of attn_out is written by
            // the copy-out below before attn_out is read by hc_post.
            let mut attn_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            // Batched FlashMLA decode lane: per row PREPARE (wq/wkv+RoPE,
            // compressor, CSA indexer top-k) → pack KV → gather Q; then ONE
            // `sparse_decode_fwd(b=N)`; then the per-row finish tail. Only the
            // attention kernel is batched (the 74k tiny gridX=1 launches).
            // SW/HCA pass selected_ptr=0; CSA passes the gathered per-row top-k.
            let use_batched_kernel = batched_attn_lane;
            if use_batched_kernel {
                // GLM SparseIndexed has NO main compressor — it attends the full
                // latent via the FlashMLA KV pool, so it must NOT request the
                // compressed pack (`flashmla_pack_borrow(true)` would error).
                let want_compressed = layer.mode.has_compressor();
                // CSA + GLM SparseIndexed both run the indexer top-k select; the
                // CSA-only compressor / full-flatten paths below stay gated on
                // CompressedSparse.
                // ponytail: pod-verify SparseIndexed batched-decode DSA select lane
                // end-to-end
                let runs_indexer = layer.mode.has_indexer();
                // Defer the per-row READ (paged-MQA logits + topk) into ONE
                // batch_size=N call after the prepare loop; the per-row CACHE
                // WRITES still run inside it.
                let use_batched_dsa_select = runs_indexer
                    && kv_adapter
                        .layer_dsa_and_flashmla_batch_mut(layer_idx)?
                        .1
                        .is_some();
                // N-row staging for the batched CSA select. CompressedSparse uses
                // the batched indexer-query prepass output; SparseIndexed stages
                // per-row q_i/weights here.
                let mut dsa_stage_q_i: Option<HiddenStates> = None;
                let mut dsa_stage_weights: Option<HiddenStates> = None;
                let mut dsa_key_counts =
                    use_batched_dsa_select.then(|| Vec::<i32>::with_capacity(n));
                let mut prepared: Vec<crate::attention::Dsv4MlaPrepared> = Vec::with_capacity(n);
                let mut slot_block_offsets = Vec::with_capacity(n);
                let mut page_tables = Vec::with_capacity(n);
                // Batched (m=N) slot-INDEPENDENT projection pre-pass: wq_a / wkv /
                // wq_b + RoPE over all N rows at once (weights read ONCE across
                // the N-token grid). The per-row loop below reads each row's SLICE.
                let positions = batched_positions.as_ref().ok_or_else(|| {
                    anyhow!("DSv4 batched decode lane: batched positions buffer missing")
                })?;
                // Borrow the model-wide shared FP8 prefill DeepGEMM scratch for the
                // batched (m=N) projection; `None` ⇒ scalar fallback. Scoped to the
                // pre-pass and released before the per-row loop re-borrows.
                let proj = {
                    let (_layer_pool, _dsa, _flash_batch, _flashmla_scratch, prefill_shared) =
                        kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                    crate::attention::mla_attention_prepare_proj_batch(
                        &self.ctx,
                        &self.config,
                        &layer.attention,
                        layer.compress_ratio,
                        &normed,
                        positions,
                        prefill_shared,
                        &mut keepalive,
                    )?
                };
                // Batched (m=N) compressor/indexer key + query projections: the
                // per-row m=1 GEMVs re-read the full weight per decode row.
                // Touches NO slot state.
                let (compressor_kv_score, indexer_kv_score, indexer_query_kv_score) = {
                    let (_lp, _dsa, _fb, _fs, mut prefill_shared) =
                        kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                    // Aux-stream fork of this prepass was killed: pod A/B at n=8
                    // regressed ~9% wall-clock (9.65s -> 10.53s, zero overlap).
                    let main = match layer.attention.compressor.as_ref() {
                        Some(c) => crate::attention::compressor_batch_prepass(
                            &self.ctx,
                            c,
                            &normed,
                            prefill_shared.as_deref_mut(),
                            &mut keepalive,
                        )?,
                        None => None,
                    };
                    let indexer = match (layer.mode, layer.attention.indexer.as_ref()) {
                        (DeepSeekV4AttentionMode::CompressedSparse, Some(idx)) => {
                            crate::attention::compressor_batch_prepass(
                                &self.ctx,
                                idx.compressor
                                    .as_ref()
                                    .expect("DSv4 CSA indexer has a key compressor"),
                                &normed,
                                prefill_shared.as_deref_mut(),
                                &mut keepalive,
                            )?
                        }
                        _ => None,
                    };
                    let query = match (layer.mode, layer.attention.indexer.as_ref()) {
                        (DeepSeekV4AttentionMode::CompressedSparse, Some(idx)) => {
                            Some(crate::attention::indexer_query_batch_prepass(
                                &self.ctx,
                                idx,
                                &proj.c_q_normed,
                                &normed,
                                prefill_shared,
                                &mut keepalive,
                            )?)
                        }
                        _ => None,
                    };
                    (main, indexer, query)
                };
                // Full-flatten P1a: defer each row's compressor STATE update
                // (gather ring-state pointers + advance compressed.seq_len, no
                // per-row FFI) into ONE `dsv4_compressor_update_batched`. Runs
                // BEFORE P1b's consumers so they see the written keys.
                // `indexer_rows_before` is captured per row here since P1a advances
                // seq_len. SparseIndexed has no compressor and skips this.
                let indexer_rows_before: Vec<usize> = if full_flatten {
                    let mut main_sink = crate::attention::Dsv4CompressorBatchPtrs::with_capacity(n);
                    let mut indexer_sink =
                        crate::attention::Dsv4CompressorBatchPtrs::with_capacity(n);
                    let mut before = Vec::with_capacity(n);
                    let original_seq_len = proj.rope.original_seq_len;
                    for r in 0..n {
                        // Defer mode reads NO `normed_row` data — it only gathers
                        // state pointers and advances seq_len — but the handle's
                        // `[hidden,1]` shape satisfies compressor_forward's asserts.
                        let slot = &mut slots[slot_ids[r]];
                        let b = crate::attention::mla_attention_compressor_defer_row(
                            &self.ctx,
                            &self.config,
                            &layer.attention,
                            layer.mode,
                            layer.compress_ratio,
                            &normed_row,
                            &mut slot.attention[layer_idx],
                            start_positions[r],
                            Some(&slot.start_pos_device),
                            proj.rope,
                            &mut main_sink,
                            &mut indexer_sink,
                            &mut keepalive,
                        )?;
                        before.push(b);
                    }
                    // ONE batched update each for the main + indexer compressor;
                    // the sinks are all-or-nothing per layer since `layer.mode` is
                    // uniform. `ptrs.prev_overlap_*` are per-row per-slot registers.
                    let overlap = layer.compress_ratio < 16;
                    if let (Some((kv, score)), Some(compressor)) = (
                        compressor_kv_score.as_ref(),
                        layer.attention.compressor.as_ref(),
                    ) {
                        let positions = batched_positions.as_ref().ok_or_else(|| {
                            anyhow!("DSv4 full-flatten P1a: batched positions missing")
                        })?;
                        crate::attention::dsv4_compressor_update_batched(
                            &self.ctx,
                            &self.config,
                            compressor,
                            kv,
                            score,
                            &main_sink,
                            positions,
                            n,
                            self.config.head_dim,
                            layer.compress_ratio,
                            overlap,
                            proj.rope,
                            &mut ptr_keepalive,
                        )?;
                    }
                    if layer.mode == DeepSeekV4AttentionMode::CompressedSparse
                        && let (Some((kv, score)), Some(indexer)) =
                            (indexer_kv_score.as_ref(), layer.attention.indexer.as_ref())
                    {
                        let positions = batched_positions.as_ref().ok_or_else(|| {
                            anyhow!("DSv4 full-flatten P1a: batched positions missing")
                        })?;
                        crate::attention::dsv4_compressor_update_batched(
                            &self.ctx,
                            &self.config,
                            indexer
                                .compressor
                                .as_ref()
                                .expect("DSv4 CSA indexer has a key compressor"),
                            kv,
                            score,
                            &indexer_sink,
                            positions,
                            n,
                            self.config.index_head_dim,
                            layer.compress_ratio,
                            true, // indexer compressor always overlap
                            proj.rope,
                            &mut ptr_keepalive,
                        )?;
                    }
                    before
                } else {
                    Vec::new()
                };
                // Full-flatten P1b: hoist the per-row DSA CACHE WRITE (Hadamard
                // rotate of newly-packed index keys + FP8 fused-store + packed_rows
                // advance) into ONE pre-pass. Runs AFTER P1a wrote
                // `indexer.compressed` and BEFORE the prepare loop, whose
                // `csa_select` then skips the write. SparseIndexed keeps it per-row.
                let cache_writes_in_prepass = use_batched_dsa_select
                    && full_flatten
                    && layer.mode == DeepSeekV4AttentionMode::CompressedSparse;
                if cache_writes_in_prepass {
                    let mut cache_ptrs =
                        crate::attention::Dsv4DsaCacheWriteBatchPtrs::with_capacity(n);
                    for r in 0..n {
                        let (layer_pool, dsa_shared, _fb, _fs, _pf) =
                            kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                        let dsa_shared = dsa_shared.ok_or_else(|| {
                            anyhow!("DSv4 full-flatten P1b: shared DSA scratch missing")
                        })?;
                        let slot = &mut slots[slot_ids[r]];
                        let state = &mut slot.attention[layer_idx];
                        // P1a already advanced `compressed.seq_len` to the AFTER
                        // count; `before[r]` is the captured pre-advance value.
                        let indexer_rows_after =
                            state.indexer_compressed_seq_len().ok_or_else(|| {
                                anyhow!("DSv4 full-flatten P1b: indexer state missing")
                            })?;
                        crate::attention::dsv4_dsa_cache_write_gather_row(
                            &self.ctx,
                            &self.config,
                            state,
                            layer_pool,
                            dsa_shared,
                            indexer_rows_before[r],
                            indexer_rows_after,
                            // Full history → window base 0 (matches csa_select).
                            0,
                            &mut cache_ptrs,
                        )?;
                    }
                    crate::attention::dsv4_dsa_cache_write_batched(
                        &self.ctx,
                        n,
                        &cache_ptrs,
                        &mut ptr_keepalive,
                        &mut keepalive,
                    )?;
                }
                crate::profile::profile_op(
                    ctx,
                    "attention_prepare",
                    Some(layer_idx),
                    seq_len,
                    || {
                        // Op "c" batched KV pack (MODEL1 only): the per-row pack's
                        // SW one-token + compressed-delta launches are HOISTED into
                        // ONE batched launch each after this loop (the SW-ring
                        // BOOTSTRAP stays per-row). Persistent device page tables
                        // live in `flash.device_page_table`, not per-call
                        // temporaries. A restored slot re-enters decode with
                        // fp8_kv_comp_packed_rows=0 and needs the single-row bulk
                        // rebuild, so that whole layer runs per-row this tick.
                        let comp_bulk_gap = (0..n).any(|r| {
                            slots[slot_ids[r]].attention[layer_idx].flashmla_comp_bulk_gap()
                        });
                        let pack_batched =
                            full_flatten && self.config.head_dim != 576 && !comp_bulk_gap;
                        let mut pack_nope_ptrs: Vec<u64> = Vec::new();
                        let mut pack_rope_ptrs: Vec<u64> = Vec::new();
                        let mut pack_compressed_ptrs: Vec<u64> = Vec::new();
                        let mut pack_pt_ptrs: Vec<u64> = Vec::new();
                        let mut pack_sw_blocks: usize = 0;
                        let mut pack_num_logical_pages: usize = 0;
                        if pack_batched {
                            pack_nope_ptrs.reserve(n);
                            pack_rope_ptrs.reserve(n);
                            pack_compressed_ptrs.reserve(n);
                            pack_pt_ptrs.reserve(n);
                        }
                        for r in 0..n {
                            if !full_flatten {
                                let src = normed.data.slice(r * hidden_size..(r + 1) * hidden_size);
                                ctx.stream.memcpy_dtod(&src, &mut normed_row.data).map_err(
                                    |e| anyhow!("DSv4 batched attn copy-in failed: {e}"),
                                )?;
                                let cq_src = proj
                                    .c_q_normed
                                    .data
                                    .slice(r * q_lora_rank..(r + 1) * q_lora_rank);
                                ctx.stream
                                    .memcpy_dtod(&cq_src, &mut c_q_normed_row.data)
                                    .map_err(|e| anyhow!("DSv4 batched c_q copy-in failed: {e}"))?;
                            }
                            // This row's q_prepared / k_prepared slices → fresh
                            // OWNED [*, 1] buffers taken by value into the returned
                            // Dsv4MlaPrepared: both must outlive the reused row
                            // scratch.
                            // SAFETY: each dtod copy fills the full buffer before read.
                            let mut q_prepared_owned =
                                unsafe { HiddenStates::uninit(&self.ctx, local_width, 1)? };
                            {
                                let qp_src = proj
                                    .q_prepared
                                    .data
                                    .slice(r * local_width..(r + 1) * local_width);
                                ctx.stream
                                    .memcpy_dtod(&qp_src, &mut q_prepared_owned.data)
                                    .map_err(|e| {
                                        anyhow!("DSv4 batched q_prepared owned copy failed: {e}")
                                    })?;
                            }
                            let mut k_prepared_owned =
                            // SAFETY: uninit device scratch; fully written before first
                            // read.
                            unsafe { HiddenStates::uninit(&self.ctx, mla_head_dim, 1)? };
                            {
                                let kp_src = proj
                                    .k_prepared
                                    .data
                                    .slice(r * mla_head_dim..(r + 1) * mla_head_dim);
                                ctx.stream
                                    .memcpy_dtod(&kp_src, &mut k_prepared_owned.data)
                                    .map_err(|e| {
                                        anyhow!("DSv4 batched k_prepared owned copy failed: {e}")
                                    })?;
                            }
                            let (
                                layer_pool,
                                dsa_shared,
                                flash_batch,
                                flashmla_scratch,
                                _prefill_shared,
                            ) = kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                            let flash_batch = flash_batch.ok_or_else(|| {
                                anyhow!("DSv4 batched decode lane: batch scratch missing")
                            })?;
                            let flashmla_scratch = flashmla_scratch.ok_or_else(|| {
                                anyhow!(
                                    "DSv4 batched decode lane: single-row decode scratch missing"
                                )
                            })?;
                            let slot = &mut slots[slot_ids[r]];
                            slot_block_offsets
                                .push(layer_pool.flashmla_slot_first_block_or_zero(slot_ids[r])?);
                            page_tables
                                .push(layer_pool.flashmla_page_table_padded_i32(slot_ids[r])?);
                            // Batched CSA select: thread this row's gather sink.
                            // `csa_select` then runs cache writes only and returns
                            // `selected: None`. `None` ⇒ per-row select.
                            let need_dsa_stage =
                                use_batched_dsa_select && indexer_query_kv_score.is_none();
                            if need_dsa_stage && dsa_stage_q_i.is_none() {
                                let index_heads = self.config.index_n_heads;
                                let q_width = index_heads * self.config.index_head_dim;
                                // SAFETY: each row's slice is fully written by the
                                // per-row
                                // gather before the batched select reads it.
                                let q_b = unsafe { HiddenStates::uninit(&self.ctx, q_width, n)? };
                                // SAFETY: uninit device scratch; fully written before
                                // first read.
                                let w_b =
                                    unsafe { HiddenStates::uninit(&self.ctx, index_heads, n)? };
                                keepalive.keep_hidden(&q_b);
                                keepalive.keep_hidden(&w_b);
                                dsa_stage_q_i = Some(q_b);
                                dsa_stage_weights = Some(w_b);
                            }
                            let batched_gather = if use_batched_dsa_select {
                                Some(crate::attention::Dsv4DsaBatchedGather {
                                    q_i_batch: dsa_stage_q_i.as_mut(),
                                    weights_batch: dsa_stage_weights.as_mut(),
                                    row: r,
                                    key_counts: dsa_key_counts
                                        .as_mut()
                                        .expect("batched DSA key_counts present"),
                                    // CompressedSparse full-flatten: P1b already
                                    // wrote the cache. SparseIndexed ⇒ per-row write.
                                    cache_writes_in_prepass,
                                })
                            } else {
                                None
                            };
                            // Slice this row's `[width,1]` column out of the batched
                            // compressor/indexer pre-pass outputs into a fresh owned
                            // buffer, referenced (not consumed) so it outlives the
                            // prepare call.
                            let slice_row = |src: &HiddenStates| -> Result<HiddenStates> {
                                let width = src.hidden_dim;
                                // SAFETY: the dtod copy fills the full buffer before
                                // read.
                                let mut row_buf =
                                    unsafe { HiddenStates::uninit(&self.ctx, width, 1)? };
                                let col = src.data.slice(r * width..(r + 1) * width);
                                self.ctx
                                    .stream
                                    .memcpy_dtod(&col, &mut row_buf.data)
                                    .map_err(|e| {
                                        anyhow!("DSv4 batched compressor slice copy failed: {e}")
                                    })?;
                                Ok(row_buf)
                            };
                            // Full-flatten: the compressor STATE update already ran
                            // batched in P1a, so pass `skip_compressor=true` + the
                            // P1a-captured `indexer_rows_before` and no precomputed
                            // slices. The owned slice buffers are bound at iteration
                            // scope so they outlive the prepare call that borrows
                            // them through `compressor_precomputed`.
                            let (comp_main_row, comp_indexer_row) = if full_flatten {
                                (None, None)
                            } else {
                                let main_row = match compressor_kv_score.as_ref() {
                                    Some((kv, score)) => Some((slice_row(kv)?, slice_row(score)?)),
                                    None => None,
                                };
                                let indexer_row = match indexer_kv_score.as_ref() {
                                    Some((kv, score)) => Some((slice_row(kv)?, slice_row(score)?)),
                                    None => None,
                                };
                                (main_row, indexer_row)
                            };
                            let compressor_precomputed =
                                comp_main_row.as_ref().map(|(kv, score)| {
                                    crate::attention::Dsv4CompressorPrecomputed {
                                        main: (kv, score),
                                        indexer: comp_indexer_row
                                            .as_ref()
                                            .map(|(ikv, iscore)| (ikv as &_, iscore as &_)),
                                    }
                                });
                            // CSA indexer-query batched pre-pass: borrow this row's
                            // `[width,1]` column VIEW so `csa_select` skips the
                            // per-row m=1 GEMVs AND the D2D re-copy. `None` when the
                            // pre-pass didn't run (SparseIndexed / no indexer).
                            let indexer_query_precomputed =
                                indexer_query_kv_score.as_ref().map(|(q_i, weights)| {
                                    crate::attention::Dsv4IndexerQueryPrecomputed {
                                        q_i: q_i.col(r),
                                        weights: weights.col(r),
                                    }
                                });
                            let skip_compressor = full_flatten;
                            let idx_before_override = if full_flatten {
                                indexer_rows_before.get(r).copied()
                            } else {
                                None
                            };
                            let row_prepared =
                                crate::attention::mla_attention_prepare_compressed_only(
                                    &self.ctx,
                                    &self.config,
                                    &layer.attention,
                                    layer.mode,
                                    layer.compress_ratio,
                                    &normed_row,
                                    &c_q_normed_row,
                                    q_prepared_owned,
                                    k_prepared_owned,
                                    &proj,
                                    &mut slot.attention[layer_idx],
                                    layer_pool,
                                    dsa_shared,
                                    start_positions[r],
                                    Some(&slot.start_pos_device),
                                    batched_gather,
                                    compressor_precomputed,
                                    indexer_query_precomputed,
                                    skip_compressor,
                                    idx_before_override,
                                    &mut keepalive,
                                )?;
                            // Pack this row's KV into the shared pool, then gather
                            // its global-head Q into q_batched[r].
                            let (flash, sw_window, compressed) =
                                slot.attention[layer_idx].flashmla_pack_borrow(want_compressed)?;
                            if pack_batched {
                                // Run the once-per-slot SW-ring BOOTSTRAP here, then
                                // gather this row's pack pointers + page table for
                                // the ONE batched pack issued after the loop.
                                crate::attention::flashmla_pack_sw_ring(
                                    &self.ctx,
                                    flash,
                                    flashmla_scratch,
                                    layer_pool,
                                    sw_window,
                                    &self.config,
                                )?;
                                pack_sw_blocks = flash.sw_blocks();
                                let (nope_ptr, ng) =
                                    row_prepared.k_prepared.data.device_ptr(&self.ctx.stream);
                                pack_nope_ptrs.push(nope_ptr);
                                pack_rope_ptrs.push(
                                    nope_ptr
                                        + crate::attention::flashmla_pack_rope_offset_bytes(
                                            &self.config,
                                        ),
                                );
                                drop(ng);
                                match compressed {
                                    Some(c) => {
                                        let (cp, cg) = c.data.device_ptr(&self.ctx.stream);
                                        pack_compressed_ptrs.push(cp);
                                        drop(cg);
                                    }
                                    None => pack_compressed_ptrs.push(0),
                                }
                                // Persistent device page table from flash state: a
                                // per-call temporary would be freed before the
                                // batched kernel runs (#8 graph UAF).
                                pack_num_logical_pages =
                                    pack_num_logical_pages.max(flash.device_page_table.len());
                                let (pt, pg) = flash.device_page_table.device_ptr(&self.ctx.stream);
                                pack_pt_ptrs.push(pt);
                                drop(pg);
                            } else {
                                crate::attention::flashmla_decode_pack_row(
                                    &self.ctx,
                                    &self.config,
                                    layer.compress_ratio,
                                    flash,
                                    flashmla_scratch,
                                    layer_pool,
                                    sw_window,
                                    &row_prepared.k_prepared,
                                    compressed,
                                    &slot.start_pos_device,
                                )?;
                            }
                            flash_batch.gather_q_row(
                                &self.ctx,
                                &self.config,
                                &row_prepared.q_prepared,
                                &self.tp,
                                r,
                                row_prepared.local_heads,
                            )?;
                            // Gather this row's indexer top-k `selected` into the
                            // contiguous `selected_batched[r * index_topk..]` so the
                            // ONE batched index build can read it per row. SW/HCA
                            // skip this (selected is None, selected_ptr=0). Under the
                            // batched DSA select `selected` is None — filled after
                            // this loop — so skip the per-row gather.
                            if runs_indexer && !use_batched_dsa_select {
                                let sel = row_prepared.selected.as_ref().ok_or_else(|| {
                                anyhow!(
                                    "DSv4 batched indexer decode: row {r} missing indexer selected"
                                )
                            })?;
                                flash_batch.gather_selected_row(&self.ctx, sel, r)?;
                            }
                            // Keep the prepared buffers alive to function return: the
                            // batched fwd reads the gathered Q and the finish loop
                            // reads k_prepared/local_attn (premature-free guard).
                            keepalive.keep_hidden(&row_prepared.q_prepared);
                            keepalive.keep_hidden(&row_prepared.k_prepared);
                            keepalive.keep_hidden(&row_prepared.local_attn);
                            if let Some(sel) = row_prepared.selected.as_ref() {
                                keepalive.keep_i32(sel);
                            }
                            prepared.push(row_prepared);
                        }
                        // ONE batched SW one-token + compressed-delta pack over all
                        // N rows, issued before build_layer_batch_meta / the batched
                        // fwd read the pool. Uploaded ptr arrays are kept alive past
                        // the launch via `ptr_keepalive` / `keepalive`.
                        if pack_batched && n > 0 {
                            crate::profile::profile_op(
                                ctx,
                                "attention_pack",
                                Some(layer_idx),
                                seq_len,
                                || {
                                    let pool_ptr = {
                                        let (
                                            layer_pool,
                                            _dsa,
                                            _flash_batch,
                                            _flashmla_scratch,
                                            _prefill,
                                        ) = kv_adapter
                                            .layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                                        layer_pool.flashmla_pool_base_ptr(&self.ctx)?
                                    };
                                    let positions =
                                        batched_positions.as_ref().ok_or_else(|| {
                                            anyhow!(
                                                "DSv4 op-c batched pack: batched positions missing"
                                            )
                                        })?;
                                    let nope_arr =
                                        crate::ops::upload_u64(&self.ctx, &pack_nope_ptrs)?;
                                    let rope_arr =
                                        crate::ops::upload_u64(&self.ctx, &pack_rope_ptrs)?;
                                    let compressed_arr =
                                        crate::ops::upload_u64(&self.ctx, &pack_compressed_ptrs)?;
                                    let pt_arr = crate::ops::upload_u64(&self.ctx, &pack_pt_ptrs)?;
                                    crate::attention::flashmla_decode_pack_batched(
                                        &self.ctx,
                                        &self.config,
                                        layer.compress_ratio,
                                        pack_sw_blocks,
                                        n,
                                        pool_ptr,
                                        &nope_arr,
                                        &rope_arr,
                                        &compressed_arr,
                                        positions,
                                        &pt_arr,
                                        pack_num_logical_pages,
                                    )?;
                                    ptr_keepalive.push(nope_arr);
                                    ptr_keepalive.push(rope_arr);
                                    ptr_keepalive.push(compressed_arr);
                                    ptr_keepalive.push(pt_arr);
                                    Ok(())
                                },
                            )?;
                        }
                        Ok(())
                    },
                )?;
                // ONE batched CSA select: the per-row paged-MQA logits + topk for
                // all N rows, writing into `selected_batched`. Runs AFTER all N
                // rows' DSA caches are populated and BEFORE `build_layer_batch_meta`
                // reads it.
                if use_batched_dsa_select {
                    let (q_i_batch, weights_batch) =
                        if let Some((q_i, weights)) = indexer_query_kv_score.as_ref() {
                            (q_i, weights)
                        } else {
                            (
                                dsa_stage_q_i
                                    .as_ref()
                                    .expect("batched DSA staging q present"),
                                dsa_stage_weights
                                    .as_ref()
                                    .expect("batched DSA staging weights present"),
                            )
                        };
                    let key_counts = dsa_key_counts
                        .take()
                        .expect("batched DSA key_counts present");
                    ensure!(
                        key_counts.len() == n,
                        "DSv4 batched CSA select: captured {} key_counts != n {}",
                        key_counts.len(),
                        n
                    );
                    // GLM SparseIndexed: full-sequence indexer, every token a key
                    // (ratio=1); context_lens = abs_pos / 1 = abs_pos. CSA keeps its
                    // compress_ratio. ensure still holds (1 > 0).
                    let ratio = if layer.mode == DeepSeekV4AttentionMode::SparseIndexed {
                        1
                    } else {
                        layer.compress_ratio
                    };
                    ensure!(
                        ratio > 0,
                        "DSv4 batched indexer select: indexer layer must have ratio>0"
                    );
                    // Byte-equivalent to the single-row GPU fill:
                    //   context_lens[r] = min(key_count_r, abs_pos_r / ratio)
                    //   positions[r]    = abs_pos_r   (abs_pos_r = start_positions[r])
                    let mut context_lens_host = Vec::with_capacity(n);
                    let mut positions_host = Vec::with_capacity(n);
                    for r in 0..n {
                        let abs_pos = i32::try_from(start_positions[r]).map_err(|_| {
                            anyhow!(
                                "DSv4 batched CSA abs_pos {} overflows i32",
                                start_positions[r]
                            )
                        })?;
                        let avail = (abs_pos / ratio as i32).min(key_counts[r]);
                        context_lens_host.push(avail);
                        positions_host.push(abs_pos);
                    }
                    let local_index_heads = self.config.index_n_heads;
                    let score_scale = (self.config.index_head_dim as f32).powf(-0.5)
                        * (self.config.index_n_heads as f32).powf(-0.5);
                    let (layer_pool, dsa_shared, flash_batch, _flashmla_scratch, _prefill) =
                        kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                    let dsa_shared = dsa_shared.ok_or_else(|| {
                        anyhow!("DSv4 batched CSA select: shared DSA scratch missing")
                    })?;
                    let flash_batch = flash_batch
                        .ok_or_else(|| anyhow!("DSv4 batched CSA select: batch scratch missing"))?;
                    crate::attention::csa_select_official_batched(
                        &self.ctx,
                        &self.config,
                        q_i_batch,
                        weights_batch,
                        dsa_shared,
                        layer_pool,
                        n,
                        &slot_ids[..n],
                        &context_lens_host,
                        &positions_host,
                        local_index_heads,
                        score_scale,
                        flash_batch.selected_batched_mut(),
                        &mut keepalive,
                    )?;
                }
                crate::profile::profile_op(ctx, "attention_fwd", Some(layer_idx), seq_len, || {
                    let (layer_pool, _dsa, flash_batch, _flashmla_scratch, _prefill) =
                        kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                    let flash_batch = flash_batch.ok_or_else(|| {
                        anyhow!("DSv4 batched decode lane: batch scratch missing")
                    })?;
                    // For CSA, build_layer_batch_meta sources the per-row top-k from
                    // the internal `selected_batched` buffer; SW/HCA pass
                    // selected_ptr=0 inside.
                    flash_batch.build_layer_batch_meta(
                        &self.ctx,
                        &self.config,
                        layer_idx,
                        layer.mode,
                        layer.compress_ratio,
                        &start_positions[..n],
                        &slot_block_offsets,
                        &page_tables,
                    )?;
                    let sm_scale = prepared[0].sm_scale;
                    flash_batch.decode_lane_fwd(
                        &self.ctx,
                        &self.config,
                        &layer.attention,
                        layer_pool,
                        layer_idx,
                        n,
                        sm_scale,
                    )
                })?;
                crate::profile::profile_op(
                    ctx,
                    "attention_finish",
                    Some(layer_idx),
                    seq_len,
                    || {
                        // The inverse-RoPE + SW-window WRITE switch from N per-row
                        // launches to ONE batched launch each when `full_flatten` is
                        // on; both must run BEFORE the per-row O-LoRA, which consumes
                        // the inverse-roped `local_attn`. Pass F1 slices each row's
                        // global-head output into its `local_attn` and, under
                        // full_flatten, gathers the device pointers for the batched
                        // FINISH kernels.
                        let mut out_ptrs: Vec<u64> = if full_flatten {
                            Vec::with_capacity(n)
                        } else {
                            Vec::new()
                        };
                        let mut kprep_ptrs: Vec<u64> = if full_flatten {
                            Vec::with_capacity(n)
                        } else {
                            Vec::new()
                        };
                        let mut swcache_ptrs: Vec<u64> = if full_flatten {
                            Vec::with_capacity(n)
                        } else {
                            Vec::new()
                        };
                        // F1 batched: slice ALL n rows in ONE launch.
                        if full_flatten {
                            let local_heads = prepared[0].local_heads;
                            let (_layer_pool, _dsa, flash_batch, _flashmla_scratch, _prefill) =
                                kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                            let flash_batch = flash_batch.ok_or_else(|| {
                                anyhow!("DSv4 batched decode lane: batch scratch missing")
                            })?;
                            flash_batch.slice_out_batched(
                                &self.ctx,
                                &self.config,
                                &self.tp,
                                n,
                                local_heads,
                                &mut local_attn_batched,
                            )?;
                        }
                        let local_width_row = local_attn_batched.hidden_dim;
                        for r in 0..n {
                            // Field accesses below are disjoint (k_prepared `&`,
                            // local_attn `&mut`) so the borrow checker splits them.
                            let p = &mut prepared[r];
                            if full_flatten {
                                // The output pointer is row r's slice of the
                                // contiguous `local_attn_batched`, which the F2
                                // O-LoRA then reads. Single-stream: ptrs stay valid.
                                let row_view = local_attn_batched
                                    .data
                                    .slice(r * local_width_row..(r + 1) * local_width_row);
                                let (op, og) = row_view.device_ptr(&ctx.stream);
                                out_ptrs.push(op);
                                drop(og);
                                let (kp, kg) = p.k_prepared.data.device_ptr(&ctx.stream);
                                kprep_ptrs.push(kp);
                                drop(kg);
                                let slot = &mut slots[slot_ids[r]];
                                let sw_window = slot.attention[layer_idx].sw_window_cache_mut();
                                let (cp, cg) = sw_window.device_ptr_mut(&ctx.stream);
                                swcache_ptrs.push(cp);
                                drop(cg);
                            } else {
                                {
                                    let (
                                        _layer_pool,
                                        _dsa,
                                        flash_batch,
                                        _flashmla_scratch,
                                        _prefill,
                                    ) = kv_adapter.layer_dsa_and_flashmla_batch_mut(layer_idx)?;
                                    let flash_batch = flash_batch.ok_or_else(|| {
                                        anyhow!("DSv4 batched decode lane: batch scratch missing")
                                    })?;
                                    flash_batch.slice_out_row(
                                        &self.ctx,
                                        &self.config,
                                        &self.tp,
                                        r,
                                        p.local_heads,
                                        &mut p.local_attn,
                                    )?;
                                }
                                let slot = &mut slots[slot_ids[r]];
                                let sw_window = slot.attention[layer_idx].sw_window_cache_mut();
                                crate::attention::flashmla_decode_finish_row(
                                    &self.ctx,
                                    &self.config,
                                    sw_window,
                                    &p.k_prepared,
                                    &mut p.local_attn,
                                    start_positions[r],
                                    &slot.start_pos_device,
                                    p.local_heads,
                                    p.rope,
                                )?;
                            }
                        }
                        // Batched FINISH tail: ONE inverse-RoPE + ONE SW-window write
                        // over all N rows. local_heads / rope params are uniform.
                        if full_flatten {
                            let positions = batched_positions.as_ref().ok_or_else(|| {
                                anyhow!("DSv4 full-flatten finish: batched positions missing")
                            })?;
                            let local_heads = prepared[0].local_heads;
                            let rope = prepared[0].rope;
                            let out_arr = crate::ops::upload_u64(&self.ctx, &out_ptrs)?;
                            let kprep_arr = crate::ops::upload_u64(&self.ctx, &kprep_ptrs)?;
                            let swcache_arr = crate::ops::upload_u64(&self.ctx, &swcache_ptrs)?;
                            crate::attention::flashmla_decode_inverse_rope_batched(
                                &self.ctx,
                                &self.config,
                                &out_arr,
                                positions,
                                n,
                                local_heads,
                                rope,
                            )?;
                            crate::attention::flashmla_decode_sw_window_batched(
                                &self.ctx,
                                &self.config,
                                &kprep_arr,
                                &swcache_arr,
                                positions,
                                n,
                            )?;
                            ptr_keepalive.push(out_arr);
                            ptr_keepalive.push(kprep_arr);
                            ptr_keepalive.push(swcache_arr);
                        }
                        // Pass F2: O-LoRA → attn_out (consumes the inverse-roped
                        // output). full_flatten batches over N in ONE
                        // mla_oproj(token_count=n) — plain-o, single-output-group and
                        // grouped all read `local_attn_batched` and write straight
                        // into `attn_out`. The !full_flatten lane stays per-row.
                        if full_flatten {
                            // local_attn_batched is [local_width, n], read token-major.
                            let slot = &mut slots[slot_ids[0]];
                            crate::attention::mla_oproj(
                                &self.ctx,
                                &layer.attention,
                                &mut slot.attention[layer_idx],
                                // Decode: prefill DeepGEMM lane never taken. The
                                // decode lane reuses slot 0's shared transient
                                // fused_wqkv scratch at M=n.
                                None,
                                &local_attn_batched,
                                n,
                                &mut keepalive,
                                &mut attn_out,
                            )?;
                        } else {
                            for r in 0..n {
                                let row_src = &prepared[r].local_attn;
                                let slot = &mut slots[slot_ids[r]];
                                crate::attention::mla_oproj(
                                    &self.ctx,
                                    &layer.attention,
                                    &mut slot.attention[layer_idx],
                                    None,
                                    row_src,
                                    1,
                                    &mut keepalive,
                                    &mut attn_out_row,
                                )?;
                                let mut dst = attn_out
                                    .data
                                    .slice_mut(r * hidden_size..(r + 1) * hidden_size);
                                ctx.stream
                                    .memcpy_dtod(&attn_out_row.data, &mut dst)
                                    .map_err(|e| {
                                        anyhow!("DSv4 batched attn copy-out failed: {e}")
                                    })?;
                            }
                        }
                        Ok(())
                    },
                )?;
            } else {
                crate::profile::profile_op(ctx, "attention", Some(layer_idx), seq_len, || {
                    for r in 0..n {
                        let src = normed.data.slice(r * hidden_size..(r + 1) * hidden_size);
                        ctx.stream
                            .memcpy_dtod(&src, &mut normed_row.data)
                            .map_err(|e| anyhow!("DSv4 batched attn copy-in failed: {e}"))?;
                        let (layer_pool, dsa_shared, flashmla_scratch, prefill_shared, _fp32) =
                            kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                        let slot = &mut slots[slot_ids[r]];
                        crate::attention::mla_attention(
                            &self.ctx,
                            &self.config,
                            &layer.attention,
                            layer.mode,
                            layer.compress_ratio,
                            layer_idx,
                            &normed_row,
                            &mut slot.attention[layer_idx],
                            layer_pool,
                            dsa_shared,
                            flashmla_scratch,
                            prefill_shared,
                            // Decode lane (start_pos_device Some): probe unreachable.
                            None,
                            start_positions[r],
                            Some(&slot.start_pos_device),
                            None,
                            &self.tp,
                            &mut attn_out_row,
                            &mut keepalive,
                        )?;
                        let mut dst = attn_out
                            .data
                            .slice_mut(r * hidden_size..(r + 1) * hidden_size);
                        ctx.stream
                            .memcpy_dtod(&attn_out_row.data, &mut dst)
                            .map_err(|e| anyhow!("DSv4 batched attn copy-out failed: {e}"))?;
                        // No per-row host sync: every op runs on ctx.stream, so
                        // stream ordering serializes row r's reads of the shared
                        // scratch before row r+1's writes.
                    }
                    Ok(())
                })?;
            }
            keepalive.keep_hidden(&attn_out);
            // Row-parallel O-LoRA: one all-reduce over [N, hidden]. NOT bit-identical
            // to N per-row all-reduces — NCCL tiles a [hidden,N] message differently,
            // so identical-input rows pick up ~1 bf16 ULP of drift.
            crate::profile::profile_op(ctx, "attn_allreduce", Some(layer_idx), seq_len, || {
                self.tp.all_reduce_sum(&self.ctx, &mut attn_out)
            })?;
            // SAFETY: hc_post / add_batch writes the full stream buffer.
            let mut attn_stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
            self.hc_post_fold(
                attn_mhc.as_ref(),
                HcHalf::Attn,
                layer_idx,
                seq_len,
                &attn_out,
                &stream,
                &mut attn_stream,
            )?;
            keepalive.keep_hidden(&attn_stream);
            stream = attn_stream;

            // MoE half.
            // SAFETY: uninit device scratch; fully written before first read.
            let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            let ffn_mhc = self.hc_pre_norm(
                layer,
                HcHalf::Ffn,
                layer_idx,
                seq_len,
                &stream,
                &mut normed,
                &mut keepalive,
            )?;
            keepalive.keep_hidden(&normed);
            // Routed MoE over the whole [N] batch. allreduce transport: one router
            // gemm + one DeepGEMM grouped expert GEMM over N×topk routes + one EP
            // all-reduce. deepep transport: the token-owned LL / intranode pipelines,
            // natively [N]-batched. Bit-identity vs per-row is NOT expected; gated on
            // needle retrieval, not byte-parity. A GLM dense layer replaces the whole
            // routed + shared MoE with a plain SwiGLU FFN.
            let mut moe_with_shared =
                // SAFETY: uninit device scratch; fully written before first read.
                unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
            if let Some(dense) = layer.dense_mlp.as_ref() {
                crate::profile::profile_op(ctx, "mlp", Some(layer_idx), seq_len, || {
                    dsv4_dense_mlp_forward(
                        &self.ctx,
                        dense,
                        &normed,
                        &mut moe_with_shared,
                        self.config.swiglu_limit,
                        &mut keepalive,
                    )
                })?;
                keepalive.keep_hidden(&moe_with_shared);
            } else {
                // SAFETY: uninit device scratch; fully written before first read.
                let mut moe_out = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                if use_deepep_transport {
                    #[cfg(feature = "deepep")]
                    {
                        let transport = self.deepep.as_ref().ok_or_else(|| {
                            anyhow!(
                                "ARLE_DSV4_MOE_TRANSPORT=deepep but DeepEP transport is not booted"
                            )
                        })?;
                        if transport.is_low_latency() {
                            // Token-owned LL path: DeepEP dispatch/combine are
                            // collectives every rank enters even at owned_n == 0.
                            // The LL scratch is whole-forward scratch, parked
                            // per-slot, so the batch borrows row 0's.
                            let scratch = slots[slot_ids[0]]
                                .deepep_ll_scratch
                                .as_mut()
                                .ok_or_else(|| {
                                    anyhow!("deepep_ll selected but slot LL scratch not allocated")
                                })?;
                            self.tp.shard_rows_and_allreduce(
                                &self.ctx,
                                &normed,
                                &mut moe_out,
                                hidden_size,
                                seq_len,
                                |owned_in, owned_out, start, owned_n| {
                                    keepalive.keep_hidden(owned_in);
                                    keepalive.keep_hidden(owned_out);
                                    crate::moe::dsv4_moe_forward_deepep_ll(
                                        self,
                                        transport,
                                        scratch,
                                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                                        &tokens[start..start + owned_n],
                                        tokens.len(),
                                        owned_in,
                                        owned_out,
                                        &mut keepalive,
                                    )
                                },
                            )?;
                        } else {
                            // Intranode normal-mode DeepEP: its combine reduces
                            // across EP, so no moe all-reduce is needed.
                            crate::moe::dsv4_moe_forward_deepep(
                                self,
                                transport,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                tokens,
                                &normed,
                                &mut moe_out,
                                &mut keepalive,
                            )?;
                        }
                    }
                    #[cfg(not(feature = "deepep"))]
                    anyhow::bail!(
                        "ARLE_DSV4_MOE_TRANSPORT=deepep requires infer-cuda feature deepep"
                    );
                } else {
                    let tail = kv_adapter.moe_tail_scratch_mut();
                    let needs_moe_allreduce = crate::profile::profile_op(
                        ctx,
                        "moe_route",
                        Some(layer_idx),
                        seq_len,
                        || {
                            crate::moe::dsv4_moe_forward(
                                self,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                tokens,
                                &normed,
                                &mut moe_out,
                                &mut keepalive,
                                tail,
                                mega_epoch,
                            )
                        },
                    )?;
                    keepalive.keep_hidden(&moe_out);
                    // Routed experts are EP-sharded → sum, then add the replicated
                    // shared expert once per rank. One all-reduce over [N, hidden].
                    if needs_moe_allreduce {
                        crate::profile::profile_op(
                            ctx,
                            "moe_allreduce",
                            Some(layer_idx),
                            seq_len,
                            || self.tp.all_reduce_sum(&self.ctx, &mut moe_out),
                        )?;
                    }
                }
                keepalive.keep_hidden(&moe_out);
                // Grouped shared expert over [N]: one batched SwiGLU GEMM pair.
                // Per-rank token-sharded waterfill was killed 2026-07-06: at n=64 it
                // showed no wall-clock gain (75.9s vs 73.8s, within trial noise).
                // Reuse the model-wide pooled shared-expert output + FP8 scratch
                // (#29) instead of per-layer `uninit` + fresh temporaries.
                let (shared_out, shared_scratch) = kv_adapter.shared_expert_decode_mut();
                let shared = shared_out;
                let scratch = shared_scratch
                    .ok_or_else(|| anyhow!("DSv4 batched decode requires shared-expert scratch"))?;
                shared.seq_len = seq_len; // rows ≤ MAX_SPEC_VERIFY_ROWS ≤ max_m
                ensure!(
                    shared.hidden_dim == hidden_size,
                    "DSv4 batched shared out hidden {} != {}",
                    shared.hidden_dim,
                    hidden_size
                );
                crate::profile::profile_op(ctx, "shared_expert", Some(layer_idx), seq_len, || {
                    crate::moe::dsv4_shared_expert_forward_decode_scratch(
                        &self.ctx,
                        &self.ctx.stream,
                        layer.moe.as_ref().expect("DSv4 layer.moe"),
                        &normed,
                        shared,
                        self.config.swiglu_limit,
                        scratch,
                    )
                })?;
                // SAFETY: add_batch writes the full [seq_len, hidden_size] buffer.
                crate::profile::profile_op(ctx, "shared_add", Some(layer_idx), seq_len, || {
                    crate::ops::add_batch(&self.ctx, &moe_out, shared, &mut moe_with_shared)
                })?;
                keepalive.keep_hidden(&moe_with_shared);
            }
            // SAFETY: hc_post / add_batch writes the full stream buffer.
            let mut ffn_stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
            self.hc_post_fold(
                ffn_mhc.as_ref(),
                HcHalf::Ffn,
                layer_idx,
                seq_len,
                &moe_with_shared,
                &stream,
                &mut ffn_stream,
            )?;
            keepalive.keep_hidden(&ffn_stream);
            stream = ffn_stream;
            // DSpark T3: capture each row's wide HC stream at this layer's OUTPUT
            // into its slot's tap buffer, gated so the default path is unchanged.
            if dspark
                && let Some(tap_idx) = self
                    .config
                    .dspark_target_layer_ids
                    .iter()
                    .position(|&l| l == layer_idx)
            {
                for r in 0..n {
                    let tap = &mut slots[slot_ids[r]].dspark_taps[tap_idx];
                    self.capture_mtp_stream_hidden(&stream, r, 1, tap, &mut keepalive)?;
                }
            }
        }

        for r in 0..n {
            slots[slot_ids[r]].seq_len += 1;
        }
        // Per-step per-GEMV breakdown (rank-0 only; self-gates on
        // ARLE_DSV4_LINEAR_PROFILE).
        crate::linear_profile::print_rank0("decode-step");
        Ok((stream, keepalive))
    }

    /// Cross-slot batched MTP verify: N draft chains in ONE layer-major forward
    /// over `M = Σ_s row_count_s` rows, grouped contiguously by slot. MoE / HC-wrap
    /// / norm / head run over all M rows; attention stays per-slot — one FlashMLA
    /// sparse verify call per chain chunk per layer, with ancestor metadata making
    /// row r attend slot s's committed KV plus `scheds[s].ancestors[r]`. The
    /// lm_head extraction batches fold + projection + argmax over all M rows (one
    /// GEMM, one argmax, one D2H).
    ///
    /// Returns per slot `(argmax, hiddens)`: `argmax[j]` is the target's argmax
    /// AFTER chain row j; `hiddens[j]` is that row's MTP stream hidden.
    ///
    /// PRECONDITIONS: each slot's `seq_len == start_pos_s` (contiguous append);
    /// allreduce transport only (the deepep_ll owned-shard partition is keyed on
    /// the whole-batch `seq_len`, not per-slot verify chunks); the draft already
    /// wrote each slot's frozen-layer rings.
    ///
    /// Mutates only function-local scratch plus `slot.spec_normed` (the combined
    /// `[M,hidden]` normed sliced per slot). Slot rings and `slot.seq_len` are NOT
    /// touched — the verify lane is frozen and the caller commits.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn forward_decode_batch_verify(
        &self,
        slots: &mut [Dsv4SlotState],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_ids: &[usize],
        row_tokens: &[Vec<u32>],
        start_positions: &[usize],
        scheds: &[SpecVerifySchedule],
    ) -> Result<(Vec<(Vec<u32>, Vec<DeviceVec>)>, HiddenStates)> {
        let n = slot_ids.len();
        ensure!(n > 0, "DSv4 batched verify requires at least one chain");
        ensure!(
            row_tokens.len() == n && start_positions.len() == n && scheds.len() == n,
            "DSv4 batched verify surface length mismatch (slots {n}, chains {}, starts {}, scheds {})",
            row_tokens.len(),
            start_positions.len(),
            scheds.len()
        );
        ensure!(
            !crate::runtime_flags::dsv4_moe_transport()?.is_deepep(),
            "DSv4 batched MTP verify supports the allreduce transport only \
             (the deepep_ll owned-shard partition is keyed on the whole-batch \
              seq_len, not per-slot verify chunks)"
        );

        let lens: Vec<usize> = row_tokens.iter().map(|c| c.len()).collect();
        for (s, &len) in lens.iter().enumerate() {
            ensure!(
                len >= 1 && scheds[s].positions.len() == len && scheds[s].ancestors.len() == len,
                "DSv4 batched verify slot {s} chain rows {len} != schedule rows ({}, {})",
                scheds[s].positions.len(),
                scheds[s].ancestors.len()
            );
            scheds[s].validate_sparse_at(start_positions[s])?;
            let slot = &slots[slot_ids[s]];
            ensure!(
                slot.seq_len == start_positions[s],
                "DSv4 batched verify slot {} seq_len {} != start_pos {}; verify requires contiguous appends",
                slot_ids[s],
                slot.seq_len,
                start_positions[s]
            );
            ensure!(
                start_positions[s] + len <= slot.max_seq_len,
                "DSv4 batched verify slot {} sequence {} exceeds max_seq_len {}",
                slot_ids[s],
                start_positions[s] + len,
                slot.max_seq_len
            );
        }
        let mut offsets = Vec::with_capacity(n);
        let mut acc = 0usize;
        for &len in &lens {
            offsets.push(acc);
            acc += len;
        }
        let m = acc; // total rows over all per-slot chains

        let hidden_size = self.config.hidden_size;
        let hc_mult = self.config.hc_mult;
        let stream_dim = hidden_size * hc_mult;
        let seq_len = m; // batch dimension: M verify rows
        let mega_epoch = self.begin_mega_moe_forward(seq_len)?;
        let dspark = self.config.is_dspark();
        let mut keepalive = Dsv4ForwardKeepalive::new(false);

        let tokens: Vec<u32> = row_tokens.iter().flat_map(|c| c.iter().copied()).collect();
        let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();

        let token_ids = crate::ops::upload_i32(&self.ctx, &token_ids_host)?;
        keepalive.keep_i32(&token_ids);
        // SAFETY: embed_stream writes the full stream buffer.
        let mut stream = unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
        self.embed_stream(&token_ids, seq_len, &mut stream, &mut keepalive)?;

        // Per-slot chain verify: one FlashMLA sparse forward per slot chunk per
        // layer, with prefix metadata expressing row r -> ancestors explicitly.
        let max_chunk = *lens.iter().max().unwrap_or(&1);
        // SAFETY: uninit device scratch; fully written before first read.
        let mut normed_chunk = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, max_chunk)? };
        // SAFETY: uninit device scratch; fully written before first read.
        let mut attn_chunk = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, max_chunk)? };
        keepalive.keep_hidden(&normed_chunk);
        keepalive.keep_hidden(&attn_chunk);
        let sparse_metas: Vec<crate::attention::Dsv4ChainVerifyAttnMeta> = scheds
            .iter()
            .map(|s| {
                crate::attention::Dsv4ChainVerifyAttnMeta::new(
                    &self.ctx,
                    &s.positions,
                    &s.ancestors,
                )
            })
            .collect::<Result<_>>()?;

        crate::attention::set_dsv4_verify_frozen(true);
        let result = (|| -> Result<()> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                // SAFETY: uninit device scratch; fully written before first read.
                let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                let attn_mhc = self.hc_pre_norm(
                    layer,
                    HcHalf::Attn,
                    layer_idx,
                    seq_len,
                    &stream,
                    &mut normed,
                    &mut keepalive,
                )?;
                keepalive.keep_hidden(&normed);

                // Attention PER SLOT / PER CHUNK: write the slot block into chunk
                // scratch, run the sparse chain-verify lane once, then scatter back.
                // SAFETY: each [off..off+len) block of attn_out is written by the
                // copy-out below before attn_out is read by hc_post.
                let mut attn_out =
                    unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                {
                    crate::profile::profile_op(
                        &self.ctx,
                        "attention",
                        Some(layer_idx),
                        seq_len,
                        || {
                            for s in 0..n {
                                let off = offsets[s];
                                let len = lens[s];
                                normed_chunk.seq_len = len;
                                attn_chunk.seq_len = len;
                                let src = normed
                                    .data
                                    .slice(off * hidden_size..(off + len) * hidden_size);
                                self.ctx
                                    .stream
                                    .memcpy_dtod(
                                        &src,
                                        &mut normed_chunk.data.slice_mut(0..len * hidden_size),
                                    )
                                    .map_err(|e| {
                                        anyhow!("DSv4 batched verify chunk copy-in failed: {e}")
                                    })?;
                                // Commit-fold scatter: persist THIS slot's per-layer
                                // attn-normed chain rows into the OWNING slot's
                                // `spec_normed[layer_idx]`. The combined `normed` is
                                // sliced per slot, so there is no cross-slot aliasing.
                                let slot = &mut slots[slot_ids[s]];
                                let cache = slot.spec_normed.as_mut().ok_or_else(|| {
                                    anyhow!("DSv4 batched verify without spec_normed cache")
                                })?;
                                let mut dst = cache[layer_idx].data.slice_mut(0..len * hidden_size);
                                self.ctx
                                    .stream
                                    .memcpy_dtod(
                                        &normed
                                            .data
                                            .slice(off * hidden_size..(off + len) * hidden_size),
                                        &mut dst,
                                    )
                                    .map_err(|e| {
                                        anyhow!(
                                            "DSv4 batched verify spec_normed persist failed: {e}"
                                        )
                                    })?;
                                let (
                                    layer_pool,
                                    dsa_shared,
                                    flashmla_scratch,
                                    prefill_shared,
                                    fp32,
                                ) = kv_adapter.layer_and_dsa_shared_mut(layer_idx)?;
                                let slot = &mut slots[slot_ids[s]];
                                crate::attention::mla_attention(
                                    &self.ctx,
                                    &self.config,
                                    &layer.attention,
                                    layer.mode,
                                    layer.compress_ratio,
                                    layer_idx,
                                    &normed_chunk,
                                    &mut slot.attention[layer_idx],
                                    layer_pool,
                                    dsa_shared,
                                    flashmla_scratch,
                                    prefill_shared,
                                    fp32,
                                    start_positions[s],
                                    None,
                                    Some(&sparse_metas[s]),
                                    &self.tp,
                                    &mut attn_chunk,
                                    &mut keepalive,
                                )?;
                                let mut dst = attn_out
                                    .data
                                    .slice_mut(off * hidden_size..(off + len) * hidden_size);
                                self.ctx
                                    .stream
                                    .memcpy_dtod(
                                        &attn_chunk.data.slice(0..len * hidden_size),
                                        &mut dst,
                                    )
                                    .map_err(|e| {
                                        anyhow!("DSv4 batched verify chunk copy-out failed: {e}")
                                    })?;
                            }
                            Ok(())
                        },
                    )?;
                }
                keepalive.keep_hidden(&attn_out);
                // Row-parallel O-LoRA: one all-reduce over [M, hidden].
                crate::profile::profile_op(
                    &self.ctx,
                    "attn_allreduce",
                    Some(layer_idx),
                    seq_len,
                    || self.tp.all_reduce_sum(&self.ctx, &mut attn_out),
                )?;
                // SAFETY: hc_post / add_batch writes the full stream buffer.
                let mut attn_stream =
                    unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
                self.hc_post_fold(
                    attn_mhc.as_ref(),
                    HcHalf::Attn,
                    layer_idx,
                    seq_len,
                    &attn_out,
                    &stream,
                    &mut attn_stream,
                )?;
                keepalive.keep_hidden(&attn_stream);
                stream = attn_stream;

                // MoE half over the whole [M] batch (the dominant amortization —
                // byte parity NOT expected: grouped GEMM tiles over M differently;
                // gated on needle). Allreduce transport only.
                // SAFETY: uninit device scratch; fully written before first read.
                let mut normed = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                let ffn_mhc = self.hc_pre_norm(
                    layer,
                    HcHalf::Ffn,
                    layer_idx,
                    seq_len,
                    &stream,
                    &mut normed,
                    &mut keepalive,
                )?;
                keepalive.keep_hidden(&normed);
                // GLM dense layer (`per_layer_dense_mlp[i]`): a plain SwiGLU FFN
                // replaces the routed-expert + shared-expert MoE entirely.
                let mut moe_with_shared =
                    // SAFETY: uninit device scratch; fully written before first read.
                    unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                if let Some(dense) = layer.dense_mlp.as_ref() {
                    crate::profile::profile_op(&self.ctx, "mlp", Some(layer_idx), seq_len, || {
                        dsv4_dense_mlp_forward(
                            &self.ctx,
                            dense,
                            &normed,
                            &mut moe_with_shared,
                            self.config.swiglu_limit,
                            &mut keepalive,
                        )
                    })?;
                    keepalive.keep_hidden(&moe_with_shared);
                } else {
                    // SAFETY: the MoE forward writes the full routed output buffer.
                    let mut moe_out =
                        unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                    let needs_moe_allreduce = crate::profile::profile_op(
                        &self.ctx,
                        "moe_route",
                        Some(layer_idx),
                        seq_len,
                        || {
                            crate::moe::dsv4_moe_forward(
                                self,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                &tokens,
                                &normed,
                                &mut moe_out,
                                &mut keepalive,
                                None,
                                mega_epoch,
                            )
                        },
                    )?;
                    keepalive.keep_hidden(&moe_out);
                    if needs_moe_allreduce {
                        crate::profile::profile_op(
                            &self.ctx,
                            "moe_allreduce",
                            Some(layer_idx),
                            seq_len,
                            || self.tp.all_reduce_sum(&self.ctx, &mut moe_out),
                        )?;
                    }
                    // Grouped shared expert over [M] (dense FFN, prefill path).
                    let mut shared =
                        // SAFETY: uninit device scratch; fully written before first
                        // read.
                        unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
                    crate::profile::profile_op(
                        &self.ctx,
                        "shared_expert",
                        Some(layer_idx),
                        seq_len,
                        || {
                            crate::moe::dsv4_shared_expert_forward(
                                &self.ctx,
                                &self.ctx.stream,
                                layer.moe.as_ref().expect("DSv4 layer.moe"),
                                &normed,
                                &mut shared,
                                self.config.swiglu_limit,
                                &mut keepalive,
                            )
                        },
                    )?;
                    keepalive.keep_hidden(&shared);
                    // SAFETY: add_batch writes the full [m, hidden_size] buffer.
                    crate::profile::profile_op(
                        &self.ctx,
                        "shared_add",
                        Some(layer_idx),
                        seq_len,
                        || {
                            crate::ops::add_batch(
                                &self.ctx,
                                &moe_out,
                                &shared,
                                &mut moe_with_shared,
                            )
                        },
                    )?;
                    keepalive.keep_hidden(&moe_with_shared);
                }
                // SAFETY: hc_post / add_batch writes the full stream buffer.
                let mut ffn_stream =
                    unsafe { HiddenStates::uninit(&self.ctx, stream_dim, seq_len)? };
                self.hc_post_fold(
                    ffn_mhc.as_ref(),
                    HcHalf::Ffn,
                    layer_idx,
                    seq_len,
                    &moe_with_shared,
                    &stream,
                    &mut ffn_stream,
                )?;
                keepalive.keep_hidden(&ffn_stream);
                stream = ffn_stream;
                // DSpark T3: capture each slot's chain rows at this layer's OUTPUT.
                // Row block [off_s..off_s+len_s) belongs to slot s; the full chain is
                // written (the commit fold later reads only the accepted prefix).
                if dspark
                    && let Some(tap_idx) = self
                        .config
                        .dspark_target_layer_ids
                        .iter()
                        .position(|&l| l == layer_idx)
                {
                    for s in 0..n {
                        let tap = &mut slots[slot_ids[s]].dspark_taps[tap_idx];
                        self.capture_mtp_stream_hidden(
                            &stream,
                            offsets[s],
                            lens[s],
                            tap,
                            &mut keepalive,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        crate::attention::set_dsv4_verify_frozen(false);
        result?;

        let mut hiddens_all = Vec::with_capacity(m);
        for i in 0..m {
            let mut h = DeviceVec::zeros(&self.ctx, stream_dim)?;
            self.capture_mtp_stream_hidden(&stream, i, 1, &mut h, &mut keepalive)?;
            hiddens_all.push(h);
        }

        let (logits, argmax_all) =
            crate::profile::profile_op(&self.ctx, "lm_head", None, m, || {
                let logits = self.verify_logits_from_stream(&stream, m, &mut keepalive)?;
                let argmax_all = self.mtp_argmax_batch(&logits)?;
                Ok((logits, argmax_all))
            })?;
        std::hint::black_box(keepalive.len());
        drop(keepalive);

        // Slice argmax + hiddens back per slot; hiddens are moved out.
        let mut hiddens_iter = hiddens_all.into_iter();
        let mut out = Vec::with_capacity(n);
        for s in 0..n {
            let len = lens[s];
            let off = offsets[s];
            let argmax = argmax_all[off..off + len].to_vec();
            let mut hiddens = Vec::with_capacity(len);
            for _ in 0..len {
                hiddens.push(
                    hiddens_iter
                        .next()
                        .ok_or_else(|| anyhow!("DSv4 batched verify hidden slice underflow"))?,
                );
            }
            out.push((argmax, hiddens));
        }
        Ok((out, logits))
    }
}
