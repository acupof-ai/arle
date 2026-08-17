use super::*;

pub(crate) enum Qwen35Attn {
    // Size-skewed variants; boxing keeps the enum small (clippy::large_enum_variant).
    Full(Box<FullAttn>),
    Linear(Box<LinearAttn>),
}

/// Gated full attention: the q rows carry the per-head sigmoid gate, so the q
/// part is `heads*head_dim*2` rows.
pub(crate) struct FullAttn {
    /// Row-fused `[q; k; v]` (`[q_gated + 2*kv, hidden]`).
    pub(crate) qkv_proj: DeviceMatrix,
    pub(crate) o_proj: DeviceMatrix,
    pub(crate) q_norm: DeviceVec,
    pub(crate) k_norm: DeviceVec,
    /// B2 CP decode: 1/(attn_tp x cp) head subset, loaded only when cp>1.
    /// Stays resident for the serving lifetime; the OPD offload snapshot runs
    /// cp=1 and never sees it.
    pub(crate) decode: Option<FullAttnDecode>,
}

/// B2 CP decode weight subset: the q/k/v rows and o_proj cols for this rank's
/// 1/(attn_tp x cp) head block, same quant format as the primary sharded load.
pub(crate) struct FullAttnDecode {
    pub(crate) qkv_proj: DeviceMatrix,
    pub(crate) o_proj: DeviceMatrix,
}

pub(crate) struct LinearAttn {
    /// Row-fused `[qkv; z]` (`[qkv_dim + z_dim, hidden]`).
    pub(crate) in_proj_qkvz: DeviceMatrix,
    /// Row-fused `[b; a]` (`[2*Vh, hidden]`): b = rows `0..Vh`, a = `Vh..2*Vh`.
    pub(crate) in_proj_ba: DeviceMatrix,
    pub(crate) conv1d_weight: DeviceVec,
    pub(crate) dt_bias: DeviceVec,
    /// This rank's v-head shard under TP.
    pub(crate) a_log: CudaSlice<f32>,
    /// Broadcast across heads; replicated under TP.
    pub(crate) norm_weight: CudaSlice<f32>,
    pub(crate) out_proj: DeviceMatrix,
    /// B2 CP decode: 1/(attn_tp x cp) head subset, loaded only when cp>1.
    /// dt_bias/a_log are offset into the primary buffers by the decode kernel
    /// (head-indexed), so they need no second copy; the conv weight's subset
    /// channels are three disjoint blocks, so it gets a compact copy.
    pub(crate) decode: Option<LinearAttnDecode>,
}

/// B2 CP decode weight subset: the qkvz/ba rows, out_proj cols, and the compact
/// `[qkv_dim', K]` conv weight for this rank's 1/(attn_tp x cp) v-head block,
/// same quant format as the primary.
pub(crate) struct LinearAttnDecode {
    pub(crate) in_proj_qkvz: DeviceMatrix,
    pub(crate) in_proj_ba: DeviceMatrix,
    pub(crate) out_proj: DeviceMatrix,
    pub(crate) conv1d_weight: DeviceVec,
}

/// B2 CP decode geometry for one linear layer's recurrent advance: the subset
/// dims plus the v-head offset into the primary dt_bias / a_log buffers.
#[derive(Clone, Copy)]
pub(crate) struct LinearDecodeGeom {
    qkv_dim: usize,
    z_dim: usize,
    k_heads: usize,
    v_heads: usize,
    v_off: usize,
}

impl Qwen35Model {
    /// Gated full attention over an explicit contiguous K/V cache (`max_seq_len`
    /// = `k_cache.len / kv_dim`), into `out` (`[hidden, seq]`, beta=0 o_proj
    /// GEMM). `start_pos_dev` is the GPU-resident `start_pos`, same for every
    /// layer of one call; `full_idx` is only a profiling label.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn full_attention_into(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        k_cache: &mut DeviceVec,
        v_cache: &mut DeviceVec,
        full_idx: usize,
        start_pos: usize,
        start_pos_dev: &CudaSlice<i32>,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let seq_len = normed.seq_len;
        // LOCAL per-rank widths: GEMM outputs, caches, and launches must all
        // agree on this rank's head shard.
        let q_dim = self.local_full_attn_q_dim();
        let kv_dim = self.local_full_attn_kv_dim();
        let q_proj_dim = self.local_full_attn_q_proj_dim();

        let FullAttnScratch {
            qkv_fused,
            q_full,
            k_batch,
            v_batch,
            q_prepped,
            attn_heads,
            fa3_lse,
            fa3_oaccum: _,
            fa3_lseaccum: _,
            fa3_semaphore,
            batch_partial_out: _,
            batch_partial_m: _,
            batch_partial_l: _,
            ..
        } = fw;
        let qkv_fused = qkv_fused.get(&self.ctx, q_proj_dim + 2 * kv_dim, seq_len)?;
        let q_full = q_full.get(&self.ctx, q_proj_dim, seq_len)?;
        let k_batch = k_batch.get(&self.ctx, kv_dim, seq_len)?;
        let v_batch = v_batch.get(&self.ctx, kv_dim, seq_len)?;
        crate::profile::profile_op(&self.ctx, "full/qkv_gemm", Some(full_idx), seq_len, || {
            gemm_batch(&self.ctx, &attn.qkv_proj, normed, qkv_fused)?;
            split_qkv(&self.ctx, qkv_fused, q_full, k_batch, v_batch)?;
            Ok(())
        })?;

        let q_prepped = q_prepped.get(&self.ctx, q_dim, seq_len)?;
        let attn_out = attn_heads.get(&self.ctx, q_dim, seq_len)?;

        let max_seq_len = k_cache.len / kv_dim;
        let sm_scale = 1.0f32 / (c.head_dim as f32).sqrt();
        let kv_len = start_pos + seq_len;

        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (k_ptr, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (qp_ptr, _g7) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (kc_ptr, _g8) = k_cache.data.device_ptr_mut(&self.ctx.stream);
            let (vc_ptr, _g9) = v_cache.data.device_ptr_mut(&self.ctx.stream);
            let (sp_ptr, _g10) = start_pos_dev.device_ptr(&self.ctx.stream);
            crate::profile::profile_op(&self.ctx, "full/prep", Some(full_idx), seq_len, || {
                // SAFETY: all buffers valid on ctx.stream; cache sized
                // max_seq_len*kv_dim.
                unsafe {
                    ffi::prefill_attention_hd256_prep_cuda(
                        qf_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        v_ptr as *const ffi::Half,
                        qn_ptr as *const ffi::Half,
                        kn_ptr as *const ffi::Half,
                        cos_ptr as *const ffi::Half,
                        sin_ptr as *const ffi::Half,
                        qp_ptr as *mut ffi::Half,
                        kc_ptr as *mut ffi::Half,
                        vc_ptr as *mut ffi::Half,
                        self.local_q_heads as i32,
                        self.local_kv_heads as i32,
                        c.head_dim as i32,
                        seq_len as i32,
                        sp_ptr as *const i32,
                        c.rotary_dim as i32,
                        c.rms_norm_eps,
                        max_seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            })?;
        }

        // Decode takes the devpos entry: kv_len is read from the staged
        // `start_pos` device buffer, so one captured graph replays across
        // positions. Prefill keeps the host-scalar entry (never captured).
        {
            let (q_ptr, _g0) = q_prepped.data.device_ptr(&self.ctx.stream);
            let (kc_ptr, _g1) = k_cache.data.device_ptr(&self.ctx.stream);
            let (vc_ptr, _g2) = v_cache.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q_prepped/caches/out valid on ctx.stream for the shapes
            // above; `start_pos_dev` is the forward-level staged position (one
            // i32, value == start_pos).
            crate::profile::profile_op(
                &self.ctx,
                "full/attention",
                Some(full_idx),
                seq_len,
                || {
                    // SAFETY: ptrs from live device allocations sized to the dims
                    // passed.
                    unsafe {
                        if seq_len == 1
                            && qwen35_fa2_sm70_enabled(&self.ctx)
                            && !crate::runtime_flags::qwen35_decode_graph()
                        {
                            // The host kv_len arg is not graph-replay safe, so
                            // captured decode falls through to devpos below.
                            ffi::arle_fa2_sm70_attention_cuda(
                                q_ptr as *const ffi::Half,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_ptr as *mut ffi::Half,
                                self.local_q_heads as i32,
                                self.local_kv_heads as i32,
                                c.head_dim as i32,
                                seq_len as i32,
                                kv_len as i32,
                                max_seq_len as i32,
                                sm_scale,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        } else if seq_len == 1 {
                            let (sp_ptr, _g4) = start_pos_dev.device_ptr(&self.ctx.stream);
                            ffi::nonpaged_prefill_attention_devpos_cuda(
                                q_ptr as *const ffi::Half,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_ptr as *mut ffi::Half,
                                self.local_q_heads as i32,
                                self.local_kv_heads as i32,
                                c.head_dim as i32,
                                seq_len as i32,
                                sp_ptr as *const i32,
                                max_seq_len as i32,
                                sm_scale,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        } else if c.head_dim == 256 && qwen35_fa3_enabled(&self.ctx) {
                            // q/out token-major [S, h, 256], cache head-major
                            // [h_k, max_seq, 256]. An exact `kv_len` as seqlen_k
                            // keeps the shim non-varlen; causal is bottom-right
                            // aligned = chunked-prefill semantics.
                            let lse = fa3_lse.get(&self.ctx, self.local_q_heads * seq_len)?;
                            let sem = fa3_semaphore.get(&self.ctx, 5)?;
                            let (lse_ptr, _g4) = lse.device_ptr_mut(&self.ctx.stream);
                            let (sem_ptr, _g5) = sem.device_ptr_mut(&self.ctx.stream);
                            let head_dim = c.head_dim as i64;
                            let args = ffi::ArleFa3FwdHd256Args {
                                q: q_ptr as *const ffi::Half,
                                k: kc_ptr as *const ffi::Half,
                                v: vc_ptr as *const ffi::Half,
                                o: o_ptr as *mut ffi::Half,
                                softmax_lse: lse_ptr as *mut f32,
                                out_accum: std::ptr::null_mut(),
                                softmax_lse_accum: std::ptr::null_mut(),
                                tile_count_semaphore: sem_ptr as *mut i32,
                                metadata_capacity: 5,
                                cu_seqlens_q: std::ptr::null(),
                                seqused_k: std::ptr::null(),
                                batch: 1,
                                total_q: seq_len as i32,
                                seqlen_q: seq_len as i32,
                                seqlen_k: kv_len as i32,
                                num_heads: self.local_q_heads as i32,
                                num_heads_k: self.local_kv_heads as i32,
                                head_dim: c.head_dim as i32,
                                q_row_stride: q_dim as i64,
                                k_row_stride: head_dim,
                                v_row_stride: head_dim,
                                o_row_stride: q_dim as i64,
                                q_head_stride: head_dim,
                                k_head_stride: max_seq_len as i64 * head_dim,
                                v_head_stride: max_seq_len as i64 * head_dim,
                                o_head_stride: head_dim,
                                softmax_scale: sm_scale,
                                is_causal: 1,
                                num_splits: 1,
                                page_table: std::ptr::null(),
                                page_table_batch_stride: 0,
                                page_size: 0,
                                num_pages: 0,
                                k_page_stride: 0,
                                v_page_stride: 0,
                            };
                            ffi::arle_fa3_fwd_hd256_bf16_cuda(&args, self.ctx.stream.cu_stream())
                                .result()?;
                        } else if qwen35_fa2_sm70_enabled(&self.ctx) {
                            // sm_70 lane: FA3 needs sm_80+.
                            ffi::arle_fa2_sm70_attention_cuda(
                                q_ptr as *const ffi::Half,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_ptr as *mut ffi::Half,
                                self.local_q_heads as i32,
                                self.local_kv_heads as i32,
                                c.head_dim as i32,
                                seq_len as i32,
                                kv_len as i32,
                                max_seq_len as i32,
                                sm_scale,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        } else {
                            ffi::nonpaged_prefill_attention_cuda(
                                q_ptr as *const ffi::Half,
                                kc_ptr as *const ffi::Half,
                                vc_ptr as *const ffi::Half,
                                o_ptr as *mut ffi::Half,
                                self.local_q_heads as i32,
                                self.local_kv_heads as i32,
                                c.head_dim as i32,
                                seq_len as i32,
                                kv_len as i32,
                                max_seq_len as i32,
                                sm_scale,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                    }
                    Ok(())
                },
            )?;
        }

        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g1) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            crate::profile::profile_op(&self.ctx, "full/gate", Some(full_idx), seq_len, || {
                // SAFETY: q_full/attn_out valid on ctx.stream; gate layout per
                // full-attn prep.
                unsafe {
                    ffi::attention_gate_batch_hd256_cuda(
                        qf_ptr as *const ffi::Half,
                        o_ptr as *mut ffi::Half,
                        self.local_q_heads as i32,
                        c.head_dim as i32,
                        seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            })?;
        }

        crate::profile::profile_op(&self.ctx, "full/o_proj", Some(full_idx), seq_len, || {
            gemm_batch(&self.ctx, &attn.o_proj, attn_out, out)
        })?;
        crate::profile::profile_op(&self.ctx, "full/allreduce", Some(full_idx), seq_len, || {
            self.tp.attn_all_reduce_sum(&self.ctx, out)
        })?;
        Ok(())
    }

    /// Paged full attention over `meta`'s ragged page table. RoPE is baked into
    /// the cached K at write time, so a recall-restricted page subset attends
    /// exactly those pages.
    ///
    /// `layer0_query` is the recall sink: on a multi-row prefill the post-RoPE
    /// layer-0 Q is read back head-major `[num_q_heads * head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn full_attention_paged(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        full_idx: usize,
        pool: &PagedKVPool,
        meta: &crate::loader::PageMeta,
        cp: Option<&Qwen35CpPrefill>,
        cp_decode: Option<&Qwen35CpDecode>,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
        layer0_query: Option<&mut Vec<f32>>,
    ) -> Result<()> {
        let c = &self.config;
        // One q row per batch element is the decode kernel's contract.
        let decode = meta.seq_len == 1;
        // 2D (attn_tp × cp): the pool is sequence-sharded (block-cyclic), so
        // decode is flash-decoding — FA3 over the local shard then a cross-cp
        // (lse, out) merge. Heads stay on the attn_tp shard (no cp head
        // subdivision), unlike B2 below.
        let two_d = decode && self.tp.two_d_engaged();
        // B2: cp ranks shard the attn_tp heads via model-load subset weights.
        let b2 = decode && cp_decode.is_some() && !two_d;
        // The merge needs each shard's lse; only the FA3 lane produces one.
        if two_d {
            ensure!(
                qwen35_fa3_enabled(&self.ctx),
                "2D decode merge requires the FA3 paged lane (TileLang produces no lse)"
            );
        }
        let (qkv_proj, o_proj): (&DeviceMatrix, &DeviceMatrix) = if b2 {
            let d = attn
                .decode
                .as_ref()
                .expect("B2 decode weights present when cp_decode active");
            (&d.qkv_proj, &d.o_proj)
        } else {
            (&attn.qkv_proj, &attn.o_proj)
        };
        let (q_heads, kv_heads) = if b2 {
            (self.decode_q_heads(), self.decode_kv_heads())
        } else {
            (self.local_q_heads, self.local_kv_heads)
        };
        let q_dim = q_heads * c.head_dim;
        let kv_dim = kv_heads * c.head_dim;
        let q_proj_dim = q_heads * c.head_dim * 2;
        let sm_scale = 1.0f32 / (c.head_dim as f32).sqrt();
        let stride_page = pool.kv_dim * pool.page_size;

        let FullAttnScratch {
            qkv_fused,
            q_full,
            k_batch,
            v_batch,
            q_prepped,
            attn_heads,
            fa3_lse,
            fa3_oaccum,
            fa3_lseaccum,
            fa3_semaphore,
            cp_in,
            ring_prefill,
            cp_row_gather,
            ..
        } = fw;
        // CP prefill: `normed` is the full chunk but this rank computes only
        // its contiguous q-slice; `meta` is already slice-shaped.
        let normed: &HiddenStates = match cp {
            Some(cpx) => {
                ensure!(
                    pool.format == KVFormat::BF16,
                    "CP prefill requires the BF16 KV pool (guarded at construction)"
                );
                let (off, len) = cpx.slices[self.tp.attn_cp_rank()];
                let h = c.hidden_size;
                let buf = cp_in.get(&self.ctx, h, len)?;
                self.ctx
                    .stream
                    .memcpy_dtod(
                        &normed.data.slice(off * h..(off + len) * h),
                        &mut buf.data.slice_mut(0..len * h),
                    )
                    .map_err(|e| anyhow!("CP q-slice copy failed: {e}"))?;
                buf
            }
            None => normed,
        };
        let rows = normed.seq_len;
        ensure!(
            meta.total_q == rows,
            "Qwen3.6 paged full attention: page table covers {} query tokens != {rows} rows",
            meta.total_q
        );
        let qkv_fused = qkv_fused.get(&self.ctx, q_proj_dim + 2 * kv_dim, rows)?;
        let q_full = q_full.get(&self.ctx, q_proj_dim, rows)?;
        let k_batch = k_batch.get(&self.ctx, kv_dim, rows)?;
        let v_batch = v_batch.get(&self.ctx, kv_dim, rows)?;
        crate::profile::profile_op(
            &self.ctx,
            "full_paged/qkv_gemm",
            Some(full_idx),
            rows,
            || {
                gemm_batch(&self.ctx, qkv_proj, normed, qkv_fused)?;
                split_qkv(&self.ctx, qkv_fused, q_full, k_batch, v_batch)?;
                Ok(())
            },
        )?;

        let attn_out = attn_heads.get(&self.ctx, q_dim, rows)?;

        if let Some(cpx) = cp {
            // 2D ring prefill: one ring pass over the whole prompt. The pool is
            // sharded block-cyclic (logical page g on shard g % cp); each rank
            // scatters its owned pages as the KV blocks rotate through.
            self.ring_prefill_full_attention(
                attn,
                cpx,
                full_idx,
                pool,
                ring_prefill,
                q_full,
                k_batch,
                v_batch,
                attn_out,
            )?;
        } else {
            let q_prepped = q_prepped.get(&self.ctx, q_dim, rows)?;

            // B2: history pages hold every cp rank's KV (prefill keeps the pool
            // replicated); this rank's subset lives at its natural head offset, so
            // shift the pool base for both the prep write and the attention read.
            let (k_pool_ptr, v_pool_ptr) = if b2 {
                let (kv_off, _) = cp_decode
                    .expect("b2 ⇒ cp_decode present")
                    .subset(self.local_kv_heads);
                let off = (kv_off * pool.page_size * c.head_dim * std::mem::size_of::<ffi::Half>())
                    as u64;
                (
                    pool.k_ptr(full_idx, &self.ctx.stream) + off,
                    pool.v_ptr(full_idx, &self.ctx.stream) + off,
                )
            } else {
                (
                    pool.k_ptr(full_idx, &self.ctx.stream),
                    pool.v_ptr(full_idx, &self.ctx.stream),
                )
            };

            {
                let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
                let (k_ptr, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
                let (v_ptr, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
                let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
                let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
                let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
                let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
                let (positions_ptr, _gp) = meta.positions.device_ptr(&self.ctx.stream);
                let (kv_indices_ptr, _gi) = meta.kv_indices.device_ptr(&self.ctx.stream);
                let (kv_indptr_ptr, _gpi) = meta.kv_indptr.device_ptr(&self.ctx.stream);
                let (last_page_len_ptr, _gl) = meta.kv_last_page_len.device_ptr(&self.ctx.stream);
                let (start_pos_ptr, _gs) = meta.start_positions.device_ptr(&self.ctx.stream);
                {
                    let (qp_ptr, _g7) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
                    crate::profile::profile_op(
                        &self.ctx,
                        "full_paged/prep",
                        Some(full_idx),
                        rows,
                        || {
                            // SAFETY: all buffers valid on ctx.stream; pool tail page
                            // allocated; per-row offsets come from the meta's own
                            // prefix sums, so each launch stays inside its row.
                            unsafe {
                                if decode {
                                    // B2: k_pool_ptr/v_pool_ptr are offset to this
                                    // rank's head block, so the subset K/V lands at
                                    // its natural offset in the full-head pool.
                                    // 2D: only the shard owning the new token's
                                    // page writes; the others skip. The predicate
                                    // is computed once in sharded_decode_meta
                                    // (block-cyclic owns_page) and carried on the
                                    // meta — 1 on all non-2D paths.
                                    let write_kv = meta.write_kv;
                                    ffi::decode_prep_paged_hd256_cuda(
                                        qf_ptr as *const ffi::Half,
                                        qp_ptr as *mut ffi::Half,
                                        k_ptr as *const ffi::Half,
                                        v_ptr as *const ffi::Half,
                                        qn_ptr as *const ffi::Half,
                                        kn_ptr as *const ffi::Half,
                                        cos_ptr as *const ffi::Half,
                                        sin_ptr as *const ffi::Half,
                                        positions_ptr as *const i32,
                                        k_pool_ptr as *mut ffi::Half,
                                        v_pool_ptr as *mut ffi::Half,
                                        kv_indices_ptr as *const i32,
                                        kv_indptr_ptr as *const i32,
                                        last_page_len_ptr as *const i32,
                                        q_heads as i32,
                                        kv_heads as i32,
                                        pool.page_size as i32,
                                        stride_page as i32,
                                        meta.batch as i32,
                                        c.rotary_dim as i32,
                                        c.rms_norm_eps,
                                        write_kv,
                                        self.ctx.stream.cu_stream(),
                                    )
                                    .result()?;
                                } else {
                                    // The prep reads ONE scalar start_pos off a
                                    // table based at element 0 — launch per row.
                                    let elem = std::mem::size_of::<ffi::Half>() as u64;
                                    for b in 0..meta.batch {
                                        let (col, pages) =
                                            (meta.q_offsets[b], meta.page_offsets[b]);
                                        let len = meta.q_offsets[b + 1] - col;
                                        ffi::prefill_attention_paged_prep_hd256_cuda(
                                            (qf_ptr + (col * q_proj_dim) as u64 * elem)
                                                as *const ffi::Half,
                                            (qp_ptr + (col * q_dim) as u64 * elem)
                                                as *mut ffi::Half,
                                            (k_ptr + (col * kv_dim) as u64 * elem)
                                                as *const ffi::Half,
                                            (v_ptr + (col * kv_dim) as u64 * elem)
                                                as *const ffi::Half,
                                            qn_ptr as *const ffi::Half,
                                            kn_ptr as *const ffi::Half,
                                            cos_ptr as *const ffi::Half,
                                            sin_ptr as *const ffi::Half,
                                            (kv_indices_ptr + (pages * 4) as u64) as *const i32,
                                            pool.page_size as i32,
                                            k_pool_ptr as *mut ffi::Half,
                                            v_pool_ptr as *mut ffi::Half,
                                            self.local_q_heads as i32,
                                            self.local_kv_heads as i32,
                                            len as i32,
                                            (start_pos_ptr + (b * 4) as u64) as *const i32,
                                            c.rotary_dim as i32,
                                            c.rms_norm_eps,
                                            self.ctx.stream.cu_stream(),
                                        )
                                        .result()?;
                                    }
                                }
                            }
                            Ok(())
                        },
                    )?;
                }
            }

            // Quantize the BF16 tokens the prep just wrote into `k_data[layer]`, so
            // the attention kernel reads the whole history from one quantized pool.
            if pool.format != KVFormat::BF16 {
                let new_rows = meta.new_token_rows.as_ref().ok_or_else(|| {
                    anyhow!(
                        "Qwen35 full-attn FP8/INT8 pool missing new_token_rows in PageMeta \
                     (format={:?})",
                        pool.format
                    )
                })?;
                let kv_dim = self.local_full_attn_kv_dim();
                for &(src, data, scales) in &[
                    (
                        pool.k_ptr(full_idx, &self.ctx.stream),
                        pool.k_data_ptr(full_idx, &self.ctx.stream),
                        pool.k_scales_ptr(full_idx, &self.ctx.stream),
                    ),
                    (
                        pool.v_ptr(full_idx, &self.ctx.stream),
                        pool.v_data_ptr(full_idx, &self.ctx.stream),
                        pool.v_scales_ptr(full_idx, &self.ctx.stream),
                    ),
                ] {
                    kv_quant::quantize_paged_kv_per_token(
                        &self.ctx,
                        src,
                        data,
                        scales,
                        new_rows,
                        self.local_kv_heads,
                        c.head_dim,
                        kv_dim,
                        rows,
                        pool.format,
                    )?;
                }
            }

            {
                let (bsz, total_q, max_q) =
                    (meta.batch as i32, meta.total_q as i32, meta.seq_len as i32);
                let (q_indptr_ptr, _g1) = meta.q_indptr.device_ptr(&self.ctx.stream);
                let (kv_indptr_ptr, _g2) = meta.kv_indptr.device_ptr(&self.ctx.stream);
                let (kv_indices_ptr, _g3) = meta.kv_indices.device_ptr(&self.ctx.stream);
                let (last_page_len_ptr, _g4) = meta.kv_last_page_len.device_ptr(&self.ctx.stream);
                let phase = if decode {
                    ffi::AttnPhase::Decode
                } else {
                    ffi::AttnPhase::Prefill
                };
                {
                    crate::profile::profile_op(
                        &self.ctx,
                        "full_paged/attention",
                        Some(full_idx),
                        rows,
                        || {
                            // Short queries and batch==1 prefill take FA3 paged
                            // split-KV; ragged multi-request prefill keeps TileLang
                            // (routing it here cost c=8 TTFT 12.07→18.23 s).
                            // Quantized pools try the native 1-byte kernel first;
                            // prefill rows and workspace overflow fall through to
                            // the FA3 quant shim, then the varlen kernel below.
                            if (meta.seq_len <= FA3_MAX_QLEN || meta.batch == 1)
                                && c.head_dim == 256
                            {
                                if decode
                                    && matches!(pool.format, KVFormat::FP8E4M3 | KVFormat::INT8)
                                {
                                    // Capped at 16: the pool's split-KV workspace is
                                    // sized for 16 splits at the GQA ratio 8.
                                    let splits = self
                                        .ctx
                                        .sm_count()
                                        .div_ceil(meta.batch.max(1) * kv_heads.max(1))
                                        .max(FA3_DECODE_SPLITS_FLOOR)
                                        .clamp(2, 16);
                                    let needed =
                                        kv_quant::paged_attention_quantized_fa3_workspace_bytes(
                                            meta.total_q,
                                            q_heads,
                                            c.head_dim,
                                            splits,
                                        );
                                    if needed <= pool.quantized_attn_workspace_bytes {
                                        let ws = pool.quantized_attn_workspace()?;
                                        let (rect_ptr, _fr) =
                                            meta.page_table_rect.device_ptr(&self.ctx.stream);
                                        let (q_indptr_ptr, _fq) =
                                            meta.q_indptr.device_ptr(&self.ctx.stream);
                                        let (kv_lens_ptr, _fl) =
                                            meta.kv_lens_dev.device_ptr(&self.ctx.stream);
                                        kv_quant::paged_attention_quantized_fa3(
                                            &self.ctx,
                                            q_prepped,
                                            pool.k_data_ptr(full_idx, &self.ctx.stream),
                                            pool.v_data_ptr(full_idx, &self.ctx.stream),
                                            pool.k_scales_ptr(full_idx, &self.ctx.stream),
                                            pool.v_scales_ptr(full_idx, &self.ctx.stream),
                                            rect_ptr,
                                            q_indptr_ptr,
                                            kv_lens_ptr,
                                            attn_out,
                                            q_heads,
                                            kv_heads,
                                            c.head_dim,
                                            pool.page_size,
                                            meta.page_table_stride,
                                            meta.batch,
                                            meta.total_q,
                                            sm_scale,
                                            pool.format,
                                            splits,
                                            ws,
                                            pool.quantized_attn_workspace_bytes,
                                        )?;
                                        return Ok(());
                                    }
                                }
                                if qwen35_fa3_enabled(&self.ctx)
                                    && matches!(
                                        pool.format,
                                        KVFormat::BF16 | KVFormat::FP8E4M3 | KVFormat::INT8
                                    )
                                {
                                    let (qp_ptr, _g0) =
                                        q_prepped.data.device_ptr_mut(&self.ctx.stream);
                                    // Scoped: the guard releases the attn_out borrow so
                                    // the 2D merge stage can read it below; the u64
                                    // address stays valid (attn_out owns the allocation).
                                    let ao_ptr = {
                                        let (p, _g5) =
                                            attn_out.data.device_ptr_mut(&self.ctx.stream);
                                        p
                                    };
                                    // ONE launch for the whole batch: `seqused_k`
                                    // keeps the K/V batch strides (only `cu_seqlens_k`
                                    // drops them, flash_api.cpp:105-108), which is what
                                    // lets a paged batch share a launch.
                                    // `splits` is an upper bound; FA3 picks the live
                                    // value. One tile per SM is where raising it stops
                                    // paying (+0.36% at batch 8); the floor 8 is the
                                    // measured optimum from batch 4 up and bound only
                                    // batch 1 (32 tiles, 46 of 78 SMs idle).
                                    let splits = if meta.seq_len <= FA3_MAX_QLEN {
                                        match crate::runtime_flags::qwen35_fa3_decode_splits() {
                                            0 => self
                                                .ctx
                                                .sm_count()
                                                .div_ceil(meta.batch.max(1) * kv_heads.max(1))
                                                .max(FA3_DECODE_SPLITS_FLOOR)
                                                .clamp(2, 256),
                                            n => n,
                                        }
                                    } else {
                                        1
                                    };
                                    let accum_rows = q_heads * meta.total_q;
                                    // 2D: room for every cp shard's partial; FA3
                                    // writes this rank's at its rank-major slot.
                                    let lse = if two_d {
                                        fa3_lse
                                            .get(&self.ctx, self.tp.attn_cp_size() * accum_rows)?
                                    } else {
                                        fa3_lse.get(&self.ctx, accum_rows)?
                                    };
                                    let oaccum = fa3_oaccum
                                        .get(&self.ctx, splits * accum_rows * c.head_dim)?;
                                    let lseaccum =
                                        fa3_lseaccum.get(&self.ctx, splits * accum_rows)?;
                                    let meta_cap = meta.batch.div_ceil(4) * 4 * 4 + 1;
                                    let sem = fa3_semaphore.get(&self.ctx, meta_cap)?;
                                    let (lse_ptr, _f0) = lse.device_ptr_mut(&self.ctx.stream);
                                    let my_lse_byte_off = (self.tp.attn_cp_rank()
                                        * accum_rows
                                        * std::mem::size_of::<f32>())
                                        as u64;
                                    let softmax_lse_ptr = if two_d {
                                        // lse is sized cp_size*accum_rows f32 above.
                                        lse_ptr + my_lse_byte_off
                                    } else {
                                        lse_ptr
                                    };
                                    let (oaccum_ptr, _f1) = oaccum.device_ptr_mut(&self.ctx.stream);
                                    let (lseaccum_ptr, _f2) =
                                        lseaccum.device_ptr_mut(&self.ctx.stream);
                                    let (sem_ptr, _f3) = sem.device_ptr_mut(&self.ctx.stream);
                                    let (kv_lens_ptr, _f4) =
                                        meta.kv_lens_dev.device_ptr(&self.ctx.stream);
                                    let (rect_ptr, _f5) =
                                        meta.page_table_rect.device_ptr(&self.ctx.stream);
                                    let head_dim = c.head_dim as i64;
                                    let args = ffi::ArleFa3FwdHd256Args {
                                        q: qp_ptr as *const ffi::Half,
                                        k: k_pool_ptr as *const ffi::Half,
                                        v: v_pool_ptr as *const ffi::Half,
                                        o: ao_ptr as *mut ffi::Half,
                                        softmax_lse: softmax_lse_ptr as *mut f32,
                                        out_accum: oaccum_ptr as *mut f32,
                                        softmax_lse_accum: lseaccum_ptr as *mut f32,
                                        tile_count_semaphore: sem_ptr as *mut i32,
                                        metadata_capacity: meta_cap as i32,
                                        cu_seqlens_q: q_indptr_ptr as *const i32,
                                        seqused_k: kv_lens_ptr as *const i32,
                                        batch: meta.batch as i32,
                                        total_q: meta.total_q as i32,
                                        seqlen_q: meta.seq_len as i32,
                                        seqlen_k: meta.max_kv_len() as i32,
                                        num_heads: q_heads as i32,
                                        num_heads_k: kv_heads as i32,
                                        head_dim: c.head_dim as i32,
                                        q_row_stride: (q_heads * c.head_dim) as i64,
                                        // HND pool [page, h_k, page_size, d].
                                        k_row_stride: head_dim,
                                        v_row_stride: head_dim,
                                        o_row_stride: (q_heads * c.head_dim) as i64,
                                        q_head_stride: head_dim,
                                        k_head_stride: pool.page_size as i64 * head_dim,
                                        v_head_stride: pool.page_size as i64 * head_dim,
                                        o_head_stride: head_dim,
                                        softmax_scale: sm_scale,
                                        // Bottom-right aligned; the shim demotes to
                                        // non-causal at qlen 1.
                                        is_causal: 1,
                                        num_splits: splits as i32,
                                        page_table: rect_ptr as *const i32,
                                        page_table_batch_stride: meta.page_table_stride as i64,
                                        page_size: pool.page_size as i32,
                                        num_pages: pool.max_total_pages as i32,
                                        k_page_stride: stride_page as i64,
                                        v_page_stride: stride_page as i64,
                                    };
                                    if pool.format == KVFormat::BF16 {
                                        // SAFETY: q/o are the live prepped/out buffers; k/v
                                        // are
                                        // the layer's pool base; the page table is the
                                        // meta's
                                        // rectangular mirror, `batch * page_table_stride`
                                        // long.
                                        unsafe {
                                            ffi::arle_fa3_fwd_hd256_bf16_cuda(
                                                &args,
                                                self.ctx.stream.cu_stream(),
                                            )
                                            .result()?;
                                        }
                                    } else {
                                        let quant_args = ffi::ArleFa3FwdHd256QuantArgs {
                                            base: args,
                                            k_data: pool.k_data_ptr(full_idx, &self.ctx.stream)
                                                as *const u8,
                                            v_data: pool.v_data_ptr(full_idx, &self.ctx.stream)
                                                as *const u8,
                                            k_scales: pool.k_scales_ptr(full_idx, &self.ctx.stream)
                                                as *const f32,
                                            v_scales: pool.v_scales_ptr(full_idx, &self.ctx.stream)
                                                as *const f32,
                                            is_fp8: if pool.format == KVFormat::FP8E4M3 {
                                                1
                                            } else {
                                                0
                                            },
                                        };
                                        // SAFETY: same live buffers as the bf16 arm; the
                                        // 1-byte pools and per-(token, head) scales are the
                                        // layer's quantized planes, fresh from the quantize
                                        // pass above.
                                        unsafe {
                                            ffi::arle_fa3_fwd_hd256_quant_cuda(
                                                &quant_args,
                                                self.ctx.stream.cu_stream(),
                                            )
                                            .result()?;
                                        }
                                    }
                                    if two_d {
                                        // Cross-cp flash-decoding merge: pack this
                                        // rank's (lse, out) into one rank-major
                                        // section, single all-gather, weighted
                                        // average in f32. The lse rides the bf16
                                        // collective as f32 pairs — NCCL moves
                                        // bytes, so it stays f32 end to end.
                                        let cp_size = self.tp.attn_cp_size();
                                        let cp_rank = self.tp.attn_cp_rank();
                                        let sect = accum_rows * c.head_dim;
                                        let lse_bf16 = accum_rows * 2;
                                        let section_bf16 = lse_bf16 + sect;
                                        let merge_gather =
                                            cp_row_gather.get(&self.ctx, section_bf16, cp_size)?;
                                        let (mg_ptr, _gp) =
                                            merge_gather.data.device_ptr_mut(&self.ctx.stream);
                                        // SAFETY: mg_ptr is the gather base; lse
                                        // holds cp_size*accum_rows f32 with this
                                        // rank's partial at my_lse_byte_off; both
                                        // live on ctx.stream.
                                        unsafe {
                                            cudarc::driver::result::memcpy_dtod_async(
                                                mg_ptr + cp_rank as u64 * section_bf16 as u64 * 2,
                                                lse_ptr + my_lse_byte_off,
                                                accum_rows * 4,
                                                self.ctx.stream.cu_stream(),
                                            )
                                        }
                                        .map_err(|e| anyhow!("2D merge lse stage failed: {e}"))?;
                                        drop(_gp);
                                        self.ctx
                                            .stream
                                            .memcpy_dtod(
                                                &attn_out.data.slice(0..sect),
                                                &mut merge_gather.data.slice_mut(
                                                    cp_rank * section_bf16 + lse_bf16
                                                        ..cp_rank * section_bf16 + lse_bf16 + sect,
                                                ),
                                            )
                                            .map_err(|e| {
                                                anyhow!("2D merge out stage failed: {e}")
                                            })?;
                                        // One fence bracket: both D2D stages run on
                                        // compute, the gather on comm, the merge
                                        // kernel below waits once.
                                        self.ctx.comm_waits_for_compute()?;
                                        self.cp_all_gather_in_place(merge_gather, section_bf16)?;
                                        self.ctx.compute_waits_for_comm()?;
                                        let (mg_ptr, _gm) =
                                            merge_gather.data.device_ptr(&self.ctx.stream);
                                        let (ao_ptr, _ga) =
                                            attn_out.data.device_ptr_mut(&self.ctx.stream);
                                        // SAFETY: buffers live on ctx.stream; the
                                        // gather's compute-waits-comm fence orders
                                        // it before this launch.
                                        unsafe {
                                            ffi::cross_cp_merge_bf16_hd256_cuda(
                                                mg_ptr as *const ffi::Half,
                                                (section_bf16 / 2) as i32,
                                                lse_bf16 as i32,
                                                section_bf16 as i32,
                                                ao_ptr as *mut ffi::Half,
                                                cp_size as i32,
                                                accum_rows as i32,
                                                c.head_dim as i32,
                                                self.ctx.stream.cu_stream(),
                                            )
                                            .result()?;
                                        }
                                    }
                                    return Ok(());
                                }
                            }
                            // The TileLang lane bakes `num_pages` as a host arg and
                            // would replay stale under graph capture.
                            ensure!(
                                meta.seqlen_k_capture.is_none(),
                                "paged decode graph capture requires the FA3 BF16 lane"
                            );
                            match pool.format {
                                KVFormat::BF16 => {
                                    let (qp_ptr, _g0) =
                                        q_prepped.data.device_ptr_mut(&self.ctx.stream);
                                    let (ao_ptr, _g5) =
                                        attn_out.data.device_ptr_mut(&self.ctx.stream);
                                    // SAFETY: kernel signature from paged_attn_v1 ABI
                                    // (18-arg BF16).
                                    let kernel = ffi::resolve_paged_attn_v1(
                                        c.head_dim as u32,
                                        q_heads as u32,
                                        kv_heads as u32,
                                        phase,
                                    )
                                    .ok_or_else(|| {
                                        anyhow!(
                                            "no HD256 paged {} kernel for q{}_kv{}",
                                            if decode { "decode" } else { "prefill" },
                                            q_heads,
                                            kv_heads
                                        )
                                    })?;
                                    // SAFETY: ptrs from live device allocations sized to
                                    // the dims passed.
                                    unsafe {
                                        kernel(
                                            qp_ptr as *mut ffi::Half,
                                            q_indptr_ptr as *const i32,
                                            k_pool_ptr as *mut ffi::Half,
                                            v_pool_ptr as *mut ffi::Half,
                                            kv_indptr_ptr as *const i32,
                                            kv_indices_ptr as *const i32,
                                            last_page_len_ptr as *const i32,
                                            ao_ptr as *mut ffi::Half,
                                            bsz,
                                            total_q,
                                            max_q,
                                            pool.max_total_pages as i32,
                                            meta.num_pages as i32,
                                            q_heads as i32,
                                            kv_heads as i32,
                                            pool.page_size as i32,
                                            sm_scale,
                                            self.ctx.stream.cu_stream(),
                                        )
                                        .result()?;
                                    }
                                }
                                KVFormat::FP8E4M3 | KVFormat::INT8 => {
                                    // Per-(token, kv_head) symmetric quant; both
                                    // formats share the split-KV varlen kernel.
                                    let ws = pool.quantized_attn_workspace()?;
                                    let max_kv_len = meta.max_kv_len();
                                    kv_quant::decode_attention_varlen_quantized(
                                        &self.ctx,
                                        q_prepped,
                                        q_indptr_ptr,
                                        pool.k_data_ptr(full_idx, &self.ctx.stream),
                                        pool.v_data_ptr(full_idx, &self.ctx.stream),
                                        Some(pool.k_scales_ptr(full_idx, &self.ctx.stream)),
                                        Some(pool.v_scales_ptr(full_idx, &self.ctx.stream)),
                                        kv_indptr_ptr,
                                        kv_indices_ptr,
                                        last_page_len_ptr,
                                        attn_out,
                                        q_heads,
                                        kv_heads,
                                        c.head_dim,
                                        pool.page_size,
                                        meta.batch,
                                        meta.total_q,
                                        max_kv_len,
                                        true, // causal
                                        pool.format,
                                        sm_scale,
                                        ws,
                                        pool.quantized_attn_workspace_bytes,
                                    )?;
                                }
                                other => anyhow::bail!(
                                    "Qwen35 full-attn paged attention: unsupported pool format {other:?}"
                                ),
                            }
                            Ok(())
                        },
                    )?;
                }
            }

            // The recall cycle runs once per prefill, so the D2H stays off every
            // other paged forward. The query is the mean of the last `m` prompt
            // tokens' post-RoPE queries.
            if let Some(dst) = layer0_query
                && full_idx == 0
                && rows > 1
            {
                let host: Vec<bf16> = self
                    .ctx
                    .stream
                    .clone_dtoh(&q_prepped.data)
                    .map_err(|e| anyhow!("recall layer0 q dtoh: {e}"))?;
                const RECALL_PREFILL_Q_TOKENS: usize = 16;
                let m = RECALL_PREFILL_Q_TOKENS.min(rows);
                let mut q = vec![0.0_f32; q_dim];
                for t in (rows - m)..rows {
                    let base = t * q_dim;
                    for (d, slot) in q.iter_mut().enumerate() {
                        *slot += host[base + d].to_f32();
                    }
                }
                let inv = 1.0_f32 / m as f32;
                for v in &mut q {
                    *v *= inv;
                }
                *dst = q;
            }
        }

        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g1) = attn_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q_full/attn_out valid on ctx.stream; gate iterates
            // rows * num_q_heads.
            crate::profile::profile_op(&self.ctx, "full_paged/gate", Some(full_idx), rows, || {
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    ffi::attention_gate_paged_hd256_cuda(
                        qf_ptr as *const ffi::Half,
                        o_ptr as *mut ffi::Half,
                        q_heads as i32,
                        rows as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            })?;
        }

        if let Some(cpx) = cp {
            // Reduce this slice's o_proj partial over the head-shard (attn_tp)
            // group, then gather the cp slices back into the full-chunk `out`.
            let local = fw.cp_out.get(&self.ctx, c.hidden_size, rows)?;
            crate::profile::profile_op(
                &self.ctx,
                "full_paged/o_proj",
                Some(full_idx),
                rows,
                || gemm_batch(&self.ctx, &attn.o_proj, attn_out, local),
            )?;
            crate::profile::profile_op(
                &self.ctx,
                "full_paged/allreduce",
                Some(full_idx),
                rows,
                || {
                    self.tp
                        .all_reduce_sum_over(self.tp.attn_tp(), &self.ctx, local)
                },
            )?;
            crate::profile::profile_op(
                &self.ctx,
                "full_paged/cp_row_gather",
                Some(full_idx),
                rows,
                || self.cp_all_gather_rows(cpx, local, &mut fw.cp_row_gather, out),
            )?;
        } else {
            crate::profile::profile_op(
                &self.ctx,
                "full_paged/o_proj",
                Some(full_idx),
                rows,
                || gemm_batch(&self.ctx, o_proj, attn_out, out),
            )?;
            crate::profile::profile_op(
                &self.ctx,
                "full_paged/allreduce",
                Some(full_idx),
                rows,
                || {
                    if b2 {
                        // B2: partials span attn_tp x cp == world (attn_dp=1).
                        self.tp.all_reduce_sum(&self.ctx, out)
                    } else {
                        self.tp.attn_all_reduce_sum(&self.ctx, out)
                    }
                },
            )?;
        }
        Ok(())
    }

    fn cp_all_gather_in_place(&self, gather: &mut HiddenStates, sect: usize) -> Result<()> {
        let cp_rank = self.tp.attn_cp_rank();
        let elem = std::mem::size_of::<ffi::Half>() as u64;
        let (g_ptr, _g) = gather.data.device_ptr_mut(&self.ctx.stream);
        // SAFETY: live `cp_size*sect` buffer; equal `sect` on every cp rank.
        unsafe {
            self.tp.attn_cp_all_gather_bf16_unfenced(
                &self.ctx,
                (g_ptr + (cp_rank * sect) as u64 * elem) as *const std::ffi::c_void,
                sect,
                g_ptr as *mut std::ffi::c_void,
            )
        }
    }

    /// 2D ring prefill: one ring-attention pass over the whole prompt. Each rank
    /// preps its own q-slice and KV slice into dense head-major buffers, rotates
    /// the KV slice around the cp ring, scatters its owned pages (block-cyclic)
    /// into the sharded pool as the blocks pass through, and finalizes the
    /// flash-2 accumulator into the gate's row-major bf16 `attn_out`.
    #[allow(clippy::too_many_arguments)]
    fn ring_prefill_full_attention(
        &self,
        attn: &FullAttn,
        cp: &Qwen35CpPrefill,
        full_idx: usize,
        pool: &PagedKVPool,
        ring: &mut RingPrefillScratch,
        q_full: &HiddenStates,
        k_batch: &HiddenStates,
        v_batch: &HiddenStates,
        attn_out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let head_dim = c.head_dim;
        let (cp_size, cp_rank) = (self.tp.attn_cp_size(), self.tp.attn_cp_rank());
        let (q_heads, kv_heads) = (self.local_q_heads, self.local_kv_heads);
        let rows = cp.slices[cp_rank].1;
        let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
        let stride_page = pool.kv_dim * pool.page_size;
        let pad = cp.pad;

        // Dense prep: q/k-norm + partial RoPE into head-major buffers (no pool
        // write — the scatter below owns the pool write).
        let q_ring = ring.q.get(&self.ctx, q_heads * rows * head_dim)?;
        let k0 = ring.k0.get(&self.ctx, kv_heads * pad * head_dim)?;
        let v0 = ring.v0.get(&self.ctx, kv_heads * pad * head_dim)?;
        let k1 = ring.k1.get(&self.ctx, kv_heads * pad * head_dim)?;
        let v1 = ring.v1.get(&self.ctx, kv_heads * pad * head_dim)?;
        {
            let (qf_ptr, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (k_ptr, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (qo_ptr, _g7) = q_ring.device_ptr_mut(&self.ctx.stream);
            let (ko_ptr, _g8) = k0.device_ptr_mut(&self.ctx.stream);
            let (vo_ptr, _g9) = v0.device_ptr_mut(&self.ctx.stream);
            crate::profile::profile_op(
                &self.ctx,
                "full_paged/ring_prep",
                Some(full_idx),
                rows,
                || {
                    // SAFETY: buffers live on ctx.stream, sized to the dims passed.
                    unsafe {
                        ffi::ring_prefill_dense_prep_hd256_cuda(
                            qf_ptr as *const ffi::Half,
                            k_ptr as *const ffi::Half,
                            v_ptr as *const ffi::Half,
                            qn_ptr as *const ffi::Half,
                            kn_ptr as *const ffi::Half,
                            cos_ptr as *const ffi::Half,
                            sin_ptr as *const ffi::Half,
                            qo_ptr as *mut ffi::Half,
                            ko_ptr as *mut ffi::Half,
                            vo_ptr as *mut ffi::Half,
                            q_heads as i32,
                            kv_heads as i32,
                            rows as i32,
                            cp.q_pos[0] as i32,
                            c.rotary_dim as i32,
                            c.rms_norm_eps,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                    Ok(())
                },
            )?;
        }

        // Accumulator init: m=-inf, l=0, o=0 (flash-2 online softmax).
        let acc_rows = q_heads * rows;
        let acc_o_len = acc_rows * head_dim;
        let m_a = ring.m_a.get(&self.ctx, acc_rows)?;
        let l_a = ring.l_a.get_zeroed(&self.ctx, acc_rows)?;
        let o_a = ring.o_a.get_zeroed(&self.ctx, acc_o_len)?;
        {
            let neg_inf = vec![f32::NEG_INFINITY; acc_rows];
            self.ctx
                .stream
                .memcpy_htod(&neg_inf, m_a)
                .map_err(|e| anyhow!("ring m -inf upload: {e}"))?;
        }

        let dims0 = cuda_kernels::ring_attention::RingBlockDims {
            num_q_tiles: q_heads,
            num_q_heads: q_heads,
            num_kv_heads: kv_heads,
            head_dim,
            q_rows: rows,
            blk_len: cp.k_pos[cp_rank].len(),
            sm_scale,
        };
        let fa3 = cuda_kernels::ring_attention::ring_fa3_route(
            &self.ctx.stream,
            dims0,
            &cp.q_pos,
            &cp.k_pos[cp_rank],
        );

        if fa3 {
            // FA3 route: thread internally-allocated accumulator copies.
            let mut fa3_acc: Option<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)> = None;
            let mut cur_owner = cp_rank;
            let mut cur_pair = 0usize;
            for hop in 0..cp_size {
                if hop > 0 {
                    // The current pair was recv'd at the prior hop's start; wait
                    // for that transfer before this hop's compute reads it.
                    self.ctx.compute_waits_for_comm()?;
                }
                let next_owner = (cur_owner + cp_size - 1) % cp_size;
                if hop + 1 < cp_size {
                    // Group recv+send so NCCL submits both directions together —
                    // the host never blocks in ncclSend waiting for a peer recv.
                    // The send reads the current pair (stable: compute only reads
                    // it, scatter writes to the pool), so posting it before the
                    // compute is safe. The recv lands in the idle pair and
                    // overlaps with this hop's compute.
                    self.ctx.comm_waits_for_compute()?;
                    self.tp.attn_cp_group_start()?;
                    let recv_len = cp.slices[next_owner].1;
                    if cur_pair == 0 {
                        self.ring_prefill_post_recv(
                            &mut *k1, &mut *v1, recv_len, kv_heads, head_dim, cp_size,
                        )?;
                    } else {
                        self.ring_prefill_post_recv(
                            &mut *k0, &mut *v0, recv_len, kv_heads, head_dim, cp_size,
                        )?;
                    }
                    let send_len = cp.slices[cur_owner].1;
                    if cur_pair == 0 {
                        self.ring_prefill_post_send(
                            &*k0, &*v0, send_len, kv_heads, head_dim, cp_size,
                        )?;
                    } else {
                        self.ring_prefill_post_send(
                            &*k1, &*v1, send_len, kv_heads, head_dim, cp_size,
                        )?;
                    }
                    self.tp.attn_cp_group_end()?;
                }
                let blk_len = cp.k_pos[cur_owner].len();
                let dims = cuda_kernels::ring_attention::RingBlockDims { blk_len, ..dims0 };
                let (k_ref, v_ref): (&CudaSlice<u16>, &CudaSlice<u16>) = if cur_pair == 0 {
                    (&*k0, &*v0)
                } else {
                    (&*k1, &*v1)
                };
                let (m_in, l_in, o_in): (&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>) =
                    if hop == 0 {
                        (&*m_a, &*l_a, &*o_a)
                    } else {
                        let acc = fa3_acc.as_ref().unwrap();
                        (&acc.0, &acc.1, &acc.2)
                    };
                let (m_out, l_out, o_out) = cuda_kernels::ring_attention::ring_block_fwd_merge_fa3(
                    &self.ctx.stream,
                    q_ring,
                    k_ref,
                    v_ref,
                    m_in,
                    l_in,
                    o_in,
                    &cp.q_pos,
                    &cp.k_pos[cur_owner],
                    dims,
                )?;
                fa3_acc = Some((m_out, l_out, o_out));
                self.ring_prefill_scatter(
                    cp,
                    full_idx,
                    pool,
                    k_ref,
                    v_ref,
                    cur_owner,
                    stride_page,
                    kv_heads,
                )?;
                if hop + 1 < cp_size {
                    cur_owner = next_owner;
                    cur_pair = 1 - cur_pair;
                }
            }
            let acc = fa3_acc.as_ref().unwrap();
            self.ring_prefill_finalize(&acc.1, &acc.2, attn_out, q_heads, rows, full_idx)?;
        } else {
            // Scalar route: A/B ping-pong (the merge is functional, in != out).
            let m_b = ring.m_b.get(&self.ctx, acc_rows)?;
            let l_b = ring.l_b.get(&self.ctx, acc_rows)?;
            let o_b = ring.o_b.get(&self.ctx, acc_o_len)?;
            let mut cur_owner = cp_rank;
            let mut cur_pair = 0usize;
            let mut out_in_a = false;
            for hop in 0..cp_size {
                if hop > 0 {
                    self.ctx.compute_waits_for_comm()?;
                }
                let next_owner = (cur_owner + cp_size - 1) % cp_size;
                if hop + 1 < cp_size {
                    self.ctx.comm_waits_for_compute()?;
                    self.tp.attn_cp_group_start()?;
                    let recv_len = cp.slices[next_owner].1;
                    if cur_pair == 0 {
                        self.ring_prefill_post_recv(
                            &mut *k1, &mut *v1, recv_len, kv_heads, head_dim, cp_size,
                        )?;
                    } else {
                        self.ring_prefill_post_recv(
                            &mut *k0, &mut *v0, recv_len, kv_heads, head_dim, cp_size,
                        )?;
                    }
                    let send_len = cp.slices[cur_owner].1;
                    if cur_pair == 0 {
                        self.ring_prefill_post_send(
                            &*k0, &*v0, send_len, kv_heads, head_dim, cp_size,
                        )?;
                    } else {
                        self.ring_prefill_post_send(
                            &*k1, &*v1, send_len, kv_heads, head_dim, cp_size,
                        )?;
                    }
                    self.tp.attn_cp_group_end()?;
                }
                let blk_len = cp.k_pos[cur_owner].len();
                let (k_ref, v_ref): (&CudaSlice<u16>, &CudaSlice<u16>) = if cur_pair == 0 {
                    (&*k0, &*v0)
                } else {
                    (&*k1, &*v1)
                };
                let (mi, li, oi, mo, lo, oo) = if hop % 2 == 0 {
                    (&*m_a, &*l_a, &*o_a, &mut *m_b, &mut *l_b, &mut *o_b)
                } else {
                    (&*m_b, &*l_b, &*o_b, &mut *m_a, &mut *l_a, &mut *o_a)
                };
                // Guards drop before the scatter reborrows the pair.
                {
                    let (q_ptr, _g0) = q_ring.device_ptr(&self.ctx.stream);
                    let (k_ptr, _g1) = k_ref.device_ptr(&self.ctx.stream);
                    let (v_ptr, _g2) = v_ref.device_ptr(&self.ctx.stream);
                    let (mi_ptr, _g3) = mi.device_ptr(&self.ctx.stream);
                    let (li_ptr, _g4) = li.device_ptr(&self.ctx.stream);
                    let (oi_ptr, _g5) = oi.device_ptr(&self.ctx.stream);
                    let (mo_ptr, _g6) = mo.device_ptr_mut(&self.ctx.stream);
                    let (lo_ptr, _g7) = lo.device_ptr_mut(&self.ctx.stream);
                    let (oo_ptr, _g8) = oo.device_ptr_mut(&self.ctx.stream);
                    let (qpos_ptr, _g9) = cp.q_pos_f32.device_ptr(&self.ctx.stream);
                    let (kpos_ptr, _g10) = cp.k_pos_f32[cur_owner].device_ptr(&self.ctx.stream);
                    crate::profile::profile_op(
                        &self.ctx,
                        "full_paged/ring_merge",
                        Some(full_idx),
                        rows,
                        || {
                            // SAFETY: buffers live on ctx.stream, sized to the dims passed.
                            unsafe {
                                ffi::ring_block_attention_fwd_merge_cuda(
                                    q_ptr as *const ffi::Half,
                                    k_ptr as *const ffi::Half,
                                    v_ptr as *const ffi::Half,
                                    mi_ptr as *const f32,
                                    li_ptr as *const f32,
                                    oi_ptr as *const f32,
                                    mo_ptr as *mut f32,
                                    lo_ptr as *mut f32,
                                    oo_ptr as *mut f32,
                                    qpos_ptr as *const f32,
                                    kpos_ptr as *const f32,
                                    q_heads as i32,
                                    q_heads as i32,
                                    kv_heads as i32,
                                    head_dim as i32,
                                    rows as i32,
                                    blk_len as i32,
                                    sm_scale,
                                    self.ctx.stream.cu_stream(),
                                )
                                .result()?;
                            }
                            Ok(())
                        },
                    )?;
                }
                out_in_a = hop % 2 == 1;
                self.ring_prefill_scatter(
                    cp,
                    full_idx,
                    pool,
                    k_ref,
                    v_ref,
                    cur_owner,
                    stride_page,
                    kv_heads,
                )?;
                if hop + 1 < cp_size {
                    cur_owner = next_owner;
                    cur_pair = 1 - cur_pair;
                }
            }
            let (l_fin, o_fin): (&CudaSlice<f32>, &CudaSlice<f32>) = if out_in_a {
                (&*l_a, &*o_a)
            } else {
                (&*l_b, &*o_b)
            };
            self.ring_prefill_finalize(l_fin, o_fin, attn_out, q_heads, rows, full_idx)?;
        }
        Ok(())
    }

    /// Scatter the current ring block's tokens whose global page is owned by
    /// this shard into the sharded HND pool (block-cyclic).
    #[allow(clippy::too_many_arguments)]
    fn ring_prefill_scatter(
        &self,
        cp: &Qwen35CpPrefill,
        full_idx: usize,
        pool: &PagedKVPool,
        k_dense: &CudaSlice<u16>,
        v_dense: &CudaSlice<u16>,
        owner: usize,
        stride_page: usize,
        kv_heads: usize,
    ) -> Result<()> {
        let cp_size = self.tp.attn_cp_size();
        let cp_rank = self.tp.attn_cp_rank();
        let blk_start = cp.k_pos[owner][0] as i32;
        let blk_len = cp.k_pos[owner].len();
        let (k_ptr, _g0) = k_dense.device_ptr(&self.ctx.stream);
        let (v_ptr, _g1) = v_dense.device_ptr(&self.ctx.stream);
        let (pt_ptr, _g2) = cp.kv_indices.device_ptr(&self.ctx.stream);
        let k_pool = pool.k_ptr(full_idx, &self.ctx.stream);
        let v_pool = pool.v_ptr(full_idx, &self.ctx.stream);
        // SAFETY: dense buffers hold the block's prepped K/V (head-major,
        // stride = blk_len); kv_indices is this shard's local page table.
        unsafe {
            ffi::ring_prefill_scatter_sharded_hd256_cuda(
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                pt_ptr as *const i32,
                cp.kv_indices.len() as i32,
                pool.page_size as i32,
                kv_heads as i32,
                blk_start,
                blk_len as i32,
                cp_rank as i32,
                cp_size as i32,
                stride_page as i32,
                k_pool as *mut ffi::Half,
                v_pool as *mut ffi::Half,
                self.ctx.stream.cu_stream(),
            )
            .result()?;
        }
        Ok(())
    }

    /// Post two unfenced cp recvs (k, v) into the idle pair from the upstream
    /// peer. Issued before this hop's compute so the transfer overlaps; the
    /// caller must run `compute_waits_for_comm` before the next hop's compute
    /// reads this pair.
    fn ring_prefill_post_recv(
        &self,
        recv_k: &mut CudaSlice<u16>,
        recv_v: &mut CudaSlice<u16>,
        recv_len: usize,
        kv_heads: usize,
        head_dim: usize,
        cp_size: usize,
    ) -> Result<()> {
        let cp_rank = self.tp.attn_cp_rank();
        let recv_peer = (cp_rank + cp_size - 1) % cp_size;
        let count = kv_heads * recv_len * head_dim;
        let (rk_ptr, _g0) = recv_k.device_ptr_mut(&self.ctx.stream);
        let (rv_ptr, _g1) = recv_v.device_ptr_mut(&self.ctx.stream);
        // SAFETY: recv buffers are pad-sized (>= recv_len); peers post matching
        // sends in (k, v) order.
        unsafe {
            self.tp.attn_cp_recv_unfenced(
                &self.ctx,
                rk_ptr as *mut std::ffi::c_void,
                count,
                cuda_kernels::collective::DType::BF16,
                recv_peer,
            )?;
            self.tp.attn_cp_recv_unfenced(
                &self.ctx,
                rv_ptr as *mut std::ffi::c_void,
                count,
                cuda_kernels::collective::DType::BF16,
                recv_peer,
            )?;
        }
        Ok(())
    }

    /// Post two unfenced cp sends (k, v) of the current pair to the downstream
    /// peer. Caller must run `comm_waits_for_compute` first so the compute that
    /// reads this pair has finished.
    fn ring_prefill_post_send(
        &self,
        send_k: &CudaSlice<u16>,
        send_v: &CudaSlice<u16>,
        send_len: usize,
        kv_heads: usize,
        head_dim: usize,
        cp_size: usize,
    ) -> Result<()> {
        let cp_rank = self.tp.attn_cp_rank();
        let send_peer = (cp_rank + 1) % cp_size;
        let count = kv_heads * send_len * head_dim;
        let (sk_ptr, _g0) = send_k.device_ptr(&self.ctx.stream);
        let (sv_ptr, _g1) = send_v.device_ptr(&self.ctx.stream);
        // SAFETY: send buffers hold `send_len` live rows; peers post matching
        // recvs in (k, v) order.
        unsafe {
            self.tp.attn_cp_send_unfenced(
                &self.ctx,
                sk_ptr as *const std::ffi::c_void,
                count,
                cuda_kernels::collective::DType::BF16,
                send_peer,
            )?;
            self.tp.attn_cp_send_unfenced(
                &self.ctx,
                sv_ptr as *const std::ffi::c_void,
                count,
                cuda_kernels::collective::DType::BF16,
                send_peer,
            )?;
        }
        Ok(())
    }

    /// Finalize the ring accumulator into the gate's row-major bf16 `attn_out`.
    fn ring_prefill_finalize(
        &self,
        acc_l: &CudaSlice<f32>,
        acc_o: &CudaSlice<f32>,
        attn_out: &mut HiddenStates,
        q_heads: usize,
        rows: usize,
        full_idx: usize,
    ) -> Result<()> {
        let (l_ptr, _g0) = acc_l.device_ptr(&self.ctx.stream);
        let (o_ptr, _g1) = acc_o.device_ptr(&self.ctx.stream);
        let (out_ptr, _g2) = attn_out.data.device_ptr_mut(&self.ctx.stream);
        crate::profile::profile_op(
            &self.ctx,
            "full_paged/ring_finalize",
            Some(full_idx),
            rows,
            || {
                // SAFETY: acc buffers are [q_heads*rows] / [q_heads*rows*d];
                // attn_out is [rows, q_heads*d] bf16.
                unsafe {
                    ffi::ring_prefill_finalize_bf16_hd256_cuda(
                        l_ptr as *const f32,
                        o_ptr as *const f32,
                        out_ptr as *mut ffi::Half,
                        q_heads as i32,
                        rows as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    /// All-gather cp slices of `[dim, slice_rows]` `local` back into the
    /// full-chunk `out` rows (in-place NCCL gather over the padded buffer,
    /// then per-slice D2D copies to the chunk-relative row offsets).
    fn cp_all_gather_rows(
        &self,
        cp: &Qwen35CpPrefill,
        local: &HiddenStates,
        cp_row_gather: &mut HiddenSlot,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let dim = local.hidden_dim;
        ensure!(
            out.hidden_dim == dim,
            "CP row gather: local dim {dim} != out dim {}",
            out.hidden_dim
        );
        let (cp_size, cp_rank) = (self.tp.attn_cp_size(), self.tp.attn_cp_rank());
        let pad = cp.pad;
        let sect = pad * dim;
        let my_len = cp.slices[cp_rank].1;
        ensure!(
            local.seq_len == my_len,
            "CP row gather: slice rows mismatch"
        );
        let gather = cp_row_gather.get(&self.ctx, dim, cp_size * pad)?;
        self.ctx
            .stream
            .memcpy_dtod(
                &local.data.slice(0..my_len * dim),
                &mut gather
                    .data
                    .slice_mut(cp_rank * sect..cp_rank * sect + my_len * dim),
            )
            .map_err(|e| anyhow!("CP row-gather stage failed: {e}"))?;
        self.ctx.comm_waits_for_compute()?;
        self.cp_all_gather_in_place(gather, sect)?;
        self.ctx.compute_waits_for_comm()?;
        for (peer, &(off, len)) in cp.slices.iter().enumerate() {
            self.ctx
                .stream
                .memcpy_dtod(
                    &gather.data.slice(peer * sect..peer * sect + len * dim),
                    &mut out.data.slice_mut(off * dim..(off + len) * dim),
                )
                .map_err(|e| anyhow!("CP row-gather scatter failed: {e}"))?;
        }
        Ok(())
    }

    /// Gated-delta-rule linear attention into `out` (`[hidden, rows]`, beta=0
    /// out-proj GEMM). The conv ring + recurrent state advance in place and
    /// carry across prefill/decode.
    ///
    /// `rows = normed.seq_len` is the FLAT column count: every weight-heavy
    /// step runs once over all of them, only [`LinearCore`] is per-slot.
    pub(crate) fn linear_attention(
        &self,
        attn: &LinearAttn,
        normed: &HiddenStates,
        core: LinearCore<'_, '_>,
        linear_idx: usize,
        lw: &mut LinearAttnScratch,
        cp: Option<&Qwen35CpPrefill>,
        cp_decode: Option<&Qwen35CpDecode>,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        // B2: cp ranks shard the v heads via the decode subset weights. Only
        // the single-row decode core is sharded; anything else (Tables,
        // multi-row) runs replicated for this step.
        let b2 = cp_decode.is_some()
            && cp.is_none()
            && matches!(
                &core,
                LinearCore::Rows(rs) if rs.len() == 1 && rs[0].len == 1 && rs[0].capture.is_none()
            );
        if cp_decode.is_some() && !b2 {
            log::warn!(
                "B2 CP decode: linear layer {linear_idx} core is not single-row decode; running replicated"
            );
        }
        let (in_qkvz, in_ba, out_proj): (&DeviceMatrix, &DeviceMatrix, &DeviceMatrix) = if b2 {
            let d = attn
                .decode
                .as_ref()
                .expect("B2 decode weights present when cp_decode active");
            (&d.in_proj_qkvz, &d.in_proj_ba, &d.out_proj)
        } else {
            (&attn.in_proj_qkvz, &attn.in_proj_ba, &attn.out_proj)
        };
        // LOCAL per-rank widths: conv channels, recurrent state, and launches
        // all follow this rank's linear k/v head shard (B2: the 1/cp subset).
        let (qkv_dim, z_dim) = if b2 {
            let lk = self.decode_linear_k_heads();
            let lv = self.decode_linear_v_heads();
            (
                2 * lk * c.linear_key_head_dim + lv * c.linear_value_head_dim,
                lv * c.linear_value_head_dim,
            )
        } else {
            (self.local_linear_qkv_dim(), self.local_linear_z_dim())
        };
        let b_dim = in_ba.rows / 2;
        let a_dim = b_dim;

        let LinearAttnScratch {
            capture_copy,
            qkvz,
            qkv,
            z,
            ba,
            b_proj,
            a_proj,
            qkv_conv,
            gdr_out,
            normed_out,
            fq_q,
            fq_k,
            fq_v,
            fq_a,
            fq_g,
            fq_g_cumsum,
            fq_beta,
            batch_ptrs,
            batch_len,
            batch_host,
            batch_len_host,
            cp_in,
            cp_out,
            cp_row_gather,
        } = lw;
        // CP prefill: weight-heavy steps run over this rank's q-slice only;
        // the GDN state chain is relayed across cp ranks below.
        let normed: &HiddenStates = match cp {
            Some(cpx) => {
                let (off, len) = cpx.slices[self.tp.attn_cp_rank()];
                let h = c.hidden_size;
                let buf = cp_in.get(&self.ctx, h, len)?;
                self.ctx
                    .stream
                    .memcpy_dtod(
                        &normed.data.slice(off * h..(off + len) * h),
                        &mut buf.data.slice_mut(0..len * h),
                    )
                    .map_err(|e| anyhow!("CP linear q-slice copy failed: {e}"))?;
                buf
            }
            None => normed,
        };
        let rows = normed.seq_len;
        let qkvz = qkvz.get(&self.ctx, qkv_dim + z_dim, rows)?;
        let qkv = qkv.get(&self.ctx, qkv_dim, rows)?;
        let z = z.get(&self.ctx, z_dim, rows)?;
        let ba = ba.get(&self.ctx, b_dim + a_dim, rows)?;
        let b_proj = b_proj.get(&self.ctx, b_dim, rows)?;
        let a_proj = a_proj.get(&self.ctx, a_dim, rows)?;
        crate::profile::profile_op(&self.ctx, "linear/in_proj", Some(linear_idx), rows, || {
            gemm_batch(&self.ctx, in_qkvz, normed, qkvz)?;
            split2(&self.ctx, qkvz, qkv, z)?;
            gemm_batch(&self.ctx, in_ba, normed, ba)?;
            split2(&self.ctx, ba, b_proj, a_proj)?;
            Ok(())
        })?;

        let qkv_conv = qkv_conv.get(&self.ctx, qkv_dim, rows)?;
        let gdr_out = gdr_out.get(&self.ctx, z_dim, rows)?;
        match core {
            LinearCore::Rows(rs) if cp.is_some() => {
                ensure!(
                    rs.len() == 1 && rs[0].capture.is_none(),
                    "CP prefill linear attention is single-row and capture-free"
                );
                let (cp_size, cp_rank) = (self.tp.attn_cp_size(), self.tp.attn_cp_rank());
                let slot = &mut *rs[0].slot;
                ensure!(
                    linear_idx < slot.gdr_states.len() && linear_idx < slot.conv_states.len(),
                    "CP prefill: slot recurrent state missing for linear layer {linear_idx}"
                );
                let gdr_len = slot.gdr_states[linear_idx].len();
                let conv_len = slot.conv_states[linear_idx].len;
                // Chunk-order state chain: slice r starts from slice r-1's
                // post-slice recurrent + conv-tail state (rank 0 starts from
                // the slot's own state — chunk continuity).
                use cuda_kernels::collective::DType;
                if cp_rank > 0 {
                    let (g_ptr, _g0) = slot.gdr_states[linear_idx].device_ptr_mut(&self.ctx.stream);
                    let (c_ptr, _g1) = slot.conv_states[linear_idx]
                        .data
                        .device_ptr_mut(&self.ctx.stream);
                    self.ctx.comm_waits_for_compute()?;
                    self.tp.attn_cp_group_start()?;
                    // SAFETY: live per-slot state buffers; the previous cp rank
                    // posts the matching sends in the same (gdr, conv) order.
                    unsafe {
                        self.tp.attn_cp_recv_unfenced(
                            &self.ctx,
                            g_ptr as *mut std::ffi::c_void,
                            gdr_len,
                            DType::F32,
                            cp_rank - 1,
                        )?;
                        self.tp.attn_cp_recv_unfenced(
                            &self.ctx,
                            c_ptr as *mut std::ffi::c_void,
                            conv_len,
                            DType::BF16,
                            cp_rank - 1,
                        )?;
                    }
                    self.tp.attn_cp_group_end()?;
                    self.ctx.compute_waits_for_comm()?;
                }
                self.advance_linear_conv_gdr(
                    attn,
                    &qkv.data.slice(0..rows * qkv_dim),
                    &b_proj.data.slice(0..rows * b_dim),
                    &a_proj.data.slice(0..rows * a_dim),
                    slot,
                    linear_idx,
                    rows,
                    &mut qkv_conv.data.slice_mut(0..rows * qkv_dim),
                    &mut gdr_out.data.slice_mut(0..rows * z_dim),
                    fq_q,
                    fq_k,
                    fq_v,
                    fq_a,
                    fq_g,
                    fq_g_cumsum,
                    fq_beta,
                    None,
                )?;
                {
                    let (g_ptr, _g0) = slot.gdr_states[linear_idx].device_ptr_mut(&self.ctx.stream);
                    let (c_ptr, _g1) = slot.conv_states[linear_idx]
                        .data
                        .device_ptr_mut(&self.ctx.stream);
                    self.ctx.comm_waits_for_compute()?;
                    // Group the send (if any) with the broadcast so NCCL
                    // submits both directions together — the host never blocks
                    // in ncclSend waiting for a peer recv that hasn't been
                    // posted.
                    self.tp.attn_cp_group_start()?;
                    if cp_rank + 1 < cp_size {
                        // SAFETY: same buffers, matching recvs on the next rank.
                        unsafe {
                            self.tp.attn_cp_send_unfenced(
                                &self.ctx,
                                g_ptr as *const std::ffi::c_void,
                                gdr_len,
                                DType::F32,
                                cp_rank + 1,
                            )?;
                            self.tp.attn_cp_send_unfenced(
                                &self.ctx,
                                c_ptr as *const std::ffi::c_void,
                                conv_len,
                                DType::BF16,
                                cp_rank + 1,
                            )?;
                        }
                    }
                    // The LAST slice's post-state is the true end-of-chunk state;
                    // broadcast it so every rank's slot agrees before the next
                    // chunk / decode.
                    // SAFETY: live state buffers, same count/root on every cp rank.
                    unsafe {
                        self.tp.attn_cp_broadcast_unfenced(
                            &self.ctx,
                            g_ptr as *mut std::ffi::c_void,
                            gdr_len,
                            DType::F32,
                            cp_size - 1,
                        )?;
                        self.tp.attn_cp_broadcast_unfenced(
                            &self.ctx,
                            c_ptr as *mut std::ffi::c_void,
                            conv_len,
                            DType::BF16,
                            cp_size - 1,
                        )?;
                    }
                    self.tp.attn_cp_group_end()?;
                    self.ctx.compute_waits_for_comm()?;
                }
            }
            LinearCore::Rows(rs) => {
                let total: usize = rs.iter().map(|r| r.len).sum();
                ensure!(
                    total == rows,
                    "linear rows total {total} != {rows} staged columns"
                );
                // Each row's columns land at ITS capture offset 0, so a
                // partial-accept replay re-runs only that slot's prefix.
                if rs.iter().any(|r| r.capture.is_some()) {
                    let (mut dst, mut src, mut sz) = (Vec::new(), Vec::new(), Vec::new());
                    let mut off = 0usize;
                    for r in rs.iter_mut() {
                        let len = r.len;
                        let at = off;
                        off += len;
                        let Some(cap) = r.capture.as_deref_mut() else {
                            continue;
                        };
                        ensure!(
                            linear_idx < cap.qkv.len() && len <= cap.rows,
                            "spec capture is {} layers x {} rows, cannot hold layer \
                             {linear_idx} x {len} rows",
                            cap.qkv.len(),
                            cap.rows
                        );
                        for (s_ptr, w, d) in [
                            (&qkv.data, qkv_dim, &mut cap.qkv[linear_idx]),
                            (&b_proj.data, b_dim, &mut cap.b_proj[linear_idx]),
                            (&a_proj.data, a_dim, &mut cap.a_proj[linear_idx]),
                        ] {
                            let elem = std::mem::size_of::<bf16>();
                            dst.push(d.data.device_ptr_mut(&self.ctx.stream).0);
                            src.push(s_ptr.device_ptr(&self.ctx.stream).0 + (at * w * elem) as u64);
                            sz.push(len * w * elem);
                        }
                    }
                    self.batched_copy(capture_copy, &dst, &src, &sz)?;
                }
                // B2: allocate the decode recurrent pair (first B2 step for
                // this slot) and scatter this layer's head subset into it.
                if b2 {
                    let cp = cp_decode.unwrap();
                    let (num_linear, gdr_len_d, conv_len_d) =
                        self.recurrent_dims_decode(cp.cp_size);
                    let slot = &mut *rs[0].slot;
                    slot.ensure_decode_recurrent(&self.ctx, num_linear, gdr_len_d, conv_len_d)?;
                    slot.scatter_decode_state(
                        &self.ctx,
                        linear_idx,
                        self.local_linear_k_heads,
                        self.decode_linear_k_heads(),
                        self.decode_linear_v_heads(),
                        cp.cp_rank,
                        c.linear_key_head_dim,
                        c.linear_value_head_dim,
                        c.linear_conv_kernel_dim,
                    )?;
                }
                let decode_geom = if b2 {
                    let cp = cp_decode.unwrap();
                    let (v_off, _) = cp.subset(self.local_linear_v_heads);
                    Some(LinearDecodeGeom {
                        qkv_dim,
                        z_dim,
                        k_heads: self.decode_linear_k_heads(),
                        v_heads: self.decode_linear_v_heads(),
                        v_off,
                    })
                } else {
                    None
                };
                // Uniform short rows pack identically to the varlen kernels'
                // `s * len` stride, so the whole batch is one conv + one GDR
                // launch instead of B of each.
                let uniform = rs.first().map(|r| r.len).filter(|len| {
                    (1..=LINEAR_BATCH_MAX_LEN).contains(len) && rs.iter().all(|r| r.len == *len)
                });
                if let (Some(len), true) = (uniform, rs.len() > 1) {
                    self.advance_linear_conv_gdr_batched(
                        attn,
                        rs,
                        linear_idx,
                        len,
                        qkv,
                        b_proj,
                        a_proj,
                        qkv_conv,
                        gdr_out,
                        batch_ptrs,
                        batch_len,
                        batch_host,
                        batch_len_host,
                    )?;
                } else {
                    let mut off = 0usize;
                    for r in rs.iter_mut() {
                        self.advance_linear_conv_gdr(
                            attn,
                            &qkv.data.slice(off * qkv_dim..(off + r.len) * qkv_dim),
                            &b_proj.data.slice(off * b_dim..(off + r.len) * b_dim),
                            &a_proj.data.slice(off * a_dim..(off + r.len) * a_dim),
                            r.slot,
                            linear_idx,
                            r.len,
                            &mut qkv_conv
                                .data
                                .slice_mut(off * qkv_dim..(off + r.len) * qkv_dim),
                            &mut gdr_out.data.slice_mut(off * z_dim..(off + r.len) * z_dim),
                            fq_q,
                            fq_k,
                            fq_v,
                            fq_a,
                            fq_g,
                            fq_g_cumsum,
                            fq_beta,
                            decode_geom,
                        )?;
                        off += r.len;
                    }
                }
            }
            LinearCore::Tables { conv, gdr } => {
                let (x_ptr, _g0) = qkv.data.device_ptr(&self.ctx.stream);
                let (w_ptr, _g1) = attn.conv1d_weight.data.device_ptr(&self.ctx.stream);
                let (b_ptr, _g2) = b_proj.data.device_ptr(&self.ctx.stream);
                let (a_ptr, _g3) = a_proj.data.device_ptr(&self.ctx.stream);
                let (dt_ptr, _g4) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
                let (alog_ptr, _g5) = attn.a_log.device_ptr(&self.ctx.stream);
                let (conv_tbl, _g6) = conv.device_ptr(&self.ctx.stream);
                let (gdr_tbl, _g7) = gdr.device_ptr(&self.ctx.stream);
                let (cv_ptr, _g8) = qkv_conv.data.device_ptr_mut(&self.ctx.stream);
                let (o_ptr, _g9) = gdr_out.data.device_ptr_mut(&self.ctx.stream);
                crate::profile::profile_op(
                    &self.ctx,
                    "linear/conv1d",
                    Some(linear_idx),
                    rows,
                    || {
                        // SAFETY: x/weight/out are live `[B, C]`/`[C*K]` buffers on
                        // ctx.stream; the table's first B entries point at live
                        // `[C, K-1]` conv rings.
                        unsafe {
                            ffi::conv1d_decode_batch_cuda(
                                x_ptr as *const ffi::Half,
                                w_ptr as *const ffi::Half,
                                conv_tbl as *mut *mut ffi::Half,
                                cv_ptr as *mut ffi::Half,
                                qkv_dim as i32,
                                c.linear_conv_kernel_dim as i32,
                                rows as i32,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                        Ok(())
                    },
                )?;
                crate::profile::profile_op(
                    &self.ctx,
                    "linear/gdr_recurrent",
                    Some(linear_idx),
                    rows,
                    || {
                        // SAFETY: all buffers live on ctx.stream; the table's first
                        // B entries point at live `[Vh, Kd, Vd]` f32 states.
                        unsafe {
                            ffi::gdr_decode_batch_cuda(
                                cv_ptr as *const ffi::Half,
                                b_ptr as *const ffi::Half,
                                a_ptr as *const ffi::Half,
                                dt_ptr as *const ffi::Half,
                                alog_ptr as *const f32,
                                gdr_tbl as *mut *mut f32,
                                o_ptr as *mut ffi::Half,
                                self.local_linear_k_heads as i32,
                                self.local_linear_v_heads as i32,
                                c.linear_key_head_dim as i32,
                                c.linear_value_head_dim as i32,
                                rows as i32,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                        Ok(())
                    },
                )?;
            }
        }

        let normed_out = normed_out.get(&self.ctx, z_dim, rows)?;
        {
            let (x_ptr, _g0) = gdr_out.data.device_ptr(&self.ctx.stream);
            let (w_ptr, _g1) = attn.norm_weight.device_ptr(&self.ctx.stream);
            let (gate_ptr, _g2) = z.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _g3) = normed_out.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: gdr_out/norm/z/out valid on ctx.stream; per-head layout from
            // config.
            // One block per `[val_dim]` slice, so the grid must cover all
            // rows*Vh (token, head) slices; `weight` is a per-[Vd] broadcast.
            crate::profile::profile_op(&self.ctx, "linear/norm", Some(linear_idx), rows, || {
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    ffi::rms_norm_gated_cuda(
                        x_ptr as *const ffi::Half,
                        w_ptr as *const f32,
                        gate_ptr as *const ffi::Half,
                        o_ptr as *mut ffi::Half,
                        (z_dim / c.linear_value_head_dim * rows) as i32,
                        c.linear_value_head_dim as i32,
                        c.rms_norm_eps,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            })?;
        }

        if let Some(cpx) = cp {
            let local = cp_out.get(&self.ctx, c.hidden_size, rows)?;
            crate::profile::profile_op(
                &self.ctx,
                "linear/out_proj",
                Some(linear_idx),
                rows,
                || gemm_batch(&self.ctx, &attn.out_proj, normed_out, local),
            )?;
            crate::profile::profile_op(
                &self.ctx,
                "linear/allreduce",
                Some(linear_idx),
                rows,
                || {
                    self.tp
                        .all_reduce_sum_over(self.tp.attn_tp(), &self.ctx, local)
                },
            )?;
            crate::profile::profile_op(
                &self.ctx,
                "linear/cp_row_gather",
                Some(linear_idx),
                rows,
                || self.cp_all_gather_rows(cpx, local, cp_row_gather, out),
            )?;
        } else {
            crate::profile::profile_op(
                &self.ctx,
                "linear/out_proj",
                Some(linear_idx),
                rows,
                || gemm_batch(&self.ctx, out_proj, normed_out, out),
            )?;
            crate::profile::profile_op(
                &self.ctx,
                "linear/allreduce",
                Some(linear_idx),
                rows,
                || {
                    if b2 {
                        // B2: partials span attn_tp x cp == world (attn_dp=1).
                        self.tp.all_reduce_sum(&self.ctx, out)
                    } else {
                        self.tp.attn_all_reduce_sum(&self.ctx, out)
                    }
                },
            )?;
        }
        Ok(())
    }

    /// [`Self::advance_linear_conv_gdr`] for B rows of the SAME `len` in one
    /// conv + one GDR launch. Uniform `len` makes the varlen kernels' `s * len`
    /// row stride identical to the trunk's ragged packing, so the shared
    /// scratch needs no repack.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn advance_linear_conv_gdr_batched(
        &self,
        attn: &LinearAttn,
        rs: &mut [LinearRow<'_>],
        linear_idx: usize,
        len: usize,
        qkv: &HiddenStates,
        b_proj: &HiddenStates,
        a_proj: &HiddenStates,
        qkv_conv: &mut HiddenStates,
        gdr_out: &mut HiddenStates,
        batch_ptrs: &mut SliceSlot<u64>,
        batch_len: &mut SliceSlot<i32>,
        host: &mut Vec<u64>,
        len_host: &mut Vec<i32>,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let c = &self.config;
        let b = rs.len();
        let qkv_dim = self.local_linear_qkv_dim();
        let b_dim = attn.in_proj_ba.rows / 2;
        let (qkv_base, _g0) = qkv.data.device_ptr(&ctx.stream);
        let (b_base, _g1) = b_proj.data.device_ptr(&ctx.stream);
        let (a_base, _g2) = a_proj.data.device_ptr(&ctx.stream);
        let elem = std::mem::size_of::<bf16>() as u64;
        // Five contiguous B-entry tables in one upload, in kernel argument
        // order: conv x, conv ring, b, a, GDR state.
        host.clear();
        host.resize(5 * b, 0);
        for (i, r) in rs.iter_mut().enumerate() {
            let row = (i * len) as u64;
            let conv_state = &mut r.slot.conv_states[linear_idx];
            ensure!(
                conv_state.len == qkv_dim * (c.linear_conv_kernel_dim - 1),
                "Qwen3.5 conv state len {} != qkv_dim*(kernel-1) {}",
                conv_state.len,
                qkv_dim * (c.linear_conv_kernel_dim - 1)
            );
            host[i] = qkv_base + row * qkv_dim as u64 * elem;
            host[b + i] = conv_state.data.device_ptr_mut(&ctx.stream).0;
            host[2 * b + i] = b_base + row * b_dim as u64 * elem;
            host[3 * b + i] = a_base + row * b_dim as u64 * elem;
            host[4 * b + i] = r.slot.gdr_states[linear_idx].device_ptr_mut(&ctx.stream).0;
        }
        let tbl = batch_ptrs.get(ctx, host.len())?;
        ctx.stream
            .memcpy_htod(host, tbl)
            .map_err(|e| anyhow!("H2D linear batch pointer table: {e}"))?;
        let (base, _gt) = tbl.device_ptr(&ctx.stream);
        // Same for every layer of a tick, so upload only when its shape
        // changes; `get` zero-fills a resize, so both dims must be checked.
        let d = batch_len.get(ctx, b)?;
        if len_host.len() != b || len_host[0] != len as i32 {
            len_host.clear();
            len_host.resize(b, len as i32);
            ctx.stream
                .memcpy_htod(len_host, d)
                .map_err(|e| anyhow!("H2D linear batch row lengths: {e}"))?;
        }
        let (len_ptr, _gl) = d.device_ptr(&ctx.stream);
        let (w_ptr, _g3) = attn.conv1d_weight.data.device_ptr(&ctx.stream);
        let (dt_ptr, _g4) = attn.dt_bias.data.device_ptr(&ctx.stream);
        let (alog_ptr, _g5) = attn.a_log.device_ptr(&ctx.stream);
        let (cv_ptr, _g6) = qkv_conv.data.device_ptr_mut(&ctx.stream);
        let (o_ptr, _g7) = gdr_out.data.device_ptr_mut(&ctx.stream);
        let table = |k: u64| base + k * b as u64 * 8;
        crate::profile::profile_op(ctx, "linear/conv1d", Some(linear_idx), b * len, || {
            // SAFETY: each table holds `b` live pointers staged above; the
            // shared scratch is `[b * len, dim]`.
            unsafe {
                ffi::conv1d_prefill_varlen_cuda(
                    table(0) as *const *const ffi::Half,
                    w_ptr as *const ffi::Half,
                    table(1) as *const *mut ffi::Half,
                    len_ptr as *const i32,
                    cv_ptr as *mut ffi::Half,
                    qkv_dim as i32,
                    len as i32,
                    c.linear_conv_kernel_dim as i32,
                    b as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
            Ok(())
        })?;
        crate::profile::profile_op(
            ctx,
            "linear/gdr_recurrent",
            Some(linear_idx),
            b * len,
            || {
                // SAFETY: same tables; qkv_conv/gdr_out are `[b * len, dim]`.
                unsafe {
                    ffi::gated_delta_rule_prefill_recurrent_varlen_cuda(
                        cv_ptr as *const ffi::Half,
                        table(2) as *const *const ffi::Half,
                        table(3) as *const *const ffi::Half,
                        dt_ptr as *const ffi::Half,
                        alog_ptr as *const f32,
                        table(4) as *const *mut f32,
                        len_ptr as *const i32,
                        o_ptr as *mut ffi::Half,
                        self.local_linear_k_heads as i32,
                        self.local_linear_v_heads as i32,
                        c.linear_key_head_dim as i32,
                        c.linear_value_head_dim as i32,
                        len as i32,
                        b as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    /// Conv1d (advances `slot.conv_states[linear_idx]`) + gated-delta rule
    /// (advances `slot.gdr_states[linear_idx]`) for one linear layer over
    /// `seq_len` rows. The ONLY persistent-state-mutating core of
    /// [`Self::linear_attention`], so the partial-accept replay re-runs the
    /// identical dispatch and advances the state byte-identically.
    ///
    /// `qkv_in` is the post-in_proj fused `[q|k|v]` PRE-conv1d (`qkv_dim` wide,
    /// token-major). Every view spans EXACTLY this slot's `seq_len` rows, so
    /// each slot's state sees only its own tokens.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn advance_linear_conv_gdr(
        &self,
        attn: &LinearAttn,
        qkv_in: &CudaView<'_, bf16>,
        b_in: &CudaView<'_, bf16>,
        a_in: &CudaView<'_, bf16>,
        slot: &mut Qwen35SlotState,
        linear_idx: usize,
        seq_len: usize,
        qkv_conv: &mut CudaViewMut<'_, bf16>,
        gdr_out: &mut CudaViewMut<'_, bf16>,
        fq_q: &mut HiddenSlot,
        fq_k: &mut HiddenSlot,
        fq_v: &mut HiddenSlot,
        fq_a: &mut HiddenSlot,
        fq_g: &mut SliceSlot<f32>,
        fq_g_cumsum: &mut SliceSlot<f32>,
        fq_beta: &mut SliceSlot<f32>,
        decode: Option<LinearDecodeGeom>,
    ) -> Result<()> {
        let c = &self.config;
        let b2 = decode.is_some();
        let (qkv_dim, z_dim, k_heads, v_heads, v_off) = decode.map_or(
            (
                self.local_linear_qkv_dim(),
                self.local_linear_z_dim(),
                self.local_linear_k_heads,
                self.local_linear_v_heads,
                0usize,
            ),
            |d| (d.qkv_dim, d.z_dim, d.k_heads, d.v_heads, d.v_off),
        );

        let conv_state = if b2 {
            &mut slot.conv_states_decode[linear_idx]
        } else {
            &mut slot.conv_states[linear_idx]
        };
        ensure!(
            conv_state.len == qkv_dim * (c.linear_conv_kernel_dim - 1),
            "Qwen3.5 conv state len {} != qkv_dim*(kernel-1) {}",
            conv_state.len,
            qkv_dim * (c.linear_conv_kernel_dim - 1)
        );
        let conv_weight = if b2 {
            &attn
                .decode
                .as_ref()
                .expect("B2 decode weights present when cp_decode active")
                .conv1d_weight
                .data
        } else {
            &attn.conv1d_weight.data
        };
        {
            {
                let (x_ptr, _g0) = qkv_in.device_ptr(&self.ctx.stream);
                let (w_ptr, _g1) = conv_weight.device_ptr(&self.ctx.stream);
                let (s_ptr, _g2) = conv_state.data.device_ptr_mut(&self.ctx.stream);
                let (o_ptr, _g3) = qkv_conv.device_ptr_mut(&self.ctx.stream);
                // SAFETY: qkv/weight/state/out valid on ctx.stream; weight len checked
                // by the kernel against num_channels*kernel.
                crate::profile::profile_op(
                    &self.ctx,
                    "linear/conv1d",
                    Some(linear_idx),
                    seq_len,
                    || {
                        // SAFETY: ptrs from live device allocations sized to the dims
                        // passed.
                        unsafe {
                            ffi::conv1d_prefill_cuda(
                                x_ptr as *const ffi::Half,
                                w_ptr as *const ffi::Half,
                                s_ptr as *mut ffi::Half,
                                o_ptr as *mut ffi::Half,
                                qkv_dim as i32,
                                seq_len as i32,
                                c.linear_conv_kernel_dim as i32,
                                self.ctx.stream.cu_stream(),
                            )
                            .result()?;
                        }
                        Ok(())
                    },
                )?;
            }
        }

        // The FlashQLA chunked path has one AOT instantiation per (Hg, H)
        // geometry; unknown geometry falls back to the recurrent kernel.
        let fq_fns: Option<(ffi::FqCumsumFn, ffi::FqKktFn, ffi::FqFwdFn)> = match (k_heads, v_heads)
        {
            (16, 32) => Some((
                ffi::gdr_fq_cumsum_cuda as _,
                ffi::gdr_fq_kkt_cuda as _,
                ffi::gdr_fq_fwd_cuda as _,
            )),
            (16, 48) => Some((
                ffi::gdr_fq_cumsum_h48_cuda as _,
                ffi::gdr_fq_kkt_h48_cuda as _,
                ffi::gdr_fq_fwd_h48_cuda as _,
            )),
            _ => None,
        };
        let use_fq_chunked = seq_len > 1
            && crate::runtime_flags::qwen35_gdr_chunked()
            && c.linear_key_head_dim == 128
            && c.linear_value_head_dim == 128
            && fq_fns.is_some()
            && fq_kernels_available(&self.ctx);
        if use_fq_chunked {
            // The AOT dispatch wrapper resolves SM + module via the calling
            // thread's DRIVER context, which runtime-API kernels never need;
            // without this bind the fq path returns NOT_SUPPORTED.
            self.ctx
                .ctx
                .bind_to_thread()
                .map_err(|e| anyhow!("bind CUDA context for chunked GDR failed: {e}"))?;
            let (fq_cumsum, fq_kkt, fq_fwd) = fq_fns.unwrap();
            let hg_dim = k_heads * c.linear_key_head_dim;
            let fq_q = fq_q.get(&self.ctx, hg_dim, seq_len)?;
            let fq_k = fq_k.get(&self.ctx, hg_dim, seq_len)?;
            let fq_v = fq_v.get(&self.ctx, z_dim, seq_len)?;
            let fq_a = fq_a.get(&self.ctx, v_heads * 64, seq_len)?;
            let g_len = v_heads * seq_len;
            let fq_g = fq_g.get(&self.ctx, g_len)?;
            let fq_g_cumsum = fq_g_cumsum.get(&self.ctx, g_len)?;
            let fq_beta = fq_beta.get(&self.ctx, g_len)?;
            let gdr_state = if b2 {
                &mut slot.gdr_states_decode[linear_idx]
            } else {
                &mut slot.gdr_states[linear_idx]
            };

            let (qkv_ptr, _g0) = qkv_conv.device_ptr(&self.ctx.stream);
            let (b_ptr, _g1) = b_in.device_ptr(&self.ctx.stream);
            let (a_ptr, _g2) = a_in.device_ptr(&self.ctx.stream);
            let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
            let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
            let dt_ptr = dt_ptr + (v_off * std::mem::size_of::<ffi::Half>()) as u64;
            let alog_ptr = alog_ptr + (v_off * std::mem::size_of::<f32>()) as u64;
            let (q_ptr, _g5) = fq_q.data.device_ptr_mut(&self.ctx.stream);
            let (k_ptr, _g6) = fq_k.data.device_ptr_mut(&self.ctx.stream);
            let (v_ptr, _g7) = fq_v.data.device_ptr_mut(&self.ctx.stream);
            let (a_inv_ptr, _g8) = fq_a.data.device_ptr_mut(&self.ctx.stream);
            let (g_ptr, _g9) = fq_g.device_ptr_mut(&self.ctx.stream);
            let (gc_ptr, _g10) = fq_g_cumsum.device_ptr_mut(&self.ctx.stream);
            let (beta_ptr, _g11) = fq_beta.device_ptr_mut(&self.ctx.stream);
            let (s_ptr, _g12) = gdr_state.device_ptr_mut(&self.ctx.stream);
            let (o_ptr, _g13) = gdr_out.device_ptr_mut(&self.ctx.stream);
            // SAFETY: all buffers valid on ctx.stream, shapes per the slot
            // `.get` calls above. The slot state pointer is passed as BOTH
            // h0 and ht (in-place chunk chaining): each fwd CTA reads its h0
            // slice fully before writing the same ht slice.
            crate::profile::profile_op(
                &self.ctx,
                "linear/gdr_fq",
                Some(linear_idx),
                seq_len,
                || {
                    // SAFETY: ptrs from live device allocations sized to the dims
                    // passed.
                    unsafe {
                        ffi::gdr_fq_prep_cuda(
                            qkv_ptr as *const ffi::Half,
                            b_ptr as *const ffi::Half,
                            a_ptr as *const ffi::Half,
                            dt_ptr as *const ffi::Half,
                            alog_ptr as *const f32,
                            q_ptr as *mut ffi::Half,
                            k_ptr as *mut ffi::Half,
                            v_ptr as *mut ffi::Half,
                            g_ptr as *mut f32,
                            beta_ptr as *mut f32,
                            k_heads as i32,
                            v_heads as i32,
                            c.linear_key_head_dim as i32,
                            c.linear_value_head_dim as i32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                        fq_cumsum(
                            g_ptr as *const f32,
                            gc_ptr as *mut f32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                        fq_kkt(
                            k_ptr as *const ffi::Half,
                            beta_ptr as *const f32,
                            a_inv_ptr as *mut ffi::Half,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                        fq_fwd(
                            q_ptr as *const ffi::Half,
                            k_ptr as *const ffi::Half,
                            v_ptr as *const ffi::Half,
                            a_inv_ptr as *const ffi::Half,
                            gc_ptr as *const f32,
                            beta_ptr as *const f32,
                            s_ptr as *const f32,
                            o_ptr as *mut ffi::Half,
                            s_ptr as *mut f32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                    Ok(())
                },
            )?;
        }
        if !use_fq_chunked {
            let gdr_state = if b2 {
                &mut slot.gdr_states_decode[linear_idx]
            } else {
                &mut slot.gdr_states[linear_idx]
            };
            {
                let (qkv_ptr, _g0) = qkv_conv.device_ptr(&self.ctx.stream);
                let (b_ptr, _g1) = b_in.device_ptr(&self.ctx.stream);
                let (a_ptr, _g2) = a_in.device_ptr(&self.ctx.stream);
                let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
                let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
                let dt_ptr = dt_ptr + (v_off * std::mem::size_of::<ffi::Half>()) as u64;
                let alog_ptr = alog_ptr + (v_off * std::mem::size_of::<f32>()) as u64;
                let (s_ptr, _g5) = gdr_state.device_ptr_mut(&self.ctx.stream);
                let (o_ptr, _g6) = gdr_out.device_ptr_mut(&self.ctx.stream);
                crate::profile::profile_op(
                    &self.ctx,
                    "linear/gdr_recurrent",
                    Some(linear_idx),
                    seq_len,
                    || {
                        // SAFETY: all buffers valid on ctx.stream; head dims from
                        // config.
                        unsafe {
                            if seq_len == 1 {
                                ffi::gated_delta_rule_decode_cuda(
                                    qkv_ptr as *const ffi::Half,
                                    b_ptr as *const ffi::Half,
                                    a_ptr as *const ffi::Half,
                                    dt_ptr as *const ffi::Half,
                                    alog_ptr as *const f32,
                                    s_ptr as *mut f32,
                                    o_ptr as *mut ffi::Half,
                                    k_heads as i32,
                                    v_heads as i32,
                                    c.linear_key_head_dim as i32,
                                    c.linear_value_head_dim as i32,
                                    self.ctx.stream.cu_stream(),
                                )
                                .result()?;
                            } else {
                                ffi::gated_delta_rule_prefill_recurrent_cuda(
                                    qkv_ptr as *const ffi::Half,
                                    b_ptr as *const ffi::Half,
                                    a_ptr as *const ffi::Half,
                                    dt_ptr as *const ffi::Half,
                                    alog_ptr as *const f32,
                                    s_ptr as *mut f32,
                                    o_ptr as *mut ffi::Half,
                                    k_heads as i32,
                                    v_heads as i32,
                                    c.linear_key_head_dim as i32,
                                    c.linear_value_head_dim as i32,
                                    seq_len as i32,
                                    self.ctx.stream.cu_stream(),
                                )
                                .result()?;
                            }
                        }
                        Ok(())
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Partial-accept linear-only replay: advance ONLY the gated-delta
    /// recurrent + conv states over the accepted prefix (`k+1` rows) from the
    /// verify capture.
    ///
    /// Precondition: the caller has just `restore_trunk`-ed the conv + gdr
    /// states to the pre-verify snapshot, so re-running the first `k+1`
    /// recurrent steps reproduces the verify's state bit-for-bit.
    ///
    /// Mutates `slot.conv_states[li]` / `slot.gdr_states[li]` for every linear
    /// layer plus the `ws.linear` scratch. The full-attn KV caches and
    /// `slot.seq_len` are the caller's to rewind. No H2D/D2H/sync.
    pub(crate) fn replay_linear_only(
        &self,
        slot: &mut Qwen35SlotState,
        ws: &mut Qwen35Workspace,
        capture: &Qwen35LinearCapture,
        k: usize,
    ) -> Result<()> {
        let num_linear = slot.conv_states.len();
        ensure!(
            capture.qkv.len() == num_linear
                && capture.b_proj.len() == num_linear
                && capture.a_proj.len() == num_linear,
            "spec capture linear count {}/{}/{} != slot linear layers {num_linear}",
            capture.qkv.len(),
            capture.b_proj.len(),
            capture.a_proj.len()
        );
        let rows = k + 1;
        ensure!(
            rows <= capture.rows,
            "spec replay needs {rows} rows but capture holds {}",
            capture.rows
        );
        // Each slot's capture holds ITS OWN rows token-major from offset 0, so
        // the accepted prefix is the leading `k+1` columns.
        let qkv_dim = self.local_linear_qkv_dim();
        let z_dim = self.local_linear_z_dim();
        let Qwen35Workspace { linear, .. } = ws;
        let LinearAttnScratch {
            qkv_conv,
            gdr_out,
            fq_q,
            fq_k,
            fq_v,
            fq_a,
            fq_g,
            fq_g_cumsum,
            fq_beta,
            ..
        } = linear;
        let qkv_conv = qkv_conv.get(&self.ctx, qkv_dim, rows)?;
        let gdr_out = gdr_out.get(&self.ctx, z_dim, rows)?;
        let mut li = 0usize;
        for layer in &self.layers {
            if let Qwen35Attn::Linear(attn) = &layer.attn {
                let b_dim = attn.in_proj_ba.rows / 2;
                let a_dim = b_dim;
                self.advance_linear_conv_gdr(
                    attn,
                    &capture.qkv[li].data.slice(0..rows * qkv_dim),
                    &capture.b_proj[li].data.slice(0..rows * b_dim),
                    &capture.a_proj[li].data.slice(0..rows * a_dim),
                    slot,
                    li,
                    rows,
                    &mut qkv_conv.data.slice_mut(..),
                    &mut gdr_out.data.slice_mut(..),
                    fq_q,
                    fq_k,
                    fq_v,
                    fq_a,
                    fq_g,
                    fq_g_cumsum,
                    fq_beta,
                    None,
                )?;
                li += 1;
            }
        }
        ensure!(
            li == num_linear,
            "spec replay advanced {li} linear layers != slot count {num_linear}"
        );
        Ok(())
    }

    /// [`Self::replay_linear_only`] for a whole batch: one conv1d and one
    /// gated-delta launch per layer instead of two per slot per layer. Each
    /// slot keeps its own capture and state, reached through `tables`.
    pub(crate) fn replay_linear_only_batched(
        &self,
        slots: &mut [&mut Qwen35SlotState],
        captures: &[&Qwen35LinearCapture],
        ks: &[usize],
        tables: &mut Qwen35ReplayTables,
        ws: &mut Qwen35Workspace,
    ) -> Result<()> {
        let b = slots.len();
        ensure!(
            b == captures.len() && b == ks.len(),
            "batched replay: {b} slots vs {} captures / {} ks",
            captures.len(),
            ks.len()
        );
        let num_linear = slots[0].conv_states.len();
        let max_len = ks.iter().map(|k| k + 1).max().unwrap_or(0);
        ensure!(max_len >= 1, "batched replay with no rows");
        for (s, cap) in captures.iter().enumerate() {
            ensure!(
                cap.qkv.len() == num_linear && ks[s] < cap.rows,
                "batched replay slot {s}: capture {} layers / {} rows cannot hold {} rows of \
                 {num_linear} layers",
                cap.qkv.len(),
                cap.rows,
                ks[s] + 1
            );
        }
        let ctx = &self.ctx;
        tables.stage(ctx, slots, captures, ks, num_linear)?;

        let qkv_dim = self.local_linear_qkv_dim();
        let z_dim = self.local_linear_z_dim();
        let rows = b * max_len;
        let Qwen35Workspace { linear, .. } = ws;
        let qkv_conv = linear.qkv_conv.get(ctx, qkv_dim, rows)?;
        let gdr_out = linear.gdr_out.get(ctx, z_dim, rows)?;
        let (cv_ptr, _gc) = qkv_conv.data.device_ptr_mut(&ctx.stream);
        let (go_ptr, _gg) = gdr_out.data.device_ptr_mut(&ctx.stream);
        let stride = num_linear * b;
        let (tbl, _gt) = tables
            .ptrs
            .get(ctx, REPLAY_TABLES * stride)?
            .device_ptr(&ctx.stream);
        let lay = ReplayLayout {
            base: tbl,
            ..tables.layout
        };
        let (len_ptr, _gl) = tables.row_len.get(ctx, b)?.device_ptr(&ctx.stream);
        let c = &self.config;
        let mut li = 0usize;
        for layer in &self.layers {
            let Qwen35Attn::Linear(attn) = &layer.attn else {
                continue;
            };
            let (w_ptr, _g0) = attn.conv1d_weight.data.device_ptr(&ctx.stream);
            let (dt_ptr, _g1) = attn.dt_bias.data.device_ptr(&ctx.stream);
            let (alog_ptr, _g2) = attn.a_log.device_ptr(&ctx.stream);
            let qkv_tbl = lay.table(TBL_QKV, li);
            let b_tbl = lay.table(TBL_B, li);
            let a_tbl = lay.table(TBL_A, li);
            let conv_tbl = lay.table(TBL_CONV, li);
            let gdr_tbl = lay.table(TBL_GDR, li);
            // SAFETY: each table holds `b` pointers staged above; the shared
            // scratch is `[b * max_len, dim]`.
            unsafe {
                ffi::conv1d_prefill_varlen_cuda(
                    qkv_tbl as *const *const ffi::Half,
                    w_ptr as *const ffi::Half,
                    conv_tbl as *const *mut ffi::Half,
                    len_ptr as *const i32,
                    cv_ptr as *mut ffi::Half,
                    qkv_dim as i32,
                    max_len as i32,
                    c.linear_conv_kernel_dim as i32,
                    b as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
                ffi::gated_delta_rule_prefill_recurrent_varlen_cuda(
                    cv_ptr as *const ffi::Half,
                    b_tbl as *const *const ffi::Half,
                    a_tbl as *const *const ffi::Half,
                    dt_ptr as *const ffi::Half,
                    alog_ptr as *const f32,
                    gdr_tbl as *const *mut f32,
                    len_ptr as *const i32,
                    go_ptr as *mut ffi::Half,
                    self.local_linear_k_heads as i32,
                    self.local_linear_v_heads as i32,
                    c.linear_key_head_dim as i32,
                    c.linear_value_head_dim as i32,
                    max_len as i32,
                    b as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
            li += 1;
        }
        ensure!(
            li == num_linear,
            "batched replay advanced {li} linear layers != slot count {num_linear}"
        );
        Ok(())
    }
}
