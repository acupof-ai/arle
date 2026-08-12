use super::*;

pub(crate) enum Qwen35Attn {
    // Boxed: `FullAttn`/`LinearAttn` are large (multiple DeviceMatrix) and
    // size-skewed; boxing keeps the enum small (clippy::large_enum_variant).
    Full(Box<FullAttn>),
    Linear(Box<LinearAttn>),
}

/// Gated full attention (the q rows carry the per-head sigmoid gate → q part
/// rows = `heads*head_dim*2`).
pub(crate) struct FullAttn {
    /// Row-fused `[q; k; v]` (`[q_gated + 2*kv, hidden]`): one GEMM per step,
    /// split into the q/k/v buffers downstream.
    pub(crate) qkv_proj: DeviceMatrix,
    pub(crate) o_proj: DeviceMatrix,
    pub(crate) q_norm: DeviceVec,
    pub(crate) k_norm: DeviceVec,
}

/// Gated-delta-rule linear attention.
pub(crate) struct LinearAttn {
    /// Row-fused `[qkv; z]` (`[qkv_dim + z_dim, hidden]`): one GEMM per step,
    /// split into the conv-input qkv and the z gate downstream.
    pub(crate) in_proj_qkvz: DeviceMatrix,
    /// Row-fused `[b; a]` (`[2*Vh, hidden]`): one GEMM per step, split into
    /// the per-head b/a scalars downstream. b = rows `0..Vh`, a = `Vh..2*Vh`.
    pub(crate) in_proj_ba: DeviceMatrix,
    pub(crate) conv1d_weight: DeviceVec,
    pub(crate) dt_bias: DeviceVec,
    /// This rank's v-head shard under TP.
    pub(crate) a_log: CudaSlice<f32>,
    /// Gated output RMSNorm scale, broadcast across heads; replicated under TP.
    pub(crate) norm_weight: CudaSlice<f32>,
    pub(crate) out_proj: DeviceMatrix,
}

fn format_fq_parity_metrics(reference: &[f64], actual: &[f64], rel_floor: f64) -> String {
    assert_eq!(reference.len(), actual.len());
    let mut abs_errors = Vec::with_capacity(reference.len());
    let mut squared_error = 0.0;
    let mut max_rel = 0.0f64;
    let mut dot = 0.0;
    let mut reference_norm = 0.0;
    let mut actual_norm = 0.0;
    let mut non_finite_mismatch = 0;
    for (&reference, &actual) in reference.iter().zip(actual) {
        if !reference.is_finite() || !actual.is_finite() {
            non_finite_mismatch += usize::from(reference.to_bits() != actual.to_bits());
            continue;
        }
        let abs = (reference - actual).abs();
        abs_errors.push(abs);
        squared_error += abs * abs;
        max_rel = max_rel.max(abs / reference.abs().max(rel_floor));
        dot += reference * actual;
        reference_norm += reference * reference;
        actual_norm += actual * actual;
    }
    abs_errors.sort_unstable_by(f64::total_cmp);
    let max_abs = abs_errors.last().copied().unwrap_or(0.0);
    let p99_index = (99 * abs_errors.len()).div_ceil(100).saturating_sub(1);
    let p99_abs = abs_errors.get(p99_index).copied().unwrap_or(0.0);
    let rmse = (squared_error / abs_errors.len().max(1) as f64).sqrt();
    let rel_l2 = (squared_error / reference_norm.max(1e-30)).sqrt();
    let cosine = if reference_norm == 0.0 && actual_norm == 0.0 {
        1.0
    } else if reference_norm == 0.0 || actual_norm == 0.0 {
        0.0
    } else {
        dot / (reference_norm * actual_norm).sqrt()
    };
    format!(
        "max_abs={max_abs:.9e} p99_abs={p99_abs:.9e} rmse={rmse:.9e} \
         max_rel={max_rel:.9e} rel_floor={rel_floor:.1e} rel_l2={rel_l2:.9e} \
         cosine={cosine:.9} non_finite_mismatch={non_finite_mismatch}"
    )
}

#[cfg(test)]
#[test]
fn fq_parity_metrics_reports_distribution_and_non_finite_mismatch() {
    let report = format_fq_parity_metrics(&[0.0, 1.0, f64::INFINITY], &[0.0, 1.5, 0.0], 1e-6);
    assert!(report.contains("max_abs=5.000000000e-1"));
    assert!(report.contains("rmse=3.535533906e-1"));
    assert!(report.contains("non_finite_mismatch=1"));
}

impl Qwen35Model {
    pub(crate) fn full_attention(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        slot: &mut Qwen35SlotState,
        full_idx: usize,
        start_pos: usize,
        start_pos_dev: &CudaSlice<i32>,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let k_cache = &mut slot.k_caches[full_idx];
        let v_cache = &mut slot.v_caches[full_idx];
        self.full_attention_into(
            attn,
            normed,
            k_cache,
            v_cache,
            full_idx,
            start_pos,
            start_pos_dev,
            fw,
            out,
        )
    }

    /// Gated full attention over an explicit contiguous K/V cache (`max_seq_len`
    /// derived from `k_cache.len / kv_dim`), uncached recompute over
    /// `[0, start_pos+seq_len)` each call, into `out` (`[hidden, seq]`, beta=0
    /// o_proj GEMM). The prep kernel fuses q/k RMSNorm + RoPE + cache write; the
    /// gate kernel applies the per-head sigmoid gate carried in `q_full`.
    /// `start_pos_dev` is the GPU-resident `start_pos` (same for every layer of
    /// one call). Prefill chunks (`seq_len > 1`) route through the vendored FA3
    /// hopper fwd when [`qwen35_fa3_enabled`]; decode keeps the devpos kernel
    /// (graph-captured) untouched. The trunk passes its per-slot cache via
    /// [`Self::full_attention`]; the MTP draft head passes its own fresh
    /// per-block cache. `full_idx` is only a profiling label. Numerics, kernels,
    /// and launch order are identical to the pre-extraction trunk path.
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
        // LOCAL per-rank widths (= global config on a single GPU): the sharded
        // q/k/v GEMM outputs, the per-slot caches, and the kernel launches must
        // all agree on this rank's head shard.
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
                // SAFETY: all buffers valid on ctx.stream; cache sized max_seq_len*kv_dim.
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

        // Attention over the contiguous cache (causal; decode = qlen 1).
        // Decode (`seq_len == 1`) takes the devpos entry: the kv length is read
        // from the staged `start_pos` DEVICE buffer inside the kernel (same
        // math — kv_len = start_pos + token + 1 either way), so the launch is
        // CUDA-graph capture-safe and ONE captured graph replays across
        // positions. Eager decode uses the same entry, keeping the graph lane
        // kernel-for-kernel identical to its eager warm runs. Prefill keeps
        // the host-scalar entry (multi-token, never captured).
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
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
                    unsafe {
                        if seq_len == 1
                            && qwen35_fa2_sm70_enabled(&self.ctx)
                            && !crate::runtime_flags::qwen35_decode_graph()
                        {
                            // FA2 sm_70 decode (eager). The host kv_len arg is
                            // not graph-replay safe, so captured decode falls
                            // through to the devpos kernel below.
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
                            // FA3 fwd over the SAME buffers the in-tree kernel uses:
                            // q/out token-major [S, h, 256] (HD256 prep layout),
                            // cache head-major [h_k, max_seq, 256]. Passing the
                            // exact `kv_len` as seqlen_k keeps the shim on the
                            // non-varlen path; causal is bottom-right aligned =
                            // chunked-prefill semantics. Gate + o_proj follow
                            // unchanged.
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
                            // FA2 sm_70 prefill (SOTA on V100; FA3 is sm_80+).
                            // Causal chunked-prefill; same buffers as naive.
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
                // SAFETY: q_full/attn_out valid on ctx.stream; gate layout per full-attn prep.
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
        // Row-parallel o_proj: sum the per-rank partials (no-op single-GPU).
        crate::profile::profile_op(&self.ctx, "full/allreduce", Some(full_idx), seq_len, || {
            self.tp.all_reduce_sum(&self.ctx, out)
        })?;
        Ok(())
    }

    /// Paged full attention over `meta`'s ragged page table: qkv GEMM → prep →
    /// paged attention → sigmoid gate → o_proj → all-reduce. The dispatch
    /// `(batch, total_q, max_qlen)` makes 1×T, B×1 and the B×T middle one path.
    /// RoPE is baked into the cached K at write time, so a recall-restricted
    /// page subset attends exactly those pages (the dense-Qwen3 argument).
    ///
    /// `layer0_query` is the `--kv-recall` sink: on a multi-row prefill the
    /// post-RoPE layer-0 Q is read back for the next step's recall score
    /// (head-major `[num_q_heads * head_dim]`, matching `recompute_recall_plan`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn full_attention_paged(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        full_idx: usize,
        pool: &PagedKVPool,
        meta: &crate::loader::PageMeta,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
        layer0_query: Option<&mut Vec<f32>>,
    ) -> Result<()> {
        let c = &self.config;
        let rows = normed.seq_len;
        ensure!(
            meta.total_q == rows,
            "Qwen3.6 paged full attention: page table covers {} query tokens != {rows} rows",
            meta.total_q
        );
        // `max_qlen == 1` is the decode kernel's contract (one q row per batch
        // element); anything longer goes through the ragged prefill kernel.
        let decode = meta.seq_len == 1;
        let q_dim = self.local_full_attn_q_dim();
        let kv_dim = self.local_full_attn_kv_dim();
        let q_proj_dim = self.local_full_attn_q_proj_dim();
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
            ..
        } = fw;
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
                gemm_batch(&self.ctx, &attn.qkv_proj, normed, qkv_fused)?;
                split_qkv(&self.ctx, qkv_fused, q_full, k_batch, v_batch)?;
                Ok(())
            },
        )?;

        let q_prepped = q_prepped.get(&self.ctx, q_dim, rows)?;
        let attn_out = attn_heads.get(&self.ctx, q_dim, rows)?;

        let k_pool_ptr = pool.k_ptr(full_idx, &self.ctx.stream);
        let v_pool_ptr = pool.v_ptr(full_idx, &self.ctx.stream);

        {
            #[cfg(test)]
            let prep_capture = if !decode && meta.batch == 1 {
                prep_probe::begin(
                    &self.ctx,
                    full_idx,
                    rows,
                    self.local_q_heads,
                    self.local_kv_heads,
                    c.head_dim,
                    c.rotary_dim,
                    c.rms_norm_eps,
                    &q_full.data,
                    &k_batch.data,
                    &attn.q_norm,
                    &attn.k_norm,
                    &self.cos_cache,
                    &self.sin_cache,
                    &meta.start_positions,
                )?
            } else {
                None
            };
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
                                    self.local_q_heads as i32,
                                    self.local_kv_heads as i32,
                                    pool.page_size as i32,
                                    stride_page as i32,
                                    meta.batch as i32,
                                    c.rotary_dim as i32,
                                    c.rms_norm_eps,
                                    self.ctx.stream.cu_stream(),
                                )
                                .result()?;
                            } else {
                                // The prep reads ONE scalar start_pos off a
                                // table based at element 0 — launch per row.
                                let elem = std::mem::size_of::<ffi::Half>() as u64;
                                for b in 0..meta.batch {
                                    let (col, pages) = (meta.q_offsets[b], meta.page_offsets[b]);
                                    let len = meta.q_offsets[b + 1] - col;
                                    ffi::prefill_attention_paged_prep_hd256_cuda(
                                        (qf_ptr + (col * q_proj_dim) as u64 * elem)
                                            as *const ffi::Half,
                                        (qp_ptr + (col * q_dim) as u64 * elem) as *mut ffi::Half,
                                        (k_ptr + (col * kv_dim) as u64 * elem) as *const ffi::Half,
                                        (v_ptr + (col * kv_dim) as u64 * elem) as *const ffi::Half,
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
            #[cfg(test)]
            prep_probe::finish(&self.ctx, prep_capture, q_prepped)?;
        }

        // For quantized pools: BF16 work buffer → quantized data buffer.
        // The prep kernel above wrote the new tokens BF16 into `k_work` / `v_work`
        // (= `k_ptr` / `v_ptr` for FP8/INT8 pools). Quantize them into `k_data[layer]`
        // so the FP8 attention kernel can read the complete token history (prefix from
        // prior steps + new tokens just written) from one contiguous FP8 pool.
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
            #[cfg(test)]
            let attn_capture = if !decode && meta.batch == 1 {
                attn_probe::begin(
                    &self.ctx,
                    full_idx,
                    rows,
                    self.local_q_heads,
                    self.local_kv_heads,
                    c.head_dim,
                    c.rotary_dim,
                    &q_prepped.data,
                    &k_batch.data,
                    &v_batch.data,
                    &attn.k_norm,
                    &self.cos_cache,
                    &self.sin_cache,
                    c.rms_norm_eps,
                    &meta.start_positions,
                )?
            } else {
                None
            };
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
                        // sm_90 short queries — decode (1 row) and spec verify
                        // (block+1): FA3 paged split-KV + PackGQA. The TileLang
                        // kernel it replaces pads BLOCK_M=64 around the real rows
                        // and gives one CTA per query head, 6× the KV traffic.
                        // batch==1 prefill chunks also route here: the 2026-07-28
                        // kill (c=8 TTFT 12.07→18.23 s) was the one-launch-per-
                        // request cost on ragged batches; a single request is a
                        // single launch either way. Ragged multi-request prefill
                        // and non-Hopper keep TileLang.
                        if (meta.seq_len <= FA3_MAX_QLEN || meta.batch == 1)
                            && pool.format == KVFormat::BF16
                            && c.head_dim == 256
                            && qwen35_fa3_enabled(&self.ctx)
                        {
                            let (qp_ptr, _g0) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
                            let (ao_ptr, _g5) = attn_out.data.device_ptr_mut(&self.ctx.stream);
                            // ONE launch for the whole batch: q/o are packed
                            // [total_q, h, d] behind `q_indptr`, and each row's KV
                            // extent comes from `kv_lens_dev` against the
                            // rectangular page table. `seqused_k` does not drop the
                            // K/V batch strides (only `cu_seqlens_k` does,
                            // flash_api.cpp:105-108), which is what lets a paged
                            // batch share a launch — per row it was 16 CTAs on 78
                            // SMs, serialized.
                            // Split-KV pays only when q is tiny vs kv; a long
                            // prefill chunk saturates SMs on the q axis alone.
                            // Upper bound only: FA3 picks the live value itself
                            // (flash_prepare_scheduler.cu `num_splits_dynamic`),
                            // clamped by what we pass. pack_gqa leaves batch ×
                            // kv_heads tiles before splitting, so one tile per SM
                            // is where the ceiling stops costing anything — raising
                            // it further only grows the combine scratch, which
                            // measured +0.36% at batch 8. The historical 8 is the
                            // floor: it is the measured optimum from batch 4 up,
                            // and it bound the scheduler only at batch 1, where
                            // 4×8 = 32 tiles left 46 of 78 SMs idle.
                            let splits = if meta.seq_len <= FA3_MAX_QLEN {
                                match qwen35_fa3_decode_splits() {
                                    0 => self
                                        .ctx
                                        .sm_count()
                                        .div_ceil(meta.batch.max(1) * self.local_kv_heads.max(1))
                                        .max(FA3_DECODE_SPLITS_FLOOR)
                                        .clamp(2, 256),
                                    n => n,
                                }
                            } else {
                                1
                            };
                            let accum_rows = self.local_q_heads * meta.total_q;
                            let lse = fa3_lse.get(&self.ctx, accum_rows)?;
                            let oaccum =
                                fa3_oaccum.get(&self.ctx, splits * accum_rows * c.head_dim)?;
                            let lseaccum = fa3_lseaccum.get(&self.ctx, splits * accum_rows)?;
                            let meta_cap = meta.batch.div_ceil(4) * 4 * 4 + 1;
                            let sem = fa3_semaphore.get(&self.ctx, meta_cap)?;
                            let (lse_ptr, _f0) = lse.device_ptr_mut(&self.ctx.stream);
                            let (oaccum_ptr, _f1) = oaccum.device_ptr_mut(&self.ctx.stream);
                            let (lseaccum_ptr, _f2) = lseaccum.device_ptr_mut(&self.ctx.stream);
                            let (sem_ptr, _f3) = sem.device_ptr_mut(&self.ctx.stream);
                            let (kv_lens_ptr, _f4) = meta.kv_lens_dev.device_ptr(&self.ctx.stream);
                            let (rect_ptr, _f5) = meta.page_table_rect.device_ptr(&self.ctx.stream);
                            let head_dim = c.head_dim as i64;
                            let args = ffi::ArleFa3FwdHd256Args {
                                q: qp_ptr as *const ffi::Half,
                                k: k_pool_ptr as *const ffi::Half,
                                v: v_pool_ptr as *const ffi::Half,
                                o: ao_ptr as *mut ffi::Half,
                                softmax_lse: lse_ptr as *mut f32,
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
                                num_heads: self.local_q_heads as i32,
                                num_heads_k: self.local_kv_heads as i32,
                                head_dim: c.head_dim as i32,
                                q_row_stride: (self.local_q_heads * c.head_dim) as i64,
                                // HND pool [page, h_k, page_size, d]: tokens are
                                // contiguous inside a (page, head), heads stride
                                // by a page's worth, pages by the whole page.
                                k_row_stride: head_dim,
                                v_row_stride: head_dim,
                                o_row_stride: (self.local_q_heads * c.head_dim) as i64,
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
                            // SAFETY: q/o are the live prepped/out buffers; k/v are
                            // the layer's pool base; the page table is the meta's
                            // rectangular mirror, `batch * page_table_stride` long.
                            unsafe {
                                ffi::arle_fa3_fwd_hd256_bf16_cuda(
                                    &args,
                                    self.ctx.stream.cu_stream(),
                                )
                                .result()?;
                            }
                            return Ok(());
                        }
                        // A persistent-decode meta means a graph capture may be
                        // active — the TileLang lane below bakes `num_pages`
                        // as a host arg and would replay stale, so refuse (the
                        // capture error downgrades the lane to eager).
                        ensure!(
                            meta.seqlen_k_capture.is_none(),
                            "paged decode graph capture requires the FA3 BF16 lane"
                        );
                        match pool.format {
                            KVFormat::BF16 => {
                                let (qp_ptr, _g0) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
                                let (ao_ptr, _g5) = attn_out.data.device_ptr_mut(&self.ctx.stream);
                                // SAFETY: kernel signature from paged_attn_v1 ABI (18-arg BF16).
                                let kernel = ffi::resolve_paged_attn_v1(
                                    c.head_dim as u32,
                                    self.local_q_heads as u32,
                                    self.local_kv_heads as u32,
                                    phase,
                                )
                                .ok_or_else(|| {
                                    anyhow!(
                                        "no HD256 paged {} kernel for q{}_kv{}",
                                        if decode { "decode" } else { "prefill" },
                                        self.local_q_heads,
                                        self.local_kv_heads
                                    )
                                })?;
                                // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                                        self.local_q_heads as i32,
                                        self.local_kv_heads as i32,
                                        pool.page_size as i32,
                                        sm_scale,
                                        self.ctx.stream.cu_stream(),
                                    )
                                    .result()?;
                                }
                            }
                            KVFormat::FP8E4M3 => {
                                let (qp_ptr, _g0) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
                                let (ao_ptr, _g5) = attn_out.data.device_ptr_mut(&self.ctx.stream);
                                // SAFETY: kernel signature from paged_attn_fp8_v1 ABI (20-arg).
                                // k/v data buffers are FP8; scales are per-token per-kv-head f32.
                                let kernel = ffi::resolve_paged_attn_fp8_v1(
                                    c.head_dim as u32,
                                    self.local_q_heads as u32,
                                    self.local_kv_heads as u32,
                                    phase,
                                )
                                .ok_or_else(|| {
                                    anyhow!(
                                        "no HD256 FP8 paged {} kernel for q{}_kv{}",
                                        if decode { "decode" } else { "prefill" },
                                        self.local_q_heads,
                                        self.local_kv_heads
                                    )
                                })?;
                                let k_data = pool.k_data_ptr(full_idx, &self.ctx.stream);
                                let v_data = pool.v_data_ptr(full_idx, &self.ctx.stream);
                                let k_scales = pool.k_scales_ptr(full_idx, &self.ctx.stream);
                                let v_scales = pool.v_scales_ptr(full_idx, &self.ctx.stream);
                                // SAFETY: ptrs from live device allocations sized to the dims passed.
                                unsafe {
                                    kernel(
                                        qp_ptr as *mut ffi::Half,
                                        q_indptr_ptr as *const i32,
                                        k_data as *const u8,
                                        v_data as *const u8,
                                        k_scales as *const f32,
                                        v_scales as *const f32,
                                        kv_indptr_ptr as *const i32,
                                        kv_indices_ptr as *const i32,
                                        last_page_len_ptr as *const i32,
                                        ao_ptr as *mut ffi::Half,
                                        bsz,
                                        total_q,
                                        max_q,
                                        pool.max_total_pages as i32,
                                        meta.num_pages as i32,
                                        self.local_q_heads as i32,
                                        self.local_kv_heads as i32,
                                        pool.page_size as i32,
                                        sm_scale,
                                        self.ctx.stream.cu_stream(),
                                    )
                                    .result()?;
                                }
                            }
                            KVFormat::INT8 => {
                                // Per-(token, kv_head) symmetric INT8 KV: same
                                // split-KV varlen kernel as FP8, `int8_kv` flips
                                // the load to int8 + scale.
                                let ws = pool.quantized_attn_workspace()?;
                                let max_kv_len = meta.max_kv_len();
                                kv_quant::decode_attention_varlen_int8(
                                    &self.ctx,
                                    &q_prepped,
                                    q_indptr_ptr,
                                    pool.k_data_ptr(full_idx, &self.ctx.stream),
                                    pool.v_data_ptr(full_idx, &self.ctx.stream),
                                    Some(pool.k_scales_ptr(full_idx, &self.ctx.stream)),
                                    Some(pool.v_scales_ptr(full_idx, &self.ctx.stream)),
                                    kv_indptr_ptr,
                                    kv_indices_ptr,
                                    last_page_len_ptr,
                                    attn_out,
                                    self.local_q_heads,
                                    self.local_kv_heads,
                                    c.head_dim,
                                    pool.page_size,
                                    meta.batch,
                                    meta.total_q,
                                    max_kv_len,
                                    true, // causal
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
            #[cfg(test)]
            attn_probe::finish(&self.ctx, attn_capture, attn_out)?;
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
                        self.local_q_heads as i32,
                        rows as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            })?;
        }

        // Layer-0 PREFILL: read back the post-RoPE prepped Q for the recall score
        // (head-major `[num_q_heads * head_dim]`). Under the write-through model the
        // whole recall cycle (score → evict → prefetch) runs ONCE per prefill, not
        // per decode step, so only the multi-token prefill needs this signal — the
        // D2H stays off every other paged forward. `q_prepped` is token-major
        // `[rows, q_dim]`; the recall query is the mean of the last `m` prompt
        // tokens' queries (R3 — "what am I about to generate").
        if let Some(dst) = layer0_query
            && full_idx == 0
            && rows > 1
        {
            let host: Vec<bf16> = self
                .ctx
                .stream
                .clone_dtoh(&q_prepped.data)
                .map_err(|e| anyhow!("recall layer0 q dtoh: {e}"))?;
            const RECALL_PREFILL_Q_TOKENS: usize = 16; // R3 default `m`.
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

        crate::profile::profile_op(&self.ctx, "full_paged/o_proj", Some(full_idx), rows, || {
            gemm_batch(&self.ctx, &attn.o_proj, attn_out, out)
        })?;
        // Row-parallel o_proj: sum the per-rank partials (no-op single-GPU).
        crate::profile::profile_op(
            &self.ctx,
            "full_paged/allreduce",
            Some(full_idx),
            rows,
            || self.tp.all_reduce_sum(&self.ctx, out),
        )?;
        Ok(())
    }

    /// Gated-delta-rule linear attention into `out` (`[hidden, rows]`, beta=0
    /// out-proj GEMM): in-proj → depthwise conv1d → RECURRENT gated-delta
    /// (advances the per-slot state in place) → gated output RMSNorm →
    /// out-proj. The conv ring + recurrent state carry across prefill/decode.
    ///
    /// `rows = normed.seq_len` is the FLAT column count. Every weight-heavy
    /// step runs once over all of them; only [`LinearCore`] is per-slot.
    pub(crate) fn linear_attention(
        &self,
        attn: &LinearAttn,
        normed: &HiddenStates,
        core: LinearCore<'_, '_>,
        linear_idx: usize,
        lw: &mut LinearAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let rows = normed.seq_len;
        // LOCAL per-rank widths (= global config on a single GPU): the fused
        // [q|k|v] shard, conv channels, recurrent state, and kernel launches all
        // follow this rank's linear k/v head shard. b/a widths come off the
        // sharded projection rows directly (`[local_Vh, hidden]`).
        let qkv_dim = self.local_linear_qkv_dim();
        let z_dim = self.local_linear_z_dim();
        let b_dim = attn.in_proj_ba.rows / 2;
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
        } = lw;
        let qkvz = qkvz.get(&self.ctx, qkv_dim + z_dim, rows)?;
        let qkv = qkv.get(&self.ctx, qkv_dim, rows)?;
        let z = z.get(&self.ctx, z_dim, rows)?;
        let ba = ba.get(&self.ctx, b_dim + a_dim, rows)?;
        let b_proj = b_proj.get(&self.ctx, b_dim, rows)?;
        let a_proj = a_proj.get(&self.ctx, a_dim, rows)?;
        crate::profile::profile_op(&self.ctx, "linear/in_proj", Some(linear_idx), rows, || {
            gemm_batch(&self.ctx, &attn.in_proj_qkvz, normed, qkvz)?;
            split2(&self.ctx, qkvz, qkv, z)?;
            gemm_batch(&self.ctx, &attn.in_proj_ba, normed, ba)?;
            split2(&self.ctx, ba, b_proj, a_proj)?;
            Ok(())
        })?;

        let qkv_conv = qkv_conv.get(&self.ctx, qkv_dim, rows)?;
        let gdr_out = gdr_out.get(&self.ctx, z_dim, rows)?;
        match core {
            LinearCore::Rows(rs) => {
                let total: usize = rs.iter().map(|r| r.len).sum();
                ensure!(
                    total == rows,
                    "linear rows total {total} != {rows} staged columns"
                );
                // Each row's columns land at ITS capture offset 0, so a
                // partial-accept replay re-runs only that slot's prefix. Three
                // launches for the whole batch, not three per row.
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
                // Uniform short rows (a DSpark verify tick) pack identically to
                // the varlen kernels' `s * len` stride, so the whole batch is
                // one conv + one GDR launch. The per-row loop below is B times
                // the launches for the same work.
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
            // SAFETY: gdr_out/norm/z/out valid on ctx.stream; per-head layout from config.
            // The kernel launches exactly `num_heads` blocks, each normalizing one
            // flat `[val_dim]` slice at `blockIdx.x * val_dim` — gdr_out/z are
            // `[rows, Vh*Vd]` row-major, so the grid must cover all
            // rows*Vh (token, head) slices, not just token 0. `weight[tid]`
            // is a per-[Vd] broadcast (no blockIdx dependence), so the
            // extension is exact (the monolith's `rms_norm_gated_batch_into`
            // passed `seq_len * num_heads` identically).
            crate::profile::profile_op(&self.ctx, "linear/norm", Some(linear_idx), rows, || {
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    ffi::rms_norm_gated_cuda(
                        x_ptr as *const ffi::Half,
                        w_ptr as *const f32,
                        gate_ptr as *const ffi::Half,
                        o_ptr as *mut ffi::Half,
                        (self.local_linear_v_heads * rows) as i32,
                        c.linear_value_head_dim as i32,
                        c.rms_norm_eps,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
                Ok(())
            })?;
        }

        crate::profile::profile_op(&self.ctx, "linear/out_proj", Some(linear_idx), rows, || {
            gemm_batch(&self.ctx, &attn.out_proj, normed_out, out)
        })?;
        // Row-parallel out_proj: ONE all-reduce over the exact `[hidden, rows]`
        // buffer (no-op single-GPU).
        crate::profile::profile_op(
            &self.ctx,
            "linear/allreduce",
            Some(linear_idx),
            rows,
            || self.tp.all_reduce_sum(&self.ctx, out),
        )?;
        Ok(())
    }

    /// [`Self::advance_linear_conv_gdr`] for B rows of the SAME `len` in one
    /// conv + one GDR launch, through the varlen kernels
    /// [`Self::replay_linear_only_batched`] already uses. Uniform `len` makes
    /// their `s * len` row stride identical to the trunk's ragged packing, so
    /// the shared scratch needs no repack.
    ///
    /// The recurrent lane, not FlashQLA: a verify chain fits in one FlashQLA
    /// chunk, so chunk parallelism buys nothing against B times the launches.
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
        // Five contiguous B-entry tables in one upload, in the order the
        // kernels take them: conv x, conv ring, b, a, GDR state.
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
        // Same for every layer of a tick, so upload only when the tick's shape
        // changes. `get` zero-fills a resize, so both dims must be checked.
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
    /// [`Self::linear_attention`], factored out so the partial-accept replay
    /// re-runs the IDENTICAL kernel dispatch (same conv1d + same
    /// recurrent/chunked GDR branch, same inputs) — guaranteeing the conv and
    /// recurrent state advance is byte-identical between the trunk forward and
    /// the replay.
    ///
    /// `qkv_in` is the post-in_proj fused `[q|k|v]` PRE-conv1d (`qkv_dim` wide,
    /// token-major); conv1d reads it and writes the SEPARATE `qkv_conv`, which
    /// the GDR then consumes. `b_in`/`a_in` are the `in_proj_b`/`in_proj_a`
    /// gate projections. Every view spans EXACTLY this slot's `seq_len` rows —
    /// in a ragged batch that is a column-range slice of the shared scratch, so
    /// each slot's state sees only its own tokens.
    /// `gdr_out` is written but discarded by the replay (only the state
    /// side-effect matters); the trunk path norms it.
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
    ) -> Result<()> {
        let c = &self.config;
        let qkv_dim = self.local_linear_qkv_dim();
        let z_dim = self.local_linear_z_dim();

        let conv_state = &mut slot.conv_states[linear_idx];
        ensure!(
            conv_state.len == qkv_dim * (c.linear_conv_kernel_dim - 1),
            "Qwen3.5 conv state len {} != qkv_dim*(kernel-1) {}",
            conv_state.len,
            qkv_dim * (c.linear_conv_kernel_dim - 1)
        );
        {
            #[cfg(test)]
            let conv_capture = conv_probe::begin(
                &self.ctx,
                linear_idx,
                seq_len,
                qkv_dim,
                c.linear_conv_kernel_dim,
                qkv_in,
                &attn.conv1d_weight,
                conv_state,
            )?;
            {
                let (x_ptr, _g0) = qkv_in.device_ptr(&self.ctx.stream);
                let (w_ptr, _g1) = attn.conv1d_weight.data.device_ptr(&self.ctx.stream);
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
                        // SAFETY: ptrs from live device allocations sized to the dims passed.
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
            #[cfg(test)]
            conv_probe::finish(&self.ctx, conv_capture, qkv_conv, conv_state)?;
        }

        // Gated-delta rule. Decode (seq_len==1) is always the recurrent
        // kernel. Prefill chunks default to the recurrent kernel; the
        // FlashQLA chunked path (--qwen35-gdr-chunked) replaces the
        // serial token scan with chunk-parallel TileLang kernels, one AOT
        // instantiation per (Hg, H) geometry — unknown geometry falls back
        // to recurrent. The legacy in-tree chunkwise TileLang path stays
        // dead (sm_90 hang was in ITS kernels).
        let fq_fns: Option<(ffi::FqCumsumFn, ffi::FqKktFn, ffi::FqFwdFn)> =
            match (self.local_linear_k_heads, self.local_linear_v_heads) {
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
            && qwen35_gdr_chunked_enabled()
            && c.linear_key_head_dim == 128
            && c.linear_value_head_dim == 128
            && fq_fns.is_some_and(|(cumsum, _, _)| fq_kernels_available(&self.ctx, cumsum));
        if use_fq_chunked {
            // The AOT dispatch wrapper resolves SM + module via the DRIVER
            // context of the calling thread; the engine forward thread is not
            // guaranteed to have one bound (runtime-API kernels don't need
            // it), which made the fq path a per-thread lottery returning
            // NOT_SUPPORTED. Bind explicitly.
            self.ctx
                .ctx
                .bind_to_thread()
                .map_err(|e| anyhow!("bind CUDA context for chunked GDR failed: {e}"))?;
            let (fq_cumsum, fq_kkt, fq_fwd) = fq_fns.unwrap();
            let fq_parity = std::env::var_os("ARLE_FQ_PARITY").is_some();
            let mut fq_par_snap: Option<cudarc::driver::CudaSlice<f32>> = None;
            if fq_parity {
                let st = &slot.gdr_states[linear_idx];
                let mut snap = self.ctx.stream.alloc_zeros::<f32>(st.len())?;
                self.ctx.stream.memcpy_dtod(st, &mut snap)?;
                fq_par_snap = Some(snap);
            }
            let hg_dim = self.local_linear_k_heads * c.linear_key_head_dim;
            let fq_q = fq_q.get(&self.ctx, hg_dim, seq_len)?;
            let fq_k = fq_k.get(&self.ctx, hg_dim, seq_len)?;
            let fq_v = fq_v.get(&self.ctx, z_dim, seq_len)?;
            let fq_a = fq_a.get(&self.ctx, self.local_linear_v_heads * 64, seq_len)?;
            let g_len = self.local_linear_v_heads * seq_len;
            let fq_g = fq_g.get(&self.ctx, g_len)?;
            let fq_g_cumsum = fq_g_cumsum.get(&self.ctx, g_len)?;
            let fq_beta = fq_beta.get(&self.ctx, g_len)?;
            let gdr_state = &mut slot.gdr_states[linear_idx];

            let (qkv_ptr, _g0) = qkv_conv.device_ptr(&self.ctx.stream);
            let (b_ptr, _g1) = b_in.device_ptr(&self.ctx.stream);
            let (a_ptr, _g2) = a_in.device_ptr(&self.ctx.stream);
            let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
            let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
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
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                            self.local_linear_k_heads as i32,
                            self.local_linear_v_heads as i32,
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
            if let Some(mut snap) = fq_par_snap {
                drop(_g12);
                drop(_g13);
                let o_ref = DeviceVec::zeros(&self.ctx, seq_len * z_dim)?;
                {
                    let (qkv_ptr, _g0) = qkv_conv.device_ptr(&self.ctx.stream);
                    let (b_ptr, _g1) = b_in.device_ptr(&self.ctx.stream);
                    let (a_ptr, _g2) = a_in.device_ptr(&self.ctx.stream);
                    let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
                    let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
                    let (s_ptr, _g5) = snap.device_ptr_mut(&self.ctx.stream);
                    let (o_ptr, _g6) = o_ref.data.device_ptr(&self.ctx.stream);
                    // SAFETY: same buffers/dims as the fq calls above.
                    unsafe {
                        ffi::gated_delta_rule_prefill_recurrent_cuda(
                            qkv_ptr as *const ffi::Half,
                            b_ptr as *const ffi::Half,
                            a_ptr as *const ffi::Half,
                            dt_ptr as *const ffi::Half,
                            alog_ptr as *const f32,
                            s_ptr as *mut f32,
                            o_ptr as *mut ffi::Half,
                            self.local_linear_k_heads as i32,
                            self.local_linear_v_heads as i32,
                            c.linear_key_head_dim as i32,
                            c.linear_value_head_dim as i32,
                            seq_len as i32,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                }
                let a = self.ctx.stream.clone_dtoh(&*gdr_state)?;
                let b = self.ctx.stream.clone_dtoh(&snap)?;
                let oc = self.ctx.stream.clone_dtoh(&*gdr_out)?;
                let orf = self.ctx.stream.clone_dtoh(&o_ref.data)?;
                let state_actual = a.iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
                let state_reference = b.iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
                let output_actual = oc
                    .iter()
                    .map(|value| f64::from(value.to_f32()))
                    .collect::<Vec<_>>();
                let output_reference = orf
                    .iter()
                    .map(|value| f64::from(value.to_f32()))
                    .collect::<Vec<_>>();
                eprintln!(
                    "[fq-parity] layer={linear_idx} seq={seq_len} state_{} output_{}",
                    format_fq_parity_metrics(&state_reference, &state_actual, 1e-9),
                    format_fq_parity_metrics(&output_reference, &output_actual, 1e-6),
                );
            }
        }
        if !use_fq_chunked {
            let gdr_state = &mut slot.gdr_states[linear_idx];
            #[cfg(test)]
            let gdr_capture = gdr_probe::begin(
                &self.ctx,
                linear_idx,
                seq_len,
                self.local_linear_k_heads,
                self.local_linear_v_heads,
                c.linear_key_head_dim,
                c.linear_value_head_dim,
                qkv_conv,
                b_in,
                a_in,
                &attn.dt_bias,
                &attn.a_log,
                gdr_state,
            )?;
            {
                let (qkv_ptr, _g0) = qkv_conv.device_ptr(&self.ctx.stream);
                let (b_ptr, _g1) = b_in.device_ptr(&self.ctx.stream);
                let (a_ptr, _g2) = a_in.device_ptr(&self.ctx.stream);
                let (dt_ptr, _g3) = attn.dt_bias.data.device_ptr(&self.ctx.stream);
                let (alog_ptr, _g4) = attn.a_log.device_ptr(&self.ctx.stream);
                let (s_ptr, _g5) = gdr_state.device_ptr_mut(&self.ctx.stream);
                let (o_ptr, _g6) = gdr_out.device_ptr_mut(&self.ctx.stream);
                crate::profile::profile_op(
                    &self.ctx,
                    "linear/gdr_recurrent",
                    Some(linear_idx),
                    seq_len,
                    || {
                        // SAFETY: all buffers valid on ctx.stream; head dims from config.
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
                                    self.local_linear_k_heads as i32,
                                    self.local_linear_v_heads as i32,
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
                                    self.local_linear_k_heads as i32,
                                    self.local_linear_v_heads as i32,
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
            #[cfg(test)]
            gdr_probe::finish(&self.ctx, gdr_capture, gdr_out, gdr_state)?;
        }
        Ok(())
    }

    /// Partial-accept linear-only replay: advance ONLY the 48 gated-delta
    /// recurrent + conv states over the accepted prefix `[pending, d1..dk]`
    /// (`k+1` rows) from the verify capture, leaving them byte-identical to the
    /// old full-trunk `forward_hidden` replay at a tiny fraction of its cost.
    ///
    /// Precondition: the caller has just `restore_trunk`-ed the conv + gdr
    /// states to `S_{start_pos}` (the pre-verify snapshot), so re-running the
    /// first `k+1` recurrent steps reproduces the verify's `S_{start_pos+k+1}`.
    /// This re-uses the SAME [`Self::advance_linear_conv_gdr`] dispatch the
    /// verify forward used, fed the captured per-layer inputs — same conv1d +
    /// same recurrent/chunked GDR branch + same in-place accumulation, so the
    /// result is bit-for-bit the verify's first `k+1` steps. The
    /// recurrent-vs-chunked branch is re-selected from `seq_len = k+1`
    /// IDENTICALLY to how the old `forward_hidden` replay's `linear_attention`
    /// selected it (preserving any k==0 decode-vs-chunked path choice).
    ///
    /// Mutated device buffers (full enumeration):
    ///   - `slot.conv_states[li]` for every linear layer `li` — advanced k+1
    ///     steps from the restored S_start (conv1d ring shift, content-based).
    ///   - `slot.gdr_states[li]` for every linear layer `li` — advanced k+1
    ///     steps from the restored S_start (recurrent in-place).
    ///   - `ws.linear` scratch (`qkv_conv`/`gdr_out`/`fq_*`) — fully overwritten
    ///     per layer before read; transient, no persistent meaning.
    ///
    /// Explicitly NOT touched: the 16 full-attn KV caches (self-heal via the
    /// caller's `set_seq_len` position rewind), `slot.seq_len` (caller sets it),
    /// MLP/MoE/lm_head (no persistent state). Numerics-only; no H2D/D2H/sync.
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
        // the accepted prefix is the leading `rows = k+1` columns.
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
    /// slot keeps its own capture and state, reached through `tables`. The
    /// launches are sub-100 µs, so the win is their count.
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

    /// Batched-decode full attention: batched q/k/v projections over all B
    /// rows, then one split-KV fused decode launch over grid.z = B. The kernel
    /// reads per-row positions / seq_lens and per-row K/V cache pointers from
    /// device arrays, so the host never loops over rows for full-attn decode.
    /// head_dim != 256 keeps the per-row path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn full_attention_batch_rows(
        &self,
        attn: &FullAttn,
        normed: &HiddenStates,
        slots: &mut [Qwen35SlotState],
        slot_indices: &[usize],
        full_idx: usize,
        positions_dev: &CudaSlice<i32>,
        seq_lens_dev: &CudaSlice<i32>,
        k_cache_table: &CudaSlice<u64>,
        v_cache_table: &CudaSlice<u64>,
        fw: &mut FullAttnScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        let c = &self.config;
        let b = normed.seq_len;
        let q_dim = self.local_full_attn_q_dim();
        let kv_dim = self.local_full_attn_kv_dim();
        let q_proj_dim = self.local_full_attn_q_proj_dim();
        let sm_scale = 1.0f32 / (c.head_dim as f32).sqrt();

        let FullAttnScratch {
            qkv_fused,
            q_full,
            k_batch,
            v_batch,
            q_prepped,
            attn_heads,
            fa3_lse: _,
            fa3_oaccum: _,
            fa3_lseaccum: _,
            fa3_semaphore: _,
            batch_partial_out,
            batch_partial_m,
            batch_partial_l,
        } = fw;
        let qkv_fused = qkv_fused.get(&self.ctx, q_proj_dim + 2 * kv_dim, b)?;
        let q_full = q_full.get(&self.ctx, q_proj_dim, b)?;
        let k_batch = k_batch.get(&self.ctx, kv_dim, b)?;
        let v_batch = v_batch.get(&self.ctx, kv_dim, b)?;
        gemm_batch(&self.ctx, &attn.qkv_proj, normed, qkv_fused)?;
        split_qkv(&self.ctx, qkv_fused, q_full, k_batch, v_batch)?;

        let attn_heads = attn_heads.get(&self.ctx, q_dim, b)?;

        if c.head_dim == 256 {
            ensure!(
                self.local_kv_heads > 0 && self.local_q_heads.is_multiple_of(self.local_kv_heads),
                "Qwen3.5 batched decode full-attn requires integral GQA ratio: q_heads={} kv_heads={}",
                self.local_q_heads,
                self.local_kv_heads
            );
            let partial_scalars = b * self.local_q_heads * QWEN35_BATCHED_DECODE_KV_SPLITS;
            let partial_out = batch_partial_out.get(&self.ctx, partial_scalars * c.head_dim)?;
            let partial_m = batch_partial_m.get(&self.ctx, partial_scalars)?;
            let partial_l = batch_partial_l.get(&self.ctx, partial_scalars)?;
            let (qf_base, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (k_base, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_base, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (pos_ptr, _g7) = positions_dev.device_ptr(&self.ctx.stream);
            let (seq_ptr, _g8) = seq_lens_dev.device_ptr(&self.ctx.stream);
            let (ktbl_ptr, _g9) = k_cache_table.device_ptr(&self.ctx.stream);
            let (vtbl_ptr, _g10) = v_cache_table.device_ptr(&self.ctx.stream);
            let (po_ptr, _g11) = partial_out.device_ptr_mut(&self.ctx.stream);
            let (pm_ptr, _g12) = partial_m.device_ptr_mut(&self.ctx.stream);
            let (pl_ptr, _g13) = partial_l.device_ptr_mut(&self.ctx.stream);
            let (ao_base, _g14) = attn_heads.data.device_ptr_mut(&self.ctx.stream);
            // SAFETY: q/k/v projections and output are live `[B, *]` buffers;
            // positions/seq_lens are live `[B]` i32 device arrays; K/V pointer
            // tables hold the first B live per-slot caches staged for this
            // row→slot mapping. The fused kernel writes one new K/V row per
            // batch item and all partial slots; reduce writes every attn_heads
            // element before gate reads it.
            unsafe {
                ffi::fused_gqa_attention_decode_batched(
                    qf_base as *const ffi::Half,
                    k_base as *const ffi::Half,
                    v_base as *const ffi::Half,
                    qn_ptr as *const ffi::Half,
                    kn_ptr as *const ffi::Half,
                    cos_ptr as *const ffi::Half,
                    sin_ptr as *const ffi::Half,
                    pos_ptr as *const i32,
                    seq_ptr as *const i32,
                    ktbl_ptr as *const *const ffi::Half,
                    vtbl_ptr as *const *const ffi::Half,
                    po_ptr as *mut f32,
                    pm_ptr as *mut f32,
                    pl_ptr as *mut f32,
                    self.local_q_heads as i32,
                    self.local_kv_heads as i32,
                    (self.local_q_heads / self.local_kv_heads) as i32,
                    c.head_dim as i32,
                    c.rotary_dim as i32,
                    self.max_seq_len as i32,
                    b as i32,
                    c.rms_norm_eps,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
                ffi::attention_decode_reduce_batched(
                    po_ptr as *const f32,
                    pm_ptr as *const f32,
                    pl_ptr as *const f32,
                    ao_base as *mut ffi::Half,
                    self.local_q_heads as i32,
                    c.head_dim as i32,
                    b as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
                ffi::attention_gate_batch_hd256_cuda(
                    qf_base as *const ffi::Half,
                    ao_base as *mut ffi::Half,
                    self.local_q_heads as i32,
                    c.head_dim as i32,
                    b as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()?;
            }
        } else {
            let q_prepped = q_prepped.get(&self.ctx, q_dim, b)?;
            let (qf_base, _g0) = q_full.data.device_ptr(&self.ctx.stream);
            let (k_base, _g1) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_base, _g2) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _g3) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _g4) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _g5) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _g6) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (qp_base, _g7) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (ao_base, _g8) = attn_heads.data.device_ptr_mut(&self.ctx.stream);
            let (pos_base, _g9) = positions_dev.device_ptr(&self.ctx.stream);

            for (r, &si) in slot_indices.iter().enumerate() {
                let slot = &mut slots[si];
                let k_cache = &mut slot.k_caches[full_idx];
                let v_cache = &mut slot.v_caches[full_idx];
                let max_seq_len = k_cache.len / kv_dim;
                let (kc_ptr, _gk) = k_cache.data.device_ptr_mut(&self.ctx.stream);
                let (vc_ptr, _gv) = v_cache.data.device_ptr_mut(&self.ctx.stream);
                let qf_r = qf_base + (r * q_proj_dim * 2) as u64;
                let k_r = k_base + (r * kv_dim * 2) as u64;
                let v_r = v_base + (r * kv_dim * 2) as u64;
                let qp_r = qp_base + (r * q_dim * 2) as u64;
                let ao_r = ao_base + (r * q_dim * 2) as u64;
                let pos_r = pos_base + (r * 4) as u64;
                // SAFETY: every pointer is a live device allocation on
                // ctx.stream; offsets stay inside the `[*, B]` buffers for
                // r < B; each kernel runs at seq_len == 1 so it touches only
                // row r's block + slot r's caches.
                unsafe {
                    ffi::prefill_attention_hd256_prep_cuda(
                        qf_r as *const ffi::Half,
                        k_r as *const ffi::Half,
                        v_r as *const ffi::Half,
                        qn_ptr as *const ffi::Half,
                        kn_ptr as *const ffi::Half,
                        cos_ptr as *const ffi::Half,
                        sin_ptr as *const ffi::Half,
                        qp_r as *mut ffi::Half,
                        kc_ptr as *mut ffi::Half,
                        vc_ptr as *mut ffi::Half,
                        self.local_q_heads as i32,
                        self.local_kv_heads as i32,
                        c.head_dim as i32,
                        1,
                        pos_r as *const i32,
                        c.rotary_dim as i32,
                        c.rms_norm_eps,
                        max_seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                    ffi::nonpaged_prefill_attention_devpos_cuda(
                        qp_r as *const ffi::Half,
                        kc_ptr as *const ffi::Half,
                        vc_ptr as *const ffi::Half,
                        ao_r as *mut ffi::Half,
                        self.local_q_heads as i32,
                        self.local_kv_heads as i32,
                        c.head_dim as i32,
                        1,
                        pos_r as *const i32,
                        max_seq_len as i32,
                        sm_scale,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                    ffi::attention_gate_batch_hd256_cuda(
                        qf_r as *const ffi::Half,
                        ao_r as *mut ffi::Half,
                        self.local_q_heads as i32,
                        c.head_dim as i32,
                        1,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }

        gemm_batch(&self.ctx, &attn.o_proj, attn_heads, out)?;
        // Row-parallel o_proj: ONE all-reduce over the exact `[hidden, B]`
        // buffer — message length is B valid columns by construction (no-op
        // single-GPU).
        self.tp.all_reduce_sum(&self.ctx, out)?;
        Ok(())
    }
}
