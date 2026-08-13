use super::*;

// --- Context-parallel ring attention (Track A device path) ---
// One-block-pure kernels + on-device flash-2 merge; the ring rotation and tape
// live in ops/ring_attention.rs. q/k/v arrive as f32 handles (tape tensors),
// converted to bf16 for the kernel (matching the training activation precision);
// the (m,l,o) accumulators and grads stay f32 for a stable merge.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn ring_i32(v: usize, label: &'static str) -> Result<i32> {
    i32::try_from(v).map_err(|_| AutogradError::TapeInvariant(label))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_ring_block_fwd_merge(
    backend: &CudaBackend,
    q: &DeviceHandle,
    k_blk: &DeviceHandle,
    v_blk: &DeviceHandle,
    acc_m: &DeviceHandle,
    acc_l: &DeviceHandle,
    acc_o: &DeviceHandle,
    q_pos: &DeviceHandle,
    k_pos: &DeviceHandle,
    q_pos_host: &[usize],
    k_pos_host: &[usize],
    dims: RingBlockDims,
) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
    let q_slice = backend.cuda_slice(q, "ring q")?;
    let k_slice = backend.cuda_slice(k_blk, "ring k")?;
    let v_slice = backend.cuda_slice(v_blk, "ring v")?;
    let q_bf16 = backend.local_f32_as_bf16(q_slice, q_slice.len())?;
    let k_bf16 = backend.local_f32_as_bf16(k_slice, k_slice.len())?;
    let v_bf16 = backend.local_f32_as_bf16(v_slice, v_slice.len())?;
    let m_in = backend.cuda_slice(acc_m, "ring acc_m")?;
    let l_in = backend.cuda_slice(acc_l, "ring acc_l")?;
    let o_in = backend.cuda_slice(acc_o, "ring acc_o")?;
    if ring_fa3_route(backend, dims, q_pos_host, k_pos_host) {
        return cuda_ring_block_fwd_merge_fa3(
            backend, &q_bf16, &k_bf16, &v_bf16, m_in, l_in, o_in, q_pos_host, k_pos_host, dims,
        );
    }
    let qpos_slice = backend.cuda_slice(q_pos, "ring q_pos")?;
    let kpos_slice = backend.cuda_slice(k_pos, "ring k_pos")?;
    let rows = dims.num_q_tiles * dims.q_rows;
    let mut m_out = backend
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|_| cuda_alloc_failed("ring m_out", vec![rows]))?;
    let mut l_out = backend
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|_| cuda_alloc_failed("ring l_out", vec![rows]))?;
    let mut o_out = backend
        .stream
        .alloc_zeros::<f32>(rows * dims.head_dim)
        .map_err(|_| cuda_alloc_failed("ring o_out", vec![rows * dims.head_dim]))?;
    {
        let (q_ptr, _qg) = q_bf16.device_ptr(&backend.stream);
        let (k_ptr, _kg) = k_bf16.device_ptr(&backend.stream);
        let (v_ptr, _vg) = v_bf16.device_ptr(&backend.stream);
        let (mi_ptr, _mig) = m_in.device_ptr(&backend.stream);
        let (li_ptr, _lig) = l_in.device_ptr(&backend.stream);
        let (oi_ptr, _oig) = o_in.device_ptr(&backend.stream);
        let (mo_ptr, _mog) = m_out.device_ptr_mut(&backend.stream);
        let (lo_ptr, _log) = l_out.device_ptr_mut(&backend.stream);
        let (oo_ptr, _oog) = o_out.device_ptr_mut(&backend.stream);
        let (qpos_ptr, _qpg) = qpos_slice.device_ptr(&backend.stream);
        let (kpos_ptr, _kpg) = kpos_slice.device_ptr(&backend.stream);
        check_cuda_ffi(
            // SAFETY: q/k/v are live bf16 copies; acc_*_in are f32 handles of the right
            // length; *_out are freshly allocated rows / rows*hd; q_pos/k_pos are f32
            // handles of q_rows / blk_len; dims mirror the shapes.
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
                    ring_i32(dims.num_q_tiles, "ring num_q_tiles i32")?,
                    ring_i32(dims.num_q_heads, "ring num_q_heads i32")?,
                    ring_i32(dims.num_kv_heads, "ring num_kv_heads i32")?,
                    ring_i32(dims.head_dim, "ring head_dim i32")?,
                    ring_i32(dims.q_rows, "ring q_rows i32")?,
                    ring_i32(dims.blk_len, "ring blk_len i32")?,
                    dims.sm_scale,
                    backend.stream.cu_stream(),
                )
            },
            "ring_block_attention_fwd_merge_cuda",
        )?;
    }
    Ok((
        DeviceHandle::Cuda(CudaStorage::new(m_out)),
        DeviceHandle::Cuda(CudaStorage::new(l_out)),
        DeviceHandle::Cuda(CudaStorage::new(o_out)),
    ))
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_ring_block_finalize(
    backend: &CudaBackend,
    acc_m: &DeviceHandle,
    acc_l: &DeviceHandle,
    acc_o: &DeviceHandle,
    total_rows: usize,
    head_dim: usize,
) -> Result<(DeviceHandle, DeviceHandle)> {
    let m_slice = backend.cuda_slice(acc_m, "ring fin m")?;
    let l_slice = backend.cuda_slice(acc_l, "ring fin l")?;
    let o_slice = backend.cuda_slice(acc_o, "ring fin o")?;
    let out_len = total_rows * head_dim;
    let mut out_f32 = backend
        .stream
        .alloc_zeros::<f32>(out_len)
        .map_err(|_| cuda_alloc_failed("ring finalize out", vec![out_len]))?;
    let mut lse = backend
        .stream
        .alloc_zeros::<f32>(total_rows)
        .map_err(|_| cuda_alloc_failed("ring finalize lse", vec![total_rows]))?;
    {
        let (m_ptr, _mg) = m_slice.device_ptr(&backend.stream);
        let (l_ptr, _lg) = l_slice.device_ptr(&backend.stream);
        let (o_ptr, _og) = o_slice.device_ptr(&backend.stream);
        let (out_ptr, _outg) = out_f32.device_ptr_mut(&backend.stream);
        let (lse_ptr, _lseg) = lse.device_ptr_mut(&backend.stream);
        check_cuda_ffi(
            // SAFETY: acc_* are f32 handles length rows / rows*hd; out is rows*hd f32;
            // lse is rows f32; dims mirror the shapes.
            unsafe {
                ffi::ring_block_attention_finalize_cuda(
                    m_ptr as *const f32,
                    l_ptr as *const f32,
                    o_ptr as *const f32,
                    out_ptr as *mut f32,
                    lse_ptr as *mut f32,
                    ring_i32(total_rows, "ring total_rows i32")?,
                    ring_i32(head_dim, "ring head_dim i32")?,
                    backend.stream.cu_stream(),
                )
            },
            "ring_block_attention_finalize_cuda",
        )?;
    }
    Ok((
        DeviceHandle::Cuda(CudaStorage::new(out_f32)),
        DeviceHandle::Cuda(CudaStorage::new(lse)),
    ))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_ring_block_bwd(
    backend: &CudaBackend,
    q: &DeviceHandle,
    k_blk: &DeviceHandle,
    v_blk: &DeviceHandle,
    out: &DeviceHandle,
    lse: &DeviceHandle,
    d_out: &DeviceHandle,
    grad_q: &DeviceHandle,
    q_pos: &DeviceHandle,
    k_pos: &DeviceHandle,
    q_pos_host: &[usize],
    k_pos_host: &[usize],
    dims: RingBlockDims,
) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
    let q_slice = backend.cuda_slice(q, "ring bwd q")?;
    let k_slice = backend.cuda_slice(k_blk, "ring bwd k")?;
    let v_slice = backend.cuda_slice(v_blk, "ring bwd v")?;
    let out_slice = backend.cuda_slice(out, "ring bwd out")?;
    let lse_slice = backend.cuda_slice(lse, "ring bwd lse")?;
    let do_slice = backend.cuda_slice(d_out, "ring bwd d_out")?;
    let gq_slice = backend.cuda_slice(grad_q, "ring bwd grad_q")?;
    let q_bf16 = backend.local_f32_as_bf16(q_slice, q_slice.len())?;
    let k_bf16 = backend.local_f32_as_bf16(k_slice, k_slice.len())?;
    let v_bf16 = backend.local_f32_as_bf16(v_slice, v_slice.len())?;
    let do_bf16 = backend.local_f32_as_bf16(do_slice, do_slice.len())?;
    // grad_q is in/out: copy the running accumulator so the kernel's += lands on a
    // fresh handle (the tape input stays immutable).
    let mut gq_out = backend
        .stream
        .alloc_zeros::<f32>(gq_slice.len())
        .map_err(|_| cuda_alloc_failed("ring grad_q out", vec![gq_slice.len()]))?;
    backend
        .stream
        .memcpy_dtod(gq_slice, &mut gq_out)
        .map_err(|_| AutogradError::TapeInvariant("cuda D2D copy failed (ring grad_q carry)"))?;
    let blk_elems = dims.num_q_tiles / (dims.num_q_heads / dims.num_kv_heads).max(1)
        * dims.blk_len
        * dims.head_dim;
    let mut gk = backend
        .stream
        .alloc_zeros::<f32>(blk_elems)
        .map_err(|_| cuda_alloc_failed("ring grad_k", vec![blk_elems]))?;
    let mut gv = backend
        .stream
        .alloc_zeros::<f32>(blk_elems)
        .map_err(|_| cuda_alloc_failed("ring grad_v", vec![blk_elems]))?;
    if ring_fa3_route(backend, dims, q_pos_host, k_pos_host) {
        return cuda_ring_block_bwd_fa3(
            backend, &q_bf16, &k_bf16, &v_bf16, out_slice, lse_slice, &do_bf16, gq_out, gk, gv,
            q_pos_host, k_pos_host, dims,
        );
    }
    let qpos_slice = backend.cuda_slice(q_pos, "ring bwd q_pos")?;
    let kpos_slice = backend.cuda_slice(k_pos, "ring bwd k_pos")?;
    {
        let (q_ptr, _qg) = q_bf16.device_ptr(&backend.stream);
        let (k_ptr, _kg) = k_bf16.device_ptr(&backend.stream);
        let (v_ptr, _vg) = v_bf16.device_ptr(&backend.stream);
        let (out_ptr, _og) = out_slice.device_ptr(&backend.stream);
        let (lse_ptr, _lg) = lse_slice.device_ptr(&backend.stream);
        let (do_ptr, _dg) = do_bf16.device_ptr(&backend.stream);
        let (gq_ptr, _gqg) = gq_out.device_ptr_mut(&backend.stream);
        let (gk_ptr, _gkg) = gk.device_ptr_mut(&backend.stream);
        let (gv_ptr, _gvg) = gv.device_ptr_mut(&backend.stream);
        let (qpos_ptr, _qpg) = qpos_slice.device_ptr(&backend.stream);
        let (kpos_ptr, _kpg) = kpos_slice.device_ptr(&backend.stream);
        check_cuda_ffi(
            // SAFETY: bf16 copies + f32 out/lse/grad handles of matching length; gk/gv
            // sized to one block's [Tkv, blk_len, hd]; q_pos/k_pos f32 of q_rows/blk_len;
            // dims mirror the shapes.
            unsafe {
                ffi::ring_block_attention_bwd_cuda(
                    q_ptr as *const ffi::Half,
                    k_ptr as *const ffi::Half,
                    v_ptr as *const ffi::Half,
                    out_ptr as *const f32,
                    lse_ptr as *const f32,
                    do_ptr as *const ffi::Half,
                    gq_ptr as *mut f32,
                    gk_ptr as *mut f32,
                    gv_ptr as *mut f32,
                    qpos_ptr as *const f32,
                    kpos_ptr as *const f32,
                    ring_i32(dims.num_q_tiles, "ring bwd num_q_tiles i32")?,
                    ring_i32(dims.num_q_heads, "ring bwd num_q_heads i32")?,
                    ring_i32(dims.num_kv_heads, "ring bwd num_kv_heads i32")?,
                    ring_i32(dims.head_dim, "ring bwd head_dim i32")?,
                    ring_i32(dims.q_rows, "ring bwd q_rows i32")?,
                    ring_i32(dims.blk_len, "ring bwd blk_len i32")?,
                    dims.sm_scale,
                    backend.stream.cu_stream(),
                )
            },
            "ring_block_attention_bwd_cuda",
        )?;
    }
    Ok((
        DeviceHandle::Cuda(CudaStorage::new(gq_out)),
        DeviceHandle::Cuda(CudaStorage::new(gk)),
        DeviceHandle::Cuda(CudaStorage::new(gv)),
    ))
}

// --- CP ring FA3 pair route (hd256 / sm_90) ---
// Decomposes the (q_pos, k_pos) block into rectangular FA3 calls per the run
// classification in ops/ring_attention.rs: each visible (q_run, k_run) pair is
// one non-varlen batch=1 FA3 launch over strided head-major views of the
// tile-major [tiles, seq, d] bf16 buffers (head_stride = seq*d, row_stride =
// d, run pointer offset = run_row*d). Forward merges each pair's normalized
// (o, lse) into the running accumulators as block stats (m = lse, l = 1);
// backward mirrors ring-flash-attention's global-LSE trick (final lse + final
// o + upstream d_out, per-pair causal flag). Scalar kernels remain the
// non-sm90 / non-hd256 fallback.

#[cfg(not(feature = "no-cuda"))]
pub(super) fn ring_fa3_route(
    backend: &CudaBackend,
    dims: RingBlockDims,
    q_pos: &[usize],
    k_pos: &[usize],
) -> bool {
    let shape_ok = dims.head_dim == 256
        && q_pos.len() == dims.q_rows
        && k_pos.len() == dims.blk_len
        && ring_fa3_is_sm90(backend);
    // SAFETY: pure host query exported by both the real shim and the stub.
    let real = unsafe { ffi::arle_fa3_real_kernel_marker_cuda() } == 1;
    if shape_ok && !real {
        // A stub build falls back silently at ~1/3 the speed, and such a run has
        // already been mistaken for an FA3 measurement.
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "[autograd] FA3 ring shape qualifies but the real kernel is absent — running \
                 scalar. Rebuild for an sm_90 target with vendor/flash-attention/hopper present."
            );
        });
    }
    shape_ok && real
}

/// The vendored FA3 units are sm_90a-only — dispatch strictly on 9.0.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn ring_fa3_is_sm90(backend: &CudaBackend) -> bool {
    use cudarc::driver::sys::CUdevice_attribute as Attr;
    let ctx = backend.stream.context();
    ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .is_ok_and(|v| v == 9)
        && ctx
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .is_ok_and(|v| v == 0)
}

#[cfg(not(feature = "no-cuda"))]
pub(super) struct RingFa3Pair {
    q: crate::ops::ring_attention::PosRun,
    k: crate::ops::ring_attention::PosRun,
    causal: bool,
}

/// All visible (q_run, k_run) pairs, classified up front so a mis-aligned
/// shard errors loudly before any state is touched.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn ring_fa3_pairs(q_pos: &[usize], k_pos: &[usize]) -> Result<Vec<RingFa3Pair>> {
    use crate::ops::ring_attention::{PairClass, classify_pair, contiguous_pos_runs};
    let k_runs = contiguous_pos_runs(k_pos);
    let mut pairs = Vec::new();
    for q_run in contiguous_pos_runs(q_pos) {
        for &k_run in &k_runs {
            match classify_pair(q_run, k_run)? {
                PairClass::Full => pairs.push(RingFa3Pair {
                    q: q_run,
                    k: k_run,
                    causal: false,
                }),
                PairClass::Causal => pairs.push(RingFa3Pair {
                    q: q_run,
                    k: k_run,
                    causal: true,
                }),
                PairClass::Skip => {}
            }
        }
    }
    Ok(pairs)
}

/// Non-varlen causal metadata ceiling: `round_up(batch=1, 4) * 4 + 1` (see the
/// fwd shim's carve) — covers every pair class.
#[cfg(not(feature = "no-cuda"))]
pub(super) const RING_FA3_FWD_METADATA_I32: usize = 17;

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_ring_block_fwd_merge_fa3(
    backend: &CudaBackend,
    q_bf16: &CudaSlice<u16>,
    k_bf16: &CudaSlice<u16>,
    v_bf16: &CudaSlice<u16>,
    m_in: &CudaSlice<f32>,
    l_in: &CudaSlice<f32>,
    o_in: &CudaSlice<f32>,
    q_pos: &[usize],
    k_pos: &[usize],
    dims: RingBlockDims,
) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
    let pairs = ring_fa3_pairs(q_pos, k_pos)?;
    let (h, hk, d, s, blk) = (
        dims.num_q_heads,
        dims.num_kv_heads,
        dims.head_dim,
        dims.q_rows,
        dims.blk_len,
    );
    let b = dims.num_q_tiles / h.max(1);
    let rows = dims.num_q_tiles * s;
    // Fresh accumulator copies (functional like the scalar merge); each pair
    // then rescales only its run rows in place.
    let mut m_out = backend
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|_| cuda_alloc_failed("ring fa3 m_out", vec![rows]))?;
    let mut l_out = backend
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|_| cuda_alloc_failed("ring fa3 l_out", vec![rows]))?;
    let mut o_out = backend
        .stream
        .alloc_zeros::<f32>(rows * d)
        .map_err(|_| cuda_alloc_failed("ring fa3 o_out", vec![rows * d]))?;
    let carry = AutogradError::TapeInvariant("cuda D2D copy failed (ring fa3 acc carry)");
    backend
        .stream
        .memcpy_dtod(m_in, &mut m_out)
        .map_err(|_| carry.clone())?;
    backend
        .stream
        .memcpy_dtod(l_in, &mut l_out)
        .map_err(|_| carry.clone())?;
    backend
        .stream
        .memcpy_dtod(o_in, &mut o_out)
        .map_err(|_| carry)?;

    for pair in &pairs {
        let (lq, lk) = (pair.q.len, pair.k.len);
        // Per-pair scratch, reused across the batch loop (stream-ordered).
        let mut o_pair = backend
            .stream
            .alloc_zeros::<u16>(h * lq * d)
            .map_err(|_| cuda_alloc_failed("ring fa3 o_pair", vec![h, lq, d]))?;
        let mut lse_pair = backend
            .stream
            .alloc_zeros::<f32>(h * lq)
            .map_err(|_| cuda_alloc_failed("ring fa3 lse_pair", vec![h, lq]))?;
        let mut meta = backend
            .stream
            .alloc_zeros::<i32>(RING_FA3_FWD_METADATA_I32)
            .map_err(|_| cuda_alloc_failed("ring fa3 meta", vec![RING_FA3_FWD_METADATA_I32]))?;
        for bi in 0..b {
            {
                let (q_ptr, _qg) = q_bf16.device_ptr(&backend.stream);
                let (k_ptr, _kg) = k_bf16.device_ptr(&backend.stream);
                let (v_ptr, _vg) = v_bf16.device_ptr(&backend.stream);
                let (op_ptr, _og) = o_pair.device_ptr_mut(&backend.stream);
                let (lse_ptr, _lg) = lse_pair.device_ptr_mut(&backend.stream);
                let (meta_ptr, _mg) = meta.device_ptr_mut(&backend.stream);
                let args = ffi::ArleFa3FwdHd256Args {
                    q: (q_ptr + (((bi * h * s) + pair.q.row) * d * 2) as u64) as *const ffi::Half,
                    k: (k_ptr + (((bi * hk * blk) + pair.k.row) * d * 2) as u64)
                        as *const ffi::Half,
                    v: (v_ptr + (((bi * hk * blk) + pair.k.row) * d * 2) as u64)
                        as *const ffi::Half,
                    o: op_ptr as *mut ffi::Half,
                    softmax_lse: lse_ptr as *mut f32,
                    out_accum: std::ptr::null_mut(),
                    softmax_lse_accum: std::ptr::null_mut(),
                    tile_count_semaphore: meta_ptr as *mut i32,
                    metadata_capacity: RING_FA3_FWD_METADATA_I32 as i32,
                    cu_seqlens_q: std::ptr::null(),
                    seqused_k: std::ptr::null(),
                    batch: 1,
                    total_q: ring_i32(lq, "ring fa3 total_q i32")?,
                    seqlen_q: ring_i32(lq, "ring fa3 seqlen_q i32")?,
                    seqlen_k: ring_i32(lk, "ring fa3 seqlen_k i32")?,
                    num_heads: ring_i32(h, "ring fa3 num_heads i32")?,
                    num_heads_k: ring_i32(hk, "ring fa3 num_heads_k i32")?,
                    head_dim: 256,
                    q_row_stride: d as i64,
                    k_row_stride: d as i64,
                    v_row_stride: d as i64,
                    o_row_stride: d as i64,
                    q_head_stride: (s * d) as i64,
                    k_head_stride: (blk * d) as i64,
                    v_head_stride: (blk * d) as i64,
                    o_head_stride: (lq * d) as i64,
                    softmax_scale: dims.sm_scale,
                    is_causal: pair.causal as i32,
                    num_splits: 1,
                    page_table: std::ptr::null(),
                    page_table_batch_stride: 0,
                    page_size: 0,
                    num_pages: 0,
                    k_page_stride: 0,
                    v_page_stride: 0,
                };
                check_cuda_ffi(
                    // SAFETY: strided head-major views of live bf16 buffers; o/lse
                    // are compact [h, lq(, d)] scratch; meta covers the shim's carve.
                    unsafe { ffi::arle_fa3_fwd_hd256_bf16_cuda(&args, backend.stream.cu_stream()) },
                    "arle_fa3_fwd_hd256_bf16_cuda",
                )?;
            }
            let (mo_ptr, _mog) = m_out.device_ptr_mut(&backend.stream);
            let (lo_ptr, _log) = l_out.device_ptr_mut(&backend.stream);
            let (oo_ptr, _oog) = o_out.device_ptr_mut(&backend.stream);
            let (op_ptr, _og2) = o_pair.device_ptr(&backend.stream);
            let (lse_ptr, _lg2) = lse_pair.device_ptr(&backend.stream);
            check_cuda_ffi(
                // SAFETY: acc buffers are full [tiles, s(, d)]; pair buffers compact.
                unsafe {
                    ffi::ring_fa3_merge_pair_cuda(
                        mo_ptr as *mut f32,
                        lo_ptr as *mut f32,
                        oo_ptr as *mut f32,
                        lse_ptr as *const f32,
                        op_ptr as *const ffi::Half,
                        ring_i32(h, "ring fa3 merge heads i32")?,
                        ring_i32(bi * h, "ring fa3 merge tile_base i32")?,
                        ring_i32(s, "ring fa3 merge seq i32")?,
                        ring_i32(pair.q.row, "ring fa3 merge run_start i32")?,
                        ring_i32(lq, "ring fa3 merge run_len i32")?,
                        ring_i32(d, "ring fa3 merge head_dim i32")?,
                        backend.stream.cu_stream(),
                    )
                },
                "ring_fa3_merge_pair_cuda",
            )?;
        }
    }
    Ok((
        DeviceHandle::Cuda(CudaStorage::new(m_out)),
        DeviceHandle::Cuda(CudaStorage::new(l_out)),
        DeviceHandle::Cuda(CudaStorage::new(o_out)),
    ))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_ring_block_bwd_fa3(
    backend: &CudaBackend,
    q_bf16: &CudaSlice<u16>,
    k_bf16: &CudaSlice<u16>,
    v_bf16: &CudaSlice<u16>,
    out_slice: &CudaSlice<f32>,
    lse_slice: &CudaSlice<f32>,
    do_bf16: &CudaSlice<u16>,
    mut gq_out: CudaSlice<f32>,
    mut gk: CudaSlice<f32>,
    mut gv: CudaSlice<f32>,
    q_pos: &[usize],
    k_pos: &[usize],
    dims: RingBlockDims,
) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
    let pairs = ring_fa3_pairs(q_pos, k_pos)?;
    let (h, hk, d, s, blk) = (
        dims.num_q_heads,
        dims.num_kv_heads,
        dims.head_dim,
        dims.q_rows,
        dims.blk_len,
    );
    let b = dims.num_q_tiles / h.max(1);
    let gqa = h != hk;
    // FA3 bwd wants o/d_out bf16; o is saved f32 — one quantize copy per call
    // (d_out was converted by the caller).
    let o_bf16 = backend.local_f32_as_bf16(out_slice, out_slice.len())?;

    for pair in &pairs {
        let (lq, lk) = (pair.q.len, pair.k.len);
        // hd256 sm90 bwd tiles: kBlockM=64 / kBlockN=80 (see the bwd shim).
        let sq_r = lq.div_ceil(64) * 64;
        let sk_r = lk.div_ceil(80) * 80;
        let mut lse_c = backend
            .stream
            .alloc_zeros::<f32>(h * lq)
            .map_err(|_| cuda_alloc_failed("ring fa3 bwd lse", vec![h, lq]))?;
        let mut dq = backend
            .stream
            .alloc_zeros::<u16>(h * lq * d)
            .map_err(|_| cuda_alloc_failed("ring fa3 dq", vec![h, lq, d]))?;
        let mut dk = backend
            .stream
            .alloc_zeros::<u16>(hk * lk * d)
            .map_err(|_| cuda_alloc_failed("ring fa3 dk", vec![hk, lk, d]))?;
        let mut dv = backend
            .stream
            .alloc_zeros::<u16>(hk * lk * d)
            .map_err(|_| cuda_alloc_failed("ring fa3 dv", vec![hk, lk, d]))?;
        let mut softmax_d = backend
            .stream
            .alloc_zeros::<f32>(h * sq_r)
            .map_err(|_| cuda_alloc_failed("ring fa3 softmax_d", vec![h, sq_r]))?;
        let mut lse_log2 = backend
            .stream
            .alloc_zeros::<f32>(h * sq_r)
            .map_err(|_| cuda_alloc_failed("ring fa3 lse_log2", vec![h, sq_r]))?;
        let mut dq_accum = backend
            .stream
            .alloc_zeros::<f32>(h * sq_r * 256)
            .map_err(|_| cuda_alloc_failed("ring fa3 dq_accum", vec![h, sq_r, 256]))?;
        let mut dkv_accum = if gqa {
            let dk_a = backend
                .stream
                .alloc_zeros::<f32>(hk * sk_r * 256)
                .map_err(|_| cuda_alloc_failed("ring fa3 dk_accum", vec![hk, sk_r, 256]))?;
            let dv_a = backend
                .stream
                .alloc_zeros::<f32>(hk * sk_r * 256)
                .map_err(|_| cuda_alloc_failed("ring fa3 dv_accum", vec![hk, sk_r, 256]))?;
            Some((dk_a, dv_a))
        } else {
            None
        };
        let dq_sem_len = lq.div_ceil(64) * h;
        let mut dq_sem = backend
            .stream
            .alloc_zeros::<i32>(dq_sem_len)
            .map_err(|_| cuda_alloc_failed("ring fa3 dq_semaphore", vec![dq_sem_len]))?;
        for bi in 0..b {
            {
                let (lse_ptr, _lg) = lse_slice.device_ptr(&backend.stream);
                let (lsec_ptr, _lcg) = lse_c.device_ptr_mut(&backend.stream);
                check_cuda_ffi(
                    // SAFETY: lse is full [tiles, s]; dst is compact [h, lq].
                    unsafe {
                        ffi::ring_fa3_gather_lse_cuda(
                            lsec_ptr as *mut f32,
                            lse_ptr as *const f32,
                            ring_i32(h, "ring fa3 gather heads i32")?,
                            ring_i32(bi * h, "ring fa3 gather tile_base i32")?,
                            ring_i32(s, "ring fa3 gather seq i32")?,
                            ring_i32(pair.q.row, "ring fa3 gather run_start i32")?,
                            ring_i32(lq, "ring fa3 gather run_len i32")?,
                            backend.stream.cu_stream(),
                        )
                    },
                    "ring_fa3_gather_lse_cuda",
                )?;
            }
            {
                let (q_ptr, _qg) = q_bf16.device_ptr(&backend.stream);
                let (k_ptr, _kg) = k_bf16.device_ptr(&backend.stream);
                let (v_ptr, _vg) = v_bf16.device_ptr(&backend.stream);
                let (o_ptr, _og) = o_bf16.device_ptr(&backend.stream);
                let (do_ptr, _dg) = do_bf16.device_ptr(&backend.stream);
                let (lsec_ptr, _lcg) = lse_c.device_ptr(&backend.stream);
                let (dq_ptr, _dqg) = dq.device_ptr_mut(&backend.stream);
                let (dk_ptr, _dkg) = dk.device_ptr_mut(&backend.stream);
                let (dv_ptr, _dvg) = dv.device_ptr_mut(&backend.stream);
                let (sd_ptr, _sdg) = softmax_d.device_ptr_mut(&backend.stream);
                let (ll2_ptr, _llg) = lse_log2.device_ptr_mut(&backend.stream);
                let (dqa_ptr, _dqag) = dq_accum.device_ptr_mut(&backend.stream);
                let (dka_ptr, dva_ptr, _dkag, _dvag) = match dkv_accum.as_mut() {
                    Some((dk_a, dv_a)) => {
                        let (dka, dkag) = dk_a.device_ptr_mut(&backend.stream);
                        let (dva, dvag) = dv_a.device_ptr_mut(&backend.stream);
                        (dka as *mut f32, dva as *mut f32, Some(dkag), Some(dvag))
                    }
                    None => (std::ptr::null_mut(), std::ptr::null_mut(), None, None),
                };
                let (sem_ptr, _semg) = dq_sem.device_ptr_mut(&backend.stream);
                let q_off = (((bi * h * s) + pair.q.row) * d * 2) as u64;
                let kv_off = (((bi * hk * blk) + pair.k.row) * d * 2) as u64;
                let args = ffi::ArleFa3BwdHd256Args {
                    q: (q_ptr + q_off) as *const ffi::Half,
                    k: (k_ptr + kv_off) as *const ffi::Half,
                    v: (v_ptr + kv_off) as *const ffi::Half,
                    o: (o_ptr + q_off) as *const ffi::Half,
                    dout: (do_ptr + q_off) as *const ffi::Half,
                    softmax_lse: lsec_ptr as *const f32,
                    dq: dq_ptr as *mut ffi::Half,
                    dk: dk_ptr as *mut ffi::Half,
                    dv: dv_ptr as *mut ffi::Half,
                    softmax_d: sd_ptr as *mut f32,
                    softmax_lse_log2: ll2_ptr as *mut f32,
                    dq_accum: dqa_ptr as *mut f32,
                    dk_accum: dka_ptr,
                    dv_accum: dva_ptr,
                    dq_semaphore: sem_ptr as *mut i32,
                    softmax_d_capacity: (h * sq_r) as i64,
                    softmax_lse_log2_capacity: (h * sq_r) as i64,
                    dq_accum_capacity: (h * sq_r * 256) as i64,
                    dk_accum_capacity: if gqa { (hk * sk_r * 256) as i64 } else { 0 },
                    dv_accum_capacity: if gqa { (hk * sk_r * 256) as i64 } else { 0 },
                    dq_semaphore_capacity: dq_sem_len as i64,
                    seqlen_q: ring_i32(lq, "ring fa3 bwd seqlen_q i32")?,
                    seqlen_k: ring_i32(lk, "ring fa3 bwd seqlen_k i32")?,
                    num_heads: ring_i32(h, "ring fa3 bwd num_heads i32")?,
                    num_heads_k: ring_i32(hk, "ring fa3 bwd num_heads_k i32")?,
                    head_dim: 256,
                    q_row_stride: d as i64,
                    k_row_stride: d as i64,
                    v_row_stride: d as i64,
                    o_row_stride: d as i64,
                    do_row_stride: d as i64,
                    dq_row_stride: d as i64,
                    dk_row_stride: d as i64,
                    dv_row_stride: d as i64,
                    q_head_stride: (s * d) as i64,
                    k_head_stride: (blk * d) as i64,
                    v_head_stride: (blk * d) as i64,
                    o_head_stride: (s * d) as i64,
                    do_head_stride: (s * d) as i64,
                    dq_head_stride: (lq * d) as i64,
                    dk_head_stride: (lk * d) as i64,
                    dv_head_stride: (lk * d) as i64,
                    softmax_scale: dims.sm_scale,
                    is_causal: pair.causal as i32,
                };
                check_cuda_ffi(
                    // SAFETY: strided head-major views of live bf16 buffers; compact
                    // dq/dk/dv outputs; scratch sized per the bwd shim's contract
                    // (round_up 64/80 tiles, GQA fp32 accum, dq semaphore).
                    unsafe { ffi::arle_fa3_bwd_hd256_bf16_cuda(&args, backend.stream.cu_stream()) },
                    "arle_fa3_bwd_hd256_bf16_cuda",
                )?;
            }
            // Accumulate the pair's compact bf16 grads into the running f32
            // buffers (dq += into gq carry; dk/dv += into this block's grads).
            let accum = |dst: &mut CudaSlice<f32>,
                         src: &CudaSlice<u16>,
                         heads: usize,
                         tile_base: usize,
                         seq: usize,
                         run: &crate::ops::ring_attention::PosRun|
             -> Result<()> {
                let (dst_ptr, _dg) = dst.device_ptr_mut(&backend.stream);
                let (src_ptr, _sg) = src.device_ptr(&backend.stream);
                check_cuda_ffi(
                    // SAFETY: dst full [tiles, seq, d]; src compact [heads, run.len, d].
                    unsafe {
                        ffi::ring_fa3_accum_grad_bf16_cuda(
                            dst_ptr as *mut f32,
                            src_ptr as *const ffi::Half,
                            ring_i32(heads, "ring fa3 accum heads i32")?,
                            ring_i32(tile_base, "ring fa3 accum tile_base i32")?,
                            ring_i32(seq, "ring fa3 accum seq i32")?,
                            ring_i32(run.row, "ring fa3 accum run_start i32")?,
                            ring_i32(run.len, "ring fa3 accum run_len i32")?,
                            ring_i32(d, "ring fa3 accum head_dim i32")?,
                            backend.stream.cu_stream(),
                        )
                    },
                    "ring_fa3_accum_grad_bf16_cuda",
                )
            };
            accum(&mut gq_out, &dq, h, bi * h, s, &pair.q)?;
            accum(&mut gk, &dk, hk, bi * hk, blk, &pair.k)?;
            accum(&mut gv, &dv, hk, bi * hk, blk, &pair.k)?;
        }
    }
    Ok((
        DeviceHandle::Cuda(CudaStorage::new(gq_out)),
        DeviceHandle::Cuda(CudaStorage::new(gk)),
        DeviceHandle::Cuda(CudaStorage::new(gv)),
    ))
}
