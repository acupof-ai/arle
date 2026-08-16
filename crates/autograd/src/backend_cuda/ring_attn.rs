use super::*;

// --- Context-parallel ring attention (Track A device path) ---
// Autograd-side adapters only: translate `CudaBackend` + `DeviceHandle` into
// `CudaSlice` and stage f32↔bf16. q/k/v arrive as f32 handles (tape tensors),
// converted to bf16 for the kernel (matching the training activation precision);
// the (m,l,o) accumulators and grads stay f32 for a stable merge. The FA3 pair
// route and its launches live tape-free in `cuda_kernels::ring_attention`; the
// scalar one-block kernels below remain the non-sm90 / non-hd256 fallback.
#[cfg(not(feature = "no-cuda"))]
use cuda_kernels::ring_attention::{ring_block_bwd_fa3, ring_block_fwd_merge_fa3, ring_fa3_route};

#[cfg(not(feature = "no-cuda"))]
pub(super) fn ring_i32(v: usize, label: &'static str) -> Result<i32> {
    i32::try_from(v).map_err(|_| AutogradError::TapeInvariant(label))
}

/// Map a ring-core anyhow error (alloc shapes, launch labels) into the tape's
/// static error, keeping the detail in the log.
#[cfg(not(feature = "no-cuda"))]
fn ring_core_err(e: anyhow::Error, label: &'static str) -> AutogradError {
    log::error!("{label}: {e:#}");
    AutogradError::TapeInvariant(label)
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
    if ring_fa3_route(&backend.stream, dims, q_pos_host, k_pos_host) {
        let (m_out, l_out, o_out) = ring_block_fwd_merge_fa3(
            &backend.stream,
            &q_bf16,
            &k_bf16,
            &v_bf16,
            m_in,
            l_in,
            o_in,
            q_pos_host,
            k_pos_host,
            dims,
        )
        .map_err(|e| ring_core_err(e, "ring fa3 fwd merge failed"))?;
        return Ok((
            DeviceHandle::Cuda(CudaStorage::new(m_out)),
            DeviceHandle::Cuda(CudaStorage::new(l_out)),
            DeviceHandle::Cuda(CudaStorage::new(o_out)),
        ));
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
    if ring_fa3_route(&backend.stream, dims, q_pos_host, k_pos_host) {
        // FA3 bwd wants o/d_out bf16; o is saved f32 — one quantize copy per call.
        let o_bf16 = backend.local_f32_as_bf16(out_slice, out_slice.len())?;
        let (gq_out, gk, gv) = ring_block_bwd_fa3(
            &backend.stream,
            &q_bf16,
            &k_bf16,
            &v_bf16,
            &o_bf16,
            lse_slice,
            &do_bf16,
            gq_out,
            gk,
            gv,
            q_pos_host,
            k_pos_host,
            dims,
        )
        .map_err(|e| ring_core_err(e, "ring fa3 bwd failed"))?;
        return Ok((
            DeviceHandle::Cuda(CudaStorage::new(gq_out)),
            DeviceHandle::Cuda(CudaStorage::new(gk)),
            DeviceHandle::Cuda(CudaStorage::new(gv)),
        ));
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
