use super::*;

pub(super) fn cuda_all_reduce_sum_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    shape: &[usize],
    axis: CommAxis,
) -> Result<DeviceHandle> {
    #[cfg(not(feature = "nccl"))]
    let _ = axis;
    let len = shape_size(shape);
    let src = backend.cuda_slice(x, "all_reduce_sum")?;
    if src.len() != len {
        return Err(AutogradError::DataLengthMismatch {
            len: src.len(),
            shape: shape.to_vec(),
            size: len,
        });
    }
    let mut out = alloc_zeros_retry::<f32>(backend, len)
        .map_err(|_| AutogradError::TapeInvariant("cuda all_reduce alloc failed"))?;
    backend
        .stream
        .memcpy_dtod(src, &mut out)
        .map_err(|_| AutogradError::TapeInvariant("cuda all_reduce D2D copy failed"))?;

    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    if let Some(nccl) = backend.comm(axis) {
        let (dst_ptr, _dst_guard) = out.device_ptr_mut(&backend.stream);
        // SAFETY: NCCL collective over a guarded device pointer live for the call.
        unsafe {
            nccl.all_reduce(
                dst_ptr as *mut _,
                len,
                DType::F32,
                ReduceOp::Sum,
                backend.stream.cu_stream().cast(),
            )
            .map_err(|_| AutogradError::TapeInvariant("NCCL all_reduce_sum failed"))?;
        }
    }

    Ok(DeviceHandle::Cuda(CudaStorage::new(out)))
}

pub(super) fn cuda_all_gather_seq_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    local_shape: &[usize],
    axis: CommAxis,
) -> Result<DeviceHandle> {
    #[cfg(not(feature = "nccl"))]
    let _ = axis;
    let local_len = shape_size(local_shape);
    let src = backend.cuda_slice(x, "all_gather_seq")?;
    if src.len() != local_len {
        return Err(AutogradError::DataLengthMismatch {
            len: src.len(),
            shape: local_shape.to_vec(),
            size: local_len,
        });
    }

    // No communicator (single-process / CPU parity path): identity — the
    // gathered full sequence equals this rank's local shard (world==1).
    // Buffers are functional (write-fresh; see add_into_device), and this
    // branch skips the in-place NCCL write, so sharing the input Arc is
    // safe — no alloc, no D2D copy.
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    let world = backend.comm(axis).map_or(1, |nccl| nccl.world_size());
    #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
    let world = 1usize;
    if world <= 1 {
        return Ok(x.clone());
    }

    // NCCL all-gather: shards are equal-length (seq % world == 0), so the
    // full [1, S, H] is the rank-order concatenation of each [1, S/N, H].
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    {
        let full_len = local_len * world;
        let mut out = alloc_zeros_retry::<f32>(backend, full_len)
            .map_err(|_| AutogradError::TapeInvariant("cuda all_gather_seq full alloc failed"))?;
        let nccl = backend.comm(axis).expect("world>1 implies nccl present");
        // Scope the device-ptr guards so their SyncOnDrop borrow of `out`
        // ends before `out` is moved into the handle (mirrors the implicit
        // drop in all_reduce_sum_device's `if let` block).
        {
            let (src_ptr, _src_guard) = src.device_ptr(&backend.stream);
            let (dst_ptr, _dst_guard) = out.device_ptr_mut(&backend.stream);
            // SAFETY: NCCL collective over guarded src/dst pointers live for the call.
            unsafe {
                nccl.all_gather(
                    src_ptr as *const _,
                    dst_ptr as *mut _,
                    local_len,
                    DType::F32,
                    backend.stream.cu_stream().cast(),
                )
                .map_err(|_| AutogradError::TapeInvariant("NCCL all_gather_seq failed"))?;
            }
        }
        Ok(DeviceHandle::Cuda(CudaStorage::new(out)))
    }
    #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
    unreachable!("world>1 without nccl feature")
}

pub(super) fn cuda_reduce_scatter_sum_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    local_shape: &[usize],
    axis: CommAxis,
) -> Result<DeviceHandle> {
    #[cfg(not(feature = "nccl"))]
    let _ = axis;
    let local_len = shape_size(local_shape);
    let src = backend.cuda_slice(x, "reduce_scatter_sum")?;

    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    let world = backend.comm(axis).map_or(1, |nccl| nccl.world_size());
    #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
    let world = 1usize;
    if world <= 1 {
        // Identity: input already this rank's [1, S/N, H] (== full at N=1).
        // Share the input Arc (functional buffers) — no alloc, no D2D copy.
        if src.len() != local_len {
            return Err(AutogradError::DataLengthMismatch {
                len: src.len(),
                shape: local_shape.to_vec(),
                size: local_len,
            });
        }
        return Ok(x.clone());
    }

    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    {
        if src.len() != local_len * world {
            return Err(AutogradError::DataLengthMismatch {
                len: src.len(),
                shape: local_shape.to_vec(),
                size: local_len * world,
            });
        }
        let mut out = alloc_zeros_retry::<f32>(backend, local_len)
            .map_err(|_| AutogradError::TapeInvariant("cuda reduce_scatter_sum alloc failed"))?;
        let nccl = backend.comm(axis).expect("world>1 implies nccl present");
        // Scope the device-ptr guards so their SyncOnDrop borrow of `out`
        // ends before `out` is moved into the handle.
        {
            let (src_ptr, _src_guard) = src.device_ptr(&backend.stream);
            let (dst_ptr, _dst_guard) = out.device_ptr_mut(&backend.stream);
            // SAFETY: NCCL collective over guarded src/dst pointers live for the call.
            unsafe {
                nccl.reduce_scatter(
                    src_ptr as *const _,
                    dst_ptr as *mut _,
                    local_len,
                    DType::F32,
                    ReduceOp::Sum,
                    backend.stream.cu_stream().cast(),
                )
                .map_err(|_| AutogradError::TapeInvariant("NCCL reduce_scatter_sum failed"))?;
            }
        }
        Ok(DeviceHandle::Cuda(CudaStorage::new(out)))
    }
    #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
    unreachable!("world>1 without nccl feature")
}

pub(super) fn cuda_ring_send_recv_kv(
    backend: &CudaBackend,
    block: &DeviceHandle,
    block_shape: &[usize],
) -> Result<DeviceHandle> {
    let len = shape_size(block_shape);
    let src = backend.cuda_slice(block, "ring_send_recv_kv")?;
    if src.len() != len {
        return Err(AutogradError::DataLengthMismatch {
            len: src.len(),
            shape: block_shape.to_vec(),
            size: len,
        });
    }

    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    let world = backend
        .comm(CommAxis::Seq)
        .map_or(1, |nccl| nccl.world_size());
    #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
    let world = 1usize;
    // Single rank: the ring degenerates to the local block (identity).
    if world <= 1 {
        return Ok(block.clone());
    }

    // Ring rotation: send this block to (rank+1)%world, receive the block
    // from (rank-1+world)%world, both inside one group so NCCL pairs the
    // matched send/recv without deadlock. Blocks are equal-length (the
    // launcher pads seq to a multiple of world), so recv fills `len`.
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    {
        let nccl = backend
            .comm(CommAxis::Seq)
            .expect("world>1 implies nccl present");
        let rank = nccl.rank();
        let next = (rank + 1) % world;
        let prev = (rank + world - 1) % world;
        let mut out = alloc_zeros_retry::<f32>(backend, len)
            .map_err(|_| AutogradError::TapeInvariant("cuda ring_send_recv_kv alloc failed"))?;
        {
            let (src_ptr, _src_guard) = src.device_ptr(&backend.stream);
            let (dst_ptr, _dst_guard) = out.device_ptr_mut(&backend.stream);
            let stream = backend.stream.cu_stream().cast();
            nccl.group_start()
                .map_err(|_| AutogradError::TapeInvariant("ring group_start failed"))?;
            // SAFETY: NCCL send/recv inside a group over guarded pointers live for the call.
            unsafe {
                nccl.send(src_ptr as *const _, len, DType::F32, next, stream)
                    .map_err(|_| AutogradError::TapeInvariant("ring send failed"))?;
                nccl.recv(dst_ptr as *mut _, len, DType::F32, prev, stream)
                    .map_err(|_| AutogradError::TapeInvariant("ring recv failed"))?;
            }
            nccl.group_end()
                .map_err(|_| AutogradError::TapeInvariant("ring group_end failed"))?;
        }
        Ok(DeviceHandle::Cuda(CudaStorage::new(out)))
    }
    #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
    unreachable!("world>1 without nccl feature")
}

/// Blocking point-to-point send on the CP communicator (a lone `ncclSend` is
/// safe: the peer's matching `ncclRecv` is the only other party).
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_cp_send(
    backend: &CudaBackend,
    handle: &DeviceHandle,
    len: usize,
    peer: usize,
) -> Result<()> {
    let src = backend.cuda_slice(handle, "cp_send")?;
    if src.len() < len {
        return Err(AutogradError::DataLengthMismatch {
            len: src.len(),
            shape: vec![len],
            size: len,
        });
    }
    #[cfg(feature = "nccl")]
    {
        let nccl = backend
            .comm(CommAxis::Seq)
            .ok_or(AutogradError::TapeInvariant("cp_send: no CP communicator"))?;
        let (src_ptr, _src_guard) = src.device_ptr(&backend.stream);
        let stream = backend.stream.cu_stream().cast();
        // SAFETY: guarded pointer live for the call; peer runs the matching recv.
        unsafe {
            nccl.send(src_ptr as *const _, len, DType::F32, peer, stream)
                .map_err(|_| AutogradError::TapeInvariant("cp_send failed"))?;
        }
        Ok(())
    }
    #[cfg(not(feature = "nccl"))]
    {
        let _ = peer;
        Err(AutogradError::TapeInvariant("cp_send: built without nccl"))
    }
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_cp_recv(backend: &CudaBackend, len: usize, peer: usize) -> Result<DeviceHandle> {
    #[cfg(feature = "nccl")]
    {
        let nccl = backend
            .comm(CommAxis::Seq)
            .ok_or(AutogradError::TapeInvariant("cp_recv: no CP communicator"))?;
        let mut out = alloc_zeros_retry::<f32>(backend, len)
            .map_err(|_| AutogradError::TapeInvariant("cuda cp_recv alloc failed"))?;
        {
            let (dst_ptr, _dst_guard) = out.device_ptr_mut(&backend.stream);
            let stream = backend.stream.cu_stream().cast();
            // SAFETY: guarded pointer live for the call; peer runs the matching send.
            unsafe {
                nccl.recv(dst_ptr as *mut _, len, DType::F32, peer, stream)
                    .map_err(|_| AutogradError::TapeInvariant("cp_recv failed"))?;
            }
        }
        Ok(DeviceHandle::Cuda(CudaStorage::new(out)))
    }
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (backend, len, peer);
        Err(AutogradError::TapeInvariant("cp_recv: built without nccl"))
    }
}
