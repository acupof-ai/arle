use super::*;

/// 64 MiB slot granularity: varying trajectory lengths share size classes instead
/// of exhausting the budget on exact-fit slots.
#[cfg(not(feature = "no-cuda"))]
pub(super) const PINNED_SLOT_GRANULARITY: usize = 16 << 20;

/// Reusable pinned host buffers holding parked checkpoint activations.
///
/// Slots are permanent once allocated; a slot may be larger than the activation, so
/// every copy names its own length (cudarc's typed memcpy copies the whole dst).
/// Ordering = the backend's single stream; copies pass plain slices, which record no
/// host-visible event, so `checkpoint_pin_readback` drains the stream itself.
#[cfg(not(feature = "no-cuda"))]
#[derive(Default)]
pub(super) struct PinnedCheckpointPool {
    slots: Vec<PinnedHostSlice<f32>>,
    free: Vec<u32>,
    allocated_bytes: usize,
    stream: Option<Arc<CudaStream>>,
}

#[cfg(not(feature = "no-cuda"))]
impl Drop for PinnedCheckpointPool {
    // The slots' events are never recorded, so `PinnedHostSlice::drop` would
    // `cuMemFreeHost` under an in-flight DtoH; drain the stream first.
    fn drop(&mut self) {
        if let Some(stream) = &self.stream {
            let _ = stream.synchronize();
        }
    }
}

#[cfg(not(feature = "no-cuda"))]
impl PinnedCheckpointPool {
    /// An idle slot holding at least `len` f32; `None` = budget spent.
    fn take(
        &mut self,
        ctx: &Arc<CudaContext>,
        len: usize,
        budget_bytes: usize,
    ) -> Result<Option<u32>> {
        let reuse = {
            let slots = &self.slots;
            self.free
                .iter()
                .enumerate()
                .filter(|&(_, &slot)| slots[slot as usize].len() >= len)
                .min_by_key(|&(_, &slot)| slots[slot as usize].len())
                .map(|(index, _)| index)
        };
        if let Some(index) = reuse {
            return Ok(Some(self.free.swap_remove(index)));
        }
        let fits = |cap: usize| {
            self.allocated_bytes
                .saturating_add(cap.saturating_mul(std::mem::size_of::<f32>()))
                <= budget_bytes
        };
        // Exact-fit fallback keeps a nearly-spent (or sub-granularity) budget usable.
        let rounded = len.next_multiple_of(PINNED_SLOT_GRANULARITY);
        let capacity = if fits(rounded) {
            rounded
        } else if fits(len) {
            len
        } else {
            return Ok(None);
        };
        let bytes = capacity * std::mem::size_of::<f32>();
        let slot = u32::try_from(self.slots.len())
            .map_err(|_| AutogradError::TapeInvariant("pinned checkpoint pool slot overflow"))?;
        // SAFETY: uninitialised until the DtoH the caller enqueues next writes it.
        let buf = unsafe { ctx.alloc_pinned::<f32>(capacity) }.map_err(|_| {
            AutogradError::TapeInvariant("cuMemHostAlloc for the checkpoint pool failed")
        })?;
        self.slots.push(buf);
        self.allocated_bytes += bytes;
        Ok(Some(slot))
    }

    /// Idempotent: a double release cannot hand the same slot to two tensors.
    fn release(&mut self, slot: u32) {
        if (slot as usize) < self.slots.len() && !self.free.contains(&slot) {
            self.free.push(slot);
        }
    }

    fn slot(&self, slot: u32) -> Result<&PinnedHostSlice<f32>> {
        self.slots
            .get(slot as usize)
            .ok_or(AutogradError::TapeInvariant(
                "pinned checkpoint slot out of range",
            ))
    }
}

pub(super) fn cuda_checkpoint_pin_offload(
    backend: &CudaBackend,
    handle: &DeviceHandle,
    len: usize,
) -> Result<Option<u32>> {
    let budget = crate::runtime_flags::checkpoint_pinned_offload_bytes();
    // f32 storage only: a bf16/fp8 host image is not what the reload hands back.
    let DeviceHandle::Cuda(storage) = handle else {
        return Ok(None);
    };
    if budget == 0 {
        return Ok(None);
    }
    let src = backend.cuda_storage_slice(storage)?;
    if src.len() != len {
        return Err(AutogradError::DataLengthMismatch {
            len: src.len(),
            shape: vec![len],
            size: len,
        });
    }
    let mut pool = backend
        .pinned_checkpoints
        .lock()
        .map_err(|_| AutogradError::TapeInvariant("pinned checkpoint pool poisoned"))?;
    let Some(slot) = pool.take(backend.stream.context(), len, budget)? else {
        return Ok(None);
    };
    pool.stream.get_or_insert_with(|| backend.stream.clone());
    // No synchronize: stream order covers the write before and the free after.
    let copied = match pool.slots[slot as usize].as_mut_slice() {
        Ok(host) => backend.stream.memcpy_dtoh(src, &mut host[..len]),
        Err(err) => Err(err),
    };
    if copied.is_err() {
        pool.release(slot);
        return Err(AutogradError::TapeInvariant(
            "cuda pinned checkpoint dtoh copy failed",
        ));
    }
    Ok(Some(slot))
}

pub(super) fn cuda_checkpoint_pin_reload(
    backend: &CudaBackend,
    slot: u32,
    shape: &[usize],
) -> Result<DeviceHandle> {
    let mut pool = backend
        .pinned_checkpoints
        .lock()
        .map_err(|_| AutogradError::TapeInvariant("pinned checkpoint pool poisoned"))?;
    let size = shape_size(shape);
    let device = {
        let buf = pool.slot(slot)?;
        if buf.len() < size {
            return Err(AutogradError::DataLengthMismatch {
                len: buf.len(),
                shape: shape.to_vec(),
                size,
            });
        }
        let host = buf
            .as_slice()
            .map_err(|_| AutogradError::TapeInvariant("pinned checkpoint host pointer failed"))?;
        backend
            .stream
            .clone_htod(&host[..size])
            .map_err(|_| AutogradError::TapeInvariant("cuda pinned checkpoint htod copy failed"))?
    };
    pool.release(slot);
    Ok(DeviceHandle::Cuda(CudaStorage::new(device)))
}

pub(super) fn cuda_checkpoint_pin_readback(
    backend: &CudaBackend,
    slot: u32,
    dst: &mut [f32],
) -> Result<()> {
    // Copies record no host-visible event; the drain is what proves the DtoH landed.
    backend.stream.synchronize().map_err(|_| {
        AutogradError::TapeInvariant("cuda synchronize failed (pinned checkpoint readback)")
    })?;
    let mut pool = backend
        .pinned_checkpoints
        .lock()
        .map_err(|_| AutogradError::TapeInvariant("pinned checkpoint pool poisoned"))?;
    {
        let buf = pool.slot(slot)?;
        if buf.len() < dst.len() {
            return Err(AutogradError::DataLengthMismatch {
                len: buf.len(),
                shape: vec![dst.len()],
                size: dst.len(),
            });
        }
        let src = buf
            .as_slice()
            .map_err(|_| AutogradError::TapeInvariant("pinned checkpoint host read failed"))?;
        dst.copy_from_slice(&src[..dst.len()]);
    }
    pool.release(slot);
    Ok(())
}
