//! The two accumulators a chunked forward/backward needs. Both write through one
//! device buffer in place, so neither can hold old+new at once — the quadratic
//! fold that OOM'd the 131072 forward is not expressible here.

use crate::{
    AutogradError, Result,
    backend::DeviceHandle,
    tensor::{TensorId, TensorStore},
};

/// Probe before cloning: an in-place write through a shared buffer corrupts the
/// sibling, and `TensorStore::device_handle` bumps the count past 1.
fn sole_owned_handle(id: TensorId, store: &mut TensorStore) -> Result<DeviceHandle> {
    store.ensure_device(id)?;
    let handle = store
        .tensor(id)?
        .device_handle
        .as_ref()
        .expect("ensure_device");
    if handle
        .device_buffer_strong_count()
        .is_some_and(|count| count != 1)
    {
        return Err(AutogradError::TapeInvariant(
            "chunk accumulator buffer is shared; in-place write needs sole ownership",
        ));
    }
    Ok(handle.clone())
}

/// Output rows partition by chunk: one full-size buffer, disjoint ranges assigned
/// in place.
pub struct SeqAccum {
    id: TensorId,
    shape: Vec<usize>,
    row_axis: usize,
}

impl SeqAccum {
    pub fn new(shape: Vec<usize>, row_axis: usize, store: &mut TensorStore) -> Result<Self> {
        if row_axis >= shape.len() {
            return Err(AutogradError::InvalidRank {
                expected: "row_axis < shape rank",
                got: row_axis,
            });
        }
        let handle = store.backend().zeros(&shape)?;
        let id = store.alloc_device_tensor(shape.clone(), handle)?;
        Ok(Self {
            id,
            shape,
            row_axis,
        })
    }

    /// Never clone the handle behind it: both writers mutate the buffer in place,
    /// so a live second reference turns into an error at the next write.
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Assign `src` into rows `start .. start + src.shape[row_axis]`.
    pub fn write_rows(
        &mut self,
        start: usize,
        src: TensorId,
        store: &mut TensorStore,
    ) -> Result<()> {
        let src_shape = &store.tensor(src)?.shape;
        let mut expected = self.shape.clone();
        expected[self.row_axis] = src_shape.get(self.row_axis).copied().unwrap_or(0);
        if *src_shape != expected {
            let got = src_shape.clone();
            return Err(AutogradError::ShapeMismatch { expected, got });
        }
        let mut starts = vec![0; self.shape.len()];
        starts[self.row_axis] = start;
        let mut ends = self.shape.clone();
        ends[self.row_axis] = start + expected[self.row_axis];
        let dest = sole_owned_handle(self.id, store)?;
        let src_handle = store.device_handle(src)?;
        let updated =
            store
                .backend()
                .write_slice_device(&dest, &src_handle, &self.shape, &starts, &ends)?;
        store.replace_device_handle(self.id, updated)
    }

    pub fn finish(self) -> TensorId {
        self.id
    }
}

/// Every chunk contributes to the whole tensor: one buffer summed in place.
pub struct ChunkSum {
    id: Option<TensorId>,
}

impl ChunkSum {
    pub fn new() -> Self {
        Self { id: None }
    }

    /// See `SeqAccum::id`.
    pub fn id(&self) -> Option<TensorId> {
        self.id
    }

    pub fn add(&mut self, part: TensorId, store: &mut TensorStore) -> Result<()> {
        let src = store.device_handle(part)?;
        match self.id {
            // Adopt the first chunk's buffer instead of copying it; the caller's
            // `free_new_except` drops `part`, restoring sole ownership.
            None => {
                let shape = store.tensor(part)?.shape.clone();
                self.id = Some(store.alloc_device_tensor(shape, src)?);
            }
            Some(acc) => {
                let shape = store.tensor(acc)?.shape.clone();
                let dest = sole_owned_handle(acc, store)?;
                let updated = store
                    .backend()
                    .accumulate_into_device(&dest, &src, &shape)?;
                store.replace_device_handle(acc, updated)?;
            }
        }
        Ok(())
    }

    /// Evict until the next `add`, which re-uploads. Only the caller knows
    /// another chunk follows — parking after the last one is a whole-accumulator
    /// round trip that `finish` immediately undoes. Skipped below
    /// `checkpoint_offload_min_bytes`, where the transfer costs more than the
    /// peak it saves.
    pub fn park(&mut self, store: &mut TensorStore) -> Result<()> {
        let Some(id) = self.id else { return Ok(()) };
        if store.tensor(id)?.size * size_of::<f32>()
            >= crate::runtime_flags::checkpoint_offload_min_bytes()
        {
            store.offload_to_host(id)?;
        }
        Ok(())
    }

    /// Brings a parked accumulator back to device; `None` when no chunk
    /// contributed.
    pub fn finish(self, store: &mut TensorStore) -> Result<Option<TensorId>> {
        if let Some(id) = self.id {
            store.ensure_device(id)?;
        }
        Ok(self.id)
    }
}

impl Default for ChunkSum {
    fn default() -> Self {
        Self::new()
    }
}
