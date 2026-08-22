//! Chunked forward/backward accumulators. Both write one device buffer in
//! place, so the quadratic fold that OOM'd the 131072 forward is unwritable.

use crate::{
    AutogradError, Result,
    backend::DeviceHandle,
    tensor::{TensorId, TensorStore},
};

/// Probe before cloning — `device_handle` bumps the count, and an in-place write
/// through a shared buffer corrupts the sibling.
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

    pub fn id(&self) -> TensorId {
        self.id
    }

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

pub struct ChunkSum {
    id: Option<TensorId>,
}

impl ChunkSum {
    pub fn new() -> Self {
        Self { id: None }
    }

    pub fn id(&self) -> Option<TensorId> {
        self.id
    }

    pub fn add(&mut self, part: TensorId, store: &mut TensorStore) -> Result<()> {
        let src = store.device_handle(part)?;
        match self.id {
            // Adopt rather than copy: `free_new_except` drops `part`, restoring
            // sole ownership.
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

    /// Evict until the next `add`. Never after the last chunk — `finish`
    /// re-uploads, a whole-accumulator round trip for nothing.
    pub fn park(&mut self, store: &mut TensorStore) -> Result<()> {
        let Some(id) = self.id else { return Ok(()) };
        if store.tensor(id)?.size * size_of::<f32>()
            >= crate::runtime_flags::checkpoint_offload_min_bytes()
        {
            store.offload_to_host(id)?;
        }
        Ok(())
    }

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
