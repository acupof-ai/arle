use crate::{
    AutogradError, Result,
    tensor::{TensorId, TensorStore},
};

/// One full-size device buffer whose disjoint row ranges are written in place.
///
/// There is deliberately no append/grow: folding `concat_axis2` over chunks
/// recopies every earlier chunk (O(rows²/chunk) traffic) and holds old+new live
/// at once, and the callers additionally had to keep every chunk alive until the
/// fold. That doubled peak is what OOMed the seq=131072 forward and the
/// seq=40960 LoRA backward.
pub struct SeqAccum {
    id: TensorId,
    shape: Vec<usize>,
    row_axis: usize,
}

impl SeqAccum {
    pub fn new(shape: Vec<usize>, row_axis: usize, store: &mut TensorStore) -> Result<Self> {
        let handle = store.backend().zeros(&shape)?;
        let id = store.alloc_device_tensor(shape.clone(), handle)?;
        Ok(Self {
            id,
            shape,
            row_axis,
        })
    }

    /// Keep-set entry for the caller's per-chunk `free_new_except`.
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
        store.ensure_device(src)?;
        let src_shape = store.tensor(src)?.shape.clone();
        let conforms =
            src_shape.len() == self.shape.len()
                && src_shape.iter().zip(&self.shape).enumerate().all(
                    |(axis, (src_dim, dest_dim))| axis == self.row_axis || src_dim == dest_dim,
                );
        if !conforms {
            return Err(AutogradError::ShapeMismatch {
                expected: self.shape.clone(),
                got: src_shape,
            });
        }
        let mut starts = vec![0; self.shape.len()];
        starts[self.row_axis] = start;
        let mut ends = self.shape.clone();
        ends[self.row_axis] = start + src_shape[self.row_axis];
        let dest = store
            .tensor(self.id)?
            .device_handle
            .as_ref()
            .expect("accumulator allocated on device")
            .clone();
        let src_handle = store
            .tensor(src)?
            .device_handle
            .as_ref()
            .expect("ensure_device")
            .clone();
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
