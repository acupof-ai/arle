use crate::{
    AutogradError, Result,
    backend::DeviceHandle,
    tensor::{TensorId, TensorStore},
};

/// One full-size device buffer whose disjoint row ranges are written in place.
/// No append by design — folding `concat_axis2` per chunk is quadratic and holds
/// old+new at once.
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

    /// Keep-set entry for the caller's per-chunk `free_new_except`. Do not clone
    /// the handle behind it: `write_rows` mutates the buffer in place.
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
        // Probe before the clone bumps it: in-place write would corrupt a sibling.
        if store
            .tensor(self.id)?
            .device_handle
            .as_ref()
            .and_then(DeviceHandle::device_buffer_strong_count)
            .is_some_and(|count| count != 1)
        {
            return Err(AutogradError::TapeInvariant(
                "SeqAccum buffer is shared; write_rows needs sole ownership",
            ));
        }
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
