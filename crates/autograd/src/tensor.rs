use crate::{
    AutogradError, Result,
    backend::{Backend, CpuBackend, DeviceHandle},
};
use std::{collections::HashSet, sync::Arc};

pub type TensorId = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dirty {
    Host,
    Device,
    Both,
}

#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub size: usize,
    pub requires_grad: bool,
    pub grad: Option<TensorId>,
    pub device_handle: Option<DeviceHandle>,
    pub dirty: Dirty,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        assert!(
            self.dirty != Dirty::Device,
            "ensure_host before cloning a device-resident tensor"
        );
        Self {
            data: self.data.clone(),
            // Device handles own unique backend allocations; clones fall back
            // to the host copy until an explicit re-upload repopulates them.
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            size: self.size,
            requires_grad: self.requires_grad,
            grad: self.grad,
            device_handle: None,
            dirty: Dirty::Host,
        }
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        let size = shape_size(&shape);
        if data.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: data.len(),
                shape,
                size,
            });
        }

        let strides = contiguous_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
            size,
            requires_grad,
            grad: None,
            device_handle: None,
            dirty: Dirty::Host,
        })
    }
}

#[derive(Debug)]
pub struct TensorStore {
    pub tensors: Vec<Option<Tensor>>,
    pub free_ids: Vec<TensorId>,
    backend: Arc<dyn Backend>,
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::with_backend(Arc::new(CpuBackend))
    }
}

impl TensorStore {
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self {
            tensors: Vec::new(),
            free_ids: Vec::new(),
            backend,
        }
    }

    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    pub fn set_backend(&mut self, backend: Arc<dyn Backend>) {
        self.backend = backend;
    }

    pub fn alloc(&mut self, tensor: Tensor) -> TensorId {
        if let Some(id) = self.free_ids.pop() {
            self.tensors[id] = Some(tensor);
            id
        } else {
            let id = self.tensors.len();
            self.tensors.push(Some(tensor));
            id
        }
    }

    pub fn free(&mut self, id: TensorId) -> Result<()> {
        let slot = self
            .tensors
            .get_mut(id)
            .ok_or(AutogradError::InvalidTensorId(id))?;
        if slot.is_none() {
            return Err(AutogradError::InvalidTensorId(id));
        }
        *slot = None;
        self.free_ids.push(id);
        Ok(())
    }

    pub fn retain_ids(&mut self, keep: &HashSet<TensorId>) {
        for (id, slot) in self.tensors.iter_mut().enumerate() {
            if keep.contains(&id) || slot.is_none() {
                continue;
            }
            *slot = None;
            self.free_ids.push(id);
        }
    }

    pub fn get(&self, id: TensorId) -> Option<&Tensor> {
        self.tensors.get(id).and_then(Option::as_ref)
    }

    pub fn get_mut(&mut self, id: TensorId) -> Option<&mut Tensor> {
        if matches!(
            self.tensors
                .get(id)
                .and_then(Option::as_ref)
                .map(|tensor| &tensor.dirty),
            Some(Dirty::Device)
        ) {
            self.ensure_host(id)
                .expect("ensure_host before mutable tensor access");
        }

        let tensor = self.tensors.get_mut(id).and_then(Option::as_mut)?;
        tensor.dirty = Dirty::Host;
        Some(tensor)
    }

    pub fn from_slice(&mut self, data: &[f32], shape: &[usize]) -> Result<TensorId> {
        let tensor = Tensor::new(data.to_vec(), shape.to_vec(), false)?;
        Ok(self.alloc(tensor))
    }

    pub fn ensure_host(&mut self, id: TensorId) -> Result<()> {
        if self.tensor(id)?.dirty != Dirty::Device {
            return Ok(());
        }

        let handle = self
            .tensor(id)?
            .device_handle
            .as_ref()
            .ok_or(AutogradError::TapeInvariant(
                "device-resident tensor missing device handle",
            ))?
            .clone();
        self.backend().eval(&[&handle])?;
        let host = self.backend().readback(&handle)?;
        let tensor = self.raw_tensor_mut(id)?;
        tensor.data = host;
        tensor.dirty = Dirty::Both;
        Ok(())
    }

    /// Flush a batch of `Dirty::Device` tensors to host using **one**
    /// backend `eval` call, then per-id `readback`. Equivalent to calling
    /// `ensure_host` for each id, but collapses N FFI eval boundaries
    /// into 1 — meaningful on Metal where each `mlx_eval` round-trip
    /// dominates at small shapes (see `tape::backward` flush loop).
    /// Tensors not currently `Dirty::Device` are silently skipped, so the
    /// caller can pass the entire output set without pre-filtering.
    pub fn flush_to_host_batch(&mut self, ids: &[TensorId]) -> Result<()> {
        let mut to_flush: Vec<(TensorId, DeviceHandle)> = Vec::with_capacity(ids.len());
        for &id in ids {
            let tensor = self.tensor(id)?;
            if tensor.dirty != Dirty::Device {
                continue;
            }
            let handle = tensor
                .device_handle
                .as_ref()
                .ok_or(AutogradError::TapeInvariant(
                    "device-resident tensor missing device handle",
                ))?
                .clone();
            to_flush.push((id, handle));
        }
        if to_flush.is_empty() {
            return Ok(());
        }
        let handle_refs: Vec<&DeviceHandle> = to_flush.iter().map(|(_, h)| h).collect();
        self.backend().eval(&handle_refs)?;
        for (id, handle) in to_flush {
            let host = self.backend().readback(&handle)?;
            let tensor = self.raw_tensor_mut(id)?;
            tensor.data = host;
            tensor.dirty = Dirty::Both;
        }
        Ok(())
    }

    pub fn ensure_device(&mut self, id: TensorId) -> Result<()> {
        let (dirty, has_handle, data, shape) = {
            let tensor = self.tensor(id)?;
            (
                tensor.dirty.clone(),
                tensor.device_handle.is_some(),
                tensor.data.clone(),
                tensor.shape.clone(),
            )
        };

        if has_handle && dirty != Dirty::Host {
            return Ok(());
        }

        let handle = self.backend().upload(&data, &shape)?;
        let tensor = self.raw_tensor_mut(id)?;
        tensor.device_handle = Some(handle);
        tensor.dirty = Dirty::Both;
        Ok(())
    }

    pub fn to_host(&mut self, id: TensorId) -> Result<Vec<f32>> {
        self.ensure_host(id)?;
        Ok(self.tensor(id)?.data.clone())
    }

    pub fn alloc_device_tensor(
        &mut self,
        shape: Vec<usize>,
        handle: DeviceHandle,
    ) -> Result<TensorId> {
        let tensor = Tensor {
            data: Vec::new(),
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            size: shape_size(&shape),
            requires_grad: false,
            grad: None,
            device_handle: Some(handle),
            dirty: Dirty::Device,
        };
        Ok(self.alloc(tensor))
    }

    /// Replace an existing tensor's device handle with a fresh one, marking
    /// the host copy stale (`Dirty::Device`) and clearing the cached host
    /// `data` buffer.
    ///
    /// Used by the device-backed AdamW path (M5.3b.10): the optimizer
    /// produces a new `DeviceHandle` for each updated parameter and we must
    /// install it without going through `get_mut` — `get_mut` auto-triggers
    /// `ensure_host`, which would re-download the old pre-update values and
    /// then mark the tensor `Dirty::Host`, forcing a full re-upload on the
    /// next forward pass. That is the exact churn this path exists to kill.
    pub fn replace_device_handle(&mut self, id: TensorId, handle: DeviceHandle) -> Result<()> {
        let tensor = self.raw_tensor_mut(id)?;
        tensor.device_handle = Some(handle);
        tensor.dirty = Dirty::Device;
        tensor.data.clear();
        Ok(())
    }

    pub(crate) fn set_requires_grad(&mut self, id: TensorId, requires_grad: bool) -> Result<()> {
        self.raw_tensor_mut(id)?.requires_grad = requires_grad;
        Ok(())
    }

    pub(crate) fn set_grad(&mut self, id: TensorId, grad: Option<TensorId>) -> Result<()> {
        self.raw_tensor_mut(id)?.grad = grad;
        Ok(())
    }

    fn raw_tensor_mut(&mut self, id: TensorId) -> Result<&mut Tensor> {
        self.tensors
            .get_mut(id)
            .and_then(Option::as_mut)
            .ok_or(AutogradError::InvalidTensorId(id))
    }

    pub fn zeros_like(&mut self, id: TensorId) -> Result<TensorId> {
        let source = self.tensor(id)?;
        let tensor = Tensor::new(vec![0.0; source.size], source.shape.clone(), false)?;
        Ok(self.alloc(tensor))
    }

    pub fn accumulate_grad(&mut self, param_id: TensorId, grad_id: TensorId) -> Result<()> {
        let (requires_grad, shape, existing_grad) = {
            let tensor = self.tensor(param_id)?;
            (tensor.requires_grad, tensor.shape.clone(), tensor.grad)
        };
        if !requires_grad {
            return Ok(());
        }

        let grad_shape = self.tensor(grad_id)?.shape.clone();
        if shape != grad_shape {
            return Err(AutogradError::GradientShapeMismatch {
                tensor_id: param_id,
                expected: shape,
                got: grad_shape,
            });
        }

        match existing_grad {
            Some(existing_id) => {
                let incoming = self.to_host(grad_id)?;
                let existing = self
                    .get_mut(existing_id)
                    .ok_or(AutogradError::InvalidTensorId(existing_id))?;
                for (dst, src) in existing.data.iter_mut().zip(incoming) {
                    *dst += src;
                }
            }
            None => {
                let cloned_grad_id = self.clone_tensor(grad_id)?;
                self.set_grad(param_id, Some(cloned_grad_id))?;
            }
        }

        Ok(())
    }

    pub(crate) fn tensor(&self, id: TensorId) -> Result<&Tensor> {
        self.get(id).ok_or(AutogradError::InvalidTensorId(id))
    }

    pub(crate) fn tensor_mut(&mut self, id: TensorId) -> Result<&mut Tensor> {
        self.get_mut(id).ok_or(AutogradError::InvalidTensorId(id))
    }

    pub(crate) fn clone_tensor(&mut self, id: TensorId) -> Result<TensorId> {
        self.ensure_host(id)?;
        let tensor = self.tensor(id)?.clone();
        Ok(self.alloc(tensor))
    }

    pub(crate) fn fill_like(&mut self, id: TensorId, value: f32) -> Result<TensorId> {
        let source = self.tensor(id)?;
        let tensor = Tensor::new(vec![value; source.size], source.shape.clone(), false)?;
        Ok(self.alloc(tensor))
    }
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (index, dim) in shape.iter().enumerate().rev() {
        strides[index] = stride;
        stride *= *dim;
    }
    strides
}

fn shape_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_reuses_slot() {
        let mut store = TensorStore::default();
        let first = store
            .from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
            .expect("alloc first tensor");
        store.free(first).expect("free first tensor");
        let second = store
            .from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2])
            .expect("alloc second tensor");

        assert_eq!(first, second);
    }

    #[test]
    fn from_slice_tracks_shape_and_host_data() {
        let mut store = TensorStore::default();
        let id = store
            .from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("alloc tensor");

        let tensor = store.get(id).expect("tensor exists");
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.strides, vec![3, 1]);
        assert_eq!(tensor.size, 6);
        assert_eq!(
            store.to_host(id).expect("host copy"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }
}
