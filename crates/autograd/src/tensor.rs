use crate::{
    AutogradError, Result,
    backend::{Backend, CpuBackend},
};
use std::{collections::HashSet, sync::Arc};

pub type TensorId = usize;

#[derive(Debug, Clone)]
pub struct GpuTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub size: usize,
    pub requires_grad: bool,
    pub grad: Option<TensorId>,
}

impl GpuTensor {
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
        })
    }
}

#[derive(Debug)]
pub struct TensorStore {
    pub tensors: Vec<Option<GpuTensor>>,
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

    pub fn alloc(&mut self, tensor: GpuTensor) -> TensorId {
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

    pub fn get(&self, id: TensorId) -> Option<&GpuTensor> {
        self.tensors.get(id).and_then(Option::as_ref)
    }

    pub fn get_mut(&mut self, id: TensorId) -> Option<&mut GpuTensor> {
        self.tensors.get_mut(id).and_then(Option::as_mut)
    }

    pub fn from_slice(&mut self, data: &[f32], shape: &[usize]) -> Result<TensorId> {
        let tensor = GpuTensor::new(data.to_vec(), shape.to_vec(), false)?;
        Ok(self.alloc(tensor))
    }

    pub fn to_host(&self, id: TensorId) -> Result<Vec<f32>> {
        Ok(self.tensor(id)?.data.clone())
    }

    pub fn zeros_like(&mut self, id: TensorId) -> Result<TensorId> {
        let source = self.tensor(id)?;
        let tensor = GpuTensor::new(vec![0.0; source.size], source.shape.clone(), false)?;
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
                let existing = self.tensor_mut(existing_id)?;
                for (dst, src) in existing.data.iter_mut().zip(incoming) {
                    *dst += src;
                }
            }
            None => {
                let cloned_grad_id = self.clone_tensor(grad_id)?;
                self.tensor_mut(param_id)?.grad = Some(cloned_grad_id);
            }
        }

        Ok(())
    }

    pub(crate) fn tensor(&self, id: TensorId) -> Result<&GpuTensor> {
        self.get(id).ok_or(AutogradError::InvalidTensorId(id))
    }

    pub(crate) fn tensor_mut(&mut self, id: TensorId) -> Result<&mut GpuTensor> {
        self.get_mut(id).ok_or(AutogradError::InvalidTensorId(id))
    }

    pub(crate) fn clone_tensor(&mut self, id: TensorId) -> Result<TensorId> {
        let tensor = self.tensor(id)?.clone();
        Ok(self.alloc(tensor))
    }

    pub(crate) fn fill_like(&mut self, id: TensorId, value: f32) -> Result<TensorId> {
        let source = self.tensor(id)?;
        let tensor = GpuTensor::new(vec![value; source.size], source.shape.clone(), false)?;
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
