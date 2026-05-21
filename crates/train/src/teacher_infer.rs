//! Teacher-forward abstraction for OPD.
//!
//! Phase 2 of the large-to-small OPD path needs two teacher sources behind
//! the same train-side contract: the existing in-process `Qwen35Model` teacher
//! and, next, an `infer` runtime teacher. `DeviceLogits` intentionally carries
//! a `TensorId` in the caller's `TensorStore` so the KL path can stay on the
//! same backend without a host materialization.

use autograd::{AutogradError, Tape, TensorId, TensorStore};

use crate::qwen35::{Qwen35Error, Qwen35Model};

#[derive(Debug, Clone)]
pub struct DeviceLogits {
    pub tensor_id: TensorId,
    pub shape: Vec<usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum TeacherForwardError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Qwen35(#[from] Qwen35Error),
    #[error("{0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, TeacherForwardError>;

pub trait TeacherForward {
    fn forward_logits_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<DeviceLogits>;

    fn vocab_size(&self) -> usize;

    fn parameter_ids(&self) -> &[TensorId] {
        &[]
    }
}

pub struct InProcessTeacher<'a> {
    model: &'a Qwen35Model,
    parameter_ids: Vec<TensorId>,
}

impl<'a> InProcessTeacher<'a> {
    pub fn new(model: &'a Qwen35Model) -> Self {
        Self {
            model,
            parameter_ids: model.all_parameter_ids(),
        }
    }
}

impl TeacherForward for InProcessTeacher<'_> {
    fn forward_logits_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<DeviceLogits> {
        let tensor_id = self.model.forward(store, tape, input_ids, positions)?;
        store.ensure_device(tensor_id)?;
        let shape = store
            .get(tensor_id)
            .ok_or(AutogradError::InvalidTensorId(tensor_id))?
            .shape
            .clone();
        Ok(DeviceLogits { tensor_id, shape })
    }

    fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    fn parameter_ids(&self) -> &[TensorId] {
        &self.parameter_ids
    }
}
