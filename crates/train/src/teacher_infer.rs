//! Teacher-forward abstraction for OPD.
//!
//! Phase 2 of the large-to-small OPD path needs two teacher sources behind
//! the same train-side contract: the existing in-process `Qwen35Model` teacher
//! and, next, an `infer` runtime teacher. `DeviceLogits` intentionally carries
//! a `TensorId` in the caller's `TensorStore` so the KL path can stay on the
//! same backend without a host materialization.

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use autograd::Backend;
use autograd::{AutogradError, Tape, TensorId, TensorStore};
#[cfg(feature = "cuda")]
use infer::server_engine::LoadedInferenceEngine;

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
    #[cfg(feature = "cuda")]
    #[error("infer runtime teacher forward failed: {0}")]
    InferRuntime(String),
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

#[cfg(feature = "cuda")]
pub struct InferTeacher {
    engine: Arc<Mutex<LoadedInferenceEngine>>,
    train_backend: Arc<dyn Backend>,
    vocab_size: usize,
}

#[cfg(feature = "cuda")]
impl InferTeacher {
    pub fn new(
        engine: Arc<Mutex<LoadedInferenceEngine>>,
        train_backend: Arc<dyn Backend>,
        vocab_size: usize,
    ) -> Self {
        Self {
            engine,
            train_backend,
            vocab_size,
        }
    }

    pub fn engine(&self) -> &Arc<Mutex<LoadedInferenceEngine>> {
        &self.engine
    }

    pub fn train_backend(&self) -> &Arc<dyn Backend> {
        &self.train_backend
    }
}

#[cfg(feature = "cuda")]
impl TeacherForward for InferTeacher {
    fn forward_logits_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        _tape: &mut Tape,
    ) -> Result<DeviceLogits> {
        if input_ids.is_empty() {
            return Err(TeacherForwardError::InvalidInput(
                "InferTeacher requires a non-empty token sequence".to_owned(),
            ));
        }
        if input_ids.len() != positions.len() {
            return Err(TeacherForwardError::InvalidInput(format!(
                "InferTeacher token/position length mismatch: tokens={} positions={}",
                input_ids.len(),
                positions.len()
            )));
        }

        let raw_logits = {
            let engine = self.engine.lock().map_err(|err| {
                TeacherForwardError::InferRuntime(format!(
                    "LoadedInferenceEngine lock poisoned before raw logits forward: {err}"
                ))
            })?;
            engine
                .forward_token_logits(input_ids, positions)
                .map_err(|err| TeacherForwardError::InferRuntime(err.to_string()))?
        };
        if raw_logits.vocab_size() != self.vocab_size {
            return Err(TeacherForwardError::InvalidInput(format!(
                "InferTeacher vocab mismatch: raw logits vocab={}, configured vocab={}. \
                 Hint: construct InferTeacher with the vocab size from the same infer model.",
                raw_logits.vocab_size(),
                self.vocab_size
            )));
        }
        if raw_logits.seq_len() != input_ids.len() {
            return Err(TeacherForwardError::InvalidInput(format!(
                "InferTeacher seq_len mismatch: raw logits seq_len={}, input token len={}",
                raw_logits.seq_len(),
                input_ids.len()
            )));
        }

        raw_logits
            .device
            .sync()
            .map_err(|err| TeacherForwardError::InferRuntime(err.to_string()))?;
        let shape = vec![1, raw_logits.seq_len(), raw_logits.vocab_size()];
        let handle = raw_logits.with_logits_device_ptr(|src_ptr| {
            self.train_backend
                .import_bf16_device_ptr_as_f32(src_ptr, raw_logits.logits.len, &shape)
        })?;
        let tensor_id = store.alloc_device_tensor(shape.clone(), handle)?;
        Ok(DeviceLogits { tensor_id, shape })
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
