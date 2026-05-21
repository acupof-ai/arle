//! Teacher-forward abstraction for OPD.
//!
//! Phase 2 of the large-to-small OPD path needs two teacher sources behind
//! the same train-side contract: the existing in-process `Qwen35Model` teacher
//! and, next, an `infer` runtime teacher. `DeviceLogits` intentionally carries
//! a `TensorId` in the caller's `TensorStore` so the KL path can stay on the
//! same backend without a host materialization.

use std::collections::HashSet;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "cuda")]
use std::time::Instant;

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

pub struct TeacherEntry<'a> {
    id: String,
    teacher: &'a dyn TeacherForward,
}

impl<'a> TeacherEntry<'a> {
    pub fn new(id: impl Into<String>, teacher: &'a dyn TeacherForward) -> Self {
        Self {
            id: id.into(),
            teacher,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeacherRoute {
    pub teacher_id: String,
    pub token_prefix: Vec<u32>,
}

impl TeacherRoute {
    pub fn new(teacher_id: impl Into<String>, token_prefix: Vec<u32>) -> Self {
        Self {
            teacher_id: teacher_id.into(),
            token_prefix,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedTeacherRoute {
    teacher_index: usize,
    token_prefix: Vec<u32>,
}

/// Deterministic prompt router for using multiple specialist teachers in OPD.
///
/// The router keeps the `TeacherForward` contract unchanged: prompt ownership is
/// selected from the token prefix, and the chosen teacher returns logits in the
/// caller's `TensorStore`. Longest-prefix match wins; unmatched prompts use the
/// configured default teacher.
pub struct MultiTeacher<'a> {
    entries: Vec<TeacherEntry<'a>>,
    routes: Vec<ResolvedTeacherRoute>,
    default_index: usize,
    vocab_size: usize,
    parameter_ids: Vec<TensorId>,
}

impl<'a> MultiTeacher<'a> {
    pub fn new(entries: Vec<TeacherEntry<'a>>, default_id: &str) -> Result<Self> {
        Self::with_routes(entries, default_id, Vec::new())
    }

    pub fn with_routes(
        entries: Vec<TeacherEntry<'a>>,
        default_id: &str,
        routes: Vec<TeacherRoute>,
    ) -> Result<Self> {
        if entries.is_empty() {
            return Err(TeacherForwardError::InvalidInput(
                "MultiTeacher requires at least one teacher".to_owned(),
            ));
        }

        let mut seen_ids = HashSet::with_capacity(entries.len());
        for entry in &entries {
            if entry.id.trim().is_empty() {
                return Err(TeacherForwardError::InvalidInput(
                    "MultiTeacher teacher ids must be non-empty".to_owned(),
                ));
            }
            if !seen_ids.insert(entry.id.clone()) {
                return Err(TeacherForwardError::InvalidInput(format!(
                    "MultiTeacher duplicate teacher id '{}'",
                    entry.id
                )));
            }
        }

        let default_index = entries
            .iter()
            .position(|entry| entry.id == default_id)
            .ok_or_else(|| {
                TeacherForwardError::InvalidInput(format!(
                    "MultiTeacher default teacher id '{default_id}' is not registered"
                ))
            })?;

        let vocab_size = entries[0].teacher.vocab_size();
        for entry in entries.iter().skip(1) {
            let entry_vocab = entry.teacher.vocab_size();
            if entry_vocab != vocab_size {
                return Err(TeacherForwardError::InvalidInput(format!(
                    "MultiTeacher requires all teachers to share vocab_size, \
                     got teacher '{}' vocab_size={} but expected {}",
                    entry.id, entry_vocab, vocab_size
                )));
            }
        }

        let mut resolved_routes = Vec::with_capacity(routes.len());
        for route in routes {
            if route.token_prefix.is_empty() {
                return Err(TeacherForwardError::InvalidInput(format!(
                    "MultiTeacher route for teacher '{}' has an empty token prefix; \
                     use the default teacher instead",
                    route.teacher_id
                )));
            }
            let teacher_index = entries
                .iter()
                .position(|entry| entry.id == route.teacher_id)
                .ok_or_else(|| {
                    TeacherForwardError::InvalidInput(format!(
                        "MultiTeacher route references unknown teacher id '{}'",
                        route.teacher_id
                    ))
                })?;
            resolved_routes.push(ResolvedTeacherRoute {
                teacher_index,
                token_prefix: route.token_prefix,
            });
        }

        let mut parameter_ids = Vec::new();
        let mut seen_params = HashSet::new();
        for entry in &entries {
            for &param_id in entry.teacher.parameter_ids() {
                if seen_params.insert(param_id) {
                    parameter_ids.push(param_id);
                }
            }
        }

        Ok(Self {
            entries,
            routes: resolved_routes,
            default_index,
            vocab_size,
            parameter_ids,
        })
    }

    pub fn teacher_count(&self) -> usize {
        self.entries.len()
    }

    pub fn selected_teacher_id(&self, input_ids: &[u32]) -> &str {
        self.entries[self.selected_teacher_index(input_ids)].id()
    }

    fn selected_teacher_index(&self, input_ids: &[u32]) -> usize {
        let mut selected_index = self.default_index;
        let mut selected_prefix_len = 0usize;
        for route in &self.routes {
            let prefix_len = route.token_prefix.len();
            if prefix_len >= selected_prefix_len && input_ids.starts_with(&route.token_prefix) {
                selected_index = route.teacher_index;
                selected_prefix_len = prefix_len;
            }
        }
        selected_index
    }
}

impl TeacherForward for MultiTeacher<'_> {
    fn forward_logits_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<DeviceLogits> {
        let teacher = self.entries[self.selected_teacher_index(input_ids)].teacher;
        teacher.forward_logits_device(input_ids, positions, store, tape)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn parameter_ids(&self) -> &[TensorId] {
        &self.parameter_ids
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
    last_profile: Mutex<InferTeacherProfile>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Default, Clone, Copy)]
pub struct InferTeacherProfile {
    pub total_seconds: f64,
    pub raw_forward_seconds: f64,
    pub sync_seconds: f64,
    pub d2d_bridge_import_seconds: f64,
    pub seq_len: usize,
    pub vocab_size: usize,
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
            last_profile: Mutex::new(InferTeacherProfile::default()),
        }
    }

    pub fn engine(&self) -> &Arc<Mutex<LoadedInferenceEngine>> {
        &self.engine
    }

    pub fn train_backend(&self) -> &Arc<dyn Backend> {
        &self.train_backend
    }

    pub fn last_profile(&self) -> InferTeacherProfile {
        self.last_profile
            .lock()
            .map(|profile| *profile)
            .unwrap_or_default()
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

        let total_started = Instant::now();
        let raw_started = Instant::now();
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
        let raw_forward_seconds = raw_started.elapsed().as_secs_f64();
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

        let sync_started = Instant::now();
        raw_logits
            .device
            .sync()
            .map_err(|err| TeacherForwardError::InferRuntime(err.to_string()))?;
        let sync_seconds = sync_started.elapsed().as_secs_f64();
        let shape = vec![1, raw_logits.seq_len(), raw_logits.vocab_size()];
        let bridge_started = Instant::now();
        let handle = raw_logits.with_logits_device_ptr(|src_ptr| {
            self.train_backend
                .import_bf16_device_ptr_as_f32(src_ptr, raw_logits.logits.len, &shape)
        })?;
        let d2d_bridge_import_seconds = bridge_started.elapsed().as_secs_f64();
        if let Ok(mut profile) = self.last_profile.lock() {
            *profile = InferTeacherProfile {
                total_seconds: total_started.elapsed().as_secs_f64(),
                raw_forward_seconds,
                sync_seconds,
                d2d_bridge_import_seconds,
                seq_len: raw_logits.seq_len(),
                vocab_size: raw_logits.vocab_size(),
            };
        }
        let tensor_id = store.alloc_device_tensor(shape.clone(), handle)?;
        Ok(DeviceLogits { tensor_id, shape })
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::Tensor;

    struct FakeTeacher {
        marker: f32,
        vocab_size: usize,
        parameter_ids: Vec<TensorId>,
    }

    impl FakeTeacher {
        fn new(marker: f32, vocab_size: usize, parameter_ids: Vec<TensorId>) -> Self {
            Self {
                marker,
                vocab_size,
                parameter_ids,
            }
        }
    }

    impl TeacherForward for FakeTeacher {
        fn forward_logits_device(
            &self,
            input_ids: &[u32],
            positions: &[u32],
            store: &mut TensorStore,
            _tape: &mut Tape,
        ) -> Result<DeviceLogits> {
            if input_ids.len() != positions.len() {
                return Err(TeacherForwardError::InvalidInput(
                    "fake teacher input/position mismatch".to_owned(),
                ));
            }
            let shape = vec![1, input_ids.len(), self.vocab_size];
            let mut data = vec![0.0; input_ids.len() * self.vocab_size];
            if let Some(first) = data.first_mut() {
                *first = self.marker;
            }
            let tensor_id = store.alloc(Tensor::new(data, shape.clone(), false)?);
            Ok(DeviceLogits { tensor_id, shape })
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn parameter_ids(&self) -> &[TensorId] {
            &self.parameter_ids
        }
    }

    #[test]
    fn multi_teacher_routes_by_longest_token_prefix() -> Result<()> {
        let mut store = TensorStore::default();
        let mut tape = Tape::default();
        let default_param = store.alloc(Tensor::new(vec![0.0], vec![1], false)?);
        let code_param = store.alloc(Tensor::new(vec![0.0], vec![1], false)?);
        let python_param = store.alloc(Tensor::new(vec![0.0], vec![1], false)?);
        let default_teacher = FakeTeacher::new(1.0, 4, vec![default_param]);
        let code_teacher = FakeTeacher::new(2.0, 4, vec![code_param]);
        let python_teacher = FakeTeacher::new(3.0, 4, vec![python_param, code_param]);

        let router = MultiTeacher::with_routes(
            vec![
                TeacherEntry::new("default", &default_teacher),
                TeacherEntry::new("code", &code_teacher),
                TeacherEntry::new("python", &python_teacher),
            ],
            "default",
            vec![
                TeacherRoute::new("code", vec![7]),
                TeacherRoute::new("python", vec![7, 42]),
            ],
        )?;

        assert_eq!(router.teacher_count(), 3);
        assert_eq!(router.selected_teacher_id(&[7, 42, 9]), "python");
        assert_eq!(router.selected_teacher_id(&[7, 9]), "code");
        assert_eq!(router.selected_teacher_id(&[5, 9]), "default");
        assert_eq!(
            router.parameter_ids(),
            &[default_param, code_param, python_param]
        );

        let routed_logits =
            router.forward_logits_device(&[7, 42, 9], &[0, 1, 2], &mut store, &mut tape)?;
        let routed_host = store.to_host(routed_logits.tensor_id)?;
        assert_eq!(routed_host[0], 3.0);

        Ok(())
    }

    #[test]
    fn multi_teacher_rejects_vocab_mismatch() {
        let teacher_a = FakeTeacher::new(1.0, 4, Vec::new());
        let teacher_b = FakeTeacher::new(2.0, 5, Vec::new());

        let result = MultiTeacher::new(
            vec![
                TeacherEntry::new("a", &teacher_a),
                TeacherEntry::new("b", &teacher_b),
            ],
            "a",
        );
        assert!(result.is_err(), "vocab mismatch must be rejected");
        let err = result.err().expect("checked above");
        assert!(err.to_string().contains("vocab_size"));
    }
}
