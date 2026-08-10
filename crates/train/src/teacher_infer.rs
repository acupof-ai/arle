//! Teacher-forward abstraction for OPD.
//!
//! Phase 2 of the large-to-small OPD path needs two teacher sources behind
//! the same train-side contract: the existing in-process `Qwen35Model` teacher
//! and, next, an `infer` runtime teacher. `DeviceLogits` intentionally carries
//! a `TensorId` in the caller's `TensorStore` so the KL path can stay on the
//! same backend without a host materialization.

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
    collections::HashSet,
    io::Read,
    sync::Mutex,
    time::{Duration, Instant},
};

#[cfg(feature = "cuda")]
use autograd::Backend;
#[cfg(feature = "cuda")]
use autograd::ops::slice;
use autograd::{AutogradError, Tape, Tensor, TensorId, TensorStore};
use half::bf16;
#[cfg(feature = "cuda")]
use infer_api::LoadedInferenceEngine;
use serde::Serialize;

use crate::qwen35::{Qwen35Error, Qwen35Model, SequenceWindow};

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
    #[error("API teacher forward failed: {0}")]
    ApiRuntime(String),
    #[error("API teacher logits decode failed: {0}")]
    ApiDecode(String),
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

    fn forward_logits_window_device(
        &self,
        _input_ids: &[u32],
        _positions: &[u32],
        _window: SequenceWindow,
        _store: &mut TensorStore,
        _tape: &mut Tape,
    ) -> Result<DeviceLogits> {
        Err(TeacherForwardError::InvalidInput(
            "--logits-window-size requires a teacher implementation that can \
             return the requested logits window on device"
                .to_owned(),
        ))
    }

    fn vocab_size(&self) -> usize;

    fn parameter_ids(&self) -> &[TensorId] {
        &[]
    }

    fn offload_engine_weights(&self) -> Result<usize> {
        Ok(0)
    }

    fn reload_engine_weights(&self) -> Result<()> {
        Ok(())
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

        let resolved_routes = routes
            .into_iter()
            .map(|route| {
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
                Ok(ResolvedTeacherRoute {
                    teacher_index,
                    token_prefix: route.token_prefix,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut seen_params = HashSet::new();
        let parameter_ids: Vec<_> = entries
            .iter()
            .flat_map(|entry| entry.teacher.parameter_ids())
            .filter(|id| seen_params.insert(*id))
            .copied()
            .collect();

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

    fn forward_logits_window_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        window: SequenceWindow,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<DeviceLogits> {
        let teacher = self.entries[self.selected_teacher_index(input_ids)].teacher;
        TeacherForward::forward_logits_window_device(
            teacher, input_ids, positions, window, store, tape,
        )
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn parameter_ids(&self) -> &[TensorId] {
        &self.parameter_ids
    }

    fn offload_engine_weights(&self) -> Result<usize> {
        let mut freed = 0usize;
        for entry in &self.entries {
            freed += entry.teacher.offload_engine_weights()?;
        }
        Ok(freed)
    }

    fn reload_engine_weights(&self) -> Result<()> {
        for entry in &self.entries {
            entry.teacher.reload_engine_weights()?;
        }
        Ok(())
    }
}

pub struct ApiTeacher {
    endpoint: String,
    api_key: Option<String>,
    request_dtype: String,
    vocab_size: usize,
    agent: ureq::Agent,
    last_profile: Mutex<ApiTeacherProfile>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ApiTeacherProfile {
    pub total_seconds: f64,
    pub http_seconds: f64,
    pub decode_seconds: f64,
    pub upload_seconds: f64,
    pub seq_len: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Serialize)]
struct ApiTeacherRequest<'a> {
    input_ids: &'a [u32],
    positions: &'a [u32],
    dtype: &'a str,
}

impl ApiTeacher {
    pub fn new(endpoint: impl Into<String>, vocab_size: usize) -> Self {
        Self::with_timeout(endpoint, vocab_size, Duration::from_secs(30))
    }

    pub fn with_timeout(endpoint: impl Into<String>, vocab_size: usize, timeout: Duration) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            request_dtype: "bf16".to_owned(),
            vocab_size,
            agent: ureq::AgentBuilder::new().timeout(timeout).build(),
            last_profile: Mutex::new(ApiTeacherProfile::default()),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_request_dtype(mut self, dtype: impl Into<String>) -> Self {
        self.request_dtype = dtype.into();
        self
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn last_profile(&self) -> ApiTeacherProfile {
        self.last_profile
            .lock()
            .map(|profile| *profile)
            .unwrap_or_default()
    }

    /// POST tokens/positions, return raw bf16-LE logits bytes + `[rows, cols]`.
    /// Body is `application/octet-stream` (~1 GB) — base64+JSON dominated the step.
    fn post_logits(&self, input_ids: &[u32], positions: &[u32]) -> Result<(Vec<u8>, [usize; 2])> {
        let body = ApiTeacherRequest {
            input_ids,
            positions,
            dtype: self.request_dtype.as_str(),
        };
        let body = serde_json::to_value(&body).map_err(|err| {
            TeacherForwardError::ApiRuntime(format!("serialize token-logits request: {err}"))
        })?;
        let mut request = self
            .agent
            .post(&self.endpoint)
            .set("Content-Type", "application/json");
        if let Some(api_key) = self.api_key.as_deref() {
            request = request.set("Authorization", &format!("Bearer {api_key}"));
        }
        let resp = request.send_json(body).map_err(|err| {
            TeacherForwardError::ApiRuntime(format!(
                "POST {} failed: {err}. Hint: the API teacher endpoint must accept \
                 {{input_ids, positions, dtype}} and return raw bf16 token logits.",
                self.endpoint
            ))
        })?;
        let rows: usize = resp
            .header("x-logits-rows")
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| {
                TeacherForwardError::ApiDecode("response missing x-logits-rows header".to_owned())
            })?;
        let cols: usize = resp
            .header("x-logits-cols")
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| {
                TeacherForwardError::ApiDecode("response missing x-logits-cols header".to_owned())
            })?;
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        resp.into_reader().read_to_end(&mut bytes).map_err(|err| {
            TeacherForwardError::ApiRuntime(format!(
                "read raw-logits body from {} failed: {err}",
                self.endpoint
            ))
        })?;
        Ok((bytes, [rows, cols]))
    }
}

impl TeacherForward for ApiTeacher {
    fn forward_logits_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        _tape: &mut Tape,
    ) -> Result<DeviceLogits> {
        if input_ids.is_empty() {
            return Err(TeacherForwardError::InvalidInput(
                "ApiTeacher requires a non-empty token sequence".to_owned(),
            ));
        }
        if input_ids.len() != positions.len() {
            return Err(TeacherForwardError::InvalidInput(format!(
                "ApiTeacher token/position length mismatch: tokens={} positions={}",
                input_ids.len(),
                positions.len()
            )));
        }

        let total_started = Instant::now();
        let http_started = Instant::now();
        let (bytes, raw_shape) = self.post_logits(input_ids, positions)?;
        let http_seconds = http_started.elapsed().as_secs_f64();
        let decode_started = Instant::now();
        let shape = normalize_api_teacher_shape(&raw_shape, input_ids.len(), self.vocab_size)?;
        let logits = decode_bf16_le_logits(&bytes, shape_size(&shape))?;
        let decode_seconds = decode_started.elapsed().as_secs_f64();
        let upload_started = Instant::now();
        let tensor_id = store.alloc(Tensor::new(logits, shape.clone(), false)?);
        store.ensure_device(tensor_id)?;
        let upload_seconds = upload_started.elapsed().as_secs_f64();
        if let Ok(mut profile) = self.last_profile.lock() {
            *profile = ApiTeacherProfile {
                total_seconds: total_started.elapsed().as_secs_f64(),
                http_seconds,
                decode_seconds,
                upload_seconds,
                seq_len: input_ids.len(),
                vocab_size: self.vocab_size,
            };
        }
        Ok(DeviceLogits { tensor_id, shape })
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

fn normalize_api_teacher_shape(
    shape: &[usize],
    seq_len: usize,
    vocab_size: usize,
) -> Result<Vec<usize>> {
    match shape {
        [seq, vocab] if *seq == seq_len && *vocab == vocab_size => Ok(vec![1, seq_len, vocab_size]),
        [batch, seq, vocab] if *batch == 1 && *seq == seq_len && *vocab == vocab_size => {
            Ok(vec![1, seq_len, vocab_size])
        }
        _ => Err(TeacherForwardError::InvalidInput(format!(
            "API teacher logits shape mismatch: got {:?}, expected [seq_len={}, vocab_size={}] \
             or [1, seq_len={}, vocab_size={}].",
            shape, seq_len, vocab_size, seq_len, vocab_size
        ))),
    }
}

fn decode_bf16_le_logits(bytes: &[u8], expected_len: usize) -> Result<Vec<f32>> {
    if bytes.len() != expected_len * std::mem::size_of::<u16>() {
        return Err(TeacherForwardError::ApiDecode(format!(
            "bf16 logits byte length mismatch: got {}, expected {}",
            bytes.len(),
            expected_len * std::mem::size_of::<u16>()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect())
}

fn shape_size(shape: &[usize]) -> usize {
    shape.iter().copied().product()
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

    /// Build a teacher exposing an explicit parameter-id list (not the model's
    /// full `all_parameter_ids()`). SOPD's EMA self-teacher shares the LoRA
    /// student's base, so `all_parameter_ids()` includes the student's trainable
    /// adapter ids. The OPD step rejects teacher params with `requires_grad=true`,
    /// so the EMA teacher must expose only its own frozen params (base + EMA
    /// adapter), excluding the student adapter.
    pub fn with_parameter_ids(model: &'a Qwen35Model, parameter_ids: Vec<TensorId>) -> Self {
        Self {
            model,
            parameter_ids,
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

    fn forward_logits_window_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        window: SequenceWindow,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<DeviceLogits> {
        let tensor_id = self
            .model
            .forward_logits_window(store, tape, input_ids, positions, window)?;
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
#[derive(Clone)]
pub struct InferTeacher {
    engine: Arc<Mutex<LoadedInferenceEngine>>,
    train_backend: Arc<dyn Backend>,
    vocab_size: usize,
    last_profile: Arc<Mutex<InferTeacherProfile>>,
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
            last_profile: Arc::new(Mutex::new(InferTeacherProfile::default())),
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

    fn forward_logits_window_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        window: SequenceWindow,
        store: &mut TensorStore,
        tape: &mut Tape,
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
        if window.start >= window.end || window.end > input_ids.len() {
            return Err(TeacherForwardError::InvalidInput(format!(
                "InferTeacher logits window [{}, {}) is invalid for seq_len={}",
                window.start,
                window.end,
                input_ids.len()
            )));
        }

        let full_logits = self.forward_logits_device(input_ids, positions, store, tape)?;
        let tensor_id = slice(
            full_logits.tensor_id,
            &[0, window.start, 0],
            &[1, window.end, self.vocab_size],
            store,
            tape,
        )?;
        let shape = vec![1, window.end - window.start, self.vocab_size];
        Ok(DeviceLogits { tensor_id, shape })
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn offload_engine_weights(&self) -> Result<usize> {
        let engine = self.engine.lock().map_err(|err| {
            TeacherForwardError::InferRuntime(format!(
                "LoadedInferenceEngine lock poisoned before offload: {err}"
            ))
        })?;
        engine
            .offload_engine_weights()
            .map_err(|err| TeacherForwardError::InferRuntime(err.to_string()))
    }

    fn reload_engine_weights(&self) -> Result<()> {
        let engine = self.engine.lock().map_err(|err| {
            TeacherForwardError::InferRuntime(format!(
                "LoadedInferenceEngine lock poisoned before reload: {err}"
            ))
        })?;
        engine
            .reload_engine_weights()
            .map_err(|err| TeacherForwardError::InferRuntime(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::Tensor;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

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

    #[test]
    fn decode_bf16_le_logits_bulk() -> Result<()> {
        let values = [
            bf16::from_f32(1.25),
            bf16::from_f32(-2.5),
            bf16::from_f32(3.75),
            bf16::from_f32(4.0),
        ];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        let decoded = decode_bf16_le_logits(&bytes, 4)?;
        assert_eq!(decoded, [1.25, -2.5, 3.75, 4.0]);
        assert!(decode_bf16_le_logits(&bytes, 3).is_err());
        Ok(())
    }

    #[test]
    fn api_teacher_accepts_multitoken_full_sequence_shape() -> Result<()> {
        let shape = normalize_api_teacher_shape(&[3, 4], 3, 4)?;
        assert_eq!(shape, vec![1, 3, 4]);
        Ok(())
    }

    #[test]
    fn api_teacher_rejects_last_row_shape_for_multitoken_sequence() {
        let err = normalize_api_teacher_shape(&[1, 4], 3, 4)
            .expect_err("last-row logits must not masquerade as full sequence logits");
        assert!(err.to_string().contains("shape mismatch"));
        assert!(err.to_string().contains("seq_len=3"));
    }

    #[test]
    fn api_teacher_fetches_http_logits_into_tensor_store()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let endpoint = format!("http://{}", listener.local_addr()?);
        let server = thread::spawn(move || -> std::result::Result<(), String> {
            let (mut stream, _) = listener.accept().map_err(|err| err.to_string())?;
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            let mut header_end = None;
            loop {
                let read = stream.read(&mut buffer).map_err(|err| err.to_string())?;
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                    header_end = Some(index + 4);
                    break;
                }
            }
            let header_end =
                header_end.ok_or_else(|| "missing HTTP header terminator".to_owned())?;
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.strip_prefix("Content-Length:")
                        .or_else(|| line.strip_prefix("content-length:"))
                        .and_then(|value| value.trim().parse::<usize>().ok())
                })
                .ok_or_else(|| format!("missing Content-Length header: {headers}"))?;
            let target_len = header_end + content_length;
            while request.len() < target_len {
                let read = stream.read(&mut buffer).map_err(|err| err.to_string())?;
                if read == 0 {
                    return Err(format!(
                        "request body ended early: got {} bytes, expected {}",
                        request.len().saturating_sub(header_end),
                        content_length
                    ));
                }
                request.extend_from_slice(&buffer[..read]);
            }
            let request_body = String::from_utf8_lossy(&request[header_end..target_len]);
            if !request_body.contains("\"input_ids\":[11]") {
                return Err(format!(
                    "request missing expected input_ids: {request_body}"
                ));
            }
            if !request_body.contains("\"positions\":[0]") {
                return Err(format!(
                    "request missing expected positions: {request_body}"
                ));
            }
            let values = [0.5f32, 1.5, -2.0, 4.25];
            let body: Vec<u8> = values
                .iter()
                .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
                .collect();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\
                 x-logits-rows: 1\r\nx-logits-cols: 4\r\nx-logits-dtype: bf16\r\n\
                 Content-Length: {}\r\n\r\n",
                body.len(),
            );
            stream
                .write_all(response.as_bytes())
                .map_err(|err| err.to_string())?;
            stream.write_all(&body).map_err(|err| err.to_string())?;
            Ok(())
        });

        let teacher =
            ApiTeacher::with_timeout(endpoint, 4, Duration::from_secs(5)).with_request_dtype("f32");
        let mut store = TensorStore::default();
        let mut tape = Tape::default();
        let logits = teacher.forward_logits_device(&[11], &[0], &mut store, &mut tape)?;
        let host = store.to_host(logits.tensor_id)?;
        let profile = teacher.last_profile();

        assert_eq!(logits.shape, vec![1, 1, 4]);
        assert_eq!(host, [0.5, 1.5, -2.0, 4.25]);
        assert_eq!(profile.seq_len, 1);
        assert_eq!(profile.vocab_size, 4);
        server
            .join()
            .map_err(|_| "mock API teacher server thread panicked")?
            .map_err(|err| format!("mock API teacher server failed: {err}"))?;
        Ok(())
    }
}
