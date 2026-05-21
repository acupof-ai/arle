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
    sync::Mutex,
    time::{Duration, Instant},
};

#[cfg(feature = "cuda")]
use autograd::Backend;
use autograd::{AutogradError, Tape, Tensor, TensorId, TensorStore};
use base64::{engine::general_purpose, Engine as _};
use half::bf16;
#[cfg(feature = "cuda")]
use infer::server_engine::LoadedInferenceEngine;
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Deserialize)]
struct ApiTeacherResponse {
    shape: Vec<usize>,
    dtype: String,
    logits: Option<Vec<f32>>,
    logits_b64: Option<String>,
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

    fn post_logits(&self, input_ids: &[u32], positions: &[u32]) -> Result<ApiTeacherResponse> {
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
        request
            .send_json(body)
            .map_err(|err| {
                TeacherForwardError::ApiRuntime(format!(
                    "POST {} failed: {err}. Hint: the API teacher endpoint must accept \
                     {{input_ids, positions, dtype}} and return full token logits.",
                    self.endpoint
                ))
            })?
            .into_json()
            .map_err(|err| {
                TeacherForwardError::ApiRuntime(format!(
                    "decode JSON response from {} failed: {err}",
                    self.endpoint
                ))
            })
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
        let response = self.post_logits(input_ids, positions)?;
        let http_seconds = http_started.elapsed().as_secs_f64();
        let decode_started = Instant::now();
        let shape = normalize_api_teacher_shape(&response.shape, input_ids.len(), self.vocab_size)?;
        let logits = decode_api_teacher_logits(&response, shape_size(&shape))?;
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

fn decode_api_teacher_logits(
    response: &ApiTeacherResponse,
    expected_len: usize,
) -> Result<Vec<f32>> {
    if let Some(logits) = response.logits.as_ref() {
        if logits.len() != expected_len {
            return Err(TeacherForwardError::ApiDecode(format!(
                "JSON logits length mismatch: got {}, expected {}",
                logits.len(),
                expected_len
            )));
        }
        return Ok(logits.clone());
    }

    let encoded = response.logits_b64.as_deref().ok_or_else(|| {
        TeacherForwardError::ApiDecode(
            "response must include either `logits` or `logits_b64`".to_owned(),
        )
    })?;
    let bytes = general_purpose::STANDARD.decode(encoded).map_err(|err| {
        TeacherForwardError::ApiDecode(format!("base64 decode logits_b64 failed: {err}"))
    })?;
    match response.dtype.to_ascii_lowercase().as_str() {
        "f32" | "float32" => decode_f32_le_logits(&bytes, expected_len),
        "bf16" | "bfloat16" => decode_bf16_le_logits(&bytes, expected_len),
        dtype => Err(TeacherForwardError::ApiDecode(format!(
            "unsupported logits dtype '{dtype}', expected f32 or bf16"
        ))),
    }
}

fn decode_f32_le_logits(bytes: &[u8], expected_len: usize) -> Result<Vec<f32>> {
    if bytes.len() != expected_len * std::mem::size_of::<f32>() {
        return Err(TeacherForwardError::ApiDecode(format!(
            "f32 logits byte length mismatch: got {}, expected {}",
            bytes.len(),
            expected_len * std::mem::size_of::<f32>()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
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

    #[test]
    fn api_teacher_decodes_f32_base64_logits() -> Result<()> {
        let values = [1.25f32, -2.5, 3.75, 4.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let response = ApiTeacherResponse {
            shape: vec![1, 4],
            dtype: "f32".to_owned(),
            logits: None,
            logits_b64: Some(general_purpose::STANDARD.encode(bytes)),
        };

        let shape = normalize_api_teacher_shape(&response.shape, 1, 4)?;
        let decoded = decode_api_teacher_logits(&response, shape_size(&shape))?;

        assert_eq!(shape, vec![1, 1, 4]);
        assert_eq!(decoded, values);
        Ok(())
    }

    #[test]
    fn api_teacher_decodes_bf16_base64_logits() -> Result<()> {
        let values = [
            bf16::from_f32(1.25),
            bf16::from_f32(-2.5),
            bf16::from_f32(3.75),
            bf16::from_f32(4.0),
        ];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let response = ApiTeacherResponse {
            shape: vec![1, 1, 4],
            dtype: "bf16".to_owned(),
            logits: None,
            logits_b64: Some(general_purpose::STANDARD.encode(bytes)),
        };

        let shape = normalize_api_teacher_shape(&response.shape, 1, 4)?;
        let decoded = decode_api_teacher_logits(&response, shape_size(&shape))?;

        assert_eq!(shape, vec![1, 1, 4]);
        assert_eq!(decoded, [1.25, -2.5, 3.75, 4.0]);
        Ok(())
    }
}
