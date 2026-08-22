//! Two teacher sources behind one train-side contract: the in-process
//! `Qwen35Model` teacher and an `infer` runtime teacher. `DeviceLogits`
//! intentionally carries a `TensorId` in the caller's `TensorStore` so the KL
//! path can stay on the same backend without a host materialization.

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
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

    /// Run the full-sequence forward and return the hidden states (not logits).
    /// Used by the windowed KL path to avoid re-running the growing prefix
    /// for each window: one full forward, then `logits_from_hidden_window_device`
    /// per window.
    fn forward_hidden_device(
        &self,
        _input_ids: &[u32],
        _positions: &[u32],
        _store: &mut TensorStore,
        _tape: &mut Tape,
    ) -> Result<TensorId> {
        Err(TeacherForwardError::InvalidInput(
            "forward_hidden_device not supported by this teacher".to_owned(),
        ))
    }

    fn logits_from_hidden_window_device(
        &self,
        _hidden: TensorId,
        _window: SequenceWindow,
        _store: &mut TensorStore,
        _tape: &mut Tape,
    ) -> Result<DeviceLogits> {
        Err(TeacherForwardError::InvalidInput(
            "logits_from_hidden_window_device not supported by this teacher".to_owned(),
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
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
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
        into_device_logits(store, tensor_id)
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
        into_device_logits(store, tensor_id)
    }

    fn forward_hidden_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let token_indices = input_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let pos = positions.iter().map(|&id| id as usize).collect::<Vec<_>>();
        Ok(self.model.forward_hidden_freeing_intermediates(
            store,
            tape,
            &token_indices,
            &pos,
            1,
            crate::context_parallel::CpContext::single(),
        )?)
    }

    fn logits_from_hidden_window_device(
        &self,
        hidden: TensorId,
        window: SequenceWindow,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<DeviceLogits> {
        let tensor_id = self
            .model
            .logits_from_hidden_window(store, tape, hidden, window)?;
        into_device_logits(store, tensor_id)
    }

    fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    fn parameter_ids(&self) -> &[TensorId] {
        &self.parameter_ids
    }
}

fn into_device_logits(store: &mut TensorStore, tensor_id: TensorId) -> Result<DeviceLogits> {
    store.ensure_device(tensor_id)?;
    let shape = store
        .get(tensor_id)
        .ok_or(AutogradError::InvalidTensorId(tensor_id))?
        .shape
        .clone();
    Ok(DeviceLogits { tensor_id, shape })
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

        // The bridge is stream-ordered on the logits' own stream (after the
        // lm_head GEMM), so the old pre-bridge device sync is redundant: the
        // D2D copy and the source's later free are both ordered on that
        // stream. `sync_seconds` stays in the profile for compatibility, 0.0.
        let sync_seconds = 0.0;
        let shape = vec![1, raw_logits.seq_len(), raw_logits.vocab_size()];
        let bridge_started = Instant::now();
        // The logits buffer is bound to the engine's compute stream; passing it
        // lets the bridge order the D2D copy and the source's later free on
        // that stream, skipping the bridge's context sync.
        let src_stream = raw_logits.device.stream.cu_stream() as u64;
        let handle = raw_logits.with_logits_device_ptr(|src_ptr| {
            self.train_backend.import_bf16_device_ptr_as_f32(
                src_ptr,
                src_stream,
                raw_logits.logits.len,
                &shape,
            )
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
