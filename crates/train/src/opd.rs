//! On-Policy Distillation step function.
//!
//! Per the 2026-05-18 OPD-only pivot, ARLE's one in-tree training surface
//! is OPD: a frozen teacher `Qwen35Model` and a trainable student
//! `Qwen35Model` (optionally LoRA-adapted) share a single `TensorStore`;
//! the student samples a rollout greedily, the teacher re-scores the
//! same rollout, and the forward-KL distill loss drives backward through
//! the student parameters.
//!
//! Smoke pattern (`crates/train/tests/test_opd_step.rs`):
//! - Two `Qwen35Model::new` calls into the same store, the teacher copy
//!   pinned via `clone_frozen` so its parameter ids report
//!   `requires_grad = false`.
//! - `opd_step` is invoked per training step; on return, the tape and
//!   ephemeral tensors are pruned by the function itself.
//!
//! Production wiring (`crates/cli/src/train_cli.rs::run_opd`):
//! - Teacher loaded from a separate HF/ModelScope checkpoint via
//!   `crates/train/src/qwen35_checkpoint.rs`.
//! - Student initialised from a smaller checkpoint with LoRA adapter
//!   layered on via `Qwen35Model::new_with_lora`.

use autograd::{
    AutogradError, BackwardOp, BackwardProfile, Device, Tape, TensorId, TensorStore,
    optim::{AdamW, Optimizer},
    tensor::Dirty,
};
use infer_plan::{SamplingParams, sample_token};
use std::{
    collections::HashSet,
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use crate::{
    causal_lm::live_tensor_ids,
    grad_clip::{FiniteStepError, finite_optimizer_step},
    loss::{
        DEFAULT_KL_CHUNK_SIZE, KlDirection, cross_entropy_loss, fused_linear_distill_loss,
        generalized_beta_jsd_loss, generalized_beta_jsd_loss_chunked, kl_distill_loss,
        kl_distill_loss_chunked,
    },
    qwen35::{
        Qwen35Error, Qwen35KvCache, Qwen35Model, SequenceWindow, forward_rollout_cached,
        forward_rollout_cached_device_token,
    },
    teacher_infer::{InProcessTeacher, TeacherForward, TeacherForwardError},
    trainer::{
        cleanup_after_backward, extend_keep_with_params_and_grads, retained_param_and_grad_ids,
    },
};
#[cfg(feature = "cuda")]
use crate::{infer_student::InferStudent, lora::LoraConfig};
use autograd::ops::{
    add, embedding,
    fused_linear_distill::{
        PgStats, WeightForm, fused_linear_ce_loss_indexed, fused_linear_pg_loss_indexed,
    },
    gather_last_dim, log_softmax, matmul_bt, mean, mul, mul_scalar, reshape, slice,
};

/// Routes the OPD rollout through the in-process infer engine (`InferStudent`)
/// instead of the train-crate decode. Built only when the infer arm is selected
/// and an `InferStudent` loaded; `None` → train-crate rollout (A/B baseline).
#[cfg(feature = "cuda")]
pub struct InferRolloutCtx<'a> {
    /// In-process infer student engine (LoRA-synced from the train store).
    pub student: &'a InferStudent,
    /// Train LoRA config (rank / alpha) used to export the adapter for sync.
    pub lora_config: LoraConfig,
}

/// True when the infer-engine rollout path is selected (the default). The
/// in-process infer student (CUDA graph + paged KV) is 4.99× faster than the
/// train-crate O(n²) decode (`wins/2026-05-29-opd-infer-rollout-default-p4.md`).
/// Set by `--rollout-engine {infer,train}`; unset → `infer`.
#[cfg(feature = "cuda")]
pub fn infer_rollout_flag_enabled() -> bool {
    INFER_ROLLOUT_OVERRIDE.get().copied().unwrap_or(true)
}

/// `--rollout-engine` selection; unset defaults to `infer`.
#[cfg(feature = "cuda")]
static INFER_ROLLOUT_OVERRIDE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Install the `--rollout-engine` selection (idempotent, first write wins).
#[cfg(feature = "cuda")]
pub fn set_infer_rollout_override(use_infer: bool) {
    let _ = INFER_ROLLOUT_OVERRIDE.set(use_infer);
}

/// OPD engine weight time-share mode (`--engine-offload`). When on, the
/// idle infer engines offload their device weights to host RAM during the
/// student backward, freeing VRAM so long rollouts (≥256) fit on a 16 GB card.
///
/// The vLLM-sleep-mode / verl HybridFlow equivalent ARLE previously lacked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineOffloadMode {
    /// Default — resident-weights path is unchanged.
    Off,
    /// Offload BOTH idle engines (rollout student + scoring teacher) together
    /// after the teacher scores. Frees the most VRAM (~4.4 GB at rollout-256)
    /// but reloads the teacher every step, which races the shared async pool
    /// across the three co-resident CUDA contexts (step-2 illegal address —
    /// see `errors/2026-05-30-...`).
    All,
    /// Offload ONLY the rollout infer-student; keep the teacher resident. The
    /// teacher is never reloaded, so the step-2 teacher-reload path is avoided.
    /// Frees ~1.4 GB — enough for moderate rollout-256 backwards but can OOM on
    /// the longest CoTs (the teacher's 3 GB stays put).
    Student,
    /// Offload ONLY the scoring teacher; keep the rollout infer-student
    /// resident. Frees the teacher's ~3.0 GB (more headroom than `Student`) and
    /// — because only ONE engine offloads/reloads — keeps the shared async pool
    /// free of the student↔teacher offload interleaving that corrupts the W4A8
    /// Marlin reload under `All`. Matches the single-engine offload/reload
    /// parity conditions, so the teacher's Marlin side buffers round-trip.
    Teacher,
}

impl EngineOffloadMode {
    /// True when any engine offload is active (student and/or teacher).
    pub fn is_enabled(self) -> bool {
        !matches!(self, EngineOffloadMode::Off)
    }

    /// True when the scoring teacher participates in the offload/reload
    /// time-share. `Student` keeps the teacher resident.
    pub fn offloads_teacher(self) -> bool {
        matches!(self, EngineOffloadMode::All | EngineOffloadMode::Teacher)
    }

    /// True when the rollout infer-student participates in the offload/reload
    /// time-share. `Teacher` keeps the student resident.
    pub fn offloads_student(self) -> bool {
        matches!(self, EngineOffloadMode::All | EngineOffloadMode::Student)
    }
}

/// The OPD engine offload mode (`--engine-offload off|all|student|teacher`;
/// `teacher` keeps the student resident — frees ~3 GB and avoids the
/// multi-engine pool interleaving that corrupts the W4A8 Marlin reload under
/// `all`).
#[cfg(feature = "cuda")]
pub fn engine_offload_mode() -> EngineOffloadMode {
    crate::runtime_flags::engine_offload()
}

#[derive(Debug, thiserror::Error)]
pub enum OpdError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    FiniteStep(#[from] FiniteStepError),
    #[error(transparent)]
    Qwen35(#[from] Qwen35Error),
    #[error("{0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, OpdError>;

#[derive(Debug, Clone)]
pub struct OpdStepConfig {
    /// Tokens to roll out greedily from the student starting from the prompt.
    pub rollout_len: usize,
    /// Optional rollout sampling. `None` keeps the existing greedy argmax path.
    pub rollout_sampling: Option<SamplingParams>,
    /// Gradient L2 norm clip threshold.
    pub grad_clip: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct OpdStepOutcome {
    pub loss: f32,
    pub rollout_len: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OpdStepProfile {
    pub total_seconds: f64,
    pub student_rollout_seconds: f64,
    pub teacher_forward_seconds: f64,
    pub student_forward_seconds: f64,
    pub kl_loss_seconds: f64,
    pub optimizer_zero_grad_seconds: f64,
    pub backward_seconds: f64,
    pub grad_clip_seconds: f64,
    pub optimizer_step_seconds: f64,
    pub post_step_cleanup_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GkdSftAnchor {
    StudentRollout,
    CorpusTruth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpdKlMask {
    Full,
    CompletionOnly,
}

/// Default Route-B logits window. Keep it aligned with the KL chunk size so
/// each backward materializes at most one `[window, vocab]` logits tile.
pub const DEFAULT_LOGITS_WINDOW_SIZE: usize = DEFAULT_KL_CHUNK_SIZE;

#[derive(Debug, Clone, Copy)]
pub struct GkdLossConfig<'a> {
    pub lambda: f32,
    pub sft_anchor: GkdSftAnchor,
    pub corpus_tokens: Option<&'a [u32]>,
    pub kl_chunk_size: Option<usize>,
    pub kl_direction: KlDirection,
    pub kl_temperature: f32,
    pub kl_beta: Option<f32>,
    pub teacher_topk: Option<usize>,
    pub fused_distill: bool,
    pub logits_window_size: Option<usize>,
    pub kl_mask: OpdKlMask,
}

impl Default for GkdLossConfig<'_> {
    fn default() -> Self {
        Self {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: Some(DEFAULT_KL_CHUNK_SIZE),
            kl_direction: KlDirection::Forward,
            kl_temperature: 1.0,
            kl_beta: None,
            teacher_topk: None,
            // Dense logits+KL by default: the fused lm_head+loss path ran the
            // lm_head on the HOST (~205 s/step for the 27B, GPU idle) vs ~3.9 s
            // GPU-bound dense. Opt into fused only for windows too large to
            // materialize [window, vocab]. errors/2026-06-23-opd-fused-distill-default-host-bound.
            fused_distill: false,
            logits_window_size: None,
            kl_mask: OpdKlMask::CompletionOnly,
        }
    }
}

fn record_profile(
    profile: &mut Option<&mut OpdStepProfile>,
    update: impl FnOnce(&mut OpdStepProfile),
) {
    if let Some(profile) = profile.as_deref_mut() {
        update(profile);
    }
}

fn opd_step_trace_enabled() -> bool {
    match std::env::var("ARLE_OPD_STEP_TRACE") {
        Ok(value) => !(value == "0" || value.eq_ignore_ascii_case("false")),
        Err(_) => false,
    }
}

fn log_opd_step_trace(step_started: Instant, event: &str, detail: impl AsRef<str>) {
    if opd_step_trace_enabled() {
        eprintln!(
            "opd_step_trace event={event} elapsed_seconds={:.6} {}",
            step_started.elapsed().as_secs_f64(),
            detail.as_ref()
        );
    }
}

fn log_opd_window_trace(
    kind: &str,
    event: &str,
    index: usize,
    window_started: Instant,
    detail: impl AsRef<str>,
) {
    if opd_step_trace_enabled() {
        eprintln!(
            "opd_window_trace kind={kind} event={event} index={index} \
             elapsed_seconds={:.6} {}",
            window_started.elapsed().as_secs_f64(),
            detail.as_ref()
        );
    }
}

static OPD_BACKWARD_PROFILE_WINDOWS: AtomicU64 = AtomicU64::new(0);
static OPD_BACKWARD_PROFILE_TOTALS: LazyLock<Mutex<BackwardProfile>> =
    LazyLock::new(|| Mutex::new(BackwardProfile::default()));

fn opd_backward_profile_enabled() -> bool {
    std::env::var_os("ARLE_OPD_BACKWARD_PROFILE").is_some()
}

fn backward_with_optional_profile(
    loss: TensorId,
    loss_value: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<()> {
    store.backend().trim_memory_pool()?;
    if !opd_backward_profile_enabled() {
        tape.backward_accumulate_only(loss, store)?;
        return Ok(());
    }

    let backward_profile = tape.backward_accumulate_only_profiled(loss, store)?;
    log_opd_backward_profile(loss_value, &backward_profile);
    Ok(())
}

fn log_opd_backward_profile(loss_value: f32, profile: &BackwardProfile) {
    let window_index = OPD_BACKWARD_PROFILE_WINDOWS.fetch_add(1, Ordering::Relaxed) + 1;
    print_backward_profile("window", window_index, loss_value, profile);

    let aggregate = {
        let mut aggregate = OPD_BACKWARD_PROFILE_TOTALS
            .lock()
            .expect("OPD backward profile mutex poisoned");
        aggregate.merge(profile);
        aggregate.clone()
    };
    print_backward_profile("aggregate", window_index, loss_value, &aggregate);
}

fn print_backward_profile(
    scope: &str,
    window_index: u64,
    loss_value: f32,
    profile: &BackwardProfile,
) {
    let total_secs = profile.total_duration.as_secs_f64();
    let op_secs = profile.total_op_duration().as_secs_f64();
    let merge_secs = profile.merge_grad_duration.as_secs_f64();
    let prelude_secs = profile.prelude_duration.as_secs_f64();
    let unattributed_secs = (total_secs - op_secs - merge_secs - prelude_secs).max(0.0);
    eprintln!(
        "opd_backward_profile scope={scope} windows={window_index} loss={loss_value:.12e} \
         total_seconds={total_secs:.6} op_seconds={op_secs:.6} \
         merge_grad_seconds={merge_secs:.6} prelude_seconds={prelude_secs:.6} \
         unattributed_seconds={unattributed_secs:.6}"
    );

    let mut rows = profile
        .op_totals
        .iter()
        .map(|(&op, stats)| (op, stats.count, stats.duration.as_secs_f64()))
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    for (rank, (op, count, seconds)) in rows.iter().enumerate() {
        let pct_backward = if total_secs == 0.0 {
            0.0
        } else {
            seconds / total_secs * 100.0
        };
        eprintln!(
            "opd_backward_op_profile scope={scope} windows={window_index} rank={} op={} \
             count={} seconds={seconds:.6} pct_backward={pct_backward:.3}",
            rank + 1,
            op.name(),
            count
        );
    }

    let mut site_rows = profile
        .site_totals
        .iter()
        .filter_map(|(&(op, site), stats)| {
            (op == BackwardOp::MatmulBT).then_some((op, site, stats.count, stats.duration))
        })
        .collect::<Vec<_>>();
    site_rows.sort_by(|a, b| {
        b.3.cmp(&a.3)
            .then_with(|| a.1.cmp(b.1))
            .then_with(|| a.0.cmp(&b.0))
    });
    for (rank, (op, site, count, duration)) in site_rows.iter().enumerate() {
        let seconds = duration.as_secs_f64();
        let pct_backward = if total_secs == 0.0 {
            0.0
        } else {
            seconds / total_secs * 100.0
        };
        eprintln!(
            "opd_backward_site_profile scope={scope} windows={window_index} rank={} op={} \
             site={} count={} seconds={seconds:.6} pct_backward={pct_backward:.3}",
            rank + 1,
            op.name(),
            site,
            count
        );
    }
}

/// Greedy-argmax the last-position row of a `[1, seq_len, vocab]` logits
/// tensor. The student's lm-head returns dense logits per position; OPD
/// only needs the next-token row.
fn greedy_next_token(
    logits_id: TensorId,
    seq_len: usize,
    vocab: usize,
    store: &mut TensorStore,
    sampling: Option<&SamplingParams>,
    position: u64,
) -> Result<u32> {
    let host = store.to_host(logits_id)?;
    if seq_len == 0 || vocab == 0 {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout cannot sample next token with seq_len={seq_len}, vocab={vocab}. \
             Hint: pass a non-empty prompt and a Qwen35Config with vocab_size > 0."
        )));
    }
    let expected_len = seq_len.checked_mul(vocab).ok_or_else(|| {
        OpdError::InvalidInput(format!(
            "OPD rollout logits shape overflow for seq_len={seq_len}, vocab={vocab}. \
             Hint: check the prompt length and Qwen35Config::vocab_size before calling opd_step."
        ))
    })?;
    if host.len() != expected_len {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout logits length mismatch: logits_len={}, expected exactly \
             seq_len * vocab = {expected_len} ({seq_len} * {vocab}). Hint: check \
             Qwen35Model::forward output shape and Qwen35Config::vocab_size.",
            host.len()
        )));
    }
    let last_row_start = (seq_len - 1) * vocab;
    let row = &host[last_row_start..last_row_start + vocab];
    if let Some(params) = sampling {
        for (i, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(OpdError::InvalidInput(format!(
                    "OPD rollout logits contain non-finite value at last-row vocab index {i}: {v}. \
                     Hint: check student forward numerics, checkpoint dtype conversion, and \
                     learning-rate stability before sampling the next token."
                )));
            }
        }
        return Ok(sample_token(row, params, position));
    }

    let mut best_idx: usize = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() {
            return Err(OpdError::InvalidInput(format!(
                "OPD rollout logits contain non-finite value at last-row vocab index {i}: {v}. \
                 Hint: check student forward numerics, checkpoint dtype conversion, and \
                 learning-rate stability before sampling the next token."
            )));
        }
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

fn device_argmax_token(
    logits_id: TensorId,
    vocab: usize,
    store: &mut TensorStore,
    sampling: Option<&SamplingParams>,
    position: u64,
) -> Result<TensorId> {
    if vocab == 0 {
        return Err(OpdError::InvalidInput(
            "OPD rollout cannot sample next token with vocab=0. Hint: verify \
             Qwen35Config::vocab_size before calling opd_step."
                .to_owned(),
        ));
    }
    let shape = store
        .get(logits_id)
        .ok_or(AutogradError::InvalidTensorId(logits_id))?
        .shape
        .clone();
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim != vocab {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout logits last dim mismatch: got {last_dim}, expected \
             vocab={vocab}. Hint: check Qwen35Model::forward output shape and \
             Qwen35Config::vocab_size."
        )));
    }
    let total = shape.iter().product::<usize>();
    let rows = total / vocab;
    if rows != 1 {
        return Err(OpdError::InvalidInput(format!(
            "OPD device rollout expects exactly one logits row, got {rows}. \
             Hint: rollout KV cache should return only the final next-token \
            logits row."
        )));
    }
    if let Some(params) = sampling {
        let host = store.to_host(logits_id)?;
        if host.len() != vocab {
            return Err(OpdError::InvalidInput(format!(
                "OPD sampled device rollout logits length mismatch: got {}, expected vocab={vocab}. \
                 Hint: rollout KV cache should return exactly one final next-token row.",
                host.len()
            )));
        }
        for (i, &v) in host.iter().enumerate() {
            if !v.is_finite() {
                return Err(OpdError::InvalidInput(format!(
                    "OPD sampled device rollout logits contain non-finite value at vocab index {i}: {v}. \
                     Hint: check student forward numerics before sampling the next token."
                )));
            }
        }
        let sampled = sample_token(&host, params, position);
        let token_id = store.from_slice(&[sampled as f32], &[1])?;
        store.ensure_device(token_id)?;
        return Ok(token_id);
    }

    store.ensure_device(logits_id)?;
    let logits_handle = store
        .get(logits_id)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "device_argmax_token: logits missing device handle",
        ))?;
    let token_handle = store.backend().argmax_last_dim(&logits_handle, &shape)?;
    Ok(store.alloc_device_tensor(vec![rows], token_handle)?)
}

fn write_rollout_token(
    buffer_id: TensorId,
    token_id: TensorId,
    rollout_len: usize,
    step: usize,
    store: &mut TensorStore,
) -> Result<TensorId> {
    store.ensure_device(buffer_id)?;
    store.ensure_device(token_id)?;
    let buffer_handle = store
        .get(buffer_id)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "write_rollout_token: rollout buffer missing device handle",
        ))?;
    let token_handle = store
        .get(token_id)
        .and_then(|tensor| tensor.device_handle.clone())
        .ok_or(AutogradError::TapeInvariant(
            "write_rollout_token: token missing device handle",
        ))?;
    let next_handle =
        store
            .backend()
            .write_scalar_at(&buffer_handle, &token_handle, rollout_len, step)?;
    Ok(store.alloc_device_tensor(vec![rollout_len], next_handle)?)
}

fn read_generated_rollout_tokens(
    buffer_id: TensorId,
    rollout_len: usize,
    vocab: usize,
    store: &mut TensorStore,
) -> Result<Vec<u32>> {
    let host = store.to_host(buffer_id)?;
    if host.len() != rollout_len {
        return Err(OpdError::InvalidInput(format!(
            "OPD generated rollout token buffer length mismatch: got {}, \
             expected {rollout_len}. Hint: device argmax rollout buffer shape \
             should match cfg.rollout_len.",
            host.len()
        )));
    }
    let out: Vec<u32> = host
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if !value.is_finite() {
                return Err(OpdError::InvalidInput(format!(
                    "OPD generated rollout token at index {index} is non-finite ({value}). \
                     Hint: check CUDA argmax output and student forward numerics."
                )));
            }
            let rounded = value.round();
            if (value - rounded).abs() > 0.0 {
                return Err(OpdError::InvalidInput(format!(
                    "OPD generated rollout token at index {index} is not an exact \
                     integer id ({value}). Hint: CUDA argmax should write exact \
                     f32 token ids."
                )));
            }
            if rounded < 0.0 || rounded as usize >= vocab {
                return Err(OpdError::InvalidInput(format!(
                    "OPD generated rollout token id {rounded} at index {index} is \
                     outside student.config().vocab_size={vocab}. Hint: check the \
                     argmax kernel bounds and model vocab size."
                )));
            }
            Ok(rounded as u32)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(out)
}

fn use_device_rollout_argmax(store: &TensorStore, rollout_len: usize, vocab: usize) -> bool {
    matches!(store.backend().device(), Device::Cuda) && (rollout_len >= 4 || vocab >= 65_536)
}

static ROLLOUT_PROGRESS_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("ARLE_OPD_ROLLOUT_PROGRESS").is_some());

fn should_retain_rollout_step(step: usize, rollout_len: usize) -> bool {
    let completed_steps = step + 1;
    completed_steps.is_multiple_of(crate::runtime_flags::rollout_retain_interval())
        || completed_steps == rollout_len
}

fn maybe_log_rollout_progress(path: &str, step: usize, rollout_len: usize, started: &Instant) {
    if !*ROLLOUT_PROGRESS_ENABLED {
        return;
    }
    let completed_steps = step + 1;
    if completed_steps.is_multiple_of(crate::runtime_flags::rollout_progress_interval())
        || completed_steps == rollout_len
    {
        eprintln!(
            "opd_rollout_progress path={path} step={completed_steps}/{rollout_len} \
             elapsed_seconds={:.3} retain_interval={}",
            started.elapsed().as_secs_f64(),
            crate::runtime_flags::rollout_retain_interval()
        );
    }
}

fn retain_rollout_step_tensors(
    store: &mut TensorStore,
    base_keep: &HashSet<TensorId>,
    rollout_cache: &Qwen35KvCache,
    current_device_token: Option<TensorId>,
    generated_tokens: Option<TensorId>,
) {
    let mut keep = base_keep.clone();
    rollout_cache.extend_tensor_ids(&mut keep);
    if let Some(token_id) = current_device_token {
        keep.insert(token_id);
    }
    if let Some(buffer_id) = generated_tokens {
        keep.insert(buffer_id);
    }
    store.retain_ids(&keep);
}

fn rollout_full_forward(
    student: &Qwen35Model,
    rollout: &mut Vec<u32>,
    rollout_len: usize,
    vocab: usize,
    sampling: Option<&SamplingParams>,
    store: &mut TensorStore,
    tape: &mut Tape,
    base_keep: &HashSet<TensorId>,
) -> Result<()> {
    let rollout_started = Instant::now();
    for step in 0..rollout_len {
        let positions = (0..rollout.len() as u32).collect::<Vec<_>>();
        let logits = student
            .forward(store, tape, rollout, &positions)
            .map_err(|err| map_qwen35_forward_error("student rollout", err))?;
        let next = greedy_next_token(
            logits,
            rollout.len(),
            vocab,
            store,
            sampling,
            rollout.len() as u64,
        )?;
        rollout.push(next);
        if should_retain_rollout_step(step, rollout_len) {
            store.retain_ids(base_keep);
        }
        maybe_log_rollout_progress("full-forward", step, rollout_len, &rollout_started);
    }
    store.retain_ids(base_keep);
    Ok(())
}

fn validate_token_ids(context: &str, tokens: &[u32], vocab: usize) -> Result<()> {
    if let Some((index, token_id)) = tokens
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, token_id)| token_id as usize >= vocab)
    {
        return Err(OpdError::InvalidInput(format!(
            "{context} token id {token_id} at {context}[{index}] is outside \
             student.config().vocab_size={vocab}. Hint: verify the tokenizer and \
             student model directory match before running OPD."
        )));
    }
    Ok(())
}

fn validate_forced_rollout(
    forced_rollout: &[u32],
    prompt_ids: &[u32],
    rollout_len: usize,
    vocab: usize,
) -> Result<()> {
    let expected_len = prompt_ids
        .len()
        .checked_add(rollout_len)
        .ok_or_else(|| OpdError::InvalidInput("OPD forced_rollout length overflow".to_owned()))?;
    if forced_rollout.len() != expected_len {
        return Err(OpdError::InvalidInput(format!(
            "OPD forced_rollout length mismatch: got {}, expected {} \
             (= prompt_len {} + rollout_len {}). Hint: pass the full selected \
             prompt-plus-rollout trajectory.",
            forced_rollout.len(),
            expected_len,
            prompt_ids.len(),
            rollout_len
        )));
    }
    if !forced_rollout.starts_with(prompt_ids) {
        return Err(OpdError::InvalidInput(
            "OPD forced_rollout must start with the exact prompt_ids prefix. \
             Hint: select whole candidate trajectories; do not splice response \
             suffixes into a different prompt."
                .to_owned(),
        ));
    }
    validate_token_ids("forced_rollout", forced_rollout, vocab)
}

pub fn student_rollout_only(
    student: &Qwen35Model,
    prompt_ids: &[u32],
    rollout_len: usize,
    sampling: Option<&SamplingParams>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<Vec<u32>> {
    let vocab = student.config().vocab_size;
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "student_rollout_only requires a non-empty prompt_ids slice. Hint: \
             pass at least one BOS/chat token; the rollout helper does not \
             synthesize prompts."
                .to_owned(),
        ));
    }
    if vocab == 0 {
        return Err(OpdError::InvalidInput(
            "student_rollout_only requires student.config().vocab_size > 0.".to_owned(),
        ));
    }
    validate_rollout_shape(prompt_ids.len(), rollout_len, vocab)?;
    validate_token_ids("prompt_ids", prompt_ids, vocab)?;
    let rollout_keep_base = live_tensor_ids(store);
    student_rollout_only_with_keep(
        student,
        prompt_ids,
        rollout_len,
        sampling,
        &rollout_keep_base,
        store,
        tape,
    )
}

fn student_rollout_only_with_keep(
    student: &Qwen35Model,
    prompt_ids: &[u32],
    rollout_len: usize,
    sampling: Option<&SamplingParams>,
    rollout_keep_base: &HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<Vec<u32>> {
    tape.entries.clear();
    tape.set_enabled(false);
    let mut rollout: Vec<u32> = prompt_ids.to_vec();
    let vocab = student.config().vocab_size;
    let use_rollout_kv_cache = student.supports_rollout_kv_cache();

    if use_rollout_kv_cache && use_device_rollout_argmax(store, rollout_len, vocab) {
        let mut rollout_cache = Qwen35KvCache::new(student, prompt_ids.len() + rollout_len);
        let mut generated_tokens = if rollout_len == 0 {
            None
        } else {
            let handle = store.backend().zeros(&[rollout_len])?;
            Some(store.alloc_device_tensor(vec![rollout_len], handle)?)
        };
        let mut current_device_token: Option<TensorId> = None;
        let rollout_started = Instant::now();
        for step in 0..rollout_len {
            let logits = if step == 0 {
                let positions = (0..prompt_ids.len() as u32).collect::<Vec<_>>();
                forward_rollout_cached(
                    student,
                    store,
                    tape,
                    prompt_ids,
                    &positions,
                    &mut rollout_cache,
                )
                .map_err(|err| map_qwen35_forward_error("student rollout", err))?
            } else {
                let token_id = current_device_token.ok_or_else(|| {
                    OpdError::InvalidInput(
                        "OPD rollout cache cannot decode from an empty rollout. Hint: pass a \
                         non-empty prompt before calling student_rollout_only."
                            .to_owned(),
                    )
                })?;
                let position = (prompt_ids.len() + step - 1) as u32;
                forward_rollout_cached_device_token(
                    student,
                    store,
                    tape,
                    token_id,
                    position,
                    &mut rollout_cache,
                )
                .map_err(|err| map_qwen35_forward_error("student rollout", err))?
            };
            let next_token = device_argmax_token(
                logits,
                vocab,
                store,
                sampling,
                (prompt_ids.len() + step) as u64,
            )?;
            if let Some(buffer_id) = generated_tokens {
                generated_tokens = Some(write_rollout_token(
                    buffer_id,
                    next_token,
                    rollout_len,
                    step,
                    store,
                )?);
            }
            current_device_token = Some(next_token);
            if should_retain_rollout_step(step, rollout_len) {
                retain_rollout_step_tensors(
                    store,
                    rollout_keep_base,
                    &rollout_cache,
                    current_device_token,
                    generated_tokens,
                );
            }
            maybe_log_rollout_progress("device-token", step, rollout_len, &rollout_started);
        }
        if let Some(buffer_id) = generated_tokens {
            rollout.extend(read_generated_rollout_tokens(
                buffer_id,
                rollout_len,
                vocab,
                store,
            )?);
        }
        store.retain_ids(rollout_keep_base);
    } else if use_rollout_kv_cache {
        let mut rollout_cache = Qwen35KvCache::new(student, prompt_ids.len() + rollout_len);
        let rollout_started = Instant::now();
        for step in 0..rollout_len {
            let (input_ids, positions, logits_seq_len) = if step == 0 {
                (
                    rollout.clone(),
                    (0..rollout.len() as u32).collect::<Vec<_>>(),
                    1,
                )
            } else {
                let last = *rollout.last().ok_or_else(|| {
                    OpdError::InvalidInput(
                        "OPD rollout cache cannot decode from an empty rollout. Hint: pass a \
                         non-empty prompt before calling student_rollout_only."
                            .to_owned(),
                    )
                })?;
                let position = (rollout.len() - 1) as u32;
                (vec![last], vec![position], 1)
            };
            let logits = forward_rollout_cached(
                student,
                store,
                tape,
                &input_ids,
                &positions,
                &mut rollout_cache,
            )
            .map_err(|err| map_qwen35_forward_error("student rollout", err))?;
            let next = greedy_next_token(
                logits,
                logits_seq_len,
                vocab,
                store,
                sampling,
                rollout.len() as u64,
            )?;
            rollout.push(next);
            if should_retain_rollout_step(step, rollout_len) {
                retain_rollout_step_tensors(store, rollout_keep_base, &rollout_cache, None, None);
            }
            maybe_log_rollout_progress("host-token", step, rollout_len, &rollout_started);
        }
        store.retain_ids(rollout_keep_base);
    } else {
        rollout_full_forward(
            student,
            &mut rollout,
            rollout_len,
            vocab,
            sampling,
            store,
            tape,
            rollout_keep_base,
        )?;
    }

    debug_assert!(
        tape.entries.is_empty(),
        "student_rollout_only must keep rollout candidates off the backward tape"
    );
    Ok(rollout)
}

fn validate_student_params(student_params: &[TensorId], store: &TensorStore) -> Result<()> {
    if student_params.is_empty() {
        return Err(OpdError::InvalidInput(
            "OPD step requires at least one student parameter id. Hint: pass \
             student.all_parameter_ids() from the trainable student model; an empty \
             parameter list makes the optimizer step a no-op."
                .to_owned(),
        ));
    }

    let mut trainable = 0usize;
    let mut seen = std::collections::HashSet::new();
    for (index, &param_id) in student_params.iter().enumerate() {
        if !seen.insert(param_id) {
            return Err(OpdError::InvalidInput(format!(
                "OPD student_params[{index}]={param_id} duplicates an earlier \
                 parameter id. Hint: pass each trainable student parameter \
                 exactly once; duplicate ids would apply grad clipping and \
                 optimizer updates more than once."
            )));
        }
        let tensor = store.get(param_id).ok_or_else(|| {
            OpdError::InvalidInput(format!(
                "OPD student_params[{index}]={param_id} does not exist in the TensorStore. \
                 Hint: pass parameter ids from the same student Qwen35Model and TensorStore \
                 used for this opd_step call."
            ))
        })?;
        if tensor.requires_grad {
            trainable += 1;
        }
    }

    if trainable == 0 {
        return Err(OpdError::InvalidInput(
            "OPD student_params contains no trainable tensors (requires_grad=true). \
             Hint: build the student with Qwen35Model::new for scratch training or \
             Qwen35Model::new_with_lora for LoRA; frozen teacher/eval parameter ids \
             make the OPD optimizer step a no-op."
                .to_owned(),
        ));
    }

    Ok(())
}

fn validate_student_param_ownership(
    student_params: &[TensorId],
    student_model_params: &[TensorId],
    teacher_params: &[TensorId],
) -> Result<()> {
    let student_model_param_set: HashSet<TensorId> = student_model_params.iter().copied().collect();
    let teacher_param_set: HashSet<TensorId> = teacher_params.iter().copied().collect();
    for (index, &param_id) in student_params.iter().enumerate() {
        if teacher_param_set.contains(&param_id) {
            return Err(OpdError::InvalidInput(format!(
                "OPD student_params[{index}]={param_id} belongs to the frozen \
                 teacher model. Hint: pass student parameter ids from \
                 student.all_parameter_ids() or the student's LoRA adapter ids; \
                 teacher weights must not be optimized."
            )));
        }
        if !student_model_param_set.contains(&param_id) {
            return Err(OpdError::InvalidInput(format!(
                "OPD student_params[{index}]={param_id} is not owned by the \
                 student Qwen35Model passed to opd_step. Hint: build \
                 student_params from that exact student's all_parameter_ids() \
                 or adapter ids, using the same TensorStore."
            )));
        }
    }

    Ok(())
}

fn validate_teacher_params(teacher_params: &[TensorId], store: &TensorStore) -> Result<()> {
    if teacher_params.is_empty() {
        return Err(OpdError::InvalidInput(
            "OPD teacher exposes no parameter ids. Hint: pass a Qwen35Model \
             built by Qwen35Model::new_for_eval or load_qwen35_from_hf_dir."
                .to_owned(),
        ));
    }

    for (index, &param_id) in teacher_params.iter().enumerate() {
        let tensor = store.get(param_id).ok_or_else(|| {
            OpdError::InvalidInput(format!(
                "OPD teacher parameter ids must belong to the same TensorStore, \
                 but teacher_params[{index}]={param_id} is missing. Hint: build \
                 teacher and student in the TensorStore passed to opd_step."
            ))
        })?;
        if tensor.requires_grad {
            return Err(OpdError::InvalidInput(format!(
                "OPD teacher parameter teacher_params[{index}]={param_id} has \
                 requires_grad=true. Hint: build the teacher with \
                 Qwen35Model::new_for_eval, load_qwen35_from_hf_dir, or \
                 student.clone_frozen; OPD must not optimize teacher weights."
            )));
        }
    }

    Ok(())
}

fn validate_loss_value(loss_value: f32) -> Result<()> {
    if loss_value.is_finite() {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD KL loss became non-finite ({loss_value}). Hint: check teacher/student logits \
         for NaN or Inf, verify both checkpoints use the same tokenizer/model family, and \
         reduce the learning rate before resuming."
    )))
}

fn validate_logits_shape(stage: &str, shape: &[usize], seq_len: usize, vocab: usize) -> Result<()> {
    let expected_shape = vec![1, seq_len, vocab];
    if shape == expected_shape {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD {stage} logits shape mismatch: got {shape:?}, expected \
         {expected_shape:?}. Hint: windowed Route B requires each teacher and \
         student forward to return [batch=1, window_len, vocab] for exactly \
         the current logits window."
    )))
}

fn validate_step_config(cfg: &OpdStepConfig) -> Result<()> {
    if cfg.grad_clip >= 0.0 && cfg.grad_clip.is_finite() {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD step requires cfg.grad_clip to be non-negative and finite, got {}. Hint: pass \
         a positive finite threshold to enable clipping, or pass 0.0 to \
         disable clipping explicitly.",
        cfg.grad_clip
    )))
}

fn validate_gkd_lambda(gkd_lambda: f32) -> Result<()> {
    if (0.0..=1.0).contains(&gkd_lambda) && gkd_lambda.is_finite() {
        return Ok(());
    }
    Err(OpdError::InvalidInput(format!(
        "OPD GKD lambda must be finite and in [0.0, 1.0], got {gkd_lambda}. \
         Hint: pass --gkd-lambda 0.0 for pure OPD, 0.3 for the literature \
         SFT/OPD blend probe, or 1.0 for pure hard-token SFT proxy."
    )))
}

fn validate_gkd_loss_config(config: GkdLossConfig<'_>) -> Result<()> {
    validate_gkd_lambda(config.lambda)?;
    if !config.kl_temperature.is_finite() || config.kl_temperature <= 0.0 {
        return Err(OpdError::InvalidInput(format!(
            "OPD KL temperature must be finite and > 0.0, got {}. Hint: use \
             --kl-temperature 1.0 for the baseline KL loss or a positive value \
             for pure-OPD temperature softening.",
            config.kl_temperature
        )));
    }
    if config.kl_temperature != 1.0 && config.lambda > 0.0 {
        return Err(OpdError::InvalidInput(format!(
            "OPD KL temperature is a pure-OPD lever only: got \
             kl_temperature={} with gkd_lambda={}. The SFT anchor is deliberately \
             1/vocab-scale-matched to the KL term, so T^2 KL compensation would \
             silently reweight the (1-lambda)KL + lambda*SFT blend. Set \
             --gkd-lambda 0.0 or --kl-temperature 1.0.",
            config.kl_temperature, config.lambda
        )));
    }
    if let Some(beta) = config.kl_beta
        && !(beta.is_finite() && (0.0..=1.0).contains(&beta))
    {
        return Err(OpdError::InvalidInput(format!(
            "OPD KL beta must be finite and in [0.0, 1.0], got {beta}. \
             Hint: omit --kl-beta to keep --kl-direction, or pass 0.5 for \
             TRL-style generalized JSD."
        )));
    }
    if config.kl_chunk_size == Some(0) {
        return Err(OpdError::InvalidInput(
            "OPD KL chunk size must be > 0 when set. Hint: pass \
             --kl-chunk-size 32 for the 256-token rollout bench, or set an \
             explicit larger value after a memory check."
                .to_owned(),
        ));
    }
    if let Some(teacher_topk) = config.teacher_topk {
        if teacher_topk == 0 {
            return Err(OpdError::InvalidInput(
                "OPD teacher top-k must be > 0 when set. Hint: omit \
                 --teacher-topk to keep the dense full-vocab teacher path, or \
                 pass a positive k after Piece A engine-side top-k is available."
                    .to_owned(),
            ));
        }
        if config.kl_direction != KlDirection::Forward {
            return Err(OpdError::InvalidInput(
                "OPD teacher top-k is forward-KL only. Hint: omit \
                 --teacher-topk for reverse KL, or use --kl-direction forward."
                    .to_owned(),
            ));
        }
        if config.kl_beta.is_some() {
            return Err(OpdError::InvalidInput(
                "OPD teacher top-k does not support generalized JSD beta. Hint: \
                 omit --kl-beta for sparse forward-KL teacher targets."
                    .to_owned(),
            ));
        }
    }
    if config.logits_window_size == Some(0) {
        return Err(OpdError::InvalidInput(
            "OPD logits window size must be > 0 when set. Hint: pass \
             --logits-window-size 64 with --kl-chunk-size 64 for the \
             512-token real-corpus Route B smoke, or omit it to keep the \
             baseline full-logits path."
                .to_owned(),
        ));
    }
    if config.lambda == 0.0 {
        return Ok(());
    }
    if config.sft_anchor == GkdSftAnchor::CorpusTruth
        && config.corpus_tokens.is_none_or(|tokens| tokens.is_empty())
    {
        return Err(OpdError::InvalidInput(
            "GKD corpus-truth SFT anchor requires non-empty corpus completion \
             tokens when lambda > 0. Hint: add a `completion` or `target` \
             field to each training row in --prompts-file, or use \
             --sft-anchor student-rollout."
                .to_owned(),
        ));
    }
    Ok(())
}

fn kl_distill_loss_for_config(
    student_logits: TensorId,
    teacher_logits: TensorId,
    num_positions: usize,
    kl_chunk_size: Option<usize>,
    kl_direction: KlDirection,
    kl_temperature: f32,
    kl_beta: Option<f32>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if let Some(beta) = kl_beta {
        return match kl_chunk_size {
            Some(chunk_size) => generalized_beta_jsd_loss_chunked(
                student_logits,
                teacher_logits,
                num_positions,
                chunk_size,
                kl_temperature,
                beta,
                store,
                tape,
            ),
            None => generalized_beta_jsd_loss(
                student_logits,
                teacher_logits,
                num_positions,
                kl_temperature,
                beta,
                store,
                tape,
            ),
        }
        .map_err(OpdError::from);
    }
    match kl_chunk_size {
        Some(chunk_size) => kl_distill_loss_chunked(
            student_logits,
            teacher_logits,
            num_positions,
            chunk_size,
            kl_temperature,
            kl_direction,
            store,
            tape,
        )
        .map_err(OpdError::from),
        None => kl_distill_loss(
            student_logits,
            teacher_logits,
            num_positions,
            kl_temperature,
            kl_direction,
            store,
            tape,
        )
        .map_err(OpdError::from),
    }
}

fn sequence_windows(total_positions: usize, window_size: usize) -> Result<Vec<SequenceWindow>> {
    if total_positions == 0 {
        return Err(OpdError::InvalidInput(
            "OPD windowed logits path requires at least one position. Hint: \
             pass a non-empty prompt/completion sequence before enabling \
             --logits-window-size."
                .to_owned(),
        ));
    }
    if window_size == 0 {
        return Err(OpdError::InvalidInput(
            "OPD logits window size must be > 0 when set. Hint: pass \
             --logits-window-size 64 or omit the flag for full logits."
                .to_owned(),
        ));
    }
    let mut windows = Vec::new();
    let mut start = 0usize;
    while start < total_positions {
        let end = start.saturating_add(window_size).min(total_positions);
        windows.push(SequenceWindow { start, end });
        start = end;
    }
    Ok(windows)
}

fn sequence_windows_for_range(
    range: KlLogitRange,
    window_size: usize,
) -> Result<Vec<SequenceWindow>> {
    sequence_windows(range.len(), window_size).map(|windows| {
        windows
            .into_iter()
            .map(|window| SequenceWindow {
                start: range.start + window.start,
                end: range.start + window.end,
            })
            .collect()
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KlLogitRange {
    start: usize,
    end: usize,
}

impl KlLogitRange {
    fn len(self) -> usize {
        self.end - self.start
    }
}

fn kl_logit_range(mask: OpdKlMask, prompt_len: usize, sequence_len: usize) -> Result<KlLogitRange> {
    match mask {
        OpdKlMask::Full => {
            if sequence_len == 0 {
                return Err(OpdError::InvalidInput(
                    "OPD full KL mask requires at least one sequence position.".to_owned(),
                ));
            }
            Ok(KlLogitRange {
                start: 0,
                end: sequence_len,
            })
        }
        OpdKlMask::CompletionOnly => {
            if prompt_len == 0 {
                return Err(OpdError::InvalidInput(
                    "OPD completion-only KL mask requires a non-empty prompt.".to_owned(),
                ));
            }
            if sequence_len <= prompt_len {
                return Err(OpdError::InvalidInput(format!(
                    "OPD completion-only KL mask requires at least one completion token, \
                     got prompt_len={prompt_len} and sequence_len={sequence_len}. \
                     Hint: set --rollout-len > 0 or use --opd-kl-mask full."
                )));
            }
            Ok(KlLogitRange {
                start: prompt_len - 1,
                end: sequence_len - 1,
            })
        }
    }
}

fn slice_logits_for_kl(
    logits: TensorId,
    range: KlLogitRange,
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let starts = [0, range.start, 0];
    let ends = [1, range.end, vocab];
    slice(logits, &starts, &ends, store, tape).map_err(OpdError::from)
}

#[allow(clippy::too_many_arguments)]
fn backward_chunked_kl_rollout<T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    rollout: &[u32],
    prompt_len: usize,
    vocab: usize,
    chunk_size: usize,
    kl_mask: OpdKlMask,
    kl_direction: KlDirection,
    kl_temperature: f32,
    kl_beta: Option<f32>,
    student_model_params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
    engine_offload: EngineOffloadMode,
    student_engine_offload: Option<&dyn Fn() -> Result<usize>>,
    student_engine_reload: Option<&dyn Fn() -> Result<()>>,
) -> Result<f32> {
    let kl_range = kl_logit_range(kl_mask, prompt_len, rollout.len())?;
    if kl_range.start >= kl_range.end {
        return Ok(0.0);
    }

    // Forward teacher + student over the scored prefix exactly ONCE. Causal
    // attention makes position p's logits independent of tokens after p, so a
    // single full-prefix forward yields the same per-position logits the old
    // per-chunk loop produced by re-forwarding `[0..seq_end_k]` from token 0
    // for every chunk — but without the O(n^2) redundant recompute that
    // dominated backward (each chunk re-ran the full growing prefix through
    // the dense base model). Chunking now happens only at the loss/softmax
    // level via `kl_distill_loss_chunked`, matching the already-correct
    // `kl_distill_loss_for_config` path used by the non-rollout KL callers.
    let seq_end = kl_range.end;
    let positions = (0..seq_end as u32).collect::<Vec<_>>();
    let prefix = &rollout[..seq_end];

    tape.entries.clear();
    tape.set_enabled(false);
    // OPD engine time-share: the teacher may have been offloaded to host RAM
    // during the previous step's student backward. Reload it before scoring.
    // In `Student` mode the teacher stays resident, so this is skipped.
    //
    // Fence the train backend first. The reload re-allocates the teacher's
    // weight buffers (incl. the W4A8 Marlin packed side buffers) from the
    // shared async pool on the infer scheduler thread; without this barrier
    // the previous step's still-draining train pool ops race those allocs
    // (event tracking is disabled) and the reloaded Marlin side buffer is
    // dropped → "missing W4A8 Marlin-packed side buffer".
    if engine_offload.offloads_teacher() {
        store
            .backend()
            .device_synchronize()
            .map_err(OpdError::from)?;
        teacher
            .reload_engine_weights()
            .map_err(|err| map_teacher_forward_error("teacher reload", err))?;
    }
    let phase_started = Instant::now();
    let teacher_logits = teacher
        .forward_logits_device(prefix, &positions, store, tape)
        .map_err(|err| map_teacher_forward_error("teacher scoring", err))?;
    record_profile(profile, |profile| {
        profile.teacher_forward_seconds += phase_started.elapsed().as_secs_f64();
    });
    let expected_shape = vec![1, seq_end, vocab];
    if teacher_logits.shape != expected_shape {
        return Err(OpdError::InvalidInput(format!(
            "OPD chunked teacher logits shape mismatch: got {:?}, expected {:?}. \
             Hint: the TeacherForward implementation must return \
             [batch=1, seq_len, vocab] logits for the exact prefix being scored.",
            teacher_logits.shape, expected_shape
        )));
    }

    // KL over the completion region only. The teacher is a fixed distillation
    // target — no gradient ever flows into it — so slice its completion region
    // with the tape DISABLED. This is load-bearing for VRAM: a tape-enabled
    // slice registers a `slice_bwd` node that pre-allocates a grad buffer the
    // size of the *full* teacher logits (`[1, seq_end, vocab]` ≈ 142 MiB at
    // 144×248320), which was the rollout-128 backward OOM site. With the tape
    // off, only the small `[1, kl_range, vocab]` slice is materialized. The
    // teacher runs through the infer runtime (full logits — the windowed
    // `forward_logits_window_device` is only supported by the in-process
    // Qwen35 teacher), so we then free the full-prefix teacher logits: with the
    // tape disabled it has no backward dependency, reclaiming ~142 MiB to widen
    // headroom for the student backward.
    let phase_started = Instant::now();
    let starts = [0, kl_range.start, 0];
    let ends = [1, kl_range.end, vocab];
    tape.set_enabled(false);
    let teacher_kl =
        slice(teacher_logits.tensor_id, &starts, &ends, store, tape).map_err(OpdError::from)?;
    store
        .free(teacher_logits.tensor_id)
        .map_err(OpdError::from)?;

    // OPD engine time-share: the teacher's KL target is now materialized in the
    // train store and the rollout is complete, so BOTH idle infer engines
    // (rollout student + scoring teacher) can be offloaded to host RAM to free
    // VRAM for the student backward. They are offloaded together here — on a
    // device quiesced after teacher scoring — rather than the student earlier,
    // to avoid racing the teacher forward's allocations against the student's
    // async pool frees. Reloaded at the next step (student before rollout,
    // teacher before scoring).
    //
    // Fence the train backend before the infer-thread offload frees the
    // pool blocks, same cross-context ordering reason as the reload fence.
    if engine_offload.is_enabled() {
        store
            .backend()
            .device_synchronize()
            .map_err(OpdError::from)?;
    }
    if engine_offload.offloads_student()
        && let Some(offload_student) = student_engine_offload
    {
        let freed = offload_student()?;
        eprintln!(
            "opd_engine_offload student_offloaded freed_bytes={freed} freed_mib={:.1}",
            freed as f64 / (1024.0 * 1024.0)
        );
    }
    if engine_offload.offloads_teacher() {
        let freed = teacher
            .offload_engine_weights()
            .map_err(|err| map_teacher_forward_error("teacher offload", err))?;
        eprintln!(
            "opd_engine_offload teacher_offloaded freed_bytes={freed} freed_mib={:.1}",
            freed as f64 / (1024.0 * 1024.0)
        );
    }

    tape.set_enabled(true);
    let student_phase_started = Instant::now();
    // VRAM-fit lever (2026-05-29): run the student lm_head over the KL window
    // ONLY, not the full scored prefix. The hidden-state forward still covers
    // `[0..seq_end]` (causal attention needs the full prefix), but the
    // vocab-wide lm_head projection — and its backward grad buffers — are the
    // dominant transient at seq×vocab=144×248320. `forward_logits_window`
    // slices the cheap 1024-wide hidden to the window then projects, so the
    // full `[1, seq_end, vocab]` student logits tensor (≈142 MiB) and the
    // `slice_bwd` grad buffer of the same size never materialize. Numerically
    // identical: causal logits at position p are independent of tokens after p,
    // so windowed lm_head == full lm_head sliced.
    let kl_window = SequenceWindow {
        start: kl_range.start,
        end: kl_range.end,
    };
    let student_kl = student
        .forward_logits_window(store, tape, prefix, &positions, kl_window)
        .map_err(|err| map_qwen35_forward_error("student chunk KL", err))?;
    record_profile(profile, |profile| {
        profile.student_forward_seconds += student_phase_started.elapsed().as_secs_f64();
    });

    let loss = if let Some(beta) = kl_beta {
        generalized_beta_jsd_loss_chunked(
            student_kl,
            teacher_kl,
            kl_range.len(),
            chunk_size,
            kl_temperature,
            beta,
            store,
            tape,
        )
    } else {
        kl_distill_loss_chunked(
            student_kl,
            teacher_kl,
            kl_range.len(),
            chunk_size,
            kl_temperature,
            kl_direction,
            store,
            tape,
        )
    }
    .map_err(OpdError::from)?;
    let loss_value = store.to_host(loss).map_err(OpdError::from)?[0];
    record_profile(profile, |profile| {
        profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
    });

    let phase_started = Instant::now();
    backward_with_optional_profile(loss, loss_value, store, tape)?;
    record_profile(profile, |profile| {
        profile.backward_seconds += phase_started.elapsed().as_secs_f64();
    });
    cleanup_after_backward(store, tape, student_model_params, keep_extra);

    // OPD engine time-share: the heavy student-backward transients are now
    // freed, so reload the engines we offloaded BEFORE returning. This keeps
    // the offload window strictly inside the backward and leaves both engines
    // resident for whatever the caller does next — the inter-step KL eval and
    // checkpoint save both run a teacher/student forward, which would hit an
    // offloaded (placeholder) weight and fail (W4A8 "missing Marlin-packed
    // side buffer"). Reload is idempotent, so the next step's pre-rollout /
    // pre-scoring reloads become no-ops. Fence the train backend first so the
    // backward's pool ops are ordered ahead of the reload's allocations.
    if engine_offload.is_enabled() {
        store
            .backend()
            .device_synchronize()
            .map_err(OpdError::from)?;
    }
    if engine_offload.offloads_teacher() {
        teacher
            .reload_engine_weights()
            .map_err(|err| map_teacher_forward_error("teacher reload (post-backward)", err))?;
    }
    if engine_offload.offloads_student()
        && let Some(reload_student) = student_engine_reload
    {
        reload_student()?;
    }

    Ok(loss_value)
}

fn validate_rollout_shape(prompt_len: usize, rollout_len: usize, vocab: usize) -> Result<()> {
    let total_len = prompt_len.checked_add(rollout_len).ok_or_else(|| {
        OpdError::InvalidInput(format!(
            "OPD rollout length overflow: prompt_len={prompt_len}, \
             cfg.rollout_len={rollout_len}. Hint: reduce --rollout-len or \
             split the prompt before calling opd_step."
        ))
    })?;
    if total_len > u32::MAX as usize {
        return Err(OpdError::InvalidInput(format!(
            "OPD rollout total length {total_len} exceeds u32::MAX position ids. \
             Hint: reduce --rollout-len or prompt length; the current OPD \
             Qwen3.5 path uses u32 position ids."
        )));
    }
    if vocab > u32::MAX as usize {
        return Err(OpdError::InvalidInput(format!(
            "OPD student.config().vocab_size={vocab} exceeds u32::MAX token ids. \
             Hint: verify Qwen35Config::vocab_size; greedy rollout returns u32 \
             token ids."
        )));
    }
    Ok(())
}

fn next_token_sft_loss_from_logits(
    student_logits: TensorId,
    logits_seq_len: usize,
    start_position: usize,
    target_tokens: &[u32],
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if target_tokens.is_empty() {
        return Err(OpdError::InvalidInput(
            "GKD SFT proxy requires at least one target token. Hint: provide \
             non-empty corpus completion tokens or use a rollout with at \
             least two tokens."
                .to_owned(),
        ));
    }
    let end_position = start_position
        .checked_add(target_tokens.len())
        .ok_or_else(|| {
            OpdError::InvalidInput(
                "GKD SFT proxy logits slice overflowed. Hint: check prompt \
                 and completion lengths before mixing SFT into OPD."
                    .to_owned(),
            )
        })?;
    if end_position > logits_seq_len {
        return Err(OpdError::InvalidInput(format!(
            "GKD SFT proxy target slice [{}..{}) exceeds logits_seq_len={}. \
             Hint: completion-token CE should use logits from the prompt's \
             final token through the completion prefix.",
            start_position, end_position, logits_seq_len
        )));
    }
    let shape = store
        .get(student_logits)
        .ok_or(AutogradError::InvalidTensorId(student_logits))?
        .shape
        .clone();
    let expected_shape = vec![1, logits_seq_len, vocab];
    if shape != expected_shape {
        return Err(OpdError::InvalidInput(format!(
            "GKD SFT proxy expected student logits shape {:?}, got {:?}. \
             Hint: pass logits from the same sequence used for the SFT \
             target tokens.",
            expected_shape, shape
        )));
    }
    let targets: Vec<usize> = target_tokens
        .iter()
        .enumerate()
        .map(|(index, &token_id)| {
            if token_id as usize >= vocab {
                return Err(OpdError::InvalidInput(format!(
                    "GKD SFT proxy target token {token_id} at target[{index}] is \
                     outside vocab={vocab}. Hint: verify tokenizer/model vocab \
                     alignment before mixing hard-token SFT into OPD."
                )));
            }
            Ok(token_id as usize)
        })
        .collect::<Result<Vec<_>>>()?;
    let shifted_logits = slice(
        student_logits,
        &[0, start_position, 0],
        &[1, end_position, vocab],
        store,
        tape,
    )?;
    // `cross_entropy_loss` is already per-position (`mean` over the one gathered
    // target logit per position), which matches `kl_distill_loss`'s `batchmean`
    // (`sum_v / positions`) scale. No `1/vocab` rescale: both the KL and the
    // hard-label CE are per-position, so `--gkd-lambda` blends them at face
    // value. (Pre-batchmean-fix this was scaled by `1/vocab` to match the old
    // mean-over-positions*vocab KL; that compensation is removed with the fix.)
    cross_entropy_loss(shifted_logits, &targets, store, tape).map_err(OpdError::from)
}

fn shifted_rollout_sft_loss(
    student_logits: TensorId,
    rollout: &[u32],
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if rollout.len() < 2 {
        return Err(OpdError::InvalidInput(
            "GKD student-rollout SFT anchor requires at least two rollout \
             tokens so logits can be trained against next-token labels. Hint: \
             use a prompt with length >= 2 or set --rollout-len > 0 when \
             --gkd-lambda > 0."
                .to_owned(),
        ));
    }
    next_token_sft_loss_from_logits(
        student_logits,
        rollout.len(),
        0,
        &rollout[1..],
        vocab,
        store,
        tape,
    )
}

fn corpus_truth_sft_loss(
    student: &Qwen35Model,
    prompt_ids: &[u32],
    corpus_tokens: &[u32],
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "GKD corpus-truth SFT anchor requires a non-empty prompt. Hint: \
             OPD prompts should include at least one context token."
                .to_owned(),
        ));
    }
    if corpus_tokens.is_empty() {
        return Err(OpdError::InvalidInput(
            "GKD corpus-truth SFT anchor requires non-empty completion tokens. \
             Hint: add a `completion` or `target` field to each training row \
             in --prompts-file."
                .to_owned(),
        ));
    }
    let total_len = prompt_ids
        .len()
        .checked_add(corpus_tokens.len())
        .ok_or_else(|| {
            OpdError::InvalidInput(
                "GKD corpus-truth SFT prompt+completion length overflowed. \
                 Hint: reduce prompt/completion max tokens."
                    .to_owned(),
            )
        })?;
    if total_len > u32::MAX as usize {
        return Err(OpdError::InvalidInput(format!(
            "GKD corpus-truth SFT sequence length {total_len} exceeds u32::MAX \
             RoPE position range. Hint: reduce prompt/completion max tokens."
        )));
    }
    for (index, &token_id) in corpus_tokens.iter().enumerate() {
        if token_id as usize >= vocab {
            return Err(OpdError::InvalidInput(format!(
                "GKD corpus-truth SFT completion token {token_id} at \
                 completion[{index}] is outside vocab={vocab}. Hint: verify \
                 tokenizer/model vocab alignment before training."
            )));
        }
    }
    let sft_sequence: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(corpus_tokens.iter().copied())
        .collect();
    let positions = (0..total_len as u32).collect::<Vec<_>>();
    let student_logits = student
        .forward(store, tape, &sft_sequence, &positions)
        .map_err(|err| map_qwen35_forward_error("student corpus SFT", err))?;
    next_token_sft_loss_from_logits(
        student_logits,
        total_len,
        prompt_ids.len() - 1,
        corpus_tokens,
        vocab,
        store,
        tape,
    )
}

fn gkd_sft_loss(
    config: GkdLossConfig<'_>,
    student: &Qwen35Model,
    prompt_ids: &[u32],
    student_logits: TensorId,
    rollout: &[u32],
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    match config.sft_anchor {
        GkdSftAnchor::StudentRollout => {
            shifted_rollout_sft_loss(student_logits, rollout, vocab, store, tape)
        }
        GkdSftAnchor::CorpusTruth => corpus_truth_sft_loss(
            student,
            prompt_ids,
            config.corpus_tokens.ok_or_else(|| {
                OpdError::InvalidInput(
                    "GKD corpus-truth SFT anchor requires corpus completion \
                     tokens. Hint: add completion/target fields to \
                     --prompts-file."
                        .to_owned(),
                )
            })?,
            vocab,
            store,
            tape,
        ),
    }
}

fn mix_gkd_losses(
    kl_loss: TensorId,
    sft_loss: TensorId,
    gkd_lambda: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    validate_gkd_lambda(gkd_lambda)?;
    if gkd_lambda == 0.0 {
        return Ok(kl_loss);
    }
    if gkd_lambda == 1.0 {
        return Ok(sft_loss);
    }
    let weighted_kl = mul_scalar(kl_loss, 1.0 - gkd_lambda, store, tape)?;
    let weighted_sft = mul_scalar(sft_loss, gkd_lambda, store, tape)?;
    add(weighted_kl, weighted_sft, store, tape).map_err(OpdError::from)
}

fn backward_weighted_window_loss(
    loss: TensorId,
    weight: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<f32> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(OpdError::InvalidInput(format!(
            "OPD window loss weight must be finite and non-negative, got {weight}. \
             Hint: verify lambda and window/target counts before Route B backward."
        )));
    }
    let loss_started = Instant::now();
    let weighted_loss = if (weight - 1.0).abs() < f32::EPSILON {
        loss
    } else {
        mul_scalar(loss, weight, store, tape)?
    };
    let loss_value = store.to_host(weighted_loss)?[0];
    validate_loss_value(loss_value)?;
    record_profile(profile, |profile| {
        profile.kl_loss_seconds += loss_started.elapsed().as_secs_f64();
    });

    let phase_started = Instant::now();
    backward_with_optional_profile(weighted_loss, loss_value, store, tape)?;
    record_profile(profile, |profile| {
        profile.backward_seconds += phase_started.elapsed().as_secs_f64();
    });
    Ok(loss_value)
}

fn cleanup_window_backward_retaining_base(
    store: &mut TensorStore,
    tape: &mut Tape,
    base_tape_len: usize,
    base_keep: &HashSet<TensorId>,
    student_model_params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
    keep_grads: &[TensorId],
) {
    tape.entries.truncate(base_tape_len);
    tape.set_enabled(true);
    let mut keep = base_keep.clone();
    keep.extend(keep_extra.iter().copied());
    for &grad_id in keep_grads {
        keep.insert(grad_id);
    }
    extend_keep_with_params_and_grads(&mut keep, student_model_params.iter().copied(), store);
    store.retain_ids(&keep);
}

fn accumulate_tensor_grad(
    store: &mut TensorStore,
    existing_grad_id: TensorId,
    incoming_grad_id: TensorId,
) -> Result<()> {
    let expected = store
        .get(existing_grad_id)
        .ok_or(AutogradError::InvalidTensorId(existing_grad_id))?
        .shape
        .clone();
    let incoming = store
        .get(incoming_grad_id)
        .ok_or(AutogradError::InvalidTensorId(incoming_grad_id))?
        .shape
        .clone();
    if expected != incoming {
        return Err(AutogradError::GradientShapeMismatch {
            tensor_id: existing_grad_id,
            expected,
            got: incoming,
        }
        .into());
    }

    let both_on_device = {
        let existing = store
            .get(existing_grad_id)
            .ok_or(AutogradError::InvalidTensorId(existing_grad_id))?;
        let incoming = store
            .get(incoming_grad_id)
            .ok_or(AutogradError::InvalidTensorId(incoming_grad_id))?;
        existing.dirty != Dirty::Host
            && existing.device_handle.is_some()
            && incoming.dirty != Dirty::Host
            && incoming.device_handle.is_some()
    };
    if both_on_device {
        let existing_handle = store
            .get(existing_grad_id)
            .and_then(|tensor| tensor.device_handle.as_ref())
            .ok_or(AutogradError::InvalidTensorId(existing_grad_id))?
            .clone();
        let incoming_handle = store
            .get(incoming_grad_id)
            .and_then(|tensor| tensor.device_handle.as_ref())
            .ok_or(AutogradError::InvalidTensorId(incoming_grad_id))?
            .clone();
        let sum_handle =
            store
                .backend()
                .add_into_device(&existing_handle, &incoming_handle, &incoming)?;
        store.replace_device_handle(existing_grad_id, sum_handle)?;
        return Ok(());
    }

    let incoming_data = store.to_host(incoming_grad_id)?;
    let existing = store
        .get_mut(existing_grad_id)
        .ok_or(AutogradError::InvalidTensorId(existing_grad_id))?;
    for (dst, src) in existing.data.iter_mut().zip(incoming_data) {
        *dst += src;
    }
    Ok(())
}

fn map_qwen35_forward_error(stage: &str, err: Qwen35Error) -> OpdError {
    match err {
        Qwen35Error::InputLenMismatch {
            input_len,
            expected_len,
        } => OpdError::InvalidInput(format!(
            "OPD {stage} Qwen3.5 forward input length mismatch: got \
             {input_len}, expected {expected_len}. Hint: verify prompt_ids, \
             generated rollout length, and position ids were built from the \
             same rollout."
        )),
        Qwen35Error::PositionOutOfBounds { position, upper } => OpdError::InvalidInput(format!(
            "OPD {stage} Qwen3.5 forward position id {position} is outside \
             rope cache size {upper}. Hint: reduce prompt length or \
             --rollout-len, or load/build a Qwen35Config with a larger \
             rope_cache_len_hint."
        )),
        Qwen35Error::InvalidConfig(reason) => OpdError::InvalidInput(format!(
            "OPD {stage} Qwen3.5 forward config error: {reason}. Hint: verify \
             Qwen35Config matches the checkpoint and that rope_cache_len_hint \
             covers prompt length plus rollout length."
        )),
        Qwen35Error::Autograd(err) => OpdError::InvalidInput(format!(
            "OPD {stage} Qwen3.5 forward autograd error: {err}. Hint: verify \
             the checkpoint tensor shapes match config.json, that teacher and \
             student use compatible Qwen3.5-family layouts, and include this \
             stage name in the OPD loader/model follow-up report."
        )),
        Qwen35Error::Config(err) => OpdError::InvalidInput(format!(
            "OPD {stage} Qwen3.5 config error: {err}. Hint: verify config.json \
             is a supported Qwen3/Qwen3.5-family config before running OPD."
        )),
    }
}

fn map_teacher_forward_error(stage: &str, err: TeacherForwardError) -> OpdError {
    match err {
        TeacherForwardError::Qwen35(err) => map_qwen35_forward_error(stage, err),
        TeacherForwardError::Autograd(err) => OpdError::InvalidInput(format!(
            "OPD {stage} teacher forward autograd error: {err}. Hint: verify \
             the teacher runtime shares the same TensorStore backend and returns \
             device-resident logits compatible with the student KL path."
        )),
        TeacherForwardError::InvalidInput(reason) => OpdError::InvalidInput(format!(
            "OPD {stage} teacher forward input error: {reason}. Hint: verify \
             prompt_ids, rollout ids, and positions are aligned before scoring \
             the rollout."
        )),
        TeacherForwardError::ApiRuntime(reason) => OpdError::InvalidInput(format!(
            "OPD {stage} API teacher runtime error: {reason}. Hint: verify \
             the API teacher endpoint is reachable, returns full logits for \
             every requested token position, and uses the same tokenizer/vocab \
             as the student."
        )),
        TeacherForwardError::ApiDecode(reason) => OpdError::InvalidInput(format!(
            "OPD {stage} API teacher logits decode error: {reason}. Hint: verify \
             the response shape is [seq,vocab] or [1,seq,vocab], dtype is f32 \
             or bf16, and logits_b64 is little-endian."
        )),
        #[cfg(feature = "cuda")]
        TeacherForwardError::InferRuntime(reason) => OpdError::InvalidInput(format!(
            "OPD {stage} infer teacher runtime error: {reason}. Hint: verify \
             the infer teacher model is loaded on CUDA, raw logits export is \
             available, and the token positions are contiguous from zero for \
             the current Path B bridge."
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn backward_windowed_pure_kl_cached_student_hidden<T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    rollout: &[u32],
    positions: &[u32],
    vocab: usize,
    gkd_config: GkdLossConfig<'_>,
    window_size: usize,
    student_target_params: &[TensorId],
    student_model_params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
    engine_offload: EngineOffloadMode,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<f32> {
    let kl_range = kl_logit_range(gkd_config.kl_mask, prompt_ids.len(), rollout.len())?;
    let windows = sequence_windows_for_range(kl_range, window_size)?;
    if windows.is_empty() {
        return Err(OpdError::InvalidInput(
            "OPD windowed Route B built zero KL windows. Hint: verify \
             prompt length, rollout length, and --logits-window-size."
                .to_owned(),
        ));
    }

    let mut teacher_tape = Tape::new();
    teacher_tape.set_enabled(false);
    if engine_offload.offloads_teacher() {
        let reload_started = Instant::now();
        log_opd_window_trace("kl", "teacher_reload_start", 0, reload_started, "");
        store
            .backend()
            .device_synchronize()
            .map_err(OpdError::from)?;
        teacher
            .reload_engine_weights()
            .map_err(|err| map_teacher_forward_error("teacher reload (windowed KL)", err))?;
        log_opd_window_trace("kl", "teacher_reload_done", 0, reload_started, "");
    }
    let teacher_started = Instant::now();
    log_opd_window_trace(
        "kl",
        "teacher_full_forward_start",
        0,
        teacher_started,
        format!("seq_len={}", rollout.len()),
    );
    let teacher_full_logits = teacher
        .forward_logits_device(rollout, positions, store, &mut teacher_tape)
        .map_err(|err| map_teacher_forward_error("teacher full KL cache", err))?;
    log_opd_window_trace(
        "kl",
        "teacher_full_forward_done",
        0,
        teacher_started,
        format!("shape={:?}", teacher_full_logits.shape),
    );
    let expected_teacher_shape = vec![1, rollout.len(), vocab];
    if teacher_full_logits.shape != expected_teacher_shape {
        return Err(OpdError::InvalidInput(format!(
            "OPD cached teacher logits shape mismatch: got {:?}, expected {:?}. \
             Hint: the TeacherForward implementation must return full \
             [batch=1, seq_len, vocab] logits when the windowed pure-KL path \
             caches teacher logits for reuse.",
            teacher_full_logits.shape, expected_teacher_shape
        )));
    }
    record_profile(profile, |profile| {
        profile.teacher_forward_seconds += teacher_started.elapsed().as_secs_f64();
    });

    if engine_offload.offloads_teacher() {
        store
            .backend()
            .device_synchronize()
            .map_err(OpdError::from)?;
        let freed = teacher
            .offload_engine_weights()
            .map_err(|err| map_teacher_forward_error("teacher offload (windowed KL)", err))?;
        eprintln!(
            "opd_engine_offload teacher_offloaded freed_bytes={freed} freed_mib={:.1}",
            freed as f64 / (1024.0 * 1024.0)
        );
    }

    tape.entries.clear();
    tape.set_enabled(true);
    let student_started = Instant::now();
    log_opd_window_trace(
        "kl",
        "student_hidden_forward_start",
        0,
        student_started,
        format!("seq_len={}", rollout.len()),
    );
    let student_hidden = student
        .forward_hidden_states(
            store,
            tape,
            rollout,
            positions,
            crate::context_parallel::CpContext::single(),
        )
        .map_err(|err| map_qwen35_forward_error("student windowed KL hidden", err))?;
    log_opd_window_trace(
        "kl",
        "student_hidden_forward_done",
        0,
        student_started,
        format!("tensor_id={student_hidden}"),
    );
    record_profile(profile, |profile| {
        profile.student_forward_seconds += student_started.elapsed().as_secs_f64();
    });

    let base_tape_len = tape.entries.len();
    let base_keep = store.live_ids().into_iter().collect::<HashSet<_>>();

    let mut total_loss = 0.0f32;
    let mut hidden_grad: Option<TensorId> = None;
    for (window_offset, window) in windows.into_iter().enumerate() {
        let window_index = window_offset + 1;
        let window_started = Instant::now();
        log_opd_window_trace(
            "kl",
            "start",
            window_index,
            window_started,
            format!(
                "start={} end={} len={} cached_hidden=true",
                window.start,
                window.end,
                window.len()
            ),
        );
        tape.entries.truncate(base_tape_len);
        tape.set_enabled(true);
        let mut window_tape = Tape::new();
        window_tape.set_enabled(false);

        let phase_started = Instant::now();
        log_opd_window_trace(
            "kl",
            "teacher_forward_start",
            window_index,
            window_started,
            "",
        );
        let teacher_tensor_id = slice(
            teacher_full_logits.tensor_id,
            &[0, window.start, 0],
            &[1, window.end, vocab],
            store,
            &mut window_tape,
        )?;
        let teacher_shape = vec![1, window.len(), vocab];
        log_opd_window_trace(
            "kl",
            "teacher_forward_done",
            window_index,
            window_started,
            format!("shape={teacher_shape:?} cached_full=true"),
        );
        record_profile(profile, |profile| {
            profile.teacher_forward_seconds += phase_started.elapsed().as_secs_f64();
        });
        validate_logits_shape("teacher windowed KL", &teacher_shape, window.len(), vocab)?;

        window_tape.set_enabled(true);
        let phase_started = Instant::now();
        let kl_loss = if gkd_config.kl_beta.is_some() || !gkd_config.fused_distill {
            let logits_started = Instant::now();
            log_opd_window_trace(
                "kl",
                "student_logits_window_start",
                window_index,
                window_started,
                format!(
                    "kl_beta={} fused_distill={}",
                    gkd_config.kl_beta.is_some(),
                    gkd_config.fused_distill
                ),
            );
            let student_logits = student
                .logits_from_hidden_window(store, &mut window_tape, student_hidden, window)
                .map_err(|err| map_qwen35_forward_error("student cached-hidden KL beta", err))?;
            record_profile(profile, |profile| {
                profile.student_forward_seconds += logits_started.elapsed().as_secs_f64();
            });
            let loss_started = Instant::now();
            let loss = kl_distill_loss_for_config(
                student_logits,
                teacher_tensor_id,
                window.len(),
                gkd_config.kl_chunk_size,
                gkd_config.kl_direction,
                gkd_config.kl_temperature,
                gkd_config.kl_beta,
                store,
                &mut window_tape,
            )?;
            record_profile(profile, |profile| {
                profile.kl_loss_seconds += loss_started.elapsed().as_secs_f64();
            });
            loss
        } else {
            log_opd_window_trace(
                "kl",
                "fused_linear_distill_start",
                window_index,
                window_started,
                format!(
                    "hidden_tensor_id={student_hidden} lm_head_tensor_id={}",
                    student.lm_head_weight_id()
                ),
            );
            // Opt-in fused path (--fused-distill). NOT default: H20 A/B measured it
            // ~205 s/step (lm_head on HOST, GPU 0%) vs ~3.9 s dense (GPU 98%) on the
            // 27B; loss matches dense to ~5e-4. Only worth it for windows too large to
            // materialize [window, vocab]. errors/2026-06-23-opd-fused-distill-default-host-bound.
            let loss = fused_linear_distill_loss(
                student_hidden,
                student.lm_head_weight_id(),
                teacher_tensor_id,
                window.start,
                window.len(),
                gkd_config.kl_chunk_size.unwrap_or(window.len()),
                gkd_config.kl_temperature,
                gkd_config.kl_direction,
                store,
                &mut window_tape,
            )?;
            record_profile(profile, |profile| {
                profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
            });
            loss
        };
        log_opd_window_trace(
            "kl",
            "kl_loss_done",
            window_index,
            window_started,
            format!("tensor_id={kl_loss}"),
        );
        let weight = window.len() as f32 / kl_range.len() as f32;
        log_opd_window_trace("kl", "backward_start", window_index, window_started, "");
        let loss_started = Instant::now();
        let weighted_loss = if (weight - 1.0).abs() < f32::EPSILON {
            kl_loss
        } else {
            mul_scalar(kl_loss, weight, store, &mut window_tape)?
        };
        let loss_value = store.to_host(weighted_loss)?[0];
        validate_loss_value(loss_value)?;
        total_loss += loss_value;
        record_profile(profile, |profile| {
            profile.kl_loss_seconds += loss_started.elapsed().as_secs_f64();
        });

        let phase_started = Instant::now();
        let window_grads = window_tape.backward_collect(weighted_loss, store)?;
        let window_hidden_grad = *window_grads
            .get(&student_hidden)
            .ok_or(AutogradError::MissingGradient(student_hidden))?;
        for &target_id in student_target_params {
            if let Some(&grad_id) = window_grads.get(&target_id) {
                store.accumulate_grad(target_id, grad_id)?;
            }
        }
        let keep_grads: Vec<_> = hidden_grad
            .into_iter()
            .chain([window_hidden_grad])
            .collect();
        cleanup_window_backward_retaining_base(
            store,
            tape,
            base_tape_len,
            &base_keep,
            student_model_params,
            keep_extra,
            &keep_grads,
        );
        if let Some(existing_hidden_grad) = hidden_grad {
            accumulate_tensor_grad(store, existing_hidden_grad, window_hidden_grad)?;
        } else {
            hidden_grad = Some(window_hidden_grad);
        }
        record_profile(profile, |profile| {
            profile.backward_seconds += phase_started.elapsed().as_secs_f64();
        });
        log_opd_window_trace(
            "kl",
            "backward_done",
            window_index,
            window_started,
            format!("loss_accum={total_loss:.12e}"),
        );
        log_opd_window_trace("kl", "cleanup_start", window_index, window_started, "");
        cleanup_window_backward_retaining_base(
            store,
            tape,
            base_tape_len,
            &base_keep,
            student_model_params,
            keep_extra,
            hidden_grad.as_slice(),
        );
        log_opd_window_trace("kl", "done", window_index, window_started, "");
    }

    store
        .free(teacher_full_logits.tensor_id)
        .map_err(OpdError::from)?;
    let hidden_grad = hidden_grad.ok_or(AutogradError::MissingGradient(student_hidden))?;
    let base_backward_started = Instant::now();
    log_opd_window_trace(
        "kl",
        "base_backward_start",
        0,
        base_backward_started,
        format!("seed_grad_id={hidden_grad}"),
    );
    tape.entries.truncate(base_tape_len);
    tape.set_enabled(false);
    if opd_backward_profile_enabled() {
        let (_, backward_profile) = tape.backward_from_seed_accumulate_targets_profiled(
            student_hidden,
            hidden_grad,
            store,
            student_target_params,
        )?;
        print_backward_profile("base", 0, total_loss, &backward_profile);
    } else {
        tape.backward_from_seed_accumulate_targets(
            student_hidden,
            hidden_grad,
            store,
            student_target_params,
        )?;
    }
    log_opd_window_trace("kl", "base_backward_done", 0, base_backward_started, "");
    record_profile(profile, |profile| {
        profile.backward_seconds += base_backward_started.elapsed().as_secs_f64();
    });
    cleanup_after_backward(store, tape, student_model_params, keep_extra);

    Ok(total_loss)
}

#[allow(clippy::too_many_arguments)]
fn backward_windowed_gkd_loss<T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    rollout: &[u32],
    positions: &[u32],
    vocab: usize,
    gkd_config: GkdLossConfig<'_>,
    window_size: usize,
    student_target_params: &[TensorId],
    student_model_params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
    engine_offload: EngineOffloadMode,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<f32> {
    let mut total_loss = 0.0f32;
    let mut backward_windows = 0usize;

    if gkd_config.lambda == 0.0 {
        return backward_windowed_pure_kl_cached_student_hidden(
            student,
            teacher,
            prompt_ids,
            rollout,
            positions,
            vocab,
            gkd_config,
            window_size,
            student_target_params,
            student_model_params,
            keep_extra,
            store,
            tape,
            engine_offload,
            profile,
        );
    }

    if gkd_config.lambda < 1.0 {
        let kl_range = kl_logit_range(gkd_config.kl_mask, prompt_ids.len(), rollout.len())?;
        for window in sequence_windows_for_range(kl_range, window_size)? {
            let window_index = backward_windows + 1;
            let window_started = Instant::now();
            log_opd_window_trace(
                "kl",
                "start",
                window_index,
                window_started,
                format!(
                    "start={} end={} len={}",
                    window.start,
                    window.end,
                    window.len()
                ),
            );
            tape.entries.clear();
            tape.set_enabled(false);

            let phase_started = Instant::now();
            log_opd_window_trace(
                "kl",
                "teacher_forward_start",
                window_index,
                window_started,
                "",
            );
            let teacher_logits = teacher
                .forward_logits_window_device(rollout, positions, window, store, tape)
                .map_err(|err| map_teacher_forward_error("teacher windowed KL", err))?;
            log_opd_window_trace(
                "kl",
                "teacher_forward_done",
                window_index,
                window_started,
                format!("shape={:?}", teacher_logits.shape),
            );
            record_profile(profile, |profile| {
                profile.teacher_forward_seconds += phase_started.elapsed().as_secs_f64();
            });
            validate_logits_shape(
                "teacher windowed KL",
                &teacher_logits.shape,
                window.len(),
                vocab,
            )?;

            tape.set_enabled(true);
            let phase_started = Instant::now();
            log_opd_window_trace(
                "kl",
                "student_forward_start",
                window_index,
                window_started,
                "",
            );
            let student_logits = student
                .forward_logits_window(store, tape, rollout, positions, window)
                .map_err(|err| map_qwen35_forward_error("student windowed KL", err))?;
            log_opd_window_trace(
                "kl",
                "student_forward_done",
                window_index,
                window_started,
                format!("tensor_id={student_logits}"),
            );
            let student_shape = store
                .get(student_logits)
                .ok_or(AutogradError::InvalidTensorId(student_logits))?
                .shape
                .clone();
            validate_logits_shape("student windowed KL", &student_shape, window.len(), vocab)?;
            record_profile(profile, |profile| {
                profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
            });

            let phase_started = Instant::now();
            log_opd_window_trace("kl", "kl_loss_start", window_index, window_started, "");
            let kl_loss = kl_distill_loss_for_config(
                student_logits,
                teacher_logits.tensor_id,
                window.len(),
                gkd_config.kl_chunk_size,
                gkd_config.kl_direction,
                gkd_config.kl_temperature,
                gkd_config.kl_beta,
                store,
                tape,
            )?;
            log_opd_window_trace(
                "kl",
                "kl_loss_done",
                window_index,
                window_started,
                format!("tensor_id={kl_loss}"),
            );
            record_profile(profile, |profile| {
                profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
            });
            let weight = (1.0 - gkd_config.lambda) * (window.len() as f32 / kl_range.len() as f32);
            log_opd_window_trace("kl", "backward_start", window_index, window_started, "");
            total_loss += backward_weighted_window_loss(kl_loss, weight, store, tape, profile)?;
            log_opd_window_trace(
                "kl",
                "backward_done",
                window_index,
                window_started,
                format!("loss_accum={total_loss:.12e}"),
            );
            backward_windows += 1;
            log_opd_window_trace("kl", "cleanup_start", window_index, window_started, "");
            cleanup_after_backward(store, tape, student_model_params, keep_extra);
            log_opd_window_trace("kl", "done", window_index, window_started, "");
        }
    }

    if gkd_config.lambda > 0.0 {
        match gkd_config.sft_anchor {
            GkdSftAnchor::StudentRollout => {
                let target_count = rollout.len().checked_sub(1).ok_or_else(|| {
                    OpdError::InvalidInput(
                        "GKD student-rollout SFT anchor requires at least two rollout \
                         tokens so logits can be trained against next-token labels."
                            .to_owned(),
                    )
                })?;
                if target_count == 0 {
                    return Err(OpdError::InvalidInput(
                        "GKD student-rollout SFT anchor requires at least two rollout \
                         tokens so logits can be trained against next-token labels."
                            .to_owned(),
                    ));
                }
                for target_window in sequence_windows(target_count, window_size)? {
                    tape.entries.clear();
                    tape.set_enabled(true);

                    let logits_window = target_window;
                    let phase_started = Instant::now();
                    let student_logits = student
                        .forward_logits_window(store, tape, rollout, positions, logits_window)
                        .map_err(|err| {
                            map_qwen35_forward_error("student windowed rollout SFT", err)
                        })?;
                    let student_shape = store
                        .get(student_logits)
                        .ok_or(AutogradError::InvalidTensorId(student_logits))?
                        .shape
                        .clone();
                    validate_logits_shape(
                        "student windowed rollout SFT",
                        &student_shape,
                        logits_window.len(),
                        vocab,
                    )?;
                    record_profile(profile, |profile| {
                        profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
                    });

                    let target_tokens = &rollout[target_window.start + 1..target_window.end + 1];
                    let phase_started = Instant::now();
                    let sft_loss = next_token_sft_loss_from_logits(
                        student_logits,
                        logits_window.len(),
                        0,
                        target_tokens,
                        vocab,
                        store,
                        tape,
                    )?;
                    record_profile(profile, |profile| {
                        profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
                    });
                    let weight =
                        gkd_config.lambda * (target_tokens.len() as f32 / target_count as f32);
                    total_loss +=
                        backward_weighted_window_loss(sft_loss, weight, store, tape, profile)?;
                    backward_windows += 1;
                    cleanup_after_backward(store, tape, student_model_params, keep_extra);
                }
            }
            GkdSftAnchor::CorpusTruth => {
                let corpus_tokens = gkd_config.corpus_tokens.ok_or_else(|| {
                    OpdError::InvalidInput(
                        "GKD corpus-truth SFT anchor requires corpus completion \
                         tokens. Hint: add completion/target fields to \
                         --prompts-file."
                            .to_owned(),
                    )
                })?;
                if prompt_ids.is_empty() || corpus_tokens.is_empty() {
                    return Err(OpdError::InvalidInput(
                        "GKD corpus-truth SFT anchor requires non-empty prompt and \
                         completion tokens."
                            .to_owned(),
                    ));
                }
                let total_len = prompt_ids
                    .len()
                    .checked_add(corpus_tokens.len())
                    .ok_or_else(|| {
                        OpdError::InvalidInput(
                            "GKD corpus-truth SFT prompt+completion length overflowed. \
                             Hint: reduce prompt/completion max tokens."
                                .to_owned(),
                        )
                    })?;
                if total_len > u32::MAX as usize {
                    return Err(OpdError::InvalidInput(format!(
                        "GKD corpus-truth SFT sequence length {total_len} exceeds \
                         u32::MAX RoPE position range. Hint: reduce prompt/completion \
                         max tokens."
                    )));
                }
                let sft_sequence: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(corpus_tokens.iter().copied())
                    .collect();
                let sft_positions = (0..total_len as u32).collect::<Vec<_>>();
                for target_window in sequence_windows(corpus_tokens.len(), window_size)? {
                    let logits_window = SequenceWindow {
                        start: prompt_ids.len() - 1 + target_window.start,
                        end: prompt_ids.len() - 1 + target_window.end,
                    };

                    tape.entries.clear();
                    tape.set_enabled(true);
                    let phase_started = Instant::now();
                    let student_logits = student
                        .forward_logits_window(
                            store,
                            tape,
                            &sft_sequence,
                            &sft_positions,
                            logits_window,
                        )
                        .map_err(|err| {
                            map_qwen35_forward_error("student windowed corpus SFT", err)
                        })?;
                    let student_shape = store
                        .get(student_logits)
                        .ok_or(AutogradError::InvalidTensorId(student_logits))?
                        .shape
                        .clone();
                    validate_logits_shape(
                        "student windowed corpus SFT",
                        &student_shape,
                        logits_window.len(),
                        vocab,
                    )?;
                    record_profile(profile, |profile| {
                        profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
                    });

                    let target_tokens = &corpus_tokens[target_window.start..target_window.end];
                    let phase_started = Instant::now();
                    let sft_loss = next_token_sft_loss_from_logits(
                        student_logits,
                        logits_window.len(),
                        0,
                        target_tokens,
                        vocab,
                        store,
                        tape,
                    )?;
                    record_profile(profile, |profile| {
                        profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
                    });
                    let weight = gkd_config.lambda
                        * (target_tokens.len() as f32 / corpus_tokens.len() as f32);
                    total_loss +=
                        backward_weighted_window_loss(sft_loss, weight, store, tape, profile)?;
                    backward_windows += 1;
                    cleanup_after_backward(store, tape, student_model_params, keep_extra);
                }
            }
        }
    }

    if backward_windows == 0 {
        return Err(OpdError::InvalidInput(
            "OPD windowed Route B built zero backward windows. Hint: verify \
             lambda, prompt length, rollout length, and --logits-window-size."
                .to_owned(),
        ));
    }
    Ok(total_loss)
}

/// Run one OPD step:
/// 1. Roll out `cfg.rollout_len` tokens from `student` starting from `prompt_ids`,
///    or use a whole `forced_rollout` trajectory selected by an external scorer.
/// 2. Forward `teacher` on the full rollout (tape disabled).
/// 3. Forward `student` on the full rollout (tape enabled).
/// 4. `kl_distill_loss(student_logits, teacher_logits, rollout.len(), 1.0, …)`.
/// 5. Backward + grad-clip + optimizer step.
/// 6. Clear ephemeral tensors so the next step starts from a clean store.
pub fn opd_step<O: Optimizer>(
    student: &Qwen35Model,
    teacher: &Qwen35Model,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    forced_rollout: Option<&[u32]>,
) -> Result<OpdStepOutcome> {
    let teacher = InProcessTeacher::new(teacher);
    opd_step_with_teacher_forward(
        student,
        &teacher,
        prompt_ids,
        cfg,
        student_params,
        optimizer,
        store,
        tape,
        forced_rollout,
    )
}

/// Batched rubric-OPD writeback: one forward+backward over B accepted `(prompt,
/// completion)` pairs (padded to a common length), completion-masked CE. The 27B
/// CE is overhead-bound (host op-dispatch, GPU ~0% util), so batching amortizes
/// the ~64-layer per-op host dispatch over B examples — near-B× throughput.
///
/// The forward yields post-final-norm hidden (not logits); a per-row fused chunked
/// CE projects only the masked completion positions through `lm_head`, so the dense
/// `[B, seq, vocab]` logits tile is never materialized (was ~24 GB at B=4 / 10K-tok
/// completions). Grad-checkpoints offload to host past the length that needs it
/// (`writeback_offload_for_seq`), mirroring the single-trajectory writeback.
/// `chunk_rows` bounds the transient `[chunk, vocab]` tile — same knob as the
/// single-trajectory path's `window_size` (`--writeback-window`).
#[allow(clippy::too_many_arguments)]
pub fn rubric_writeback_ce_step_batched<O: Optimizer>(
    student: &Qwen35Model,
    all_model_params: &[TensorId],
    trainable_params: &[TensorId],
    optimizer: &mut O,
    batch: &[(Vec<u32>, Vec<u32>)],
    vocab: usize,
    chunk_rows: usize,
    store: &mut TensorStore,
) -> Result<f32> {
    if batch.is_empty() {
        return Err(OpdError::InvalidInput(
            "rubric batched writeback requires a non-empty batch".to_owned(),
        ));
    }
    let mut max_len = 0usize;
    for (i, (prompt, completion)) in batch.iter().enumerate() {
        if prompt.is_empty() || completion.is_empty() {
            return Err(OpdError::InvalidInput(
                "rubric batched writeback: empty prompt or completion in batch".to_owned(),
            ));
        }
        validate_token_ids(
            &format!("rubric batched writeback row {i} completion"),
            completion,
            vocab,
        )?;
        max_len = max_len.max(prompt.len() + completion.len());
    }
    let b = batch.len();

    let mut tape = Tape::new();
    // The batched forward spans b*max_len tokens — offload grad-checkpoints past
    // the length that needs it.
    tape.set_offload_checkpoints(crate::runtime_flags::writeback_offload_for_seq(b * max_len));

    // Flat [B * max_len] input, each row = prompt ++ completion ++ pad(0). Causal
    // attention + right-padding means padding never affects earlier positions, and
    // the per-row CE only targets completion positions, so padding is inert.
    let flat: Vec<usize> = batch
        .iter()
        .flat_map(|(prompt, completion)| {
            prompt
                .iter()
                .map(|&t| t as usize)
                .chain(completion.iter().map(|&t| t as usize))
                .chain(std::iter::repeat_n(
                    0usize,
                    max_len - prompt.len() - completion.len(),
                ))
        })
        .collect();
    log_writeback_vram(store, "batched", "pre forward");
    let hidden = student
        .forward_batch_hidden(&flat, b, max_len, store, &mut tape)
        .map_err(OpdError::from)?;
    log_writeback_vram(store, "batched", "post forward");

    // Per-row completion-masked CE via the fused indexed path, preserving the prior
    // mean-of-row-means reduction: each row's fused call returns that row's token-
    // mean CE (fused scales by 1/row_targets), summed and scaled by 1/b below. Row
    // i occupies flat hidden rows [i*max_len, (i+1)*max_len); predicting position
    // p = prompt.len()-1+j targets completion[j] (next-token convention, identical
    // to `next_token_sft_loss_from_logits`).
    let mut total: Option<TensorId> = None;
    for (i, (prompt, completion)) in batch.iter().enumerate() {
        let row_base = i * max_len + (prompt.len() - 1);
        let position_indices: Vec<i32> = (row_base..row_base + completion.len())
            .map(|p| p as i32)
            .collect();
        let target_tokens: Vec<i32> = completion.iter().map(|&t| t as i32).collect();
        let ce = fused_linear_ce_loss_indexed(
            hidden,
            student.lm_head_weight_id(),
            &position_indices,
            &target_tokens,
            chunk_rows,
            None,
            store,
            &mut tape,
        )
        .map_err(OpdError::from)?;
        total = Some(match total {
            None => ce,
            Some(prev) => add(prev, ce, store, &mut tape).map_err(OpdError::from)?,
        });
    }
    let total = total.expect("batch is non-empty");
    let mean = mul_scalar(total, 1.0 / b as f32, store, &mut tape).map_err(OpdError::from)?;
    let loss_value = store.to_host(mean).map_err(OpdError::from)?[0];
    validate_loss_value(loss_value)?;
    log_writeback_vram(store, "batched", "post ce");
    backward_with_optional_profile(mean, loss_value, store, &mut tape)?;
    log_writeback_vram(store, "batched", "post backward");
    finite_optimizer_step(loss_value, trainable_params, 0.0, optimizer, store)?;
    let keep_extra: HashSet<TensorId> = HashSet::new();
    cleanup_after_backward(store, &mut tape, all_model_params, &keep_extra);
    log_writeback_vram(store, "batched", "post cleanup");
    Ok(loss_value)
}

/// Build the `(predicting-position p, target token id)` list for masked
/// next-token CE over `prompt_ids ++ response_ids`.
///
/// Logits at sequence position `p` predict token `p + 1` (the
/// `next_token_sft_loss_from_logits` convention). A target counts iff:
/// - it lies inside the response span: `p + 1 >= prompt_len`, AND
/// - the response token at that index is LLM-generated:
///   `response_mask[(p + 1) - prompt_len] == 1`.
///
/// Tool / environment tokens (`response_mask == 0`) are excluded as *targets*
/// but still appear in the context that earlier predictions condition on, so
/// the student never learns to hallucinate tool output. Pure (no device work)
/// so it is unit-testable without a model.
fn build_masked_loss_targets(
    full: &[u32],
    prompt_len: usize,
    response_mask: &[u8],
) -> Vec<(usize, usize)> {
    let seq_len = full.len();
    let mut targets = Vec::new();
    if seq_len < 2 || prompt_len == 0 {
        return targets;
    }
    let p_start = prompt_len - 1;
    for p in p_start..=(seq_len - 2) {
        let target_pos = p + 1;
        let resp_idx = target_pos - prompt_len;
        if resp_idx < response_mask.len() && response_mask[resp_idx] == 1 {
            targets.push((p, full[target_pos] as usize));
        }
    }
    targets
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Copy)]
struct VramSample {
    free: usize,
    total: usize,
    /// Mempool hoard in MiB; `None` = probe unavailable, not a measured 0.
    hoarded_mib: Option<u64>,
}

impl VramSample {
    fn used(self) -> usize {
        self.total.saturating_sub(self.free)
    }

    fn used_mib(self) -> usize {
        self.used() >> 20
    }

    fn free_mib(self) -> usize {
        self.free >> 20
    }

    fn total_mib(self) -> usize {
        self.total >> 20
    }
}

fn writeback_vram_trace_enabled() -> bool {
    std::env::var("ARLE_OPD_VRAM_TRACE").is_ok()
}

/// `n/a` keeps an unavailable probe distinct from a measured `0MiB`.
pub fn fmt_hoarded(mib: Option<u64>) -> String {
    mib.map_or_else(|| "n/a".to_string(), |m| format!("{m}MiB"))
}

fn log_writeback_vram(store: &TensorStore, scope: &str, label: &str) -> Option<VramSample> {
    if !writeback_vram_trace_enabled() {
        return None;
    }
    match store.backend().device_mem_info() {
        Some((free, total)) => {
            let sample = VramSample {
                free,
                total,
                hoarded_mib: store.backend().hoarded_mib(),
            };
            eprintln!(
                "[opd-vram] {scope} {label}: used={}MiB free={}MiB total={}MiB hoarded={}",
                sample.used_mib(),
                sample.free_mib(),
                sample.total_mib(),
                fmt_hoarded(sample.hoarded_mib),
            );
            Some(sample)
        }
        None => {
            eprintln!("[opd-vram] {scope} {label}: device_mem_info unavailable");
            None
        }
    }
}

fn log_writeback_vram_ledger(
    scope: &str,
    base: Option<VramSample>,
    post_forward: Option<VramSample>,
    post_backward: Option<VramSample>,
    post_cleanup: Option<VramSample>,
) {
    if !writeback_vram_trace_enabled() {
        return;
    }
    let used = |sample: Option<VramSample>| sample.map(|s| s.used_mib()).unwrap_or(0);
    let hoarded = |sample: Option<VramSample>| fmt_hoarded(sample.and_then(|s| s.hoarded_mib));
    let allocator_delta = match (base, post_cleanup) {
        (Some(base), Some(cleanup)) => cleanup.used().saturating_sub(base.used()) >> 20,
        _ => 0,
    };
    eprintln!(
        "[opd-vram-ledger] {scope} base_used_mib={} post_forward_used_mib={} \
         post_backward_used_mib={} post_cleanup_used_mib={} allocator_retained_delta_mib={} \
         hoarded_fwd/bwd/clean_mib={}/{}/{}",
        used(base),
        used(post_forward),
        used(post_backward),
        used(post_cleanup),
        allocator_delta,
        hoarded(post_forward),
        hoarded(post_backward),
        hoarded(post_cleanup),
    );
}

/// Loss applied by the shared writeback step.
pub enum WritebackLoss<'a> {
    Ce,
    Pg {
        rollout_logprobs: &'a [f32],
        weight: &'a [f32],
        form: WeightForm,
        kl_coef: f32,
    },
}

/// CE compatibility wrapper.
pub fn masked_writeback_ce_step<O: Optimizer>(
    student: &Qwen35Model,
    all_model_params: &[TensorId],
    trainable_params: &[TensorId],
    optimizer: &mut O,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    vocab: usize,
    window_size: usize,
    store: &mut TensorStore,
) -> Result<f32> {
    masked_writeback_step(
        WritebackLoss::Ce,
        student,
        all_model_params,
        trainable_params,
        optimizer,
        true,
        prompt_ids,
        response_ids,
        response_mask,
        vocab,
        window_size,
        crate::context_parallel::CpContext::single(),
        crate::context_parallel::DpContext::single(),
        store,
    )
    .map(|(loss, _)| loss)
}

/// Frozen prompt and trainable generated segment.
struct GenSegment {
    prompt_prefix: Vec<u32>,
    gen_ids: Vec<u32>,
    prompt_positions: Vec<u32>,
    gen_positions: Vec<u32>,
}

impl GenSegment {
    fn split(full: &[u32], prompt_len: usize) -> Self {
        let gen_start = prompt_len - 1;
        let seq_len = full.len();
        Self {
            prompt_prefix: full[0..gen_start].to_vec(),
            gen_ids: full[gen_start..].to_vec(),
            prompt_positions: (0..gen_start as u32).collect(),
            gen_positions: (gen_start as u32..seq_len as u32).collect(),
        }
    }
}

/// One checkpointed forward, chunked loss, backward, and optional optimizer step.
#[allow(clippy::too_many_arguments)]
pub fn masked_writeback_step<O: Optimizer>(
    loss_kind: WritebackLoss,
    student: &Qwen35Model,
    all_model_params: &[TensorId],
    trainable_params: &[TensorId],
    optimizer: &mut O,
    // False accumulates gradients for a later optimizer step.
    step_optimizer: bool,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    vocab: usize,
    window_size: usize,
    cp: crate::context_parallel::CpContext,
    dp: crate::context_parallel::DpContext,
    store: &mut TensorStore,
) -> Result<(f32, PgStats)> {
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "masked writeback requires a non-empty prompt".to_owned(),
        ));
    }
    if window_size == 0 {
        return Err(OpdError::InvalidInput(
            "masked writeback window_size must be > 0".to_owned(),
        ));
    }

    let prompt_len = prompt_ids.len();
    let mut full: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(response_ids.iter().copied())
        .collect();
    let mut seq_len = full.len();
    if seq_len > u32::MAX as usize {
        return Err(OpdError::InvalidInput(format!(
            "masked writeback trajectory length {seq_len} exceeds u32::MAX \
             position ids."
        )));
    }
    let mut positions: Vec<u32> = (0..seq_len as u32).collect();

    let loss_targets = build_masked_loss_targets(&full, prompt_len, response_mask);
    if loss_targets.is_empty() {
        eprintln!(
            "[masked-writeback] no LLM-generated targets (prompt_len={prompt_len}, \
             response_len={}, mask_ones={}); skipping (nothing to train)",
            response_ids.len(),
            response_mask.iter().filter(|&&m| m == 1).count(),
        );
        return Ok((0.0, PgStats::default()));
    }
    let total_targets = loss_targets.len();
    let chunk_rows = window_size; // reused: positions per fused-CE chunk.

    // Pad the tail so CP zigzag can split evenly; pad rows carry no target and are
    // never attended (see padded_seq_len). Single card is identity.
    let padded = cp.padded_seq_len(seq_len);
    if padded != seq_len {
        full.resize(padded, 0);
        positions.extend(seq_len as u32..padded as u32);
        seq_len = padded;
    }

    // Context-parallel sequence shard: this rank owns a zigzag pair of chunks
    // (front + back) for balanced causal work — `shard.local_rows()` is the gather
    // index. `total_targets` stays GLOBAL (built over `full`) so inv_n = 1/global
    // weights every rank's grad correctly; the post-backward all-reduce-sum then
    // yields the exact global mean.
    let shard = cp.shard(seq_len);
    // inv_n override under CP or DP (single-card keeps the fused op's local default,
    // byte-identical). CP shares ONE trajectory so total_targets is already global;
    // DP sums per-replica counts. The count reduce runs over the WORLD comm, so only
    // cp rank 0 contributes — every cp rank carries the same replica-global count and
    // would over-count by cp_size (G3: losses came back exactly /world).
    let inv_n_override = if dp.is_enabled() {
        let contribution = if cp.rank == 0 { total_targets } else { 0 };
        let global =
            crate::grad_clip::dp_group_sum_count(contribution, store).map_err(OpdError::from)?;
        crate::context_parallel::global_inv_n(global)
    } else if cp.is_enabled() {
        Some(1.0_f32 / total_targets as f32)
    } else {
        None
    };

    // Frozen-prompt-KV: forward only the gen segment, rebasing masked positions
    // into it. gen_start=0 keeps positions absolute for the byte-identical full
    // path. Every masked target p >= prompt_len-1 = gen_start, so p-gen_start>=0.
    // Disabled under CP (the gen-segment path is single-rank; CP shards the full
    // sequence instead).
    let frozen =
        crate::runtime_flags::writeback_frozen_prompt_kv() && prompt_len > 1 && !cp.is_enabled();
    let gen_start = if frozen { prompt_len - 1 } else { 0 };

    // Split the (predicting-position p, target token) pairs into the parallel
    // i32 index/target arrays the fused chunked CE consumes. `p` indexes the
    // hidden tensor's sequence rows; the targets are the hard next tokens.
    //
    // Vocab bound-check runs over the FULL global set on every rank (not per
    // shard): a bad token must fail all ranks together, or one rank errors while
    // the others wedge in the next collective. Clear OPD error before the autograd
    // bounds check (the lm_head's [vocab, hidden] also enforces it).
    for &(p, target) in &loss_targets {
        if target >= vocab {
            return Err(OpdError::InvalidInput(format!(
                "masked writeback target token {target} at position {p} exceeds vocab={vocab}"
            )));
        }
    }
    // Map each masked target to its local hidden row. `shard.local_of` handles both
    // paths: under CP the shard is zigzag so it maps position→local row via the chunk
    // map (and gen_start==0, frozen disabled); off-CP the shard is the whole sequence
    // so local_of(p)==Some(p) and the `- gen_start` rebases into the frozen-prompt gen
    // segment. Every frozen target p >= gen_start, so the subtraction can't underflow.
    let (position_indices, target_tokens): (Vec<i32>, Vec<i32>) = loss_targets
        .iter()
        .filter_map(|&(p, target)| {
            shard
                .local_of(p)
                .map(|local| ((local - gen_start) as i32, target as i32))
        })
        .unzip();

    let mut tape = Tape::new();
    // Offload per-layer grad-checkpoints to host RAM only past a length that needs
    // it: the H2D re-upload serializes on the host thread and starves the GPU on
    // short trajectories, but resident checkpoints OOM the allocator on long ones.
    // Seq-adaptive gate + rationale + anchors: `writeback_offload_for_seq`.
    let offload_checkpoints = crate::runtime_flags::writeback_offload_for_seq(seq_len);
    tape.set_offload_checkpoints(offload_checkpoints);
    tape.set_enabled(true);
    eprintln!("[masked-writeback] offload_checkpoints={offload_checkpoints} seq_len={seq_len}");
    let keep_extra: HashSet<TensorId> = HashSet::new();
    eprintln!(
        "[masked-writeback] seq_len={seq_len} total_targets={total_targets} \
         chunk_rows={chunk_rows} frozen={frozen} gen_start={gen_start}"
    );

    // ONE checkpointed forward → [1, rows, hidden] (rows = seq or gen_len). No
    // per-window re-forward of the growing prefix (was O(N²)). Frozen: only the
    // gen segment is forwarded/backwarded, seeded from the off-tape prompt KV.
    let vram_base = log_writeback_vram(store, "masked-writeback", "pre forward_hidden_states");
    let t_fwd = Instant::now();
    let hidden = if frozen {
        let seg = GenSegment::split(&full, prompt_len);
        student
            .forward_hidden_states_gen_segment(
                store,
                &mut tape,
                &seg.prompt_prefix,
                &seg.gen_ids,
                &seg.prompt_positions,
                &seg.gen_positions,
            )
            .map_err(|err| {
                map_qwen35_forward_error("masked writeback frozen-prompt-KV student hidden", err)
            })?
    } else if cp.is_enabled() {
        // CP: embed only this rank's zigzag shard rows; attention gathers the full
        // KV. positions carry ABSOLUTE ids so RoPE cos/sin match the gathered prefix.
        // local_rows() is the gather index (two chunks, front+back, in local order).
        let rows = shard.local_rows();
        let shard_ids: Vec<u32> = rows.iter().map(|&r| full[r]).collect();
        let shard_pos: Vec<u32> = rows.iter().map(|&r| positions[r]).collect();
        student
            .forward_hidden_states(store, &mut tape, &shard_ids, &shard_pos, cp)
            .map_err(|err| map_qwen35_forward_error("masked writeback CP student hidden", err))?
    } else {
        student
            .forward_hidden_states(store, &mut tape, &full, &positions, cp)
            .map_err(|err| map_qwen35_forward_error("masked writeback student hidden", err))?
    };
    let fwd_secs = t_fwd.elapsed().as_secs_f64();
    let vram_post_forward =
        log_writeback_vram(store, "masked-writeback", "post forward_hidden_states");
    eprintln!("[masked-writeback] phase=forward_hidden_states seconds={fwd_secs:.3}");

    // Chunked fused CE: per chunk computes hidden_chunk @ lm_headᵀ → logits → CE
    // → gradient, freeing each chunk. Never materializes [seq, vocab]. The loss
    // is already the mean CE per masked token and the gradient is scaled by 1/N,
    // so backward (seed 1.0) applies the per-token-mean update directly.
    let t_ce = Instant::now();
    let (loss, pg_stats) = match loss_kind {
        WritebackLoss::Ce => {
            // A CP sequence shard can legitimately own ZERO masked targets (e.g. a
            // prompt-heavy prefix shard), which the fused CE rejects. That rank must
            // still backprop through `hidden` so the forward's all_gather_seq KV
            // gather fires its reduce_scatter adjoint — else the CP group deadlocks.
            // A zero loss that depends on hidden (sum·0) yields exactly that: value
            // 0, grad 0, collectives in lockstep, and the post-backward grad
            // all-reduce sums a genuine zero contribution from this rank.
            let loss = if cp.is_enabled() && position_indices.is_empty() {
                let s = autograd::ops::sum(hidden, store, &mut tape).map_err(OpdError::from)?;
                mul_scalar(s, 0.0, store, &mut tape).map_err(OpdError::from)?
            } else {
                fused_linear_ce_loss_indexed(
                    hidden,
                    student.lm_head_weight_id(),
                    &position_indices,
                    &target_tokens,
                    chunk_rows,
                    inv_n_override,
                    store,
                    &mut tape,
                )
                .map_err(OpdError::from)?
            };
            (loss, PgStats::default())
        }
        WritebackLoss::Pg {
            rollout_logprobs,
            weight,
            form,
            kl_coef,
        } => {
            // PG under CP/DP needs per-shard rollout_logprobs/weight + a global-count
            // inv_n in the PG op — deferred (brick scope is CE writeback). CE is the
            // agent-OPD writeback loss; PG-parallel lands with the PG inv_n_override.
            if cp.is_enabled() || dp.is_enabled() {
                return Err(OpdError::InvalidInput(
                    "parallel writeback supports the CE loss only (PG-CP/DP deferred)".to_owned(),
                ));
            }
            if rollout_logprobs.len() != total_targets || weight.len() != total_targets {
                return Err(OpdError::InvalidInput(format!(
                    "masked writeback PG rollout_logprobs/weight len {}/{} != masked targets {total_targets}",
                    rollout_logprobs.len(),
                    weight.len(),
                )));
            }
            fused_linear_pg_loss_indexed(
                hidden,
                student.lm_head_weight_id(),
                &position_indices,
                &target_tokens,
                rollout_logprobs,
                weight,
                form,
                kl_coef,
                chunk_rows,
                store,
                &mut tape,
            )
            .map_err(OpdError::from)?
        }
    };

    let loss_value = store.to_host(loss).map_err(OpdError::from)?[0];
    // Reported loss: the local numerator is only this rank's partial sum (a
    // cp rank holds its sequence shard's targets, a dp replica its own data)
    // while inv_n is already 1/global_count — so the world sum of every
    // rank's partial IS the true global mean, printed identically on every
    // rank. Measured: cp=2 shards report 4.805783/6.064485, NOT identical.
    // Grads are unaffected: they already sum to the exact global mean via
    // all_reduce_cp_grads.
    let loss_value = if cp.is_enabled() || dp.is_enabled() {
        crate::grad_clip::dp_group_sum_scalar(loss_value, store).map_err(OpdError::from)?
    } else {
        loss_value
    };
    validate_loss_value(loss_value)?;
    let ce_secs = t_ce.elapsed().as_secs_f64();
    eprintln!("[masked-writeback] phase=fused_ce seconds={ce_secs:.3}");
    let t_bwd = Instant::now();
    backward_with_optional_profile(loss, loss_value, store, &mut tape)?;
    let bwd_secs = t_bwd.elapsed().as_secs_f64();
    let vram_post_backward = log_writeback_vram(store, "masked-writeback", "post backward");
    eprintln!("[masked-writeback] phase=backward seconds={bwd_secs:.3}");
    // CP replicates weights across the sequence-shard group and DP across the
    // batch-shard group; either way each rank produced only its contribution to
    // every weight grad. All-reduce-sum the trainable grads to the exact global
    // grad — MUST be after backward and before the optimizer step, and gated on
    // `step_optimizer` so PG's grad-accumulation (step_optimizer=false) doesn't
    // re-scale earlier microsteps by the group size. The collective is the same
    // sum over whatever NCCL group the backend holds (CP or DP); world==1 is a
    // no-op, so single-card stays byte-identical.
    if (cp.is_enabled() || dp.is_enabled()) && step_optimizer {
        crate::grad_clip::all_reduce_cp_grads(trainable_params, store).map_err(OpdError::from)?;
    }
    // Pre-step grad-norm telemetry. The prod `clip_grad_norm` logger sits on the
    // non-writeback OPD paths; the agentic-OPD writeback steps the optimizer here
    // directly, so surface the global L2 norm at this step too (env-gated).
    if std::env::var("ARLE_OPD_LOG_GRAD_NORM").is_ok() {
        let gn = crate::grad_clip::compute_global_norm_f64(trainable_params, store);
        eprintln!("[writeback-grad] grad_norm={gn:.6e}");
    }
    let t_opt = Instant::now();
    if step_optimizer {
        finite_optimizer_step(loss_value, trainable_params, 0.0, optimizer, store)?;
    }
    cleanup_after_backward(store, &mut tape, all_model_params, &keep_extra);
    let opt_secs = t_opt.elapsed().as_secs_f64();
    let vram_post_cleanup = log_writeback_vram(store, "masked-writeback", "post cleanup");
    log_writeback_vram_ledger(
        "masked-writeback",
        vram_base,
        vram_post_forward,
        vram_post_backward,
        vram_post_cleanup,
    );
    eprintln!("[masked-writeback] phase=optimizer_cleanup seconds={opt_secs:.3}");

    eprintln!("[masked-writeback] DONE loss={loss_value:.6} total_targets={total_targets}");
    // Mean CE per masked token (the fused loss already applied the 1/N mean).
    Ok((loss_value, pg_stats))
}

/// Positions per `[rows, vocab]` logits tile in [`capture_rollout_logprobs`];
/// bounds the transient softmax tile (≈ rows·vocab·4 B) since the capture has no
/// `window_size` knob of its own.
const CAPTURE_LOGPROB_CHUNK_ROWS: usize = 256;

/// Capture π_rollout: `log_softmax(logits)[target]` at each masked
/// (LLM-generated) response position, in the SAME order the fused writeback ops
/// consume (`build_masked_loss_targets`). Tape-OFF forward over `prompt ++
/// response` — call BEFORE any optimizer step so θ is still the rollout policy
/// (V0). Returns one f32 per masked target position (empty if none).
pub fn capture_rollout_logprobs(
    student: &Qwen35Model,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    store: &mut TensorStore,
) -> Result<Vec<f32>> {
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "capture_rollout_logprobs requires a non-empty prompt".to_owned(),
        ));
    }
    let prompt_len = prompt_ids.len();
    let full: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(response_ids.iter().copied())
        .collect();
    let seq_len = full.len();
    let loss_targets = build_masked_loss_targets(&full, prompt_len, response_mask);
    if loss_targets.is_empty() {
        return Ok(Vec::new());
    }
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    // Frozen-prompt-KV: forward only the gen segment. gen_start=0 keeps the full
    // path byte-identical. Masked rows p >= prompt_len-1 = gen_start, so the
    // gathered `rows` rebase to p-gen_start (>= 0) into the [gen_len, hidden]
    // tensor; output logprobs stay in the same target order.
    let frozen = crate::runtime_flags::writeback_frozen_prompt_kv() && prompt_len > 1;
    let gen_start = if frozen { prompt_len - 1 } else { 0 };
    let hidden_rows = seq_len - gen_start;

    // Snapshot live tensors so the forward + per-chunk projection intermediates
    // (hidden, logits, logp, gathered) + the tape's checkpoints are reclaimed at
    // the end — `to_host` copies but does not free them, so trajectories leak.
    let keep_ids: HashSet<TensorId> = store.live_ids().into_iter().collect();

    // Tape ENABLED (but never backwarded) so `should_checkpoint` engages and the
    // forward frees per-layer activations instead of piling all 64 layers'
    // [seq, hidden] resident — a tape-OFF forward has no checkpointing and OOMs on
    // long trajectories atop the two resident 27B models. Checkpointing is
    // numerically exact, so π_rollout is unchanged. Needs `--gradient-checkpointing`.
    let mut tape = Tape::new();
    tape.set_enabled(true);
    // Same seq-adaptive host-offload gate as the writeback forward. Capture is
    // never backwarded, so offloaded checkpoint inputs are never re-uploaded —
    // the forward retains no device activations across layers (resident
    // checkpoints spiked 50.7→97.4 GB and OOMed at seq≈15K; offload is
    // numerically transparent, see `checkpoint_offload_is_transparent`).
    tape.set_offload_checkpoints(crate::runtime_flags::writeback_offload_for_seq(seq_len));
    let hidden = if frozen {
        let seg = GenSegment::split(&full, prompt_len);
        student
            .forward_hidden_states_gen_segment(
                store,
                &mut tape,
                &seg.prompt_prefix,
                &seg.gen_ids,
                &seg.prompt_positions,
                &seg.gen_positions,
            )
            .map_err(|err| {
                map_qwen35_forward_error("rollout-logprob frozen-prompt-KV student hidden", err)
            })?
    } else {
        student
            .forward_hidden_states(
                store,
                &mut tape,
                &full,
                &positions,
                crate::context_parallel::CpContext::single(),
            )
            .map_err(|err| map_qwen35_forward_error("rollout-logprob student hidden", err))?
    };
    let hidden_dim = *store
        .get(hidden)
        .ok_or(AutogradError::InvalidTensorId(hidden))?
        .shape
        .last()
        .ok_or_else(|| OpdError::InvalidInput("rollout-logprob: empty hidden shape".to_owned()))?;
    let dbg = std::env::var("ARLE_OPD_LOG_DIS_STATS").is_ok();
    let shp = |store: &TensorStore, id| store.get(id).map(|t| t.shape.clone());
    if dbg {
        eprintln!(
            "[capture] seq_len={seq_len} targets={} hidden_shape={:?} hidden_dim={hidden_dim}",
            loss_targets.len(),
            shp(store, hidden),
        );
    }
    let hidden_2d =
        reshape(hidden, &[hidden_rows, hidden_dim], store, &mut tape).map_err(OpdError::from)?;
    let lm_head = student.lm_head_weight_id();

    let mut logprobs = Vec::with_capacity(loss_targets.len());
    for chunk in loss_targets.chunks(CAPTURE_LOGPROB_CHUNK_ROWS) {
        let rows: Vec<usize> = chunk.iter().map(|&(p, _)| p - gen_start).collect();
        let targets: Vec<usize> = chunk.iter().map(|&(_, t)| t).collect();
        // Gather this chunk's hidden rows, project through lm_head, log-softmax,
        // then pick each row's target logprob. Never materializes [seq, vocab].
        // `embedding` row-gathers but emits [1, chunk, hidden]; reshape to rank-2
        // so the bf16 matmul_bt forward (which rejects rank-3) accepts it.
        let rows_hidden_3d =
            embedding(hidden_2d, &rows, store, &mut tape).map_err(OpdError::from)?;
        let rows_hidden = reshape(rows_hidden_3d, &[rows.len(), hidden_dim], store, &mut tape)
            .map_err(OpdError::from)?;
        if dbg {
            eprintln!(
                "[capture] chunk rows={} rows_hidden={:?} lm_head={:?}",
                rows.len(),
                shp(store, rows_hidden),
                shp(store, lm_head),
            );
        }
        let logits = matmul_bt(rows_hidden, lm_head, store, &mut tape).map_err(OpdError::from)?;
        let logp = log_softmax(logits, store, &mut tape).map_err(OpdError::from)?;
        let gathered = gather_last_dim(logp, &targets, store, &mut tape).map_err(OpdError::from)?;
        logprobs.extend_from_slice(&store.to_host(gathered).map_err(OpdError::from)?);
    }
    store.retain_ids(&keep_ids);
    Ok(logprobs)
}

/// Skip-Observation Generalized Advantage Estimation (SAO Phase 2). `values`
/// holds V(s_t) at the masked (LLM-generated) positions ONLY, in trajectory
/// order — so the recursion's "next value" is the next LLM token's, and
/// environment/tool observation tokens are skipped for free. `terminal_reward`
/// lands on the final generated token (agentic reward is trajectory-terminal).
/// Returns `(advantages, returns)`, `returns = advantages + values` (the MSE
/// target for the critic). γ=discount, λ=GAE trace.
pub fn skip_obs_gae(
    values: &[f32],
    terminal_reward: f32,
    gamma: f32,
    lam: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = values.len();
    let mut advantages = vec![0.0f32; n];
    let mut gae = 0.0f32;
    for t in (0..n).rev() {
        let reward = if t == n - 1 { terminal_reward } else { 0.0 };
        let next_value = if t == n - 1 { 0.0 } else { values[t + 1] };
        let delta = reward + gamma * next_value - values[t];
        gae = delta + gamma * lam * gae;
        advantages[t] = gae;
    }
    let returns = advantages.iter().zip(values).map(|(a, v)| a + v).collect();
    (advantages, returns)
}

/// SAO Phase 2 value critic: a single linear head `V(s) = hidden · wᵀ` on the
/// (frozen) student trunk, its own AdamW + LR. **Frozen-Attention**: both the
/// GAE value read and the MSE update project a DETACHED copy of the masked
/// hidden rows (host round-trip), so gradient reaches `weight` only — never the
/// base. Zero-init → V₀(s)=0 → round-0 GAE = discounted reward-to-go (MC), a
/// stable cold start (no separate value-pretraining phase).
///
/// ponytail: one 27B forward per trajectory (`masked_hidden`) feeds BOTH the GAE
/// values and the MSE update; K=1 update/policy-step. Upgrade to K=2 / a fused
/// value-in-writeback forward only if the critic's fit lags or the writeback
/// wall dominates rollout — neither observed yet.
pub struct ValueCritic {
    weight: TensorId,
    params: [TensorId; 1],
    opt: AdamW,
    gamma: f32,
    lam: f32,
}

impl ValueCritic {
    pub fn new(
        hidden_dim: usize,
        lr: f32,
        gamma: f32,
        lam: f32,
        store: &mut TensorStore,
    ) -> Result<Self> {
        let weight = store
            .from_slice(&vec![0.0f32; hidden_dim], &[1, hidden_dim])
            .map_err(OpdError::from)?;
        store
            .get_mut(weight)
            .ok_or(AutogradError::InvalidTensorId(weight))?
            .requires_grad = true;
        Ok(Self {
            weight,
            params: [weight],
            opt: AdamW::new(lr, (0.9, 0.999), 1.0e-8, 0.0),
            gamma,
            lam,
        })
    }

    /// The critic's parameter ids — the policy writeback must keep these in its
    /// cleanup set (`cleanup_after_backward` frees everything not in
    /// `all_model_params`, and the critic weight is deliberately NOT a student
    /// param), else the next `update` hits a freed weight.
    pub fn param_ids(&self) -> &[TensorId] {
        &self.params
    }

    /// The masked (LLM-generated) hidden rows as a DETACHED host copy, in
    /// `build_masked_loss_targets` order. One checkpointed forward (tape-on but
    /// never backwarded, like `capture_rollout_logprobs`); returns
    /// `(rows_flat[n*hidden], n)`, empty when the trajectory has no LLM tokens.
    fn masked_hidden(
        &self,
        student: &Qwen35Model,
        prompt_ids: &[u32],
        response_ids: &[u32],
        response_mask: &[u8],
        store: &mut TensorStore,
    ) -> Result<(Vec<f32>, usize)> {
        let prompt_len = prompt_ids.len();
        let full: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(response_ids.iter().copied())
            .collect();
        let seq_len = full.len();
        let loss_targets = build_masked_loss_targets(&full, prompt_len, response_mask);
        if loss_targets.is_empty() {
            return Ok((Vec::new(), 0));
        }
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        // Frozen-prompt-KV: gen-segment forward, gather masked rows rebased by
        // -gen_start. gen_start=0 keeps the full path byte-identical. Returned
        // (rows_flat, n) semantics are unchanged.
        let frozen = crate::runtime_flags::writeback_frozen_prompt_kv() && prompt_len > 1;
        let gen_start = if frozen { prompt_len - 1 } else { 0 };
        let hidden_rows = seq_len - gen_start;
        let keep_ids: HashSet<TensorId> = store.live_ids().into_iter().collect();
        let mut tape = Tape::new();
        tape.set_enabled(true);
        let hidden = if frozen {
            let seg = GenSegment::split(&full, prompt_len);
            student
                .forward_hidden_states_gen_segment(
                    store,
                    &mut tape,
                    &seg.prompt_prefix,
                    &seg.gen_ids,
                    &seg.prompt_positions,
                    &seg.gen_positions,
                )
                .map_err(|err| {
                    map_qwen35_forward_error("value-critic frozen-prompt-KV student hidden", err)
                })?
        } else {
            student
                .forward_hidden_states(
                    store,
                    &mut tape,
                    &full,
                    &positions,
                    crate::context_parallel::CpContext::single(),
                )
                .map_err(|err| map_qwen35_forward_error("value-critic student hidden", err))?
        };
        let hidden_dim = *store
            .get(hidden)
            .ok_or(AutogradError::InvalidTensorId(hidden))?
            .shape
            .last()
            .ok_or_else(|| OpdError::InvalidInput("value-critic: empty hidden shape".to_owned()))?;
        let hidden_2d = reshape(hidden, &[hidden_rows, hidden_dim], store, &mut tape)
            .map_err(OpdError::from)?;
        let rows: Vec<usize> = loss_targets.iter().map(|&(p, _)| p - gen_start).collect();
        let n = rows.len();
        let rows_hidden_3d =
            embedding(hidden_2d, &rows, store, &mut tape).map_err(OpdError::from)?;
        let rows_hidden =
            reshape(rows_hidden_3d, &[n, hidden_dim], store, &mut tape).map_err(OpdError::from)?;
        let rows_flat = store.to_host(rows_hidden).map_err(OpdError::from)?;
        store.retain_ids(&keep_ids);
        Ok((rows_flat, n))
    }

    /// Skip-Obs GAE advantages + MSE-target returns for one trajectory, from the
    /// current (detached) critic values. `advantages`/`returns` are empty when
    /// the trajectory has no LLM tokens (caller skips it).
    pub fn advantages(
        &self,
        student: &Qwen35Model,
        prompt_ids: &[u32],
        response_ids: &[u32],
        response_mask: &[u8],
        terminal_reward: f32,
        store: &mut TensorStore,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (rows_flat, n) =
            self.masked_hidden(student, prompt_ids, response_ids, response_mask, store)?;
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let hidden_dim = rows_flat.len() / n;
        let w = store.to_host(self.weight).map_err(OpdError::from)?;
        let values: Vec<f32> = (0..n)
            .map(|i| {
                rows_flat[i * hidden_dim..(i + 1) * hidden_dim]
                    .iter()
                    .zip(&w)
                    .map(|(h, wj)| h * wj)
                    .sum()
            })
            .collect();
        Ok(skip_obs_gae(&values, terminal_reward, self.gamma, self.lam))
    }

    /// One MSE(V(s), returns) update. Frozen-attention: the masked hidden rows
    /// are a detached constant, so backward accumulates grad on `weight` only.
    /// Returns the MSE loss (0.0 when nothing to fit).
    pub fn update(
        &mut self,
        student: &Qwen35Model,
        prompt_ids: &[u32],
        response_ids: &[u32],
        response_mask: &[u8],
        returns: &[f32],
        store: &mut TensorStore,
    ) -> Result<f32> {
        let (rows_flat, n) =
            self.masked_hidden(student, prompt_ids, response_ids, response_mask, store)?;
        if n == 0 {
            return Ok(0.0);
        }
        if returns.len() != n {
            return Err(OpdError::InvalidInput(format!(
                "value-critic returns len {} != masked targets {n}",
                returns.len()
            )));
        }
        let hidden_dim = rows_flat.len() / n;
        // Free the MSE graph after the step (rows is [n, hidden] — tens of MB);
        // `weight` + optimizer moments live in `keep_ids` and survive.
        let keep_ids: HashSet<TensorId> = store.live_ids().into_iter().collect();
        let mut tape = Tape::new();
        tape.set_enabled(true);
        // Detached hidden rows (rg=false) → grad flows to `weight` only.
        let rows = store
            .from_slice(&rows_flat, &[n, hidden_dim])
            .map_err(OpdError::from)?;
        let value = matmul_bt(rows, self.weight, store, &mut tape).map_err(OpdError::from)?; // [n,1]
        let value_1d = reshape(value, &[n], store, &mut tape).map_err(OpdError::from)?;
        let neg_returns: Vec<f32> = returns.iter().map(|r| -r).collect();
        let neg_returns_id = store
            .from_slice(&neg_returns, &[n])
            .map_err(OpdError::from)?;
        let diff = add(value_1d, neg_returns_id, store, &mut tape).map_err(OpdError::from)?;
        let sq = mul(diff, diff, store, &mut tape).map_err(OpdError::from)?;
        let mse = mean(sq, store, &mut tape).map_err(OpdError::from)?;
        let mse_value = store.to_host(mse).map_err(OpdError::from)?[0];
        validate_loss_value(mse_value)?;
        tape.backward(mse, store).map_err(OpdError::from)?;
        finite_optimizer_step(mse_value, &self.params, 0.0, &mut self.opt, store)?;
        store.retain_ids(&keep_ids);
        Ok(mse_value)
    }
}

/// Frozen-prompt compatibility wrapper.
#[allow(clippy::too_many_arguments)]
pub fn masked_writeback_ce_step_frozen_prompt_kv<O: Optimizer>(
    student: &Qwen35Model,
    all_model_params: &[TensorId],
    trainable_params: &[TensorId],
    optimizer: &mut O,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    vocab: usize,
    window_size: usize,
    store: &mut TensorStore,
) -> Result<f32> {
    masked_writeback_step(
        WritebackLoss::Ce,
        student,
        all_model_params,
        trainable_params,
        optimizer,
        true,
        prompt_ids,
        response_ids,
        response_mask,
        vocab,
        window_size,
        crate::context_parallel::CpContext::single(),
        crate::context_parallel::DpContext::single(),
        store,
    )
    .map(|(loss, _)| loss)
}

/// Dispatch CE writeback with optional context parallelism.
#[allow(clippy::too_many_arguments)]
pub fn masked_writeback_ce_step_dispatch<O: Optimizer>(
    student: &Qwen35Model,
    all_model_params: &[TensorId],
    trainable_params: &[TensorId],
    optimizer: &mut O,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    vocab: usize,
    window_size: usize,
    cp: crate::context_parallel::CpContext,
    dp: crate::context_parallel::DpContext,
    store: &mut TensorStore,
) -> Result<f32> {
    masked_writeback_step(
        WritebackLoss::Ce,
        student,
        all_model_params,
        trainable_params,
        optimizer,
        true,
        prompt_ids,
        response_ids,
        response_mask,
        vocab,
        window_size,
        cp,
        dp,
        store,
    )
    .map(|(loss, _)| loss)
}

/// Group the masked (LLM-generated) predicting positions into maximal
/// contiguous `[start, end)` runs, then sub-window each run by `window_size`.
///
/// `masked_positions` are the predicting positions `p` (sorted ascending) whose
/// next token is LLM-generated — the ones that receive KL loss. Tool/environment
/// tokens leave gaps, so consecutive `p`s form runs; scoring one run's logit tile
/// at a time keeps tool-token positions out of the loss (mirroring the masked-CE
/// path) while bounding each `[1, window, vocab]` tile to `window_size` rows.
fn masked_gkd_windows(masked_positions: &[usize], window_size: usize) -> Vec<SequenceWindow> {
    let mut windows = Vec::new();
    let mut i = 0;
    while i < masked_positions.len() {
        let start = masked_positions[i];
        let mut end = start + 1;
        let mut j = i + 1;
        while j < masked_positions.len() && masked_positions[j] == end {
            end += 1;
            j += 1;
        }
        // Sub-window the contiguous run so no logit tile exceeds window_size rows.
        let mut s = start;
        while s < end {
            let e = (s + window_size).min(end);
            windows.push(SequenceWindow { start: s, end: e });
            s = e;
        }
        i = j;
    }
    windows
}

/// GKD (per-token teacher-KL) writeback for agent-OPD replay — the distillation
/// sibling of [`masked_writeback_ce_step`]. Instead of hard next-token CE on the
/// passing trajectory tokens, it distils a TEACHER's per-position distribution on
/// the SAME masked (LLM-generated) positions via forward-KL, so the signal is
/// dense (every vocab logit, not one hard target) rather than pure reproduction.
///
/// Mirrors the CE path's structure: ONE gradient-checkpointed
/// `forward_hidden_states` over `prompt ++ response` → `[1, seq, hidden]`, then
/// per masked window `[ws, we)` computes the student logits
/// `hidden[ws..we] @ lm_headᵀ` → `[1, w, vocab]`, forwards the frozen TEACHER over
/// the same window → `[1, w, vocab]`, and reuses
/// [`kl_distill_loss_chunked`](crate::loss::kl_distill_loss_chunked) (forward KL,
/// `batchmean`) — chunking the softmax intermediates. Windows track contiguous
/// runs of the `response_mask` so tool/environment positions never receive loss.
/// The returned scalar is the target-count-weighted mean KL per masked token.
///
/// `window_size` (reuse `--writeback-window`) bounds each `[1, w, vocab]` tile;
/// unlike the CE path a full-vocab logit tile is inherent to KL, so prefer a
/// smaller window here than for masked CE.
///
/// `entropy_weight > 0` (AEPO) is not yet wired — per-position entropy weighting
/// needs the pre-reduction per-position KL, which `kl_distill_loss_chunked`
/// collapses; it is flagged (loud TODO log), never silently dropped.
#[allow(clippy::too_many_arguments)]
pub fn gkd_writeback_step<O: Optimizer, T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    all_model_params: &[TensorId],
    trainable_params: &[TensorId],
    optimizer: &mut O,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    vocab: usize,
    window_size: usize,
    temperature: f32,
    entropy_weight: f32,
    store: &mut TensorStore,
) -> Result<f32> {
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "GKD writeback requires a non-empty prompt".to_owned(),
        ));
    }
    if window_size == 0 {
        return Err(OpdError::InvalidInput(
            "GKD writeback window_size must be > 0".to_owned(),
        ));
    }
    if teacher.vocab_size() != vocab {
        return Err(OpdError::InvalidInput(format!(
            "GKD writeback teacher vocab_size {} != student vocab {vocab}",
            teacher.vocab_size()
        )));
    }
    if entropy_weight > 0.0 {
        // AEPO entropy weighting stub — flagged, not silently dropped. TODO: expose
        // the per-position KL before reduction so each position can be scaled by
        // (1 + w * normalized_student_entropy); kl_distill_loss_chunked currently
        // returns only the reduced mean.
        eprintln!(
            "[gkd-writeback] TODO --gkd-entropy-weight={entropy_weight} is NOT YET \
             IMPLEMENTED (needs pre-reduction per-position KL); running UNWEIGHTED KL"
        );
    }

    let prompt_len = prompt_ids.len();
    let full: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(response_ids.iter().copied())
        .collect();
    let seq_len = full.len();
    if seq_len > u32::MAX as usize {
        return Err(OpdError::InvalidInput(format!(
            "GKD writeback trajectory length {seq_len} exceeds u32::MAX position ids."
        )));
    }
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let loss_targets = build_masked_loss_targets(&full, prompt_len, response_mask);
    if loss_targets.is_empty() {
        eprintln!(
            "[gkd-writeback] no LLM-generated targets (prompt_len={prompt_len}, \
             response_len={}, mask_ones={}); skipping (nothing to train)",
            response_ids.len(),
            response_mask.iter().filter(|&&m| m == 1).count(),
        );
        return Ok(0.0);
    }
    let masked_positions: Vec<usize> = loss_targets.iter().map(|&(p, _)| p).collect();
    let total_targets = masked_positions.len();
    let windows = masked_gkd_windows(&masked_positions, window_size);

    let mut tape = Tape::new();
    let offload_checkpoints = crate::runtime_flags::writeback_offload_for_seq(seq_len);
    tape.set_offload_checkpoints(offload_checkpoints);
    tape.set_enabled(true);
    // Retain the teacher's params across the post-backward `retain_ids` prune:
    // the EMA adapter (and any teacher-only weights) are NOT in the student's
    // `all_model_params`, so without this they would be freed and the next step's
    // teacher forward / EMA update would read dropped tensors.
    let keep_extra: HashSet<TensorId> = teacher.parameter_ids().iter().copied().collect();
    eprintln!(
        "[gkd-writeback] seq_len={seq_len} total_targets={total_targets} \
         windows={} temperature={temperature} offload_checkpoints={offload_checkpoints}",
        windows.len()
    );

    // ONE checkpointed forward over prompt++response → [1, seq, hidden]; the
    // student logits for each window are hidden[ws..we] @ lm_headᵀ (never the full
    // [seq, vocab]). The teacher scores the same window separately (frozen).
    let hidden = student
        .forward_hidden_states(
            store,
            &mut tape,
            &full,
            &positions,
            crate::context_parallel::CpContext::single(),
        )
        .map_err(|err| map_qwen35_forward_error("GKD writeback student hidden", err))?;
    let hidden_dim = *store
        .get(hidden)
        .ok_or(AutogradError::InvalidTensorId(hidden))?
        .shape
        .last()
        .ok_or_else(|| OpdError::InvalidInput("GKD writeback: empty hidden shape".to_owned()))?;
    let lm_head = student.lm_head_weight_id();

    let mut total_loss: Option<TensorId> = None;
    for window in &windows {
        let w = window.end - window.start;
        // Student logits for this window: slice the hidden rows, project through
        // the lm_head, reshape to [1, w, vocab] to match the teacher window shape.
        let hidden_slice = slice(
            hidden,
            &[0, window.start, 0],
            &[1, window.end, hidden_dim],
            store,
            &mut tape,
        )
        .map_err(OpdError::from)?;
        let hidden_2d =
            reshape(hidden_slice, &[w, hidden_dim], store, &mut tape).map_err(OpdError::from)?;
        let student_logits_2d =
            matmul_bt(hidden_2d, lm_head, store, &mut tape).map_err(OpdError::from)?;
        let student_logits =
            reshape(student_logits_2d, &[1, w, vocab], store, &mut tape).map_err(OpdError::from)?;

        let teacher_logits = teacher
            .forward_logits_window_device(&full, &positions, *window, store, &mut tape)
            .map_err(|err| OpdError::InvalidInput(format!("GKD teacher window forward: {err}")))?;

        let kl = kl_distill_loss_chunked(
            student_logits,
            teacher_logits.tensor_id,
            w,
            DEFAULT_KL_CHUNK_SIZE,
            temperature,
            KlDirection::Forward,
            store,
            &mut tape,
        )
        .map_err(OpdError::from)?;
        // Target-count weight so the accumulated loss is the mean KL per masked
        // token (each window contributes its share of the total masked positions).
        let weighted = mul_scalar(kl, w as f32 / total_targets as f32, store, &mut tape)
            .map_err(OpdError::from)?;
        total_loss = Some(match total_loss {
            Some(prev) => add(prev, weighted, store, &mut tape).map_err(OpdError::from)?,
            None => weighted,
        });
    }

    let loss = total_loss.ok_or_else(|| {
        OpdError::InvalidInput(
            "GKD writeback produced no windows despite non-empty targets".to_owned(),
        )
    })?;
    let loss_value = store.to_host(loss).map_err(OpdError::from)?[0];
    validate_loss_value(loss_value)?;
    backward_with_optional_profile(loss, loss_value, store, &mut tape)?;
    finite_optimizer_step(loss_value, trainable_params, 0.0, optimizer, store)?;
    cleanup_after_backward(store, &mut tape, all_model_params, &keep_extra);

    eprintln!("[gkd-writeback] DONE mean_kl={loss_value:.6} total_targets={total_targets}");
    Ok(loss_value)
}

pub fn opd_step_with_teacher_forward<O: Optimizer, T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    forced_rollout: Option<&[u32]>,
) -> Result<OpdStepOutcome> {
    opd_step_with_teacher_forward_profiled(
        student,
        teacher,
        prompt_ids,
        cfg,
        student_params,
        optimizer,
        store,
        tape,
        forced_rollout,
        None,
    )
}

pub fn opd_step_with_teacher_forward_profiled<O: Optimizer, T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    forced_rollout: Option<&[u32]>,
    profile: Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    opd_step_with_teacher_forward_profiled_gkd(
        student,
        teacher,
        prompt_ids,
        cfg,
        student_params,
        optimizer,
        store,
        tape,
        0.0,
        forced_rollout,
        profile,
    )
}

pub fn opd_step_with_teacher_forward_profiled_gkd<O: Optimizer, T: TeacherForward + ?Sized>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    gkd_lambda: f32,
    forced_rollout: Option<&[u32]>,
    profile: Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    opd_step_with_teacher_forward_profiled_gkd_anchor(
        student,
        teacher,
        prompt_ids,
        cfg,
        student_params,
        optimizer,
        store,
        tape,
        GkdLossConfig {
            lambda: gkd_lambda,
            ..GkdLossConfig::default()
        },
        forced_rollout,
        #[cfg(feature = "cuda")]
        None,
        profile,
    )
}

/// Student rollout phase: infer-student decode (cuda) or train-crate decode,
/// producing the `rollout` that downstream KL/backward consumes.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn run_opd_rollout_phase(
    student: &Qwen35Model,
    prompt_ids: &[u32],
    cfg: &OpdStepConfig,
    vocab: usize,
    forced_rollout: Option<&[u32]>,
    rollout_keep_base: &HashSet<TensorId>,
    engine_offload: EngineOffloadMode,
    total_started: Instant,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
    #[cfg(feature = "cuda")] infer_rollout: Option<&InferRolloutCtx<'_>>,
) -> Result<Vec<u32>> {
    // 1. Student rollout — tape disabled, no backward graph for sample tokens.
    let phase_started = Instant::now();
    store.retain_ids(rollout_keep_base);
    let rollout = if let Some(forced_rollout) = forced_rollout {
        tape.entries.clear();
        tape.set_enabled(false);
        log_opd_step_trace(total_started, "forced_rollout_start", "");
        validate_forced_rollout(forced_rollout, prompt_ids, cfg.rollout_len, vocab)?;
        store.retain_ids(rollout_keep_base);
        forced_rollout.to_vec()
    } else {
        let rollout_sampling = cfg.rollout_sampling.as_ref();
        // Infer-engine rollout: mirror the train LoRA into the
        // infer student once per step, then decode `rollout_len` tokens via
        // the infer engine. Produces the same `rollout: Vec<u32>` the
        // train-crate helper would, so downstream KL/backward is unchanged.
        // When the ctx is absent, the public train-crate helper runs as
        // the A/B baseline.
        #[cfg(feature = "cuda")]
        if let Some(ctx) = infer_rollout {
            log_opd_step_trace(total_started, "infer_rollout_reload_start", "");
            // OPD engine time-share: the rollout student may have been
            // offloaded to host RAM during the previous step's backward.
            // Reload it before the LoRA sync (which re-merges resident base
            // weights) and the rollout decode. Fence the train backend first
            // so the previous step's optimizer/cleanup pool ops are ordered
            // ahead of the reload's pool allocations (same cross-context
            // ordering reason as the teacher reload fence).
            if engine_offload.offloads_student() {
                store
                    .backend()
                    .device_synchronize()
                    .map_err(OpdError::from)?;
                ctx.student.reload_engine_weights().map_err(|err| {
                    OpdError::InvalidInput(format!("infer student reload failed: {err}"))
                })?;
            }
            log_opd_step_trace(total_started, "infer_rollout_sync_lora_start", "");
            ctx.student
                .sync_lora_from_store(store, &student.adapter_name_map(), ctx.lora_config)
                .map_err(|err| {
                    OpdError::InvalidInput(format!("infer student LoRA sync failed: {err}"))
                })?;
            log_opd_step_trace(total_started, "infer_rollout_generate_start", "");
            let rollout = ctx
                .student
                .generate_rollout(prompt_ids, cfg.rollout_len, rollout_sampling)
                .map_err(|err| {
                    OpdError::InvalidInput(format!("infer student rollout failed: {err}"))
                })?;
            log_opd_step_trace(
                total_started,
                "infer_rollout_generate_done",
                format!("actual_rollout_len={}", rollout.len()),
            );
            store.retain_ids(rollout_keep_base);
            // NB: the infer student engine is idle after the rollout, but we do
            // NOT offload it here. Offloading mid-step (before the teacher
            // forward) churns the shared device memory pool while the teacher
            // forward allocates from it, racing the async frees → illegal
            // address. Instead both idle engines are offloaded together AFTER
            // the teacher scores, just before the student backward (see
            // `backward_chunked_kl_rollout`), on a quiesced device.
            rollout
        } else {
            log_opd_step_trace(total_started, "train_rollout_start", "");
            student_rollout_only(
                student,
                prompt_ids,
                cfg.rollout_len,
                rollout_sampling,
                store,
                tape,
            )?
        }
        #[cfg(not(feature = "cuda"))]
        {
            log_opd_step_trace(total_started, "train_rollout_start", "");
            student_rollout_only(
                student,
                prompt_ids,
                cfg.rollout_len,
                rollout_sampling,
                store,
                tape,
            )?
        }
    };
    record_profile(profile, |profile| {
        profile.student_rollout_seconds += phase_started.elapsed().as_secs_f64();
    });
    log_opd_step_trace(
        total_started,
        "student_rollout_done",
        format!("actual_rollout_len={}", rollout.len()),
    );
    Ok(rollout)
}

/// Shared borrowed context for the two GKD backward routes. Holds the `&`/Copy
/// fields common to both; `&mut` store/tape/optimizer/profile stay explicit.
struct GkdRouteCtx<'a, T: TeacherForward + ?Sized> {
    student: &'a Qwen35Model,
    teacher: &'a T,
    prompt_ids: &'a [u32],
    cfg: &'a OpdStepConfig,
    gkd_config: GkdLossConfig<'a>,
    student_params: &'a [TensorId],
    student_model_params: &'a [TensorId],
    keep_extra: &'a HashSet<TensorId>,
    engine_offload: EngineOffloadMode,
    total_started: Instant,
    vocab: usize,
    rollout: &'a [u32],
    positions: &'a [u32],
    #[cfg(feature = "cuda")]
    infer_rollout: Option<&'a InferRolloutCtx<'a>>,
}

/// Route B — windowed scoring/backward: the memory path for long rollouts /
/// large vocabs. `kl_chunk_size` chunks softmax inside each window.
fn run_windowed_gkd_route<O: Optimizer, T: TeacherForward + ?Sized>(
    rt: &GkdRouteCtx<'_, T>,
    window_size: usize,
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    log_opd_step_trace(
        rt.total_started,
        "windowed_route_start",
        format!("window_size={window_size}"),
    );
    let phase_started = Instant::now();
    optimizer.zero_grad(store, rt.student_params);
    record_profile(profile, |profile| {
        profile.optimizer_zero_grad_seconds += phase_started.elapsed().as_secs_f64();
    });

    #[cfg(feature = "cuda")]
    let student_engine_offloaded = if rt.engine_offload.offloads_student() {
        if let Some(ctx) = rt.infer_rollout {
            log_opd_step_trace(rt.total_started, "infer_rollout_offload_start", "");
            store
                .backend()
                .device_synchronize()
                .map_err(OpdError::from)?;
            let freed = ctx.student.offload_engine_weights().map_err(|err| {
                OpdError::InvalidInput(format!("infer student offload failed: {err}"))
            })?;
            eprintln!(
                "opd_engine_offload student_offloaded freed_bytes={freed} freed_mib={:.1}",
                freed as f64 / (1024.0 * 1024.0)
            );
            true
        } else {
            false
        }
    } else {
        false
    };

    log_opd_step_trace(rt.total_started, "windowed_backward_start", "");
    let loss_result = backward_windowed_gkd_loss(
        rt.student,
        rt.teacher,
        rt.prompt_ids,
        rt.rollout,
        rt.positions,
        rt.vocab,
        rt.gkd_config,
        window_size,
        rt.student_params,
        rt.student_model_params,
        rt.keep_extra,
        store,
        tape,
        rt.engine_offload,
        profile,
    );
    log_opd_step_trace(rt.total_started, "windowed_backward_done", "");

    #[cfg(feature = "cuda")]
    if student_engine_offloaded {
        // Keep the rollout engine offloaded until the next rollout.
        // Reloading here happens before `cleanup_after_backward` prunes
        // the long-sequence autograd tape and can OOM on 2K-token OPD
        // steps. The next-step reload path above runs after cleanup and
        // before LoRA sync/generation, which is the first point where the
        // infer student must be resident again.
        log_opd_step_trace(
            rt.total_started,
            "infer_rollout_reload_deferred_until_next_step",
            "",
        );
    }

    let loss_value = loss_result?;

    let phase_started = Instant::now();
    finite_optimizer_step(
        loss_value,
        rt.student_params,
        rt.cfg.grad_clip,
        optimizer,
        store,
    )?;
    record_profile(profile, |profile| {
        profile.grad_clip_seconds += phase_started.elapsed().as_secs_f64();
    });
    log_opd_step_trace(rt.total_started, "optimizer_step_done", "");

    Ok(OpdStepOutcome {
        loss: loss_value,
        rollout_len: rt.rollout.len(),
    })
}

/// Route — pure chunked-KL (lambda == 0.0): full-prefix teacher + student
/// forward, chunked softmax in the backward, cuda-only offload/reload hooks
/// around teacher scoring.
fn run_chunked_kl_route<O: Optimizer, T: TeacherForward + ?Sized>(
    rt: &GkdRouteCtx<'_, T>,
    chunk_size: usize,
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    profile: &mut Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    let phase_started = Instant::now();
    optimizer.zero_grad(store, rt.student_params);
    record_profile(profile, |profile| {
        profile.optimizer_zero_grad_seconds += phase_started.elapsed().as_secs_f64();
    });

    // Build the rollout-student offload/reload hooks (cuda-only). The
    // offload is invoked inside the backward right after the teacher
    // scores (idle engines offloaded on a quiesced device); the reload
    // runs at the end of the backward so the student is resident again
    // for the inter-step eval / checkpoint / next rollout.
    #[cfg(feature = "cuda")]
    let student_offload_fn: Option<Box<dyn Fn() -> Result<usize>>> =
        if rt.engine_offload.offloads_student() {
            rt.infer_rollout.map(|ctx| {
                let student = ctx.student;
                Box::new(move || {
                    student.offload_engine_weights().map_err(|err| {
                        OpdError::InvalidInput(format!("infer student offload failed: {err}"))
                    })
                }) as Box<dyn Fn() -> Result<usize>>
            })
        } else {
            None
        };
    #[cfg(not(feature = "cuda"))]
    let student_offload_fn: Option<Box<dyn Fn() -> Result<usize>>> = None;

    #[cfg(feature = "cuda")]
    let student_reload_fn: Option<Box<dyn Fn() -> Result<()>>> =
        if rt.engine_offload.offloads_student() {
            rt.infer_rollout.map(|ctx| {
                let student = ctx.student;
                Box::new(move || {
                    student.reload_engine_weights().map_err(|err| {
                        OpdError::InvalidInput(format!(
                            "infer student reload (post-backward) failed: {err}"
                        ))
                    })
                }) as Box<dyn Fn() -> Result<()>>
            })
        } else {
            None
        };
    #[cfg(not(feature = "cuda"))]
    let student_reload_fn: Option<Box<dyn Fn() -> Result<()>>> = None;

    let loss_value = backward_chunked_kl_rollout(
        rt.student,
        rt.teacher,
        rt.rollout,
        rt.prompt_ids.len(),
        rt.vocab,
        chunk_size,
        rt.gkd_config.kl_mask,
        rt.gkd_config.kl_direction,
        rt.gkd_config.kl_temperature,
        rt.gkd_config.kl_beta,
        rt.student_model_params,
        rt.keep_extra,
        store,
        tape,
        profile,
        rt.engine_offload,
        student_offload_fn.as_deref(),
        student_reload_fn.as_deref(),
    )?;

    let phase_started = Instant::now();
    finite_optimizer_step(
        loss_value,
        rt.student_params,
        rt.cfg.grad_clip,
        optimizer,
        store,
    )?;
    record_profile(profile, |profile| {
        profile.grad_clip_seconds += phase_started.elapsed().as_secs_f64();
    });

    Ok(OpdStepOutcome {
        loss: loss_value,
        rollout_len: rt.rollout.len(),
    })
}

pub fn opd_step_with_teacher_forward_profiled_gkd_anchor<
    O: Optimizer,
    T: TeacherForward + ?Sized,
>(
    student: &Qwen35Model,
    teacher: &T,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
    gkd_config: GkdLossConfig<'_>,
    forced_rollout: Option<&[u32]>,
    #[cfg(feature = "cuda")] infer_rollout: Option<InferRolloutCtx<'_>>,
    profile: Option<&mut OpdStepProfile>,
) -> Result<OpdStepOutcome> {
    let mut profile = profile;
    if let Some(profile) = profile.as_deref_mut() {
        *profile = OpdStepProfile::default();
    }
    let total_started = Instant::now();
    log_opd_step_trace(
        total_started,
        "start",
        format!(
            "prompt_len={} rollout_len={}",
            prompt_ids.len(),
            cfg.rollout_len
        ),
    );
    validate_step_config(&cfg)?;
    validate_gkd_loss_config(gkd_config)?;
    if let Some(teacher_topk) = gkd_config.teacher_topk {
        return Err(OpdError::InvalidInput(format!(
            "OPD teacher_topk={teacher_topk} requires Piece A engine-side \
             top-k teacher targets on H20/CUDA. Local Piece C only wires the \
             config gate; omit --teacher-topk to keep the dense full-vocab path."
        )));
    }
    let vocab = student.config().vocab_size;
    if prompt_ids.is_empty() {
        return Err(OpdError::InvalidInput(
            "OPD step requires a non-empty prompt_ids slice. Hint: pass at least \
             one BOS/chat token; the OPD substrate does not synthesize prompts."
                .to_owned(),
        ));
    }
    if vocab == 0 {
        return Err(OpdError::InvalidInput(
            "OPD step requires student.config().vocab_size > 0. Hint: verify the \
             loaded Qwen35Config before constructing the student model."
                .to_owned(),
        ));
    }
    validate_rollout_shape(prompt_ids.len(), cfg.rollout_len, vocab)?;
    let teacher_vocab = teacher.vocab_size();
    if teacher_vocab != vocab {
        return Err(OpdError::InvalidInput(format!(
            "OPD requires teacher/student vocab_size to match, got \
             teacher.vocab_size()={teacher_vocab} and \
             student.config().vocab_size={vocab}. Hint: use model directories \
             that share the same tokenizer before running OPD."
        )));
    }
    validate_token_ids("prompt_ids", prompt_ids, vocab)?;
    let teacher_params = teacher.parameter_ids().to_vec();
    let student_model_params = student.all_parameter_ids();
    if !teacher_params.is_empty() {
        validate_teacher_params(&teacher_params, store)?;
    }
    validate_student_params(student_params, store)?;
    validate_student_param_ownership(student_params, &student_model_params, &teacher_params)?;
    let keep_extra = retained_param_and_grad_ids(&teacher_params, store);
    let mut rollout_keep_base = retained_param_and_grad_ids(&student_model_params, store);
    rollout_keep_base.extend(keep_extra.iter().copied());

    let result = (|| {
        #[cfg(feature = "cuda")]
        let engine_offload = engine_offload_mode();
        #[cfg(not(feature = "cuda"))]
        let engine_offload = EngineOffloadMode::Off;

        let rollout = run_opd_rollout_phase(
            student,
            prompt_ids,
            &cfg,
            vocab,
            forced_rollout,
            &rollout_keep_base,
            engine_offload,
            total_started,
            store,
            tape,
            &mut profile,
            #[cfg(feature = "cuda")]
            infer_rollout.as_ref(),
        )?;

        // Route B (windowed) is the memory path for long rollouts and large
        // vocabs; pure chunked-KL (lambda == 0.0) is next. Both early-return.
        let positions: Vec<u32> = (0..rollout.len() as u32).collect();
        let rt = GkdRouteCtx {
            student,
            teacher,
            prompt_ids,
            cfg: &cfg,
            gkd_config,
            student_params,
            student_model_params: &student_model_params,
            keep_extra: &keep_extra,
            engine_offload,
            total_started,
            vocab,
            rollout: &rollout,
            positions: &positions,
            #[cfg(feature = "cuda")]
            infer_rollout: infer_rollout.as_ref(),
        };
        if let Some(window_size) = gkd_config.logits_window_size {
            return run_windowed_gkd_route(&rt, window_size, optimizer, store, tape, &mut profile);
        }
        if let Some(chunk_size) = gkd_config.kl_chunk_size
            && gkd_config.lambda == 0.0
        {
            return run_chunked_kl_route(&rt, chunk_size, optimizer, store, tape, &mut profile);
        }

        // 3. Teacher forward — still tape-disabled. Teacher params carry
        //    `requires_grad = false` so no entries record even if tape was on,
        //    but disabling cheap-defends against any rogue grad-bearing weight.
        let phase_started = Instant::now();
        let teacher_logits = teacher
            .forward_logits_device(&rollout, &positions, store, tape)
            .map_err(|err| map_teacher_forward_error("teacher scoring", err))?;
        record_profile(&mut profile, |profile| {
            profile.teacher_forward_seconds += phase_started.elapsed().as_secs_f64();
        });
        let expected_teacher_shape = vec![1, rollout.len(), vocab];
        if teacher_logits.shape != expected_teacher_shape {
            return Err(OpdError::InvalidInput(format!(
                "OPD teacher logits shape mismatch: got {:?}, expected {:?}. \
                 Hint: the TeacherForward implementation must return \
                 [batch=1, seq_len, vocab] logits for the exact rollout \
                 scored by the student.",
                teacher_logits.shape, expected_teacher_shape
            )));
        }
        let mut keep_teacher_logits = rollout_keep_base.clone();
        keep_teacher_logits.insert(teacher_logits.tensor_id);
        store.retain_ids(&keep_teacher_logits);

        // 3. Student forward — tape enabled now so backward can flow.
        tape.set_enabled(true);
        let phase_started = Instant::now();
        let student_logits = student
            .forward(store, tape, &rollout, &positions)
            .map_err(|err| map_qwen35_forward_error("student KL", err))?;
        record_profile(&mut profile, |profile| {
            profile.student_forward_seconds += phase_started.elapsed().as_secs_f64();
        });

        // 4. KL distill loss, optionally blended with a GKD hard-token
        //    SFT anchor. The legacy anchor uses the on-policy rollout;
        //    corpus-truth mode re-forwards the student over prompt+target.
        let phase_started = Instant::now();
        let kl_range = kl_logit_range(gkd_config.kl_mask, prompt_ids.len(), rollout.len())?;
        let (student_kl_logits, teacher_kl_logits) =
            if kl_range.start == 0 && kl_range.end == rollout.len() {
                (student_logits, teacher_logits.tensor_id)
            } else {
                (
                    slice_logits_for_kl(student_logits, kl_range, vocab, store, tape)?,
                    slice_logits_for_kl(teacher_logits.tensor_id, kl_range, vocab, store, tape)?,
                )
            };
        let kl_loss = kl_distill_loss_for_config(
            student_kl_logits,
            teacher_kl_logits,
            kl_range.len(),
            gkd_config.kl_chunk_size,
            gkd_config.kl_direction,
            gkd_config.kl_temperature,
            gkd_config.kl_beta,
            store,
            tape,
        )?;
        let loss = if gkd_config.lambda == 0.0 {
            kl_loss
        } else {
            let sft_loss = gkd_sft_loss(
                gkd_config,
                student,
                prompt_ids,
                student_logits,
                &rollout,
                vocab,
                store,
                tape,
            )?;
            mix_gkd_losses(kl_loss, sft_loss, gkd_config.lambda, store, tape)?
        };
        let loss_value = store.to_host(loss)?[0];
        validate_loss_value(loss_value)?;
        record_profile(&mut profile, |profile| {
            profile.kl_loss_seconds += phase_started.elapsed().as_secs_f64();
        });

        // 5. Backward + grad clip + optimizer step.
        let phase_started = Instant::now();
        optimizer.zero_grad(store, student_params);
        record_profile(&mut profile, |profile| {
            profile.optimizer_zero_grad_seconds += phase_started.elapsed().as_secs_f64();
        });
        let phase_started = Instant::now();
        backward_with_optional_profile(loss, loss_value, store, tape)?;
        record_profile(&mut profile, |profile| {
            profile.backward_seconds += phase_started.elapsed().as_secs_f64();
        });
        let phase_started = Instant::now();
        finite_optimizer_step(loss_value, student_params, cfg.grad_clip, optimizer, store)?;
        record_profile(&mut profile, |profile| {
            profile.grad_clip_seconds += phase_started.elapsed().as_secs_f64();
        });

        Ok(OpdStepOutcome {
            loss: loss_value,
            rollout_len: rollout.len(),
        })
    })();

    // 6. Prune rollout/teacher/student forward temporaries on both success
    //    and failure. Teacher params live in `keep_extra`. Retain the full
    //    student model, not just the optimizer target slice, because LoRA-only
    //    OPD optimizes adapter ids while still needing frozen base weights for
    //    the next forward pass.
    let phase_started = Instant::now();
    cleanup_after_backward(store, tape, &student_model_params, &keep_extra);
    record_profile(&mut profile, |profile| {
        profile.post_step_cleanup_seconds += phase_started.elapsed().as_secs_f64();
        profile.total_seconds = total_started.elapsed().as_secs_f64();
    });
    result
}

#[cfg(test)]
mod tests {
    use autograd::{AutogradError, Tape, Tensor, TensorStore};

    use crate::loss::KlDirection;

    use super::{
        GkdLossConfig, GkdSftAnchor, KlLogitRange, OpdError, OpdKlMask, OpdStepConfig,
        build_masked_loss_targets, greedy_next_token, kl_distill_loss_for_config, kl_logit_range,
        map_qwen35_forward_error, mix_gkd_losses, next_token_sft_loss_from_logits,
        sequence_windows, shifted_rollout_sft_loss, skip_obs_gae, validate_gkd_lambda,
        validate_gkd_loss_config, validate_logits_shape, validate_loss_value,
        validate_rollout_shape, validate_step_config, validate_student_param_ownership,
        validate_student_params, validate_teacher_params,
    };
    use crate::qwen35::Qwen35Error;

    #[test]
    fn skip_obs_gae_cold_start_is_reward_to_go() {
        // V≈0 (zero-init critic) with γ=λ=1 and terminal reward 1 → every
        // masked position's advantage/return is the reward-to-go = 1.0.
        let (adv, ret) = skip_obs_gae(&[0.0; 4], 1.0, 1.0, 1.0);
        assert_eq!(adv, vec![1.0; 4]);
        assert_eq!(ret, vec![1.0; 4]);
        // A perfectly-fit critic (V = its own returns) yields advantage 0 → the
        // baseline cancels the reward, no policy push. Here V predicts the MC
        // return exactly, so δ telescopes to 0 at every step.
        let (adv0, _) = skip_obs_gae(&[1.0, 1.0, 1.0, 1.0], 1.0, 1.0, 1.0);
        assert!(
            adv0.iter().all(|a| a.abs() < 1e-6),
            "fit critic → adv≈0, got {adv0:?}"
        );
        // Discount γ=0.5, reward 1 at the last of 3 → adv = [0.25, 0.5, 1.0].
        let (adv1, _) = skip_obs_gae(&[0.0; 3], 1.0, 0.5, 1.0);
        assert_eq!(adv1, vec![0.25, 0.5, 1.0]);
    }

    #[test]
    fn greedy_next_token_rejects_logits_len_mismatch() {
        let mut store = TensorStore::default();
        let logits =
            store.alloc(Tensor::new(vec![0.0; 8], vec![1, 2, 4], false).expect("logits tensor"));

        let err = greedy_next_token(logits, 1, 4, &mut store, None, 0)
            .expect_err("extra logits rows must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("logits length mismatch"));
        assert!(message.contains("expected exactly"));
        assert!(message.contains("1 * 4"));
    }

    #[test]
    fn greedy_next_token_rejects_non_finite_logits() {
        let mut store = TensorStore::default();
        let logits = store.alloc(
            Tensor::new(vec![0.0, 0.5, f32::NAN, 0.25], vec![1, 1, 4], false)
                .expect("logits tensor"),
        );

        let err = greedy_next_token(logits, 1, 4, &mut store, None, 0)
            .expect_err("NaN logits must not silently sample token 0");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("non-finite"));
        assert!(message.contains("vocab index 2"));
        assert!(message.contains("student forward numerics"));
    }

    #[test]
    fn validate_student_params_rejects_empty_list() {
        let store = TensorStore::default();
        let err = validate_student_params(&[], &store)
            .expect_err("empty student parameter list must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("at least one student parameter id"));
        assert!(message.contains("optimizer step a no-op"));
    }

    #[test]
    fn validate_student_params_rejects_missing_tensor_id() {
        let store = TensorStore::default();
        let err = validate_student_params(&[17], &store)
            .expect_err("missing TensorStore id must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("student_params[0]=17"));
        assert!(message.contains("same student Qwen35Model"));
    }

    #[test]
    fn validate_student_params_rejects_duplicate_tensor_id() {
        let mut store = TensorStore::default();
        let trainable =
            store.alloc(Tensor::new(vec![0.0; 4], vec![2, 2], true).expect("trainable tensor"));

        let err = validate_student_params(&[trainable, trainable], &store)
            .expect_err("duplicate student parameter ids must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("student_params[1]"));
        assert!(message.contains("duplicates"));
        assert!(message.contains("exactly once"));
        assert!(message.contains("optimizer updates more than once"));
    }

    #[test]
    fn validate_student_params_rejects_frozen_only_params() {
        let mut store = TensorStore::default();
        let frozen =
            store.alloc(Tensor::new(vec![0.0; 4], vec![2, 2], false).expect("frozen tensor"));

        let err = validate_student_params(&[frozen], &store)
            .expect_err("frozen-only parameter list must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("no trainable tensors"));
        assert!(message.contains("requires_grad=true"));
        assert!(message.contains("optimizer step a no-op"));
    }

    #[test]
    fn validate_student_param_ownership_rejects_teacher_param_ids() {
        let student_param = 10;
        let teacher_param = 20;
        let err = validate_student_param_ownership(
            &[student_param, teacher_param],
            &[student_param],
            &[teacher_param],
        )
        .expect_err("teacher ids must not be accepted as student params");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("student_params[1]=20"));
        assert!(message.contains("frozen teacher"));
        assert!(message.contains("must not be optimized"));
    }

    #[test]
    fn validate_student_param_ownership_rejects_non_student_param_ids() {
        let student_param = 10;
        let foreign_param = 30;
        let err = validate_student_param_ownership(&[foreign_param], &[student_param], &[])
            .expect_err("foreign ids must not be accepted as student params");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("student_params[0]=30"));
        assert!(message.contains("not owned by the student"));
        assert!(message.contains("same TensorStore"));
    }

    #[test]
    fn validate_teacher_params_rejects_trainable_params() {
        let mut store = TensorStore::default();
        let trainable_teacher =
            store.alloc(Tensor::new(vec![0.0; 4], vec![2, 2], true).expect("teacher tensor"));

        let err = validate_teacher_params(&[trainable_teacher], &store)
            .expect_err("trainable teacher parameters must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("teacher_params[0]"));
        assert!(message.contains("requires_grad=true"));
        assert!(message.contains("new_for_eval"));
        assert!(message.contains("must not optimize teacher weights"));
    }

    #[test]
    fn validate_loss_value_rejects_non_finite_loss() {
        for loss_value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_loss_value(loss_value)
                .expect_err("non-finite OPD loss must be rejected before backward");

            let OpdError::InvalidInput(message) = err else {
                panic!("expected InvalidInput, got {err:?}");
            };
            assert!(message.contains("non-finite"));
            assert!(message.contains("teacher/student logits"));
            assert!(message.contains("learning rate"));
        }
    }

    #[test]
    fn validate_step_config_rejects_non_finite_grad_clip() {
        for grad_clip in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_step_config(&OpdStepConfig {
                rollout_len: 1,
                rollout_sampling: None,
                grad_clip,
            })
            .expect_err("non-finite OPD grad_clip must not silently disable clipping");

            let OpdError::InvalidInput(message) = err else {
                panic!("expected InvalidInput, got {err:?}");
            };
            assert!(message.contains("cfg.grad_clip"));
            assert!(message.contains("finite"));
            assert!(message.contains("0.0"));
        }
    }

    #[test]
    fn validate_step_config_rejects_negative_grad_clip() {
        let err = validate_step_config(&OpdStepConfig {
            rollout_len: 1,
            rollout_sampling: None,
            grad_clip: -1.0,
        })
        .expect_err("negative OPD grad_clip must not silently disable clipping");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("cfg.grad_clip"));
        assert!(message.contains("non-negative"));
        assert!(message.contains("0.0"));
    }

    #[test]
    fn validate_gkd_lambda_rejects_out_of_range_values() {
        for gkd_lambda in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            let err =
                validate_gkd_lambda(gkd_lambda).expect_err("invalid GKD lambda must be rejected");

            let OpdError::InvalidInput(message) = err else {
                panic!("expected InvalidInput, got {err:?}");
            };
            assert!(message.contains("GKD lambda"));
            assert!(message.contains("[0.0, 1.0]"));
        }
    }

    #[test]
    fn validate_gkd_loss_config_requires_corpus_tokens_for_corpus_anchor() {
        let missing = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.3,
            sft_anchor: GkdSftAnchor::CorpusTruth,
            corpus_tokens: None,
            kl_chunk_size: None,
            kl_direction: KlDirection::Forward,
            kl_temperature: 1.0,
            kl_beta: None,
            teacher_topk: None,
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::Full,
        })
        .expect_err("corpus anchor must require target tokens");
        let OpdError::InvalidInput(message) = missing else {
            panic!("expected InvalidInput, got {missing:?}");
        };
        assert!(message.contains("corpus-truth"));
        assert!(message.contains("completion"));

        let empty = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.3,
            sft_anchor: GkdSftAnchor::CorpusTruth,
            corpus_tokens: Some(&[]),
            kl_chunk_size: None,
            kl_direction: KlDirection::Forward,
            kl_temperature: 1.0,
            kl_beta: None,
            teacher_topk: None,
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::Full,
        })
        .expect_err("empty corpus anchor tokens must be rejected");
        let OpdError::InvalidInput(message) = empty else {
            panic!("expected InvalidInput, got {empty:?}");
        };
        assert!(message.contains("non-empty"));

        validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::CorpusTruth,
            corpus_tokens: None,
            kl_chunk_size: None,
            kl_direction: KlDirection::Forward,
            kl_temperature: 1.0,
            kl_beta: None,
            teacher_topk: None,
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::Full,
        })
        .expect("lambda=0 should not require unused corpus targets");
        validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.3,
            sft_anchor: GkdSftAnchor::CorpusTruth,
            corpus_tokens: Some(&[1, 2]),
            kl_chunk_size: None,
            kl_direction: KlDirection::Forward,
            kl_temperature: 1.0,
            kl_beta: None,
            teacher_topk: None,
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::Full,
        })
        .expect("non-empty corpus targets should be accepted");
    }

    #[test]
    fn validate_gkd_loss_config_rejects_zero_kl_chunk_size() {
        let err = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: Some(0),
            kl_direction: KlDirection::Forward,
            kl_temperature: 1.0,
            kl_beta: None,
            teacher_topk: None,
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::Full,
        })
        .expect_err("zero KL chunk size must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("KL chunk size"));
        assert!(message.contains("> 0"));
        assert!(message.contains("--kl-chunk-size"));
    }

    #[test]
    fn validate_gkd_loss_config_gates_teacher_topk_shape_and_objective() {
        validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: None,
            kl_direction: KlDirection::Forward,
            kl_temperature: 1.0,
            kl_beta: None,
            teacher_topk: Some(64),
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::CompletionOnly,
        })
        .expect("positive teacher top-k should be valid for sparse forward KL");

        let zero = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: None,
            kl_direction: KlDirection::Forward,
            kl_temperature: 1.0,
            kl_beta: None,
            teacher_topk: Some(0),
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::CompletionOnly,
        })
        .expect_err("zero teacher top-k must be rejected");
        assert!(format!("{zero}").contains("teacher top-k"));

        let reverse = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: None,
            kl_direction: KlDirection::Reverse,
            kl_temperature: 1.0,
            kl_beta: None,
            teacher_topk: Some(64),
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::CompletionOnly,
        })
        .expect_err("sparse teacher top-k must reject reverse KL");
        assert!(format!("{reverse}").contains("forward-KL only"));

        let beta = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: None,
            kl_direction: KlDirection::Forward,
            kl_temperature: 1.0,
            kl_beta: Some(0.5),
            teacher_topk: Some(64),
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::CompletionOnly,
        })
        .expect_err("sparse teacher top-k must reject generalized JSD");
        assert!(format!("{beta}").contains("JSD beta"));
    }

    #[test]
    fn validate_gkd_loss_config_rejects_temperature_with_sft_blend() {
        let err = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.5,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: None,
            kl_direction: KlDirection::Forward,
            kl_temperature: 2.0,
            kl_beta: None,
            teacher_topk: None,
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::CompletionOnly,
        })
        .expect_err("KL temperature must not silently reweight GKD SFT blend");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("pure-OPD"));
        assert!(message.contains("1/vocab"));
        assert!(message.contains("T^2"));
        assert!(message.contains("--gkd-lambda 0.0"));

        validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: None,
            kl_direction: KlDirection::Forward,
            kl_temperature: 2.0,
            kl_beta: None,
            teacher_topk: None,
            fused_distill: true,
            logits_window_size: None,
            kl_mask: OpdKlMask::CompletionOnly,
        })
        .expect("KL temperature should be accepted for pure OPD");
    }

    #[test]
    fn validate_gkd_loss_config_rejects_zero_logits_window_size() {
        let err = validate_gkd_loss_config(GkdLossConfig {
            lambda: 0.0,
            sft_anchor: GkdSftAnchor::StudentRollout,
            corpus_tokens: None,
            kl_chunk_size: None,
            kl_direction: KlDirection::Forward,
            kl_temperature: 1.0,
            kl_beta: None,
            teacher_topk: None,
            fused_distill: true,
            logits_window_size: Some(0),
            kl_mask: OpdKlMask::Full,
        })
        .expect_err("zero logits window size must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("logits window size"));
        assert!(message.contains("> 0"));
        assert!(message.contains("--logits-window-size"));
    }

    #[test]
    fn sequence_windows_cover_tail_without_overlap() {
        let windows = sequence_windows(10, 4).expect("windows");
        let spans = windows
            .into_iter()
            .map(|window| (window.start, window.end))
            .collect::<Vec<_>>();
        assert_eq!(spans, vec![(0, 4), (4, 8), (8, 10)]);
    }

    #[test]
    fn kl_logit_range_completion_only_slices_completion_targets() {
        let range = kl_logit_range(OpdKlMask::CompletionOnly, 16, 144).expect("range");
        assert_eq!(
            range,
            KlLogitRange {
                start: 15,
                end: 143
            }
        );
        assert_eq!(range.len(), 128);
    }

    #[test]
    fn kl_logit_range_full_preserves_legacy_positions() {
        let range = kl_logit_range(OpdKlMask::Full, 16, 144).expect("range");
        assert_eq!(range, KlLogitRange { start: 0, end: 144 });
        assert_eq!(range.len(), 144);
    }

    #[test]
    fn kl_distill_loss_for_config_accepts_chunked_path() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let student = store.alloc(
            Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 2, 2], true).expect("student logits"),
        );
        let teacher = store.alloc(
            Tensor::new(vec![0.4, 0.3, 0.2, 0.1], vec![1, 2, 2], false).expect("teacher logits"),
        );

        let loss = kl_distill_loss_for_config(
            student,
            teacher,
            2,
            Some(1),
            KlDirection::Forward,
            1.0,
            None,
            &mut store,
            &mut tape,
        )
        .expect("chunked KL config should run");
        let value = store.to_host(loss).expect("loss host")[0];
        assert!(value.is_finite(), "chunked KL loss must be finite");
    }

    #[test]
    fn validate_logits_shape_rejects_last_row_teacher_logits_for_full_kl() {
        let err = validate_logits_shape("teacher KL", &[1, 1, 8], 4, 8)
            .expect_err("last-row teacher logits must not score a 4-token KL window");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("OPD teacher KL logits shape mismatch"));
        assert!(message.contains("expected [1, 4, 8]"));
        assert!(message.contains("current logits window"));
    }

    #[test]
    fn mix_gkd_losses_respects_lambda_endpoints_and_weighted_blend() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let kl = store.alloc(Tensor::new(vec![2.0], vec![1], true).expect("kl scalar"));
        let sft = store.alloc(Tensor::new(vec![10.0], vec![1], true).expect("sft scalar"));

        let pure_kl = mix_gkd_losses(kl, sft, 0.0, &mut store, &mut tape)
            .expect("lambda=0 should produce pure KL");
        assert_eq!(store.to_host(pure_kl).expect("pure kl host")[0], 2.0);

        let pure_sft = mix_gkd_losses(kl, sft, 1.0, &mut store, &mut tape)
            .expect("lambda=1 should produce pure SFT");
        assert_eq!(store.to_host(pure_sft).expect("pure sft host")[0], 10.0);

        let midpoint = mix_gkd_losses(kl, sft, 0.5, &mut store, &mut tape)
            .expect("lambda=0.5 should mix losses evenly");
        let value = store.to_host(midpoint).expect("midpoint host")[0];
        assert!((value - 6.0).abs() < 1.0e-6, "got {value}");

        let lambda03 = mix_gkd_losses(kl, sft, 0.3, &mut store, &mut tape)
            .expect("lambda=0.3 should mix losses with SFT weight 0.3");
        let value = store.to_host(lambda03).expect("lambda03 host")[0];
        assert!((value - 4.4).abs() < 1.0e-6, "got {value}");
    }

    #[test]
    fn shifted_rollout_sft_loss_is_per_position_ce() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let logits = store.alloc(
            Tensor::new(
                vec![
                    0.0, 1.0, 2.0, 3.0, // position 0, target 2
                    4.0, 3.0, 2.0, 1.0, // position 1, target 1
                    0.5, 0.25, 0.0, -0.25, // position 2 ignored
                ],
                vec![1, 3, 4],
                true,
            )
            .expect("student logits"),
        );

        let loss = shifted_rollout_sft_loss(logits, &[0, 2, 1], 4, &mut store, &mut tape)
            .expect("shifted sft loss");
        let value = store.to_host(loss).expect("loss host")[0];

        // Per-position CE (mean over positions 0..1, NO `1/vocab`): now matches
        // the `batchmean` KL scale so `--gkd-lambda` blends KL and CE at face
        // value.
        let ce0 = (0.0f32.exp() + 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln() - 2.0;
        let ce1 = (4.0f32.exp() + 3.0f32.exp() + 2.0f32.exp() + 1.0f32.exp()).ln() - 3.0;
        let expected = (ce0 + ce1) * 0.5;
        assert!(
            (value - expected).abs() < 1.0e-6,
            "got {value}, expected {expected}"
        );
    }

    #[test]
    fn corpus_target_sft_loss_uses_completion_logits_after_prompt() {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let logits = store.alloc(
            Tensor::new(
                vec![
                    0.0, 0.0, 0.0, 0.0, // prompt position 0 ignored
                    0.0, 1.0, 2.0, 3.0, // prompt final position -> target 3
                    4.0, 3.0, 2.0, 1.0, // completion prefix -> target 1
                    0.5, 0.25, 0.0, -0.25, // final position ignored
                ],
                vec![1, 4, 4],
                true,
            )
            .expect("student logits"),
        );

        let loss = next_token_sft_loss_from_logits(logits, 4, 1, &[3, 1], 4, &mut store, &mut tape)
            .expect("corpus sft loss");
        let value = store.to_host(loss).expect("loss host")[0];

        let ce0 = (0.0f32.exp() + 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln() - 3.0;
        let ce1 = (4.0f32.exp() + 3.0f32.exp() + 2.0f32.exp() + 1.0f32.exp()).ln() - 3.0;
        // Per-position CE (mean over 2 positions, NO `1/vocab`) — see
        // `shifted_rollout_sft_loss_is_per_position_ce`.
        let expected = (ce0 + ce1) * 0.5;
        assert!(
            (value - expected).abs() < 1.0e-6,
            "got {value}, expected {expected}"
        );
    }

    #[test]
    fn map_qwen35_forward_error_wraps_autograd_errors_with_opd_context() {
        let err = map_qwen35_forward_error(
            "student KL",
            Qwen35Error::Autograd(AutogradError::ShapeMismatch {
                expected: vec![1, 2, 3],
                got: vec![1, 2],
            }),
        );

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("OPD student KL"));
        assert!(message.contains("autograd error"));
        assert!(message.contains("checkpoint tensor shapes"));
        assert!(message.contains("OPD loader/model follow-up"));
    }

    #[test]
    fn validate_rollout_shape_rejects_total_len_overflow() {
        let err = validate_rollout_shape(2, usize::MAX, 16)
            .expect_err("rollout length overflow must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("rollout length overflow"));
        assert!(message.contains("prompt_len=2"));
        assert!(message.contains("rollout-len"));
    }

    #[test]
    fn validate_rollout_shape_rejects_u32_position_overflow() {
        let err = validate_rollout_shape(u32::MAX as usize, 1, 16)
            .expect_err("position id overflow must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("exceeds u32::MAX position ids"));
        assert!(message.contains("u32 position ids"));
    }

    #[test]
    fn validate_rollout_shape_rejects_u32_vocab_overflow() {
        let err = validate_rollout_shape(1, 1, u32::MAX as usize + 1)
            .expect_err("token id overflow must be rejected");

        let OpdError::InvalidInput(message) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("vocab_size"));
        assert!(message.contains("u32 token ids"));
    }

    #[test]
    fn masked_loss_targets_select_only_response_llm_positions() {
        // prompt = [100, 101] (prompt_len = 2); response = [10, 11, 20, 21, 30]
        // with mask [1, 1, 0, 0, 1] (two tool tokens at response idx 2,3).
        // full = [100, 101, 10, 11, 20, 21, 30], seq_len = 7.
        //
        // Predicting position p targets full[p + 1]; it counts iff
        // target_pos = p + 1 is a response position with mask == 1:
        //   p=1 -> target full[2]=10  (resp idx 0, mask 1)  ✓
        //   p=2 -> target full[3]=11  (resp idx 1, mask 1)  ✓
        //   p=3 -> target full[4]=20  (resp idx 2, mask 0)  ✗ tool
        //   p=4 -> target full[5]=21  (resp idx 3, mask 0)  ✗ tool
        //   p=5 -> target full[6]=30  (resp idx 4, mask 1)  ✓
        let full = [100u32, 101, 10, 11, 20, 21, 30];
        let prompt_len = 2;
        let mask = [1u8, 1, 0, 0, 1];
        let targets = build_masked_loss_targets(&full, prompt_len, &mask);
        assert_eq!(
            targets,
            vec![(1usize, 10usize), (2, 11), (5, 30)],
            "loss targets must be the LLM-generated response positions only \
             (tool tokens at p=3,4 excluded); position p predicts token p+1"
        );
    }

    #[test]
    fn masked_loss_targets_empty_when_no_llm_tokens() {
        // All-tool response yields no targets (nothing to train).
        let full = [1u32, 2, 5, 6];
        assert!(build_masked_loss_targets(&full, 2, &[0u8, 0]).is_empty());
        // Empty / single-token sequences yield no targets.
        assert!(build_masked_loss_targets(&[1u32], 1, &[]).is_empty());
        assert!(build_masked_loss_targets(&[], 0, &[]).is_empty());
    }

    #[test]
    fn masked_loss_targets_clamp_ragged_mask() {
        // mask shorter than the response: positions past the mask are excluded
        // (resp_idx >= mask.len() -> skipped), never panicking.
        // full = [1, 7, 8, 9], prompt_len = 1; mask = [1, 1] covers resp idx 0,1.
        //   p=0 -> target full[1]=7 (resp idx 0, mask 1) ✓
        //   p=1 -> target full[2]=8 (resp idx 1, mask 1) ✓
        //   p=2 -> target full[3]=9 (resp idx 2, beyond mask) ✗
        let full = [1u32, 7, 8, 9];
        let targets = build_masked_loss_targets(&full, 1, &[1u8, 1]);
        assert_eq!(targets, vec![(0usize, 7usize), (1, 8)]);
    }

    // The pod showed CP loss-sum 5.5% off the f32 single-card loss while the CP
    // HIDDEN tracks f32 — isolating the bug to the masked-CE aggregation over the
    // zigzag shards, which is pure host arithmetic. This reproduces exactly that
    // aggregation on CPU (no ring, no GPU): given a FIXED hidden, the single-card
    // CE over all targets must equal the sum of per-zigzag-shard CE with
    // inv_n=1/global. The shard gathers its rows (`local_rows`), so the target at
    // global position p uses shard-local row `local_of(p)` of the gathered hidden —
    // the same remap opd.rs does. A wrong remap / inv_n / double-count breaks this.
    #[test]
    fn cp_sharded_ce_sum_equals_single_card_ce() {
        use crate::context_parallel::CpContext;
        use autograd::ops::fused_linear_distill::fused_linear_ce_loss_indexed;

        let (seq, hidden_dim, vocab) = (16usize, 8usize, 16usize);
        let cp_size = 2;
        // Deterministic hidden [1, seq, hidden] and lm_head [vocab, hidden].
        let synth = |n: usize, seed: u64| -> Vec<f32> {
            let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                })
                .collect()
        };
        let hidden_data = synth(seq * hidden_dim, 7);
        let weight_data = synth(vocab * hidden_dim, 9);

        // Targets: predict positions 3..=14 (prompt_len 4), token cycling in-vocab.
        let targets: Vec<(usize, usize)> =
            (3..=seq - 2).map(|p| (p, (p * 3 + 1) % vocab)).collect();
        let global_targets = targets.len();
        let inv_n = 1.0f32 / global_targets as f32;

        let ce = |store: &mut TensorStore,
                  hidden: &[f32],
                  rows: usize,
                  pos: &[i32],
                  tgt: &[i32],
                  inv: Option<f32>|
         -> f32 {
            let h = store
                .alloc(Tensor::new(hidden.to_vec(), vec![1, rows, hidden_dim], false).unwrap());
            let w = store
                .alloc(Tensor::new(weight_data.clone(), vec![vocab, hidden_dim], false).unwrap());
            let mut tape = Tape::new();
            tape.set_enabled(false);
            let loss =
                fused_linear_ce_loss_indexed(h, w, pos, tgt, 64, inv, store, &mut tape).unwrap();
            store.to_host(loss).unwrap()[0]
        };

        // Single card: whole sequence, all targets, inv_n = 1/global (the default
        // 1/num_targets == 1/global here since num_targets == global).
        let mut store = TensorStore::default();
        let (pos_all, tgt_all): (Vec<i32>, Vec<i32>) =
            targets.iter().map(|&(p, t)| (p as i32, t as i32)).unzip();
        let loss_single = ce(&mut store, &hidden_data, seq, &pos_all, &tgt_all, None);

        // CP: each zigzag shard gathers its rows, remaps targets via local_of,
        // inv_n = 1/global; sum over shards must equal the single-card loss.
        let mut loss_cp = 0.0f32;
        for rank in 0..cp_size {
            let shard = CpContext::new(rank, cp_size).shard(seq);
            let rows = shard.local_rows();
            let mut shard_hidden = Vec::with_capacity(rows.len() * hidden_dim);
            for &r in &rows {
                shard_hidden.extend_from_slice(&hidden_data[r * hidden_dim..(r + 1) * hidden_dim]);
            }
            let (pos, tgt): (Vec<i32>, Vec<i32>) = targets
                .iter()
                .filter_map(|&(p, t)| shard.local_of(p).map(|l| (l as i32, t as i32)))
                .unzip();
            if pos.is_empty() {
                continue;
            }
            let mut s = TensorStore::default();
            loss_cp += ce(&mut s, &shard_hidden, rows.len(), &pos, &tgt, Some(inv_n));
        }

        let rel = (loss_single - loss_cp).abs() / loss_single.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "CP-sharded CE sum {loss_cp} != single-card CE {loss_single} (rel {rel}) — \
             masked-CE aggregation over zigzag shards is wrong"
        );
    }
}
