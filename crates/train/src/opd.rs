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

use std::{
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use autograd::{AutogradError, BackwardOp, BackwardProfile, Tape, TensorId, TensorStore};
use infer_plan::SamplingParams;

use crate::{
    grad_clip::FiniteStepError,
    loss::{DEFAULT_KL_CHUNK_SIZE, KlDirection},
    qwen35::Qwen35Error,
    teacher_infer::TeacherForwardError,
};
#[cfg(feature = "cuda")]
use crate::{infer_student::InferStudent, lora::LoraConfig};

#[path = "opd/backward.rs"]
mod backward;
#[path = "opd/critic.rs"]
mod critic;
#[path = "opd/loss.rs"]
mod loss;
#[path = "opd/rollout.rs"]
mod rollout;
#[path = "opd/step.rs"]
mod step;
#[path = "opd/validation.rs"]
mod validation;
#[path = "opd/windowing.rs"]
mod windowing;
#[path = "opd/writeback.rs"]
mod writeback;

pub use critic::{ValueCritic, skip_obs_gae};
pub use rollout::student_rollout_only;
pub use step::{OpdStepInputs, opd_step, opd_step_with_teacher};
pub use writeback::{
    WritebackLoss, capture_rollout_logprobs, fmt_hoarded, full_batch_ce_writeback_step,
    masked_gkd_writeback_step, masked_writeback_step,
};

/// Routes the OPD rollout through the in-process infer engine (`InferStudent`)
/// instead of the train-crate decode. `None` → train-crate rollout (A/B baseline).
#[cfg(feature = "cuda")]
pub struct InferRolloutCtx<'a> {
    pub student: &'a InferStudent,
    pub lora_config: LoraConfig,
}

/// True when the infer-engine rollout path is selected (default). The in-process
/// infer student (CUDA graph + paged KV) is 4.99× faster than train-crate O(n²)
/// decode (`wins/2026-05-29-opd-infer-rollout-default-p4.md`).
#[cfg(feature = "cuda")]
pub fn infer_rollout_flag_enabled() -> bool {
    INFER_ROLLOUT_OVERRIDE.get().copied().unwrap_or(true)
}

#[cfg(feature = "cuda")]
static INFER_ROLLOUT_OVERRIDE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Install the `--rollout-engine` selection (idempotent, first write wins).
#[cfg(feature = "cuda")]
pub fn set_infer_rollout_override(use_infer: bool) {
    let _ = INFER_ROLLOUT_OVERRIDE.set(use_infer);
}

/// OPD engine weight time-share mode (`--engine-offload`). Offloads idle infer
/// engines to host RAM during student backward, freeing VRAM for long rollouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineOffloadMode {
    Off,
    /// Offload BOTH engines after teacher scores. Frees ~4.4 GB at rollout-256
    /// but reloads the teacher every step, which races the shared async pool
    /// across three co-resident CUDA contexts (step-2 illegal address — see
    /// `errors/2026-05-30-...`).
    All,
    /// Offload ONLY the rollout infer-student; keep teacher resident. Avoids
    /// the step-2 teacher-reload path. Frees ~1.4 GB.
    Student,
    /// Offload ONLY the scoring teacher; keep student resident. Frees ~3.0 GB
    /// and avoids the student↔teacher offload interleaving that corrupts the
    /// W4A8 Marlin reload under `All`.
    Teacher,
}

impl EngineOffloadMode {
    pub fn is_enabled(self) -> bool {
        !matches!(self, EngineOffloadMode::Off)
    }

    pub fn offloads_teacher(self) -> bool {
        matches!(self, EngineOffloadMode::All | EngineOffloadMode::Teacher)
    }

    pub fn offloads_student(self) -> bool {
        matches!(self, EngineOffloadMode::All | EngineOffloadMode::Student)
    }
}

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
    pub rollout_len: usize,
    /// `None` keeps the existing greedy argmax path.
    pub rollout_sampling: Option<SamplingParams>,
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

/// Default Route-B logits window. Aligned with the KL chunk size so each
/// backward materializes at most one `[window, vocab]` logits tile.
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
            // Dense logits+KL by default: fused lm_head+loss ran lm_head on
            // HOST (~205 s/step for 27B, GPU idle) vs ~3.9 s GPU-bound dense.
            // Opt into fused only for windows too large to materialize
            // [window, vocab]. errors/2026-06-23-opd-fused-distill-default-host-bound.
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

fn step_trace_enabled() -> bool {
    match std::env::var("ARLE_OPD_STEP_TRACE") {
        Ok(value) => !(value == "0" || value.eq_ignore_ascii_case("false")),
        Err(_) => false,
    }
}

fn log_opd_step_trace(step_started: Instant, event: &str, detail: impl AsRef<str>) {
    if step_trace_enabled() {
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
    if step_trace_enabled() {
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
