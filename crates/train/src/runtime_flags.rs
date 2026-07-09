//! OPD runtime toggles: `arle train … --flag` → [`apply_runtime_flags`] once
//! at CLI start (also forwards the autograd knobs). The statics are the single
//! truth — no env reads.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering::Relaxed};

use crate::opd::EngineOffloadMode;

/// Train-side knobs the OPD CLI flags control (defaults = shipped behavior).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainRuntimeFlags {
    /// Offload per-layer grad-checkpoints to host RAM during the writeback
    /// forward (`--writeback-offload`; default on — long trajectories need it).
    pub writeback_offload: bool,
    /// Idle-engine offload time-share (`--engine-offload off|all|student|teacher`).
    pub engine_offload: EngineOffloadMode,
    /// Per-layer gradient checkpointing at student load (`--gradient-checkpointing`).
    pub gradient_checkpointing: bool,
    /// Trim the device pool before backward (`--trim-before-backward`).
    pub trim_before_backward: bool,
    /// Trim the device pool after writeback (`--trim-after-writeback`).
    pub trim_after_writeback: bool,
    /// Frozen-prompt-KV writeback path (`--writeback-frozen-prompt-kv`).
    pub writeback_frozen_prompt_kv: bool,
    /// Rollout tensor retain interval (`--rollout-retain-interval`).
    pub rollout_retain_interval: usize,
    /// Rollout progress log interval (`--rollout-progress-interval`).
    pub rollout_progress_interval: usize,
    /// Autograd-crate knobs, forwarded to `autograd::apply_runtime_flags`.
    pub autograd: autograd::AutogradRuntimeFlags,
}

impl Default for TrainRuntimeFlags {
    fn default() -> Self {
        Self {
            writeback_offload: true,
            engine_offload: EngineOffloadMode::Off,
            gradient_checkpointing: false,
            trim_before_backward: false,
            trim_after_writeback: false,
            writeback_frozen_prompt_kv: false,
            rollout_retain_interval: 2,
            rollout_progress_interval: 16,
            autograd: autograd::AutogradRuntimeFlags::default(),
        }
    }
}

static WRITEBACK_OFFLOAD: AtomicBool = AtomicBool::new(true);
static ENGINE_OFFLOAD: AtomicU8 = AtomicU8::new(0);
static GRADIENT_CHECKPOINTING: AtomicBool = AtomicBool::new(false);
static TRIM_BEFORE_BACKWARD: AtomicBool = AtomicBool::new(false);
static TRIM_AFTER_WRITEBACK: AtomicBool = AtomicBool::new(false);
static WRITEBACK_FROZEN_PROMPT_KV: AtomicBool = AtomicBool::new(false);
static ROLLOUT_RETAIN_INTERVAL: AtomicUsize = AtomicUsize::new(2);
static ROLLOUT_PROGRESS_INTERVAL: AtomicUsize = AtomicUsize::new(16);

pub fn apply_runtime_flags(f: &TrainRuntimeFlags) {
    WRITEBACK_OFFLOAD.store(f.writeback_offload, Relaxed);
    ENGINE_OFFLOAD.store(f.engine_offload as u8, Relaxed);
    GRADIENT_CHECKPOINTING.store(f.gradient_checkpointing, Relaxed);
    TRIM_BEFORE_BACKWARD.store(f.trim_before_backward, Relaxed);
    TRIM_AFTER_WRITEBACK.store(f.trim_after_writeback, Relaxed);
    WRITEBACK_FROZEN_PROMPT_KV.store(f.writeback_frozen_prompt_kv, Relaxed);
    ROLLOUT_RETAIN_INTERVAL.store(f.rollout_retain_interval.max(1), Relaxed);
    ROLLOUT_PROGRESS_INTERVAL.store(f.rollout_progress_interval.max(1), Relaxed);
    autograd::apply_runtime_flags(&f.autograd);
}

pub(crate) fn writeback_offload() -> bool {
    WRITEBACK_OFFLOAD.load(Relaxed)
}
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn engine_offload() -> EngineOffloadMode {
    match ENGINE_OFFLOAD.load(Relaxed) {
        1 => EngineOffloadMode::All,
        2 => EngineOffloadMode::Student,
        3 => EngineOffloadMode::Teacher,
        _ => EngineOffloadMode::Off,
    }
}
pub(crate) fn gradient_checkpointing() -> bool {
    GRADIENT_CHECKPOINTING.load(Relaxed)
}
pub(crate) fn trim_before_backward() -> bool {
    TRIM_BEFORE_BACKWARD.load(Relaxed)
}
pub(crate) fn trim_after_writeback() -> bool {
    TRIM_AFTER_WRITEBACK.load(Relaxed)
}
pub(crate) fn writeback_frozen_prompt_kv() -> bool {
    WRITEBACK_FROZEN_PROMPT_KV.load(Relaxed)
}
pub(crate) fn rollout_retain_interval() -> usize {
    ROLLOUT_RETAIN_INTERVAL.load(Relaxed)
}
pub(crate) fn rollout_progress_interval() -> usize {
    ROLLOUT_PROGRESS_INTERVAL.load(Relaxed)
}
