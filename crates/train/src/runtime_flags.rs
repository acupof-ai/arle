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
    /// Skip update records longer than this (`--max-update-seq`; 0 = unlimited).
    pub max_update_seq: usize,
    /// Autograd-crate knobs, forwarded to `autograd::apply_runtime_flags`.
    pub autograd: autograd::AutogradRuntimeFlags,
}

impl Default for TrainRuntimeFlags {
    fn default() -> Self {
        Self {
            writeback_offload: true,
            engine_offload: EngineOffloadMode::Off,
            gradient_checkpointing: true,
            trim_before_backward: false,
            trim_after_writeback: false,
            writeback_frozen_prompt_kv: false,
            rollout_retain_interval: 2,
            rollout_progress_interval: 16,
            max_update_seq: 23_000,
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
static MAX_UPDATE_SEQ: AtomicUsize = AtomicUsize::new(23_000);

pub fn apply_runtime_flags(f: &TrainRuntimeFlags) {
    WRITEBACK_OFFLOAD.store(f.writeback_offload, Relaxed);
    ENGINE_OFFLOAD.store(f.engine_offload as u8, Relaxed);
    GRADIENT_CHECKPOINTING.store(f.gradient_checkpointing, Relaxed);
    TRIM_BEFORE_BACKWARD.store(f.trim_before_backward, Relaxed);
    TRIM_AFTER_WRITEBACK.store(f.trim_after_writeback, Relaxed);
    WRITEBACK_FROZEN_PROMPT_KV.store(f.writeback_frozen_prompt_kv, Relaxed);
    ROLLOUT_RETAIN_INTERVAL.store(f.rollout_retain_interval.max(1), Relaxed);
    ROLLOUT_PROGRESS_INTERVAL.store(f.rollout_progress_interval.max(1), Relaxed);
    MAX_UPDATE_SEQ.store(f.max_update_seq, Relaxed);
    autograd::apply_runtime_flags(&f.autograd);
}

/// VRAM wall: H20-96GB OOMs the writeback backward at seq≈30K even with
/// offload + trims (alloc_zeros mul_backward, 2026-07-18); 22K peaked 90.7GB.
/// 0 = unlimited.
pub(crate) fn max_update_seq() -> usize {
    MAX_UPDATE_SEQ.load(Relaxed)
}

pub(crate) fn writeback_offload() -> bool {
    WRITEBACK_OFFLOAD.load(Relaxed)
}

/// Grad-checkpoint host-offload gate: seq-adaptive under the default flag, hard
/// off when the user passes `--writeback-offload false`.
///
/// The H2D re-upload serializes on the host thread and starves the GPU on short
/// trajectories (measured seq sweep 5K-12K on 27B: backward −29…−38%, zero peak
/// headroom vs resident — offload only slows this band). Post fused-CE +
/// batched-LA, resident checkpoints survive to seq=24576 and OOM at 28672
/// (wins/2026-07-24-writeback-offload-dial-back); 16384 keeps 1.5× margin below
/// last-proven-good and 25 GiB peak headroom on a 96 GB H20.
pub(crate) fn writeback_offload_for_seq(seq_len: usize) -> bool {
    const WRITEBACK_OFFLOAD_MIN_SEQ: usize = 16384;
    writeback_offload() && seq_len >= WRITEBACK_OFFLOAD_MIN_SEQ
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
