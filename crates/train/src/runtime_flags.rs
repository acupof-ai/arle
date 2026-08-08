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
static GRADIENT_CHECKPOINTING: AtomicBool = AtomicBool::new(true);
static WRITEBACK_FROZEN_PROMPT_KV: AtomicBool = AtomicBool::new(false);
static ROLLOUT_RETAIN_INTERVAL: AtomicUsize = AtomicUsize::new(2);
static ROLLOUT_PROGRESS_INTERVAL: AtomicUsize = AtomicUsize::new(16);
static MAX_UPDATE_SEQ: AtomicUsize = AtomicUsize::new(23_000);

pub fn apply_runtime_flags(f: &TrainRuntimeFlags) {
    WRITEBACK_OFFLOAD.store(f.writeback_offload, Relaxed);
    ENGINE_OFFLOAD.store(f.engine_offload as u8, Relaxed);
    GRADIENT_CHECKPOINTING.store(f.gradient_checkpointing, Relaxed);
    // The gen-segment forward is single-rank; CP shards the full sequence
    // instead. Normalized here rather than at each call site — two of the three
    // sites have no `cp` in scope, and one guarded site plus two unguarded ones
    // is how a latent flag becomes an active bug the day it is switched on.
    WRITEBACK_FROZEN_PROMPT_KV.store(
        f.writeback_frozen_prompt_kv
            && !crate::context_parallel::CpContext::from_env().is_enabled(),
        Relaxed,
    );
    ROLLOUT_RETAIN_INTERVAL.store(f.rollout_retain_interval.max(1), Relaxed);
    ROLLOUT_PROGRESS_INTERVAL.store(f.rollout_progress_interval.max(1), Relaxed);
    MAX_UPDATE_SEQ.store(f.max_update_seq, Relaxed);
    autograd::apply_runtime_flags(&f.autograd);
}

pub(crate) fn max_update_seq() -> usize {
    MAX_UPDATE_SEQ.load(Relaxed)
}

/// MLP + full-attention recompute chunk. Always on — both are exact under
/// position-wise / q-tiling chunking.
pub const OPD_SEQ_CHUNK: usize = 4096;

pub(crate) fn writeback_offload() -> bool {
    WRITEBACK_OFFLOAD.load(Relaxed)
}

/// Offload only where resident checkpoints no longer fit.
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
pub(crate) fn writeback_frozen_prompt_kv() -> bool {
    WRITEBACK_FROZEN_PROMPT_KV.load(Relaxed)
}
pub(crate) fn rollout_retain_interval() -> usize {
    ROLLOUT_RETAIN_INTERVAL.load(Relaxed)
}
pub(crate) fn rollout_progress_interval() -> usize {
    ROLLOUT_PROGRESS_INTERVAL.load(Relaxed)
}
