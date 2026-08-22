//! Train flags -> statics, applied once at CLI start; no env reads.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering::Relaxed};

use crate::opd::EngineOffloadMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainRuntimeFlags {
    pub writeback_offload: bool,
    pub engine_offload: EngineOffloadMode,
    pub gradient_checkpointing: bool,
    pub writeback_frozen_prompt_kv: bool,
    pub rollout_retain_interval: usize,
    pub rollout_progress_interval: usize,
    pub max_update_seq: usize,
    pub opd_seq_chunk: usize,
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
            opd_seq_chunk: 4096,
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
static OPD_SEQ_CHUNK: AtomicUsize = AtomicUsize::new(4096);

pub fn apply_runtime_flags(f: &TrainRuntimeFlags) {
    WRITEBACK_OFFLOAD.store(f.writeback_offload, Relaxed);
    ENGINE_OFFLOAD.store(f.engine_offload as u8, Relaxed);
    GRADIENT_CHECKPOINTING.store(f.gradient_checkpointing, Relaxed);
    // Works under CP: gen segment is CP-sharded, attention seeds from the
    // full prompt prefix K/V. Cost scales with supervised span, not full seq.
    WRITEBACK_FROZEN_PROMPT_KV.store(f.writeback_frozen_prompt_kv, Relaxed);
    ROLLOUT_RETAIN_INTERVAL.store(f.rollout_retain_interval.max(1), Relaxed);
    ROLLOUT_PROGRESS_INTERVAL.store(f.rollout_progress_interval.max(1), Relaxed);
    MAX_UPDATE_SEQ.store(f.max_update_seq, Relaxed);
    OPD_SEQ_CHUNK.store(f.opd_seq_chunk.max(1), Relaxed);
    autograd::apply_runtime_flags(&f.autograd);
}

pub(crate) fn max_update_seq() -> usize {
    MAX_UPDATE_SEQ.load(Relaxed)
}

/// MLP + full-attention recompute chunk. Always on — exact under position-wise/q-tiling.
pub fn opd_seq_chunk() -> usize {
    OPD_SEQ_CHUNK.load(Relaxed)
}

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
