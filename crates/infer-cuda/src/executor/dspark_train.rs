//! DSpark train sidecar: experience capture from the inference hot path.
//!
//! After each DSpark verify+accept step, the (draft_tokens, draft_logits,
//! accepted_count) tuple is pushed into a bounded ring buffer. A separate
//! training thread drains it and runs acceptance-weighted policy gradient against the acceptance reward.
//!
//! The capture runs AFTER the verify forward has completed, on already-computed
//! data. It copies the logits to host (small: block_size × vocab bf16) via a
//! stream sync — the sync is bounded (it waits for the just-issued verify
//! forward + the small D2H copy) but does add per-step latency. A fully async
//! capture (separate stream + event fence) is future work.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};
use half::bf16;

/// One DSpark spec step's experience for RL training.
pub struct DsparkExperience {
    /// Draft tokens proposed by the draft head `[block_size]`.
    pub draft_tokens: Vec<u32>,
    /// Draft logits `[block_size, vocab]` as f32, copied from device.
    pub draft_logits: Vec<f32>,
    /// Target (trunk) logits `[block_size, vocab]` as f32, copied from device.
    /// Used for the L1 regularization term that prevents reward hacking.
    pub target_logits: Vec<f32>,
    /// Number of accepted tokens (the reward signal, 0..=block_size).
    pub accepted: usize,
    /// Block size (== draft_tokens.len()).
    pub block_size: usize,
    /// Vocab size (== draft_logits.len() / block_size).
    pub vocab_size: usize,
}

/// Bounded ring buffer of [`DsparkExperience`].
///
/// Capacity is fixed at construction; when full, the oldest entry is dropped.
/// A `Mutex` is fine here because the hot path only does a cheap push (the
/// D2H copy dominates, not the lock), and the trainer drains in batches.
pub struct DsparkExperienceBuffer {
    buf: Mutex<VecDeque<DsparkExperience>>,
    capacity: usize,
}

impl DsparkExperienceBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buf: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    /// Push one experience, dropping the oldest if at capacity.
    pub fn push(&self, exp: DsparkExperience) {
        let mut buf = self.buf.lock().expect("dspark experience buffer poisoned");
        if buf.len() >= self.capacity {
            buf.pop_front();
        }
        buf.push_back(exp);
    }

    /// Drain up to `n` experiences (FIFO). Returns fewer if the buffer has less.
    pub fn drain(&self, n: usize) -> Vec<DsparkExperience> {
        let mut buf = self.buf.lock().expect("dspark experience buffer poisoned");
        let take = n.min(buf.len());
        buf.drain(..take).collect()
    }

    /// Current length.
    pub fn len(&self) -> usize {
        self.buf
            .lock()
            .expect("dspark experience buffer poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Global experience buffer. Lazily initialized on first capture.
static BUFFER: OnceLock<DsparkExperienceBuffer> = OnceLock::new();

/// Default capacity: enough for ~10s of B=1 traffic at 50 tok/s with block_size=7.
const DEFAULT_CAPACITY: usize = 4096;

/// Get or create the global buffer.
pub fn buffer() -> &'static DsparkExperienceBuffer {
    BUFFER.get_or_init(|| DsparkExperienceBuffer::new(DEFAULT_CAPACITY))
}

/// Copy `HiddenStates` (bf16) to host as f32.
fn hidden_states_to_host(ctx: &DeviceContext, hs: &HiddenStates) -> Vec<f32> {
    match ctx.stream.clone_dtoh(&hs.data) {
        Ok(host_bf16) => {
            let _ = ctx.sync();
            host_bf16.iter().map(|x| x.to_f32()).collect()
        }
        Err(e) => {
            log::warn!("dspark_train: D2H draft logits copy failed: {e}");
            Vec::new()
        }
    }
}

/// Capture a DSpark experience from the hot path.
///
/// Called after `dspark_accept_commit` returns the accepted count `k`.
/// `draft_logits` are the draft head's outputs (`scratch.logits`); `target_logits`
/// are the trunk's verify outputs. Both are device-resident; this function copies
/// them to host (via a stream sync) and pushes the tuple to the global buffer.
///
/// Best-effort: returns immediately on any error so a capture failure never
/// aborts the decode step. The D2H copy + sync does add per-step latency.
pub fn capture_dspark_experience(
    ctx: &DeviceContext,
    draft_tokens: &[u32],
    draft_logits: &HiddenStates,
    target_logits: &DeviceVec,
    accepted: usize,
) {
    let block_size = draft_tokens.len();
    if block_size == 0 {
        return;
    }
    let draft_logits_host = hidden_states_to_host(ctx, draft_logits);
    let target_logits_host = match target_logits.to_host(ctx) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("dspark_train: D2H target logits copy failed: {e}");
            return;
        }
    };
    if draft_logits_host.is_empty() || target_logits_host.is_empty() {
        return;
    }
    let vocab_size = draft_logits_host.len() / block_size;
    // Validate target logits shape matches draft (block_size * vocab). A
    // mismatch here would silently feed garbage to the trainer's L1 loss.
    if target_logits_host.len() != block_size * vocab_size {
        log::warn!(
            "dspark_train: draft/target shape mismatch: draft_len={} target_len={} block_size={block_size} vocab={vocab_size}",
            draft_logits_host.len(),
            target_logits_host.len()
        );
        return;
    }
    buffer().push(DsparkExperience {
        draft_tokens: draft_tokens.to_vec(),
        draft_logits: draft_logits_host,
        target_logits: target_logits_host,
        accepted,
        block_size,
        vocab_size,
    });
}

/// DSv4 variant: target logits are a `[total_m, vocab]` `HiddenStates`
/// (`seq_len` = total tokens across all slots, `hidden_dim` = vocab); only
/// rows `[col_offset, col_offset + col_len)` belong to this slot.
pub fn capture_dspark_experience_hidden(
    ctx: &DeviceContext,
    draft_tokens: &[u32],
    draft_logits: &HiddenStates,
    target_logits: &HiddenStates,
    col_offset: usize,
    col_len: usize,
    accepted: usize,
) {
    let block_size = draft_tokens.len();
    if block_size == 0 || col_len == 0 {
        return;
    }
    let vocab = target_logits.hidden_dim;
    // Validate the slice is within bounds before touching device memory.
    if col_offset + col_len > target_logits.seq_len {
        log::warn!(
            "dspark_train: target logits slice out of bounds: offset={col_offset} len={col_len} seq_len={}",
            target_logits.seq_len
        );
        return;
    }
    let draft_logits_host = hidden_states_to_host(ctx, draft_logits);
    // Slice the target logits to this slot's rows: [col_offset, col_offset+col_len).
    let byte_off = col_offset * vocab * std::mem::size_of::<bf16>();
    let byte_len = col_len * vocab * std::mem::size_of::<bf16>();
    let target_logits_host: Vec<f32> = match ctx
        .stream
        .clone_dtoh(&target_logits.data.slice(byte_off..byte_off + byte_len))
    {
        Ok(host_bf16) => {
            let _ = ctx.sync();
            host_bf16.iter().map(|x| x.to_f32()).collect()
        }
        Err(e) => {
            log::warn!("dspark_train: D2H target logits copy failed: {e}");
            return;
        }
    };
    if draft_logits_host.is_empty() || target_logits_host.is_empty() {
        return;
    }
    // Validate draft/target vocab agree before pushing (a mismatch here means
    // the slice above grabbed the wrong rows and would silently corrupt training).
    let vocab_size = draft_logits_host.len() / block_size;
    if target_logits_host.len() != col_len * vocab_size {
        log::warn!(
            "dspark_train: draft/target vocab mismatch: draft_vocab={vocab_size} target_len={} col_len={col_len}",
            target_logits_host.len()
        );
        return;
    }
    buffer().push(DsparkExperience {
        draft_tokens: draft_tokens.to_vec(),
        draft_logits: draft_logits_host,
        target_logits: target_logits_host,
        accepted,
        block_size,
        vocab_size,
    });
}
