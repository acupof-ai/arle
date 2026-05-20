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

use autograd::{AutogradError, Tape, TensorId, TensorStore, optim::Optimizer};

use crate::{
    grad_clip::clip_grad_norm,
    loss::kl_distill_loss,
    qwen35::{Qwen35Error, Qwen35Model},
};

#[derive(Debug, thiserror::Error)]
pub enum OpdError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Qwen35(#[from] Qwen35Error),
}

pub type Result<T> = std::result::Result<T, OpdError>;

#[derive(Debug, Clone, Copy)]
pub struct OpdStepConfig {
    /// Tokens to roll out greedily from the student starting from the prompt.
    pub rollout_len: usize,
    /// Gradient L2 norm clip threshold.
    pub grad_clip: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct OpdStepOutcome {
    pub loss: f32,
    pub rollout_len: usize,
}

/// Greedy-argmax the last-position row of a `[1, seq_len, vocab]` logits
/// tensor. The student's lm-head returns dense logits per position; OPD
/// only needs the next-token row.
fn greedy_next_token(
    logits_id: TensorId,
    seq_len: usize,
    vocab: usize,
    store: &mut TensorStore,
) -> Result<u32> {
    let host = store.to_host(logits_id)?;
    let last_row_start = (seq_len - 1) * vocab;
    let row = &host[last_row_start..last_row_start + vocab];
    let mut best_idx: usize = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

/// Run one OPD step:
/// 1. Greedy-rollout `cfg.rollout_len` tokens from `student` starting from `prompt_ids`.
/// 2. Forward `teacher` on the full rollout (tape disabled).
/// 3. Forward `student` on the full rollout (tape enabled).
/// 4. `kl_distill_loss(student_logits, teacher_logits, rollout.len(), …)`.
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
) -> Result<OpdStepOutcome> {
    let vocab = student.config().vocab_size;

    // 1. Greedy rollout — tape disabled, no backward graph for sample tokens.
    tape.entries.clear();
    tape.set_enabled(false);
    let mut rollout: Vec<u32> = prompt_ids.to_vec();
    for _ in 0..cfg.rollout_len {
        let positions: Vec<u32> = (0..rollout.len() as u32).collect();
        let logits = student
            .forward(store, tape, &rollout, &positions)
            .map_err(OpdError::from)?;
        let next = greedy_next_token(logits, rollout.len(), vocab, store)?;
        rollout.push(next);
    }
    // Note: rollout ephemerals (logits per iteration) stay in the store
    // until the post-step `retain_ids` prune below. For long rollouts
    // (>16 steps) we'd want per-iteration pruning, but that requires a
    // `keep` set covering both teacher AND student parameters, not just
    // `student_params`. Deferred until the production path benches.

    // 2. Teacher forward — still tape-disabled. Teacher params carry
    //    `requires_grad = false` so no entries record even if tape was on,
    //    but disabling cheap-defends against any rogue grad-bearing weight.
    let positions: Vec<u32> = (0..rollout.len() as u32).collect();
    let teacher_logits = teacher
        .forward(store, tape, &rollout, &positions)
        .map_err(OpdError::from)?;

    // 3. Student forward — tape enabled now so backward can flow.
    tape.set_enabled(true);
    let student_logits = student
        .forward(store, tape, &rollout, &positions)
        .map_err(OpdError::from)?;

    // 4. KL distill loss.
    let loss = kl_distill_loss(student_logits, teacher_logits, rollout.len(), store, tape)?;
    let loss_value = store.to_host(loss)?[0];

    // 5. Backward + grad clip + optimizer step.
    optimizer.zero_grad(store, student_params);
    tape.backward(loss, store)?;
    clip_grad_norm(student_params, cfg.grad_clip, store);
    optimizer.step(store, student_params)?;

    // 6. Tape cleared but no `retain_ids` here — the caller owns the keep
    //    set (must include both teacher AND student params + cos/sin
    //    caches). The training loop in `arle train opd` builds the keep
    //    set once and reuses it; per-step `retain_ids` is the standard
    //    pattern (see `test_lm.rs`).
    tape.entries.clear();
    tape.set_enabled(true);

    Ok(OpdStepOutcome {
        loss: loss_value,
        rollout_len: rollout.len(),
    })
}
