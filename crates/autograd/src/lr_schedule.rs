//! Learning-rate schedules for the autograd optimizer stack.
//!
//! Every schedule is a pure function of the global optimizer step. Values are
//! `f32` to match [`crate::optim::AdamW`]'s LR precision.

use std::f32::consts::PI;

use libm::cosf;

/// Implementations must be deterministic across calls and cheap enough to
/// invoke every optimizer step.
pub trait LrSchedule: Send + Sync {
    /// Learning rate for the given optimizer step (1-indexed or 0-indexed —
    /// schedule defines). The wiring layer must pass a consistent step
    /// counter across resumes.
    fn lr(&self, step: u64) -> f32;

    fn describe(&self) -> String;
}

/// Linear warmup to `base_lr`, then half-cosine decay to `min_lr`; clamped to
/// `min_lr` after `total_steps`.
#[derive(Debug, Clone, Copy)]
pub struct CosineWithWarmup {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: u64,
    pub total_steps: u64,
}

impl LrSchedule for CosineWithWarmup {
    fn lr(&self, step: u64) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            let frac = step as f32 / self.warmup_steps as f32;
            return self.base_lr * frac;
        }

        if step >= self.total_steps {
            return self.min_lr;
        }

        // If total_steps <= warmup_steps there is no decay window; return
        // base_lr so callers never get NaN from a zero-length denominator.
        let decay_span = self.total_steps.saturating_sub(self.warmup_steps);
        if decay_span == 0 {
            return self.base_lr;
        }

        let progress = (step - self.warmup_steps) as f32 / decay_span as f32;
        let cosine = 0.5 * (1.0 + cosf(PI * progress));
        self.min_lr + (self.base_lr - self.min_lr) * cosine
    }

    fn describe(&self) -> String {
        format!(
            "cosine-with-warmup(base_lr={}, min_lr={}, warmup_steps={}, total_steps={})",
            self.base_lr, self.min_lr, self.warmup_steps, self.total_steps
        )
    }
}
