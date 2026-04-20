//! Learning-rate schedules for the autograd optimizer stack.
//!
//! Every schedule is a pure function of the global optimizer step, so the
//! trait objects returned by [`parse_lr_schedule`] can be plugged into any
//! training binary without persisted state. Values are `f32` to match
//! [`crate::optim::AdamW`]'s LR precision.

use std::f32::consts::PI;

use anyhow::{Result, bail};
use libm::cosf;

/// Schedule contract used by training binaries. Implementations must be
/// deterministic across calls and cheap enough to invoke every optimizer step.
pub trait LrSchedule: Send + Sync {
    /// Learning rate for the given (1-indexed or 0-indexed — schedule
    /// defines) optimizer step. The wiring layer is responsible for passing
    /// a consistent step counter across resumes.
    fn lr(&self, step: u64) -> f32;

    /// Human-readable one-liner for logs / config dumps.
    fn describe(&self) -> String;
}

/// Constant LR — useful as a no-op fallback or to disable warmup for short
/// debug runs.
#[derive(Debug, Clone, Copy)]
pub struct ConstantLr(pub f32);

impl LrSchedule for ConstantLr {
    fn lr(&self, _step: u64) -> f32 {
        self.0
    }

    fn describe(&self) -> String {
        format!("constant(lr={})", self.0)
    }
}

/// Linear warmup from 0 to `base_lr` over `warmup_steps`, then flat.
/// `warmup_steps == 0` degenerates to constant `base_lr`.
#[derive(Debug, Clone, Copy)]
pub struct LinearWarmup {
    pub base_lr: f32,
    pub warmup_steps: u64,
}

impl LrSchedule for LinearWarmup {
    fn lr(&self, step: u64) -> f32 {
        if self.warmup_steps == 0 {
            return self.base_lr;
        }
        if step >= self.warmup_steps {
            return self.base_lr;
        }
        // step in [0, warmup_steps): linear 0 -> base_lr.
        let frac = step as f32 / self.warmup_steps as f32;
        self.base_lr * frac
    }

    fn describe(&self) -> String {
        format!(
            "linear-warmup(base_lr={}, warmup_steps={})",
            self.base_lr, self.warmup_steps
        )
    }
}

/// Linear warmup to `base_lr`, then half-cosine decay down to `min_lr`
/// across the remaining `total_steps - warmup_steps`. After `total_steps`
/// the LR is clamped to `min_lr`.
#[derive(Debug, Clone, Copy)]
pub struct CosineWithWarmup {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: u64,
    pub total_steps: u64,
}

impl LrSchedule for CosineWithWarmup {
    fn lr(&self, step: u64) -> f32 {
        // Warmup branch (matches LinearWarmup semantics for warmup_steps > 0).
        if self.warmup_steps > 0 && step < self.warmup_steps {
            let frac = step as f32 / self.warmup_steps as f32;
            return self.base_lr * frac;
        }

        // Post-total clamp.
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

/// Parse a schedule spec name into a boxed trait object. Training binaries
/// pass the CLI string through here so the allow-list lives in one spot.
pub fn parse_lr_schedule(
    spec: &str,
    base_lr: f32,
    warmup: u64,
    total: u64,
    min_lr: f32,
) -> Result<Box<dyn LrSchedule>> {
    match spec {
        "constant" => Ok(Box::new(ConstantLr(base_lr))),
        "linear-warmup" => Ok(Box::new(LinearWarmup {
            base_lr,
            warmup_steps: warmup,
        })),
        "cosine-with-warmup" => Ok(Box::new(CosineWithWarmup {
            base_lr,
            min_lr,
            warmup_steps: warmup,
            total_steps: total,
        })),
        other => bail!(
            "unknown lr schedule '{other}'; expected one of: constant, linear-warmup, cosine-with-warmup"
        ),
    }
}
