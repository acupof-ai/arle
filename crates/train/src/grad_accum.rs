//! Gradient-accumulation bookkeeper.
//!
//! Pure state machine: the training loop multiplies each micro-batch loss by
//! `loss_scale()` before `backward()`, calls `observe_and_check_ready()` per
//! micro-batch, and only runs `optimizer.step()` + `tape.zero_grad()` when
//! the observation returns `true` (i.e. the accumulation window closed).
//! After stepping, the loop calls `reset_after_step()` to start the next
//! window.
//!
//! No tensor or tape imports — this struct is deliberately inert. Tape
//! accumulation itself lives in `autograd::tape::Tape::accumulate_grad`.

/// Tracks where we are inside an accumulation window of size `accum_steps`.
#[derive(Debug)]
pub struct GradAccumulator {
    accum_steps: u64,
    current: u64,
}

impl GradAccumulator {
    /// Create a bookkeeper for a window of `accum_steps` micro-batches.
    ///
    /// # Panics
    /// Panics if `accum_steps == 0`.
    pub fn new(accum_steps: u64) -> Self {
        assert!(
            accum_steps >= 1,
            "GradAccumulator requires accum_steps >= 1"
        );
        Self {
            accum_steps,
            current: 0,
        }
    }

    /// Multiplier to apply to the micro-batch loss before `backward()`.
    ///
    /// Equal to `1.0 / accum_steps` so that the summed gradients across the
    /// window match the gradient of the mean loss.
    pub fn loss_scale(&self) -> f32 {
        1.0 / self.accum_steps as f32
    }

    /// Record one observed micro-batch. Returns `true` iff this observation
    /// filled the current accumulation window — the caller should then run
    /// `optimizer.step()`, `tape.zero_grad()`, and `reset_after_step()`.
    pub fn observe_and_check_ready(&mut self) -> bool {
        self.current += 1;
        self.current >= self.accum_steps
    }

    /// Window size this bookkeeper was constructed with.
    pub fn accum_steps(&self) -> u64 {
        self.accum_steps
    }

    /// Reset the in-window counter. Call exactly once after a successful
    /// `optimizer.step()` at the end of a window.
    pub fn reset_after_step(&mut self) {
        self.current = 0;
    }
}
