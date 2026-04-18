//! Shared-state control plane for long-running training jobs.
//!
//! A `TrainingController` wraps the trainer's live counters plus a
//! cooperative `should_stop` flag. The trainer reads/writes through
//! `update` + `should_stop`; external drivers (HTTP handlers, tests)
//! read via `snapshot` and signal via `request_stop` / `request_save`.
//! Locking is coarse — one `Mutex` around a ~hundred-byte struct —
//! because iterations are orders of magnitude slower than the
//! contended section.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Default)]
pub struct TrainingStatus {
    pub iter: usize,
    pub total_iters: usize,
    pub mean_reward: f32,
    pub best_reward: f32,
    pub last_kl: f32,
    pub last_loss: f32,
    pub wall_secs: f32,
    pub started: bool,
    pub finished: bool,
}

#[derive(Debug, Default)]
pub struct TrainingController {
    status: Mutex<TrainingStatus>,
    stop: AtomicBool,
    save: AtomicBool,
}

impl TrainingController {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn snapshot(&self) -> TrainingStatus {
        self.status.lock().expect("status poisoned").clone()
    }

    pub fn update(&self, f: impl FnOnce(&mut TrainingStatus)) {
        let mut guard = self.status.lock().expect("status poisoned");
        f(&mut guard);
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// External driver requests the trainer to flush a checkpoint at
    /// the next iteration boundary. Flag is edge-triggered — the
    /// trainer resets it via `take_save_request` once handled.
    pub fn request_save(&self) {
        self.save.store(true, Ordering::Release);
    }

    pub fn take_save_request(&self) -> bool {
        self.save.swap(false, Ordering::AcqRel)
    }
}
