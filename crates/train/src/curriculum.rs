//! Basic curriculum pool (M4.5).
//!
//! A `TaskPool` holds parameterized synthetic tasks with per-task rolling
//! pass@1 statistics. Tasks whose rolling pass@1 crosses `retire_threshold`
//! are retired so the sampler stops returning them; retired slots free up
//! compute for harder tasks that the operator adds later.
//!
//! This module is intentionally decoupled from the training loop — callers
//! sample a task, run a rollout, call `record(task_id, passed)`, and
//! periodically call `maybe_retire()`. No I/O, no model state.

use std::collections::VecDeque;

use crate::dataset::LcgRng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

/// A synthetic task. `target_range` / `prompt_len` are the knobs the
/// trainer currently cares about; extend this struct (not `Difficulty`)
/// when new task families come online.
#[derive(Debug, Clone)]
pub struct Task {
    pub id: usize,
    pub difficulty: Difficulty,
    pub target_range: usize,
    pub prompt_len: usize,
}

#[derive(Debug, Clone)]
struct TaskStats {
    window: VecDeque<bool>,
    retired: bool,
}

impl TaskStats {
    fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            retired: false,
        }
    }

    fn record(&mut self, passed: bool, window_size: usize) {
        if self.window.len() == window_size {
            self.window.pop_front();
        }
        self.window.push_back(passed);
    }

    /// Rolling pass@1 — or `None` if the window is empty (no data yet).
    fn pass_at_1(&self) -> Option<f32> {
        if self.window.is_empty() {
            return None;
        }
        let hits = self.window.iter().filter(|v| **v).count() as f32;
        Some(hits / self.window.len() as f32)
    }
}

#[derive(Debug)]
pub struct TaskPool {
    tasks: Vec<Task>,
    stats: Vec<TaskStats>,
    window_size: usize,
    retire_threshold: f32,
    min_samples_before_retire: usize,
}

impl TaskPool {
    pub fn new(window_size: usize, retire_threshold: f32) -> Self {
        Self::with_min_samples(window_size, retire_threshold, window_size)
    }

    /// Same as `new` but lets the caller require fewer samples before a
    /// task is eligible for retirement. Useful in tests.
    pub fn with_min_samples(
        window_size: usize,
        retire_threshold: f32,
        min_samples_before_retire: usize,
    ) -> Self {
        assert!(window_size > 0, "window_size must be positive");
        assert!(
            retire_threshold > 0.0 && retire_threshold <= 1.0,
            "retire_threshold must be in (0, 1]",
        );
        Self {
            tasks: Vec::new(),
            stats: Vec::new(),
            window_size,
            retire_threshold,
            min_samples_before_retire,
        }
    }

    /// Append a task; returns its index inside the pool (not the task id).
    pub fn add(&mut self, task: Task) -> usize {
        let index = self.tasks.len();
        self.tasks.push(task);
        self.stats.push(TaskStats::new(self.window_size));
        index
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    pub fn task(&self, index: usize) -> &Task {
        &self.tasks[index]
    }

    pub fn is_retired(&self, index: usize) -> bool {
        self.stats[index].retired
    }

    pub fn pass_at_1(&self, index: usize) -> Option<f32> {
        self.stats[index].pass_at_1()
    }

    /// Sample an active (non-retired) task uniformly. Returns `None` if
    /// every task has retired.
    pub fn sample(&self, rng: &mut LcgRng) -> Option<usize> {
        let active: Vec<usize> = (0..self.tasks.len())
            .filter(|i| !self.stats[*i].retired)
            .collect();
        if active.is_empty() {
            return None;
        }
        let pick = (rng.next_u64() as usize) % active.len();
        Some(active[pick])
    }

    /// Record a rollout outcome against a task.
    pub fn record(&mut self, index: usize, passed: bool) {
        self.stats[index].record(passed, self.window_size);
    }

    /// Mark every eligible task with rolling pass@1 ≥ threshold as retired.
    /// Returns the number of newly retired tasks this call.
    pub fn maybe_retire(&mut self) -> usize {
        let mut retired = 0usize;
        for stats in &mut self.stats {
            if stats.retired {
                continue;
            }
            if stats.window.len() < self.min_samples_before_retire {
                continue;
            }
            if let Some(rate) = stats.pass_at_1()
                && rate >= self.retire_threshold
            {
                stats.retired = true;
                retired += 1;
            }
        }
        retired
    }

    /// Counts of active (non-retired) tasks per difficulty tier. Returned
    /// in `(Easy, Medium, Hard)` order for stable logging.
    pub fn active_distribution(&self) -> [(Difficulty, usize); 3] {
        let mut easy = 0usize;
        let mut medium = 0usize;
        let mut hard = 0usize;
        for (task, stats) in self.tasks.iter().zip(self.stats.iter()) {
            if stats.retired {
                continue;
            }
            match task.difficulty {
                Difficulty::Easy => easy += 1,
                Difficulty::Medium => medium += 1,
                Difficulty::Hard => hard += 1,
            }
        }
        [
            (Difficulty::Easy, easy),
            (Difficulty::Medium, medium),
            (Difficulty::Hard, hard),
        ]
    }
}
