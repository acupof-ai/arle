use super::opd_runtime::PromptSampler;

#[derive(Debug, Clone, Copy, Default)]
struct TaskStats {
    ema_pass: Option<f32>,
    hot_rounds: u32,
    retired: bool,
}

/// Pass-rate task selection: variance-weighted sampling — concentrate rollout
/// on the p≈0.5 max-variance band where reward-bearing R(p,k)=1−p^k−(1−p)^k peaks
/// — plus EMA-pass retirement of mastered tasks, with a 0.1 exploration floor.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) struct TaskSelection {
    rng: PromptSampler,
    stats: Vec<TaskStats>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl TaskSelection {
    pub(super) fn new(n_tasks: usize) -> Self {
        Self {
            // Fixed seed → runs are reproducible.
            rng: PromptSampler::new(0x5EED),
            stats: vec![TaskStats::default(); n_tasks],
        }
    }

    /// Predictive keep-probability from the online pass-estimate: run a task in
    /// proportion to its reward-bearing variance v = p(1−p) (∈[0,0.25], max at
    /// p=0.5), normalized so p≈0.5 always runs and a 0.1 floor keeps the tails
    /// explored (a skipped task's EMA can only refresh on a round it runs).
    fn keep_prob(ema_pass: Option<f32>) -> f64 {
        let Some(p) = ema_pass else {
            return 1.0; // unseen → run: need an estimate before we can weight
        };
        let v = f64::from(p) * f64::from(1.0 - p);
        (v / 0.25).max(0.1)
    }

    /// Update from a completed group; skipped rounds freeze the estimate.
    pub(super) fn record(&mut self, task: usize, pass_rate: f32) {
        let s = &mut self.stats[task];
        let ema = s.ema_pass.map_or(pass_rate, |e| 0.3 * pass_rate + 0.7 * e);
        s.ema_pass = Some(ema);
        s.hot_rounds = if ema >= 0.9 { s.hot_rounds + 1 } else { 0 };
        s.retired = s.retired || s.hot_rounds >= 3;
    }

    /// A mastered task (EMA pass ≥ 0.9 for 3 rounds) — never a DAPO refill target.
    pub(super) fn is_retired(&self, task: usize) -> bool {
        self.stats[task].retired
    }

    /// Task indices to run this round + (skipped, retired) counts.
    /// Round 0 runs ALL tasks: no history yet, and the baseline needs it.
    pub(super) fn select(&mut self, round: usize) -> (Vec<usize>, usize, usize) {
        let (mut run, mut skipped, mut retired) = (Vec::new(), 0, 0);
        for i in 0..self.stats.len() {
            let s = self.stats[i];
            if s.retired {
                retired += 1;
            } else if round > 0 && self.rng.next_unit() >= Self::keep_prob(s.ema_pass) {
                skipped += 1;
            } else {
                run.push(i);
            }
        }
        (run, skipped, retired)
    }
}

/// Experience replay: age-bounded, |A|-prioritized buffer of trained groups.
/// Entries retain the generation-time behavior sidecars, so every reuse stays
/// corrected against the policy that sampled the trajectories.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone)]
pub(super) struct ReplayEntry {
    pub(super) batch: Vec<train::update_strategy::ScoredTrajectory>,
    pub(super) task_id: String,
    pub(super) round: usize,
    pub(super) priority: f32,
}

/// Survey-grounded staleness bound: evict entries older than 10 rounds.
const REPLAY_MAX_AGE: usize = 10;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Default)]
pub(super) struct ReplayBuffer {
    entries: Vec<ReplayEntry>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl ReplayBuffer {
    /// Priority = mean |reward − group mean|; 0 ⇔ zero-variance (nothing to learn).
    fn priority(rewards: &[f32]) -> f32 {
        if rewards.is_empty() {
            return 0.0;
        }
        let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        rewards.iter().map(|r| (r - mean).abs()).sum::<f32>() / rewards.len() as f32
    }

    /// Zero-variance groups never enter the buffer.
    pub(super) fn push(
        &mut self,
        round: usize,
        task_id: String,
        batch: Vec<train::update_strategy::ScoredTrajectory>,
    ) {
        let rewards: Vec<f32> = batch.iter().map(|t| t.reward).collect();
        let priority = Self::priority(&rewards);
        if priority > 0.0 {
            self.entries.push(ReplayEntry {
                batch,
                task_id,
                round,
                priority,
            });
        }
    }

    /// Evict age > [`REPLAY_MAX_AGE`], then draw up to `n` entries without
    /// replacement, P(entry) ∝ priority — deterministic given the fixed-seed rng.
    pub(super) fn draw(
        &mut self,
        round: usize,
        n: usize,
        rng: &mut PromptSampler,
    ) -> Vec<ReplayEntry> {
        self.entries.retain(|e| round - e.round <= REPLAY_MAX_AGE);
        let mut pool: Vec<usize> = (0..self.entries.len()).collect();
        let mut out = Vec::new();
        while out.len() < n && !pool.is_empty() {
            let total: f64 = pool
                .iter()
                .map(|&i| f64::from(self.entries[i].priority))
                .sum();
            let mut x = rng.next_unit() * total;
            let pick = pool
                .iter()
                .position(|&i| {
                    x -= f64::from(self.entries[i].priority);
                    x <= 0.0
                })
                .unwrap_or(pool.len() - 1);
            out.push(self.entries[pool.remove(pick)].clone());
        }
        out
    }
}
