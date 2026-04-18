//! Task generator (M4.6 scaffold).
//!
//! Emits `GeneratedTask`s that bundle a `Task` with a `VerifierKind`.
//! Pairing is structural — you cannot construct a generated task without
//! a verifier, which enforces the "verifier-grounded" invariant called
//! for in the M4.6 plan. Difficulty is sampled from caller-declared
//! weights, and per-difficulty parameter bands keep generated tasks
//! inside each tier's intended shape.
//!
//! This is a synthetic stand-in for true self-play. When the trainer
//! later grows a task-proposing model, that component slots in behind
//! the same `GeneratedTask` boundary.

use crate::curriculum::{Difficulty, Task};
use crate::dataset::LcgRng;
use crate::verifier::VerifierKind;

/// Declarative spec for one difficulty tier: parameter ranges plus the
/// verifier kinds eligible for this tier. Ranges are inclusive.
#[derive(Debug, Clone)]
pub struct TierSpec {
    pub difficulty: Difficulty,
    pub weight: f32,
    pub target_range: (usize, usize),
    pub prompt_len: (usize, usize),
    pub verifiers: Vec<VerifierKind>,
}

impl TierSpec {
    fn validate(&self) {
        assert!(self.weight > 0.0, "tier weight must be positive");
        assert!(
            self.target_range.0 > 0 && self.target_range.0 <= self.target_range.1,
            "target_range must be positive and non-empty: got {:?}",
            self.target_range,
        );
        assert!(
            self.prompt_len.0 > 0 && self.prompt_len.0 <= self.prompt_len.1,
            "prompt_len must be positive and non-empty: got {:?}",
            self.prompt_len,
        );
        assert!(
            !self.verifiers.is_empty(),
            "tier must list ≥1 verifier kind"
        );
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedTask {
    pub task: Task,
    pub verifier: VerifierKind,
}

/// Samples tasks across tiers according to declared weights.
#[derive(Debug, Clone)]
pub struct TaskGenerator {
    tiers: Vec<TierSpec>,
    cumulative: Vec<f32>,
    total_weight: f32,
    next_id: usize,
}

impl TaskGenerator {
    pub fn new(tiers: Vec<TierSpec>) -> Self {
        assert!(!tiers.is_empty(), "generator needs ≥1 tier");
        for tier in &tiers {
            tier.validate();
        }
        let mut cumulative = Vec::with_capacity(tiers.len());
        let mut running = 0.0_f32;
        for tier in &tiers {
            running += tier.weight;
            cumulative.push(running);
        }
        Self {
            tiers,
            cumulative,
            total_weight: running,
            next_id: 0,
        }
    }

    pub fn tiers(&self) -> &[TierSpec] {
        &self.tiers
    }

    /// Draw one task. Every call mints a fresh monotonic `Task.id`.
    pub fn generate(&mut self, rng: &mut LcgRng) -> GeneratedTask {
        let tier_index = self.sample_tier(rng);
        let tier = &self.tiers[tier_index];
        let target_range = sample_in_range(rng, tier.target_range);
        let prompt_len = sample_in_range(rng, tier.prompt_len);
        let verifier = tier.verifiers[pick_index(rng, tier.verifiers.len())].clone();
        let task = Task {
            id: self.next_id,
            difficulty: tier.difficulty,
            target_range,
            prompt_len,
        };
        self.next_id += 1;
        GeneratedTask { task, verifier }
    }

    fn sample_tier(&self, rng: &mut LcgRng) -> usize {
        let draw = next_unit(rng) * self.total_weight;
        for (idx, cutoff) in self.cumulative.iter().enumerate() {
            if draw <= *cutoff {
                return idx;
            }
        }
        self.tiers.len() - 1
    }
}

fn sample_in_range(rng: &mut LcgRng, (lo, hi): (usize, usize)) -> usize {
    if lo == hi {
        return lo;
    }
    let span = hi - lo + 1;
    lo + pick_index(rng, span)
}

fn pick_index(rng: &mut LcgRng, upper: usize) -> usize {
    if upper <= 1 {
        0
    } else {
        (rng.next_u64() % upper as u64) as usize
    }
}

fn next_unit(rng: &mut LcgRng) -> f32 {
    // Top 24 bits → [0, 1). Keeps mantissa precision without bias.
    let bits = (rng.next_u64() >> 40) as u32;
    bits as f32 / (1u32 << 24) as f32
}
