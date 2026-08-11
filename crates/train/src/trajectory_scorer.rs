use autograd::{Tape, TensorStore};

use crate::opd::OpdError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactMatchScorer {
    reference: Vec<u32>,
}

impl ExactMatchScorer {
    pub fn new(reference: impl Into<Vec<u32>>) -> Self {
        Self {
            reference: reference.into(),
        }
    }

    pub fn reference(&self) -> &[u32] {
        &self.reference
    }

    pub fn score(
        &self,
        _prompt: &[u32],
        rollouts: &[Vec<u32>],
        _store: &mut TensorStore,
        _tape: &mut Tape,
    ) -> Result<Vec<f32>, OpdError> {
        Ok(rollouts
            .iter()
            .map(|rollout| {
                if rollout.as_slice() == self.reference.as_slice() {
                    1.0
                } else {
                    0.0
                }
            })
            .collect())
    }
}

pub fn select_best(rollouts: &[Vec<u32>], scores: &[f32]) -> usize {
    assert!(
        !rollouts.is_empty(),
        "select_best requires at least one candidate rollout"
    );
    assert_eq!(
        rollouts.len(),
        scores.len(),
        "select_best requires one score per candidate rollout"
    );

    let mut best_index = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (index, &score) in scores.iter().enumerate() {
        assert!(
            score.is_finite(),
            "select_best requires finite scores, got {score} at index {index}"
        );
        if score > best_score {
            best_index = index;
            best_score = score;
        }
    }
    best_index
}
