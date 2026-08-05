//! `ℒ = α_ce·ℒ_ce + α_tv·ℒ_tv + α_conf·ℒ_conf`, term for term with
//! deepseek-ai/DeepSpec (MIT) `deepspec/modeling/dspark/loss.py`. All three
//! share one denominator, `Σ w`, over [`crate::block::row_weights`].

use anyhow::{Result, ensure};
use autograd::{Tape, TensorId, TensorStore, ops};

/// Objective mix. Defaults are the reference's (`config/dspark/*.py`).
#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub ce: f32,
    pub tv: f32,
    pub confidence: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            ce: 0.1,
            tv: 0.9,
            confidence: 1.0,
        }
    }
}

pub struct Batch<'a> {
    /// `[rows, vocab]`, Markov bias already applied.
    pub draft_logits: TensorId,
    /// `[rows, vocab]` trunk logits for the same positions.
    pub target_logits: TensorId,
    pub targets: &'a [usize],
    /// Zero on rows `eval` masked out.
    pub weights: &'a [f32],
    /// `[rows]` pre-sigmoid. `None` drops `ℒ_conf`.
    pub conf_logits: Option<TensorId>,
}

/// `Σ(x ⊙ w) / denom`. Sign and position decay fold into the constant `w`, so
/// this is also how the L1 subgradient is taken without an `abs` op.
fn weighted_mean(
    x: TensorId,
    w: &[f32],
    shape: &[usize],
    denom: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let w_id = store.from_slice(w, shape)?;
    let prod = ops::mul(x, w_id, store, tape)?;
    let total = ops::sum(prod, store, tape)?;
    Ok(ops::mul_scalar(total, 1.0 / denom, store, tape)?)
}

/// The acceptance a rejection-sampling verify would give, and the confidence
/// head's target.
#[must_use]
pub fn accept_targets(l1_per_row: &[f32]) -> Vec<f32> {
    l1_per_row
        .iter()
        .map(|&d| (1.0 - 0.5 * d).clamp(0.0, 1.0))
        .collect()
}

pub fn dspark_loss(
    batch: &Batch<'_>,
    weights: Weights,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let &Batch {
        draft_logits,
        target_logits,
        targets,
        weights: w,
        conf_logits,
    } = batch;
    let rows = targets.len();
    ensure!(rows > 0, "empty batch");
    ensure!(w.len() == rows, "weights {} != rows {rows}", w.len());
    let denom: f32 = w.iter().sum();
    ensure!(denom > 0.0, "every row is masked out");
    let vocab = *store
        .get(draft_logits)
        .and_then(|t| t.shape.last())
        .ok_or_else(|| anyhow::anyhow!("draft logits missing or rank-0"))?;

    let log_p = ops::log_softmax(draft_logits, store, tape)?;
    let tok_lp = ops::gather_last_dim(log_p, targets, store, tape)?;
    let neg_w: Vec<f32> = w.iter().map(|&x| -x).collect();
    let ce = weighted_mean(tok_lp, &neg_w, &[rows], denom, store, tape)?;

    // The vocab axis stays a sum; only the row axis is averaged. Squaring in
    // place of |·| is not "the same direction" at production vocab, where
    // probabilities run ~1e-5: ∂(p−q)² rescales each class by its own residual
    // and buries the tail TV weighs equally.
    let p_draft = ops::softmax(draft_logits, store, tape)?;
    let p_target = ops::softmax(target_logits, store, tape)?;
    let neg_target = ops::mul_scalar(p_target, -1.0, store, tape)?;
    let diff = ops::add(p_draft, neg_target, store, tape)?;
    let diff_host = store.to_host(diff)?;
    let signed_w: Vec<f32> = diff_host
        .iter()
        .enumerate()
        .map(|(i, &d)| w[i / vocab].copysign(d))
        .collect();
    let tv = weighted_mean(diff, &signed_w, &[rows, vocab], denom, store, tape)?;

    let mut loss = ops::add(
        ops::mul_scalar(ce, weights.ce, store, tape)?,
        ops::mul_scalar(tv, weights.tv, store, tape)?,
        store,
        tape,
    )?;

    if let Some(conf) = conf_logits {
        let l1: Vec<f32> = diff_host
            .chunks_exact(vocab)
            .map(|row| row.iter().map(|d| d.abs()).sum())
            .collect();
        let accept = accept_targets(&l1);
        // BCE-with-logits without a softplus op: `[z, 0]` through log_softmax
        // gives column 0 = log σ(z), column 1 = log(1 − σ(z)).
        let z = ops::reshape(conf, &[rows, 1], store, tape)?;
        let zeros = store.from_slice(&vec![0.0; rows], &[rows, 1])?;
        let pair = ops::cat(&[z, zeros], 1, store, tape)?;
        let log_sig = ops::log_softmax(pair, store, tape)?;
        let bce_w: Vec<f32> = (0..rows)
            .flat_map(|r| [-w[r] * accept[r], -w[r] * (1.0 - accept[r])])
            .collect();
        let conf_loss = weighted_mean(log_sig, &bce_w, &[rows, 2], denom, store, tape)?;
        loss = ops::add(
            loss,
            ops::mul_scalar(conf_loss, weights.confidence, store, tape)?,
            store,
            tape,
        )?;
    }

    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::{CpuBackend, Tensor};
    use std::sync::Arc;

    const VOCAB: usize = 4;

    fn setup() -> (TensorStore, Tape) {
        (
            TensorStore::with_backend(Arc::new(CpuBackend) as Arc<dyn autograd::Backend>),
            Tape::new(),
        )
    }

    fn put(store: &mut TensorStore, v: &[f32], shape: &[usize], grad: bool) -> TensorId {
        store.alloc(Tensor::new(v.to_vec(), shape.to_vec(), grad).unwrap())
    }

    fn only(term: &str) -> Weights {
        let mut w = Weights {
            ce: 0.0,
            tv: 0.0,
            confidence: 0.0,
        };
        match term {
            "ce" => w.ce = 1.0,
            "tv" => w.tv = 1.0,
            _ => w.confidence = 1.0,
        }
        w
    }

    /// Reference re-derived by hand, not read off the implementation.
    #[test]
    fn matches_a_hand_computed_reference() {
        let (mut s, mut t) = setup();
        let d = put(&mut s, &[2.0, 0.0, 0.0, 0.0], &[1, VOCAB], true);
        let g = put(&mut s, &[0.0, 2.0, 0.0, 0.0], &[1, VOCAB], false);
        let b = Batch {
            draft_logits: d,
            target_logits: g,
            targets: &[1],
            weights: &[1.0],
            conf_logits: None,
        };
        let loss = dspark_loss(&b, only("ce"), &mut s, &mut t).unwrap();
        let expect = ((2.0f32).exp() + 3.0).ln();
        assert!((s.to_host(loss).unwrap()[0] - expect).abs() < 1e-5);
    }

    #[test]
    fn tv_is_zero_when_the_distributions_agree_and_grows_as_they_part() {
        let run = |draft: &[f32], target: &[f32]| {
            let (mut s, mut t) = setup();
            let d = put(&mut s, draft, &[1, VOCAB], true);
            let g = put(&mut s, target, &[1, VOCAB], false);
            let b = Batch {
                draft_logits: d,
                target_logits: g,
                targets: &[0],
                weights: &[1.0],
                conf_logits: None,
            };
            let l = dspark_loss(&b, only("tv"), &mut s, &mut t).unwrap();
            s.to_host(l).unwrap()[0]
        };
        assert!(run(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]).abs() < 1e-6);
        assert!(run(&[9.0, 0.0, 0.0, 0.0], &[0.0, 9.0, 0.0, 0.0]) > 1.9);
    }

    #[test]
    fn dead_rows_do_not_reach_the_loss() {
        let mk = |row1: f32| {
            let (mut s, mut t) = setup();
            let d = put(
                &mut s,
                &[2.0, 0.0, 0.0, 0.0, row1, 0.0, 0.0, 0.0],
                &[2, VOCAB],
                true,
            );
            let g = put(
                &mut s,
                &[0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                &[2, VOCAB],
                false,
            );
            let b = Batch {
                draft_logits: d,
                target_logits: g,
                targets: &[1, 1],
                weights: &[1.0, 0.0],
                conf_logits: None,
            };
            let l = dspark_loss(&b, Weights::default(), &mut s, &mut t).unwrap();
            s.to_host(l).unwrap()[0]
        };
        assert!((mk(0.0) - mk(7.0)).abs() < 1e-6);
    }

    #[test]
    fn the_confidence_term_is_minimized_at_the_accept_rate_it_predicts() {
        let at = |z: f32| {
            let (mut s, mut t) = setup();
            let d = put(&mut s, &[1.0, 0.0, 0.0, 0.0], &[1, VOCAB], true);
            let g = put(&mut s, &[1.0, 0.0, 0.0, 0.0], &[1, VOCAB], false);
            let c = put(&mut s, &[z], &[1], true);
            let b = Batch {
                draft_logits: d,
                target_logits: g,
                targets: &[0],
                weights: &[1.0],
                conf_logits: Some(c),
            };
            let l = dspark_loss(&b, only("conf"), &mut s, &mut t).unwrap();
            s.to_host(l).unwrap()[0]
        };
        assert!(at(6.0) < at(0.0), "confident is better when accept == 1");
        assert!(at(0.0) < at(-6.0));
        assert!(at(6.0) < 0.01);
    }

    #[test]
    fn accept_target_is_the_rejection_sampling_rate() {
        assert_eq!(
            accept_targets(&[0.0, 1.0, 2.0, 3.0]),
            vec![1.0, 0.5, 0.0, 0.0]
        );
    }
}
