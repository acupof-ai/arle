//! `ℒ = α_ce·ℒ_ce + α_tv·ℒ_tv + α_conf·ℒ_conf`, term for term with
//! deepseek-ai/DeepSpec (MIT) `deepspec/modeling/dspark/loss.py`. All three
//! share one denominator, [`Batch::denom`].

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
    /// Normalizer. The sample-wide weight sum, so chunking the backward does
    /// not re-weight rows; callers pass it and the loss never recomputes it.
    pub denom: f32,
    /// `[rows]` pre-sigmoid. `None` drops `ℒ_conf`.
    pub conf_logits: Option<TensorId>,
    /// Rows are block-major: row `r` sits at within-block position `r % block_size`.
    pub block_size: usize,
}

/// Host-side value of each term, for logging.
#[derive(Debug, Clone, Default)]
pub struct Terms {
    pub ce: f32,
    pub tv: f32,
    pub conf: f32,
    pub mean_accept: f32,
    /// mean(sigmoid(z) - accept). Sign of the confidence head's error; BCE alone
    /// cannot distinguish an over-confident head from a saturated-low one.
    pub confidence_bias: f32,
    /// mean|sigmoid(z) - accept|.
    pub confidence_abs_error: f32,
    /// mean of (cumprod(sigmoid(z)) - cumprod(accept)) over a block — the statistic
    /// the serve actually thresholds.
    pub confidence_cumprod_bias: f32,
    /// mean over blocks of 1 + sum(cumprod(accept)): expected tokens committed per
    /// block. Floor 1.0, ceiling 1 + block_size.
    pub tau: f32,
    /// accept rate at each within-block position, length block_size.
    pub accept_at: Vec<f32>,
}

/// `Σ(x ⊙ w) / denom`. Position decay folds into the constant `w`.
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

/// The reference's gradient-free training diagnostics (`loss.py:134-198`).
///
/// `tau` and `accept_at` read the raw eval mask where the loss terms read the
/// decay weights — a row is live exactly when its weight is nonzero.
fn fill_diagnostics(
    terms: &mut Terms,
    accept: &[f32],
    sigma: Option<&[f32]>,
    w: &[f32],
    denom: f32,
    block_size: usize,
) {
    let mut pos_accept = vec![0.0f32; block_size];
    let mut pos_live = vec![0.0f32; block_size];
    let (mut tau_sum, mut blocks) = (0.0f32, 0usize);
    let (mut bias, mut abs_error, mut cumprod_bias) = (0.0f32, 0.0f32, 0.0f32);

    let by_block = w.chunks(block_size).zip(accept.chunks(block_size));
    for (b, (block_w, block_accept)) in by_block.enumerate() {
        let (mut accept_prefix, mut sigma_prefix, mut expected) = (1.0f32, 1.0f32, 0.0f32);
        for (k, (&wk, &a)) in block_w.iter().zip(block_accept).enumerate() {
            let live = f32::from(wk > 0.0);
            pos_accept[k] += a * live;
            pos_live[k] += live;
            accept_prefix *= a * live;
            expected += accept_prefix;
            if let Some(sigma) = sigma {
                let p = sigma[b * block_size + k];
                bias += (p - a) * wk;
                abs_error += (p - a).abs() * wk;
                sigma_prefix *= p * live;
                cumprod_bias += (sigma_prefix - accept_prefix) * wk;
            }
        }
        if block_w.iter().any(|&wk| wk > 0.0) {
            tau_sum += 1.0 + expected;
            blocks += 1;
        }
    }

    terms.confidence_bias = bias / denom;
    terms.confidence_abs_error = abs_error / denom;
    terms.confidence_cumprod_bias = cumprod_bias / denom;
    terms.tau = if blocks == 0 {
        0.0
    } else {
        tau_sum / blocks as f32
    };
    terms.accept_at = pos_accept
        .iter()
        .zip(&pos_live)
        .map(|(&sum, &n)| if n > 0.0 { sum / n } else { 0.0 })
        .collect();
}

pub fn dspark_loss(
    batch: &Batch<'_>,
    weights: Weights,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<(TensorId, Terms)> {
    let &Batch {
        draft_logits,
        target_logits,
        targets,
        weights: w,
        denom,
        conf_logits,
        block_size,
    } = batch;
    let rows = targets.len();
    ensure!(rows > 0, "empty batch");
    ensure!(w.len() == rows, "weights {} != rows {rows}", w.len());
    ensure!(denom > 0.0, "every row is masked out");
    ensure!(block_size > 0, "block_size must be positive");
    ensure!(
        rows % block_size == 0,
        "{rows} rows is not a whole number of {block_size}-row blocks"
    );
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
    let abs_diff = ops::abs(diff, store, tape)?;
    // Fold the vocab axis on device with a ones-GEMV: the row weights then
    // only need `rows` values, and the per-row L1 the confidence target wants
    // comes back as `rows` floats instead of the whole `[rows, vocab]` diff.
    let ones = store.from_slice(&vec![1.0; vocab], &[vocab, 1])?;
    let l1 = ops::matmul(abs_diff, ones, store, tape)?;
    let tv = weighted_mean(l1, w, &[rows, 1], denom, store, tape)?;

    let accept = accept_targets(&store.to_host(l1)?);
    let mean_accept = w
        .iter()
        .zip(accept.iter())
        .map(|(&wr, &a)| wr * a)
        .sum::<f32>()
        / denom;

    let mut loss = ops::add(
        ops::mul_scalar(ce, weights.ce, store, tape)?,
        ops::mul_scalar(tv, weights.tv, store, tape)?,
        store,
        tape,
    )?;

    let mut conf_value = 0.0;
    let mut sigma: Option<Vec<f32>> = None;
    if let Some(conf) = conf_logits {
        sigma = Some(
            store
                .to_host(conf)?
                .iter()
                .map(|&z| 1.0 / (1.0 + (-z).exp()))
                .collect(),
        );
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
        conf_value = store.to_host(conf_loss)?[0];
        loss = ops::add(
            loss,
            ops::mul_scalar(conf_loss, weights.confidence, store, tape)?,
            store,
            tape,
        )?;
    }

    let mut terms = Terms {
        ce: store.to_host(ce)?[0],
        tv: store.to_host(tv)?[0],
        conf: conf_value,
        mean_accept,
        ..Default::default()
    };
    fill_diagnostics(&mut terms, &accept, sigma.as_deref(), w, denom, block_size);
    Ok((loss, terms))
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
            denom: 1.0,
            conf_logits: None,
            block_size: 1,
        };
        let (loss, terms) = dspark_loss(&b, only("ce"), &mut s, &mut t).unwrap();
        let expect = ((2.0f32).exp() + 3.0).ln();
        assert!((s.to_host(loss).unwrap()[0] - expect).abs() < 1e-5);
        assert!((terms.ce - expect).abs() < 1e-5);
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
                denom: 1.0,
                conf_logits: None,
                block_size: 1,
            };
            let (l, _) = dspark_loss(&b, only("tv"), &mut s, &mut t).unwrap();
            s.to_host(l).unwrap()[0]
        };
        assert!(run(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]).abs() < 1e-6);
        assert!(run(&[9.0, 0.0, 0.0, 0.0], &[0.0, 9.0, 0.0, 0.0]) > 1.9);
    }

    /// TV's gradient only exists when the target genuinely differs — a test
    /// that passes the draft's own tensor as the target cancels it exactly.
    #[test]
    fn tv_gradient_moves_the_draft_toward_the_target() {
        let target = [0.0f32, 3.0, 0.0, -1.0];
        let tv_and_grad = |draft: &[f32]| {
            let (mut s, mut t) = setup();
            let d = put(&mut s, draft, &[1, VOCAB], true);
            let g = put(&mut s, &target, &[1, VOCAB], false);
            let b = Batch {
                draft_logits: d,
                target_logits: g,
                targets: &[0],
                weights: &[1.0],
                denom: 1.0,
                conf_logits: None,
                block_size: 1,
            };
            let (l, terms) = dspark_loss(&b, only("tv"), &mut s, &mut t).unwrap();
            let value = s.to_host(l).unwrap()[0];
            let grads = t.backward(l, &mut s).unwrap();
            assert!((terms.tv - value).abs() < 1e-6);
            (value, s.to_host(grads[&d]).unwrap())
        };

        let draft = [1.5f32, 0.0, 0.5, 0.0];
        let (before, grad) = tv_and_grad(&draft);
        assert!(before > 0.1, "target must actually differ: tv = {before}");
        assert!(
            grad.iter().any(|g| g.abs() > 1e-6),
            "TV gradient vanished: {grad:?}"
        );

        let stepped: Vec<f32> = draft.iter().zip(&grad).map(|(x, g)| x - 0.1 * g).collect();
        let (after, _) = tv_and_grad(&stepped);
        assert!(after < before, "TV did not fall: {before} -> {after}");
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
                denom: 1.0,
                conf_logits: None,
                block_size: 2,
            };
            let (l, _) = dspark_loss(&b, Weights::default(), &mut s, &mut t).unwrap();
            s.to_host(l).unwrap()[0]
        };
        assert!((mk(0.0) - mk(7.0)).abs() < 1e-6);
    }

    /// The `blocks_per_backward` gate: with the sample-wide `denom` held fixed,
    /// splitting the rows across chunks must leave the summed loss unchanged.
    /// Recomputing the denominator per chunk breaks this.
    #[test]
    fn splitting_the_rows_leaves_the_loss_unchanged() {
        let draft: Vec<f32> = (0..3 * VOCAB)
            .map(|i| (i as f32 * 0.7).sin() * 2.0)
            .collect();
        let target: Vec<f32> = (0..3 * VOCAB)
            .map(|i| (i as f32 * 0.4).cos() * 2.0)
            .collect();
        let conf = [0.3f32, -1.2, 0.8];
        let tokens = [1usize, 3, 0];
        let row_w = [1.0f32, 0.5, 0.25];
        let denom: f32 = row_w.iter().sum();

        let run = |sel: &[usize]| {
            let (mut s, mut t) = setup();
            let take = |src: &[f32]| -> Vec<f32> {
                sel.iter()
                    .flat_map(|&r| src[r * VOCAB..][..VOCAB].to_vec())
                    .collect()
            };
            let n = sel.len();
            let d = put(&mut s, &take(&draft), &[n, VOCAB], true);
            let g = put(&mut s, &take(&target), &[n, VOCAB], false);
            let c = put(
                &mut s,
                &sel.iter().map(|&r| conf[r]).collect::<Vec<_>>(),
                &[n],
                true,
            );
            let tk: Vec<usize> = sel.iter().map(|&r| tokens[r]).collect();
            let ws: Vec<f32> = sel.iter().map(|&r| row_w[r]).collect();
            let b = Batch {
                draft_logits: d,
                target_logits: g,
                targets: &tk,
                weights: &ws,
                denom,
                conf_logits: Some(c),
                block_size: 1,
            };
            let (l, _) = dspark_loss(&b, Weights::default(), &mut s, &mut t).unwrap();
            s.to_host(l).unwrap()[0]
        };

        let whole = run(&[0, 1, 2]);
        let split = run(&[0]) + run(&[1, 2]);
        assert!(
            (whole - split).abs() < 1e-5,
            "chunking changed the objective: {whole} vs {split}"
        );
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
                denom: 1.0,
                conf_logits: Some(c),
                block_size: 1,
            };
            let (l, terms) = dspark_loss(&b, only("conf"), &mut s, &mut t).unwrap();
            assert!((terms.mean_accept - 1.0).abs() < 1e-6);
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

    const FAR: f32 = -60.0;

    /// A draft/target logit row whose softmax L1 distance is `2(1 − a)`, so the
    /// row's accept rate is exactly `a`.
    fn accept_pair(a: f32) -> ([f32; VOCAB], [f32; VOCAB]) {
        let h = a / 2.0;
        let l = if h <= 0.0 { FAR } else { (h / (1.0 - h)).ln() };
        ([0.0, l, FAR, FAR], [l, 0.0, FAR, FAR])
    }

    /// BCE cannot tell a saturated-low head from an over-confident one: at accept
    /// 0.05 both cost ~2.9, and only the bias's sign says which one drafts nothing.
    #[test]
    fn confidence_bias_separates_a_saturated_head_from_an_over_confident_one() {
        let at = |z: f32| {
            let (mut s, mut t) = setup();
            let (draft, target) = accept_pair(0.05);
            let d = put(&mut s, &[draft, draft].concat(), &[2, VOCAB], true);
            let g = put(&mut s, &[target, target].concat(), &[2, VOCAB], false);
            let c = put(&mut s, &[z, z], &[2], true);
            let b = Batch {
                draft_logits: d,
                target_logits: g,
                targets: &[0, 0],
                weights: &[1.0, 1.0],
                denom: 2.0,
                conf_logits: Some(c),
                block_size: 2,
            };
            dspark_loss(&b, only("conf"), &mut s, &mut t).unwrap().1
        };

        let low = at(-60.0);
        let high = at(3.0);
        assert!((low.mean_accept - 0.05).abs() < 1e-4);
        assert!(
            (low.conf - high.conf).abs() < 0.2,
            "BCE must not separate them: {} vs {}",
            low.conf,
            high.conf
        );
        assert!(
            (low.confidence_bias + 0.05).abs() < 1e-3,
            "saturated low: {}",
            low.confidence_bias
        );
        assert!(
            (high.confidence_bias - 0.9026).abs() < 1e-3,
            "over-confident: {}",
            high.confidence_bias
        );
        assert!((low.confidence_abs_error - 0.05).abs() < 1e-3);
        assert!((high.confidence_abs_error - 0.9026).abs() < 1e-3);
        // The block statistic the serve thresholds: cumprod(0.05) = [0.05, 0.0025].
        assert!(
            (low.confidence_cumprod_bias + 0.02625).abs() < 1e-3,
            "saturated low: {}",
            low.confidence_cumprod_bias
        );
        assert!(
            (high.confidence_cumprod_bias - 0.9037).abs() < 1e-3,
            "over-confident: {}",
            high.confidence_cumprod_bias
        );
    }

    /// `tau` is the expected commits per block: `1 + Σ cumprod(accept)`.
    #[test]
    fn tau_counts_the_expected_commits_in_a_block() {
        let (mut s, mut t) = setup();
        let (draft, target) = accept_pair(0.5);
        let d = put(&mut s, &[draft, draft, draft].concat(), &[3, VOCAB], true);
        let g = put(
            &mut s,
            &[target, target, target].concat(),
            &[3, VOCAB],
            false,
        );
        let b = Batch {
            draft_logits: d,
            target_logits: g,
            targets: &[0, 0, 0],
            weights: &[1.0, 1.0, 1.0],
            denom: 3.0,
            conf_logits: None,
            block_size: 3,
        };
        let (_, terms) = dspark_loss(&b, Weights::default(), &mut s, &mut t).unwrap();
        assert!((terms.tau - 1.875).abs() < 1e-3, "tau = {}", terms.tau);
        assert_eq!(terms.confidence_bias, 0.0, "no head, no confidence stats");
    }

    #[test]
    fn accept_at_is_per_position_over_the_raw_eval_mask() {
        let rates = [1.0f32, 0.5, 0.5, 0.0];
        let run = |w: &[f32]| {
            let (mut s, mut t) = setup();
            let pairs: Vec<_> = rates.iter().map(|&a| accept_pair(a)).collect();
            let draft: Vec<f32> = pairs.iter().flat_map(|&(d, _)| d).collect();
            let target: Vec<f32> = pairs.iter().flat_map(|&(_, g)| g).collect();
            let d = put(&mut s, &draft, &[4, VOCAB], true);
            let g = put(&mut s, &target, &[4, VOCAB], false);
            let b = Batch {
                draft_logits: d,
                target_logits: g,
                targets: &[0, 0, 0, 0],
                weights: w,
                denom: w.iter().sum(),
                conf_logits: None,
                block_size: 2,
            };
            dspark_loss(&b, Weights::default(), &mut s, &mut t)
                .unwrap()
                .1
        };

        let decay = (-0.25f32).exp();
        let live = run(&[1.0, decay, 1.0, decay]);
        assert!(
            (live.accept_at[0] - 0.75).abs() < 1e-4,
            "{:?}",
            live.accept_at
        );
        assert!(
            (live.accept_at[1] - 0.25).abs() < 1e-4,
            "{:?}",
            live.accept_at
        );
        // Raw: the decay that pulls mean_accept off the 0.5 row mean leaves accept_at alone.
        assert!(
            (live.mean_accept - 0.5311).abs() < 1e-3,
            "{}",
            live.mean_accept
        );

        let dead = run(&[1.0, decay, 1.0, 0.0]);
        assert!(
            (dead.accept_at[1] - 0.5).abs() < 1e-4,
            "{:?}",
            dead.accept_at
        );
        assert!((dead.tau - 2.0).abs() < 1e-3, "tau = {}", dead.tau);
    }
}
