//! `ℒ = α_ce·ℒ_ce + α_tv·ℒ_tv + α_conf·ℒ_conf`, term for term with
//! deepseek-ai/DeepSpec (MIT) `deepspec/modeling/dspark/loss.py`. All three
//! share one denominator, [`Batch::denom`].

use anyhow::{Result, ensure};
use autograd::{Tape, TensorId, TensorStore, ops};

/// Defaults are the reference's (`config/dspark/*.py`).
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
    /// Numerator of `tau`: Σ over live blocks of 1 + sum(cumprod(accept)).
    /// Kept as raw sums so folding across chunks and samples is pooled —
    /// a per-chunk mean would move with `blocks_per_backward`.
    pub tau_sum: f32,
    /// Denominator of `tau`: live block count.
    pub tau_blocks: f32,
    /// Numerator of `accept_at`: Σ accept over live rows at each within-block
    /// position, length block_size.
    pub accept_at_sum: Vec<f32>,
    /// Denominator of `accept_at`: live row count at each position.
    pub accept_at_live: Vec<f32>,
}

impl Terms {
    /// Expected tokens committed per block: 1 + Σ cumprod(accept).
    /// Floor 1.0, ceiling 1 + block_size.
    #[must_use]
    pub fn tau(&self) -> f32 {
        if self.tau_blocks > 0.0 {
            self.tau_sum / self.tau_blocks
        } else {
            0.0
        }
    }

    /// Accept rate at each within-block position, length block_size.
    #[must_use]
    pub fn accept_at(&self) -> Vec<f32> {
        self.accept_at_sum
            .iter()
            .zip(&self.accept_at_live)
            .map(|(&sum, &n)| if n > 0.0 { sum / n } else { 0.0 })
            .collect()
    }
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
    terms.tau_sum = tau_sum;
    terms.tau_blocks = blocks as f32;
    terms.accept_at_sum = pos_accept;
    terms.accept_at_live = pos_live;
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
