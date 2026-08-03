//! DSpark train sidecar: hybrid policy-gradient + supervised logit matching.
//!
//! Drains the experience buffer populated by the inference hot path and runs
//! updates against the Markov head (the DSpark-specific trainable component).
//! The backbone transformer layers stay frozen; only `markov_w1` (embedding)
//! and `markov_w2` (linear projection) are updated.
//!
//! Two complementary loss signals (cf. DeepSpec `deepspec/modeling/dspark/loss.py`):
//! - **Policy gradient**: reward = accepted/block_size, baseline = EMA.
//!   `pg_loss = -log π(draft_tokens) * (reward - baseline)`.
//! - **Supervised probability matching**: dense signal from the captured
//!   `target_logits`. `loss = Σ(softmax(draft) - softmax(target))²`.
//!   Squared difference is a differentiable surrogate for L1/total-variation
//!   — same gradient direction, no `abs()` op needed. Directly optimises
//!   acceptance rate (`accept ≈ 1 - 0.5·TV`).
//!
//! Per-position exponential decay (`loss_decay_gamma`) up-weights earlier
//! tokens in the block: a mistake at position 0 voids positions 1..k.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use autograd::ops;
use autograd::{AdamW, CpuBackend, Tape, Tensor, TensorId, TensorStore};

use crate::grad_clip::{ensure_finite_loss, finite_optimizer_step};

/// One DSpark spec step's experience for RL training.
///
/// Mirrors `infer_cuda::DsparkExperience`; redefined here to avoid a hard
/// dependency on the infer-cuda crate from the train crate's non-cuda build.
#[derive(Clone)]
pub struct DsparkExperience {
    pub draft_tokens: Vec<u32>,
    pub draft_logits: Vec<f32>,
    pub target_logits: Vec<f32>,
    pub accepted: usize,
    pub block_size: usize,
    pub vocab_size: usize,
    /// Draft head mode (serve `first_row = !next_token_heads`): true = draft
    /// row j drafted chain[j+1] conditioned on chain[j]; false = same-position,
    /// row j drafted chain[j] conditioned on chain[j-1] and row 0 never drafts.
    pub next_token_heads: bool,
}

/// Saved-head tensor names — the draft loader's own, so a head trained here and
/// one read off a checkpoint are the same artifact.
const MARKOV_W1: &str = "markov_head.markov_w1.weight";
const MARKOV_W2: &str = "markov_head.markov_w2.weight";

/// Read a head written by [`DsparkTrainer::save_weights`] back as host f32
/// `(w1, w2)`, ready for `LoadedInferenceEngine::update_dspark_markov_weights`.
pub fn load_markov_head(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read markov head {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| anyhow::anyhow!("parse markov head {}: {e}", path.display()))?;
    let read = |name: &str| -> Result<Vec<f32>> {
        let t = st
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("{} missing {name}: {e}", path.display()))?;
        ensure!(
            t.dtype() == safetensors::Dtype::BF16,
            "{} {name}: expected BF16, got {:?}",
            path.display(),
            t.dtype()
        );
        Ok(t.data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect())
    };
    Ok((read(MARKOV_W1)?, read(MARKOV_W2)?))
}

/// Trait abstracting the experience buffer so the trainer does not depend on
/// the concrete `infer_cuda::DsparkExperienceBuffer` type.
pub trait ExperienceSource: Send + Sync {
    fn drain(&self, n: usize) -> Vec<DsparkExperience>;
}

/// Configuration for the DSpark trainer.
///
/// `vocab_size` is intentionally absent: the Markov head is lazily sized to the
/// actual vocab from the first drained experience, so the trainer is
/// model-agnostic at construction.
pub struct DsparkTrainConfig {
    /// Markov head rank. Only used as a fallback when the trainer cannot read
    /// the actual rank from the engine's loaded checkpoint.
    pub markov_rank: usize,
    pub learning_rate: f32,
    /// Experiences per optimizer step. Every cost in a step is linear in
    /// experiences (~200 ms each at vocab 248320, against a ~110 ms fixed floor),
    /// so batching does NOT change how fast experiences are consumed — it only
    /// divides how many optimizer steps you get out of them. Measured at the real
    /// head shape: 64 -> 20.9 s/step, 8 -> 1.70 s/step. Small wins.
    pub batch_size: usize,
    pub baseline_ema_alpha: f32,
    /// Starting value for the EMA baseline (the reward's running mean).
    /// `None` (default) seeds it from the first batch's mean reward. A pinned
    /// value biases every advantage until the EMA walks to the true mean, which
    /// at `baseline_ema_alpha` 0.01 takes hundreds of steps: the old 0.5 default
    /// against a measured mean reward of 0.20 left `r - baseline` negative on
    /// every step for ~340 steps, so the policy-gradient term pushed down the
    /// log-prob of every drafted token regardless of whether it was accepted.
    pub baseline_init: Option<f32>,
    /// Weight on the supervised probability-matching loss. The policy-gradient
    /// loss receives weight `1.0 - prob_match_alpha`. Default 0.5.
    pub prob_match_alpha: f32,
    /// Exponential decay scale for per-position loss weighting. Position `k` in
    /// the block is weighted `exp(-k/gamma)`. `None` disables decay (uniform).
    /// Default 4.0 (DeepSpec).
    pub loss_decay_gamma: Option<f32>,
    /// Global L2 gradient norm cap. `None` disables clipping. Default 1.0.
    pub max_grad_norm: Option<f32>,
    /// Where to write the trained Markov head (`markov_head.markov_w{1,2}.weight`,
    /// bf16 safetensors — the draft loader's own frame, so the file can be
    /// overlaid on the draft dir). `None` = train in-process only and lose the
    /// head at shutdown.
    pub save_path: Option<PathBuf>,
    /// Constrain the head to its base checkpoint's singular spectrum, letting
    /// the optimizer move only the singular frames (ISO, arXiv:2607.19331). The
    /// acceptance reward is the sparse verifiable kind the paper's observation
    /// was made on; see [`crate::iso_spectrum`]. Off by default — it is an
    /// unmeasured algorithm change on this head.
    pub iso_fixed_spectrum: bool,
    /// Steps between engine hot-swaps and checkpoint writes. Each swap is a
    /// `vocab*rank*2` bf16 H2D plus a full serve-stream sync, so doing it every
    /// step stalls decode for a gradient step's worth of drift.
    pub swap_every: usize,
}

impl Default for DsparkTrainConfig {
    fn default() -> Self {
        Self {
            markov_rank: 256,
            learning_rate: 1e-4,
            batch_size: 8,
            baseline_ema_alpha: 0.01,
            baseline_init: None,
            prob_match_alpha: 0.5,
            loss_decay_gamma: Some(4.0),
            max_grad_norm: Some(1.0),
            save_path: None,
            iso_fixed_spectrum: false,
            swap_every: 8,
        }
    }
}

/// The trainable Markov head parameters.
struct MarkovParams {
    w1: TensorId, // [vocab, rank]
    w2: TensorId, // [vocab, rank] — the serve gemm_batch weight frame
    rank: usize,
}

/// DSpark trainer: runs in a background thread, drains the experience
/// buffer, and runs acceptance-weighted updates on the Markov head.
pub struct DsparkTrainer {
    config: DsparkTrainConfig,
    store: TensorStore,
    tape: Tape,
    params: Option<MarkovParams>,
    /// Initial weights seeded from the engine's loaded checkpoint. `None` =
    /// random init (fallback when the engine has no Markov head to read).
    init_weights: Option<(Vec<f32>, Vec<f32>, usize)>,
    optim: AdamW,
    /// ISO frame optimizer, captured from the seeded head on the first step.
    /// `None` when `iso_fixed_spectrum` is off; the head then trains dense w1/w2.
    iso: Option<crate::iso_spectrum::IsoFrames>,
    /// Non-mutating spectrum probe, captured on BOTH arms — measures the head's
    /// σ²-drift from its base so an ISO-off run reports whether unconstrained
    /// updates stay near-isospectral (the paper's premise), not just the ISO-on
    /// retraction residual.
    spectrum_probe: Option<crate::iso_spectrum::SpectrumProbe>,
    baseline_ema: f32,
    /// Per-parameter relative spectral drift measured at the last cadence — the
    /// paper's premise on this head, emitted on both ISO on/off. Empty until the
    /// first cadence.
    last_iso_drift: Vec<f32>,
    /// Completed steps. Drives one shared cadence for retraction and publish,
    /// which is what guarantees a published head is always on the manifold.
    steps: usize,
    running: Arc<AtomicBool>,
}

impl DsparkTrainer {
    /// Create a new trainer.
    ///
    /// `init_weights` = `(w1 [vocab*rank], w2 [vocab*rank], rank)` read from
    /// the engine's loaded checkpoint. When provided, the trainer seeds from
    /// these (so acceptance never regresses at startup) and the actual `rank`
    /// overrides `config.markov_rank`. When `None`, falls back to Xavier-ish
    /// random init using `config.markov_rank`.
    ///
    /// `running` is the shared stop flag — the caller holds one end for the
    /// guard, the trainer checks it each loop iteration. This lets the caller
    /// grab the handle before construction (which may block on a GPU weight
    /// read) without a partially-initialized trainer.
    pub fn new(
        config: DsparkTrainConfig,
        init_weights: Option<(Vec<f32>, Vec<f32>, usize)>,
        running: Arc<AtomicBool>,
    ) -> Result<Self> {
        // ISO adapts a seeded head; a cold start makes w2 = 0 (no base spectrum
        // to freeze), so reject the combination up front rather than failing
        // every train step on the thread. Plan: reject cold heads before training.
        ensure!(
            !(config.iso_fixed_spectrum && init_weights.is_none()),
            "--dspark-train-iso requires a seeded head (--dspark-markov-init): ISO \
             freezes the base spectrum and cannot grow a head from a cold w2 = 0 start"
        );
        let backend: Arc<dyn autograd::Backend> = Arc::new(CpuBackend);
        let store = TensorStore::with_backend(backend);
        let tape = Tape::new();
        let optim = AdamW::new(config.learning_rate, (0.9, 0.999), 1e-8, 0.0);
        let baseline_ema = config.baseline_init.unwrap_or(0.0);

        Ok(Self {
            config,
            store,
            tape,
            params: None,
            iso: None,
            spectrum_probe: None,
            last_iso_drift: Vec::new(),
            steps: 0,
            init_weights,
            optim,
            baseline_ema,
            running,
        })
    }

    /// Build the Markov head tensors with the given vocab size.
    ///
    /// If `init_weights` was provided at construction, seed from the checkpoint
    /// (and use its rank). Otherwise use Xavier-ish random init with
    /// `config.markov_rank`.
    fn init_params(&mut self, vocab_size: usize) -> Result<()> {
        let (rank, w1_data, w2_data) = match self.init_weights.take() {
            Some((w1, w2, rank)) => {
                let expected_w1 = vocab_size * rank;
                let expected_w2 = vocab_size * rank;
                ensure!(
                    w1.len() == expected_w1,
                    "init w1 size mismatch: got {}, expected {expected_w1} (vocab={vocab_size}, rank={rank})",
                    w1.len()
                );
                ensure!(
                    w2.len() == expected_w2,
                    "init w2 size mismatch: got {}, expected {expected_w2} (vocab={vocab_size}, rank={rank})",
                    w2.len()
                );
                (rank, w1, w2)
            }
            None => {
                // Cold start: growing a head on a draft that ships without one
                // (a DFlash backbone — `fc` fusion, no `markov_head.*`). `w2 = 0`
                // makes `bias = Σ_r w2[v][r]·w1[c][r]` identically zero, so step 0
                // is an exact no-op and acceptance cannot regress below the
                // unbiased draft. `∂bias/∂w2 = w1[c] ≠ 0`, so w2 still trains and
                // unfreezes w1 as it grows. Random-initializing both bolts a
                // random rank-`markov_rank` bias onto a working drafter.
                let rank = self.config.markov_rank;
                let scale = 0.02;
                let w1: Vec<f32> = (0..vocab_size * rank)
                    .map(|i| {
                        let s = (i % 1000) as f32;
                        scale * (s * 0.1).sin()
                    })
                    .collect();
                (rank, w1, vec![0.0; vocab_size * rank])
            }
        };

        let w1 = self
            .store
            .alloc(Tensor::new(w1_data, vec![vocab_size, rank], true)?);
        let w2 = self
            .store
            .alloc(Tensor::new(w2_data, vec![vocab_size, rank], true)?);

        self.params = Some(MarkovParams { w1, w2, rank });
        Ok(())
    }

    /// #169 case probe (`ARLE_DSPARK_DUMP=<path>`): one JSONL line per trained
    /// row — the decoded ground truth aggregate metrics kept hiding. Capped by
    /// `ARLE_DSPARK_DUMP_ROWS` (default 512), process-global.
    fn dump_case_rows(
        path: &std::ffi::OsStr,
        draft: &[f32],
        target: &[f32],
        pg: &[usize],
        cond: &[usize],
        rewards: &[f32],
        vocab: usize,
    ) {
        use std::io::Write;
        static DUMPED: AtomicUsize = AtomicUsize::new(0);
        let cap = std::env::var("ARLE_DSPARK_DUMP_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let already = DUMPED.load(Ordering::Relaxed);
        if already >= cap {
            return;
        }
        let take = pg.len().min(cap - already);
        let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        else {
            return;
        };
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i)
        };
        for i in 0..take {
            let d = argmax(&draft[i * vocab..(i + 1) * vocab]);
            let t = argmax(&target[i * vocab..(i + 1) * vocab]);
            let _ = writeln!(
                f,
                "{{\"cond\":{},\"drafted\":{},\"draft_argmax\":{d},\"target_argmax\":{t},\"reward\":{}}}",
                cond[i], pg[i], rewards[i]
            );
        }
        DUMPED.fetch_add(take, Ordering::Relaxed);
    }

    /// Run one training step on a batch of experiences.
    pub fn train_step(&mut self, experiences: &[DsparkExperience]) -> Result<f32> {
        if experiences.is_empty() {
            return Ok(0.0);
        }

        let batch = experiences.len().min(self.config.batch_size);
        let experiences = &experiences[..batch];
        let vocab_size = experiences[0].vocab_size;
        let next_token_heads = experiences[0].next_token_heads;
        ensure!(
            experiences
                .iter()
                .all(|e| e.vocab_size == vocab_size && e.next_token_heads == next_token_heads),
            "heterogeneous experience batch (vocab_size or next_token_heads)"
        );

        // Lazily build Markov params with the actual vocab size from the
        // first experience, so the trainer is model-agnostic at construction.
        if self.params.is_none() {
            self.init_params(vocab_size)?;
            let p = self.params.as_ref().unwrap();
            let (w1, w2) = (p.w1, p.w2);
            // Base spectrum, captured on BOTH arms before any step — so an ISO-off
            // run measures whether unconstrained updates stay near-isospectral.
            self.spectrum_probe = Some(crate::iso_spectrum::SpectrumProbe::capture(
                &[w1, w2],
                &mut self.store,
            )?);
            if self.config.iso_fixed_spectrum {
                // The w1/w2 leaves stay as the reconstruction target; the frames
                // become trainable. Σ₀ is defined against these base weights.
                self.iso = Some(crate::iso_spectrum::IsoFrames::capture(
                    &[w1, w2],
                    &mut self.store,
                )?);
            }
        }
        let (rank, w1_leaf, w2_leaf) = {
            let p = self.params.as_ref().unwrap();
            (p.rank, p.w1, p.w2)
        };

        // Snapshot live tensors before the forward pass so we can free every
        // intermediate created during this step in one call (no manual ID list).
        let live_before: HashSet<TensorId> = self.store.live_ids().into_iter().collect();

        // ISO on: the head reads reconstructed `W = U diag(Σ₀) Vᵀ` (on the tape,
        // so backward flows to the frames); the trainable set is the frames. Off:
        // train the dense w1/w2 leaves directly. Both build the same forward.
        let (w1_id, w2_id, trainable) = if let Some(iso) = self.iso.as_ref() {
            let w1 = iso.reconstruct(0, &mut self.store, &mut self.tape)?;
            let w2 = iso.reconstruct(1, &mut self.store, &mut self.tape)?;
            (w1, w2, iso.frame_ids())
        } else {
            (w1_leaf, w2_leaf, vec![w1_leaf, w2_leaf])
        };

        // Row alignment mirrors the serve drafting loop (qwen35/dspark.rs
        // `first_row = !next_token_heads`): draft position t drafted
        // chain[t+1] at draft-logits row t + d, conditioned on chain[t];
        // the trunk verify (target) row t predicts chain[t+1]. Same-position
        // heads (d = 1) never draft from row 0, so it is excluded entirely.
        let d = usize::from(!next_token_heads);
        let mut draft_rows: Vec<f32> = Vec::new();
        let mut target_rows: Vec<f32> = Vec::new();
        let mut pg_tokens: Vec<usize> = Vec::new(); // chain[t+1] — the drafted token
        let mut cond_tokens: Vec<usize> = Vec::new(); // chain[t] — the bias condition
        let mut token_weights: Vec<f32> = Vec::new();
        let mut row_rewards: Vec<f32> = Vec::new();
        for exp in experiences {
            let chain = &exp.draft_tokens;
            let draft_len = exp.draft_logits.len() / vocab_size;
            let target_len = exp.target_logits.len() / vocab_size;
            let trained = (chain.len().saturating_sub(1))
                .min(draft_len.saturating_sub(d))
                .min(target_len);
            for t in 0..trained {
                let j = t + d;
                draft_rows
                    .extend_from_slice(&exp.draft_logits[j * vocab_size..(j + 1) * vocab_size]);
                target_rows
                    .extend_from_slice(&exp.target_logits[t * vocab_size..(t + 1) * vocab_size]);
                pg_tokens.push(chain[t + 1] as usize);
                cond_tokens.push(chain[t] as usize);
                // Exponential decay over the DRAFT position t within trained
                // rows (not the raw row index): a mistake at draft position 0
                // voids the rest of the block.
                token_weights.push(match self.config.loss_decay_gamma {
                    Some(gamma) if gamma > 0.0 => (-(t as f32) / gamma).exp(),
                    _ => 1.0,
                });
                // Per-token credit, not the chain mean: verify stops at the
                // first rejection, so `accepted` is the accepted PREFIX length
                // and row t drafted chain[t+1], accepted iff t < accepted.
                // Broadcasting the chain mean pushed up the log-prob of the
                // rejected tail on every above-baseline chain — the tokens that
                // caused the rejection.
                row_rewards.push(f32::from(t < exp.accepted));
            }
        }
        let n_rows = pg_tokens.len();
        if n_rows == 0 {
            return Ok(0.0);
        }
        // #169 case probe: decoded per-row ground truth (drafted token vs base
        // draft argmax vs target argmax). Env-gated, capped, append-only.
        if let Some(path) = std::env::var_os("ARLE_DSPARK_DUMP") {
            Self::dump_case_rows(
                &path,
                &draft_rows,
                &target_rows,
                &pg_tokens,
                &cond_tokens,
                &row_rewards,
                vocab_size,
            );
        }
        let weight_sum: f32 = token_weights.iter().sum();

        let logits_id = self.store.from_slice(&draft_rows, &[n_rows, vocab_size])?;
        let target_logits_id = self.store.from_slice(&target_rows, &[n_rows, vocab_size])?;

        // Markov bias in the SERVE coordinate frame: w2 is [vocab, rank]
        // row-major (gemm_batch weight), bias[v] = Σ_r w2[v][r] · w1[cond][r]
        // — i.e. emb [n, rank] · w2ᵀ.
        let emb_id = ops::embedding(w1_id, &cond_tokens, &mut self.store, &mut self.tape)?;
        let emb_flat_id = ops::reshape(emb_id, &[n_rows, rank], &mut self.store, &mut self.tape)?;
        let bias_id = ops::matmul_bt(emb_flat_id, w2_id, &mut self.store, &mut self.tape)?;
        let corrected_id = ops::add(logits_id, bias_id, &mut self.store, &mut self.tape)?;

        // ---- Policy-gradient loss (acceptance-weighted) ----
        let log_probs_id = ops::log_softmax(corrected_id, &mut self.store, &mut self.tape)?;
        let token_lp_id =
            ops::gather_last_dim(log_probs_id, &pg_tokens, &mut self.store, &mut self.tape)?;

        // Baseline must centre the quantity the advantage subtracts it from:
        // the per-token credit, under the same position weighting as the loss.
        let mean_reward: f32 = row_rewards
            .iter()
            .zip(&token_weights)
            .map(|(&r, &w)| r * w)
            .sum::<f32>()
            / weight_sum;
        let next_baseline = if self.steps == 0 && self.config.baseline_init.is_none() {
            // Seed from data, not from a guess: an advantage centred on the
            // wrong constant is a uniform push away from whatever the drafter
            // currently proposes.
            mean_reward
        } else {
            (1.0 - self.config.baseline_ema_alpha) * self.baseline_ema
                + self.config.baseline_ema_alpha * mean_reward
        };
        let baseline = next_baseline;
        // Per-token advantage × position weight.
        let weighted_adv: Vec<f32> = row_rewards
            .iter()
            .zip(&token_weights)
            .map(|(&r, &w)| (r - baseline) * w)
            .collect();
        let adv_id = self.store.from_slice(&weighted_adv, &[n_rows])?;

        let pg_weighted_id = ops::mul(token_lp_id, adv_id, &mut self.store, &mut self.tape)?;
        let pg_neg_id = ops::mul_scalar(pg_weighted_id, -1.0, &mut self.store, &mut self.tape)?;
        // Weighted mean: divide by sum of position weights (not token count).
        let pg_loss_id = ops::mul_scalar(
            ops::sum(pg_neg_id, &mut self.store, &mut self.tape)?,
            1.0 / weight_sum,
            &mut self.store,
            &mut self.tape,
        )?;

        // ---- Supervised probability-matching loss ----
        // Directly optimises acceptance rate: accept ≈ 1 - 0.5·TV(draft, target).
        // Squared difference (Frobenius) is a differentiable surrogate for L1/TV.
        let draft_probs_id = ops::softmax(corrected_id, &mut self.store, &mut self.tape)?;
        let target_probs_id = ops::softmax(target_logits_id, &mut self.store, &mut self.tape)?;
        // diff = softmax(draft) - softmax(target), computed in-graph.
        let neg_target_id =
            ops::mul_scalar(target_probs_id, -1.0, &mut self.store, &mut self.tape)?;
        let diff_id = ops::add(
            draft_probs_id,
            neg_target_id,
            &mut self.store,
            &mut self.tape,
        )?;
        let sq_diff_id = ops::mul(diff_id, diff_id, &mut self.store, &mut self.tape)?;
        // Expand per-token weights to [n_rows, vocab] for element-wise mul.
        let expanded_weights: Vec<f32> = token_weights
            .iter()
            .flat_map(|&w| vec![w; vocab_size])
            .collect();
        let exp_weight_id = self
            .store
            .from_slice(&expanded_weights, &[n_rows, vocab_size])?;
        let weighted_sq_id = ops::mul(sq_diff_id, exp_weight_id, &mut self.store, &mut self.tape)?;
        // Per-token mean of the squared-L2 distribution distance ‖softmax(draft) −
        // softmax(target)‖² — the TV surrogate from line 504, which is a SUM over
        // vocab (TV = ½Σ|p−q|), not a mean. Dividing by vocab_size too averaged a
        // sum-over-classes quantity as a per-class mean → ~1/248320 → the term
        // (and its gradient) underflowed f32, making prob_match_alpha inert at
        // production vocab scale. Normalize per token only.
        let prob_match_loss_id = ops::mul_scalar(
            ops::sum(weighted_sq_id, &mut self.store, &mut self.tape)?,
            1.0 / weight_sum,
            &mut self.store,
            &mut self.tape,
        )?;

        // ---- Combined loss ----
        let pg_alpha = 1.0 - self.config.prob_match_alpha;
        let pg_scaled_id = ops::mul_scalar(pg_loss_id, pg_alpha, &mut self.store, &mut self.tape)?;
        let pm_scaled_id = ops::mul_scalar(
            prob_match_loss_id,
            self.config.prob_match_alpha,
            &mut self.store,
            &mut self.tape,
        )?;
        let loss_id = ops::add(pg_scaled_id, pm_scaled_id, &mut self.store, &mut self.tape)?;
        let loss = self
            .store
            .to_host(loss_id)?
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("DSpark loss tensor is empty"))?;
        ensure_finite_loss(loss, &trainable, &mut self.optim, &mut self.store)?;

        self.tape.backward(loss_id, &mut self.store)?;
        finite_optimizer_step(
            loss,
            &trainable,
            self.config.max_grad_norm.unwrap_or(0.0),
            &mut self.optim,
            &mut self.store,
        )?;
        self.baseline_ema = next_baseline;
        self.steps += 1;
        // ISO: polar-retract both frames back onto the orthonormal manifold, so
        // σ(U diag(Σ₀) Vᵀ) = Σ₀ exactly.
        //
        // On the `swap_every` cadence, not every step: measured 5.0 s per
        // retraction at the real head shape (151936 × 256, both frames, M4 Pro
        // single-thread) against a ~1 s step, so per-step retraction would cost
        // more wall-clock than the paper's 2.2× step reduction buys back.
        // Measured inter-retraction drift is ~1e-3 relative, so the iterate never
        // leaves a tight neighbourhood of ℱ(W₀), and sharing the publish cadence
        // means the head handed to the engine is always freshly on the manifold.
        let on_cadence = self.steps.is_multiple_of(self.config.swap_every);
        let retract_result = if on_cadence && let Some(iso) = self.iso.as_mut() {
            iso.retract(&mut self.store)
        } else {
            Ok(())
        };
        // Spectrum drift on the cadence, measured on the trainable dense leaves.
        // ISO-OFF: the leaves ARE the trained head, so this is the paper's premise
        // — did the unconstrained step stay near-isospectral. ISO-ON: the leaves
        // are the frozen capture source (frames train, leaves don't move), so drift
        // is ~0 by construction — the served head's spectrum is Σ₀ definitionally,
        // and the ISO-off arm is the one that tests the premise.
        if on_cadence
            && retract_result.is_ok()
            && let Some(probe) = self.spectrum_probe.as_ref()
        {
            self.last_iso_drift = probe.drift(&mut self.store)?;
        }

        // Free every intermediate tensor created during this step (params w1/w2
        // are in `live_before`, so they survive). Runs even if retraction failed,
        // so a failed step never leaves stale tape entries for the next batch.
        let _ = self.store.free_new_except(&live_before, &HashSet::new());
        self.tape = Tape::new();
        retract_result?;

        Ok(loss)
    }

    /// Current Markov head as `(w1 [vocab*rank], w2 [vocab*rank])`. With ISO on,
    /// materialize `W = U diag(Σ₀) Vᵀ` from the frames — callers retract first,
    /// so the returned head is on ℱ(W₀). Off: read the dense w1/w2 leaves.
    pub fn get_weights(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        let Some(params) = self.params.as_ref() else {
            anyhow::bail!("Markov params not yet initialized");
        };
        let (w1_leaf, w2_leaf) = (params.w1, params.w2);
        if let Some(iso) = self.iso.as_ref() {
            let w1 = iso.materialize(0, &mut self.store)?;
            let w2 = iso.materialize(1, &mut self.store)?;
            return Ok((w1, w2));
        }
        let w1 = self.store.to_host(w1_leaf)?;
        let w2 = self.store.to_host(w2_leaf)?;
        Ok((w1, w2))
    }

    /// Write the current Markov head to `path` as a standalone bf16 safetensors
    /// file. Load it back with [`load_markov_head`] + `--dspark-markov-init` —
    /// **not** by copying into the draft dir: `SafetensorLoader` reads only the
    /// shards `model.safetensors.index.json` lists, so a loose file there is
    /// silently ignored.
    pub fn save_weights(&mut self, path: &Path) -> Result<()> {
        let rank = self
            .params
            .as_ref()
            .map(|p| p.rank)
            .ok_or_else(|| anyhow::anyhow!("Markov params not yet initialized"))?;
        let (w1, w2) = self.get_weights()?;
        let rows = w1.len() / rank;
        let to_bf16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
                .collect()
        };
        let (w1_b, w2_b) = (to_bf16(&w1), to_bf16(&w2));
        let tensors = [(MARKOV_W1, &w1_b), (MARKOV_W2, &w2_b)].map(|(name, bytes)| {
            (
                name.to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    vec![rows, rank],
                    bytes,
                )
                .expect("bf16 view over own buffer"),
            )
        });
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        // Write-then-rename: `serialize_to_file` truncates first, so a failed
        // periodic save would otherwise destroy the last good checkpoint.
        let tmp = path.with_extension("safetensors.tmp");
        safetensors::serialize_to_file(tensors, None, &tmp)
            .map_err(|e| anyhow::anyhow!("markov head save to {} failed: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Run the training loop. Blocks until `running` is set to false.
    ///
    /// Trains on full batches only — a 3-row step on an idle serve is gradient
    /// noise — and swaps/checkpoints every `swap_every` steps rather than every
    /// step (each swap costs the serve a full stream sync).
    pub fn run_loop(
        &mut self,
        source: &dyn ExperienceSource,
        on_weights: impl Fn(Vec<f32>, Vec<f32>) + Send,
    ) {
        self.running.store(true, Ordering::SeqCst);
        let mut pending: Vec<DsparkExperience> = Vec::new();
        while self.running.load(Ordering::SeqCst) {
            pending.extend(source.drain(self.config.batch_size - pending.len()));
            if pending.len() < self.config.batch_size {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            let batch = std::mem::take(&mut pending);
            match self.train_step(&batch) {
                Ok(loss) => {
                    let steps = self.steps;
                    // `pm_alpha` rides along with the drift: only the PG half of
                    // the objective is the RLVR regime ISO's fixed-spectrum
                    // premise was observed in, so a drift number is only
                    // interpretable next to how much of the step was dense
                    // self-distillation. Emitted on both ISO on/off (the probe
                    // fills `last_iso_drift` each cadence), so the α-sweep reads it.
                    let iso = match self.last_iso_drift.as_slice() {
                        [] => String::new(),
                        drift => format!(
                            " pm_alpha={:.2} spectrum_drift=[{}]",
                            self.config.prob_match_alpha,
                            drift
                                .iter()
                                .map(|d| format!("{d:.2e}"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    };
                    eprintln!(
                        "dspark_train: step={steps} loss={loss:.4} accept_ema={:.4} n={}{iso}",
                        self.baseline_ema,
                        batch.len()
                    );
                    if steps.is_multiple_of(self.config.swap_every) {
                        self.publish(&on_weights, steps);
                    }
                }
                Err(e) => eprintln!("dspark_train: train step failed: {e}"),
            }
        }
        if self.steps > 0 && !self.steps.is_multiple_of(self.config.swap_every) {
            // Final publish lands off-cadence, so retract first — never hand out
            // or checkpoint a head that is off ℱ(W₀).
            if let Some(iso) = self.iso.as_mut()
                && let Err(e) = iso.retract(&mut self.store)
            {
                eprintln!("dspark_train: final retraction failed: {e}");
            }
            let steps = self.steps;
            self.publish(&on_weights, steps);
        }
    }

    /// Hot-swap into the engine and checkpoint, if a save path is configured.
    fn publish(&mut self, on_weights: &(impl Fn(Vec<f32>, Vec<f32>) + Send), steps: usize) {
        match self.get_weights() {
            Ok((w1, w2)) => on_weights(w1, w2),
            Err(e) => {
                eprintln!("dspark_train: weight read failed: {e}");
                return;
            }
        }
        if let Some(path) = self.config.save_path.clone() {
            match self.save_weights(&path) {
                Ok(()) => eprintln!(
                    "dspark_train: saved markov head at step {steps} -> {}",
                    path.display()
                ),
                Err(e) => eprintln!("dspark_train: save failed: {e}"),
            }
        }
    }
}

/// Adapter that wraps an `infer_api::DsparkExperienceBuffer` and implements
/// [`ExperienceSource`], converting the infer-cuda experience type to the
/// train-crate local type.
#[cfg(feature = "cuda")]
pub struct InferCudaExperienceSource<'a> {
    buf: &'a infer_api::DsparkExperienceBuffer,
}

#[cfg(feature = "cuda")]
impl<'a> InferCudaExperienceSource<'a> {
    pub fn new(buf: &'a infer_api::DsparkExperienceBuffer) -> Self {
        Self { buf }
    }
}

#[cfg(feature = "cuda")]
impl<'a> ExperienceSource for InferCudaExperienceSource<'a> {
    fn drain(&self, n: usize) -> Vec<DsparkExperience> {
        self.buf
            .drain(n)
            .into_iter()
            .map(|e| DsparkExperience {
                draft_tokens: e.draft_tokens,
                draft_logits: e.draft_logits,
                target_logits: e.target_logits,
                accepted: e.accepted,
                block_size: e.block_size,
                vocab_size: e.vocab_size,
                next_token_heads: e.next_token_heads,
            })
            .collect()
    }
}

/// RAII guard for a spawned DSpark train sidecar training thread.
///
/// Dropping the guard signals the training loop to stop and waits for it to
/// exit. The guard is `Send` so it can live across the serve thread boundary.
#[cfg(feature = "cuda")]
pub struct DsparkTrainSidecarGuard {
    running: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "cuda")]
impl Drop for DsparkTrainSidecarGuard {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Spawn the DSpark train sidecar thread.
///
/// Drains the experience buffer populated by the CUDA inference hot path, runs
/// acceptance-weighted updates on the Markov head, and pushes updated weights
/// back into the running engine via
/// [`LoadedInferenceEngine::update_dspark_markov_weights`].
///
/// The sidecar reads the engine's loaded checkpoint weights (a blocking D2H
/// copy + stream sync) *inside* the spawned thread — it must not block serve
/// startup. The trainer is constructed there with those weights, so there is a
/// single init path and no partially-initialized trainer state.
///
/// Returns a guard that stops the thread on drop. No-ops (returns `None`) on
/// non-CUDA backends or when the experience buffer is unavailable.
#[cfg(feature = "cuda")]
pub fn spawn_dspark_train_sidecar(
    engine: Arc<infer_api::LoadedInferenceEngine>,
    config: DsparkTrainConfig,
) -> Result<Option<DsparkTrainSidecarGuard>> {
    let Some(buf) = engine.dspark_experience_buffer() else {
        return Ok(None);
    };
    // Create the stop flag on the serve thread so the guard can hold it before
    // the (potentially slow) trainer construction runs in the spawned thread.
    let running = Arc::new(AtomicBool::new(false));
    let running_for_guard = Arc::clone(&running);

    let join = std::thread::Builder::new()
        .name("dspark-train-sidecar".to_string())
        .spawn(move || {
            // Read checkpoint weights here (blocking D2H + sync), not on the
            // serve thread. Best-effort: fall back to random init on failure.
            let init_weights = match engine.get_dspark_markov_weights() {
                Ok(w) => Some(w),
                Err(e) => {
                    eprintln!(
                        "dspark_train: could not seed from checkpoint weights ({e}); using random init"
                    );
                    None
                }
            };
            let mut trainer = match DsparkTrainer::new(config, init_weights, running) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("dspark_train: failed to construct trainer: {e}");
                    return;
                }
            };
            let source = InferCudaExperienceSource::new(buf);
            trainer.run_loop(&source, |w1, w2| {
                if let Err(e) = engine.update_dspark_markov_weights(&w1, &w2) {
                    eprintln!("dspark_train: weight update failed: {e}");
                }
            });
        })?;

    Ok(Some(DsparkTrainSidecarGuard {
        running: running_for_guard,
        join: Some(join),
    }))
}
