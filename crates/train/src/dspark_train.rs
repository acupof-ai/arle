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
//! - **Supervised L1 on probabilities**: dense signal from the captured
//!   `target_logits`. `l1_loss = Σ|softmax(draft) - softmax(target)|`.
//!   This directly optimises acceptance rate (`accept ≈ 1 - 0.5·TV`).
//!
//! Per-position exponential decay (`loss_decay_gamma`) up-weights earlier
//! tokens in the block: a mistake at position 0 voids positions 1..k.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use autograd::ops;
use autograd::{AdamW, CpuBackend, Tape, Tensor, TensorId, TensorStore};

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
}

/// Trait abstracting the experience buffer so the trainer does not depend on
/// the concrete `infer_cuda::DsparkExperienceBuffer` type.
pub trait ExperienceSource: Send + Sync {
    fn drain(&self, n: usize) -> Vec<DsparkExperience>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Configuration for the DSpark trainer.
///
/// `vocab_size` is intentionally absent: the Markov head is lazily sized to the
/// actual vocab from the first drained experience, so the trainer is
/// model-agnostic at construction.
pub struct DsparkTrainConfig {
    pub markov_rank: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub baseline_ema_alpha: f32,
    /// Weight on the supervised L1 probability-matching loss. The policy-gradient
    /// loss receives weight `1.0 - l1_loss_alpha`. Default 0.9 (DeepSpec).
    pub l1_loss_alpha: f32,
    /// Exponential decay scale for per-position loss weighting. Position `k` in
    /// the block is weighted `exp(-k/gamma)`. `None` disables decay (uniform).
    /// Default 4.0 (DeepSpec).
    pub loss_decay_gamma: Option<f32>,
}

impl Default for DsparkTrainConfig {
    fn default() -> Self {
        Self {
            markov_rank: 256,
            learning_rate: 1e-4,
            batch_size: 64,
            baseline_ema_alpha: 0.01,
            l1_loss_alpha: 0.9,
            loss_decay_gamma: Some(4.0),
        }
    }
}

/// The trainable Markov head parameters.
struct MarkovParams {
    w1: TensorId, // [vocab, rank]
    w2: TensorId, // [rank, vocab]
}

/// DSpark trainer: runs in a background thread, drains the experience
/// buffer, and runs acceptance-weighted updates on the Markov head.
pub struct DsparkTrainer {
    config: DsparkTrainConfig,
    store: TensorStore,
    tape: Tape,
    params: Option<MarkovParams>,
    optim: AdamW,
    baseline_ema: f32,
    running: Arc<AtomicBool>,
}

impl DsparkTrainer {
    /// Create a new trainer. Markov params are lazily initialized on the first
    /// `train_step` using the actual vocab size from the experience buffer,
    /// so the trainer is model-agnostic at construction time.
    pub fn new(config: DsparkTrainConfig) -> Result<Self> {
        let backend: Arc<dyn autograd::Backend> = Arc::new(CpuBackend);
        let store = TensorStore::with_backend(backend);
        let tape = Tape::new();
        let optim = AdamW::new(config.learning_rate, (0.9, 0.999), 1e-8, 0.0);

        Ok(Self {
            config,
            store,
            tape,
            params: None,
            optim,
            baseline_ema: 0.5,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Build the Markov head tensors with the given vocab size.
    fn init_params(&mut self, vocab_size: usize) -> Result<()> {
        let rank = self.config.markov_rank;

        let w1_data: Vec<f32> = (0..vocab_size * rank)
            .map(|i| {
                let s = (i % 1000) as f32;
                0.02 * (s * 0.1).sin()
            })
            .collect();
        let w1 = self
            .store
            .alloc(Tensor::new(w1_data, vec![vocab_size, rank], true)?);

        let w2_data: Vec<f32> = (0..rank * vocab_size)
            .map(|i| {
                let s = (i % 1000) as f32;
                0.02 * (s * 0.1).cos()
            })
            .collect();
        let w2 = self
            .store
            .alloc(Tensor::new(w2_data, vec![rank, vocab_size], true)?);

        self.params = Some(MarkovParams { w1, w2 });
        Ok(())
    }

    /// Get a handle to the running flag (for stopping the trainer).
    pub fn running_handle(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    /// Run one training step on a batch of experiences.
    pub fn train_step(&mut self, experiences: &[DsparkExperience]) -> Result<f32> {
        if experiences.is_empty() {
            return Ok(0.0);
        }

        let batch = experiences.len().min(self.config.batch_size);
        let block_size = experiences[0].block_size;
        let vocab_size = experiences[0].vocab_size;

        // Lazily build Markov params with the actual vocab size from the
        // first experience, so the trainer is model-agnostic at construction.
        if self.params.is_none() {
            self.init_params(vocab_size)?;
        }
        let params = self.params.as_ref().unwrap();

        let mut all_logits = Vec::with_capacity(batch * block_size * vocab_size);
        let mut all_target_logits = Vec::with_capacity(batch * block_size * vocab_size);
        let mut all_tokens_usize = Vec::with_capacity(batch * block_size);
        let mut rewards = Vec::with_capacity(batch);
        for exp in &experiences[..batch] {
            all_logits.extend_from_slice(&exp.draft_logits);
            all_target_logits.extend_from_slice(&exp.target_logits);
            all_tokens_usize.extend(exp.draft_tokens.iter().map(|&t| t as usize));
            rewards.push(exp.accepted as f32 / exp.block_size as f32);
        }

        let logits_id = self
            .store
            .from_slice(&all_logits, &[batch * block_size, vocab_size])?;
        let target_logits_id = self
            .store
            .from_slice(&all_target_logits, &[batch * block_size, vocab_size])?;

        // Markov bias: w2 @ w1[tokens]
        let emb_id = ops::embedding(
            params.w1,
            &all_tokens_usize,
            &mut self.store,
            &mut self.tape,
        )?;
        let emb_flat_id = ops::reshape(
            emb_id,
            &[batch * block_size, self.config.markov_rank],
            &mut self.store,
            &mut self.tape,
        )?;
        let bias_id = ops::matmul(emb_flat_id, params.w2, &mut self.store, &mut self.tape)?;
        let corrected_id = ops::add(logits_id, bias_id, &mut self.store, &mut self.tape)?;

        // Per-position exponential decay: position k in the block is weighted
        // exp(-k/gamma). Early tokens matter more — a mistake at position 0
        // voids the rest of the block.
        let pos_weights: Vec<f32> = match self.config.loss_decay_gamma {
            Some(gamma) if gamma > 0.0 => (0..block_size)
                .map(|k| (-(k as f32) / gamma).exp())
                .collect(),
            _ => vec![1.0; block_size],
        };
        // Expand to per-token weights [batch*block].
        let token_weights: Vec<f32> = (0..batch)
            .flat_map(|_| pos_weights.iter().copied())
            .collect();
        let weight_sum: f32 = token_weights.iter().sum();

        // ---- Policy-gradient loss (acceptance-weighted) ----
        let log_probs_id = ops::log_softmax(corrected_id, &mut self.store, &mut self.tape)?;
        let token_lp_id = ops::gather_last_dim(
            log_probs_id,
            &all_tokens_usize,
            &mut self.store,
            &mut self.tape,
        )?;

        let mean_reward: f32 = rewards.iter().sum::<f32>() / batch as f32;
        self.baseline_ema = (1.0 - self.config.baseline_ema_alpha) * self.baseline_ema
            + self.config.baseline_ema_alpha * mean_reward;
        let baseline = self.baseline_ema;
        // Per-token advantage × position weight.
        let weighted_adv: Vec<f32> = rewards
            .iter()
            .zip(token_weights.chunks(block_size))
            .flat_map(|(&r, w)| w.iter().map(move |&wk| (r - baseline) * wk))
            .collect();
        let adv_id = self
            .store
            .from_slice(&weighted_adv, &[batch * block_size])?;

        let pg_weighted_id = ops::mul(token_lp_id, adv_id, &mut self.store, &mut self.tape)?;
        let pg_neg_id = ops::mul_scalar(pg_weighted_id, -1.0, &mut self.store, &mut self.tape)?;
        // Weighted mean: divide by sum of position weights (not token count).
        let pg_loss_id = ops::mul_scalar(
            ops::sum(pg_neg_id, &mut self.store, &mut self.tape)?,
            1.0 / weight_sum,
            &mut self.store,
            &mut self.tape,
        )?;

        // ---- Supervised L1 loss on probability distributions ----
        // Directly optimises acceptance rate: accept ≈ 1 - 0.5·TV(draft, target).
        // We use squared difference (Frobenius) as a differentiable surrogate
        // for L1/TV — same gradient direction, no abs() op needed.
        let draft_probs_id = ops::softmax(corrected_id, &mut self.store, &mut self.tape)?;
        let target_probs_id = ops::softmax(target_logits_id, &mut self.store, &mut self.tape)?;
        // diff = draft - target; negate target on host then add.
        let neg_target_probs: Vec<f32> = all_target_logits.iter().map(|&x| -x).collect();
        let neg_target_id = self
            .store
            .from_slice(&neg_target_probs, &[batch * block_size, vocab_size])?;
        let diff_id = ops::add(
            draft_probs_id,
            neg_target_id,
            &mut self.store,
            &mut self.tape,
        )?;
        let sq_diff_id = ops::mul(diff_id, diff_id, &mut self.store, &mut self.tape)?;
        // Expand per-token weights to [batch*block, vocab] for element-wise mul.
        let expanded_weights: Vec<f32> = token_weights
            .iter()
            .flat_map(|&w| vec![w; vocab_size])
            .collect();
        let exp_weight_id = self
            .store
            .from_slice(&expanded_weights, &[batch * block_size, vocab_size])?;
        let weighted_sq_id = ops::mul(sq_diff_id, exp_weight_id, &mut self.store, &mut self.tape)?;
        let l1_loss_id = ops::mul_scalar(
            ops::sum(weighted_sq_id, &mut self.store, &mut self.tape)?,
            1.0 / (weight_sum * vocab_size as f32),
            &mut self.store,
            &mut self.tape,
        )?;

        // ---- Combined loss ----
        let pg_alpha = 1.0 - self.config.l1_loss_alpha;
        let pg_scaled_id = ops::mul_scalar(pg_loss_id, pg_alpha, &mut self.store, &mut self.tape)?;
        let l1_scaled_id = ops::mul_scalar(
            l1_loss_id,
            self.config.l1_loss_alpha,
            &mut self.store,
            &mut self.tape,
        )?;
        let loss_id = ops::add(pg_scaled_id, l1_scaled_id, &mut self.store, &mut self.tape)?;

        self.tape.backward(loss_id, &mut self.store)?;
        self.optim.step(&[params.w1, params.w2], &mut self.store);
        self.optim
            .zero_grad(&[params.w1, params.w2], &mut self.store);

        let loss_val = self.store.to_host(loss_id).unwrap_or_default();
        let loss = if loss_val.is_empty() {
            0.0
        } else {
            loss_val[0]
        };

        for id in [
            logits_id,
            target_logits_id,
            emb_id,
            emb_flat_id,
            bias_id,
            corrected_id,
            log_probs_id,
            token_lp_id,
            adv_id,
            pg_weighted_id,
            pg_neg_id,
            pg_loss_id,
            draft_probs_id,
            target_probs_id,
            neg_target_id,
            diff_id,
            sq_diff_id,
            exp_weight_id,
            weighted_sq_id,
            l1_loss_id,
            pg_scaled_id,
            l1_scaled_id,
            loss_id,
        ] {
            let _ = self.store.free(id);
        }
        self.tape = Tape::new();

        Ok(loss)
    }

    /// Get current Markov head weights as (w1 [vocab*rank], w2 [rank*vocab]).
    pub fn get_weights(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        let Some(params) = self.params.as_ref() else {
            anyhow::bail!("Markov params not yet initialized");
        };
        let w1 = self.store.to_host(params.w1)?;
        let w2 = self.store.to_host(params.w2)?;
        Ok((w1, w2))
    }

    /// Run the training loop. Blocks until `running` is set to false.
    pub fn run_loop(
        &mut self,
        source: &dyn ExperienceSource,
        on_weights: impl Fn(Vec<f32>, Vec<f32>) + Send,
    ) {
        self.running.store(true, Ordering::SeqCst);
        while self.running.load(Ordering::SeqCst) {
            let experiences = source.drain(self.config.batch_size);
            if experiences.is_empty() {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            match self.train_step(&experiences) {
                Ok(loss) => {
                    eprintln!(
                        "dspark_train: loss={loss:.4} baseline={:.4} n={}",
                        self.baseline_ema,
                        experiences.len()
                    );
                    if let Ok((w1, w2)) = self.get_weights() {
                        on_weights(w1, w2);
                    }
                }
                Err(e) => eprintln!("dspark_train: train step failed: {e}"),
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
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.buf.len()
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
/// Drains the global experience buffer populated by the CUDA inference hot
/// path, runs acceptance-weighted updates on the Markov head, and pushes updated weights
/// back into the running engine via [`LoadedInferenceEngine::update_dspark_markov_weights`].
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
    let source = InferCudaExperienceSource::new(buf);
    let mut trainer = DsparkTrainer::new(config)?;
    let running = trainer.running_handle();

    let engine_for_thread = Arc::clone(&engine);
    let join = std::thread::Builder::new()
        .name("dspark-train-sidecar".to_string())
        .spawn(move || {
            trainer.run_loop(&source, move |w1, w2| {
                if let Err(e) = engine_for_thread.update_dspark_markov_weights(&w1, &w2) {
                    eprintln!("dspark_train: weight update failed: {e}");
                }
            });
        })?;

    Ok(Some(DsparkTrainSidecarGuard {
        running,
        join: Some(join),
    }))
}
