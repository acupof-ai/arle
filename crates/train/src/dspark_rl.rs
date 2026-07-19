//! DSpark RL sidecar trainer: REINFORCE on the acceptance reward.
//!
//! Drains the experience buffer populated by the inference hot path and runs
//! policy-gradient updates against the Markov head (the DSpark-specific
//! trainable component). The backbone transformer layers stay frozen; only
//! `markov_w1` (embedding) and `markov_w2` (linear projection) are updated.
//!
//! Reward = accepted_count / block_size (normalized [0, 1]).
//! Baseline = EMA of reward (stabilizes the gradient).
//! Loss = -log π(draft_tokens) * (reward - baseline).

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

/// Configuration for the DSpark RL trainer.
pub struct DsparkRlConfig {
    pub vocab_size: usize,
    pub markov_rank: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub baseline_ema_alpha: f32,
}

impl Default for DsparkRlConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            markov_rank: 256,
            learning_rate: 1e-4,
            batch_size: 64,
            baseline_ema_alpha: 0.01,
        }
    }
}

/// The trainable Markov head parameters.
struct MarkovParams {
    w1: TensorId, // [vocab, rank]
    w2: TensorId, // [rank, vocab]
}

/// DSpark RL trainer: runs in a background thread, drains the experience
/// buffer, and runs REINFORCE updates on the Markov head.
pub struct DsparkRlTrainer {
    config: DsparkRlConfig,
    store: TensorStore,
    tape: Tape,
    params: MarkovParams,
    optim: AdamW,
    baseline_ema: f32,
    running: Arc<AtomicBool>,
}

impl DsparkRlTrainer {
    /// Create a new trainer with randomly initialized Markov head.
    pub fn new(config: DsparkRlConfig) -> Result<Self> {
        let backend: Arc<dyn autograd::Backend> = Arc::new(CpuBackend);
        let mut store = TensorStore::with_backend(backend);
        let tape = Tape::new();

        let w1_data: Vec<f32> = (0..config.vocab_size * config.markov_rank)
            .map(|i| {
                let s = (i % 1000) as f32;
                0.02 * (s * 0.1).sin()
            })
            .collect();
        let w1 = store.alloc(Tensor::new(
            w1_data,
            vec![config.vocab_size, config.markov_rank],
            true,
        )?);

        let w2_data: Vec<f32> = (0..config.markov_rank * config.vocab_size)
            .map(|i| {
                let s = (i % 1000) as f32;
                0.02 * (s * 0.1).cos()
            })
            .collect();
        let w2 = store.alloc(Tensor::new(
            w2_data,
            vec![config.markov_rank, config.vocab_size],
            true,
        )?);

        let optim = AdamW::new(config.learning_rate, (0.9, 0.999), 1e-8, 0.0);

        Ok(Self {
            config,
            store,
            tape,
            params: MarkovParams { w1, w2 },
            optim,
            baseline_ema: 0.5,
            running: Arc::new(AtomicBool::new(false)),
        })
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

        let mut all_logits = Vec::with_capacity(batch * block_size * vocab_size);
        let mut all_tokens_usize = Vec::with_capacity(batch * block_size);
        let mut rewards = Vec::with_capacity(batch);
        for exp in &experiences[..batch] {
            all_logits.extend_from_slice(&exp.draft_logits);
            all_tokens_usize.extend(exp.draft_tokens.iter().map(|&t| t as usize));
            rewards.push(exp.accepted as f32 / exp.block_size as f32);
        }

        let logits_id = self
            .store
            .from_slice(&all_logits, &[batch * block_size, vocab_size])?;

        // Markov bias: w2 @ w1[tokens]
        // embedding output: [1, batch*block, rank] (rank 3); reshape to [batch*block, rank]
        let emb_id = ops::embedding(
            self.params.w1,
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
        let bias_id = ops::matmul(emb_flat_id, self.params.w2, &mut self.store, &mut self.tape)?;
        let corrected_id = ops::add(logits_id, bias_id, &mut self.store, &mut self.tape)?;
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
        // Expand per-experience advantage to per-token: each token in a block
        // shares the same reward signal.
        let advantages: Vec<f32> = rewards
            .iter()
            .flat_map(|&r| vec![r - self.baseline_ema; block_size])
            .collect();
        let adv_id = self.store.from_slice(&advantages, &[batch * block_size])?;

        let weighted_id = ops::mul(token_lp_id, adv_id, &mut self.store, &mut self.tape)?;
        let neg_id = ops::mul_scalar(weighted_id, -1.0, &mut self.store, &mut self.tape)?;
        let loss_id = ops::mean(neg_id, &mut self.store, &mut self.tape)?;

        self.tape.backward(loss_id, &mut self.store)?;
        self.optim
            .step(&[self.params.w1, self.params.w2], &mut self.store);
        self.optim
            .zero_grad(&[self.params.w1, self.params.w2], &mut self.store);

        let loss_val = self.store.to_host(loss_id).unwrap_or_default();
        let loss = if loss_val.is_empty() {
            0.0
        } else {
            loss_val[0]
        };

        for id in [
            logits_id,
            emb_id,
            emb_flat_id,
            bias_id,
            corrected_id,
            log_probs_id,
            token_lp_id,
            adv_id,
            weighted_id,
            neg_id,
            loss_id,
        ] {
            let _ = self.store.free(id);
        }
        self.tape = Tape::new();

        Ok(loss)
    }

    /// Get current Markov head weights as (w1 [vocab*rank], w2 [rank*vocab]).
    pub fn get_weights(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        let w1 = self.store.to_host(self.params.w1)?;
        let w2 = self.store.to_host(self.params.w2)?;
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
                        "dspark_rl: loss={loss:.4} baseline={:.4} n={}",
                        self.baseline_ema,
                        experiences.len()
                    );
                    if let Ok((w1, w2)) = self.get_weights() {
                        on_weights(w1, w2);
                    }
                }
                Err(e) => eprintln!("dspark_rl: train step failed: {e}"),
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
