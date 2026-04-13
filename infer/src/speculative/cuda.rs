use std::sync::{Mutex, PoisonError};

use anyhow::{Result, bail};
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::{DraftModel, TokenProposal};
use crate::backend::cuda::bootstrap::{EngineOptions, load_qwen3_components};
use crate::model::{ModelForward, ModelRuntimeConfig, Qwen3Model};
use crate::sampler::SamplingParams;

/// Default draft-model identifier for the first CUDA implementation pass.
pub const DEFAULT_QWEN3_DRAFT_MODEL_ID: &str = "Qwen/Qwen3-0.5B";

/// Runtime configuration for [`DraftEngine`].
///
/// The initial CUDA skeleton is intentionally narrow:
/// - Qwen3-only
/// - greedy-only draft sampling
/// - stateless per-call prefill/decode, no persistent per-request KV yet
#[derive(Clone, Debug)]
pub struct DraftEngineConfig {
    pub model_path: String,
    pub runtime: ModelRuntimeConfig,
    pub sampling: SamplingParams,
}

impl DraftEngineConfig {
    pub fn qwen3_0_5b() -> Self {
        Self {
            model_path: DEFAULT_QWEN3_DRAFT_MODEL_ID.to_string(),
            runtime: ModelRuntimeConfig::default(),
            sampling: SamplingParams::default(),
        }
    }
}

impl Default for DraftEngineConfig {
    fn default() -> Self {
        Self::qwen3_0_5b()
    }
}

/// CUDA draft-model wrapper around a small Qwen3 checkpoint.
///
/// This is a phase-0 skeleton for speculative decoding:
/// - loads a second CUDA model (expected: Qwen3-0.5B)
/// - generates K greedy draft tokens for a single prefix
/// - reports draft-token probabilities when the greedy logprob fast path exists
///
/// It deliberately does **not** keep request-local draft KV across calls yet.
/// `SpeculativeScheduler` will need a dedicated per-request draft state before
/// this becomes production-usable.
pub struct DraftEngine {
    model_id: String,
    vocab_size: usize,
    sampling: SamplingParams,
    model: Mutex<Qwen3Model>,
}

impl DraftEngine {
    pub fn load(config: DraftEngineConfig) -> Result<Self> {
        Self::validate_sampling(&config.sampling)?;

        let components = load_qwen3_components(
            &config.model_path,
            EngineOptions {
                enable_cuda_graph: config.runtime.enable_cuda_graph,
            },
        )?;
        let vocab_size = components.tokenizer.vocab_size();

        Ok(Self {
            model_id: components.model_id,
            vocab_size,
            sampling: config.sampling,
            model: Mutex::new(components.model),
        })
    }

    pub fn load_qwen3(model_path: &str) -> Result<Self> {
        Self::load(DraftEngineConfig {
            model_path: model_path.to_string(),
            ..DraftEngineConfig::default()
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn sampling(&self) -> &SamplingParams {
        &self.sampling
    }

    fn validate_sampling(sampling: &SamplingParams) -> Result<()> {
        if !sampling.is_greedy() || sampling.has_penalties() {
            bail!(
                "DraftEngine skeleton currently supports greedy, penalty-free draft sampling only"
            );
        }
        Ok(())
    }
}

impl DraftModel for DraftEngine {
    fn draft_batch(&self, token_ids: &[u32], num_draft_tokens: usize) -> Result<TokenProposal> {
        Self::validate_sampling(&self.sampling)?;

        if token_ids.is_empty() {
            bail!("DraftEngine requires at least one prefix token until BOS bootstrap is wired");
        }
        if num_draft_tokens == 0 {
            return Ok(TokenProposal {
                tokens: Vec::new(),
                draft_probs: Vec::new(),
                target_probs: Vec::new(),
                target_bonus_dist: Vec::new(),
            });
        }

        let model = self.model.lock().unwrap_or_else(PoisonError::into_inner);
        let mut state = model.create_state()?;
        model.forward_prefill(token_ids, &mut state)?;

        let mut rng = StdRng::seed_from_u64(42);
        let mut tokens = Vec::with_capacity(num_draft_tokens);
        let mut draft_probs = Vec::with_capacity(num_draft_tokens);

        for _ in 0..num_draft_tokens {
            let (token, logprob) =
                model.select_token_with_logprob(&mut state, &self.sampling, &mut rng)?;
            tokens.push(token);
            draft_probs.push(logprob.map(|lp| lp.exp()).unwrap_or(0.0));

            if model.is_stop_token(token) {
                break;
            }
            model.forward_decode(token, &mut state)?;
        }

        // `target_probs` / `target_bonus_dist` are owned by the target-model
        // verify pass. Keep them zero/empty so this skeleton can compile and be
        // instantiated before the scheduler wires in verification.
        let target_probs = vec![0.0; tokens.len()];
        Ok(TokenProposal {
            tokens,
            draft_probs,
            target_probs,
            target_bonus_dist: Vec::new(),
        })
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}
