use std::sync::{Mutex, PoisonError};

use anyhow::{Result, bail};
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::{DraftModel, TokenProposal};
use crate::backend::cuda::bootstrap::{InferenceEngineOptions, load_qwen3_components};
use crate::backend::cuda::tensor::{DeviceContext, DeviceVec};
use crate::model::{GenerationState, ModelForward, ModelRuntimeConfig, Qwen3Model};
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
            InferenceEngineOptions {
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

    pub fn draft_then_verify_with_target<M: ModelForward>(
        &self,
        target_model: &M,
        token_ids: &[u32],
        num_draft_tokens: usize,
    ) -> Result<TokenProposal> {
        let mut proposal = self.draft_batch(token_ids, num_draft_tokens)?;
        let mut target_scratch = target_model.create_state()?;
        target_model.forward_prefill(token_ids, &mut target_scratch)?;
        Self::fill_target_probs_from_state(&mut proposal, target_model, &mut target_scratch)?;
        Ok(proposal)
    }

    /// Fill target probabilities by advancing a scratch target state.
    ///
    /// The state is intentionally mutable because verifier logits for token
    /// `i+1` depend on feeding draft token `i`. Callers must pass a scratch or
    /// rollback-able state and commit accepted tokens separately.
    pub fn fill_target_probs_from_state<M: ModelForward>(
        proposal: &mut TokenProposal,
        target_model: &M,
        target_state: &mut M::State,
    ) -> Result<()> {
        proposal.target_probs.clear();
        proposal.target_probs.reserve(proposal.tokens.len());
        for i in 0..proposal.tokens.len() {
            let token = proposal.tokens[i];
            let logits = target_state
                .logits()
                .to_host(target_model.device_context())?;
            proposal
                .target_probs
                .push(target_token_prob_from_host_logits(token, &logits)?);
            target_model.forward_with_logits(&[token], target_state)?;
        }
        Ok(())
    }

    pub fn fill_target_probs_from_logits(
        proposal: &mut TokenProposal,
        target_logits: &DeviceVec,
        target_ctx: &DeviceContext,
    ) -> Result<usize> {
        let logits = target_logits.to_host(target_ctx)?;
        fill_target_probs_from_host_logits(proposal, &logits)
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

pub(super) fn fill_target_probs_from_host_logits(
    proposal: &mut TokenProposal,
    logits: &[f32],
) -> Result<usize> {
    if proposal.tokens.is_empty() {
        proposal.target_probs.clear();
        return Ok(0);
    }
    if proposal.tokens.len() > 1 {
        bail!("single-logits target probability fill only supports one draft token");
    }
    if logits.is_empty() {
        bail!("target logits must not be empty when draft tokens are present");
    }
    let vocab_size = logits.len();
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    proposal.target_probs = proposal
        .tokens
        .iter()
        .map(|&token| {
            target_token_prob_from_host_logits_with_norm(
                token, logits, max_logit, denom, vocab_size,
            )
        })
        .collect();
    Ok(argmax)
}

fn target_token_prob_from_host_logits(token: u32, logits: &[f32]) -> Result<f32> {
    if logits.is_empty() {
        bail!("target logits must not be empty when draft tokens are present");
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    Ok(target_token_prob_from_host_logits_with_norm(
        token,
        logits,
        max_logit,
        denom,
        logits.len(),
    ))
}

fn target_token_prob_from_host_logits_with_norm(
    token: u32,
    logits: &[f32],
    max_logit: f32,
    denom: f32,
    vocab_size: usize,
) -> f32 {
    let idx = token as usize;
    if idx >= vocab_size {
        0.0
    } else {
        (logits[idx] - max_logit).exp() / denom
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
            draft_probs.push(logprob.map_or(0.0, f32::exp));

            if model.is_stop_token(token) {
                break;
            }
            model.forward_decode(token, &mut state)?;
        }

        Ok(TokenProposal {
            target_probs: vec![0.0; tokens.len()],
            tokens,
            draft_probs,
            target_bonus_dist: Vec::new(),
        })
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
mod tests {
    use super::fill_target_probs_from_host_logits;
    use crate::speculative::TokenProposal;

    #[test]
    fn target_logits_fill_probs_and_report_argmax() {
        let mut proposal = TokenProposal {
            tokens: vec![2],
            draft_probs: vec![1.0],
            target_probs: Vec::new(),
            target_bonus_dist: Vec::new(),
        };

        let argmax = fill_target_probs_from_host_logits(&mut proposal, &[0.0, 1.0, 3.0]).unwrap();

        assert_eq!(argmax, 2);
        assert_eq!(proposal.target_probs.len(), 1);
        assert!(proposal.target_probs[0] > 0.0);
        proposal.validate().unwrap();
    }
}
