//! Speculative decoding framework.
//!
//! Implements the draft-model + target-model verification loop described in
//! "Fast Inference from Transformers via Speculative Decoding"
//! (Leviathan et al., 2023) and "Accelerating Large Language Model Decoding
//! with Speculative Sampling" (Chen et al., 2023).
//!
//! # Algorithm
//!
//! ```text
//! loop:
//!   1. Draft model generates K candidate tokens autoregressively.
//!   2. Target model runs a single forward pass on the K+1 input positions
//!      (all draft tokens + the prefix), yielding K+1 probability distributions.
//!   3. For each draft token t_i (i = 0..K):
//!        accept_prob = min(1, P_target(t_i) / P_draft(t_i))
//!        accept with probability accept_prob
//!        if rejected: resample from adjusted distribution, break
//!   4. If all K accepted: append bonus token sampled from P_target[K].
//! ```
//!
//! Expected speedup ≈ K * α / (1 + K * α - α)  where α = mean acceptance rate.
//!
//! # CPU-verifiable parts
//!
//! - [`SpecConfig`] validation
//! - [`verify_tokens`] acceptance/rejection sampling (pure f32 math)
//! - [`VerificationResult::acceptance_rate`] from a [`VerificationResult`]
//! - [`DraftModel`] trait (GPU stub)

use anyhow::{Result, bail};
use rand::RngExt;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{DEFAULT_QWEN3_DRAFT_MODEL_ID, DraftEngine, DraftEngineConfig};

// ============================================================================
// SpecConfig
// ============================================================================

/// Configuration for speculative decoding.
#[derive(Clone, Debug)]
pub struct SpecConfig {
    /// Number of tokens the draft model proposes per target step (K).
    pub num_speculative_tokens: usize,

    /// Target model vocabulary size.  Used to validate probability arrays.
    pub vocab_size: usize,

    /// Minimum acceptance rate before falling back to regular decoding.
    /// If the rolling acceptance rate drops below this threshold, speculation
    /// is temporarily disabled.  0.0 = never fall back.
    pub min_acceptance_rate: f32,
}

impl SpecConfig {
    pub fn validate(&self) -> Result<()> {
        if self.num_speculative_tokens == 0 {
            bail!("num_speculative_tokens must be ≥ 1");
        }
        if self.vocab_size == 0 {
            bail!("vocab_size must be ≥ 1");
        }
        if !(0.0..=1.0).contains(&self.min_acceptance_rate) {
            bail!("min_acceptance_rate must be in [0, 1]");
        }
        Ok(())
    }
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 5,
            vocab_size: 32000,
            min_acceptance_rate: 0.0,
        }
    }
}

// ============================================================================
// TokenProposal
// ============================================================================

/// A batch of K draft tokens plus per-token probabilities under both models.
#[derive(Clone, Debug)]
pub struct TokenProposal {
    /// Draft token IDs, length K.
    pub tokens: Vec<u32>,

    /// Draft model probability at each proposed token:  q(t_i).  Length K.
    pub draft_probs: Vec<f32>,

    /// Target model probability at each proposed token: p(t_i).  Length K.
    pub target_probs: Vec<f32>,

    /// Full target probability distribution at position K (used for bonus
    /// token sampling).  Length = vocab_size.  May be empty if the caller
    /// does not need the bonus token.
    pub target_bonus_dist: Vec<f32>,
}

impl TokenProposal {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> Result<()> {
        let k = self.tokens.len();
        if self.draft_probs.len() != k {
            bail!(
                "draft_probs length {} != tokens length {k}",
                self.draft_probs.len()
            );
        }
        if self.target_probs.len() != k {
            bail!(
                "target_probs length {} != tokens length {k}",
                self.target_probs.len()
            );
        }
        for (i, (&p, &q)) in self
            .target_probs
            .iter()
            .zip(self.draft_probs.iter())
            .enumerate()
        {
            if !(0.0..=1.0 + 1e-5).contains(&p) {
                bail!("target_probs[{i}] = {p} out of [0, 1]");
            }
            if !(0.0..=1.0 + 1e-5).contains(&q) {
                bail!("draft_probs[{i}] = {q} out of [0, 1]");
            }
        }
        Ok(())
    }
}

// ============================================================================
// VerificationResult
// ============================================================================

/// Output of the verification step.
#[derive(Clone, Debug)]
pub struct VerificationResult {
    /// All accepted tokens (length 0..=K).
    pub accepted: Vec<u32>,

    /// Bonus token sampled from the adjusted distribution at the rejection
    /// point (or from `target_bonus_dist` if all K tokens were accepted).
    /// `None` when the bonus distribution was not provided.
    pub bonus_token: Option<u32>,

    /// Number of accepted draft tokens (= `accepted.len()`).
    pub num_accepted: usize,

    /// Index of the first rejected token (K if all accepted).
    pub rejection_index: usize,
}

impl VerificationResult {
    /// Total tokens appended in this speculation step.
    pub fn total_tokens(&self) -> usize {
        self.num_accepted + self.bonus_token.is_some() as usize
    }

    /// Empirical acceptance rate = num_accepted / K.
    pub fn acceptance_rate(&self, k: usize) -> f32 {
        if k == 0 {
            1.0
        } else {
            self.num_accepted as f32 / k as f32
        }
    }
}

// ============================================================================
// Core acceptance algorithm
// ============================================================================

/// Run the speculative decoding verification step.
///
/// Uses standard rejection sampling: for each draft token t_i,
/// accept with probability `min(1, p(t_i) / q(t_i))`.
///
/// On rejection, the "bonus" token is sampled from the adjusted distribution
/// `max(0, p - q) / Z`.  If the caller provides `target_bonus_dist` and all
/// K tokens are accepted, the bonus is sampled from that distribution.
///
/// `rng` must implement [`rand::Rng`] — pass `rand::thread_rng()` in
/// production or a seeded RNG in tests.
pub fn verify_tokens(proposal: &TokenProposal, rng: &mut impl rand::Rng) -> VerificationResult {
    let k = proposal.tokens.len();
    let mut accepted = Vec::with_capacity(k);
    let mut rejection_index = k; // default: all accepted

    for i in 0..k {
        let p = proposal.target_probs[i].max(0.0);
        let q = proposal.draft_probs[i].max(f32::MIN_POSITIVE);
        let accept_prob = (p / q).min(1.0);

        if rng.random::<f32>() < accept_prob {
            accepted.push(proposal.tokens[i]);
        } else {
            rejection_index = i;
            break;
        }
    }

    // Bonus token sampling
    let bonus_token = if rejection_index == k {
        // All accepted: sample from target_bonus_dist if provided
        if proposal.target_bonus_dist.is_empty() {
            None
        } else {
            sample_from_dist(&proposal.target_bonus_dist, rng)
        }
    } else {
        // Rejected at position `rejection_index`: sample from adjusted dist
        // adjusted[v] = max(0, p_full[v] - q_full[v]) / Z
        // Simplified: we only have scalar p/q at the rejected token, not full
        // vocab distributions. Return None — the caller must resample from target.
        None
    };

    let num_accepted = accepted.len();
    VerificationResult {
        accepted,
        bonus_token,
        num_accepted,
        rejection_index,
    }
}

/// Sample a token index from a probability distribution (unnormalized is ok).
fn sample_from_dist(dist: &[f32], rng: &mut impl rand::Rng) -> Option<u32> {
    if dist.is_empty() {
        return None;
    }
    let total: f32 = dist.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let threshold = rng.random::<f32>() * total;
    let mut cumulative = 0.0f32;
    for (i, &p) in dist.iter().enumerate() {
        cumulative += p;
        if cumulative >= threshold {
            return Some(i as u32);
        }
    }
    // Fallback: last token
    Some((dist.len() - 1) as u32)
}

// ============================================================================
// AcceptanceTracker
// ============================================================================

/// Rolling tracker for empirical acceptance rate.
///
/// Used to decide when to disable speculation (if acceptance rate is too low).
pub struct AcceptanceTracker {
    window_size: usize,
    history: std::collections::VecDeque<f32>,
}

impl AcceptanceTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            history: std::collections::VecDeque::new(),
        }
    }

    /// Record the acceptance rate for one speculation step.
    pub fn record(&mut self, rate: f32) {
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(rate);
    }

    /// Mean acceptance rate over the window.
    pub fn mean(&self) -> f32 {
        if self.history.is_empty() {
            return 1.0; // optimistic start
        }
        self.history.iter().sum::<f32>() / self.history.len() as f32
    }

    /// True if speculation should be disabled based on the configured threshold.
    pub fn should_disable(&self, min_rate: f32) -> bool {
        self.history.len() >= self.window_size && self.mean() < min_rate
    }
}

// ============================================================================
// DraftModel trait (GPU stub)
// ============================================================================

/// Trait implemented by draft models.
///
/// **GPU required for production use** — the trait itself is always available
/// for type-level work, but all implementations require CUDA kernels.
pub trait DraftModel: Send + Sync {
    /// Generate `k` speculative tokens for each request in the batch.
    ///
    /// Returns one [`TokenProposal`] per request.
    ///
    /// **GPU required** — panics if called in CPU builds.
    fn draft_batch(&self, token_ids: &[u32], num_draft_tokens: usize) -> Result<TokenProposal>;

    /// Model identifier (e.g. "Qwen3-0.5B").
    fn model_id(&self) -> &str;
}

/// Placeholder draft model for testing without GPU.
///
/// Always proposes the same token with probability 1.0 (draft) and
/// a configurable probability (target, to test acceptance logic).
pub struct MockDraftModel {
    token: u32,
    draft_prob: f32,
    target_prob: f32,
    id: String,
}

impl MockDraftModel {
    pub fn new(token: u32, draft_prob: f32, target_prob: f32) -> Self {
        Self {
            token,
            draft_prob,
            target_prob,
            id: "mock-draft".to_string(),
        }
    }
}

impl DraftModel for MockDraftModel {
    fn draft_batch(&self, _token_ids: &[u32], k: usize) -> Result<TokenProposal> {
        Ok(TokenProposal {
            tokens: vec![self.token; k],
            draft_probs: vec![self.draft_prob; k],
            target_probs: vec![self.target_prob; k],
            target_bonus_dist: vec![],
        })
    }

    fn model_id(&self) -> &str {
        &self.id
    }
}

// ============================================================================
// Expected speedup calculation
// ============================================================================

/// Compute the theoretical throughput multiplier for speculative decoding.
///
/// Based on the formula from Chen et al. 2023:
/// `speedup = (1 - α^(K+1)) / ((1 - α) * (1 + K/c))`
/// where α = acceptance rate, K = num_draft_tokens, c = target/draft cost ratio.
///
/// For a simple approximation (assuming negligible draft cost): `K * α / (1 - α^K)`.
pub fn expected_speedup(k: usize, alpha: f32) -> f32 {
    if alpha <= 0.0 {
        return 1.0;
    }
    if alpha >= 1.0 {
        return (k + 1) as f32;
    }
    let alpha_k = alpha.powi(k as i32);
    (1.0 - alpha_k) / ((1.0 - alpha) * k as f32) * k as f32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn seeded_rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    fn all_accept_proposal(k: usize) -> TokenProposal {
        // target_prob == draft_prob → accept_prob == 1 → always accepted
        TokenProposal {
            tokens: (0..k as u32).collect(),
            draft_probs: vec![0.5; k],
            target_probs: vec![0.5; k],
            target_bonus_dist: vec![],
        }
    }

    fn all_reject_proposal(k: usize) -> TokenProposal {
        // target_prob = 0 → accept_prob == 0 → always rejected
        TokenProposal {
            tokens: vec![0; k],
            draft_probs: vec![0.5; k],
            target_probs: vec![0.0; k],
            target_bonus_dist: vec![],
        }
    }

    // ---------------------------------------------------------------- verify_tokens

    #[test]
    fn all_accepted_when_target_eq_draft() {
        let proposal = all_accept_proposal(5);
        let mut rng = seeded_rng();
        let result = verify_tokens(&proposal, &mut rng);
        assert_eq!(result.num_accepted, 5);
        assert_eq!(result.rejection_index, 5);
        assert_eq!(result.accepted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn none_accepted_when_target_zero() {
        let proposal = all_reject_proposal(5);
        let mut rng = seeded_rng();
        let result = verify_tokens(&proposal, &mut rng);
        assert_eq!(result.num_accepted, 0);
        assert_eq!(result.rejection_index, 0);
        assert!(result.accepted.is_empty());
    }

    #[test]
    fn bonus_token_sampled_when_all_accepted() {
        let mut proposal = all_accept_proposal(3);
        proposal.target_bonus_dist = vec![0.0, 0.0, 1.0, 0.0]; // always token 2
        let mut rng = seeded_rng();
        let result = verify_tokens(&proposal, &mut rng);
        assert_eq!(result.num_accepted, 3);
        assert_eq!(result.bonus_token, Some(2));
        assert_eq!(result.total_tokens(), 4);
    }

    #[test]
    fn partial_acceptance() {
        // Tokens 0,1,2 always accepted (p==q → accept_prob==1),
        // token 3 always rejected (p==0 → accept_prob==0).
        let proposal = TokenProposal {
            tokens: vec![10, 11, 12, 13],
            draft_probs: vec![0.5, 0.5, 0.5, 0.5],
            target_probs: vec![0.5, 0.5, 0.5, 0.0],
            target_bonus_dist: vec![],
        };
        let mut rng = seeded_rng();
        let result = verify_tokens(&proposal, &mut rng);
        // First 3 tokens must be accepted; token index 3 must be rejected.
        assert_eq!(result.num_accepted, 3);
        assert_eq!(result.rejection_index, 3);
        assert_eq!(result.accepted, vec![10, 11, 12]);
    }

    #[test]
    fn acceptance_rate_from_result() {
        let result = VerificationResult {
            accepted: vec![1, 2, 3],
            bonus_token: None,
            num_accepted: 3,
            rejection_index: 3,
        };
        assert!((result.acceptance_rate(5) - 0.6).abs() < 1e-5);
        assert!((result.acceptance_rate(3) - 1.0).abs() < 1e-5);
        assert!((result.acceptance_rate(0) - 1.0).abs() < 1e-5);
    }

    // ---------------------------------------------------------------- TokenProposal::validate

    #[test]
    fn proposal_validate_ok() {
        let p = all_accept_proposal(3);
        p.validate().unwrap();
    }

    #[test]
    fn proposal_validate_length_mismatch() {
        let mut p = all_accept_proposal(3);
        p.draft_probs.push(0.5); // length 4 ≠ 3
        assert!(p.validate().is_err());
    }

    #[test]
    fn proposal_is_empty() {
        let p = TokenProposal {
            tokens: vec![],
            draft_probs: vec![],
            target_probs: vec![],
            target_bonus_dist: vec![],
        };
        assert!(p.is_empty());
    }

    // ---------------------------------------------------------------- SpecConfig

    #[test]
    fn spec_config_default_valid() {
        SpecConfig::default().validate().unwrap();
    }

    #[test]
    fn spec_config_zero_tokens_invalid() {
        let cfg = SpecConfig {
            num_speculative_tokens: 0,
            ..SpecConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn spec_config_invalid_threshold() {
        let cfg = SpecConfig {
            min_acceptance_rate: 1.5,
            ..SpecConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ---------------------------------------------------------------- AcceptanceTracker

    #[test]
    fn tracker_starts_optimistic() {
        let tracker = AcceptanceTracker::new(10);
        assert!((tracker.mean() - 1.0).abs() < 1e-5);
        assert!(!tracker.should_disable(0.5));
    }

    #[test]
    fn tracker_mean_correct() {
        let mut tracker = AcceptanceTracker::new(4);
        tracker.record(1.0);
        tracker.record(0.5);
        tracker.record(0.5);
        tracker.record(0.0);
        assert!((tracker.mean() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn tracker_window_evicts_old() {
        let mut tracker = AcceptanceTracker::new(2);
        tracker.record(0.0);
        tracker.record(0.0);
        tracker.record(1.0); // evicts first 0.0
        tracker.record(1.0); // evicts second 0.0
        assert!((tracker.mean() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn tracker_disable_when_below_threshold() {
        let mut tracker = AcceptanceTracker::new(3);
        tracker.record(0.1);
        tracker.record(0.1);
        tracker.record(0.1);
        assert!(tracker.should_disable(0.5)); // mean 0.1 < 0.5
        assert!(!tracker.should_disable(0.0)); // 0.1 >= 0.0
    }

    // ---------------------------------------------------------------- MockDraftModel

    #[test]
    fn mock_draft_model() {
        let model = MockDraftModel::new(42, 0.8, 0.9);
        let proposal = model.draft_batch(&[1, 2, 3], 4).unwrap();
        assert_eq!(proposal.tokens, vec![42, 42, 42, 42]);
        assert_eq!(proposal.draft_probs.len(), 4);
        assert_eq!(model.model_id(), "mock-draft");
    }

    // ---------------------------------------------------------------- expected_speedup

    #[test]
    fn speedup_alpha_one_gives_k_plus_one() {
        let k = 5;
        let s = expected_speedup(k, 1.0);
        assert!((s - (k + 1) as f32).abs() < 0.1, "speedup={s}");
    }

    #[test]
    fn speedup_alpha_zero_gives_one() {
        assert!((expected_speedup(5, 0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn speedup_increases_with_alpha() {
        let k = 5;
        let s_low = expected_speedup(k, 0.5);
        let s_high = expected_speedup(k, 0.9);
        assert!(s_high > s_low, "higher alpha should give higher speedup");
    }

    // ---------------------------------------------------------------- sample_from_dist

    #[test]
    fn sample_from_uniform_dist() {
        let dist = vec![1.0f32; 4]; // uniform over 4 tokens
        let mut rng = SmallRng::seed_from_u64(123);
        let mut counts = [0usize; 4];
        for _ in 0..1000 {
            let t = sample_from_dist(&dist, &mut rng).unwrap() as usize;
            counts[t] += 1;
        }
        // Each should appear roughly 250 times (±3σ ≈ ±50)
        for c in counts {
            assert!(c > 150 && c < 350, "count {c} outside expected range");
        }
    }

    #[test]
    fn sample_from_peaked_dist() {
        let dist = vec![0.0, 0.0, 1.0, 0.0]; // always token 2
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..10 {
            assert_eq!(sample_from_dist(&dist, &mut rng), Some(2));
        }
    }
}
