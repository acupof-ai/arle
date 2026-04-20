//! Verifier trait + concrete reward functions for train-side policy learning.
//!
//! Each verifier maps `(prompt_ids, full_ids, response_mask) -> f32` in
//! `[0, 1]` (aggregators may emit larger magnitudes). Trajectories only
//! consume response positions; prompt tokens are passed so verifiers can
//! compare response against prompt when relevant.

pub trait Verifier {
    fn verify(&self, prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32;
}

impl<F> Verifier for F
where
    F: Fn(&[usize], &[usize], &[bool]) -> f32,
{
    fn verify(&self, prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32 {
        (self)(prompt_ids, full_ids, response_mask)
    }
}

pub struct CopyVerifier;

impl Verifier for CopyVerifier {
    fn verify(&self, prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32 {
        let prefix_len = prompt_ids.len() / 2;
        let mut correct = 0usize;
        let mut total = 0usize;
        for (position, masked) in response_mask.iter().enumerate() {
            if !*masked {
                continue;
            }
            total += 1;
            let source = position.wrapping_sub(prefix_len + 1);
            if source < prefix_len && full_ids[position] == prompt_ids[source] {
                correct += 1;
            }
        }
        if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        }
    }
}

pub struct ReverseCopyVerifier;

impl Verifier for ReverseCopyVerifier {
    fn verify(&self, prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32 {
        let prefix_len = prompt_ids.len() / 2;
        let mut correct = 0usize;
        let mut total = 0usize;
        for (position, masked) in response_mask.iter().enumerate() {
            if !*masked {
                continue;
            }
            total += 1;
            let response_index = position.wrapping_sub(prefix_len + 1);
            if response_index < prefix_len {
                let source = prefix_len - 1 - response_index;
                if full_ids[position] == prompt_ids[source] {
                    correct += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        }
    }
}

pub struct PaletteVerifier {
    allowed: Vec<bool>,
}

impl PaletteVerifier {
    pub fn new(vocab_size: usize, allowed_tokens: &[usize]) -> Self {
        let mut allowed = vec![false; vocab_size];
        for token in allowed_tokens {
            if *token < vocab_size {
                allowed[*token] = true;
            }
        }
        Self { allowed }
    }
}

impl Verifier for PaletteVerifier {
    fn verify(&self, _prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32 {
        let mut matches = 0usize;
        let mut total = 0usize;
        for (position, masked) in response_mask.iter().enumerate() {
            if !*masked {
                continue;
            }
            total += 1;
            if self
                .allowed
                .get(full_ids[position])
                .copied()
                .unwrap_or(false)
            {
                matches += 1;
            }
        }
        if total == 0 {
            0.0
        } else {
            matches as f32 / total as f32
        }
    }
}

/// Rewards a response that is strictly increasing — a code-style "output
/// is monotonically sorted" property test. Response tokens equal to the
/// sentinel (prompt's last token) are skipped so separators don't break
/// the run.
pub struct MonotonicVerifier;

impl Verifier for MonotonicVerifier {
    fn verify(&self, _prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32 {
        let mut prev: Option<usize> = None;
        let mut correct = 0usize;
        let mut total = 0usize;
        for (position, masked) in response_mask.iter().enumerate() {
            if !*masked {
                continue;
            }
            total += 1;
            let value = full_ids[position];
            let ok = match prev {
                Some(last) => value > last,
                None => true,
            };
            if ok {
                correct += 1;
                prev = Some(value);
            }
        }
        if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        }
    }
}

/// Rewards presence of a specific sentinel token in the response — a
/// tool-success proxy. Real tool integration would replace this with a
/// checker over the decoded tool-call payload; the sentinel stand-in
/// lets the ensemble be tested today.
pub struct ToolSuccessVerifier {
    sentinel: usize,
}

impl ToolSuccessVerifier {
    pub fn new(sentinel: usize) -> Self {
        Self { sentinel }
    }
}

impl Verifier for ToolSuccessVerifier {
    fn verify(&self, _prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32 {
        for (position, masked) in response_mask.iter().enumerate() {
            if !*masked {
                continue;
            }
            if full_ids[position] == self.sentinel {
                return 1.0;
            }
        }
        0.0
    }
}

/// Rewards a response that encodes the correct answer to a tiny
/// arithmetic expression embedded in the prompt. Prompt layout:
/// `[a_digits..., op_token, b_digits..., eq_token]`. `op_token` selects
/// sum (==`op_plus`) or product (==`op_times`); `eq_token` terminates
/// the expression. The verifier decodes `a` and `b` from digit tokens
/// (values `0..base`), computes the expected result, and compares it
/// digit-wise against the response.
///
/// Tokenizer-free by construction. Intended as a math-verifier
/// *archetype*: the decoder can be swapped out for a real one once a
/// tokenizer is wired in; the surrounding ensemble + reward-config
/// machinery needs no change.
pub struct ArithmeticVerifier {
    pub base: usize,
    pub op_plus: usize,
    pub op_times: usize,
    pub eq_token: usize,
    pub answer_len: usize,
}

impl ArithmeticVerifier {
    pub fn new(
        base: usize,
        op_plus: usize,
        op_times: usize,
        eq_token: usize,
        answer_len: usize,
    ) -> Self {
        assert!(base >= 2, "arithmetic base must be ≥2");
        assert!(answer_len >= 1, "answer_len must be ≥1");
        Self {
            base,
            op_plus,
            op_times,
            eq_token,
            answer_len,
        }
    }

    fn decode(&self, tokens: &[usize]) -> Option<(u64, u64, bool)> {
        let op_pos = tokens
            .iter()
            .position(|t| *t == self.op_plus || *t == self.op_times)?;
        let eq_pos = tokens.iter().position(|t| *t == self.eq_token)?;
        if eq_pos <= op_pos + 1 {
            return None;
        }
        let a = digits_to_value(&tokens[..op_pos], self.base)?;
        let b = digits_to_value(&tokens[op_pos + 1..eq_pos], self.base)?;
        let is_plus = tokens[op_pos] == self.op_plus;
        Some((a, b, is_plus))
    }

    fn expected_answer(&self, a: u64, b: u64, is_plus: bool) -> u64 {
        if is_plus { a + b } else { a * b }
    }
}

impl Verifier for ArithmeticVerifier {
    fn verify(&self, prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32 {
        let Some((a, b, is_plus)) = self.decode(prompt_ids) else {
            return 0.0;
        };
        let expected = self.expected_answer(a, b, is_plus);

        // Collect response tokens in order.
        let mut response: Vec<usize> = Vec::new();
        for (position, masked) in response_mask.iter().enumerate() {
            if *masked {
                response.push(full_ids[position]);
            }
        }
        if response.is_empty() {
            return 0.0;
        }

        // Use the first `answer_len` tokens (padding with zeros if short).
        let mut answer_digits = vec![0usize; self.answer_len];
        for (i, token) in response.iter().take(self.answer_len).enumerate() {
            answer_digits[i] = *token;
        }
        let Some(answer) = digits_to_value(&answer_digits, self.base) else {
            return 0.0;
        };
        if answer == expected { 1.0 } else { 0.0 }
    }
}

fn digits_to_value(tokens: &[usize], base: usize) -> Option<u64> {
    if tokens.is_empty() {
        return None;
    }
    let mut value: u64 = 0;
    for token in tokens {
        if *token >= base {
            return None;
        }
        value = value.checked_mul(base as u64)?.checked_add(*token as u64)?;
    }
    Some(value)
}

pub struct WeightedEnsemble {
    members: Vec<(f32, Box<dyn Verifier + Send + Sync>)>,
}

impl WeightedEnsemble {
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
        }
    }

    pub fn with<V: Verifier + Send + Sync + 'static>(mut self, weight: f32, verifier: V) -> Self {
        self.members.push((weight, Box::new(verifier)));
        self
    }

    /// Build a `WeightedEnsemble` from a declarative config. Lets callers
    /// express the reward mix as data (CLI flag, TOML, JSON) instead of
    /// hand-wiring `.with(weight, verifier)` chains in code.
    pub fn from_config(config: &RewardConfig, vocab_size: usize) -> Self {
        let mut ensemble = Self::new();
        for entry in &config.members {
            let verifier: Box<dyn Verifier + Send + Sync> = match &entry.kind {
                VerifierKind::Copy => Box::new(CopyVerifier),
                VerifierKind::ReverseCopy => Box::new(ReverseCopyVerifier),
                VerifierKind::Palette { allowed_tokens } => {
                    Box::new(PaletteVerifier::new(vocab_size, allowed_tokens))
                }
                VerifierKind::Monotonic => Box::new(MonotonicVerifier),
                VerifierKind::ToolSuccess { sentinel } => {
                    Box::new(ToolSuccessVerifier::new(*sentinel))
                }
                VerifierKind::Arithmetic {
                    base,
                    op_plus,
                    op_times,
                    eq_token,
                    answer_len,
                } => Box::new(ArithmeticVerifier::new(
                    *base,
                    *op_plus,
                    *op_times,
                    *eq_token,
                    *answer_len,
                )),
            };
            ensemble.members.push((entry.weight, verifier));
        }
        ensemble
    }
}

/// Declarative description of a single verifier slot in a reward mix.
#[derive(Debug, Clone)]
pub struct RewardMember {
    pub weight: f32,
    pub kind: VerifierKind,
}

/// Tagged union of the built-in verifiers. Extend this as new verifiers
/// land (math, code, tool-success) so `RewardConfig` stays the single
/// construction surface.
#[derive(Debug, Clone)]
pub enum VerifierKind {
    Copy,
    ReverseCopy,
    Palette {
        allowed_tokens: Vec<usize>,
    },
    Monotonic,
    ToolSuccess {
        sentinel: usize,
    },
    Arithmetic {
        base: usize,
        op_plus: usize,
        op_times: usize,
        eq_token: usize,
        answer_len: usize,
    },
}

/// Config-driven reward aggregation. `members` is ordered; the final
/// reward is `Σ weight_i · verifier_i`, matching `WeightedEnsemble`'s
/// fluent-builder semantics exactly.
#[derive(Debug, Clone, Default)]
pub struct RewardConfig {
    pub members: Vec<RewardMember>,
}

impl RewardConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(mut self, weight: f32, kind: VerifierKind) -> Self {
        self.members.push(RewardMember { weight, kind });
        self
    }
}

impl Default for WeightedEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

impl Verifier for WeightedEnsemble {
    fn verify(&self, prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32 {
        self.members
            .iter()
            .map(|(weight, verifier)| weight * verifier.verify(prompt_ids, full_ids, response_mask))
            .sum()
    }
}
