//! Verifier trait + concrete reward functions for TinyLM training.
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
