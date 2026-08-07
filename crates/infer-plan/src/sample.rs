//! Host-side token sampling from a logits row.
//!
//! Pure and backend-independent: greedy decoding is bit-identical to the
//! backends' device argmax, and temperature sampling derives its randomness
//! deterministically from `(seed, position)` so a poll needs no per-slot RNG
//! state. Sampling parameters live in [`crate::SamplingParams`].

use crate::SamplingParams;

/// Index of the maximum logit (greedy / argmax). Ties resolve to the lowest index.
#[must_use]
pub fn argmax_logit(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Merge per-rank `(max_logit, global_argmax)` pairs from a vocab-sharded
/// lm_head (`ARLE_DSV4_LM_HEAD_SHARD=1`, #99) into the global greedy token.
/// Exactly reproduces the CUDA device argmax it replaces (`sampling.cu`
/// `warp_reduce_argmax`): max value wins, ties resolve to the LOWEST index.
/// (NB: [`argmax_logit`]'s `max_by` resolves ties to the highest index — the
/// device kernel, not the host helper, is this merge's parity reference.)
/// Pure host math, so every TP rank derives the same winner from the same
/// gathered pairs. `None` iff `pairs` is empty.
#[must_use]
pub fn merge_vocab_shard_argmax(pairs: impl IntoIterator<Item = (f32, u32)>) -> Option<u32> {
    pairs
        .into_iter()
        .max_by(|(a_val, a_idx), (b_val, b_idx)| a_val.total_cmp(b_val).then(b_idx.cmp(a_idx)))
        .map(|(_, idx)| idx)
}

/// SplitMix64 — a tiny dependency-free mixer turning a seed into a u64 stream.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xgrammar bitmask semantics: bit `t` set = token `t` allowed. An all-zero
/// mask would leave nothing to draw, so it is treated as "no constraint"
/// rather than returning a token the grammar rejects.
#[must_use]
pub fn apply_grammar_bitmask(logits: &[f32], mask: &[u32]) -> Vec<f32> {
    let allowed = |t: usize| mask.get(t / 32).is_some_and(|w| w >> (t % 32) & 1 == 1);
    if !(0..logits.len()).any(allowed) {
        return logits.to_vec();
    }
    logits
        .iter()
        .enumerate()
        .map(|(t, &l)| if allowed(t) { l } else { f32::NEG_INFINITY })
        .collect()
}

/// Sample one token id from a logits row under `params`.
///
/// Greedy (`temperature <= 0`) returns `argmax_logit` — bit-identical to the
/// backends' device argmax, so it preserves greedy parity. For `temperature > 0`
/// it applies temperature scaling, then optional top-k / top-p (nucleus) / min-p
/// filtering, then a multinomial draw. Randomness is derived deterministically
/// from `(params.seed, position)` so a run is reproducible and a discrete poll
/// needs no per-slot RNG state. Pure and host-side: one logits copy at c=1 is
/// sub-millisecond, so no new GPU sampling kernel is required.
#[must_use]
pub fn sample_token(logits: &[f32], params: &SamplingParams, position: u64) -> u32 {
    sample_token_logprob(logits, params, position).0
}

/// [`sample_token`] plus the behavior log-probability of the drawn token under
/// the SAME filtered, renormalized distribution the draw came from — the IS
/// ratio denominator for on-policy RL. `None` for greedy (a delta policy) and
/// for the degenerate-distribution argmax fallback.
#[must_use]
pub fn sample_token_logprob(
    logits: &[f32],
    params: &SamplingParams,
    position: u64,
) -> (u32, Option<f32>) {
    if let Some(mask) = params.grammar_bitmask.as_deref() {
        let masked = apply_grammar_bitmask(logits, mask);
        let mut p = params.clone();
        p.grammar_bitmask = None;
        return sample_token_logprob(&masked, &p, position);
    }
    // Apply logit_bias before any temperature/filtering.
    let biased = if params.logit_bias.is_empty() {
        None
    } else {
        let mut v = logits.to_vec();
        for (&tok, &bias) in &params.logit_bias {
            if (tok as usize) < v.len() {
                v[tok as usize] += bias;
            }
        }
        Some(v)
    };
    let logits = biased.as_deref().unwrap_or(logits);
    if params.is_greedy() || logits.is_empty() {
        return (argmax_logit(logits), None);
    }

    // Temperature-scaled, numerically stable softmax over all candidates.
    let inv_t = 1.0 / params.temperature;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if params.top_k <= 0 && params.top_p >= 1.0 && params.min_p <= 0.0 {
        return sample_unfiltered_temperature(logits, inv_t, max, params, position);
    }

    let mut cand: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, ((l - max) * inv_t).exp()))
        .collect();
    let sum: f32 = cand.iter().map(|(_, p)| *p).sum();
    if sum > 0.0 {
        for c in &mut cand {
            c.1 /= sum;
        }
    }

    // Descending by probability for top-k / top-p truncation.
    cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    if params.top_k > 0 && (params.top_k as usize) < cand.len() {
        cand.truncate(params.top_k as usize);
    }
    if params.top_p < 1.0 {
        let mut cum = 0.0;
        let mut cut = cand.len();
        for (i, (_, p)) in cand.iter().enumerate() {
            cum += *p;
            if cum >= params.top_p {
                cut = i + 1;
                break;
            }
        }
        cand.truncate(cut.max(1));
    }
    if params.min_p > 0.0 {
        let top = cand.first().map_or(0.0, |(_, p)| *p);
        let thresh = params.min_p * top;
        cand.retain(|(_, p)| *p >= thresh);
    }

    // Degenerate distribution (all logits -inf/NaN, or filters emptied the set):
    // fall back to greedy rather than silently returning token 0.
    let total: f32 = cand.iter().map(|(_, p)| *p).sum();
    if cand.is_empty() || !total.is_finite() || total <= 0.0 {
        return (argmax_logit(logits), None);
    }

    // Multinomial draw over the surviving candidates.
    let bits = splitmix64(
        params
            .seed
            .unwrap_or(0)
            .wrapping_add(position)
            .wrapping_add(1),
    );
    let unit = (bits >> 40) as f32 / (1u32 << 24) as f32; // [0, 1)
    let mut acc = 0.0;
    let target = unit * total;
    for (idx, p) in &cand {
        acc += *p;
        if target < acc {
            return (*idx, Some((*p / total).ln()));
        }
    }
    cand.last()
        .map_or((0, None), |(idx, p)| (*idx, Some((*p / total).ln())))
}

fn sample_unfiltered_temperature(
    logits: &[f32],
    inv_t: f32,
    max: f32,
    params: &SamplingParams,
    position: u64,
) -> (u32, Option<f32>) {
    let mut total = 0.0f32;
    for &logit in logits {
        total += ((logit - max) * inv_t).exp();
    }
    if !total.is_finite() || total <= 0.0 {
        return (argmax_logit(logits), None);
    }

    let bits = splitmix64(
        params
            .seed
            .unwrap_or(0)
            .wrapping_add(position)
            .wrapping_add(1),
    );
    let unit = (bits >> 40) as f32 / (1u32 << 24) as f32; // [0, 1)
    let target = unit * total;
    let mut acc = 0.0;
    let logprob = |logit: f32| Some((logit - max) * inv_t - total.ln());
    for (idx, &logit) in logits.iter().enumerate() {
        acc += ((logit - max) * inv_t).exp();
        if target < acc {
            return (idx as u32, logprob(logit));
        }
    }
    let last = logits.len().saturating_sub(1);
    (last as u32, logprob(logits[last]))
}

#[cfg(test)]
mod sampler_tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.1).collect()
    }

    #[test]
    fn grammar_bitmask_binds_greedy_and_sampled_draws() {
        let logits = ramp(64);
        let mut p = SamplingParams::default();
        assert_eq!(sample_token_logprob(&logits, &p, 0).0, 63);
        // Allow only token 7: both decode modes must land on it, not on the
        // unconstrained argmax.
        let mut mask = vec![0u32; 2];
        mask[0] = 1 << 7;
        p.grammar_bitmask = Some(mask.clone().into());
        assert_eq!(sample_token_logprob(&logits, &p, 0).0, 7);
        p.temperature = 1.0;
        assert_eq!(sample_token_logprob(&logits, &p, 0).0, 7);
        // An empty mask has no legal draw; fall through rather than emit a
        // token the grammar rejects.
        p.temperature = 0.0;
        p.grammar_bitmask = Some(vec![0u32; 2].into());
        assert_eq!(sample_token_logprob(&logits, &p, 0).0, 63);
    }

    #[test]
    fn greedy_equals_argmax() {
        let logits = vec![0.2, 5.0, 1.0, 4.9];
        let p = SamplingParams::default(); // temperature 0.0
        assert!(p.is_greedy());
        assert_eq!(sample_token(&logits, &p, 0), 1);
        assert_eq!(argmax_logit(&logits), 1);
    }

    #[test]
    fn degenerate_logits_fall_back_to_greedy() {
        // All -inf (e.g. a fully-masked row): must NOT silently return token 0
        // via a NaN-emptied candidate set — fall back to argmax cleanly.
        let all_neg = vec![f32::NEG_INFINITY; 8];
        let p = SamplingParams {
            temperature: 1.0,
            min_p: 0.5,
            ..SamplingParams::default()
        };
        assert_eq!(sample_token(&all_neg, &p, 3), argmax_logit(&all_neg));
        // One finite logit among -inf with aggressive min_p: pick the finite one,
        // never an empty-set token 0.
        let mut one = vec![f32::NEG_INFINITY; 8];
        one[5] = 4.0;
        assert_eq!(sample_token(&one, &p, 0), 5);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let logits = ramp(50);
        let p = SamplingParams {
            temperature: 1.5,
            top_k: 1,
            ..SamplingParams::default()
        };
        // Only the max survives, so every position samples it.
        for pos in 0..16 {
            assert_eq!(sample_token(&logits, &p, pos), 49);
        }
    }

    #[test]
    fn seeded_sampling_is_deterministic_and_varies_by_position() {
        let logits = ramp(64);
        let p = SamplingParams {
            temperature: 1.0,
            seed: Some(42),
            ..SamplingParams::default()
        };
        // Reproducible for a fixed (seed, position).
        assert_eq!(sample_token(&logits, &p, 3), sample_token(&logits, &p, 3));
        // Advancing position changes the stream (not stuck on one token).
        let draws: Vec<u32> = (0..32).map(|pos| sample_token(&logits, &p, pos)).collect();
        assert!(draws.iter().any(|&t| t != draws[0]));
    }

    #[test]
    fn unfiltered_temperature_sampling_uses_full_distribution() {
        let logits = vec![0.0, 0.5, 1.0, 1.5];
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: Some(11),
            ..SamplingParams::default()
        };
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let total: f32 = weights.iter().sum();
        for position in 0..16 {
            let bits = splitmix64(p.seed.unwrap().wrapping_add(position).wrapping_add(1));
            let target = ((bits >> 40) as f32 / (1u32 << 24) as f32) * total;
            let mut acc = 0.0;
            let mut expected = weights.len() as u32 - 1;
            for (idx, weight) in weights.iter().enumerate() {
                acc += *weight;
                if target < acc {
                    expected = idx as u32;
                    break;
                }
            }
            assert_eq!(sample_token(&logits, &p, position), expected);
        }
    }

    /// Host mimic of the device argmax tie rule (`sampling.cu`): first
    /// (lowest-index) maximum wins. `argmax_logit` differs on ties (max_by
    /// keeps the LAST maximum), so it is not the parity reference here.
    fn device_argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best as u32
    }

    #[test]
    fn vocab_shard_merge_matches_full_device_argmax() {
        // 4 "ranks" of 4 logits each: the merge over per-slice (max, argmax)
        // must equal the device argmax over the concatenated vector.
        let slices: Vec<Vec<f32>> = vec![
            vec![0.1, -2.0, 0.9, 0.3],
            vec![4.0, 0.0, 0.0, 0.0],
            vec![-1.0, 4.0, 2.0, 3.9],
            vec![0.0, 0.0, 0.0, 3.99],
        ];
        let full: Vec<f32> = slices.iter().flatten().copied().collect();
        let pairs = slices.iter().enumerate().map(|(rank, s)| {
            let local = device_argmax(s);
            (s[local as usize], rank as u32 * 4 + local)
        });
        // Tie between global 4 (rank 1) and global 9 (rank 2): lowest index
        // wins, matching the device kernel's `idx <` tie rule.
        assert_eq!(merge_vocab_shard_argmax(pairs), Some(4));
        assert_eq!(device_argmax(&full), 4);
    }

    #[test]
    fn vocab_shard_merge_tie_resolves_to_lowest_index() {
        let pairs = [(1.5, 700u32), (1.5, 3u32), (1.5, 42u32)];
        assert_eq!(merge_vocab_shard_argmax(pairs), Some(3));
        assert_eq!(merge_vocab_shard_argmax([]), None);
        assert_eq!(merge_vocab_shard_argmax([(f32::NEG_INFINITY, 9)]), Some(9));
    }

    #[test]
    fn top_p_keeps_only_the_nucleus() {
        // One dominant logit: nucleus collapses to it regardless of temperature.
        let mut logits = vec![0.0; 100];
        logits[7] = 50.0;
        let p = SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            seed: Some(1),
            ..SamplingParams::default()
        };
        assert_eq!(sample_token(&logits, &p, 0), 7);
    }
}
