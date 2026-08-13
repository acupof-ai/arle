//! Host-side token sampling from a logits row.
//!
//! Pure and backend-independent: greedy decoding is bit-identical to the
//! backends' device argmax, and temperature sampling derives its randomness
//! deterministically from `(seed, position)` so a poll needs no per-slot RNG
//! state. Sampling parameters live in [`crate::SamplingParams`].

use crate::SamplingParams;

/// Index of the maximum logit. Ties resolve to the lowest index.
#[must_use]
/// Ties resolve to the LOWEST index, matching `sampling.cu`'s
/// `warp_reduce_argmax` — the device fast path and this must agree or greedy
/// decode depends on which one ran.
pub fn argmax_logit(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then(bi.cmp(ai)))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Merge per-rank `(max_logit, global_argmax)` pairs from a vocab-sharded
/// lm_head into the global greedy token.
///
/// Ties resolve to the lowest index, like [`argmax_logit`] and the device
/// kernel. Pure host math, so every TP rank derives the same winner.
#[must_use]
pub fn merge_vocab_shard_argmax(pairs: impl IntoIterator<Item = (f32, u32)>) -> Option<u32> {
    pairs
        .into_iter()
        .max_by(|(a_val, a_idx), (b_val, b_idx)| a_val.total_cmp(b_val).then(b_idx.cmp(a_idx)))
        .map(|(_, idx)| idx)
}

/// SplitMix64 — a tiny dependency-free mixer.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xgrammar bitmask: bit `t` set = token `t` allowed. An all-zero mask leaves
/// nothing to draw, so it is treated as "no constraint" rather than returning
/// a token the grammar rejects.
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
/// backends' device argmax. For `temperature > 0` it applies temperature
/// scaling, then optional top-k / top-p / min-p filtering, then a multinomial
/// draw. Randomness is derived deterministically from `(params.seed, position)`
/// so a run is reproducible. Pure host-side: one logits copy at c=1 is
/// sub-millisecond, so no GPU sampling kernel is required.
#[must_use]
pub fn sample_token(logits: &[f32], params: &SamplingParams, position: u64) -> u32 {
    sample_token_logprob(logits, params, position).0
}

/// [`sample_token`] plus the behavior log-probability of the drawn token under
/// the same filtered, renormalized distribution — the IS ratio denominator for
/// on-policy RL. `None` for greedy (a delta policy) and for the
/// degenerate-distribution argmax fallback.
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

    // Temperature-scaled, numerically stable softmax.
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

    // Degenerate distribution (all logits -inf/NaN, or filters emptied the
    // set): fall back to greedy rather than silently returning token 0.
    let total: f32 = cand.iter().map(|(_, p)| *p).sum();
    if cand.is_empty() || !total.is_finite() || total <= 0.0 {
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
