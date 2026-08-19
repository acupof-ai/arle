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

/// Token history the penalties are scored against.
///
/// `tokens` is `prompt ++ generated`; `prompt_len` is the split. Repetition
/// penalty scores the whole slice (HF/vLLM semantics), frequency and presence
/// only `tokens[prompt_len..]` (OpenAI's "text so far" is the completion).
#[derive(Debug, Clone, Copy, Default)]
pub struct PenaltyHistory<'a> {
    pub tokens: &'a [u32],
    pub prompt_len: usize,
}

/// [`sample_token`] with the penalties applied against `history`.
#[must_use]
pub fn sample_token_penalized(
    logits: &[f32],
    params: &SamplingParams,
    position: u64,
    history: PenaltyHistory<'_>,
) -> u32 {
    sample_token_logprob_penalized(logits, params, position, history).0
}

/// Rewrite `logits` in place per the repetition / frequency / presence
/// penalties. Order is fixed: repetition (multiplicative, sign-sensitive) then
/// the two additive ones, matching vLLM.
fn apply_penalties(logits: &mut [f32], params: &SamplingParams, history: PenaltyHistory<'_>) {
    let prompt_len = history.prompt_len.min(history.tokens.len());
    // `p <= 0` maps -inf to NaN, and NaN outranks +inf in `total_cmp`.
    if params.repetition_penalty != 1.0 && params.repetition_penalty > 0.0 {
        let p = params.repetition_penalty;
        let mut seen = std::collections::HashSet::with_capacity(history.tokens.len());
        for &tok in history.tokens {
            if !seen.insert(tok) {
                continue;
            }
            if let Some(l) = logits.get_mut(tok as usize)
                && l.is_finite()
            {
                *l = if *l > 0.0 { *l / p } else { *l * p };
            }
        }
    }
    if params.frequency_penalty == 0.0 && params.presence_penalty == 0.0 {
        return;
    }
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &tok in &history.tokens[prompt_len..] {
        *counts.entry(tok).or_default() += 1;
    }
    for (tok, count) in counts {
        if let Some(l) = logits.get_mut(tok as usize)
            && l.is_finite()
        {
            *l -= params.frequency_penalty * count as f32 + params.presence_penalty;
        }
    }
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
    sample_token_logprob_penalized(logits, params, position, PenaltyHistory::default())
}

/// [`sample_token_logprob`] with the penalties applied against `history`.
///
/// Pipeline order matches vLLM: grammar mask, then `logit_bias`, then
/// repetition / frequency / presence, then temperature and the filters. The
/// penalties precede the greedy early return because they move the argmax.
#[must_use]
pub fn sample_token_logprob_penalized(
    logits: &[f32],
    params: &SamplingParams,
    position: u64,
    history: PenaltyHistory<'_>,
) -> (u32, Option<f32>) {
    if let Some(forced) = params.force_next_token {
        return (forced, None);
    }
    let rewritten = rewrite_logits(logits, params, history);
    let logits = rewritten.as_deref().unwrap_or(logits);
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

/// The grammar-mask / logit-bias / penalty rewrite the sampler draws from.
/// `None` = logits unchanged (no rewriting parameter is active).
fn rewrite_logits(
    logits: &[f32],
    params: &SamplingParams,
    history: PenaltyHistory<'_>,
) -> Option<Vec<f32>> {
    let masked = params
        .grammar_bitmask
        .as_deref()
        .map(|mask| apply_grammar_bitmask(logits, mask));
    let penalized = params.has_penalty() && !history.tokens.is_empty();
    if masked.is_none() && params.logit_bias.is_empty() && !penalized {
        return None;
    }
    let mut v = masked.unwrap_or_else(|| logits.to_vec());
    for &(tok, bias) in &params.logit_bias {
        if (tok as usize) < v.len() {
            v[tok as usize] += bias;
        }
    }
    if penalized {
        apply_penalties(&mut v, params, history);
    }
    Some(v)
}

/// OpenAI logprobs capture for one sampled position, keyed off
/// [`SamplingParams::top_logprobs`]: entry 0 = the sampled token's logprob
/// under the FULL (rewritten, temperature-scaled) softmax; entries 1.. = the
/// top-N alternatives, probability-descending (ties resolve to the lowest
/// token id). Temperature <= 0 scores the unscaled logits. Empty when the
/// capture was not requested.
#[must_use]
pub fn sampled_top_logprobs(
    logits: &[f32],
    params: &SamplingParams,
    history: PenaltyHistory<'_>,
    sampled: u32,
) -> Vec<(u32, f32)> {
    let Some(n) = params.top_logprobs else {
        return Vec::new();
    };
    if logits.is_empty() {
        return Vec::new();
    }
    let rewritten = rewrite_logits(logits, params, history);
    let logits = rewritten.as_deref().unwrap_or(logits);
    let inv_t = if params.temperature > 0.0 {
        1.0 / params.temperature
    } else {
        1.0
    };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    // Top-n scaled logits, descending; ties keep the earlier (lower) token id.
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(n + 1);
    for (i, &l) in logits.iter().enumerate() {
        let s = (l - max) * inv_t;
        sum += s.exp();
        if n > 0 && (top.len() < n || s > top[top.len() - 1].1) {
            let at = top.partition_point(|&(_, ts)| ts >= s);
            top.insert(at, (i as u32, s));
            top.truncate(n);
        }
    }
    // All-(-inf)/NaN row: no distribution to report.
    if !(sum.is_finite() && sum > 0.0) {
        return vec![(sampled, f32::NEG_INFINITY)];
    }
    let ln_sum = sum.ln();
    let sampled_lp = logits
        .get(sampled as usize)
        .map_or(f32::NEG_INFINITY, |&l| (l - max) * inv_t - ln_sum);
    let mut out = Vec::with_capacity(top.len() + 1);
    out.push((sampled, sampled_lp));
    out.extend(top.into_iter().map(|(i, s)| (i, s - ln_sum)));
    out
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
