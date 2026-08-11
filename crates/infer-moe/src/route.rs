//! CPU reference MoE routing — the device-independent ground truth, unified in
//! [`route`] driven by [`MoeConfig`]. Faithful port of two legacy routers
//! (source line refs inline); they share a selection-then-weight skeleton and
//! differ only in scoring func, selection key (bias vs none), and renorm timing.
//!
//! - **DSv4** (`v4.rs` + `dsv4_route.cu`): scores via softmax/sigmoid/
//!   sqrt(softplus); top-k by bias-corrected key; `denom = 1.0` (softmax) else
//!   `selected_sum + 1e-9`; `× routed_scaling_factor`. No softmax renorm.
//! - **Qwen3.6** (`qwen35/moe.rs` + `qwen36_route.cu`): softmax scores; greedy
//!   top-k (zero bias); raw probs, then a separate `norm_topk_prob` renorm.

use crate::config::{MoeConfig, ScoringFunc, TopkMethod};
use crate::error::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertWeight {
    /// Selected routed-expert index, in `0..num_experts`.
    pub expert: usize,
    pub weight: f32,
}

/// Selected experts + weights, in selection order (highest selection-key first).
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingDecision {
    /// `top_k` pairs, ordered by descending selection key (legacy serial order).
    pub experts: Vec<ExpertWeight>,
}

impl RoutingDecision {
    #[must_use]
    pub fn expert_ids(&self) -> Vec<usize> {
        self.experts.iter().map(|ew| ew.expert).collect()
    }

    #[must_use]
    pub fn weights(&self) -> Vec<f32> {
        self.experts.iter().map(|ew| ew.weight).collect()
    }
}

// Scoring primitives — ported from `v4.rs:511-543`.

/// Stable softmax: subtract the max before `exp` (numerical stability).
/// Port of `stable_softmax` (`v4.rs:511`).
#[must_use]
pub fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0_f32;
    let exp = logits
        .iter()
        .map(|&value| {
            let value = (value - max).exp();
            denom += value;
            value
        })
        .collect::<Vec<_>>();
    exp.into_iter().map(|value| value / denom).collect()
}

/// Numerically-stable sigmoid. Port of `sigmoid` (`v4.rs:528`).
#[must_use]
pub fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

/// Numerically-stable softplus with the `> 20 → identity` cutoff.
/// Port of `stable_softplus` (`v4.rs:537`).
#[must_use]
pub fn stable_softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

/// Port of `DeepSeekV4Config::router_scores_from_logits` (`v4.rs:292`).
pub fn scores_from_logits(logits: &[f32], scoring_func: ScoringFunc) -> Result<Vec<f32>> {
    if logits.iter().any(|value| !value.is_finite()) {
        bail!("router logits must be finite");
    }
    Ok(match scoring_func {
        ScoringFunc::Softmax => stable_softmax(logits),
        ScoringFunc::Sigmoid => logits.iter().map(|&v| sigmoid(v)).collect(),
        ScoringFunc::SqrtSoftplus => logits.iter().map(|&v| stable_softplus(v).sqrt()).collect(),
    })
}

// Selection — ported from `topk_indices_by_score` (`v4.rs:556`).

/// Top-`k` experts by descending `scores[e] + bias[e]`, tie-broken by lower
/// index. `bias` is empty (identity) or `scores.len()` long; `mask` (if
/// present) marks ineligible experts `false` (group-limited routing).
fn topk_indices_by_score(
    scores: &[f32],
    bias: &[f32],
    k: usize,
    mask: Option<&[bool]>,
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len())
        .filter(|&e| mask.is_none_or(|m| m[e]))
        .collect();
    indices.sort_by(|&a, &b| {
        let bias_a = bias.get(a).copied().unwrap_or(0.0);
        let bias_b = bias.get(b).copied().unwrap_or(0.0);
        let score_b = scores[b] + bias_b;
        let score_a = scores[a] + bias_a;
        score_b.total_cmp(&score_a).then_with(|| a.cmp(&b))
    });
    indices.truncate(k);
    indices
}

// Group-limited routing — DeepSeek-V2/V3 `n_group`/`topk_group`.
// Not exercised by either ARLE router; here so this reference can verify a
// grouped kernel. Group score = top-2 expert scores summed (DeepSeek-V3 rule);
// the `topk_group` highest groups stay eligible.

/// Per-group selection score: sum of the top-2 corrected scores in each
/// contiguous equal-size group.
fn group_scores(corrected: &[f32], n_group: usize) -> Vec<f32> {
    let experts_per_group = corrected.len() / n_group;
    (0..n_group)
        .map(|g| {
            let start = g * experts_per_group;
            let group = &corrected[start..start + experts_per_group];
            // Top-2 sum (DeepSeek-V3). With group size 1 this is just the max.
            let mut top1 = f32::NEG_INFINITY;
            let mut top2 = f32::NEG_INFINITY;
            for &v in group {
                if v > top1 {
                    top2 = top1;
                    top1 = v;
                } else if v > top2 {
                    top2 = v;
                }
            }
            if top2.is_finite() { top1 + top2 } else { top1 }
        })
        .collect()
}

/// Eligibility mask for group-limited routing: `true` for experts in one of the
/// `topk_group` highest-scoring groups, `false` otherwise.
///
/// `corrected[e] = scores[e] + bias[e]` (the selection key). Groups are scored
/// by their top-2 corrected-score sum; the `topk_group` highest groups (ties →
/// lower group index) keep all their experts eligible.
pub fn group_limited_mask(corrected: &[f32], n_group: usize, topk_group: usize) -> Vec<bool> {
    let experts_per_group = corrected.len() / n_group;
    let gscores = group_scores(corrected, n_group);
    let mut group_order: Vec<usize> = (0..n_group).collect();
    group_order.sort_by(|&a, &b| gscores[b].total_cmp(&gscores[a]).then_with(|| a.cmp(&b)));
    group_order.truncate(topk_group);

    let mut mask = vec![false; corrected.len()];
    for &g in &group_order {
        let start = g * experts_per_group;
        for slot in mask.iter_mut().skip(start).take(experts_per_group) {
            *slot = true;
        }
    }
    mask
}

/// Route one token. `gate_logits` is `[num_experts]`; `bias` is the optional DSv4
/// `e_score_correction_bias` (or empty).
pub fn route_token(gate_logits: &[f32], bias: &[f32], cfg: &MoeConfig) -> Result<RoutingDecision> {
    cfg.validate()?;
    if gate_logits.len() != cfg.num_experts {
        bail!(
            "router logits length {} does not match num_experts {}",
            gate_logits.len(),
            cfg.num_experts
        );
    }
    if !bias.is_empty() && bias.len() != cfg.num_experts {
        bail!(
            "router bias length {} does not match num_experts {}",
            bias.len(),
            cfg.num_experts
        );
    }
    if bias.iter().any(|value| !value.is_finite()) {
        bail!("router bias must be finite");
    }

    let scores = scores_from_logits(gate_logits, cfg.scoring_func)?;
    // DSv4 asserts scores finite + non-negative (`v4.rs:334`).
    if scores.iter().any(|v| !v.is_finite() || *v < 0.0) {
        bail!("router scores must be finite and non-negative");
    }

    let mask = match (cfg.n_group, cfg.topk_group) {
        (Some(n_group), Some(topk_group)) => {
            let corrected: Vec<f32> = scores
                .iter()
                .enumerate()
                .map(|(e, &s)| s + bias.get(e).copied().unwrap_or(0.0))
                .collect();
            Some(group_limited_mask(&corrected, n_group, topk_group))
        }
        _ => None,
    };

    // `noaux_tc` selects with the bias-corrected key; greedy selects with the
    // raw scores (Qwen3.6 supplies a zero bias, which is the same thing).
    let selection_bias: &[f32] = match cfg.topk_method {
        TopkMethod::NoAuxTc => bias,
        TopkMethod::Greedy => &[],
    };
    let selected = topk_indices_by_score(&scores, selection_bias, cfg.top_k, mask.as_deref());

    // DSv4 rule (`moe_routes_from_scores`, `v4.rs:386`): normalize iff scoring
    // func != softmax; `weight = scores[e] / denom * scaling`. Softmax keeps raw
    // probs (denom 1.0); the Qwen3.6 `norm_topk_prob` renorm is a separate step.
    let selected_sum: f32 = selected.iter().map(|&e| scores[e]).sum();
    let normalize = cfg.scoring_func != ScoringFunc::Softmax;
    let denom = if normalize {
        selected_sum + 1.0e-9
    } else {
        1.0
    };

    let mut experts: Vec<ExpertWeight> = selected
        .into_iter()
        .map(|expert| ExpertWeight {
            expert,
            weight: scores[expert] / denom * cfg.routed_scaling_factor,
        })
        .collect();

    // Qwen3.6 `norm_topk_prob` renorm (`qwen36_route.cu:23`). Greedy path only:
    // renorm the selected weights to sum 1. DSv4 (`NoAuxTc`) never runs it.
    if cfg.norm_topk_prob && cfg.topk_method == TopkMethod::Greedy && !normalize {
        let sum: f32 = experts.iter().map(|ew| ew.weight).sum();
        let inv = if sum > 1.0e-20 { 1.0 / sum } else { 0.0 };
        for ew in &mut experts {
            ew.weight *= inv;
        }
    }

    Ok(RoutingDecision { experts })
}

/// Route a flat batch. `gate_logits` is `[num_tokens * num_experts]` token-major;
/// `bias` is the shared `[num_experts]` correction (or empty).
pub fn route(gate_logits: &[f32], bias: &[f32], cfg: &MoeConfig) -> Result<Vec<RoutingDecision>> {
    cfg.validate()?;
    if cfg.num_experts == 0 || !gate_logits.len().is_multiple_of(cfg.num_experts) {
        bail!(
            "gate_logits length {} is not a multiple of num_experts {}",
            gate_logits.len(),
            cfg.num_experts
        );
    }
    let n = cfg.num_experts;
    let out = gate_logits
        .chunks(n)
        .map(|row| route_token(row, bias, cfg))
        .collect::<Result<Vec<_>>>()?;
    Ok(out)
}

/// CPU end-to-end MoE reference for inference-side routed experts:
/// `logits -> scores -> group mask -> top-k -> weights -> weighted sum`.
/// `gate_logits` is `[num_tokens * num_experts]`; `expert_outputs` is
/// `[num_tokens * num_experts * hidden_dim]`, token-major then expert-major.
/// Shared experts are model-specific always-on paths and stay outside this
/// routed-expert reference.
pub fn route_and_combine(
    gate_logits: &[f32],
    bias: &[f32],
    expert_outputs: &[f32],
    hidden_dim: usize,
    cfg: &MoeConfig,
) -> Result<(Vec<RoutingDecision>, Vec<f32>)> {
    if hidden_dim == 0 {
        bail!("hidden_dim must be > 0");
    }
    let decisions = route(gate_logits, bias, cfg)?;
    let num_tokens = decisions.len();
    let expected = match num_tokens
        .checked_mul(cfg.num_experts)
        .and_then(|v| v.checked_mul(hidden_dim))
    {
        Some(value) => value,
        None => bail!("expert_outputs shape overflow"),
    };
    if expert_outputs.len() != expected {
        bail!(
            "expert_outputs length {} does not match [tokens={}, experts={}, hidden_dim={}]",
            expert_outputs.len(),
            num_tokens,
            cfg.num_experts,
            hidden_dim
        );
    }

    let mut out = vec![0.0_f32; num_tokens * hidden_dim];
    for (token, decision) in decisions.iter().enumerate() {
        let out_row = &mut out[token * hidden_dim..(token + 1) * hidden_dim];
        for ExpertWeight { expert, weight } in &decision.experts {
            let expert_offset = (token * cfg.num_experts + *expert) * hidden_dim;
            let expert_row = &expert_outputs[expert_offset..expert_offset + hidden_dim];
            for (dst, &value) in out_row.iter_mut().zip(expert_row) {
                *dst += value * *weight;
            }
        }
    }

    Ok((decisions, out))
}
