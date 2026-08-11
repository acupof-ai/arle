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

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    fn assert_close(actual: f32, expected: f32, label: &str) {
        assert!(
            (actual - expected).abs() <= EPS,
            "{label}: actual={actual} expected={expected}"
        );
    }

    fn assert_slice_close(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch actual={} expected={}",
            actual.len(),
            expected.len()
        );
        for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(actual, expected, &format!("{label}[{i}]"));
        }
    }

    fn noaux_cfg(
        num_experts: usize,
        top_k: usize,
        scoring_func: ScoringFunc,
        routed_scaling_factor: f32,
        n_group: Option<usize>,
        topk_group: Option<usize>,
    ) -> MoeConfig {
        MoeConfig {
            num_experts,
            num_shared_experts: 0,
            top_k,
            scoring_func,
            topk_method: TopkMethod::NoAuxTc,
            norm_topk_prob: false,
            routed_scaling_factor,
            n_group,
            topk_group,
            hidden_size: 16,
        }
    }

    fn expert_outputs(num_tokens: usize, num_experts: usize, hidden_dim: usize) -> Vec<f32> {
        (0..num_tokens)
            .flat_map(|token| {
                (0..num_experts).flat_map(move |expert| {
                    (0..hidden_dim).map(move |dim| {
                        (token as f32 + 1.0) * 1000.0 + expert as f32 * 10.0 + dim as f32
                    })
                })
            })
            .collect()
    }

    fn expert_row(
        expert_outputs: &[f32],
        token: usize,
        expert: usize,
        num_experts: usize,
        hidden_dim: usize,
    ) -> &[f32] {
        let offset = (token * num_experts + expert) * hidden_dim;
        &expert_outputs[offset..offset + hidden_dim]
    }

    fn expected_weighted_sum(
        expert_outputs: &[f32],
        token: usize,
        decision: &RoutingDecision,
        num_experts: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; hidden_dim];
        for ExpertWeight { expert, weight } in &decision.experts {
            let row = expert_row(expert_outputs, token, *expert, num_experts, hidden_dim);
            for (dst, &value) in out.iter_mut().zip(row) {
                *dst += value * *weight;
            }
        }
        out
    }

    #[test]
    fn group_scores_top2_sum_for_four_by_four_groups() {
        let corrected = [
            1.0, 4.0, 2.0, 3.0, // g0 top-2 = 4 + 3 = 7
            -1.0, -2.0, -3.0, -4.0, // g1 top-2 = -1 + -2 = -3
            0.5, 2.5, 2.0, 1.0, // g2 top-2 = 2.5 + 2 = 4.5
            6.0, 6.0, 5.0, 0.0, // g3 top-2 = 6 + 6 = 12
        ];
        let got = group_scores(&corrected, 4);
        assert_eq!(got.len(), 4);
        for (i, (&actual, expected)) in got.iter().zip([7.0, -3.0, 4.5, 12.0]).enumerate() {
            assert_close(actual, expected, &format!("group score {i}"));
        }
    }

    #[test]
    fn group_limited_route_excludes_global_top_outside_top_group() {
        let logits = [0.0; 8];
        // All sigmoid scores are 0.5. Bias makes expert 0 the global top key
        // (10.5), but group 1 wins by top-2-sum: 6.5 + 6.5 > 10.5 + 0.5.
        let bias = [10.0, 0.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0];
        let scores = scores_from_logits(&logits, ScoringFunc::Sigmoid).unwrap();
        assert_eq!(
            topk_indices_by_score(&scores, &bias, 2, None),
            vec![0, 4],
            "without group mask the globally best expert wins"
        );

        let cfg = noaux_cfg(8, 2, ScoringFunc::Sigmoid, 1.0, Some(2), Some(1));
        let decision = route_token(&logits, &bias, &cfg).unwrap();
        assert_eq!(
            decision.expert_ids(),
            vec![4, 5],
            "topk_group mask keeps only the highest top-2-sum group"
        );
    }

    #[test]
    fn bias_corrected_selection_returns_unbiased_weight() {
        let logits = [0.0, 0.0, -5.0, 0.0];
        let bias = [0.0, 0.0, 10.0, 0.0];
        let cfg = noaux_cfg(4, 2, ScoringFunc::Sigmoid, 1.0, None, None);
        let scores = scores_from_logits(&logits, ScoringFunc::Sigmoid).unwrap();

        let decision = route_token(&logits, &bias, &cfg).unwrap();
        assert_eq!(
            decision.expert_ids(),
            vec![2, 0],
            "bias changes selection order"
        );

        let denom = scores[2] + scores[0] + 1.0e-9;
        assert_close(
            decision.experts[0].weight,
            scores[2] / denom,
            "selected expert weight uses pre-bias score",
        );
        assert!(
            decision.experts[0].weight < 0.02,
            "large bias must not leak into emitted route weight"
        );
    }

    #[test]
    fn topk_ties_break_by_lower_index() {
        let scores = [1.0, 2.0, 2.0, 2.0, 0.0];
        assert_eq!(
            topk_indices_by_score(&scores, &[], 3, None),
            vec![1, 2, 3],
            "equal raw scores prefer lower expert index"
        );

        let bias = [1.0, 0.0, 1.0];
        let scores = [1.0, 2.0, 1.0];
        assert_eq!(
            topk_indices_by_score(&scores, &bias, 3, None),
            vec![0, 1, 2],
            "equal corrected keys prefer lower expert index"
        );

        let cfg = noaux_cfg(5, 3, ScoringFunc::Sigmoid, 1.0, None, None);
        let decision = route_token(&[0.0; 5], &[0.0; 5], &cfg).unwrap();
        assert_eq!(
            decision.expert_ids(),
            vec![0, 1, 2],
            "route_token inherits lower-index tie break"
        );
    }

    #[test]
    fn denom_rules_and_scaling_for_softmax_and_sigmoid() {
        let softmax_logits = [0.0, 1.0, 2.0, 3.0];
        let softmax_scores = stable_softmax(&softmax_logits);
        let softmax_cfg = noaux_cfg(4, 2, ScoringFunc::Softmax, 3.0, None, None);
        let softmax = route_token(&softmax_logits, &[0.0; 4], &softmax_cfg).unwrap();
        assert_eq!(softmax.expert_ids(), vec![3, 2]);
        assert_close(
            softmax.weights()[0],
            softmax_scores[3] * 3.0,
            "softmax denom is 1.0, then scaling applies",
        );
        assert_close(
            softmax.weights()[1],
            softmax_scores[2] * 3.0,
            "softmax second selected weight",
        );

        let sigmoid_logits = [2.0, 1.0, 0.0, -1.0];
        let sigmoid_scores = scores_from_logits(&sigmoid_logits, ScoringFunc::Sigmoid).unwrap();
        let sigmoid_cfg = noaux_cfg(4, 2, ScoringFunc::Sigmoid, 1.5, None, None);
        let sigmoid = route_token(&sigmoid_logits, &[0.0; 4], &sigmoid_cfg).unwrap();
        assert_eq!(sigmoid.expert_ids(), vec![0, 1]);
        let denom = sigmoid_scores[0] + sigmoid_scores[1] + 1.0e-9;
        assert_close(
            sigmoid.weights()[0],
            sigmoid_scores[0] / denom * 1.5,
            "sigmoid denom is selected_sum + 1e-9, then scaling applies",
        );
        assert_close(
            sigmoid.weights()[1],
            sigmoid_scores[1] / denom * 1.5,
            "sigmoid second selected weight",
        );
    }

    #[test]
    fn group_degenerate_masks_are_all_true_when_all_groups_kept() {
        let corrected = [0.2, 0.7, -1.0, 1.2];
        assert_eq!(
            group_scores(&corrected, 4),
            corrected,
            "singleton groups score as their single expert max"
        );

        assert_eq!(
            group_limited_mask(&corrected, 1, 1),
            vec![true; corrected.len()],
            "n_group=1, topk_group=n_group keeps the only group"
        );
        assert_eq!(
            group_limited_mask(&corrected, 4, 4),
            vec![true; corrected.len()],
            "topk_group==n_group keeps every group"
        );

        let logits = [0.1, 0.3, -0.2, 2.0, 1.0, 0.0, 0.7, 0.8];
        let ungrouped = noaux_cfg(8, 3, ScoringFunc::Sigmoid, 1.0, None, None);
        let all_groups = noaux_cfg(8, 3, ScoringFunc::Sigmoid, 1.0, Some(2), Some(2));
        assert_eq!(
            route_token(&logits, &[0.0; 8], &all_groups).unwrap(),
            route_token(&logits, &[0.0; 8], &ungrouped).unwrap(),
            "topk_group==n_group route matches ungrouped route"
        );
    }

    #[test]
    fn route_e2e_noaux_tc_group_limited_pipeline() {
        let cfg = noaux_cfg(16, 3, ScoringFunc::Sigmoid, 2.0, Some(4), Some(2));
        let logits = [0.0; 16];
        // All scores are 0.5. Corrected top-2 group scores:
        // g0: (7.5 + 6.5), g1: (1.5 + 1.5), g2: (3.5 + 2.5), g3: (5.5 + 5.0).
        // The kept groups are g0 and g3. Expert 8 has a higher corrected key
        // than expert 15, but it is in masked-out g2 and must not be selected.
        let bias = [
            7.0, 6.0, 0.0, 0.0, // g0
            1.0, 1.0, 1.0, 1.0, // g1
            3.0, 2.0, 0.0, 0.0, // g2
            5.0, 4.5, 0.0, 0.0, // g3
        ];

        let decisions = route(&logits, &bias, &cfg).unwrap();
        assert_eq!(decisions.len(), 1);
        let decision = &decisions[0];
        assert_eq!(
            decision.expert_ids(),
            vec![0, 1, 12],
            "route() applies scoring, group top-2-sum mask, noaux_tc bias, and top-k order"
        );

        // Returned weights are still pre-bias sigmoid scores normalized over
        // the selected experts, then scaled by routed_scaling_factor.
        for (i, weight) in decision.weights().iter().enumerate() {
            assert_close(*weight, 2.0 / 3.0, &format!("selected weight {i}"));
        }
    }

    #[test]
    fn route_e2e_bias_changes_selection_not_weight() {
        let cfg = noaux_cfg(4, 2, ScoringFunc::Sigmoid, 1.0, None, None);
        let logits = [0.0, 0.0, -5.0, 0.0];
        let bias = [0.0, 0.0, 10.0, 0.0];
        let decisions = route(&logits, &bias, &cfg).unwrap();
        let decision = &decisions[0];

        let scores = scores_from_logits(&logits, ScoringFunc::Sigmoid).unwrap();
        let denom = scores[2] + scores[0] + 1.0e-9;
        assert_eq!(decision.expert_ids(), vec![2, 0]);
        assert_close(
            decision.weights()[0],
            scores[2] / denom,
            "route() selected weight remains pre-bias",
        );
    }

    #[test]
    fn route_e2e_softmax_and_sigmoid_denoms_are_distinct() {
        let softmax_logits = [0.0, 1.0, 2.0, 3.0];
        let softmax_cfg = noaux_cfg(4, 2, ScoringFunc::Softmax, 3.0, None, None);
        let softmax = route(&softmax_logits, &[0.0; 4], &softmax_cfg).unwrap();
        let softmax_scores = stable_softmax(&softmax_logits);
        assert_eq!(softmax[0].expert_ids(), vec![3, 2]);
        assert_close(
            softmax[0].weights()[0],
            softmax_scores[3] * 3.0,
            "route() softmax denom is 1.0",
        );

        let sigmoid_logits = [2.0, 1.0, 0.0, -1.0];
        let sigmoid_cfg = noaux_cfg(4, 2, ScoringFunc::Sigmoid, 1.5, None, None);
        let sigmoid = route(&sigmoid_logits, &[0.0; 4], &sigmoid_cfg).unwrap();
        let sigmoid_scores = scores_from_logits(&sigmoid_logits, ScoringFunc::Sigmoid).unwrap();
        let denom = sigmoid_scores[0] + sigmoid_scores[1] + 1.0e-9;
        assert_eq!(sigmoid[0].expert_ids(), vec![0, 1]);
        assert_close(
            sigmoid[0].weights()[0],
            sigmoid_scores[0] / denom * 1.5,
            "route() sigmoid denom is selected_sum + 1e-9",
        );
    }

    #[test]
    fn route_e2e_ties_break_lower_index_across_batch() {
        let cfg = noaux_cfg(5, 3, ScoringFunc::Sigmoid, 1.0, None, None);
        let flat = [
            0.0, 0.0, 0.0, 0.0, 0.0, // all tied
            -1.0, 1.0, 1.0, 1.0, -1.0, // experts 1/2/3 tied for top
        ];
        let decisions = route(&flat, &[0.0; 5], &cfg).unwrap();
        assert_eq!(decisions[0].expert_ids(), vec![0, 1, 2]);
        assert_eq!(decisions[1].expert_ids(), vec![1, 2, 3]);
    }

    #[test]
    fn route_e2e_group_degenerate_matches_ungrouped() {
        let logits = [0.1, 0.3, -0.2, 2.0, 1.0, 0.0, 0.7, 0.8];
        let ungrouped = noaux_cfg(8, 3, ScoringFunc::Sigmoid, 1.0, None, None);
        let one_group = noaux_cfg(8, 3, ScoringFunc::Sigmoid, 1.0, Some(1), Some(1));
        let all_groups = noaux_cfg(8, 3, ScoringFunc::Sigmoid, 1.0, Some(4), Some(4));

        let base = route(&logits, &[0.0; 8], &ungrouped).unwrap();
        assert_eq!(
            route(&logits, &[0.0; 8], &one_group).unwrap(),
            base,
            "n_group=1 keeps all experts and matches ungrouped route()"
        );
        assert_eq!(
            route(&logits, &[0.0; 8], &all_groups).unwrap(),
            base,
            "topk_group==n_group keeps all experts and matches ungrouped route()"
        );
    }

    #[test]
    fn route_and_combine_e2e_noaux_tc_group_limited_pipeline() {
        let cfg = noaux_cfg(16, 3, ScoringFunc::Sigmoid, 2.0, Some(4), Some(2));
        let logits = [0.0; 16];
        let bias = [
            7.0, 6.0, 0.0, 0.0, // g0
            1.0, 1.0, 1.0, 1.0, // g1
            3.0, 2.0, 0.0, 0.0, // g2
            5.0, 4.5, 0.0, 0.0, // g3
        ];
        let hidden_dim = 4;
        let expert_outputs = expert_outputs(1, 16, hidden_dim);

        let (decisions, combined) =
            route_and_combine(&logits, &bias, &expert_outputs, hidden_dim, &cfg).unwrap();

        let decision = &decisions[0];
        assert_eq!(
            decision.expert_ids(),
            vec![0, 1, 12],
            "full reference applies group mask before routed weighted combine"
        );
        for (i, weight) in decision.weights().iter().enumerate() {
            assert_close(*weight, 2.0 / 3.0, &format!("selected weight {i}"));
        }
        let expected = expected_weighted_sum(&expert_outputs, 0, decision, 16, hidden_dim);
        assert_slice_close(&combined, &expected, "group-limited combined output");
    }

    #[test]
    fn route_and_combine_e2e_bias_changes_selection_not_weight() {
        let cfg = noaux_cfg(4, 2, ScoringFunc::Sigmoid, 1.0, None, None);
        let logits = [0.0, 0.0, -5.0, 0.0];
        let bias = [0.0, 0.0, 10.0, 0.0];
        let hidden_dim = 3;
        let expert_outputs = expert_outputs(1, 4, hidden_dim);

        let (decisions, combined) =
            route_and_combine(&logits, &bias, &expert_outputs, hidden_dim, &cfg).unwrap();

        let scores = scores_from_logits(&logits, ScoringFunc::Sigmoid).unwrap();
        let denom = scores[2] + scores[0] + 1.0e-9;
        let decision = &decisions[0];
        assert_eq!(decision.expert_ids(), vec![2, 0]);
        assert_close(
            decision.weights()[0],
            scores[2] / denom,
            "biased expert keeps pre-bias weight in full combine path",
        );
        let expected = expected_weighted_sum(&expert_outputs, 0, decision, 4, hidden_dim);
        assert_slice_close(&combined, &expected, "bias-corrected combined output");
    }

    #[test]
    fn route_and_combine_e2e_batched_tokens() {
        let cfg = noaux_cfg(5, 2, ScoringFunc::Softmax, 1.25, None, None);
        let logits = [
            0.0, 1.0, 2.0, 3.0, 4.0, // token 0 -> experts 4, 3
            5.0, 4.0, 3.0, 2.0, 1.0, // token 1 -> experts 0, 1
        ];
        let hidden_dim = 2;
        let expert_outputs = expert_outputs(2, 5, hidden_dim);

        let (decisions, combined) =
            route_and_combine(&logits, &[0.0; 5], &expert_outputs, hidden_dim, &cfg).unwrap();

        assert_eq!(decisions[0].expert_ids(), vec![4, 3]);
        assert_eq!(decisions[1].expert_ids(), vec![0, 1]);
        for token in 0..2 {
            let expected =
                expected_weighted_sum(&expert_outputs, token, &decisions[token], 5, hidden_dim);
            assert_slice_close(
                &combined[token * hidden_dim..(token + 1) * hidden_dim],
                &expected,
                &format!("batched token {token} combined output"),
            );
        }
    }

    #[test]
    fn route_and_combine_rejects_bad_shapes() {
        let cfg = noaux_cfg(4, 2, ScoringFunc::Sigmoid, 1.0, None, None);
        let logits = [0.0, 0.0, 0.0, 0.0];
        assert!(
            route_and_combine(&logits, &[0.0; 4], &[0.0; 4], 0, &cfg).is_err(),
            "hidden_dim=0 is invalid"
        );
        assert!(
            route_and_combine(&logits, &[0.0; 4], &[0.0; 7], 2, &cfg).is_err(),
            "expert output shape must match [tokens, experts, hidden_dim]"
        );
    }
}
