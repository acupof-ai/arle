//! CPU reference MoE routing — the device-independent ground truth.
//!
//! Every numeric here is a faithful port; the source line refs are inline. Two
//! legacy routers are reproduced:
//!
//! ## DSv4 (`crates/deepseek-spec/src/v4.rs` + `csrc/moe/dsv4_route.cu`)
//!
//! 1. `scores = router_scores_from_logits(logits)` — softmax / sigmoid /
//!    sqrt(softplus) over all experts (`v4.rs:305`).
//! 2. Select top-k with the bias-corrected key `scores[e] + bias[e]`, tie-break
//!    lower expert index (`topk_indices_by_score`, `v4.rs:556`). The bias is the
//!    `e_score_correction_bias` and is used **only** for selection.
//! 3. `normalize = scoring_func != softmax`; `denom = selected_sum + 1e-9` if
//!    normalize else `1.0`; `weight = scores[e] / denom * routed_scaling_factor`
//!    (`moe_routes_from_scores`, `v4.rs:382`). Note: softmax DSv4 does **not**
//!    renormalize the top-k in the router.
//!
//! ## Qwen3.6 (`infer/src/model/qwen35/moe.rs` + `csrc/moe/qwen36_route.cu`)
//!
//! 1. `scores = softmax(logits)` over all experts.
//! 2. Greedy top-k over the scores (no bias; the CUDA kernel feeds an all-zero
//!    bias). Masked argmax: strictly-greater wins, tie-break lower index
//!    (`dsv4_route.cu:421`).
//! 3. Weights are the raw softmax probs (`routed_scaling_factor = 1.0`,
//!    `denom = 1.0`), then **if `norm_topk_prob`** a separate renorm divides
//!    each selected weight by the sum of the selected weights
//!    (`qwen36_renorm_topk_weights`, `qwen36_route.cu:23`).
//!
//! Both share the same selection-then-weight skeleton; the only differences are
//! the scoring func, the selection key (bias vs none), and when the top-k
//! renormalization runs. They are unified in [`route`] driven by [`MoeConfig`].

use crate::config::{MoeConfig, ScoringFunc, TopkMethod};
use crate::error::{Result, bail};

/// One selected `(expert, weight)` pair for a single token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertWeight {
    /// The selected routed-expert index, in `0..num_experts`.
    pub expert: usize,
    /// The gating weight applied to this expert's output for this token.
    pub weight: f32,
}

/// The routing decision for one token: its selected experts and their weights,
/// in selection order (highest selection-key first).
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingDecision {
    /// `top_k` selected `(expert, weight)` pairs, ordered by descending
    /// selection key (matching the legacy serial selection order).
    pub experts: Vec<ExpertWeight>,
}

impl RoutingDecision {
    /// The selected expert indices, in selection order.
    #[must_use]
    pub fn expert_ids(&self) -> Vec<usize> {
        self.experts.iter().map(|ew| ew.expert).collect()
    }

    /// The selected weights, in selection order.
    #[must_use]
    pub fn weights(&self) -> Vec<f32> {
        self.experts.iter().map(|ew| ew.weight).collect()
    }
}

// ── Scoring primitives — ported verbatim from `v4.rs:511-543`. ──────────────

/// Stable softmax: subtract the max before `exp`, divide by the sum.
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

/// Turn the per-expert router logits into selection scores per the scoring func.
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

// ── Selection — ported from `topk_indices_by_score` (`v4.rs:556`). ──────────

/// Select the top-`k` expert indices by descending `scores[e] + bias[e]`,
/// tie-broken by lower expert index. Port of `topk_indices_by_score`.
///
/// `bias` must be either empty (no correction — additive identity) or exactly
/// `scores.len()` long. The mask, if present, marks ineligible experts (used by
/// group-limited routing) with `false`; ineligible experts are never selected.
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
        // Descending by corrected score; ties → lower index first.
        score_b.total_cmp(&score_a).then_with(|| a.cmp(&b))
    });
    indices.truncate(k);
    indices
}

// ── Group-limited routing — DeepSeek-V2/V3 `n_group`/`topk_group`. ──────────
//
// NOT exercised by either ARLE legacy router (DSv4 here is `noaux_tc` with no
// grouping, Qwen3.6 is plain greedy). Provided so this same CPU reference can
// verify a grouped GPU kernel if a future checkpoint sets `n_group`. The group
// score is the sum of the top-2 expert scores within the group — the canonical
// DeepSeek-V3 rule — and the `topk_group` highest-scoring groups stay eligible.

/// Compute the per-group selection score: the sum of the top-2 (corrected)
/// scores inside each contiguous equal-size group.
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

/// Route one token from its router logits.
///
/// `gate_logits` is `[num_experts]` (the `gate @ x` output for this token).
/// `bias` is the optional DSv4 `e_score_correction_bias` (`[num_experts]`, or
/// empty for none). Returns the [`RoutingDecision`] — selected experts and
/// their normalized weights, matching the legacy math exactly.
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

    // ── 1. Scores over all experts. ────────────────────────────────────────
    let scores = scores_from_logits(gate_logits, cfg.scoring_func)?;
    // The DSv4 reference asserts scores are finite and non-negative
    // (`v4.rs:334`). Softmax / sigmoid / sqrt(softplus) all satisfy this.
    if scores.iter().any(|v| !v.is_finite() || *v < 0.0) {
        bail!("router scores must be finite and non-negative");
    }

    // ── 2. Optional group-limited eligibility mask. ────────────────────────
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

    // ── 3. Top-k selection. ─────────────────────────────────────────────────
    // `noaux_tc` selects with the bias-corrected key; greedy selects with the
    // raw scores (Qwen3.6 supplies a zero bias, which is the same thing).
    let selection_bias: &[f32] = match cfg.topk_method {
        TopkMethod::NoAuxTc => bias,
        TopkMethod::Greedy => &[],
    };
    let selected = topk_indices_by_score(&scores, selection_bias, cfg.top_k, mask.as_deref());

    // ── 4. Weights. ─────────────────────────────────────────────────────────
    // DSv4 rule (`moe_routes_from_scores`, `v4.rs:386`): normalize iff the
    // scoring func is not softmax; weight = scores[e] / denom * scaling.
    // For softmax this leaves the raw probs (denom = 1.0). The Qwen3.6
    // `norm_topk_prob` renorm — which only the greedy/softmax path uses — is a
    // separate step below, mirroring that neither the DSv4 reference nor the
    // CUDA `dsv4_route` kernel renormalizes the softmax path inside the router.
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

    // ── 5. Qwen3.6 norm_topk_prob renorm. ───────────────────────────────────
    // Port of `qwen36_renorm_topk_weights` (`qwen36_route.cu:23`): divide each
    // selected weight by the sum of the selected weights so they sum to 1.
    //
    // This is the **Qwen3.6 path only** (`TopkMethod::Greedy`), applied as a
    // separate step gated on `norm_topk_prob`. The DSv4 reference router
    // (`TopkMethod::NoAuxTc`) never runs it — its softmax path is fixed to
    // denom = 1.0 (`v4.rs:386` `normalize = scoring_func != softmax`), and its
    // sigmoid/sqrtsoftplus path was already normalized above by the DSv4 denom
    // rule. Guarding on `!normalize` here is belt-and-suspenders: the Qwen path
    // is always softmax, so it is always un-normalized at this point.
    if cfg.norm_topk_prob && cfg.topk_method == TopkMethod::Greedy && !normalize {
        let sum: f32 = experts.iter().map(|ew| ew.weight).sum();
        let inv = if sum > 1.0e-20 { 1.0 / sum } else { 0.0 };
        for ew in &mut experts {
            ew.weight *= inv;
        }
    }

    Ok(RoutingDecision { experts })
}

/// Route a flat batch of token logits.
///
/// `gate_logits` is `[num_tokens * num_experts]` row-major (token-major, the
/// `gate @ x` layout the CUDA `dsv4_route` kernel consumes). `bias` is the
/// shared `[num_experts]` correction bias (or empty). Returns one
/// [`RoutingDecision`] per token.
pub fn route(gate_logits: &[f32], bias: &[f32], cfg: &MoeConfig) -> Result<Vec<RoutingDecision>> {
    cfg.validate()?;
    if cfg.num_experts == 0 || !gate_logits.len().is_multiple_of(cfg.num_experts) {
        bail!(
            "gate_logits length {} is not a multiple of num_experts {}",
            gate_logits.len(),
            cfg.num_experts
        );
    }
    let num_tokens = gate_logits.len() / cfg.num_experts;
    let mut out = Vec::with_capacity(num_tokens);
    for t in 0..num_tokens {
        let row = &gate_logits[t * cfg.num_experts..(t + 1) * cfg.num_experts];
        out.push(route_token(row, bias, cfg)?);
    }
    Ok(out)
}
