//! CPU-verifiable unit tests for the MoE routing reference.
//!
//! These pin the legacy numerics: every expected value is derived from the
//! ported formula (and several mirror the DSv4 reference's own arithmetic as an
//! independent oracle), not from a captured GPU run.

use super::*;

const EPS: f32 = 1e-6;

fn assert_close(a: f32, b: f32, ctx: &str) {
    assert!((a - b).abs() <= EPS, "{ctx}: {a} vs {b}");
}

// ── Qwen3.6: plain softmax + greedy top-k. ──────────────────────────────────

fn qwen36_cfg(num_experts: usize, top_k: usize, norm: bool) -> MoeConfig {
    MoeConfig::qwen36(num_experts, top_k, norm, /*hidden_size=*/ 16)
}

/// top-k selects the k highest gate logits (softmax is monotone, so highest
/// logit ⇒ highest prob).
#[test]
fn topk_selects_highest_logits() {
    let cfg = qwen36_cfg(6, 3, false);
    // Logits: expert 4 > 1 > 5 > others.
    let logits = [0.1, 2.0, 0.0, -1.0, 3.0, 1.5];
    let dec = route_token(&logits, &[], &cfg).unwrap();
    assert_eq!(
        dec.expert_ids(),
        vec![4, 1, 5],
        "highest-logit experts in order"
    );
}

/// Qwen3.6 softmax weights with `norm_topk_prob = false` are the raw softmax
/// probabilities (denom = 1.0, scaling = 1.0).
#[test]
fn qwen36_raw_softmax_weights() {
    let cfg = qwen36_cfg(4, 2, false);
    let logits = [1.0, 2.0, 0.5, -0.5];
    let probs = stable_softmax(&logits);
    let dec = route_token(&logits, &[], &cfg).unwrap();
    assert_eq!(dec.expert_ids(), vec![1, 0]);
    assert_close(dec.weights()[0], probs[1], "raw prob expert 1");
    assert_close(dec.weights()[1], probs[0], "raw prob expert 0");
}

/// `norm_topk_prob = true` (Qwen3.6) renormalizes the selected softmax probs to
/// sum to 1 — port of `qwen36_renorm_topk_weights`.
#[test]
fn qwen36_norm_topk_prob_renorm() {
    let cfg = qwen36_cfg(4, 2, true);
    let logits = [1.0, 2.0, 0.5, -0.5];
    let probs = stable_softmax(&logits);
    let sel_sum = probs[1] + probs[0];
    let dec = route_token(&logits, &[], &cfg).unwrap();
    let w = dec.weights();
    assert_close(w[0], probs[1] / sel_sum, "renorm expert 1");
    assert_close(w[1], probs[0] / sel_sum, "renorm expert 0");
    assert_close(w[0] + w[1], 1.0, "renormed weights sum to 1");
}

// ── DSv4: scoring funcs + noaux_tc bias selection + normalize rule. ─────────

fn dsv4_cfg(
    num_experts: usize,
    top_k: usize,
    scoring: ScoringFunc,
    scaling: f32,
    norm_topk_prob: bool,
) -> MoeConfig {
    MoeConfig {
        num_experts,
        num_shared_experts: 1,
        top_k,
        scoring_func: scoring,
        topk_method: TopkMethod::NoAuxTc,
        norm_topk_prob,
        routed_scaling_factor: scaling,
        n_group: None,
        topk_group: None,
        hidden_size: 16,
    }
}

/// Independent oracle: re-implements `moe_routes_from_scores` (`v4.rs:382`)
/// directly to cross-check the crate's output for the sigmoid/sqrtsoftplus
/// (always-normalize) DSv4 paths.
fn dsv4_oracle(
    logits: &[f32],
    bias: &[f32],
    scoring: ScoringFunc,
    top_k: usize,
    scaling: f32,
) -> (Vec<usize>, Vec<f32>) {
    let scores = scores_from_logits(logits, scoring).unwrap();
    // topk_indices_by_score: sort by score+bias desc, tie lower index.
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        let sb = scores[b] + bias[b];
        let sa = scores[a] + bias[a];
        sb.total_cmp(&sa).then_with(|| a.cmp(&b))
    });
    idx.truncate(top_k);
    let sel_sum: f32 = idx.iter().map(|&e| scores[e]).sum();
    let normalize = scoring != ScoringFunc::Softmax;
    let denom = if normalize { sel_sum + 1e-9 } else { 1.0 };
    let w: Vec<f32> = idx.iter().map(|&e| scores[e] / denom * scaling).collect();
    (idx, w)
}

/// DSv4 `noaux_tc`: the bias steers *selection*, but the weight reads the
/// un-biased score. A large positive bias on an otherwise-low expert pulls it
/// into the top-k.
#[test]
fn dsv4_noaux_tc_bias_drives_selection_not_weight() {
    let cfg = dsv4_cfg(4, 2, ScoringFunc::Sigmoid, 1.0, false);
    let logits = [0.0, 0.0, -5.0, 0.0]; // expert 2 has the lowest score
    let bias = [0.0, 0.0, 10.0, 0.0]; // but a huge selection bias
    let scores = scores_from_logits(&logits, ScoringFunc::Sigmoid).unwrap();
    let dec = route_token(&logits, &bias, &cfg).unwrap();
    assert_eq!(dec.experts[0].expert, 2, "bias pulls expert 2 to the top");
    // Its weight uses the *un-biased* low score, not the corrected key.
    let sel_sum: f32 = dec.expert_ids().iter().map(|&e| scores[e]).sum();
    assert_close(
        dec.experts[0].weight,
        scores[2] / (sel_sum + 1e-9),
        "weight uses unbiased score, normalized",
    );
}

/// DSv4 sigmoid path always normalizes over the selected scores
/// (`denom = selected_sum + 1e-9`), independent of `norm_topk_prob`.
#[test]
fn dsv4_sigmoid_always_normalizes() {
    let logits = [1.5, -0.5, 2.0, 0.3, -1.0];
    let bias = [0.0; 5];
    for &scaling in &[1.0_f32, 1.5] {
        let cfg = dsv4_cfg(5, 3, ScoringFunc::Sigmoid, scaling, false);
        let (oid, ow) = dsv4_oracle(&logits, &bias, ScoringFunc::Sigmoid, 3, scaling);
        let dec = route_token(&logits, &bias, &cfg).unwrap();
        assert_eq!(dec.expert_ids(), oid, "sigmoid selection matches oracle");
        for (i, (&w, &o)) in dec.weights().iter().zip(ow.iter()).enumerate() {
            assert_close(w, o, &format!("sigmoid weight[{i}] scaling={scaling}"));
        }
    }
}

/// DSv4 sqrt(softplus) path matches the oracle (selection + always-normalize).
#[test]
fn dsv4_sqrtsoftplus_matches_oracle() {
    let logits = [0.7, 3.1, -2.0, 1.1, 0.0, 25.0];
    let bias = [0.1, -0.2, 0.0, 0.05, 0.0, 0.0];
    let cfg = dsv4_cfg(6, 4, ScoringFunc::SqrtSoftplus, 1.5, true);
    let (oid, ow) = dsv4_oracle(&logits, &bias, ScoringFunc::SqrtSoftplus, 4, 1.5);
    let dec = route_token(&logits, &bias, &cfg).unwrap();
    assert_eq!(
        dec.expert_ids(),
        oid,
        "sqrtsoftplus selection matches oracle"
    );
    for (i, (&w, &o)) in dec.weights().iter().zip(ow.iter()).enumerate() {
        assert_close(w, o, &format!("sqrtsoftplus weight[{i}]"));
    }
    // `>20` softplus cutoff: expert 5 (logit 25) ⇒ softplus≈25 ⇒ score≈5.
    let scores = scores_from_logits(&logits, ScoringFunc::SqrtSoftplus).unwrap();
    assert_close(
        scores[5],
        25.0_f32.sqrt(),
        "softplus identity cutoff at logit 25",
    );
}

/// The `MoeConfig::dsv4` constructor wires the DSv4 router fields and routes
/// identically to the hand-built `dsv4_cfg` fixture (sqrtsoftplus + noaux_tc +
/// bias-driven selection, config scaling, no grouping).
#[test]
fn dsv4_constructor_matches_fixture_and_routes() {
    let cfg = MoeConfig::dsv4(
        /*num_experts=*/ 6, /*num_shared_experts=*/ 1, /*top_k=*/ 4,
        /*routed_scaling_factor=*/ 1.5, /*hidden_size=*/ 16,
    );
    assert_eq!(cfg.scoring_func, ScoringFunc::SqrtSoftplus);
    assert_eq!(cfg.topk_method, TopkMethod::NoAuxTc);
    assert_eq!(cfg.num_shared_experts, 1);
    assert_eq!(cfg.routed_scaling_factor, 1.5);
    assert!(cfg.n_group.is_none() && cfg.topk_group.is_none());
    cfg.validate().unwrap();

    // Routes identically to the fixture + oracle for a biased top-k selection.
    let logits = [0.7, 3.1, -2.0, 1.1, 0.0, 25.0];
    let bias = [0.1, -0.2, 0.0, 0.05, 0.0, 0.0];
    let (oid, ow) = dsv4_oracle(&logits, &bias, ScoringFunc::SqrtSoftplus, 4, 1.5);
    let dec = route_token(&logits, &bias, &cfg).unwrap();
    assert_eq!(dec.expert_ids(), oid, "ctor selection matches oracle");
    for (i, (&w, &o)) in dec.weights().iter().zip(ow.iter()).enumerate() {
        assert_close(w, o, &format!("ctor weight[{i}]"));
    }
}

/// DSv4 softmax path does NOT renormalize the top-k in the router (denom=1.0),
/// even though `norm_topk_prob = true` — the renorm is a Qwen-only step. This is
/// the faithful legacy behavior (`v4.rs:386` normalize = scoring != softmax).
#[test]
fn dsv4_softmax_does_not_renorm_topk() {
    let logits = [1.0, 2.0, 0.5, -0.5];
    let probs = stable_softmax(&logits);
    // DSv4 config with softmax + norm_topk_prob true: still no renorm because
    // topk_method is NoAuxTc (DSv4 path), not the Qwen greedy path.
    let cfg = dsv4_cfg(4, 2, ScoringFunc::Softmax, 1.0, true);
    let dec = route_token(&logits, &[0.0; 4], &cfg).unwrap();
    assert_close(
        dec.weights()[0],
        probs[1],
        "softmax DSv4 weight = raw prob (no renorm)",
    );
    assert_close(
        dec.weights()[1],
        probs[0],
        "softmax DSv4 weight = raw prob (no renorm)",
    );
}

/// `routed_scaling_factor` multiplies every routed weight.
#[test]
fn routed_scaling_factor_applied() {
    let logits = [1.5, -0.5, 2.0, 0.3];
    let bias = [0.0; 4];
    let base = dsv4_cfg(4, 2, ScoringFunc::Sigmoid, 1.0, false);
    let scaled = dsv4_cfg(4, 2, ScoringFunc::Sigmoid, 2.5, false);
    let d0 = route_token(&logits, &bias, &base).unwrap();
    let d1 = route_token(&logits, &bias, &scaled).unwrap();
    for (a, b) in d0.weights().iter().zip(d1.weights().iter()) {
        assert_close(*b, a * 2.5, "scaling multiplies weight");
    }
}

// ── Group-limited routing (DeepSeek-V2/V3 n_group / topk_group). ────────────

/// With grouping, experts outside the kept groups are never selected.
#[test]
fn group_limited_selects_within_kept_groups() {
    // 6 experts, 3 groups of 2. Group 1 (experts 2,3) has the strongest pair.
    let logits = [0.0, 0.1, 5.0, 4.0, 0.2, 0.3];
    let cfg = MoeConfig {
        num_experts: 6,
        num_shared_experts: 0,
        top_k: 2,
        scoring_func: ScoringFunc::Softmax,
        topk_method: TopkMethod::Greedy,
        norm_topk_prob: false,
        routed_scaling_factor: 1.0,
        n_group: Some(3),
        topk_group: Some(1), // keep only the single best group
        hidden_size: 16,
    };
    let dec = route_token(&logits, &[], &cfg).unwrap();
    let ids = dec.expert_ids();
    assert_eq!(
        ids,
        vec![2, 3],
        "only the best group's experts are eligible"
    );
}

/// The group mask keeps exactly `topk_group` groups by top-2-sum score.
#[test]
fn group_limited_mask_keeps_topk_group() {
    // 3 groups of 2. corrected scores below; top-2-sum per group:
    // g0 = 1+0.5=1.5, g1 = 9+8=17, g2 = 2+0=2. Keep top-2 groups ⇒ g1, g2.
    let corrected = [1.0, 0.5, 9.0, 8.0, 2.0, 0.0];
    let mask = group_limited_mask(&corrected, 3, 2);
    assert_eq!(mask, vec![false, false, true, true, true, true]);
}

// ── Shared-expert accounting. ───────────────────────────────────────────────

#[test]
fn shared_expert_accounted() {
    let q = qwen36_cfg(8, 2, true);
    assert!(
        q.has_shared_expert(),
        "Qwen3.6 has one always-on shared expert"
    );
    assert_eq!(q.num_shared_experts, 1);

    let mut d = dsv4_cfg(8, 2, ScoringFunc::Sigmoid, 1.0, false);
    d.num_shared_experts = 0;
    assert!(
        !d.has_shared_expert(),
        "DSv4 with n_shared_experts=0 has none"
    );
    d.num_shared_experts = 2;
    assert!(d.has_shared_expert());
    // Shared experts never appear in the routed decision.
    let dec = route_token(&[1.0; 8], &[0.0; 8], &d).unwrap();
    assert!(
        dec.expert_ids().iter().all(|&e| e < d.num_experts),
        "routed ids are routed experts only, never shared"
    );
}

// ── Edge cases. ─────────────────────────────────────────────────────────────

/// k == num_experts selects every expert; softmax weights still sum to 1
/// (they are the full distribution). Selection order is by descending logit.
#[test]
fn k_equals_num_experts() {
    let cfg = qwen36_cfg(4, 4, false);
    let logits = [0.3, 1.0, -0.2, 0.7];
    let dec = route_token(&logits, &[], &cfg).unwrap();
    assert_eq!(dec.experts.len(), 4, "all experts selected");
    assert_eq!(dec.expert_ids(), vec![1, 3, 0, 2], "descending-logit order");
    let total: f32 = dec.weights().iter().sum();
    assert_close(total, 1.0, "full softmax distribution sums to 1");
}

/// Ties in the selection key break toward the lower expert index — both for the
/// raw-score greedy path and the bias-corrected noaux_tc path.
#[test]
fn ties_break_to_lower_index() {
    // All-equal logits: greedy picks the lowest indices.
    let cfg = qwen36_cfg(5, 3, false);
    let dec = route_token(&[2.0; 5], &[], &cfg).unwrap();
    assert_eq!(
        dec.expert_ids(),
        vec![0, 1, 2],
        "equal scores ⇒ lowest indices"
    );

    // noaux_tc: equal corrected keys also break to the lower index.
    let dcfg = dsv4_cfg(5, 2, ScoringFunc::Sigmoid, 1.0, false);
    // logits chosen so score+bias ties between experts 3 and 4; lower (3) wins.
    let logits = [-9.0, -9.0, -9.0, 0.0, 0.0];
    let bias = [0.0, 0.0, 0.0, 0.0, 0.0];
    let dec = route_token(&logits, &bias, &dcfg).unwrap();
    assert_eq!(dec.expert_ids(), vec![3, 4], "tie ⇒ lower index first");
}

/// Batched `route` is consistent with per-token `route_token`.
#[test]
fn batch_route_matches_per_token() {
    let cfg = qwen36_cfg(4, 2, true);
    let t0 = [1.0, 2.0, 0.5, -0.5];
    let t1 = [-1.0, 0.0, 3.0, 1.0];
    let flat: Vec<f32> = t0.iter().chain(t1.iter()).copied().collect();
    let batch = route(&flat, &[], &cfg).unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0], route_token(&t0, &[], &cfg).unwrap());
    assert_eq!(batch[1], route_token(&t1, &[], &cfg).unwrap());
}

// ── Config validation. ──────────────────────────────────────────────────────

#[test]
fn config_validation_rejects_bad_shapes() {
    let mut cfg = qwen36_cfg(4, 2, false);
    cfg.top_k = 5; // > num_experts
    assert!(cfg.validate().is_err(), "top_k > num_experts rejected");

    cfg.top_k = 0;
    assert!(cfg.validate().is_err(), "top_k == 0 rejected");

    cfg.num_experts = 0;
    assert!(cfg.validate().is_err(), "num_experts == 0 rejected");

    // Half-specified grouping is rejected.
    let mut g = qwen36_cfg(8, 2, false);
    g.n_group = Some(2);
    g.topk_group = None;
    assert!(g.validate().is_err(), "n_group without topk_group rejected");

    // num_experts not divisible by n_group rejected.
    g.n_group = Some(3);
    g.topk_group = Some(2);
    assert!(g.validate().is_err(), "8 not divisible by 3 rejected");
}

#[test]
fn scoring_and_topk_parse_from_config_strings() {
    assert_eq!(
        ScoringFunc::from_config_str("softmax").unwrap(),
        ScoringFunc::Softmax
    );
    assert_eq!(
        ScoringFunc::from_config_str("sigmoid").unwrap(),
        ScoringFunc::Sigmoid
    );
    assert_eq!(
        ScoringFunc::from_config_str("sqrtsoftplus").unwrap(),
        ScoringFunc::SqrtSoftplus
    );
    assert!(ScoringFunc::from_config_str("relu").is_err());
    assert_eq!(ScoringFunc::Softmax.scoring_kind(), 0);
    assert_eq!(ScoringFunc::Sigmoid.scoring_kind(), 1);
    assert_eq!(ScoringFunc::SqrtSoftplus.scoring_kind(), 2);

    assert_eq!(
        TopkMethod::from_config_str("noaux_tc").unwrap(),
        TopkMethod::NoAuxTc
    );
    assert_eq!(
        TopkMethod::from_config_str("greedy").unwrap(),
        TopkMethod::Greedy
    );
}

/// Non-finite logits are rejected (mirrors the DSv4 reference guard).
#[test]
fn rejects_non_finite_logits() {
    let cfg = qwen36_cfg(4, 2, false);
    assert!(route_token(&[1.0, f32::NAN, 0.0, 0.0], &[], &cfg).is_err());
    assert!(route_token(&[1.0, f32::INFINITY, 0.0, 0.0], &[], &cfg).is_err());
}
