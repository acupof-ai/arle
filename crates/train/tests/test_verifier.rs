use train::verifier::{
    ArithmeticVerifier, CopyVerifier, MonotonicVerifier, PaletteVerifier, ReverseCopyVerifier,
    RewardConfig, ToolSuccessVerifier, Verifier, VerifierKind, WeightedEnsemble,
};

fn with_mask(prompt_ids: Vec<usize>, full_ids: Vec<usize>) -> (Vec<usize>, Vec<usize>, Vec<bool>) {
    let seq_len = prompt_ids.len();
    let prefix_len = seq_len / 2;
    let mut mask = vec![false; seq_len];
    for slot in mask.iter_mut().skip(prefix_len + 1) {
        *slot = true;
    }
    (prompt_ids, full_ids, mask)
}

#[test]
fn copy_verifier_rewards_exact_copy() {
    let (prompt_ids, full_ids, mask) =
        with_mask(vec![1, 2, 3, 255, 0, 0, 0], vec![1, 2, 3, 255, 1, 2, 3]);
    assert!((CopyVerifier.verify(&prompt_ids, &full_ids, &mask) - 1.0).abs() < 1e-6);

    let (prompt_ids, full_ids, mask) =
        with_mask(vec![1, 2, 3, 255, 0, 0, 0], vec![1, 2, 3, 255, 1, 9, 3]);
    let reward = CopyVerifier.verify(&prompt_ids, &full_ids, &mask);
    assert!((reward - 2.0 / 3.0).abs() < 1e-6, "got {reward}");
}

#[test]
fn reverse_copy_verifier_rewards_reversed_prefix() {
    let (prompt_ids, full_ids, mask) =
        with_mask(vec![1, 2, 3, 255, 0, 0, 0], vec![1, 2, 3, 255, 3, 2, 1]);
    assert!((ReverseCopyVerifier.verify(&prompt_ids, &full_ids, &mask) - 1.0).abs() < 1e-6);
}

#[test]
fn palette_verifier_counts_in_vocab_tokens() {
    let palette = PaletteVerifier::new(16, &[4, 5, 6]);
    let (prompt_ids, full_ids, mask) =
        with_mask(vec![0, 0, 0, 15, 0, 0, 0], vec![0, 0, 0, 15, 4, 6, 9]);
    let reward = palette.verify(&prompt_ids, &full_ids, &mask);
    assert!((reward - 2.0 / 3.0).abs() < 1e-6, "got {reward}");
}

#[test]
fn weighted_ensemble_sums_member_scores() {
    let (prompt_ids, full_ids, mask) =
        with_mask(vec![1, 2, 3, 255, 0, 0, 0], vec![1, 2, 3, 255, 1, 2, 3]);
    let ensemble = WeightedEnsemble::new()
        .with(0.75, CopyVerifier)
        .with(0.25, PaletteVerifier::new(256, &[1, 2, 3]));
    let reward = ensemble.verify(&prompt_ids, &full_ids, &mask);
    assert!((reward - 1.0).abs() < 1e-6, "got {reward}");
}

#[test]
fn empty_response_mask_returns_zero() {
    let prompt_ids = vec![1, 2, 3, 255, 0, 0, 0];
    let full_ids = vec![1, 2, 3, 255, 0, 0, 0];
    let mask = vec![false; 7];
    assert_eq!(CopyVerifier.verify(&prompt_ids, &full_ids, &mask), 0.0);
    assert_eq!(
        ReverseCopyVerifier.verify(&prompt_ids, &full_ids, &mask),
        0.0
    );
    assert_eq!(
        PaletteVerifier::new(256, &[1]).verify(&prompt_ids, &full_ids, &mask),
        0.0
    );
}

#[test]
fn reward_config_matches_fluent_builder() {
    let (prompt_ids, full_ids, mask) =
        with_mask(vec![1, 2, 3, 255, 0, 0, 0], vec![1, 2, 3, 255, 1, 2, 3]);

    let fluent = WeightedEnsemble::new()
        .with(0.75, CopyVerifier)
        .with(0.25, PaletteVerifier::new(256, &[1, 2, 3]));

    let config = RewardConfig::new().push(0.75, VerifierKind::Copy).push(
        0.25,
        VerifierKind::Palette {
            allowed_tokens: vec![1, 2, 3],
        },
    );
    let from_config = WeightedEnsemble::from_config(&config, 256);

    let fluent_reward = fluent.verify(&prompt_ids, &full_ids, &mask);
    let config_reward = from_config.verify(&prompt_ids, &full_ids, &mask);
    assert!(
        (fluent_reward - config_reward).abs() < 1e-6,
        "fluent {fluent_reward} != config {config_reward}",
    );
}

#[test]
fn monotonic_verifier_rewards_strictly_increasing_response() {
    let (prompt_ids, full_ids, mask) =
        with_mask(vec![0, 0, 0, 255, 0, 0, 0], vec![0, 0, 0, 255, 2, 5, 9]);
    let reward = MonotonicVerifier.verify(&prompt_ids, &full_ids, &mask);
    assert!((reward - 1.0).abs() < 1e-6, "got {reward}");

    // A single out-of-order position drops reward to 2/3.
    let (prompt_ids, full_ids, mask) =
        with_mask(vec![0, 0, 0, 255, 0, 0, 0], vec![0, 0, 0, 255, 2, 1, 9]);
    let reward = MonotonicVerifier.verify(&prompt_ids, &full_ids, &mask);
    assert!((reward - 2.0 / 3.0).abs() < 1e-6, "got {reward}");
}

#[test]
fn tool_success_verifier_rewards_sentinel_presence() {
    let (prompt_ids, full_ids, mask) =
        with_mask(vec![0, 0, 0, 255, 0, 0, 0], vec![0, 0, 0, 255, 4, 7, 7]);
    let sentinel = ToolSuccessVerifier::new(7);
    let reward = sentinel.verify(&prompt_ids, &full_ids, &mask);
    assert!((reward - 1.0).abs() < 1e-6);

    let absent = ToolSuccessVerifier::new(9);
    let reward = absent.verify(&prompt_ids, &full_ids, &mask);
    assert_eq!(reward, 0.0);
}

#[test]
fn arithmetic_verifier_scores_correct_digit_answer() {
    // Prompt: 1 2 [+] 3 4 [=]  (i.e. 12 + 34 = 46); response: 4 6
    let prompt_ids = vec![1, 2, 10, 3, 4, 11];
    let full_ids = vec![1, 2, 10, 3, 4, 11, 4, 6];
    let mask = vec![false, false, false, false, false, false, true, true];
    let verifier = ArithmeticVerifier::new(
        /*base*/ 10, /*+*/ 10, /**/ 12, /*=*/ 11, /*len*/ 2,
    );
    let reward = verifier.verify(&prompt_ids, &full_ids, &mask);
    assert!((reward - 1.0).abs() < 1e-6);

    // Wrong answer → 0.
    let full_ids = vec![1, 2, 10, 3, 4, 11, 9, 9];
    let reward = verifier.verify(&prompt_ids, &full_ids, &mask);
    assert_eq!(reward, 0.0);

    // Product: 3 × 4 = 12 → digits [1, 2]
    let prompt_ids = vec![3, 12, 4, 11];
    let full_ids = vec![3, 12, 4, 11, 1, 2];
    let mask = vec![false, false, false, false, true, true];
    let reward = verifier.verify(&prompt_ids, &full_ids, &mask);
    assert!((reward - 1.0).abs() < 1e-6);
}

#[test]
fn reward_config_covers_new_archetype_kinds() {
    let (prompt_ids, full_ids, mask) =
        with_mask(vec![0, 0, 0, 255, 0, 0, 0], vec![0, 0, 0, 255, 2, 5, 9]);
    let config = RewardConfig::new()
        .push(0.5, VerifierKind::Monotonic)
        .push(0.5, VerifierKind::ToolSuccess { sentinel: 9 });
    let ensemble = WeightedEnsemble::from_config(&config, 256);
    let reward = ensemble.verify(&prompt_ids, &full_ids, &mask);
    // Monotonic → 1.0, ToolSuccess → 1.0 → weighted sum 1.0.
    assert!((reward - 1.0).abs() < 1e-6, "got {reward}");
}

#[test]
fn reward_config_supports_reverse_copy() {
    let (prompt_ids, full_ids, mask) =
        with_mask(vec![1, 2, 3, 255, 0, 0, 0], vec![1, 2, 3, 255, 3, 2, 1]);
    let config = RewardConfig::new().push(1.0, VerifierKind::ReverseCopy);
    let ensemble = WeightedEnsemble::from_config(&config, 256);
    let reward = ensemble.verify(&prompt_ids, &full_ids, &mask);
    assert!((reward - 1.0).abs() < 1e-6, "got {reward}");
}
