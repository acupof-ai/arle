use train::verifier::{
    CopyVerifier, PaletteVerifier, ReverseCopyVerifier, Verifier, WeightedEnsemble,
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
