use train::reward::{discounted_returns, group_normalize, returns_to_per_position};

#[test]
fn discounted_returns_undiscounted_matches_suffix_sum() {
    let returns = discounted_returns(&[1.0, 2.0, 3.0], 1.0);
    assert_eq!(returns, vec![6.0, 5.0, 3.0]);
}

#[test]
fn discounted_returns_with_gamma() {
    let returns = discounted_returns(&[1.0, 2.0, 3.0], 0.5);
    // G_2 = 3; G_1 = 2 + 0.5*3 = 3.5; G_0 = 1 + 0.5*3.5 = 2.75
    let expected = [2.75, 3.5, 3.0];
    for (actual, expected) in returns.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "got {actual} expected {expected}"
        );
    }
}

#[test]
fn returns_to_per_position_fans_out() {
    let returns = vec![2.0, 5.0];
    let boundaries = vec![(1, 3), (5, 7)];
    let per_position = returns_to_per_position(&returns, &boundaries, 8);
    assert_eq!(per_position, vec![0.0, 2.0, 2.0, 0.0, 0.0, 5.0, 5.0, 0.0]);
}

#[test]
fn group_normalize_zero_mean_unit_std_per_group() {
    let returns = vec![1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0];
    let advantages = group_normalize(&returns, 4);

    // First group: std ≈ 1.118, advantages ≈ -1.34, -0.45, 0.45, 1.34
    let group1_mean: f32 = advantages[..4].iter().sum::<f32>() / 4.0;
    assert!(group1_mean.abs() < 1e-4);
    // Second group: all identical → std = 0 → all zeros (divided by eps).
    for value in &advantages[4..] {
        assert!(value.abs() < 1e-3, "got {value}");
    }
}
