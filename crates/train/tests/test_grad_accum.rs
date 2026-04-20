//! Pure logic tests for the `GradAccumulator` bookkeeper. No tensors, no
//! tape — just the state machine semantics the training loops rely on.

use train::GradAccumulator;

#[test]
fn n_equals_one_is_noop_per_micro_batch() {
    let mut acc = GradAccumulator::new(1);
    assert_eq!(acc.accum_steps(), 1);
    assert_eq!(acc.loss_scale(), 1.0);

    for i in 0..10 {
        let ready = acc.observe_and_check_ready();
        assert!(ready, "observation {i} should immediately be ready at N=1");
        // Counter is at 1 when we inspect, not above.
        // (Reset after every step keeps it bounded.)
        acc.reset_after_step();
    }
}

#[test]
fn n_equals_four_ready_only_on_fourth_observation() {
    let mut acc = GradAccumulator::new(4);
    assert_eq!(acc.accum_steps(), 4);
    assert_eq!(acc.loss_scale(), 0.25);

    // First window.
    assert!(!acc.observe_and_check_ready(), "obs #1 should be not ready");
    assert!(!acc.observe_and_check_ready(), "obs #2 should be not ready");
    assert!(!acc.observe_and_check_ready(), "obs #3 should be not ready");
    assert!(acc.observe_and_check_ready(), "obs #4 should be ready");
    acc.reset_after_step();

    // Second window — same pattern post-reset.
    assert!(
        !acc.observe_and_check_ready(),
        "obs #1 of window 2 should be not ready"
    );
    assert!(
        !acc.observe_and_check_ready(),
        "obs #2 of window 2 should be not ready"
    );
    assert!(
        !acc.observe_and_check_ready(),
        "obs #3 of window 2 should be not ready"
    );
    assert!(
        acc.observe_and_check_ready(),
        "obs #4 of window 2 should be ready"
    );
}

#[test]
fn accum_steps_and_loss_scale_match_construction() {
    for n in [1u64, 2, 3, 4, 8, 16, 32, 64] {
        let acc = GradAccumulator::new(n);
        assert_eq!(acc.accum_steps(), n, "accum_steps mismatch for N={n}");
        let expected = 1.0_f32 / n as f32;
        assert_eq!(acc.loss_scale(), expected, "loss_scale mismatch for N={n}");
    }
}

#[test]
#[should_panic(expected = "accum_steps >= 1")]
fn n_equals_zero_panics() {
    let _ = GradAccumulator::new(0);
}
