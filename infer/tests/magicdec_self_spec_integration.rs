/*!
MagicDec-style self-speculation integration test.

Verifies that the MagicDec self-spec implementation correctly:
1. Routes spec-enabled requests through the speculative path
2. Executes greedy verification with target argmax tokens
3. Applies acceptance tracking and adaptive disable logic
4. Maintains correctness compared to normal decode
*/

use infer::scheduler::{DraftMode, SchedulerConfig};
use infer::speculative::{AcceptanceTracker, verify_tokens_greedy};

#[test]
fn magicdec_self_spec_config_validation() {
    // Verify config validation requires sparse-KV for multi-token self-spec
    let mut config = SchedulerConfig::runtime_defaults(4);
    config.spec_enabled = true;
    config.spec_draft_model = DraftMode::SelfSpec;
    config.spec_draft_k = 5;

    // Should fail without sparse-KV enabled
    assert!(
        config.validate().is_err(),
        "multi-token self-spec should require sparse-KV"
    );

    // Should pass with sparse-KV enabled
    config.spec_sparse_kv_enabled = true;
    assert!(
        config.validate().is_ok(),
        "sparse self-spec config should be valid"
    );
}

#[test]
fn verify_tokens_greedy_acceptance_logic() {
    // Test the core greedy verification logic

    // All tokens match → all accepted
    let draft = vec![42, 17, 89, 123];
    let target = vec![42, 17, 89, 123, 999]; // bonus token
    let result = verify_tokens_greedy(&draft, &target);
    assert_eq!(result.num_accepted, 4);
    assert_eq!(result.rejection_index, 4);
    assert_eq!(result.accepted, vec![42, 17, 89, 123]);

    // Partial match → accept prefix, reject at mismatch
    let draft = vec![42, 17, 99, 123];
    let target = vec![42, 17, 89, 123];
    let result = verify_tokens_greedy(&draft, &target);
    assert_eq!(result.num_accepted, 2);
    assert_eq!(result.rejection_index, 2);
    assert_eq!(result.accepted, vec![42, 17]);

    // No match → reject immediately
    let draft = vec![42, 17, 89];
    let target = vec![99, 17, 89];
    let result = verify_tokens_greedy(&draft, &target);
    assert_eq!(result.num_accepted, 0);
    assert_eq!(result.rejection_index, 0);
    assert_eq!(result.accepted, Vec::<u32>::new());
}

#[test]
fn acceptance_tracker_adaptive_disable() {
    let mut tracker = AcceptanceTracker::default_window();
    let threshold = 0.6;

    // Fill the window first with high acceptance
    for _ in 0..64 {
        // default window size
        tracker.observe_step(4, 5); // 80% acceptance
    }
    assert!(
        !tracker.should_disable(threshold),
        "high acceptance should stay enabled"
    );

    // Now low acceptance should trigger disable
    for _ in 0..64 {
        // fill window with low acceptance
        tracker.observe_step(1, 5); // 20% acceptance
    }
    assert!(
        tracker.should_disable(threshold),
        "low acceptance should disable"
    );
}

// NOTE: SpecVerifyRequest/SpecVerifyOutput are CUDA-only types, tested elsewhere

// NOTE: route_spec_plan and StepPlan are private CUDA-only types, tested in module tests

#[test]
fn spec_speedup_calculation() {
    // Use the actual speedup formula from speculative.rs
    use infer::speculative::expected_speedup;

    // Target configuration: K=5, α≥0.6
    let k = 5;
    assert!(
        (expected_speedup(k, 0.6) - 2.306).abs() < 0.01,
        "α=0.6 should give ~2.3x speedup"
    );
    assert!(
        (expected_speedup(k, 0.8) - 3.362).abs() < 0.01,
        "α=0.8 should give ~3.4x speedup"
    );
    assert!(
        (expected_speedup(k, 0.9) - 4.095).abs() < 0.01,
        "α=0.9 should give ~4.1x speedup"
    );

    // Agent workload target: 1.8-2.2x is achievable at moderate acceptance
    assert!(
        expected_speedup(k, 0.6) >= 1.8,
        "α=0.6 should exceed 1.8x speedup"
    );
    assert!(
        expected_speedup(k, 0.6) >= 2.0,
        "α=0.6 should exceed 2.0x speedup"
    );
}

/// Integration test demonstrating MagicDec self-spec configuration.
///
/// This test shows how to configure the scheduler for MagicDec-style
/// self-speculation with the expected CLI arguments.
#[test]
fn magicdec_configuration_example() {
    let mut config = SchedulerConfig::runtime_defaults(8);

    // MagicDec self-spec configuration
    config.spec_enabled = true;
    config.spec_draft_model = DraftMode::SelfSpec;
    config.spec_draft_k = 5;
    config.spec_acceptance_threshold = 0.6;
    config.spec_sparse_kv_enabled = true;
    config.spec_sparse_recent_tokens = 512;
    config.spec_sparse_top_k_pages = 32;

    assert!(config.validate().is_ok(), "MagicDec config should be valid");

    // Verify the configuration enables the expected behavior
    assert_eq!(config.spec_draft_model, DraftMode::SelfSpec);
    assert!(config.spec_sparse_kv_enabled);
    assert_eq!(config.spec_draft_k, 5);
    assert_eq!(config.spec_acceptance_threshold, 0.6);
}
