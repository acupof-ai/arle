use super::{
    BlockSelectionIntent, can_publish_prefix_pages, host_spill_target_bytes, is_full_sealed_prefix,
    prefix_cache_retain_hard_cap_pages, sealed_block_token_count,
};

const HARD_CAP: f64 = 0.90;

#[test]
fn retain_hard_cap_is_ninety_percent_of_pool() {
    assert_eq!(prefix_cache_retain_hard_cap_pages(100, HARD_CAP), 90);
    assert_eq!(prefix_cache_retain_hard_cap_pages(16, HARD_CAP), 14);
}

#[test]
fn publish_is_denied_once_new_pages_cross_hard_cap() {
    assert!(can_publish_prefix_pages(70, 100, 20, HARD_CAP));
    assert!(can_publish_prefix_pages(80, 100, 10, HARD_CAP));
    assert!(!can_publish_prefix_pages(81, 100, 10, HARD_CAP));
    assert!(!can_publish_prefix_pages(90, 100, 1, HARD_CAP));
}

#[test]
fn sealed_prefix_helpers_require_full_blocks() {
    assert_eq!(sealed_block_token_count(16, 3), 48);
    assert!(is_full_sealed_prefix(48, 16, 3));
    assert!(!is_full_sealed_prefix(47, 16, 3));
    assert!(!is_full_sealed_prefix(0, 16, 0));
}

#[test]
fn spill_target_requires_high_water_on_hot_path() {
    assert_eq!(
        host_spill_target_bytes(199, 1000, 0.20, 0.10, BlockSelectionIntent::Spill),
        0
    );
    assert_eq!(
        host_spill_target_bytes(200, 1000, 0.20, 0.10, BlockSelectionIntent::Spill),
        100
    );
}

#[test]
fn drain_target_uses_low_water_even_below_high_water() {
    assert_eq!(
        host_spill_target_bytes(199, 1000, 0.20, 0.10, BlockSelectionIntent::Drain),
        99
    );
    assert_eq!(
        host_spill_target_bytes(100, 1000, 0.20, 0.10, BlockSelectionIntent::Drain),
        0
    );
}
