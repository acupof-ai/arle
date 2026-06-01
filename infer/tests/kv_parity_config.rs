#[path = "support/kv_parity_config.rs"]
mod kv_parity_config;

use kv_parity_config::{FULL_DEFAULT_MAX_TOKENS, SMOKE_DEFAULT_MAX_TOKENS, max_tokens_for_config};

#[test]
fn kv_parity_default_uses_full_horizon() {
    let full_tokens = max_tokens_for_config(None, None).expect("default tokens");
    let smoke_tokens = max_tokens_for_config(None, Some("smoke")).expect("smoke tokens");
    assert_eq!(full_tokens, FULL_DEFAULT_MAX_TOKENS);
    assert!(full_tokens > smoke_tokens);
}

#[test]
fn kv_parity_smoke_profile_is_explicit() {
    assert_eq!(
        max_tokens_for_config(None, Some("smoke")).expect("smoke tokens"),
        SMOKE_DEFAULT_MAX_TOKENS
    );
}

#[test]
fn kv_parity_max_tokens_overrides_profile() {
    assert_eq!(
        max_tokens_for_config(Some("16"), Some("smoke")).expect("override tokens"),
        16
    );
}

#[test]
fn kv_parity_rejects_zero_max_tokens() {
    let err = max_tokens_for_config(Some("0"), None).expect_err("zero must fail");
    assert!(err.to_string().contains("must be > 0"));
}
