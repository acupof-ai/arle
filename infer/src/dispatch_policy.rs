//! Dispatch policy — the single source of truth for environment-gated GPU
//! dispatch knobs.
//!
//! Governance background:
//! [`docs/reviews/2026-05-29-gpu-dispatch-governance-analysis.md`] and
//! [`docs/plans/gpu-dispatch-governance.md`]. The execution path used to be an
//! emergent property of `std::env::var` reads scattered across the hot path —
//! some cached behind ad-hoc `OnceLock`s, some re-read on *every* dispatch call.
//! This module is the **Declare** gate: every dispatch-affecting env knob is
//! parsed exactly once, here, into one inspectable struct.
//!
//! This is the ops-layer subset (the kernel-selection knobs in
//! `ops/linear.rs` + `ops/attention.rs`). Model-layer knobs
//! (`INFER_PREFILL_GRAPH`, `INFER_BYPASS_TILELANG_PREFILL`, the `ARLE_DSV4_*`
//! family, `INFER_QUANT_FORMAT_OVERRIDE`) migrate here in a follow-up tranche.
//!
//! Behaviour is preserved bit-for-bit: each field reproduces the exact accepted
//! token set of the call site it replaced. Knobs are deliberately NOT unified
//! onto one truthy parser, because the legacy sites disagreed
//! (`INFER_R4_W4A16_GEMV_OVERRIDE` accepted only `"1"`;
//! `INFER_TILELANG_BF16_SPLIT_KV` additionally accepted `"YES"`). The parsers
//! are pure functions so the preserved token sets are unit-tested directly.

use std::sync::OnceLock;

/// Common truthy set shared by the four Marlin / W4A8 / deterministic knobs.
fn parse_truthy_common(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true" | "TRUE" | "yes" | "on" | "ON"))
}

/// `INFER_R4_W4A16_GEMV_OVERRIDE` legacy semantics: only the literal `"1"`.
fn parse_r4_override(value: Option<&str>) -> bool {
    value == Some("1")
}

/// `INFER_TILELANG_BF16_SPLIT_KV` legacy semantics: the common set plus `"YES"`.
fn parse_split_kv(value: Option<&str>) -> bool {
    matches!(
        value,
        Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
    )
}

/// Resolved-once view of every dispatch-affecting env knob in the ops layer.
///
/// Fields are public for the `explain-dispatch` introspection path and for
/// test construction; they are read-only after `from_env`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DispatchPolicy {
    /// `INFER_MARLIN_W4_FP8_PREFILL` — opt-in prefill-only W4+FP8 Marlin GEMM.
    pub marlin_w4_fp8_prefill: bool,
    /// `INFER_HYBRID_W4A8_PREFILL` — opt-in hybrid W4A8 prefill dispatch.
    pub hybrid_w4a8_prefill: bool,
    /// `INFER_MARLIN_W4A8_AUTOCONFIG` — let Marlin auto-pick thread config.
    pub marlin_w4a8_autoconfig: bool,
    /// `INFER_R4_W4A16_GEMV_OVERRIDE` — prefer W4A16 batch-GEMV over Marlin for
    /// decode-batched (batch ∈ 2..=8). Legacy site accepted only `"1"`.
    pub r4_w4a16_gemv_override: bool,
    /// `INFER_DETERMINISTIC` — force the deterministic BF16 GEMM path.
    pub deterministic_gemm: bool,
    /// `INFER_TILELANG_BF16_SPLIT_KV` — request the TileLang BF16 split-KV
    /// decode kernel. Legacy site additionally accepted `"YES"`.
    pub tilelang_bf16_split_kv: bool,
}

impl DispatchPolicy {
    /// Parse every knob from the environment, preserving each legacy site's
    /// exact accepted token set.
    pub fn from_env() -> Self {
        let read = |name: &str| std::env::var(name).ok();
        Self {
            marlin_w4_fp8_prefill: parse_truthy_common(
                read("INFER_MARLIN_W4_FP8_PREFILL").as_deref(),
            ),
            hybrid_w4a8_prefill: parse_truthy_common(read("INFER_HYBRID_W4A8_PREFILL").as_deref()),
            marlin_w4a8_autoconfig: parse_truthy_common(
                read("INFER_MARLIN_W4A8_AUTOCONFIG").as_deref(),
            ),
            r4_w4a16_gemv_override: parse_r4_override(
                read("INFER_R4_W4A16_GEMV_OVERRIDE").as_deref(),
            ),
            deterministic_gemm: parse_truthy_common(read("INFER_DETERMINISTIC").as_deref()),
            tilelang_bf16_split_kv: parse_split_kv(read("INFER_TILELANG_BF16_SPLIT_KV").as_deref()),
        }
    }
}

/// Process-wide dispatch policy, resolved from the environment on first access.
///
/// Mirrors the previous `OnceLock`-cached helpers: env is read once, then the
/// cached struct is returned for the lifetime of the process.
pub fn dispatch_policy() -> &'static DispatchPolicy {
    static POLICY: OnceLock<DispatchPolicy> = OnceLock::new();
    POLICY.get_or_init(DispatchPolicy::from_env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_common_accepts_legacy_set() {
        for v in ["1", "true", "TRUE", "yes", "on", "ON"] {
            assert!(parse_truthy_common(Some(v)), "{v} should be truthy");
        }
    }

    #[test]
    fn truthy_common_rejects_others() {
        for v in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("YES"),
            Some("y"),
        ] {
            assert!(!parse_truthy_common(v), "{v:?} should be falsy");
        }
    }

    #[test]
    fn r4_override_only_literal_one() {
        // Preserves the stricter legacy semantics: "true"/"yes"/"ON" must NOT enable it.
        assert!(parse_r4_override(Some("1")));
        for v in [
            None,
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some("on"),
            Some("ON"),
        ] {
            assert!(!parse_r4_override(v), "{v:?} must not enable r4 override");
        }
    }

    #[test]
    fn split_kv_accepts_common_set_plus_yes() {
        for v in ["1", "true", "TRUE", "on", "ON", "yes", "YES"] {
            assert!(parse_split_kv(Some(v)), "{v} should enable split-kv");
        }
        // "YES" is the one token split-kv accepts that the common set does not.
        assert!(parse_split_kv(Some("YES")) && !parse_truthy_common(Some("YES")));
        for v in [None, Some(""), Some("0"), Some("false")] {
            assert!(!parse_split_kv(v), "{v:?} should not enable split-kv");
        }
    }
}
