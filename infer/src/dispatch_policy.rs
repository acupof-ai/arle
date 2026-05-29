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
//! This covers the ops-layer subset (the kernel-selection knobs in
//! `ops/linear.rs` + `ops/attention.rs`), the model-layer path-selection
//! subset (`INFER_PREFILL_GRAPH` in `model/qwen3/prefill.rs`,
//! `ARLE_DSV4_GROUPED_GEMM_M_THRESHOLD` in `model/deepseek/mlp.rs`), and the
//! scheduler-layer subset (`INFER_BYPASS_TILELANG_PREFILL` in
//! `scheduler/cuda/prefill.rs`). Load-time CONFIG knobs (the `ARLE_DSV4_*`
//! pool/feature family, `INFER_QUANT_FORMAT_OVERRIDE`,
//! `INFER_QWEN3_FUSED_GATE_UP`) and the `*_DEBUG`/`*_DUMP` diagnostics are
//! deliberately NOT dispatch knobs and stay at their original sites.
//!
//! Behaviour is preserved bit-for-bit: each field reproduces the exact accepted
//! token set (or numeric parse) of the call site it replaced. Knobs are
//! deliberately NOT unified onto one truthy parser, because the legacy sites
//! disagreed (`INFER_R4_W4A16_GEMV_OVERRIDE` accepted only `"1"`;
//! `INFER_TILELANG_BF16_SPLIT_KV` additionally accepted `"YES"`). The parsers
//! are pure functions so the preserved token sets are unit-tested directly.

use std::sync::RwLock;

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

/// `INFER_PREFILL_GRAPH` legacy semantics: the common truthy set (the
/// `model/qwen3/prefill.rs` site matched exactly `1`/`true`/`TRUE`/`yes`/`on`/`ON`).
fn parse_prefill_graph(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true" | "TRUE" | "yes" | "on" | "ON"))
}

/// `INFER_BYPASS_TILELANG_PREFILL` legacy semantics: the `scheduler/cuda/prefill.rs`
/// site gated on `std::env::var(..).is_ok()`, i.e. the variable being *present* at
/// all enabled it — any value, including the empty string. Only an unset variable
/// (`None`) is falsy.
fn parse_bypass_tilelang_prefill(value: Option<&str>) -> bool {
    value.is_some()
}

/// `ARLE_DSV4_GROUPED_GEMM_M_THRESHOLD` legacy semantics: parse as `usize`,
/// keep only values `>= 1`; anything unset / unparseable / `< 1` falls back to
/// the default `4`. Returns the resolved threshold directly so the field holds
/// the effective value (not an `Option`), matching the `unwrap_or(4)` site.
fn parse_dsv4_grouped_gemm_m_threshold(value: Option<&str>) -> usize {
    value
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(4)
}

/// Resolved-once view of every dispatch-affecting env knob in the ops and
/// model layers.
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
    /// `INFER_PREFILL_GRAPH` — opt-in CUDA-Graph capture for Qwen3 paged
    /// prefill (`model/qwen3/prefill.rs`). Common truthy set.
    pub prefill_graph: bool,
    /// `INFER_BYPASS_TILELANG_PREFILL` — opt-in route-around forcing every
    /// paged-pool prefill onto the contig CUDA C path
    /// (`scheduler/cuda/prefill.rs`). Legacy site gated on variable presence
    /// (`is_ok()`): any value, including the empty string, enables it.
    pub bypass_tilelang_prefill: bool,
    /// `ARLE_DSV4_GROUPED_GEMM_M_THRESHOLD` — the M (active-row) count at/above
    /// which DSv4 grouped MoE switches from block-scaled GEMV to the 32-way
    /// M-tile grouped GEMM (`model/deepseek/mlp.rs`). Resolved value, default
    /// `4`; values `< 1` or unparseable fall back to the default.
    pub dsv4_grouped_gemm_m_threshold: usize,
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
            prefill_graph: parse_prefill_graph(read("INFER_PREFILL_GRAPH").as_deref()),
            bypass_tilelang_prefill: parse_bypass_tilelang_prefill(
                read("INFER_BYPASS_TILELANG_PREFILL").as_deref(),
            ),
            dsv4_grouped_gemm_m_threshold: parse_dsv4_grouped_gemm_m_threshold(
                read("ARLE_DSV4_GROUPED_GEMM_M_THRESHOLD").as_deref(),
            ),
        }
    }
}

static POLICY_CACHE: RwLock<Option<DispatchPolicy>> = RwLock::new(None);

/// Process-wide dispatch policy, resolved from the environment on first access
/// and cached. Production sets dispatch env vars at startup and never mutates
/// them afterward, so the cache is effectively immutable for the process
/// lifetime. Returned by value (`DispatchPolicy` is `Copy`) so the cache can be
/// invalidated without handing out borrowed references.
///
/// Tests that mutate a dispatch env var in-process MUST call
/// [`reset_dispatch_policy_cache`] afterward, or the stale cached value wins.
pub fn dispatch_policy() -> DispatchPolicy {
    if let Some(policy) = *POLICY_CACHE.read().expect("dispatch policy cache poisoned") {
        return policy;
    }
    let mut slot = POLICY_CACHE
        .write()
        .expect("dispatch policy cache poisoned");
    *slot.get_or_insert_with(DispatchPolicy::from_env)
}

/// Drop the cached [`DispatchPolicy`] so the next [`dispatch_policy`] call
/// re-reads the environment. Intended for tests that mutate dispatch env vars
/// in-process — production resolves the policy once at startup and never needs
/// this.
pub fn reset_dispatch_policy_cache() {
    *POLICY_CACHE
        .write()
        .expect("dispatch policy cache poisoned") = None;
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

    #[test]
    fn prefill_graph_accepts_common_set() {
        for v in ["1", "true", "TRUE", "yes", "on", "ON"] {
            assert!(
                parse_prefill_graph(Some(v)),
                "{v} should enable prefill-graph"
            );
        }
        // Same strictness as the common truthy set: "YES"/"y"/"0" must not enable it.
        for v in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("YES"),
            Some("y"),
        ] {
            assert!(
                !parse_prefill_graph(v),
                "{v:?} should not enable prefill-graph"
            );
        }
    }

    #[test]
    fn bypass_tilelang_prefill_enabled_by_presence() {
        // Legacy `is_ok()` semantics: presence enables it regardless of value,
        // including the empty string. Only `None` (unset) is falsy.
        assert!(!parse_bypass_tilelang_prefill(None));
        assert!(parse_bypass_tilelang_prefill(Some("")));
        assert!(parse_bypass_tilelang_prefill(Some("1")));
        assert!(parse_bypass_tilelang_prefill(Some("0")));
        assert!(parse_bypass_tilelang_prefill(Some("false")));
        assert!(parse_bypass_tilelang_prefill(Some("anything")));
    }

    #[test]
    fn dsv4_grouped_gemm_m_threshold_parse_and_default() {
        // Unset / unparseable / sub-1 fall back to the legacy default 4.
        assert_eq!(parse_dsv4_grouped_gemm_m_threshold(None), 4);
        assert_eq!(parse_dsv4_grouped_gemm_m_threshold(Some("")), 4);
        assert_eq!(parse_dsv4_grouped_gemm_m_threshold(Some("abc")), 4);
        assert_eq!(parse_dsv4_grouped_gemm_m_threshold(Some("0")), 4);
        assert_eq!(parse_dsv4_grouped_gemm_m_threshold(Some("-1")), 4);
        // Valid >= 1 values pass through verbatim.
        assert_eq!(parse_dsv4_grouped_gemm_m_threshold(Some("1")), 1);
        assert_eq!(parse_dsv4_grouped_gemm_m_threshold(Some("4")), 4);
        assert_eq!(parse_dsv4_grouped_gemm_m_threshold(Some("32")), 32);
    }
}
