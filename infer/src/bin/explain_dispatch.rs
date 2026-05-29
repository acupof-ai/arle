//! `explain-dispatch` — operator-facing introspection for the resolved
//! [`DispatchPolicy`].
//!
//! Governance background:
//! [`docs/reviews/2026-05-29-gpu-dispatch-governance-analysis.md`] and
//! [`docs/plans/gpu-dispatch-governance.md`]. `dispatch_policy.rs` is the
//! **Declare** gate: every dispatch-affecting env knob is parsed exactly once
//! into one inspectable struct. This binary is the operator-facing answer to
//! "which dispatch knobs are active right now" — it resolves
//! [`DispatchPolicy::from_env`] in the current environment and prints each
//! field with its env var name and active/default state, one readable line per
//! knob, e.g.:
//!
//! ```text
//! INFER_MARLIN_W4_FP8_PREFILL          marlin_w4_fp8_prefill = false (default)
//! ```
//!
//! This is policy-only introspection. Printing a resolved `ExecutionPlan` /
//! oplib plan (the actual kernel selection a given shape would dispatch to) is
//! a later tranche — deliberately out of scope here so this stays a pure,
//! zero-runtime-behaviour view of the Declare gate.
//!
//! No feature gate: `infer::dispatch_policy` is backend-independent (the struct
//! and its parsers are pure functions), so this resolves and prints under every
//! feature set, including the host-only `no-cuda` / `cpu` builds an operator
//! would use to inspect a deployment's knobs.

use infer::dispatch_policy::DispatchPolicy;

/// Render a boolean knob: env var name, field name, value, and whether the
/// value is the compiled-in default (`false`) or an active opt-in (`true`).
fn line_bool(env_var: &str, field: &str, value: bool) -> String {
    let state = if value { "active" } else { "default" };
    format!("{env_var:<36} {field} = {value} ({state})")
}

/// Render the numeric DSv4 threshold knob. The default is `4`; any other
/// resolved value reflects an active override (legacy `< 1` / unparseable
/// inputs already fold back to `4` inside `DispatchPolicy::from_env`).
fn line_usize(env_var: &str, field: &str, value: usize, default: usize) -> String {
    let state = if value == default {
        "default"
    } else {
        "active"
    };
    format!("{env_var:<36} {field} = {value} ({state})")
}

fn main() {
    // Resolve directly (not via the process-wide `dispatch_policy()` cache) so
    // this is a clean, side-effect-free read of the current environment.
    let policy = DispatchPolicy::from_env();

    println!("resolved DispatchPolicy (Declare gate):");
    println!(
        "{}",
        line_bool(
            "INFER_MARLIN_W4_FP8_PREFILL",
            "marlin_w4_fp8_prefill",
            policy.marlin_w4_fp8_prefill,
        )
    );
    println!(
        "{}",
        line_bool(
            "INFER_HYBRID_W4A8_PREFILL",
            "hybrid_w4a8_prefill",
            policy.hybrid_w4a8_prefill,
        )
    );
    println!(
        "{}",
        line_bool(
            "INFER_MARLIN_W4A8_AUTOCONFIG",
            "marlin_w4a8_autoconfig",
            policy.marlin_w4a8_autoconfig,
        )
    );
    println!(
        "{}",
        line_bool(
            "INFER_R4_W4A16_GEMV_OVERRIDE",
            "r4_w4a16_gemv_override",
            policy.r4_w4a16_gemv_override,
        )
    );
    println!(
        "{}",
        line_bool(
            "INFER_DETERMINISTIC",
            "deterministic_gemm",
            policy.deterministic_gemm,
        )
    );
    println!(
        "{}",
        line_bool(
            "INFER_TILELANG_BF16_SPLIT_KV",
            "tilelang_bf16_split_kv",
            policy.tilelang_bf16_split_kv,
        )
    );
    println!(
        "{}",
        line_bool("INFER_PREFILL_GRAPH", "prefill_graph", policy.prefill_graph)
    );
    println!(
        "{}",
        line_usize(
            "ARLE_DSV4_GROUPED_GEMM_M_THRESHOLD",
            "dsv4_grouped_gemm_m_threshold",
            policy.dsv4_grouped_gemm_m_threshold,
            4,
        )
    );
}
