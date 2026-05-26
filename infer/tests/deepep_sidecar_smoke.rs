//! Smoke test for the native DeepEP sidecar pool — phase 1.1.5/6.
//!
//! Skips unless ALL of the following hold:
//!   - `cuda` feature was compiled in
//!   - the sidecar binary was built (i.e. `ARLE_DEEPEP_DIR` was set when
//!     `cuda-kernels` was compiled, so `option_env!("ARLE_DEEPEP_SIDECAR_PATH")`
//!     resolves at compile time)
//!   - `ARLE_DEEPEP_RUN_SMOKE=1` is set at runtime (so the test doesn't
//!     accidentally fork 8 CUDA processes in CI / dev box runs)
//!
//! On a properly-configured 8 × H20 pod, the expected sha256 / preview for
//! each rank match the phase 1.0a-iv spike (see
//! `docs/experience/wins/2026-05-26-dsv4-deepep-cpp-full-dispatch-combine.md`):
//!
//! | rank | first8 preview                          |
//! |------|-----------------------------------------|
//! |    0 | {0.0000, 0.0006, 0.0012, ..., 0.0042}  |
//! |    1 | {6.0000, 6.0000, ..., 6.0000}          |
//! |    7 | {42.0000, 42.0000, ..., 42.0000}       |

#![cfg(feature = "cuda")]

use std::path::PathBuf;

#[test]
fn deepep_sidecar_round_trip_smoke() {
    if std::env::var("ARLE_DEEPEP_RUN_SMOKE").ok().as_deref() != Some("1") {
        eprintln!(
            "[skip] ARLE_DEEPEP_RUN_SMOKE != 1 — sidecar smoke test is opt-in (requires 8xH20)."
        );
        return;
    }

    let baked = match infer::backend::cuda::deepep_sidecar::baked_binary_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "[skip] ARLE_DEEPEP_SIDECAR_PATH not baked in — set ARLE_DEEPEP_DIR at build time to enable."
            );
            return;
        }
    };
    let bin = PathBuf::from(baked);
    if !bin.exists() {
        panic!(
            "ARLE_DEEPEP_SIDECAR_PATH baked at compile time ({}) but file is missing — clean build needed.",
            bin.display()
        );
    }

    use infer::backend::cuda::deepep_sidecar::{RoundTripRequest, SidecarPool, SidecarPoolConfig};

    let pool = SidecarPool::spawn(SidecarPoolConfig {
        binary: &bin,
        world_size: 8,
    })
    .expect("SidecarPool::spawn");

    // Phase 1.0a-iv reference shape — 1 token, hidden=4096, topk=6,
    // experts=256, num_sms=20 (channels=10), nvl_chunked send=6/recv=256.
    let req = RoundTripRequest {
        num_tokens: 1,
        hidden: 4096,
        num_topk: 6,
        num_experts: 256,
        num_sms: 20,
        nvl_chunked_send: 6,
        nvl_chunked_recv: 256,
        reserved: 0,
    };

    let responses = pool.round_trip_all(req).expect("round_trip_all");
    assert_eq!(responses.len(), 8);

    // Sanity check the rank-tag pattern: rank R's combined output averages
    // 6 copies of bf16(R + j*1e-4) — for j=0 each rank's first preview
    // value should be ≈ 6 * R.
    for (rank, resp) in responses.iter().enumerate() {
        assert_eq!(
            resp.num_recv_tokens, 6,
            "rank {rank} expected num_recv_tokens=6, got {}",
            resp.num_recv_tokens
        );
        let expected_first = 6.0_f32 * rank as f32;
        let actual_first = resp.preview[0];
        assert!(
            (actual_first - expected_first).abs() < 1.0,
            "rank {rank} first preview want ~{expected_first}, got {actual_first}"
        );
    }

    // Determinism: a second round-trip on the same pool must yield the
    // same sha256 on every rank (phase 1.0a-iv proved this; protect the
    // invariant against pool-reuse regressions).
    let responses2 = pool.round_trip_all(req).expect("round_trip_all (second)");
    for (rank, (a, b)) in responses.iter().zip(responses2.iter()).enumerate() {
        assert_eq!(
            a.sha256, b.sha256,
            "rank {rank} sha256 not deterministic across calls"
        );
    }
}
