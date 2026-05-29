//! Backend-neutral attention **head-config selection**, relocated out of the
//! CUDA launch path.
//!
//! This module owns the *selection* half of the second oplib operator family:
//! given the per-SKU `(num_qo_heads, num_kv_heads)` pair, [`head_config`]
//! resolves which AOT-precompiled TileLang head specialization a paged-attention
//! launch should target — or rejects the pair with the canonical
//! "no precompiled kernel for this config" error.
//!
//! ## The CPU-testable property
//!
//! [`head_config`] is a pure function. It names **no** CUDA/cudarc type, touches
//! no device memory, launches no kernel, and reads only the host-side head
//! counts. The consequence — and the headline of
//! [`docs/plans/backend-operator-library.md`](../../docs/plans/backend-operator-library.md)
//! §"Paged-decode attention" — is that "does this SKU's `(qo,kv)` head config
//! have a precompiled kernel?" becomes a GPU-free unit test
//! (`assert_eq!(head_config(qo, kv), Ok(Expected))`), answered under the crate's
//! default feature set on a machine with no nvcc and no GPU, **instead of a
//! runtime hard-fail** the first time an unprecompiled SKU reaches the launch.
//!
//! Before this resolver, the identical `(qo,kv)` validation + hard-fail string
//! was duplicated across every HD256 TileLang launch site in
//! `infer/src/ops/attention.rs` (paged prefill + paged decode). Each site
//! carried its own `match (qo, kv) { … other => return Err(anyhow!("…")) }`. The
//! validation + error now live here exactly once; each CUDA launch site keeps
//! only the (cuda-typed) `HeadConfig`→FFI-fn-pointer mapping, which is exhaustive
//! over the three supported variants — no hard-fail arm survives on the launch
//! side.

/// The AOT-precompiled `(num_qo_heads, num_kv_heads)` head specializations the
/// HD256 TileLang paged-attention kernels ship. Backend-neutral: it names the
/// *logical* head config, not a device function pointer. The CUDA launch path
/// maps each variant onto its `tilelang_*_q{Q}_kv{KV}_run_cuda` FFI symbol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadConfig {
    /// `(num_qo_heads, num_kv_heads) == (8, 2)`.
    Q8Kv2,
    /// `(num_qo_heads, num_kv_heads) == (16, 2)`.
    Q16Kv2,
    /// `(num_qo_heads, num_kv_heads) == (16, 4)`.
    Q16Kv4,
}

/// PURE. Resolve the AOT head specialization for `(num_qo_heads, num_kv_heads)`.
///
/// Returns `Ok(HeadConfig)` for the precompiled set `{(8,2),(16,2),(16,4)}` and
/// `Err(String)` — the canonical "no precompiled kernel for this config" message
/// the legacy HD256 launch sites emitted — for any other pair. No device memory
/// is touched and no CUDA type is named, so this runs on CPU under the default
/// feature set.
///
/// This is the relocated body of the duplicated `match (qo, kv)` head-config
/// guards in `infer/src/ops/attention.rs` (HD256 paged prefill + decode). The
/// hard-fail that used to surface only at launch time is now answerable as a
/// CPU unit test.
pub fn head_config(num_qo_heads: usize, num_kv_heads: usize) -> Result<HeadConfig, String> {
    match (num_qo_heads, num_kv_heads) {
        (8, 2) => Ok(HeadConfig::Q8Kv2),
        (16, 2) => Ok(HeadConfig::Q16Kv2),
        (16, 4) => Ok(HeadConfig::Q16Kv4),
        other => Err(format!(
            "TileLang: no specialized HD256 kernel for \
             (num_qo_heads, num_kv_heads) = {other:?}; supported configs \
             are (8,2), (16,2), (16,4). Extend SUPPORTED_HEADS \
             in the matching tools/tilelang/batch_*_paged_hd256.py, \
             TILELANG_*_HD256_HEAD_CONFIGS in cuda-kernels/build.rs, \
             and the FFI macro + this match in lockstep, then rebuild."
        )),
    }
}

/// The AOT-precompiled `(num_qo_heads, num_kv_heads)` head specializations the
/// HD128 TileLang paged-attention kernels ship. Backend-neutral: it names the
/// *logical* head config, not a device function pointer. Each HD128 launch site
/// (paged prefill, pure decode, the TC-decode alias that reuses the prefill
/// kernels, and the split-KV partial / merge pair) maps each variant onto its
/// own `tilelang_*_paged_hd128_q{Q}_kv{KV}_run_cuda` FFI symbol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadConfigHd128 {
    /// `(num_qo_heads, num_kv_heads) == (16, 8)`.
    Q16Kv8,
    /// `(num_qo_heads, num_kv_heads) == (32, 8)`.
    Q32Kv8,
    /// `(num_qo_heads, num_kv_heads) == (40, 8)`.
    Q40Kv8,
    /// `(num_qo_heads, num_kv_heads) == (64, 8)`.
    Q64Kv8,
}

/// PURE. Resolve the AOT head specialization for `(num_qo_heads, num_kv_heads)`
/// on the HD128 kernels.
///
/// Returns `Ok(HeadConfigHd128)` for the precompiled set
/// `{(16,8),(32,8),(40,8),(64,8)}` and `Err(String)` — the canonical "no
/// precompiled kernel for this config" message the legacy HD128 launch sites
/// emitted — for any other pair. No device memory is touched and no CUDA type is
/// named, so this runs on CPU under the default feature set.
///
/// This is the relocated body of the five duplicated `match (qo, kv)`
/// head-config guards in `infer/src/ops/attention.rs` (HD128 paged prefill,
/// pure decode, TC-decode alias, split-partial, split-merge). The hard-fail that
/// used to surface only at launch time is now answerable as a CPU unit test.
pub fn head_config_hd128(
    num_qo_heads: usize,
    num_kv_heads: usize,
) -> Result<HeadConfigHd128, String> {
    match (num_qo_heads, num_kv_heads) {
        (16, 8) => Ok(HeadConfigHd128::Q16Kv8),
        (32, 8) => Ok(HeadConfigHd128::Q32Kv8),
        (40, 8) => Ok(HeadConfigHd128::Q40Kv8),
        (64, 8) => Ok(HeadConfigHd128::Q64Kv8),
        other => Err(format!(
            "TileLang: no specialized HD128 kernel for \
             (num_qo_heads, num_kv_heads) = {other:?}; supported configs \
             are (16,8), (32,8), (40,8), (64,8). Extend SUPPORTED_HEADS \
             in the matching tools/tilelang/batch_*_paged_hd128.py, \
             TILELANG_*_HD128_HEAD_CONFIGS in cuda-kernels/build.rs, \
             and the FFI macro + this match in lockstep, then rebuild."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every precompiled `(qo,kv)` pair resolves to its variant — the
    /// "is my SKU's head config precompiled?" question answered on CPU.
    #[test]
    fn supported_configs_resolve_to_expected_variant() {
        assert_eq!(head_config(8, 2), Ok(HeadConfig::Q8Kv2));
        assert_eq!(head_config(16, 2), Ok(HeadConfig::Q16Kv2));
        assert_eq!(head_config(16, 4), Ok(HeadConfig::Q16Kv4));
    }

    /// A sweep of unprecompiled pairs all reject with the canonical message —
    /// the former runtime hard-fail, now a CPU assertion.
    #[test]
    fn unsupported_configs_reject_with_canonical_message() {
        // Pairs that never had an AOT kernel: a degenerate (1,1), the HD128-only
        // (32,8) shape, and a (8,1) that is supported nowhere.
        for &(qo, kv) in &[(1usize, 1usize), (32, 8), (8, 1)] {
            let err = head_config(qo, kv).expect_err("must reject unprecompiled config");
            // The invariant core of the legacy hard-fail string, verbatim.
            assert!(
                err.contains("TileLang: no specialized HD256 kernel for"),
                "missing canonical prefix for ({qo},{kv}): {err}"
            );
            assert!(
                err.contains("supported configs"),
                "missing canonical 'supported configs' clause for ({qo},{kv}): {err}"
            );
            assert!(
                err.contains("are (8,2), (16,2), (16,4)."),
                "missing canonical supported-set list for ({qo},{kv}): {err}"
            );
            // The rejected pair is reported via `{other:?}` exactly as the
            // legacy sites formatted it.
            assert!(
                err.contains(&format!("(num_qo_heads, num_kv_heads) = {:?}", (qo, kv))),
                "missing rejected-pair echo for ({qo},{kv}): {err}"
            );
        }
    }

    /// The supported set is exactly `{(8,2),(16,2),(16,4)}` — no neighbouring
    /// pair leaks in (e.g. (8,4), (16,8), (16,1)).
    #[test]
    fn supported_set_is_exactly_three_pairs() {
        let mut ok = 0usize;
        for qo in 0usize..=64 {
            for kv in 0usize..=16 {
                if head_config(qo, kv).is_ok() {
                    ok += 1;
                    assert!(
                        matches!((qo, kv), (8, 2) | (16, 2) | (16, 4)),
                        "unexpected pair accepted: ({qo},{kv})"
                    );
                }
            }
        }
        assert_eq!(ok, 3, "exactly three head configs must be precompiled");
    }

    /// Every precompiled HD128 `(qo,kv)` pair resolves to its variant — the
    /// "is my SKU's HD128 head config precompiled?" question answered on CPU.
    #[test]
    fn hd128_supported_configs_resolve_to_expected_variant() {
        assert_eq!(head_config_hd128(16, 8), Ok(HeadConfigHd128::Q16Kv8));
        assert_eq!(head_config_hd128(32, 8), Ok(HeadConfigHd128::Q32Kv8));
        assert_eq!(head_config_hd128(40, 8), Ok(HeadConfigHd128::Q40Kv8));
        assert_eq!(head_config_hd128(64, 8), Ok(HeadConfigHd128::Q64Kv8));
    }

    /// A sweep of unprecompiled HD128 pairs all reject with the canonical
    /// message — the former runtime hard-fail, now a CPU assertion.
    #[test]
    fn hd128_unsupported_configs_reject_with_canonical_message() {
        // Pairs that never had an HD128 AOT kernel: a degenerate (1,1), the
        // HD256-only (8,2) shape, and (16,4)/(8,8) that are supported nowhere
        // on HD128.
        for &(qo, kv) in &[(1usize, 1usize), (8, 2), (16, 4), (8, 8)] {
            let err =
                head_config_hd128(qo, kv).expect_err("must reject unprecompiled HD128 config");
            // The invariant core of the legacy hard-fail string, verbatim.
            assert!(
                err.contains("TileLang: no specialized HD128 kernel for"),
                "missing canonical prefix for ({qo},{kv}): {err}"
            );
            assert!(
                err.contains("supported configs"),
                "missing canonical 'supported configs' clause for ({qo},{kv}): {err}"
            );
            assert!(
                err.contains("are (16,8), (32,8), (40,8), (64,8)."),
                "missing canonical supported-set list for ({qo},{kv}): {err}"
            );
            // The rejected pair is reported via `{other:?}` exactly as the
            // legacy sites formatted it.
            assert!(
                err.contains(&format!("(num_qo_heads, num_kv_heads) = {:?}", (qo, kv))),
                "missing rejected-pair echo for ({qo},{kv}): {err}"
            );
        }
    }

    /// The HD128 supported set is exactly `{(16,8),(32,8),(40,8),(64,8)}` — no
    /// neighbouring pair leaks in (e.g. (8,8), (16,4), (48,8), (64,16)).
    #[test]
    fn hd128_supported_set_is_exactly_four_pairs() {
        let mut ok = 0usize;
        for qo in 0usize..=64 {
            for kv in 0usize..=16 {
                if head_config_hd128(qo, kv).is_ok() {
                    ok += 1;
                    assert!(
                        matches!((qo, kv), (16, 8) | (32, 8) | (40, 8) | (64, 8)),
                        "unexpected HD128 pair accepted: ({qo},{kv})"
                    );
                }
            }
        }
        assert_eq!(ok, 4, "exactly four HD128 head configs must be precompiled");
    }
}
