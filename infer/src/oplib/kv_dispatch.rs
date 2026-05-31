//! Backend-neutral KV-format dispatch **selection**, relocated out of the
//! Qwen3.5 decode / prefill launch paths.
//!
//! This module owns the *selection* half of the KV operator family: given the
//! pool's [`KVFormat`], [`dispatch`] returns the canonical [`KvScheme`] partition
//! that the per-format kernel-launch `match`es in
//! `infer/src/model/qwen35/{batch_decode.rs,prefill.rs}` switch on. The single
//! place the `KVFormat → scheme` partition lives.
//!
//! ## The pure-`dispatch()` property
//!
//! [`dispatch`] is a pure function. It names **no** CUDA/cudarc type, touches no
//! device memory, launches no kernel, and reads only the host-side
//! [`KVFormat`] tag (which is itself feature-free — `cuda_kernels::kv_types`
//! carries no `cudarc`). The consequence — and the headline of
//! [`docs/plans/backend-operator-library.md`](../../docs/plans/backend-operator-library.md)
//! — is that "which KV scheme does format X resolve to?" becomes a GPU-free unit
//! test (`assert_eq!(dispatch(fmt).scheme, Expected)`), runnable under the
//! crate's default feature set on a machine with no nvcc and no GPU.
//!
//! Before this resolver, the identical `match <pool>.format { KVFormat::… }`
//! partition was duplicated across three kernel-launch sites: the decode
//! quantize-on-write match, the decode attention-read match, and the prefill
//! quantize-scatter match. The partition now lives here once; each launch site
//! keeps only its bespoke per-arm kernel body (genuinely incompatible arg lists),
//! switching on the returned [`KvScheme`] instead of the raw [`KVFormat`].
//!
//! Per the canonical grouping the launch sites already used, **both**
//! `KVFormat::TurboQuant { .. }` configs map to the single [`KvScheme::TurboQuant`]
//! arm (the launch bodies branch on `tq_*_state`, not on the bit-pair).

// `KVFormat` lives in `cuda_kernels::kv_types` (feature-free, no `cudarc`) and
// is re-exported by `crate::model::kv_cache`. The model re-export is
// `#[cfg(feature = "cuda")]`-gated, so this pure resolver imports from
// `cuda_kernels` directly — that crate is a dependency under both `cuda` and
// `no-cuda`, keeping `dispatch()` resolvable (and CPU-testable) under the
// crate's default / `no-cuda` feature set with no GPU.
use cuda_kernels::KVFormat;

/// The canonical KV quantization-scheme partition the Qwen3.5 decode / prefill
/// kernel-launch `match`es switch on.
///
/// Backend-neutral: it names the *logical* scheme, not a device function
/// pointer. Each variant maps 1:1 onto one arm of the per-format kernel-launch
/// `match`es. Both `KVFormat::TurboQuant` bit-pair configs collapse into the
/// single [`KvScheme::TurboQuant`] arm — exactly the
/// `KVFormat::TurboQuant { .. }` grouping the launch sites already used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvScheme {
    /// Unquantized BF16 pool (`KVFormat::BF16`).
    Bf16,
    /// INT8 with KIVI per-channel K + per-(row, head) V (`KVFormat::INT8`).
    Int8PerChannel,
    /// FP8 E4M3 with KIVI per-channel K + per-(row, head) V (`KVFormat::FP8E4M3`).
    Fp8PerChannel,
    /// INT4 (packed nibbles) with KIVI per-channel K + per-(row, head) V
    /// (`KVFormat::INT4`).
    Int4PerChannel,
    /// TurboQuant (Hadamard-rotated, codebook) — both bit-pair configs of
    /// `KVFormat::TurboQuant { .. }`.
    TurboQuant,
}

/// The named KV dispatch artifact — which [`KvScheme`] `dispatch` selected.
///
/// Carries **only** the `scheme` field: of the dispatch sites surveyed in
/// `infer/src/model/qwen35/{batch_decode.rs,prefill.rs}`, every `match` on a
/// `KVFormat` value is a *kernel-launch* match (different bespoke kernel fns per
/// arm) — there are **no** fact-derivation matches that compute an inline
/// scalar/bool from the format (element byte size, whether per-channel K scales
/// are needed, etc. all already live as methods on `KVFormat` in
/// `cuda_kernels::kv_types`). Adding speculative `element_bytes` / `needs_k_scales`
/// fields with no real consumer would violate
/// `memory/feedback_no_speculative_interface_shaping`, so the struct stays
/// scheme-only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvDispatch {
    /// The canonical scheme the kernel-launch `match`es switch on.
    pub scheme: KvScheme,
}

/// PURE. Resolve the [`KvScheme`] for `format`.
///
/// This is the single home of the `KVFormat → scheme` partition the Qwen3.5
/// decode / prefill kernel-launch `match`es used to inline. Behavior-preserving
/// and bit-identical with those inline `match`es, proven by the sweep test
/// below. No device memory is touched and no CUDA type is named, so this runs on
/// CPU under the default feature set.
///
/// Both `KVFormat::TurboQuant { .. }` configs map to [`KvScheme::TurboQuant`],
/// exactly as the launch sites grouped them.
#[must_use]
pub fn dispatch(format: KVFormat) -> KvDispatch {
    let scheme = match format {
        KVFormat::BF16 => KvScheme::Bf16,
        KVFormat::INT8 => KvScheme::Int8PerChannel,
        KVFormat::FP8E4M3 => KvScheme::Fp8PerChannel,
        KVFormat::INT4 => KvScheme::Int4PerChannel,
        KVFormat::TurboQuant { .. } => KvScheme::TurboQuant,
    };
    KvDispatch { scheme }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `KVFormat` variant resolves to its canonical scheme — the
    /// "which scheme does this format select?" question answered on CPU. Sweeps
    /// all five `KVFormat` variant kinds (the six conceptual formats: BF16,
    /// FP8E4M3, INT8, INT4, and both TurboQuant bit-pair configs collapsing onto
    /// the single TurboQuant scheme).
    #[test]
    fn dispatch_matches_canonical_partition_over_all_formats() {
        assert_eq!(dispatch(KVFormat::BF16).scheme, KvScheme::Bf16);
        assert_eq!(dispatch(KVFormat::INT8).scheme, KvScheme::Int8PerChannel);
        assert_eq!(dispatch(KVFormat::FP8E4M3).scheme, KvScheme::Fp8PerChannel);
        assert_eq!(dispatch(KVFormat::INT4).scheme, KvScheme::Int4PerChannel);
        // BOTH TurboQuant configs collapse onto the single TurboQuant arm —
        // exactly the `KVFormat::TurboQuant { .. }` grouping the launch sites use.
        assert_eq!(
            dispatch(KVFormat::TurboQuant {
                key_bits: 2,
                val_bits: 2,
            })
            .scheme,
            KvScheme::TurboQuant
        );
        assert_eq!(
            dispatch(KVFormat::TurboQuant {
                key_bits: 4,
                val_bits: 4,
            })
            .scheme,
            KvScheme::TurboQuant
        );
    }

    /// The partition is a function: equal formats resolve to equal dispatches,
    /// and the two distinct TurboQuant bit-pairs are *not* distinguished by the
    /// scheme (both → TurboQuant), which is the whole point of the collapse.
    #[test]
    fn turboquant_bit_pairs_are_indistinguishable_by_scheme() {
        let tq22 = dispatch(KVFormat::TurboQuant {
            key_bits: 2,
            val_bits: 2,
        });
        let tq44 = dispatch(KVFormat::TurboQuant {
            key_bits: 4,
            val_bits: 4,
        });
        assert_eq!(tq22, tq44);
        assert_eq!(tq22.scheme, KvScheme::TurboQuant);
    }

    /// The five schemes are pairwise distinct — no two `KVFormat` kinds (other
    /// than the TurboQuant bit-pairs) collide onto the same scheme.
    #[test]
    fn distinct_formats_resolve_to_distinct_schemes() {
        let schemes = [
            dispatch(KVFormat::BF16).scheme,
            dispatch(KVFormat::INT8).scheme,
            dispatch(KVFormat::FP8E4M3).scheme,
            dispatch(KVFormat::INT4).scheme,
            dispatch(KVFormat::TurboQuant {
                key_bits: 4,
                val_bits: 4,
            })
            .scheme,
        ];
        for i in 0..schemes.len() {
            for j in (i + 1)..schemes.len() {
                assert_ne!(
                    schemes[i], schemes[j],
                    "scheme {i} and {j} collided: {:?}",
                    schemes[i]
                );
            }
        }
    }
}
