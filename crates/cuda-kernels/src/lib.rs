#[cfg(feature = "cuda")]
pub mod attention;
pub mod collective;
#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(feature = "cuda")]
pub mod kv_quant;
#[cfg(feature = "cuda")]
pub mod kv_types;
#[cfg(feature = "cuda")]
pub mod moe;
#[cfg(feature = "cuda")]
pub mod paged_kv;
#[cfg(feature = "cuda")]
pub mod prelude;
// Host math compiles everywhere (autograd's CPU gates use it); device glue is
// feature-gated inside.
pub mod ring_attention;
#[cfg(feature = "cuda")]
pub mod tensor;
#[cfg(feature = "cuda")]
pub mod turboquant_state;

#[cfg(feature = "cuda")]
pub use kv_types::{KVCacheDtype, KVFormat};

/// So downstream crates (autograd) can name the `RawDevicePtr<bf16>` the DeepGEMM
/// wrappers expect without their own `half` dep.
pub use half::bf16;

include!(concat!(env!("OUT_DIR"), "/kernel_build_identity.rs"));

#[cfg(feature = "cuda")]
pub use paged_kv::{BandPage, TokenKVPool};

/// `build.rs` sets the backing `cfg` from the SAME enable-flags that gate nvcc
/// compilation, so these consts are the single source of truth for "is this
/// kernel present" — consumers (e.g. `infer-cuda`) gate on them instead of an
/// env var, and a build that did not compile the kernel takes its scalar
/// fallback. (`cfg` is per-crate; a const is how downstream crates see what
/// `cuda-kernels` compiled.)
#[cfg(arle_flashmla)]
pub const HAS_FLASHMLA: bool = true;
/// See [`HAS_FLASHMLA`].
#[cfg(not(arle_flashmla))]
pub const HAS_FLASHMLA: bool = false;

/// A **runtime** capability probe (cached), NOT a build cfg: the non-native stub
/// exports the SAME bridge symbols as the real build
/// (`dsv4_deepgemm_native_preflight_cuda` is defined by both), so only the
/// device preflight can tell them apart. Mirrors SGLang's `is_*_available()`
/// guard; consumers fall back to scalar when false. (`HAS_FLASHMLA` above stays
/// a cfg const because its real-kernel marker is real-only — the stub does not
/// define it — so FlashMLA IS build-determinable; DeepGEMM is not.)
#[cfg(feature = "cuda")]
pub fn has_deepgemm_native() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| moe::dsv4_deepgemm_native_preflight().is_ok())
}
/// See [`has_deepgemm_native`] (no CUDA build → no native DeepGEMM).
#[cfg(not(feature = "cuda"))]
pub fn has_deepgemm_native() -> bool {
    false
}
